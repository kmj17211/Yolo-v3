import torch
from torchvision.transforms.functional import to_pil_image
from PIL import ImageDraw
import matplotlib.pyplot as plt
import copy
import numpy as np
import time

from loss import get_loss_batch

# VOC class names
classes = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

COLORS = np.random.randint(0, 255, size = (80, 3), dtype = 'uint8')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def rescale_bbox(bb, W, H):
    """
    Bounding Box Rescale (0 ~ 1) to (0 ~ Image Size)
    """
    x, y, w, h = bb
    return [x*W, y*H, w*W, h*H]

def show_img_bbox(img, targets, classes = classes):
    """
    Plot Image and Bounding Box for Test
    """
    if torch.is_tensor(img):
        img = to_pil_image(img)
    if torch.is_tensor(targets):
        targets = targets.numpy()[:, 1:]
    
    W, H = img.size
    draw = ImageDraw.Draw(img)

    for tg in targets:
        if not bool(np.array(tg).sum()):
            continue
        # Class
        id = int(tg[4])
        # x, y, w, h (0 ~ 1)
        bbox = tg[:4]
        # x, y, w, h (0 ~ Image_Size)
        bbox = rescale_bbox(bbox, W, H)
        xc, yc, w, h = bbox

        name = classes[id]
        color = [int(c) for c in COLORS[id]]
        draw.rectangle(((xc-w/2, yc-h/2), (xc+w/2, yc+h/2)), outline=tuple(color), width=3)
        draw.text((xc-w/2, yc-h/2), name, fill=(255,255,255,0))
    plt.imshow(np.array(img))
    plt.savefig('aa.png')

def collate_fn(batch):
    """
    이미지마다 Bounding Box의 개수가 다른데, 이를 하나의 Batch로 묶고 DataLoader가 불러오면 에러가 발생한다.
    이를 해결하기 위한 함수로, 하나의 Tensor로 묶으며 맨 앞의 숫자에 Batch 내의 몇 번째 이미지의 Bounding Box인지 표기 (Batch 내의 몇 번째 이미지, x, y, w, h)
    이미지도 Tuple에서 Tensor Batch로 묶음
    """
    imgs, targets, paths = list(zip(*batch))
    # 빈 박스 제거
    targets = [boxes for boxes in targets if boxes is not None]
    # index 설정
    for b_i, boxes in enumerate(targets):
        boxes[:, 0] = b_i
    targets = torch.cat(targets, 0)
    imgs = torch.stack([img for img in imgs])
    return imgs, targets, paths

def get_lr(opt):
    """
    현재 Learning Rate
    """
    for param_group in opt.param_groups:
        return param_group['lr']

def loss_epoch(model,params_loss,dataset_dl,sanity_check=False,opt=None):
    """
    epoch마다 loss 계산
    """
    running_loss=0.0
    len_data=len(dataset_dl.dataset)
    running_metrics= {}
    
    for xb, yb,_ in dataset_dl:
        yb=yb.to(device)
        _,output=model(xb.to(device))
        loss_b = get_loss_batch(output,yb, params_loss,opt)
        running_loss+=loss_b
        if sanity_check is True:
            break 
    loss=running_loss/float(len_data)
    return loss

def train_val(model, params):
    num_epochs=params["num_epochs"]
    params_loss=params["params_loss"]
    opt=params["optimizer"]
    train_dl=params["train_dl"]
    val_dl=params["val_dl"]
    sanity_check=params["sanity_check"]
    lr_scheduler=params["lr_scheduler"]
    path2weights=params["path2weights"]
    
    
    loss_history={
        "train": [],
        "val": [],
    }
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss=float('inf') 
    
    start_time = time.time()
    for epoch in range(num_epochs):
        current_lr=get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr)) 
        model.train()
        train_loss=loss_epoch(model,params_loss,train_dl,sanity_check,opt)
        loss_history["train"].append(train_loss)  
        
        model.eval()
        with torch.no_grad():
            val_loss=loss_epoch(model,params_loss,val_dl,sanity_check)
        loss_history["val"].append(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path2weights)
            print("Copied best model weights!")
            print('Get best val loss')
            
        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            print("Loading best model weights!")
            model.load_state_dict(best_model_wts) 
        print("train loss: %.6f, val loss: %.6f, time: %.4f min" %(train_loss, val_loss, (time.time()-start_time)/60))
        print("-"*10) 
    model.load_state_dict(best_model_wts)
    return model, loss_history

def NonMaxSuppression(bbox_pred, obj_thres = 0.0005, nms_thres = 0.5):
    bbox_pred[..., :4] = calc_coordinate(bbox_pred[..., :4])
    # [None] * Batch_size
    output = [None] * len(bbox_pred)

    for i, bbox_pr in enumerate(bbox_pred):
        # Objectness Score가 obj_thres보다 낮은 Box 삭제
        bbox_pr = bbox_pr[bbox_pr[:, 4] >= obj_thres]

        if not bbox_pr.size(0):
            continue
        
        # Box 별로 Objectness Score * Class Score 가장 큰 값
        score = bbox_pr[:, 4] * bbox_pr[:, 5:].max(1)[0]
        # score 큰 값 순서대로 정렬
        bbox_pr = bbox_pr[(-score).argsort()]

        cls_probs, cls_preds = bbox_pr[:, 5:].max(1, keepdim = True)
        detections = torch.cat((bbox_pr[:, :5], # Box 좌표, Objectness
                                cls_probs.float(), # Max Class Probability
                                cls_preds.float()), 1) # Max Class Probability Index
        
        bbox_nms = []
        while detections.size(0):
            high_iou_indexs = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            
            cls_match_indexs = detections[0, -1] == detections[:, -1]
            supp_indexs = high_iou_indexs & cls_match_indexs

            ww = detections[supp_indexs, 4]
            detections[0, :4] = torch.matmul(ww, detections[supp_indexs, :4]).sum(0) / ww.sum()
            
            bbox_nms += [detections[0]]
            detections = detections[~supp_indexs]
            
        if bbox_nms:
            output[i] = torch.stack(bbox_nms)
            output[i] = inverse_calc_coordinate(output[i])
    return output         

def calc_coordinate(xywh):
    """
    Bounding Box의 좌표점 4개로 변환
    (x, y, w, h) -> (x - w/2, y - h/2, x + w/2, y + h/2)
    """
    xyxy = torch.zeros_like(xywh)
    xyxy[..., 0] = xywh[..., 0] - xywh[..., 2] / 2.0
    xyxy[..., 1] = xywh[..., 1] - xywh[..., 3] / 2.0
    xyxy[..., 2] = xywh[..., 0] + xywh[..., 2] / 2.0
    xyxy[..., 3] = xywh[..., 1] + xywh[..., 3] / 2.0
    return xyxy

def inverse_calc_coordinate(xyxy, img_size = 416):
    """
    (x - w/2, y - h/2, x + w/2, y + h/2) -> (x, y, w, h)
    """
    xywh = torch.zeros(xyxy.shape[0],6)
    xywh[:,2] = (xyxy[:, 0] + xyxy[:, 2]) / 2./img_size
    xywh[:,3] = (xyxy[:, 1] + xyxy[:, 3]) / 2./img_size
    xywh[:,5] = (xyxy[:, 2] - xyxy[:, 0])/img_size 
    xywh[:,4] = (xyxy[:, 3] - xyxy[:, 1])/img_size
    xywh[:,1]= xyxy[:,6]    
    return xywh

def bbox_iou(box1, box2):
    """
    Bounding Box IoU 계산
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    
    # box1, box2 겹치는 부분
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) \
                    *torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    
    # box1 + box2
    b1_area = (b1_x2 - b1_x1 + 1.0) * (b1_y2 - b1_y1 + 1.0)
    b2_area = (b2_x2 - b2_x1 + 1.0) * (b2_y2 - b2_y1 + 1.0)
    all_area = b1_area + b2_area - inter_area + 1e-16

    iou = inter_area / all_area
    return iou