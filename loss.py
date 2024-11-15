import numpy as np
import torch

def get_loss_batch(output, targets, params_loss, opt = None):
    ignore_thres = params_loss["ignore_thres"]
    scaled_anchors = params_loss["scaled_anchors"]
    mse_loss = params_loss["mse_loss"]
    bce_loss = params_loss["bce_loss"]

    num_yolos = params_loss["num_yolos"]
    num_anchors = params_loss["num_anchors"]
    obj_scale = params_loss["obj_scale"]
    noobj_scale = params_loss["noobj_scale"]

    loss = 0.0

    for yolo_ind in range(num_yolos):
        # batch, num_bbx, coordinate + class
        yolo_out = output[yolo_ind]
        batch_size, num_bbx, _ = yolo_out.shape

        # get grid size
        # ex) 13x13 => 507 / 3
        gz_2 = num_bbx / num_anchors
        grid_size = int(np.sqrt(gz_2))

        # (batch, num_boxes, coordinate + class) -> (batch, num_anchors, grid_size, grid_size, coordinate + class)
        yolo_out = yolo_out.view(batch_size, num_anchors, grid_size, grid_size, -1)

        # box coordinate
        pred_boxes = yolo_out[...,:4]
        x, y, w, h = transform_bbox(pred_boxes, scaled_anchors[yolo_ind])
        pred_conf = yolo_out[...,4]
        pred_cls_prob = yolo_out[...,5:]

        yolo_targets = get_yolo_targets({
            'pred_cls_prob':pred_cls_prob,
            'pred_boxes':pred_boxes,
            'targets':targets,
            'anchors':scaled_anchors[yolo_ind],
            'ignore_thres':ignore_thres,
        })

        obj_mask=yolo_targets["obj_mask"].bool()
        noobj_mask=yolo_targets["noobj_mask"].bool()
        tx=yolo_targets["tx"]                
        ty=yolo_targets["ty"]                    
        tw=yolo_targets["tw"]                        
        th=yolo_targets["th"]                            
        tcls=yolo_targets["tcls"]                                
        t_conf=yolo_targets["t_conf"]

        loss_x = mse_loss(x[obj_mask], tx[obj_mask])
        loss_y = mse_loss(y[obj_mask], ty[obj_mask])
        loss_w = mse_loss(w[obj_mask], tw[obj_mask])
        loss_h = mse_loss(h[obj_mask], th[obj_mask])
        
        loss_conf_obj = bce_loss(pred_conf[obj_mask], t_conf[obj_mask])
        loss_conf_noobj = bce_loss(pred_conf[noobj_mask], t_conf[noobj_mask])
        loss_conf = obj_scale * loss_conf_obj + noobj_scale * loss_conf_noobj
        loss_cls = bce_loss(pred_cls_prob[obj_mask], tcls[obj_mask])
        loss += loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
        
    return loss.item()

def transform_bbox(bbox, anchors):
    """
    bbox : predicted bbox coordinates
    anchors : scaled anchors
    """
    ep = 1e-16

    x = bbox[...,0]
    y = bbox[...,1]
    w = bbox[...,2]
    h = bbox[...,3]
    anchor_w = anchors[:,0].view((1,3,1,1))
    anchor_h = anchors[:,1].view((1,3,1,1))

    # 전체 이미지 좌표에서 셀 내의 좌표로 변경
    # width, hegith도 anchor box에 맞게 log 스케일로 변경
    x = x - x.floor()
    y = y - y.floor()
    w = torch.log(w / anchor_w + ep)
    h = torch.log(h / anchor_h + ep)

    return x, y, w, h

def get_yolo_targets(params):
    """
    GT와 IoU가 가장 높은 anchor box를 object가 있다고 할당
    IoU > Threshold인 anchor box도 object가 있다고 할당
    나머지 anchor는 무시
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pred_boxes = params['pred_boxes']
    pred_cls_prob = params['pred_cls_prob']
    target = params['targets']
    anchors = params['anchors']
    ignore_thres = params['ignore_thres']

    batch_size = pred_boxes.size(0)
    num_anchors = pred_boxes.size(1)
    grid_size = pred_boxes.size(2)
    num_cls = pred_cls_prob.size(-1)

    sizeT = batch_size, num_anchors, grid_size, grid_size
    obj_mask = torch.zeros(sizeT, device=device, dtype=torch.uint8)
    noobj_mask = torch.ones(sizeT, device=device, dtype=torch.uint8)
    tx = torch.zeros(sizeT, device=device, dtype=torch.float32)
    ty = torch.zeros(sizeT, device=device, dtype=torch.float32)
    tw = torch.zeros(sizeT, device=device, dtype=torch.float32)
    th = torch.zeros(sizeT, device=device, dtype=torch.float32)

    sizeT = batch_size, num_anchors, grid_size, grid_size, num_cls
    tcls = torch.zeros(sizeT, device=device, dtype=torch.float32)

    # target = batch, cx, cy, w, h, class
    target_bboxes = target[:, 2:] * grid_size
    t_xy = target_bboxes[:, :2]
    t_wh = target_bboxes[:, 2:]
    t_x, t_y = t_xy.t() # .t(): 전치
    t_w, t_h = t_wh.t() # .t(): 전치

    grid_i, grid_j = t_xy.long().t() # .long(): int로 변환

    # anchor와 target의 iou 계산
    iou_with_anchors = [get_iou_WH(anchor, t_wh) for anchor in anchors]
    iou_with_anchors = torch.stack(iou_with_anchors)
    best_iou_wa, best_anchor_ind = iou_with_anchors.max(0) # iou가 가장 높은 anchor 추출

    batch_inds, target_labels = target[:, :2].long().t()
    obj_mask[batch_inds, best_anchor_ind, grid_j, grid_i] = 1 # iou가 가장 높은 anchor 할당
    noobj_mask[batch_inds, best_anchor_ind, grid_j, grid_i] = 0

    # threshold 보다 높은 iou를 지닌 anchor
    # iou가 가장 높은 anchor만 할당하면 되기 때문입니다.
    for ind, iou_wa in enumerate(iou_with_anchors.t()):
        noobj_mask[batch_inds[ind], iou_wa > ignore_thres, grid_j[ind], grid_i[ind]] = 0

    # cell 내에서 x,y로 변환
    tx[batch_inds, best_anchor_ind, grid_j, grid_i] = t_x - t_x.float()
    ty[batch_inds, best_anchor_ind, grid_j, grid_i] = t_y - t_y.float()

    anchor_w = anchors[best_anchor_ind][:, 0]
    tw[batch_inds, best_anchor_ind, grid_j, grid_i] = torch.log(t_w / anchor_w + 1e-16)

    anchor_h = anchors[best_anchor_ind][:, 1]
    th[batch_inds, best_anchor_ind, grid_j, grid_i] = torch.log(t_h / anchor_h + 1e-16)

    tcls[batch_inds, best_anchor_ind, grid_j, grid_i, target_labels] = 1

    output = {
        'obj_mask': obj_mask,
        'noobj_mask': noobj_mask,
        'tx': tx,
        'ty': ty,
        'tw': tw,
        'th': th,
        'tcls': tcls,
        't_conf': obj_mask.float(),
    }
    return output

def get_iou_WH(wh1, wh2):
    """
    anchor와 bounding box의 IoU 계산
    """
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area