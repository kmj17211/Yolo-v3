"""
Made by KMJ
Start Project Date : 23.08.04
Last Edited Date : 23.08.14

Purpose : Study
Reference : https://deep-learning-study.tistory.com/568

"""
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import VOCDataset
from network import YOLOv3
from utils import train_val, collate_fn
from utils import show_img_bbox

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    # PreProcessing & Augmentation & Set HyperParameter & Path
    Image_Size = 416
    batch_size = 32

    train_csv_file = 'Data/PASCAL_VOC/train.csv'
    val_csv_file = 'Data/PASCAL_VOC/test.csv'
    label_dir = 'Data/PASCAL_VOC/labels'
    img_dir = 'Data/PASCAL_VOC/images'

    train_transform = A.Compose([A.LongestMaxSize(max_size = int(Image_Size)),
                                # 이미지의 width나 height 중 큰 것을 416으로, ratio 유지
                                A.PadIfNeeded(min_height = Image_Size, min_width = Image_Size, border_mode = cv2.BORDER_CONSTANT, value = [0, 0, 0]),
                                # 416x416이 되도록 빈 부분 padding
                                A.RandomCrop(width = Image_Size, height = Image_Size),
                                # 랜덤으로 자르기, Image_Size를 조정하지 않는 이상 의미없다
                                A.ColorJitter(brightness = 0.4, contrast = 0.4, saturation = 0.4, hue = 0.4, p = 0.3),
                                # 밝기, 대조, 최대값 무작위 변경
                                A.ShiftScaleRotate(rotate_limit = 20, p = 0.5, border_mode = cv2.BORDER_CONSTANT, value = [0, 0, 0]),
                                # Shift, Scale, Rotate, Shear를 무작위 적용
                                A.HorizontalFlip(p = 0.5),
                                # 뒤집기
                                A.CLAHE(p = 0.1),
                                # Contrast Limited Adaptive Histogram Equalization
                                A.ToGray(p = 0.1),
                                # Gray로 변환
                                A.Normalize(mean = [0.5,0.5,0.5], std = [0.5,0.5,0.5], max_pixel_value = 255),
                                # A.Normalize(mean = [0,0,0], std = [1,1,1], max_pixel_value = 255),
                                ToTensorV2()],
                                bbox_params = A.BboxParams(format = 'yolo', min_visibility = 0.4, label_fields = []))
    
    val_transforms = A.Compose([A.LongestMaxSize(max_size = int(Image_Size)),
                                A.PadIfNeeded(min_height = Image_Size, min_width = Image_Size, border_mode = cv2.BORDER_CONSTANT, value = [0, 0, 0]),
                                A.Normalize(mean = [0.5,0.5,0.5], std = [0.5,0.5,0.5], max_pixel_value = 255),
                                ToTensorV2()],
                                bbox_params = A.BboxParams(format = 'yolo', min_visibility = 0.4, label_fields = []))

    # Train Dataset
    train_ds = VOCDataset(train_csv_file, img_dir, label_dir, train_transform)

    # # Test
    # img, labels, _ = train_ds[100]
    # print('number of data:',len(train_ds))
    # print('image size:', img.shape, type(img)) # HxWxC
    # print('labels shape:', labels.shape, type(labels))  # x1,y1,x2,y2
    # print('lables \n', labels)

    # Validation Dataset << Test set을 validation으로 사용하는거 같다
    val_ds = VOCDataset(val_csv_file, img_dir, label_dir, val_transforms)

    # # Test
    # img, labels, _ = val_ds[1]
    # print('number of data:',len(val_ds))
    # print('image size:', img.shape, type(img))
    # print('labels shape:', labels.shape, type(labels))
    # print('lables \n', labels)

    # rnd_ind = np.random.randint(0, len(train_ds), grid_size)
    # print('Image Indices : ', rnd_ind)
    
    # Plot & Save Test Image
    rnd_ind = [100, 101]
    grid_size = len(rnd_ind)
    plt.figure(figsize = (20, 20))
    for i, indice in enumerate(rnd_ind):
        img, label, _ = val_ds[indice]
        plt.subplot(1, grid_size, i + 1)
        show_img_bbox(img * 0.5 + 0.5, label)

    # DataLoader
    train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle = True, collate_fn = collate_fn)
    val_dl = DataLoader(val_ds, batch_size = batch_size, shuffle=True, collate_fn = collate_fn)

    ## Test
    # torch.manual_seed(1)
    # for imgs_batch, tg_batch, path_batch in train_dl:
    #     break
    # print(imgs_batch.shape)
    # print(tg_batch.shape, tg_batch.dtype)
    # print(tg_batch)
    anchors = [[(10,13),(16,30),(33,23)],[(30,61),(62,45),(59,119)],[(116,90),(156,198),(373,326)]]

    model = YOLOv3(anchors).to(device)

    opt = optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = ReduceLROnPlateau(opt, mode='min',factor=0.5, patience=10,verbose=1)

    path2models= "./Weight/YoloV3"
    if not os.path.exists(path2models):
            os.mkdir(path2models)
            
    scaled_anchors=[]
    
    scaled_anchors.append([(a_w / 32, a_h / 32) for a_w, a_h in anchors[2]])
    scaled_anchors.append([(a_w / 16, a_h / 16) for a_w, a_h in anchors[1]])
    scaled_anchors.append([(a_w / 8, a_h / 8) for a_w, a_h in anchors[0]])
    scaled_anchors = torch.tensor(scaled_anchors, device = device)

    mse_loss = nn.MSELoss(reduction="sum")
    bce_loss = nn.BCELoss(reduction="sum")

    params_loss={
        "scaled_anchors" : scaled_anchors,
        "ignore_thres": 0.5,
        "mse_loss": mse_loss,
        "bce_loss": bce_loss,
        "num_yolos": 3,
        "num_anchors": 3,
        "obj_scale": 1,
        "noobj_scale": 100,
    }
    params_train={
        "num_epochs": 50,
        "optimizer": opt,
        "params_loss": params_loss,
        "train_dl": train_dl,
        "val_dl": val_dl,
        "sanity_check": False,
        "lr_scheduler": lr_scheduler,
        "path2weights": path2models+"/weights.pt",
    }
    model,loss_hist=train_val(model,params_train)

    num_epochs = params_train['num_epochs']

    # Plot train-val loss
    plt.title('Train-Val Loss')
    plt.plot(range(1, num_epochs+1), loss_hist['train'], label='train')
    plt.plot(range(1, num_epochs+1), loss_hist['val'], label='val')
    plt.ylabel('Loss')
    plt.xlabel('Training Epochs')
    plt.legend()
    plt.show()