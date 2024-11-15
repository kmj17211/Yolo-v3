import torch
import cv2
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import VOCDataset
from network import YOLOv3
from utils import show_img_bbox, NonMaxSuppression

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Image_size = 416
    batch_size = 32

    test_csv_file = 'Data/PASCAL_VOC/test.csv'
    label_dir = 'Data/PASCAL_VOC/labels'
    img_dir = 'Data/PASCAL_VOC/images'

    path2weight = './Weight/YoloV3/weights.pt'

    test_trasnform = A.Compose([A.LongestMaxSize(max_size = int(Image_size)),
                                A.PadIfNeeded(min_height = Image_size, min_width = Image_size, border_mode = cv2.BORDER_CONSTANT, value = [0, 0, 0]),
                                A.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5], max_pixel_value = 255),
                                ToTensorV2()],
                                bbox_params = A.BboxParams(format = 'yolo', min_visibility = 0.4, label_fields = []))
    
    test_ds = VOCDataset(test_csv_file, img_dir, label_dir, test_trasnform)
    # test_dl = DataLoader(test_ds, batch_size = batch_size, shuffle = True, collate_fn = collate_fn)

    anchors = [[(10,13),(16,30),(33,23)],[(30,61),(62,45),(59,119)],[(116,90),(156,198),(373,326)]]

    model = YOLOv3(anchors).to(device)
    model.load_state_dict(torch.load(path2weight))
    model.eval()

    start_index = 100
    num_test_img = 2
    test_result = []
    test_img = []
    with torch.no_grad():
        for i in range(start_index, start_index + num_test_img):
            img, _, _ = test_ds[i]
            result = model(img.unsqueeze(0).to(device))[0].detach().cpu()
            result_nms = NonMaxSuppression(result)
            test_result.append(result_nms)
            test_img.append(img)

    plt.figure(figsize = (20, 20))
    for i in range(num_test_img):
        plt.subplot(1, num_test_img, i + 1)
        show_img_bbox(test_img[i] * 0.5 + 0.5, test_result[i])
        
        