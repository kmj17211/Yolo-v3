import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import pandas as pd


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VOCDataset(Dataset):
    def __init__(self, csv_file, img_dir, label_dir, transform = None, trans_params = None):
        super().__init__()

        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.trans_params = trans_params

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):

        # Load Label
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        labels = None
        if os.path.exists(label_path):
            labels = np.array(np.roll(np.loadtxt(fname = label_path, delimiter = ' ', ndmin = 2), 4, axis = 1).tolist())

        # Load Image
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert('RGB'))

        if self.transform:
            augmentations = self.transform(image = image, bboxes = labels)
            image = augmentations['image']
            targets = augmentations['bboxes']

            # Dim : [batch, cx, cy, w, h, class]
            if targets is not None:
                targets = torch.zeros((len(labels), 6))
                targets[:, 1:] = torch.tensor(labels)
        
        else:
            targets = labels

        return image, targets, label_path
    
