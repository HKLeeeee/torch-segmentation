import os
from glob import glob
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from .augmentation import get_train_augmentation, get_valid_augmentation


class SegmentationDataset(Dataset):
    '''
    <dataset folder structure>
    data-
        |
        |- images
            |-train
            |-test
            |-valid
        |- masks (must be png!)
            |-train
            |-test
            |-valid
            
    이미지들은 모두 png!
    '''
    def __init__(self, data_path, mode='train', image_size=240):
        '''
            mode string is same with dataset split folder name
            train, test, val을 폴더명으로 폴더를 나누었으면 mode도 train 또는 val 
            train은 무조건 train이어야함...!
        '''
        self.image_path = os.path.join(os.path.join(data_path, 'images'), mode)
        self.mask_path = os.path.join(os.path.join(data_path, 'masks'), mode)
        self.mode = mode 
        print(self.image_path, self.mask_path)

        # list of image files
        self.images = glob(os.path.join(self.image_path, '*.jpg'))
        # list of mask files
        self.masks = glob(os.path.join(self.mask_path, '*.png'))
        
        if len(self.images)==0 or len(self.masks)==0 :
            print('image : ', self.image_path)
            print('mask : ', self.mask_path)
            print('Check the file paths')
        
        
        self.image_size = image_size
        
        image = cv2.imread(self.images[0])
        if self.mode == 'train' :
            # load augmentation
            self.aug = get_train_augmentation(min(image.shape), self.image_size)
        else :
            self.aug = get_valid_augmentation(min(image.shape), self.image_size)
             
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        filename = image_path.split('/')[-1].replace('jpg', 'png')
        mask_path = os.path.join(self.mask_path, filename)

        image = np.array(Image.open(image_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)

        # 전처리, augmentation
        augmentation = self.aug(image=image, mask=mask)
        image = augmentation['image'] / 255.0
        mask = augmentation['mask']
        mask[mask == 255.0] = 1.0   
        mask = mask.unsqueeze(0)
        
        return image, mask
    

class MultiClassSegmentationDataset(SegmentationDataset):
    def __init__(self, data_path, num_classes, mode='train', image_size=240):
        super().__init__(data_path=data_path,
                         mode=mode,
                         image_size=image_size)
        self.num_classes = num_classes
        
    def __getitem__(self, idx):
        image, mask = super().__getitem__(idx)
        mask = mask.squeeze().to(torch.int64)
        mask = F.one_hot(mask, num_classes=self.num_classes)
        mask = mask.permute(2,0,1)
        return image, mask