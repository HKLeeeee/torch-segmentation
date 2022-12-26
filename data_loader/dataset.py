import os
from glob import glob
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from .augmentation import get_augmentation


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
        
        if self.mode == 'train' :
            # load augmentation
            image = cv2.imread(self.images[0])
            self.aug = get_augmentation(image.shape[0])
            del image
        self.image_size = image_size
             
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        filename = image_path.split('/')[-1].replace('jpg', 'png')
        mask_path = os.path.join(self.mask_path, filename)

        # image = Image.open(image_path)
        # mask = Image.open(mask_path)
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        
        
        if mask.ndim == 3 :
            if np.all(mask[:,:, 0] == mask[:,:,1]) and np.all(mask[:,:, 0] == mask[:,:,2]):
                mask = mask[:,:,0]  # .squeeze()
            else : 
                print('마스크가 이상해요!')
                
        if self.mode == 'train':
            # augmentation = self.aug(image=image, mask=mask)
            # image = augmentation['image'].astype(np.float32) / 255.0
            # mask = augmentation['mask'].astype(np.float32)
            pass
        
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size))
    
            
        image = image.transpose(2,0,1)
        mask = mask.reshape((1,)+mask.shape)
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32)
        
        return image, mask
    