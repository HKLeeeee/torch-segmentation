from torchvision import datasets, transforms
from base import BaseDataLoader
import os
import numpy as np
import cv2
from .dataset import SegmentationDataset, MultiClassSegmentationDataset, CityscapeDataset

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, 
                 validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class SegmentationLoader(BaseDataLoader):
    # TODO
    def __init__(self, 
                 data_dir, 
                 batch_size, 
                 shuffle=True, 
                 num_workers=1, 
                 mode='train', 
                 image_size=240):
        
        self.dataset = SegmentationDataset(data_path=data_dir, 
                                           mode=mode, 
                                           image_size=image_size)
        validation_split = 0.0
        super().__init__(self.dataset, 
                         batch_size, 
                         shuffle, 
                         validation_split, 
                         num_workers)

class CityLoader(BaseDataLoader):
    # TODO
    def __init__(self, 
                 data_dir, 
                 batch_size, 
                 shuffle=True, 
                 num_workers=1, 
                 mode='train', 
                 image_size=240):
        
        self.dataset = CityscapeDataset(data_path=data_dir, 
                                           mode=mode, 
                                           image_size=image_size)
        validation_split = 0.0
        super().__init__(self.dataset, 
                         batch_size, 
                         shuffle, 
                         validation_split, 
                         num_workers)
        
class MultiClassSegmentationLoader(BaseDataLoader):
    # TODO
    def __init__(self, 
                 data_dir, 
                 num_classes,
                 batch_size, 
                 shuffle=True, 
                 num_workers=1, 
                 mode='train', 
                 image_size=240):
        
        self.dataset = MultiClassSegmentationDataset(data_path=data_dir,
                                                     num_classes=num_classes, 
                                                     mode=mode, 
                                                     image_size=image_size)
        validation_split = 0.0
        super().__init__(self.dataset, 
                         batch_size, 
                         shuffle, 
                         validation_split, 
                         num_workers)
