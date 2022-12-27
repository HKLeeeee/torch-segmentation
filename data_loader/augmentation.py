import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_augmentation(ORI_IMAGE_SIZE, TARGET_IMAGE_SIZE):
    '''
     Albumentation augmentation function
     
     !!!!!!!!
     !!demo!!
     !!!!!!!!
     
     수정보완 필요!
     reference : https://albumentations.ai/docs/examples/pytorch_semantic_segmentation/
    '''
    aug =  A.Compose([
             A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, \
                                border_mode=0),

            A.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
            A.RandomSizedCrop(min_max_height=(100, 201), height=ORI_IMAGE_SIZE, width=ORI_IMAGE_SIZE, \
                                always_apply=True),

            A.GaussNoise(p=0.2),
            A.Perspective(p=0.5),

            A.OneOf(
                [
                    A.CLAHE(p=1),
                    A.RandomBrightnessContrast(p=1),
                    A.RandomGamma(p=1),
                ],
                p=0.9,
            ),

            A.OneOf(
                [
                    A.Sharpen(p=1),
                    A.Blur(blur_limit=3, p=1),
                    A.MotionBlur(blur_limit=3, p=1),
                ],
                p=0.9,
            ),

            A.OneOf(
                [
                    A.RandomBrightnessContrast(p=1),
                    A.HueSaturationValue(p=1),
                ],
                p=0.9,
            ),
        
            A.CLAHE(p=0.8),
            A.RandomBrightnessContrast(p=0.8),    
            A.RandomGamma(p=0.8),
            A.Resize(height=TARGET_IMAGE_SIZE, width=TARGET_IMAGE_SIZE),
            A.VerticalFlip(p=0.1),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            # A.Normalize(
            #     mean=[0.0, 0.0, 0.0],
            #     std=[1.0, 1.0, 1.0],
            #     max_pixel_value=255.0,
            # ),
            ToTensorV2()
        ])
    
    return aug

def get_valid_augmentation(ORI_IMAGE_SIZE, TARGET_IMAGE_SIZE):
    aug =  A.Compose([
            A.Resize(height=TARGET_IMAGE_SIZE, width=TARGET_IMAGE_SIZE),
            # A.Normalize(
            #     mean=[0.0, 0.0, 0.0],
            #     std=[1.0, 1.0, 1.0],
            #     max_pixel_value=255.0,
            # ),
            ToTensorV2()
        ])
    
    return aug



'''
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )
'''