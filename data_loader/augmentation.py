import albumentations as A

def get_augmentation(IMAGE_SIZE):
    '''
     Albumentation augmentation function
     
     !!!!!!!!
     !!demo!!
     !!!!!!!!
     
     수정보완 필요!
     reference : https://albumentations.ai/docs/examples/pytorch_semantic_segmentation/
    '''
    aug =  A.Compose([
            A.HorizontalFlip(p=0.5),

            A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, \
                                border_mode=0),

            A.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
            A.RandomSizedCrop(min_max_height=(100, 201), height=IMAGE_SIZE, width=IMAGE_SIZE, \
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
            A.RandomGamma(p=0.8)
        ])
    
    return aug