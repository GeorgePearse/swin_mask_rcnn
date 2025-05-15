"""
Data augmentation transforms for instance segmentation.
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transform(train=True):
    """Get transforms for training or validation."""
    if train:
        return A.Compose([
            A.RandomResizedCrop(
                height=512,
                width=512,
                scale=(0.8, 1.0),
                ratio=(0.75, 1.33)
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        return A.Compose([
            A.Resize(height=512, width=512),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))