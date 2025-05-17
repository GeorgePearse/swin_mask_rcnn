"""Simple transforms without ToTensorV2."""
import albumentations as A


def get_transform_simple(train=True):
    """Get transforms without ToTensorV2."""
    if train:
        return A.Compose([
            A.RandomResizedCrop(
                size=(512, 512),
                scale=(0.8, 1.0),
                ratio=(0.75, 1.33)
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.Affine(
                translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
                scale=(0.9, 1.1),
                rotate=(-15, 15),
                p=0.5
            ),
            # Add normalization to match inference
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0  # For uint8 images
            ),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        return A.Compose([
            A.Resize(height=512, width=512),
            # Add normalization to match inference
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0  # For uint8 images
            ),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))