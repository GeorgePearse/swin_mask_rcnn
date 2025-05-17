"""Test that normalization is consistent between training and inference."""
import torch
import numpy as np
from swin_maskrcnn.data.transforms_simple import get_transform_simple
from swin_maskrcnn.data.dataset import CocoDataset
from scripts.config import TrainingConfig
from swin_maskrcnn.inference.predictor import MaskRCNNPredictor
from PIL import Image


def main():
    # Load config
    config = TrainingConfig()
    
    # Create dataset with simple transforms
    dataset = CocoDataset(
        root_dir=str(config.img_root),
        annotation_file=str(config.train_ann),
        transforms=get_transform_simple(train=False),
        mode='train'
    )
    
    # Get first image
    image, target = dataset[0]
    print(f"Dataset image shape: {image.shape}")
    print(f"Dataset image mean: {image.numpy().mean():.3f}")
    print(f"Dataset image std: {image.numpy().std():.3f}")
    print(f"Dataset image min: {image.numpy().min():.3f}")
    print(f"Dataset image max: {image.numpy().max():.3f}")
    
    # Check if the image is normalized (should be around 0 mean with std ~1)
    expected_mean = 0.0  # After normalization
    expected_std = 1.0   # After normalization
    
    # Now test predictor normalization
    # Get original image path
    image_path = dataset.coco.loadImgs(dataset.image_ids[0])[0]['file_name']
    full_path = dataset.root_dir / image_path
    
    # Manually apply predictor normalization to compare
    from torchvision.transforms import functional as F
    pil_image = Image.open(full_path).convert('RGB')
    predictor_image = F.to_tensor(pil_image)
    predictor_image = F.normalize(
        predictor_image, 
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
    print(f"\nPredictor image shape: {predictor_image.shape}")
    print(f"Predictor image mean: {predictor_image.numpy().mean():.3f}")
    print(f"Predictor image std: {predictor_image.numpy().std():.3f}")
    print(f"Predictor image min: {predictor_image.numpy().min():.3f}")
    print(f"Predictor image max: {predictor_image.numpy().max():.3f}")
    
    # Compare stats to check if both are normalized consistently
    print("\nNormalization check:")
    print(f"Dataset normalized: {abs(image.numpy().mean()) < 1.0 and abs(image.numpy().std() - 1.0) < 1.0}")
    print(f"Predictor normalized: {abs(predictor_image.numpy().mean()) < 1.0 and abs(predictor_image.numpy().std() - 1.0) < 1.0}")


if __name__ == '__main__':
    main()