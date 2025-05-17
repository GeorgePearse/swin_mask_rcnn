"""Simple test to check normalization."""
import numpy as np
from swin_maskrcnn.data.transforms_simple import get_transform_simple


def main():
    # Create dummy image (uint8 RGB)
    dummy_image = np.ones((512, 512, 3), dtype=np.uint8) * 128  # Mid-gray image
    
    # Apply transforms
    train_transform = get_transform_simple(train=True)
    val_transform = get_transform_simple(train=False)
    
    # Test train transform with dummy bbox data
    dummy_bboxes = [[10, 10, 100, 100]]  # One dummy bbox
    dummy_labels = [1]  # One dummy label
    
    train_result = train_transform(image=dummy_image, bboxes=dummy_bboxes, labels=dummy_labels)
    train_image = train_result['image']
    
    print("Train transform:")
    print(f"  Input shape: {dummy_image.shape}, dtype: {dummy_image.dtype}")
    print(f"  Input mean: {dummy_image.mean():.3f}, range: [{dummy_image.min()}, {dummy_image.max()}]")
    print(f"  Output shape: {train_image.shape}, dtype: {train_image.dtype}")
    print(f"  Output mean: {train_image.mean():.3f}, range: [{train_image.min():.3f}, {train_image.max():.3f}]")
    
    # Test val transform  
    val_result = val_transform(image=dummy_image, bboxes=dummy_bboxes, labels=dummy_labels)
    val_image = val_result['image']
    
    print("\nValidation transform:")
    print(f"  Input shape: {dummy_image.shape}, dtype: {dummy_image.dtype}")
    print(f"  Input mean: {dummy_image.mean():.3f}, range: [{dummy_image.min()}, {dummy_image.max()}]")
    print(f"  Output shape: {val_image.shape}, dtype: {val_image.dtype}")
    print(f"  Output mean: {val_image.mean():.3f}, range: [{val_image.min():.3f}, {val_image.max():.3f}]")
    
    # Test with a real RGB value
    print("\nTesting normalization of a typical RGB value (128/255):")
    pixel_value = 128 / 255.0  # normalized to [0,1]
    
    # Apply ImageNet normalization manually
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    normalized_r = (pixel_value - imagenet_mean[0]) / imagenet_std[0]
    normalized_g = (pixel_value - imagenet_mean[1]) / imagenet_std[1]
    normalized_b = (pixel_value - imagenet_mean[2]) / imagenet_std[2]
    
    print(f"  Normalized R: {normalized_r:.3f}")
    print(f"  Normalized G: {normalized_g:.3f}")
    print(f"  Normalized B: {normalized_b:.3f}")
    print(f"  Average: {np.mean([normalized_r, normalized_g, normalized_b]):.3f}")


if __name__ == '__main__':
    main()