"""Simple inference test to check detection behavior."""

import torch
import numpy as np
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN


def create_dummy_image(size=(3, 224, 224)):
    """Create a dummy image with some structure."""
    img = torch.rand(size)
    # Add some structured patterns
    h, w = size[1], size[2]
    # Add a bright rectangle in the center
    img[:, h//4:3*h//4, w//4:3*w//4] += 0.5
    # Add another smaller rectangle
    img[:, h//3:2*h//3, w//3:2*w//3] += 0.3
    return img.clamp(0, 1)


def main():
    # Create model without pretrained weights
    print("Creating model...")
    model = SwinMaskRCNN(num_classes=69, frozen_backbone_stages=-1)
    model.eval()
    
    # Test with dummy data
    print("\nTesting with dummy images...")
    batch_size = 2
    images = [create_dummy_image() for _ in range(batch_size)]
    
    with torch.no_grad():
        # Test initial prediction behavior
        predictions = model(images)
        
        print(f"\nResults for {batch_size} images:")
        for i, pred in enumerate(predictions):
            if pred and 'boxes' in pred:
                num_boxes = len(pred['boxes'])
                print(f"Image {i}: {num_boxes} detections")
                if num_boxes > 0:
                    scores = pred['scores'].numpy()
                    print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]")
                    print(f"  Mean score: {scores.mean():.3f}")
                    # Count by score threshold
                    thresholds = [0.01, 0.05, 0.1, 0.3, 0.5]
                    for t in thresholds:
                        count = (scores >= t).sum()
                        print(f"  Detections with score >= {t}: {count}")
            else:
                print(f"Image {i}: No detections")
    
    print("\nNote: This is with randomly initialized weights (no pretraining)")
    print("The key is that the model should still produce some detections")
    print("rather than suppressing everything to background.")


if __name__ == "__main__":
    main()