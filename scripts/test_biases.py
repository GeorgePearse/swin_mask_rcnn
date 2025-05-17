"""Quick test to verify bias initialization fix."""

import torch
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN


def main():
    # Create model (no need to load weights for this test)
    print("Creating model...")
    model = SwinMaskRCNN(num_classes=69, frozen_backbone_stages=3)
    
    # Check bias values to verify our changes
    print("\nBias initialization check:")
    
    # RPN bias
    rpn_bias = model.rpn_head.rpn_cls.bias.data
    print(f"RPN cls bias: {rpn_bias.mean().item():.3f}")
    print(f"  Shape: {rpn_bias.shape}")
    print(f"  Values: {rpn_bias[:5].tolist()}")  # First 5 values
    
    # ROI head bias
    if hasattr(model.roi_head, 'bbox_head'):
        roi_cls_bias = model.roi_head.bbox_head.fc_cls.bias.data
    else:
        roi_cls_bias = model.roi_head.fc_cls.bias.data
    
    print(f"\nROI cls bias:")
    print(f"  Background (class 0): {roi_cls_bias[0].item():.3f}")
    print(f"  Object classes mean: {roi_cls_bias[1:].mean().item():.3f}")
    print(f"  Object classes range: [{roi_cls_bias[1:].min().item():.3f}, {roi_cls_bias[1:].max().item():.3f}]")
    print(f"  First 5 object classes: {roi_cls_bias[1:6].tolist()}")
    
    # Expected values after our fix:
    print("\nExpected values:")
    print("  RPN cls bias: ~-0.5")
    print("  ROI cls background: 0.0")
    print("  ROI cls objects: -0.1")


if __name__ == "__main__":
    main()