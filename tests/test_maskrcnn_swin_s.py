#!/usr/bin/env python
"""Test the SwinMaskRCNN model with Swin-S configuration."""
import torch
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN

def test_swin_s_in_maskrcnn():
    # Create model with Swin-S backbone
    model = SwinMaskRCNN(
        num_classes=69,  # CMR dataset has 69 classes
        pretrained_backbone=None,  # We'll test without pretrained for now
    )
    
    # Check the backbone configuration
    print("Backbone configuration:")
    print(f"  embed_dims: {model.backbone.embed_dims}")
    print(f"  depths: {[model.backbone.layers[i].depth for i in range(4)]}")
    print(f"  num_heads: [3, 6, 12, 24]")  # These are hardcoded in the model
    
    # Test forward pass
    x = torch.randn(1, 3, 640, 640)
    model.eval()
    
    with torch.no_grad():
        outputs = model(x)
    
    print(f"\nModel outputs: {outputs}")
    print(f"Number of output tensors: {len(outputs)}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Backbone parameters: {backbone_params:,}")
    
    # Expected Swin-S size is ~50M parameters for the backbone
    print(f"\nBackbone size is appropriate for Swin-S: {40e6 < backbone_params < 60e6}")

if __name__ == "__main__":
    test_swin_s_in_maskrcnn()