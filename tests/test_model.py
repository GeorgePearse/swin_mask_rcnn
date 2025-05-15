"""
Test script to verify the model can be instantiated and run.
"""
import torch
import numpy as np
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.models.swin import SwinTransformer
from swin_maskrcnn.models.fpn import FPN
from swin_maskrcnn.models.rpn import RPNHead
from swin_maskrcnn.models.roi_head import StandardRoIHead


def test_swin_backbone():
    """Test SWIN backbone forward pass."""
    print("Testing SWIN backbone...")
    
    # Create model
    model = SwinTransformer(
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        out_indices=(0, 1, 2, 3)
    )
    
    # Create dummy input
    batch_size = 2
    img_size = 224
    x = torch.randn(batch_size, 3, img_size, img_size)
    
    # Forward pass
    outputs = model(x)
    
    # Check outputs
    assert len(outputs) == 4, f"Expected 4 outputs, got {len(outputs)}"
    
    expected_sizes = [
        (batch_size, 96, img_size // 4, img_size // 4),
        (batch_size, 192, img_size // 8, img_size // 8),
        (batch_size, 384, img_size // 16, img_size // 16),
        (batch_size, 768, img_size // 32, img_size // 32)
    ]
    
    for i, (output, expected) in enumerate(zip(outputs, expected_sizes)):
        assert output.shape == expected, f"Output {i} shape mismatch: {output.shape} vs {expected}"
    
    print("✓ SWIN backbone test passed!")


def test_fpn():
    """Test FPN forward pass."""
    print("Testing FPN...")
    
    # Create model
    fpn = FPN(
        in_channels_list=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5
    )
    
    # Create dummy inputs
    batch_size = 2
    features = [
        torch.randn(batch_size, 96, 56, 56),
        torch.randn(batch_size, 192, 28, 28),
        torch.randn(batch_size, 384, 14, 14),
        torch.randn(batch_size, 768, 7, 7)
    ]
    
    # Forward pass
    outputs = fpn(features)
    
    # Check outputs
    assert len(outputs) == 5, f"Expected 5 outputs, got {len(outputs)}"
    
    for i, output in enumerate(outputs):
        assert output.shape[1] == 256, f"Output {i} channel mismatch"
    
    print("✓ FPN test passed!")


def test_rpn():
    """Test RPN forward pass."""
    print("Testing RPN...")
    
    # Create model
    rpn = RPNHead(in_channels=256)
    
    # Create dummy inputs
    batch_size = 2
    features = [
        torch.randn(batch_size, 256, 56, 56),
        torch.randn(batch_size, 256, 28, 28),
        torch.randn(batch_size, 256, 14, 14),
        torch.randn(batch_size, 256, 7, 7),
        torch.randn(batch_size, 256, 4, 4)
    ]
    
    # Forward pass
    cls_scores, bbox_preds = rpn(features)
    
    # Check outputs
    assert len(cls_scores) == len(features), "Number of classification outputs mismatch"
    assert len(bbox_preds) == len(features), "Number of regression outputs mismatch"
    
    print("✓ RPN test passed!")


def test_roi_head():
    """Test ROI head forward pass."""
    print("Testing ROI head...")
    
    # Create model
    roi_head = StandardRoIHead(num_classes=80)
    
    # Create dummy inputs
    batch_size = 2
    features = [torch.randn(batch_size, 256, 56, 56)]
    proposals = [
        torch.tensor([[10, 10, 50, 50], [100, 100, 200, 200]], dtype=torch.float32),
        torch.tensor([[20, 20, 60, 60]], dtype=torch.float32)
    ]
    
    # Forward pass (inference mode)
    roi_head.eval()
    with torch.no_grad():
        results = roi_head(features, proposals)
    
    assert 'boxes' in results, "Missing boxes in results"
    assert 'labels' in results, "Missing labels in results"
    assert 'scores' in results, "Missing scores in results"
    
    print("✓ ROI head test passed!")


def test_full_model():
    """Test full Mask R-CNN model."""
    print("Testing full Mask R-CNN model...")
    
    # Create model
    model = SwinMaskRCNN(num_classes=80)
    
    # Create dummy inputs
    batch_size = 2
    img_size = 800
    images = torch.randn(batch_size, 3, img_size, img_size)
    
    # Test training mode
    model.train()
    targets = []
    for i in range(batch_size):
        target = {
            'boxes': torch.tensor([[10, 10, 100, 100], [200, 200, 300, 300]], dtype=torch.float32),
            'labels': torch.tensor([1, 2], dtype=torch.int64),
            'masks': torch.randint(0, 2, (2, img_size, img_size), dtype=torch.uint8)
        }
        targets.append(target)
    
    # Forward pass (training)
    losses = model(images, targets)
    
    assert 'rpn_cls_loss' in losses, "Missing RPN classification loss"
    assert 'rpn_bbox_loss' in losses, "Missing RPN bbox loss"
    assert 'roi_cls_loss' in losses, "Missing ROI classification loss"
    assert 'roi_bbox_loss' in losses, "Missing ROI bbox loss"
    assert 'roi_mask_loss' in losses, "Missing ROI mask loss"
    
    # Test inference mode
    model.eval()
    with torch.no_grad():
        predictions = model(images)
    
    assert isinstance(predictions, dict), "Predictions should be a dictionary"
    assert 'boxes' in predictions, "Missing boxes in predictions"
    assert 'labels' in predictions, "Missing labels in predictions"
    assert 'scores' in predictions, "Missing scores in predictions"
    
    print("✓ Full model test passed!")


def test_model_with_small_input():
    """Test model with smaller input for quick verification."""
    print("Testing model with small input...")
    
    # Create model
    model = SwinMaskRCNN(num_classes=10)
    
    # Create small inputs
    batch_size = 1
    img_size = 224
    images = torch.randn(batch_size, 3, img_size, img_size)
    
    # Test inference
    model.eval()
    with torch.no_grad():
        outputs = model(images)
    
    print(f"Got outputs: {outputs.keys() if isinstance(outputs, dict) else type(outputs)}")
    print("✓ Small input test passed!")


if __name__ == '__main__':
    print("Running SWIN Mask R-CNN tests...\n")
    
    # Run tests
    test_swin_backbone()
    print()
    test_fpn()
    print()
    test_rpn()
    print()
    test_roi_head()
    print()
    test_full_model()
    print()
    test_model_with_small_input()
    
    print("\nAll tests passed! ✓")