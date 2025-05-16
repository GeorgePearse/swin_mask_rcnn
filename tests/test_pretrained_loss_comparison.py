"""Test that pretrained initialization provides lower initial loss values."""
import torch
import torch.nn as nn
import numpy as np
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.utils.pretrained_loader import load_pretrained_from_url
from scripts.config.training_config import TrainingConfig


def compute_average_loss(model: nn.Module, inputs: list, targets: list, num_samples: int = 5) -> dict:
    """Compute average loss over multiple forward passes."""
    model.train()
    
    total_losses = {}
    
    for i in range(num_samples):
        # Use same input/target for consistency
        losses = model(inputs, targets)
        
        # Accumulate losses
        for k, v in losses.items():
            if k not in total_losses:
                total_losses[k] = 0.0
            total_losses[k] += v.item()
    
    # Average the losses
    avg_losses = {k: v / num_samples for k, v in total_losses.items()}
    return avg_losses


def test_pretrained_vs_scratch_loss():
    """Test that pretrained initialization provides lower initial loss."""
    
    # Configuration
    num_classes = 69  # CMR dataset
    pretrained_url = "https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth"
    
    # Create test data
    torch.manual_seed(42)  # For reproducibility
    batch_size = 2
    inputs = [torch.randn(3, 224, 224) for _ in range(batch_size)]
    targets = [
        {
            'boxes': torch.tensor([
                [10, 10, 50, 50],
                [60, 60, 120, 120]
            ], dtype=torch.float32),
            'labels': torch.tensor([1, 5], dtype=torch.int64),
            'masks': torch.stack([
                torch.ones(224, 224, dtype=torch.uint8),
                torch.ones(224, 224, dtype=torch.uint8)
            ]),
        }
        for _ in range(batch_size)
    ]
    
    # Model 1: From scratch
    print("Creating model from scratch...")
    model_scratch = SwinMaskRCNN(num_classes=num_classes)
    losses_scratch = compute_average_loss(model_scratch, inputs, targets)
    
    # Model 2: With pretrained weights
    print("\nCreating model with pretrained weights...")
    model_pretrained = SwinMaskRCNN(num_classes=num_classes)
    load_pretrained_from_url(model_pretrained, pretrained_url, strict=False)
    losses_pretrained = compute_average_loss(model_pretrained, inputs, targets)
    
    # Compare losses
    print("\n=== Loss Comparison ===")
    print("Loss Type             | From Scratch | Pretrained | Improvement")
    print("-" * 60)
    
    all_better = True
    for loss_name in sorted(losses_scratch.keys()):
        scratch_loss = losses_scratch[loss_name]
        pretrained_loss = losses_pretrained[loss_name]
        improvement = (scratch_loss - pretrained_loss) / scratch_loss * 100
        
        print(f"{loss_name:20} | {scratch_loss:11.6f} | {pretrained_loss:10.6f} | {improvement:10.2f}%")
        
        # Check if pretrained is better (lower loss)
        if pretrained_loss >= scratch_loss:
            all_better = False
            print(f"  WARNING: Pretrained loss is not lower for {loss_name}")
    
    # Compute total loss
    total_scratch = sum(losses_scratch.values())
    total_pretrained = sum(losses_pretrained.values())
    total_improvement = (total_scratch - total_pretrained) / total_scratch * 100
    
    print("-" * 60)
    print(f"{'Total Loss':20} | {total_scratch:11.6f} | {total_pretrained:10.6f} | {total_improvement:10.2f}%")
    
    # Assertions
    assert total_pretrained < total_scratch, (
        f"Total pretrained loss ({total_pretrained:.6f}) should be lower than scratch loss ({total_scratch:.6f})"
    )
    
    # Check specific loss components that should improve significantly
    # ROI losses should show significant improvement
    roi_improvement = (losses_scratch['roi_bbox_loss'] - losses_pretrained['roi_bbox_loss']) / losses_scratch['roi_bbox_loss'] * 100
    rpn_bbox_improvement = (losses_scratch['rpn_bbox_loss'] - losses_pretrained['rpn_bbox_loss']) / losses_scratch['rpn_bbox_loss'] * 100
    
    print(f"\nDetailed analysis:")
    print(f"ROI bbox loss improvement: {roi_improvement:.2f}%")
    print(f"RPN bbox loss improvement: {rpn_bbox_improvement:.2f}%")
    
    # At least bbox regression losses should show significant improvement
    assert roi_improvement > 50, (
        f"Expected significant ROI bbox loss improvement, got {roi_improvement:.2f}%"
    )
    assert rpn_bbox_improvement > 50, (
        f"Expected significant RPN bbox loss improvement, got {rpn_bbox_improvement:.2f}%"
    )
    
    # Total improvement should be positive (even if small due to dominating RPN cls loss)
    assert total_improvement > 0, (
        f"Expected positive total improvement, got {total_improvement:.2f}%"
    )
    
    print(f"\n✓ Test passed! Pretrained model shows better bbox regression performance")
    print(f"  - ROI bbox loss: {roi_improvement:.2f}% improvement")
    print(f"  - RPN bbox loss: {rpn_bbox_improvement:.2f}% improvement")
    print(f"  - Total loss: {total_improvement:.2f}% improvement")


def test_pretrained_backbone_features():
    """Test that pretrained backbone produces different features than random initialization."""
    
    # Configuration
    num_classes = 69
    pretrained_url = "https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth"
    
    # Create input
    torch.manual_seed(42)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Model 1: From scratch
    model_scratch = SwinMaskRCNN(num_classes=num_classes)
    model_scratch.eval()
    
    # Model 2: With pretrained weights
    model_pretrained = SwinMaskRCNN(num_classes=num_classes)
    load_pretrained_from_url(model_pretrained, pretrained_url, strict=False)
    model_pretrained.eval()
    
    # Extract backbone features
    with torch.no_grad():
        features_scratch = model_scratch.backbone(dummy_input)
        features_pretrained = model_pretrained.backbone(dummy_input)
    
    # Compare features at each level
    print("\n=== Backbone Feature Comparison ===")
    for level, (feat_s, feat_p) in enumerate(zip(features_scratch, features_pretrained)):
        # Compute L2 distance between features
        diff = (feat_s - feat_p).pow(2).mean().sqrt().item()
        
        # Compute relative magnitude
        scratch_norm = feat_s.norm().item()
        pretrained_norm = feat_p.norm().item()
        
        print(f"Level {level}: Feature diff = {diff:.6f}, "
              f"Scratch norm = {scratch_norm:.6f}, "
              f"Pretrained norm = {pretrained_norm:.6f}")
        
        # Features should be significantly different
        assert diff > 0.1, f"Features at level {level} are too similar"
    
    print("\n✓ Pretrained backbone produces significantly different features")


def test_pretrained_initial_loss():
    """Test that pretrained model has lower initial loss."""
    
    # Configuration
    num_classes = 69
    pretrained_url = "https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth"
    
    # Create test data with diverse examples
    torch.manual_seed(42)
    num_samples = 3
    
    print("Comparing initial loss across multiple samples...")
    
    total_improvement = 0
    for i in range(num_samples):
        # Generate different test data for each sample
        batch_size = 2
        inputs = [torch.randn(3, 224, 224) for _ in range(batch_size)]
        targets = [
            {
                'boxes': torch.tensor([
                    [10+i*10, 10+i*10, 50+i*10, 50+i*10],
                    [60+i*5, 60+i*5, 120+i*5, 120+i*5]
                ], dtype=torch.float32),
                'labels': torch.tensor([i+1, i+5], dtype=torch.int64),
                'masks': torch.stack([
                    torch.ones(224, 224, dtype=torch.uint8),
                    torch.ones(224, 224, dtype=torch.uint8)
                ]),
            }
            for _ in range(batch_size)
        ]
        
        # Model from scratch
        model_scratch = SwinMaskRCNN(num_classes=num_classes)
        model_scratch.train()
        losses_scratch = model_scratch(inputs, targets)
        total_loss_scratch = sum(losses_scratch.values()).item()
        
        # Model with pretrained weights
        model_pretrained = SwinMaskRCNN(num_classes=num_classes)
        load_pretrained_from_url(model_pretrained, pretrained_url, strict=False)
        model_pretrained.train()
        losses_pretrained = model_pretrained(inputs, targets)
        total_loss_pretrained = sum(losses_pretrained.values()).item()
        
        # Calculate improvement
        improvement = (total_loss_scratch - total_loss_pretrained) / total_loss_scratch * 100
        total_improvement += improvement
        
        print(f"\nSample {i+1}:")
        print(f"  From scratch: {total_loss_scratch:.6f}")
        print(f"  Pretrained:   {total_loss_pretrained:.6f}")
        print(f"  Improvement:  {improvement:.2f}%")
    
    # Average improvement
    avg_improvement = total_improvement / num_samples
    print(f"\nAverage initial loss improvement: {avg_improvement:.2f}%")
    
    assert avg_improvement > 0, (
        f"Expected positive average improvement, got {avg_improvement:.2f}%"
    )
    
    print(f"\n✓ Pretrained model shows {avg_improvement:.2f}% lower initial loss on average")


if __name__ == "__main__":
    print("Testing pretrained vs scratch loss comparison...")
    test_pretrained_vs_scratch_loss()
    
    print("\n" + "="*60 + "\n")
    
    print("Testing pretrained backbone features...")
    test_pretrained_backbone_features()
    
    print("\n" + "="*60 + "\n")
    
    print("Testing pretrained initial loss...")
    test_pretrained_initial_loss()
    
    print("\nAll tests passed!")