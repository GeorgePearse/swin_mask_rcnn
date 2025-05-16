"""Test pretrained weight loading from mmdetection URL."""
import torch
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.utils.pretrained_loader import load_pretrained_from_url
from scripts.config.training_config import TrainingConfig


def test_pretrained_loading():
    """Test loading pretrained weights."""
    # Create config with pretrained URL
    config = TrainingConfig(
        num_classes=69,
        pretrained_backbone=True,
        pretrained_checkpoint_url="https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth"
    )
    
    # Create model
    model = SwinMaskRCNN(
        num_classes=config.num_classes,
        frozen_backbone_stages=config.frozen_backbone_stages
    )
    
    # Count parameters before loading
    param_count_before = sum(p.numel() for p in model.backbone.parameters())
    print(f"Backbone parameters before loading: {param_count_before:,}")
    
    # Load pretrained weights
    print(f"\nLoading pretrained weights from:\n{config.pretrained_checkpoint_url}")
    load_pretrained_from_url(model, config.pretrained_checkpoint_url, strict=False)
    
    # Count parameters after loading
    param_count_after = sum(p.numel() for p in model.backbone.parameters())
    print(f"\nBackbone parameters after loading: {param_count_after:,}")
    
    # Check if some weights were loaded
    loaded_params = 0
    for name, param in model.backbone.named_parameters():
        if param.abs().sum() > 0:  # Non-zero parameters
            loaded_params += 1
    
    print(f"Non-zero parameter tensors: {loaded_params}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = [torch.randn(3, 224, 224) for _ in range(2)]
    dummy_targets = [
        {
            'boxes': torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
            'labels': torch.tensor([1], dtype=torch.int64),
            'masks': torch.ones(1, 224, 224, dtype=torch.uint8),
        }
        for _ in range(2)
    ]
    
    model.train()
    try:
        losses = model(dummy_input, dummy_targets)
        print("Forward pass successful!")
        print(f"Loss keys: {list(losses.keys())}")
    except Exception as e:
        print(f"Forward pass failed: {e}")


if __name__ == "__main__":
    test_pretrained_loading()