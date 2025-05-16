"""Test frozen backbone stages functionality."""
import torch
import pytest
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from scripts.config.training_config import TrainingConfig


def test_frozen_stages_config():
    """Test that frozen_backbone_stages config parameter works correctly."""
    # Test different frozen stage values
    configs = [
        TrainingConfig(frozen_backbone_stages=-1),  # No freezing
        TrainingConfig(frozen_backbone_stages=0),   # Freeze patch embedding
        TrainingConfig(frozen_backbone_stages=2),   # Freeze first 2 stages
        TrainingConfig(frozen_backbone_stages=4),   # Freeze all stages
    ]
    
    for config in configs:
        # Create model with configuration
        model = SwinMaskRCNN(
            num_classes=config.num_classes,
            frozen_backbone_stages=config.frozen_backbone_stages
        )
        
        # Check that frozen_stages is set correctly
        assert model.backbone.frozen_stages == config.frozen_backbone_stages
        
        # For each stage, check if parameters are frozen
        if config.frozen_backbone_stages >= 0:
            # Check patch embed is frozen if frozen_stages >= 0
            for param in model.backbone.patch_embed.parameters():
                assert not param.requires_grad, f"Patch embed should be frozen when frozen_stages={config.frozen_backbone_stages}"
        
        # Test that we can create a model and run a forward pass
        batch_size = 2
        images = [torch.randn(3, 224, 224) for _ in range(batch_size)]
        targets = [
            {
                'boxes': torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
                'labels': torch.tensor([1], dtype=torch.int64),
                'masks': torch.ones(1, 224, 224, dtype=torch.uint8),
            }
            for _ in range(batch_size)
        ]
        
        # Run forward pass to ensure model works
        model.train()
        losses = model(images, targets)
        assert isinstance(losses, dict)
        assert 'rpn_cls_loss' in losses


def test_frozen_stages_validation():
    """Test that frozen_backbone_stages is validated correctly."""
    # Valid values should work
    for stage in [-1, 0, 1, 2, 3, 4]:
        config = TrainingConfig(frozen_backbone_stages=stage)
        assert config.frozen_backbone_stages == stage
    
    # Invalid values should fail
    with pytest.raises(ValueError):
        TrainingConfig(frozen_backbone_stages=-2)  # Less than -1
    
    with pytest.raises(ValueError):
        TrainingConfig(frozen_backbone_stages=5)  # Greater than 4


def test_backward_compatibility():
    """Test that freeze_backbone still works for backward compatibility."""
    # Create model with old freeze_backbone parameter
    model = SwinMaskRCNN(
        num_classes=80,
        freeze_backbone=True  # Old parameter
    )
    
    # Should freeze all stages
    assert model.backbone.frozen_stages == 4


if __name__ == "__main__":
    test_frozen_stages_config()
    test_frozen_stages_validation()
    test_backward_compatibility()
    print("All tests passed!")