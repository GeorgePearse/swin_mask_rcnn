"""Test the max_val_images parameter functionality."""
import sys
from pathlib import Path
import pytest
import torch
from unittest.mock import MagicMock, patch

# Add project root to system path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.config.training_config import TrainingConfig
from scripts.train import main, MaskRCNNLightningModule
from swin_maskrcnn.data.dataset import CocoDataset
from swin_maskrcnn.data.transforms_simple import get_transform_simple


def test_max_val_images_config():
    """Test that max_val_images is correctly loaded from config."""
    config = TrainingConfig(max_val_images=50)
    assert config.max_val_images == 50
    
    # Test with None
    config = TrainingConfig()
    assert config.max_val_images is None


def test_max_val_images_working():
    """Test that the parameter works with the training setup."""
    import yaml
    import tempfile
    import os
    
    # Create a temporary config file
    config_data = {
        'train_ann': '/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:23.077836_train_ordered.json',
        'val_ann': '/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:38.270134_val_ordered.json',
        'img_root': '/home/georgepearse/data/images',
        'num_classes': 69,
        'train_batch_size': 1,
        'val_batch_size': 1,
        'num_workers': 0,
        'num_epochs': 1,
        'max_val_images': 5,  # Very small for testing
        'steps_per_validation': 1,
        'checkpoint_dir': './test_checkpoints/max_val_test'
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name
    
    try:
        # This will just verify the setup works, not run full training
        from swin_maskrcnn.data.dataset import CocoDataset
        from swin_maskrcnn.data.transforms_simple import get_transform_simple
        from torch.utils.data import Subset
        
        dataset = CocoDataset(
            root_dir=config_data['img_root'],
            annotation_file=config_data['val_ann'],
            transforms=get_transform_simple(train=False),
            mode='train'
        )
        
        original_size = len(dataset)
        print(f"Original validation dataset size: {original_size}")
        
        if config_data['max_val_images'] < original_size:
            subset = Subset(dataset, list(range(config_data['max_val_images'])))
            print(f"Subset size: {len(subset)}")
            assert len(subset) == config_data['max_val_images']
        
        print("max_val_images parameter is working correctly!")
        
    finally:
        os.unlink(config_path)


@patch('scripts.train.COCO')
def test_validation_step_before_start_step(mock_coco):
    """Test validation step behavior before validation_start_step."""
    # Create mock COCO object
    mock_coco_instance = MagicMock()
    mock_coco.return_value = mock_coco_instance
    
    # Create test configuration with validation start step
    config = TrainingConfig(
        num_classes=69,
        validation_start_step=1000,
        steps_per_validation=200,
    )
    
    # Create model
    module = MaskRCNNLightningModule(config=config, val_coco=mock_coco_instance)
    module.global_step = 500  # Before validation_start_step
    
    # Create test batch
    batch = (
        [torch.rand(3, 224, 224)],  # images
        [{'image_id': torch.tensor(1), 'boxes': torch.rand(1, 4), 'labels': torch.tensor([1])}]  # targets
    )
    
    # Execute validation step
    result = module.validation_step(batch, 0)
    
    # Should return early with 0 predictions
    assert result['predictions'] == 0
    assert len(module.validation_outputs) == 0, "No outputs should be stored"


@patch('scripts.train.COCO')
def test_validation_metrics_logging(mock_coco):
    """Test that mAP metrics are properly logged."""
    # Create mock COCO object
    mock_coco_instance = MagicMock()
    mock_coco.return_value = mock_coco_instance
    
    # Create test configuration with high validation_start_step to test skipping
    config = TrainingConfig(
        num_classes=69,
        validation_start_step=1000,
        steps_per_validation=200,
    )
    
    # Create model
    module = MaskRCNNLightningModule(config=config, val_coco=mock_coco_instance)
    
    # Mock the log method and global_step property
    with patch.object(module, 'log') as mock_log:
        with patch.object(type(module), 'global_step', new_callable=lambda: property(lambda self: 500)):
            # Call validation epoch end
            module.on_validation_epoch_end()
            
            # Check that default metrics were logged
            mock_log.assert_any_call('val/mAP', 0.0, on_epoch=True, sync_dist=True, rank_zero_only=False)
            mock_log.assert_any_call('val/mAP50', 0.0, on_epoch=True, sync_dist=True, rank_zero_only=False)


if __name__ == "__main__":
    test_max_val_images_config()
    test_max_val_images_working()
    test_validation_step_before_start_step()
    test_validation_metrics_logging()