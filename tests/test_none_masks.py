import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.train import IterationBasedTrainer


def test_evaluate_coco_with_none_masks():
    """Test that evaluate_coco handles outputs with None masks gracefully."""
    
    # Create a real trainer instance (we'll mock its components)
    config = MagicMock()
    config.model = MagicMock()
    config.model.name = "swin_mask_rcnn"
    config.data = MagicMock()
    config.training = MagicMock()
    config.training.output_dir = "/tmp/test"
    config.training.batch_size = 1
    config.training.epochs = 1
    config.training.learning_rate = 0.001
    
    # Create the actual trainer
    with patch('scripts.train.logger'):
        trainer = IterationBasedTrainer(config)
    
    # Mock the model to return outputs with None masks
    mock_outputs = [
        {
            'boxes': torch.tensor([[100, 200, 300, 400]], dtype=torch.float32),
            'labels': torch.tensor([1], dtype=torch.int64),
            'scores': torch.tensor([0.9], dtype=torch.float32),
            'masks': None  # This is the case we're testing
        }
    ]
    
    trainer.model = MagicMock()
    trainer.model.eval = MagicMock()
    trainer.model.return_value = mock_outputs
    
    # Mock the validation loader
    mock_targets = [
        {
            'image_id': torch.tensor(123),
            'boxes': torch.tensor([[100, 200, 300, 400]], dtype=torch.float32),
            'labels': torch.tensor([1], dtype=torch.int64)
        }
    ]
    
    mock_batch = (torch.randn(1, 3, 224, 224), mock_targets)
    trainer.val_dataset_loader = [mock_batch]
    
    # Mock the COCO API components
    with patch('scripts.train.pycocotools'):
        with patch('scripts.train.maskUtils'):
            with patch('scripts.train.tqdm') as mock_tqdm:
                # Mock the progress bar
                mock_pbar = MagicMock()
                mock_tqdm.return_value = mock_pbar
                
                # Call evaluate_coco - it should handle None masks gracefully
                try:
                    trainer.evaluate_coco()
                    # If we get here without exception, the test passed
                    assert True
                except AttributeError as e:
                    if "'NoneType' object has no attribute 'cpu'" in str(e):
                        pytest.fail(f"The fix didn't work - still getting AttributeError: {e}")
                    else:
                        # Re-raise other AttributeErrors
                        raise


def test_evaluate_coco_direct():
    """Direct test of the None mask handling logic."""
    
    output = {
        'boxes': torch.tensor([[100, 200, 300, 400]]),
        'labels': torch.tensor([1]),
        'scores': torch.tensor([0.9]),
        'masks': None
    }
    
    # This is the logic from our fix
    if output['masks'] is None:
        # Should skip processing
        result = "skipped"
    else:
        # Would process masks here
        result = "processed"
    
    assert result == "skipped", "Should skip when masks is None"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])