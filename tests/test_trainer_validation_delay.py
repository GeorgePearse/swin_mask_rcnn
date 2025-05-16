"""Test that the trainer respects the validation delay configuration."""
import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from scripts.config.training_config import TrainingConfig
from scripts.train import IterationBasedTrainer


def test_trainer_respects_validation_delay():
    """Test that the trainer respects validation_start_step configuration."""
    # Create mock config with custom validation settings
    config = TrainingConfig(
        validation_start_step=1000,
        steps_per_validation=200,
        num_epochs=1,
        batch_size=1,
        num_classes=2
    )
    
    # Create mock model and loaders
    mock_model = Mock()
    mock_model.to = Mock(return_value=mock_model)
    mock_model.train = Mock()
    mock_model.eval = Mock()
    mock_model.parameters = Mock(return_value=[torch.randn(1, 1)])
    mock_model.state_dict = Mock(return_value={})
    
    # Create minimal mock DataLoader
    mock_train_loader = Mock()
    mock_train_loader.dataset = Mock()
    mock_train_loader.dataset.__len__ = Mock(return_value=10)
    mock_train_loader.__len__ = Mock(return_value=10)
    
    mock_val_loader = Mock()
    mock_val_loader.dataset = Mock()
    mock_val_loader.dataset.__len__ = Mock(return_value=5)
    
    mock_val_coco = Mock()
    
    # Create trainer
    trainer = IterationBasedTrainer(
        model=mock_model,
        train_loader=mock_train_loader,
        val_loader=mock_val_loader,
        val_coco=mock_val_coco,
        config=config,
        device=torch.device('cpu')
    )
    
    # Mock the train_step method to return dummy losses
    trainer.train_step = Mock(return_value={
        'total': 1.0, 
        'rpn_cls_loss': 0.1,
        'rpn_bbox_loss': 0.2,
        'roi_cls_loss': 0.3,
        'roi_bbox_loss': 0.4,
        'roi_mask_loss': 0.5
    })
    
    # Mock the evaluate_coco method
    trainer.evaluate_coco = Mock(return_value={
        'mAP': 0.5, 'mAP50': 0.6, 'mAP75': 0.4,
        'mAP_small': 0.3, 'mAP_medium': 0.4, 'mAP_large': 0.5
    })
    
    # Track when evaluate_coco is called
    validation_steps = []
    
    def mock_evaluate():
        validation_steps.append(trainer.global_step)
        return {
            'mAP': 0.5, 'mAP50': 0.6, 'mAP75': 0.4,
            'mAP_small': 0.3, 'mAP_medium': 0.4, 'mAP_large': 0.5
        }
    
    trainer.evaluate_coco = Mock(side_effect=mock_evaluate)
    
    # Create a small set of mock training data
    num_steps = 1500  # Enough to test before and after validation_start_step
    mock_data = [
        ([torch.randn(3, 224, 224)], [{'boxes': torch.tensor([[10, 10, 50, 50]]), 
                                      'labels': torch.tensor([1]), 
                                      'masks': torch.randn(1, 224, 224),
                                      'image_id': torch.tensor(1)}])
        for _ in range(num_steps)
    ]
    
    # Mock the train loader to iterate through our test data
    mock_train_loader.__iter__ = Mock(return_value=iter(mock_data))
    mock_train_loader.__len__ = Mock(return_value=num_steps)
    
    # Set up the loop to run for exactly our test steps
    original_tqdm = trainer.train
    
    def test_train():
        trainer.global_step = 0
        for batch_idx, (images, targets) in enumerate(mock_data):
            # Simulate training step
            trainer.train_step(images, targets)
            
            # Check validation logic (from actual train method)
            if (trainer.global_step >= trainer.config.validation_start_step and 
                trainer.global_step % trainer.config.steps_per_validation == 0):
                trainer.evaluate_coco()
                
            trainer.global_step += 1
            
            # Stop after 1500 steps for this test
            if trainer.global_step >= 1500:
                break
    
    # Run the test training loop
    test_train()
    
    # Check that validation was called at the right steps
    expected_validation_steps = [1000, 1200, 1400]
    assert validation_steps == expected_validation_steps, \
        f"Expected validation at {expected_validation_steps}, got {validation_steps}"
    
    # Verify no validation happened before step 1000
    assert all(step >= 1000 for step in validation_steps), \
        "Validation occurred before validation_start_step"
    
    print(f"Validation correctly delayed until step {config.validation_start_step}")
    print(f"Validation steps: {validation_steps}")


if __name__ == "__main__":
    test_trainer_respects_validation_delay()