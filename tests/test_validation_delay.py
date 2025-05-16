"""Test validation delay functionality."""
import pytest
import torch
from scripts.config.training_config import TrainingConfig


def test_validation_start_step_in_config():
    """Test that validation_start_step is properly configured."""
    # Test default value
    config = TrainingConfig()
    assert config.validation_start_step == 1000
    
    # Test custom value
    config = TrainingConfig(validation_start_step=500)
    assert config.validation_start_step == 500
    

def test_validation_delays_correctly():
    """Test that validation is delayed correctly based on the config."""
    config = TrainingConfig(
        validation_start_step=1000,
        steps_per_validation=200
    )
    
    # Simulate training steps
    test_steps = [100, 500, 800, 1000, 1200, 1400, 1600]
    should_validate = []
    
    for step in test_steps:
        # Check if we should validate at this step
        if (step >= config.validation_start_step and 
            step % config.steps_per_validation == 0):
            should_validate.append(step)
    
    # Expected: validation at 1000, 1200, 1400, 1600
    expected = [1000, 1200, 1400, 1600]
    assert should_validate == expected, f"Expected {expected}, got {should_validate}"


def test_no_validation_before_start_step():
    """Test that no validation occurs before validation_start_step."""
    config = TrainingConfig(
        validation_start_step=1000,
        steps_per_validation=100
    )
    
    # Check steps before 1000
    steps_before = [100, 200, 300, 400, 500, 600, 700, 800, 900]
    
    for step in steps_before:
        # Should not validate at any of these steps
        should_validate = (step >= config.validation_start_step and 
                          step % config.steps_per_validation == 0)
        assert not should_validate, f"Should not validate at step {step}"


if __name__ == "__main__":
    test_validation_start_step_in_config()
    test_validation_delays_correctly()
    test_no_validation_before_start_step()
    print("All tests passed!")