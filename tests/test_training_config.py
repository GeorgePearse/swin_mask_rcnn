"""Test training configuration."""
import tempfile
from pathlib import Path

import pytest
import yaml

from scripts.config import TrainingConfig


def test_default_config():
    """Test default configuration values."""
    config = TrainingConfig()
    
    assert config.lr == 1e-4
    assert config.momentum == 0.9
    assert config.num_classes == 69
    assert config.steps_per_validation == 5
    assert config.optimizer == "adamw"
    assert config.use_scheduler is True
    assert config.batch_size == 1


def test_custom_config():
    """Test creating config with custom values."""
    config = TrainingConfig(
        lr=0.001,
        momentum=0.95,
        optimizer="sgd",
        batch_size=2
    )
    
    assert config.lr == 0.001
    assert config.momentum == 0.95
    assert config.optimizer == "sgd"
    assert config.batch_size == 2


def test_load_from_yaml():
    """Test loading configuration from YAML file."""
    config_dict = {
        'lr': 0.002,
        'momentum': 0.85,
        'optimizer': 'sgd',
        'steps_per_validation': 10
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as f:
        yaml.dump(config_dict, f)
        f.flush()
        
        # Load config from file
        with open(f.name, 'r') as yaml_file:
            loaded_dict = yaml.safe_load(yaml_file)
        
        config = TrainingConfig(**loaded_dict)
        
        assert config.lr == 0.002
        assert config.momentum == 0.85
        assert config.optimizer == 'sgd'
        assert config.steps_per_validation == 10


def test_invalid_optimizer():
    """Test that invalid optimizer raises error."""
    with pytest.raises(ValueError):
        TrainingConfig(optimizer="invalid")


def test_path_types():
    """Test that path fields are properly handled."""
    config = TrainingConfig()
    
    assert isinstance(config.train_ann, Path)
    assert isinstance(config.val_ann, Path)
    assert isinstance(config.img_root, Path)
    assert isinstance(config.checkpoint_dir, Path)