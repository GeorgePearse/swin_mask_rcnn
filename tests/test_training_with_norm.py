"""Quick training test with normalization fix."""
import yaml
from pathlib import Path
import torch
from scripts.config import TrainingConfig
from scripts.train import main


def main_test():
    # Create a test config with minimal settings
    config_dict = {
        'train_ann': '/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:23.077836_train_ordered.json',
        'val_ann': '/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:38.270134_val_ordered.json',
        'img_root': '/home/georgepearse/data/images',
        'num_classes': 69,
        'train_batch_size': 2,
        'val_batch_size': 2,
        'num_workers': 0,
        'num_epochs': 1,
        'steps_per_validation': 10,
        'log_interval': 5,
        'checkpoint_dir': './test_norm_checkpoints',
        'max_val_images': 10  # Small validation set for quick test
    }
    
    # Save test config
    test_config_path = Path('test_norm_config.yaml')
    with open(test_config_path, 'w') as f:
        yaml.dump(config_dict, f)
    
    # Run training
    print("Starting training with normalization fix...")
    main(config_path=str(test_config_path))
    
    # Clean up
    test_config_path.unlink()
    print("Training test completed successfully!")


if __name__ == '__main__':
    main_test()