"""Example training script with configurable backbone freezing."""
from scripts.config.training_config import TrainingConfig
from scripts.train import main


# Example 1: Train with no backbone freezing (default behavior)
config_no_freeze = TrainingConfig(
    frozen_backbone_stages=-1,  # Don't freeze any layers
    num_epochs=5,
    train_batch_size=2
)

# Example 2: Use default freezing (2 stages)
config_default_freeze = TrainingConfig(
    # frozen_backbone_stages=2 is the default
    num_epochs=5,
    train_batch_size=2
)

# Example 3: Explicitly set to freeze 2 stages
config_partial_freeze = TrainingConfig(
    frozen_backbone_stages=2,  # Freeze patch embedding and first 2 stages (default)
    num_epochs=5,
    train_batch_size=2
)

# Example 3: Freeze the entire backbone (all 4 stages)
config_full_freeze = TrainingConfig(
    frozen_backbone_stages=4,  # Freeze all backbone stages
    num_epochs=5,
    train_batch_size=2
)

if __name__ == "__main__":
    # Save configuration to YAML and run training
    import yaml
    import tempfile
    
    # Example: Train with partial freezing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config_dict = config_partial_freeze.model_dump()
        yaml.dump(config_dict, f)
        temp_config_path = f.name
    
    print(f"Training with frozen_backbone_stages={config_partial_freeze.frozen_backbone_stages}")
    print("This will freeze the patch embedding layer and the first backbone stage.")
    print("Stages 2, 3, and 4 will remain trainable.\n")
    
    main(config_path=temp_config_path)