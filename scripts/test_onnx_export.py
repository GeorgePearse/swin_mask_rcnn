"""Test script to verify ONNX export functionality."""
import torch
import yaml
from pathlib import Path
from scripts.train import main

# Create a test config with minimal training for quick verification
test_config = {
    "train_ann": "/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:23.077836_train_ordered.json",
    "val_ann": "/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:38.270134_val_ordered.json",
    "img_root": "/home/georgepearse/data/images",
    "num_classes": 69,
    "num_epochs": 1,
    "train_batch_size": 2,
    "val_batch_size": 2,
    "max_val_images": 4,  # Use very few validation images for quick test
    "steps_per_validation": 10,  # Run validation every 10 steps
    "log_interval": 5,
    "checkpoint_dir": "./test_checkpoints",
    "pretrained_backbone": False,  # Skip pretrained loading for speed
}

# Save test config to file
test_config_path = Path("scripts/config/test_onnx_export.yaml")
with open(test_config_path, 'w') as f:
    yaml.dump(test_config, f)

print("Starting test training with ONNX export...")
main(config_path=str(test_config_path))

# Check if ONNX files were created
checkpoint_dir = Path(test_config["checkpoint_dir"])
latest_run = sorted(checkpoint_dir.glob("run_*"))[-1]

onnx_files = list(latest_run.glob("*.onnx"))
pth_files = list(latest_run.glob("weights_*.pth"))

print(f"\nONNX files created: {len(onnx_files)}")
for f in onnx_files:
    print(f"  - {f.name}")
    
print(f"\nWeight files created: {len(pth_files)}")
for f in pth_files:
    print(f"  - {f.name}")

print("\nTest completed!")