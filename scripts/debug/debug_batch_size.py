"""Debug script to check batch size issue."""
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from swin_maskrcnn.data.dataset import CocoDataset
from swin_maskrcnn.data.transforms_simple import get_transform_simple
from swin_maskrcnn.utils.collate import collate_fn
from scripts.config import TrainingConfig


def main():
    # Load config
    config_path = Path('scripts/config/config.yaml')
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = TrainingConfig(**config_dict)
    
    print(f"Config train batch size: {config.train_batch_size}")
    print(f"Config val batch size: {config.val_batch_size}")
    
    # Create dataset
    train_dataset = CocoDataset(
        root_dir=str(config.img_root),
        annotation_file=str(config.train_ann),
        transforms=get_transform_simple(train=True),
        mode='train'
    )
    
    print(f"Dataset size: {len(train_dataset)}")
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for debugging
        collate_fn=collate_fn,
        drop_last=True
    )
    
    print(f"Number of batches: {len(train_loader)}")
    
    # Check first few train batches
    print("\nTrain batches:")
    for i, (images, targets) in enumerate(train_loader):
        print(f"Batch {i}: {len(images)} images")
        if i >= 5:
            break
    
    # Also check validation loader
    val_dataset = CocoDataset(
        root_dir=str(config.img_root),
        annotation_file=str(config.val_ann),
        transforms=get_transform_simple(train=False),
        mode='train'
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False
    )
    
    print(f"\nValidation dataset size: {len(val_dataset)}")
    print(f"Validation batches: {len(val_loader)}")
    
    print("\nValidation batches:")
    for i, (images, targets) in enumerate(val_loader):
        print(f"Batch {i}: {len(images)} images")
        if i >= 5:
            break


if __name__ == '__main__':
    main()