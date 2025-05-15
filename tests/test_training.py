"""
Test training script using the specific paths from CLAUDE.md
"""
from pathlib import Path
import torch

from swin_maskrcnn import SwinMaskRCNN, create_dataloaders, train_mask_rcnn


def main():
    """Test training with specific CMR dataset."""
    # Paths from CLAUDE.md
    train_ann = "/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:23.077836_train_ordered.json"
    val_ann = "/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:38.270134_val_ordered.json"
    images_root = "/home/georgepearse/data/images"
    
    print(f"Training with:")
    print(f"  Train annotations: {train_ann}")
    print(f"  Val annotations: {val_ann}")
    print(f"  Images root: {images_root}")
    
    # Check files exist
    if not Path(train_ann).exists():
        raise FileNotFoundError(f"Training annotations not found: {train_ann}")
    if not Path(val_ann).exists():
        raise FileNotFoundError(f"Validation annotations not found: {val_ann}")
    if not Path(images_root).exists():
        raise FileNotFoundError(f"Images directory not found: {images_root}")
    
    # Create dataloaders with small batch size for testing
    train_loader, val_loader = create_dataloaders(
        train_root=images_root,
        train_ann_file=train_ann,
        val_root=images_root,
        val_ann_file=val_ann,
        batch_size=1,  # Small batch size for testing
        num_workers=2,
        img_size=400   # Smaller image size for testing
    )
    
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Val dataset size: {len(val_loader.dataset)}")
    
    # Create model
    # Not sure how many classes in CMR dataset, using default 80
    model = SwinMaskRCNN(num_classes=80)
    
    # Training configuration for quick test
    config = {
        'num_epochs': 1,  # Just 1 epoch for testing
        'learning_rate': 0.0001,
        'checkpoint_dir': './test_checkpoints',
        'log_interval': 10,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    
    print(f"Using device: {config['device']}")
    
    # Test training
    trainer = train_mask_rcnn(model, train_loader, val_loader, config)
    
    print("Test training completed successfully!")
    
    # Test one batch inference
    model.eval()
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(config['device'])
            outputs = model(images)
            print(f"Inference test - got outputs: {outputs.keys() if isinstance(outputs, dict) else type(outputs)}")
            break
    
    print("All tests passed!")


if __name__ == '__main__':
    main()