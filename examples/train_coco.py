"""
Example training script for COCO dataset.
"""
import argparse
from pathlib import Path

from swin_maskrcnn import SwinMaskRCNN, create_dataloaders, train_mask_rcnn


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train SWIN Mask R-CNN on COCO')
    parser.add_argument('--train-ann', type=str, required=True,
                        help='Path to COCO train annotations JSON')
    parser.add_argument('--val-ann', type=str, required=True,
                        help='Path to COCO val annotations JSON')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=12,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    args = parser.parse_args()
    
    # Extract image directories from annotation paths
    # Assumes structure: annotations/instances_train2017.json -> train2017/
    train_root = str(Path(args.train_ann).parent.parent / Path(args.train_ann).stem.split('_')[-1])
    val_root = str(Path(args.val_ann).parent.parent / Path(args.val_ann).stem.split('_')[-1])
    
    print(f"Training with:")
    print(f"  Train annotations: {args.train_ann}")
    print(f"  Train images: {train_root}")
    print(f"  Val annotations: {args.val_ann}")
    print(f"  Val images: {val_root}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_root=train_root,
        train_ann_file=args.train_ann,
        val_root=val_root,
        val_ann_file=args.val_ann,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    # Create model
    model = SwinMaskRCNN(num_classes=80)
    
    # Training configuration
    config = {
        'num_epochs': args.num_epochs,
        'learning_rate': args.lr,
        'checkpoint_dir': './checkpoints'
    }
    
    # Train model
    trainer = train_mask_rcnn(model, train_loader, val_loader, config)
    
    print("Training completed!")


if __name__ == '__main__':
    main()