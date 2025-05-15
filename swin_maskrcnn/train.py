"""
Training script for SWIN-based Mask R-CNN.
"""
import argparse
import torch
from pathlib import Path

from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.data.dataset import create_dataloaders
from swin_maskrcnn.training.trainer import train_mask_rcnn


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train SWIN-based Mask R-CNN on COCO dataset')
    
    # Dataset arguments
    parser.add_argument('--train-ann', type=str, required=True,
                        help='Path to COCO train annotations JSON file')
    parser.add_argument('--val-ann', type=str, required=True,
                        help='Path to COCO val annotations JSON file')
    parser.add_argument('--images-root', type=str, default=None,
                        help='Common root directory for images (if train/val share same root)')
    
    # Model arguments
    parser.add_argument('--num-classes', type=int, default=80,
                        help='Number of object classes')
    parser.add_argument('--pretrained-backbone', type=str, default=None,
                        help='Path to pretrained SWIN weights')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=12,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--img-size', type=int, default=800,
                        help='Input image size')
    
    # Other arguments
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--test-only', action='store_true',
                        help='Only run test/validation')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Determine image roots from annotation files if not provided
    if args.images_root:
        train_root = val_root = args.images_root
    else:
        # Extract image root from annotation path
        # Assumes standard COCO structure: annotations/instances_train2017.json -> train2017/
        train_root = str(Path(args.train_ann).parent.parent / Path(args.train_ann).stem.split('_')[-1])
        val_root = str(Path(args.val_ann).parent.parent / Path(args.val_ann).stem.split('_')[-1])
    
    print(f"Train images root: {train_root}")
    print(f"Val images root: {val_root}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_root=train_root,
        train_ann_file=args.train_ann,
        val_root=val_root,
        val_ann_file=args.val_ann,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size
    )
    
    # Create model
    model = SwinMaskRCNN(
        num_classes=args.num_classes,
        pretrained_backbone=args.pretrained_backbone
    )
    
    # Training configuration
    config = {
        'num_epochs': args.num_epochs,
        'learning_rate': args.lr,
        'weight_decay': args.wd,
        'checkpoint_dir': args.checkpoint_dir,
        'device': device
    }
    
    # Train model
    if not args.test_only:
        trainer = train_mask_rcnn(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config
        )
        
        # Plot training history
        history_path = Path(args.checkpoint_dir) / 'training_history.png'
        trainer.plot_history(save_path=str(history_path))
    else:
        # Test mode
        model.to(device)
        model.eval()
        
        if args.resume:
            checkpoint = torch.load(args.resume, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded checkpoint from {args.resume}")
        
        # Run validation
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                images = images.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # Get predictions (in inference mode)
                outputs = model(images)
                print(f"Batch {batch_idx}: Got {len(outputs)} predictions")
                
                # You could add evaluation metrics here
                
                if batch_idx >= 10:  # Test first 10 batches
                    break
        
        print("Test completed!")


if __name__ == '__main__':
    main()