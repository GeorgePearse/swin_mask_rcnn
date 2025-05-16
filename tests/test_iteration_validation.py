"""Quick test of iteration-based validation."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
from pycocotools.coco import COCO
import json

from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.data.dataset import CocoDataset
from swin_maskrcnn.data.transforms_simple import get_transform_simple
from swin_maskrcnn.utils.collate import collate_fn

# Configuration
config = {
    'train_ann': '/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:23.077836_train_ordered.json',
    'val_ann': '/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:38.270134_val_ordered.json',
    'img_root': '/home/georgepearse/data/images',
    'num_classes': 69,
    'batch_size': 1,
    'num_workers': 0,
    'lr': 1e-4,
    'steps_per_validation': 3,  # Very low for testing
    'max_steps': 10  # Stop after 10 steps
}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create small datasets
    train_dataset = CocoDataset(
        root_dir=config['img_root'],
        annotation_file=config['train_ann'],
        transforms=get_transform_simple(train=True),
        mode='train'
    )
    
    val_dataset = CocoDataset(
        root_dir=config['img_root'],
        annotation_file=config['val_ann'],
        transforms=get_transform_simple(train=False),
        mode='train'
    )
    
    # Use only first 10 samples for quick test
    train_dataset = torch.utils.data.Subset(train_dataset, range(10))
    val_dataset = torch.utils.data.Subset(val_dataset, range(5))
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn
    )
    
    # Create model
    model = SwinMaskRCNN(num_classes=config['num_classes'])
    model = model.to(device)
    
    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=config['lr'])
    
    # Training loop
    model.train()
    global_step = 0
    
    pbar = tqdm(range(config['max_steps']), desc="Training")
    
    train_iter = iter(train_loader)
    
    for step in pbar:
        # Get next batch (cycle through data)
        try:
            images, targets = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, targets = next(train_iter)
        
        # Move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        total_loss = sum(loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})
        
        global_step += 1
        
        # Run validation every N steps
        if global_step > 0 and global_step % config['steps_per_validation'] == 0:
            print(f"\nRunning validation at step {global_step}")
            
            # Calculate validation loss
            model.train()  # Keep in train mode for losses
            val_losses = []
            
            for val_images, val_targets in val_loader:
                val_images = [img.to(device) for img in val_images]
                val_targets = [{k: v.to(device) for k, v in t.items()} for t in val_targets]
                
                with torch.no_grad():
                    loss_dict = model(val_images, val_targets)
                    val_loss = sum(loss_dict.values())
                    val_losses.append(val_loss.item())
            
            avg_val_loss = np.mean(val_losses)
            print(f"  Validation Loss: {avg_val_loss:.4f}")
            
            # Simple inference test (no full COCO eval for speed)
            model.eval()
            with torch.no_grad():
                outputs = model(val_images[:1])  # Test on one image
                print(f"  Sample output: {len(outputs[0]['boxes'])} detections")
            
            model.train()  # Back to training
    
    print("\nTest completed successfully!")
    print("The iteration-based validation system is working.")


if __name__ == '__main__':
    main()