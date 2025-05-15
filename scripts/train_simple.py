"""Simple training script."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.data.dataset import CocoDataset
from swin_maskrcnn.data.transforms_simple import get_transform_simple
from swin_maskrcnn.utils.collate import collate_fn

# Configuration
config = {
    'train_ann': '/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:23.077836_train_ordered.json',
    'img_root': '/home/georgepearse/data/images',
    'num_classes': 69,
    'batch_size': 1,
    'num_workers': 0,
    'lr': 1e-4,
}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    train_dataset = CocoDataset(
        root_dir=config['img_root'],
        annotation_file=config['train_ann'],
        transforms=get_transform_simple(train=True),
        mode='train'
    )
    
    print(f"Dataset size: {len(train_dataset)}")
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn
    )
    
    # Create model
    model = SwinMaskRCNN(num_classes=config['num_classes'])
    model = model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    
    # Training loop
    model.train()
    for epoch in range(5):  # Just 5 epochs for testing
        print(f"\nEpoch {epoch+1}")
        
        total_loss = 0
        pbar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            # Move to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        avg_loss = total_loss / (batch_idx + 1)
        print(f"Average loss: {avg_loss:.4f}")
    
    print("Training test completed!")

if __name__ == '__main__':
    main()
