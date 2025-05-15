"""
Train SWIN MaskRCNN on CMR dataset.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.data.dataset import CocoDataset
from swin_maskrcnn.data.transforms import get_transform
from swin_maskrcnn.training.trainer import MaskRCNNTrainer
from swin_maskrcnn.utils.collate import collate_fn

# Configuration
config = {
    'train_ann': '/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:23.077836_train_ordered.json',
    'val_ann': '/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:38.270134_val_ordered.json',
    'img_root': '/home/georgepearse/data/images',
    'num_classes': 69,  # CMR dataset classes (0-68)
    'batch_size': 2,
    'num_workers': 0,
    'num_epochs': 50,
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'checkpoint_dir': './checkpoints',
    'log_interval': 20,
    'save_interval': 5,
}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = CocoDataset(
        root_dir=config['img_root'],
        annotation_file=config['train_ann'],
        transforms=get_transform(train=True),
        mode='train'
    )
    
    val_dataset = CocoDataset(
        root_dir=config['img_root'],
        annotation_file=config['val_ann'],
        transforms=get_transform(train=False),
        mode='val'
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # Create trainer
    trainer = MaskRCNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        learning_rate=config['lr'],
        weight_decay=config['weight_decay'],
        checkpoint_dir=config['checkpoint_dir'],
        log_interval=config['log_interval'],
        device=device
    )
    
    # Train model
    print("Starting training...")
    trainer.train()

if __name__ == '__main__':
    main()