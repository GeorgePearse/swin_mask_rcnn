"""Debug training script."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.data.dataset import CocoDataset
from swin_maskrcnn.data.transforms import get_transform
from swin_maskrcnn.utils.collate import collate_fn

# Configuration
config = {
    'train_ann': '/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:23.077836_train_ordered.json',
    'img_root': '/home/georgepearse/data/images',
    'num_classes': 69,
    'batch_size': 1,  # Start with batch size 1
    'num_workers': 0,
}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    train_dataset = CocoDataset(
        root_dir=config['img_root'],
        annotation_file=config['train_ann'],
        transforms=get_transform(train=True),
        mode='train'
    )
    
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
    model.train()
    
    # Try one batch
    print("Testing one batch...")
    images, targets = next(iter(train_loader))
    
    # Move to device
    print(f"Images type: {type(images)}")
    print(f"Images length: {len(images)}")
    print(f"First image shape: {images[0].shape}")
    print(f"First image device: {images[0].device}")
    
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    
    # Forward pass
    print("Running forward pass...")
    try:
        loss_dict = model(images, targets)
        print(f"Losses: {loss_dict}")
        print("Success!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()