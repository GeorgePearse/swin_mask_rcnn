"""Test the batch handling issue that was causing errors"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from swin_maskrcnn.utils.collate import collate_fn
from swin_maskrcnn.data.dataset import CocoDataset
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN

# Test with batch size 1 and 2
for batch_size in [1, 2]:
    print(f"\n{'='*50}")
    print(f"Testing with batch_size={batch_size}")
    print(f"{'='*50}")
    
    # Create dataset
    dataset = CocoDataset(
        root_dir="/home/georgepearse/data/images",
        annotation_file="/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:23.077836_train_ordered.json",
        transforms=None,
        mode='train'
    )
    
    # Create dataloader with specific batch size
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Create model
    num_classes = dataset.num_classes
    print(f"Dataset has {num_classes} classes")
    model = SwinMaskRCNN(num_classes=num_classes)
    model.train()
    
    # Test forward pass
    for batch_idx, (images, targets) in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Images: {len(images)} x {images[0].shape}")
        print(f"  Targets: {len(targets)}")
        
        try:
            # Forward pass
            loss_dict = model(images, targets)
            print("  Forward pass successful!")
            for k, v in loss_dict.items():
                print(f"    {k}: {v.item():.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        # Just test first batch
        break