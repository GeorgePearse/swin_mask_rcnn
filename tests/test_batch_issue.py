"""Test to reproduce the exact batch handling issue"""

import torch
import torch.nn as nn
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.utils.collate import collate_fn
from torch.utils.data import DataLoader

# Mock dataset
class MockDataset:
    def __init__(self, size=4):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        image = torch.randn(3, 224, 224)
        target = {
            'boxes': torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100]], dtype=torch.float32),
            'labels': torch.tensor([1, 2], dtype=torch.long),
            'masks': torch.zeros(2, 224, 224, dtype=torch.uint8),
            'image_id': torch.tensor(idx)
        }
        return image, target

# Create dataset and dataloader  
dataset = MockDataset(4)
dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

# Create model
model = SwinMaskRCNN(num_classes=3)
model.train()

# Test one batch
for batch_idx, (images, targets) in enumerate(dataloader):
    print(f"\nBatch {batch_idx}:")
    print(f"  Number of images: {len(images)}")
    print(f"  Number of targets: {len(targets)}")
    for i, (img, tgt) in enumerate(zip(images, targets)):
        print(f"  Image {i}: shape={img.shape}")
        print(f"  Target {i}: boxes shape={tgt['boxes'].shape}, labels shape={tgt['labels'].shape}")
    
    # Try forward pass
    try:
        with torch.no_grad():
            loss_dict = model(images, targets)
            print("  Loss computation successful")
            for k, v in loss_dict.items():
                print(f"    {k}: {v.item():.4f}")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    break  # Just test first batch