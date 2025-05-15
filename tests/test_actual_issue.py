import torch
import sys
sys.path.append('/home/georgepearse/worktrees/remove-mmdetection/machine_learning/packages/swin_maskrcnn')

from swin_maskrcnn.models.rpn import RPNHead
from swin_maskrcnn.data.dataset import CMRDataset
from torch.utils.data import DataLoader
from swin_maskrcnn.utils.collate import collate_fn

def test_actual_data_flow():
    # Let's test with actual data to see what's being passed
    train_dataset = CMRDataset(
        img_dir='/home/georgepearse/data/images',
        ann_file='/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:23.077836_train_ordered.json',
        train=True,
        transform=None
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=2,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Get a batch
    for images, targets in train_loader:
        print(f"Number of images: {len(images)}")
        print(f"Number of targets: {len(targets)}")
        
        for i, target in enumerate(targets):
            print(f"\nTarget {i}:")
            print(f"  boxes shape: {target['boxes'].shape}")
            print(f"  labels shape: {target['labels'].shape}")
            print(f"  First box: {target['boxes'][0] if len(target['boxes']) > 0 else 'No boxes'}")
        
        # Now let's see what gets passed to RPN loss
        gt_bboxes = [t['boxes'] for t in targets]
        print(f"\ngt_bboxes passed to RPN loss:")
        for i, gt_bbox in enumerate(gt_bboxes):
            print(f"  Image {i}: shape={gt_bbox.shape}, type={type(gt_bbox)}")
            
        break

if __name__ == "__main__":
    test_actual_data_flow()