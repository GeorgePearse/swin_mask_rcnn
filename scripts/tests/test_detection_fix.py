"""Quick test to verify detection fix works."""

import torch
import torch.nn as nn
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.data.dataset import CocoDataset
from swin_maskrcnn.data.transforms_simple import get_transform_simple
from swin_maskrcnn.utils.load_coco_weights import load_coco_weights
from torch.utils.data import DataLoader
from swin_maskrcnn.utils.collate import collate_fn


def main():
    # Create model
    print("Creating model...")
    model = SwinMaskRCNN(num_classes=69, frozen_backbone_stages=3)
    
    # Load COCO pretrained weights
    print("Loading COCO pretrained weights...")
    missing_keys, unexpected_keys = load_coco_weights(model, num_classes=69)
    print(f"Missing keys: {len(missing_keys)}")
    print(f"Unexpected keys: {len(unexpected_keys)}")
    
    # Use CPU to avoid memory issues
    device = torch.device("cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Create small dataset
    print("Creating dataset...")
    dataset = CocoDataset(
        root_dir="/home/georgepearse/data/images",
        annotation_file="/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:38.270134_val_ordered.json",
        transforms=get_transform_simple(train=False),
        mode='val'
    )
    
    # Create dataloader with batch size 1
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Test a few batches
    print("\nTesting detection output...")
    model.eval()
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloader):
            if i >= 5:  # Test only 5 images
                break
                
            # Move to device
            images = [img.to(device) for img in images]
            
            # Get predictions
            predictions = model(images)
            
            # Check predictions
            if predictions and len(predictions) > 0:
                pred = predictions[0]
                if 'boxes' in pred and len(pred['boxes']) > 0:
                    print(f"Image {i}: {len(pred['boxes'])} detections")
                    # Print score statistics
                    scores = pred['scores'].cpu().numpy()
                    print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]")
                    print(f"  Mean score: {scores.mean():.3f}")
                else:
                    print(f"Image {i}: No detections")
            else:
                print(f"Image {i}: No predictions")
    
    # Print bias values to verify our changes
    print("\n\nBias initialization check:")
    print(f"RPN cls bias: {model.rpn_head.rpn_cls.bias.data.mean().item():.3f}")
    if hasattr(model.roi_head, 'bbox_head'):
        print(f"ROI cls bias (background): {model.roi_head.bbox_head.fc_cls.bias.data[0].item():.3f}")
        print(f"ROI cls bias (mean objects): {model.roi_head.bbox_head.fc_cls.bias.data[1:].mean().item():.3f}")
    else:
        print(f"ROI cls bias (background): {model.roi_head.fc_cls.bias.data[0].item():.3f}")
        print(f"ROI cls bias (mean objects): {model.roi_head.fc_cls.bias.data[1:].mean().item():.3f}")


if __name__ == "__main__":
    main()