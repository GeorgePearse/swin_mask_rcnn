"""Simple test to verify predictions after bias fix."""
import torch
import numpy as np
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.data.dataset import CocoDataset
from swin_maskrcnn.data.transforms_simple import get_transform_simple
from torch.utils.data import DataLoader
from swin_maskrcnn.utils.collate import collate_fn
from swin_maskrcnn.utils.load_coco_weights import load_coco_weights


def test_simple():
    """Simple prediction test."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = SwinMaskRCNN(num_classes=69)
    
    # Load weights
    print("Loading COCO weights...")
    load_coco_weights(model, num_classes=69)
    
    # Check biases
    fc_cls = model.roi_head.bbox_head.fc_cls
    print("\nClassifier biases:")
    print(f"  Background: {fc_cls.bias[0].item():.4f}")
    print(f"  Foreground mean: {fc_cls.bias[1:].mean().item():.4f}")
    print(f"  Foreground range: [{fc_cls.bias[1:].min().item():.4f}, {fc_cls.bias[1:].max().item():.4f}]")
    
    model = model.to(device)
    model.eval()
    
    # Create test data
    val_dataset = CocoDataset(
        root_dir='/home/georgepearse/data/images',
        annotation_file='/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:38.270134_val_ordered.json',
        transforms=get_transform_simple(train=False),
        mode='train'
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Test on first batch
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            if batch_idx > 0:
                break
            
            images = [img.to(device) for img in images]
            
            # Get predictions
            outputs = model(images)
            
            # Analyze output 
            print("\nPrediction results:")
            for i, output in enumerate(outputs):
                if output is not None and 'boxes' in output:
                    num_preds = len(output['boxes'])
                    print(f"  Image {i}: {num_preds} detections")
                    
                    if num_preds > 0:
                        scores = output['scores'].cpu().numpy()
                        labels = output['labels'].cpu().numpy()
                        
                        print(f"    Score range: [{scores.min():.4f}, {scores.max():.4f}]")
                        print(f"    Score mean: {scores.mean():.4f}")
                        print(f"    Label distribution: {np.bincount(labels)}")
                        
                        # Check score thresholds
                        for thresh in [0.05, 0.1, 0.3, 0.5]:
                            count = (scores >= thresh).sum()
                            print(f"    Scores >= {thresh}: {count}")
                else:
                    print(f"  Image {i}: No detections")
    
    print("\nDone!")


if __name__ == "__main__":
    test_simple()