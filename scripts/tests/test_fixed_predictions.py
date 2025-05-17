"""Test script to verify predictions are now being generated."""
import torch
from pathlib import Path
import yaml

from swin_maskrcnn.data.dataset import CocoDataset
from swin_maskrcnn.data.transforms_simple import get_transform_simple
from torch.utils.data import DataLoader
from swin_maskrcnn.utils.collate import collate_fn
from scripts.train import MaskRCNNLightningModule
from scripts.config.training_config import TrainingConfig
from pycocotools.coco import COCO


def test_predictions():
    """Test if the fix generates predictions."""
    
    # Load config
    config_path = Path("scripts/config/config.yaml")
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = TrainingConfig(**config_dict)
    
    # Create validation dataset  
    val_dataset = CocoDataset(
        root_dir=str(config.img_root),
        annotation_file=str(config.val_ann),
        transforms=get_transform_simple(train=False),
        mode='train'
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create COCO object for validation
    val_coco = COCO(str(config.val_ann))
    
    # Create model
    model = MaskRCNNLightningModule(
        config=config,
        val_coco=val_coco
    )
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"Testing on device: {device}")
    
    # Check biases
    fc_cls = model.model.roi_head.bbox_head.fc_cls
    bias = fc_cls.bias.detach().cpu().numpy()
    print(f"\nClassifier biases:")
    print(f"  Background bias: {bias[0]:.4f}")
    print(f"  Object bias mean: {bias[1:].mean():.4f}")
    print(f"  Object bias range: [{bias[1:].min():.4f}, {bias[1:].max():.4f}]")
    
    # Test on a few batches
    total_predictions = 0
    batch_count = 0
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            if batch_idx >= 5:  # Test only 5 batches
                break
                
            # Move to device
            images = [img.to(device) for img in images]
            
            # Make predictions
            outputs = model.model(images)
            
            # Count predictions
            batch_predictions = 0
            for i, output in enumerate(outputs):
                if output is not None and 'boxes' in output:
                    num_preds = len(output['boxes'])
                    batch_predictions += num_preds
                    
                    if num_preds > 0:
                        scores = output['scores'].cpu().numpy()
                        print(f"\nBatch {batch_idx}, Image {i}:")
                        print(f"  Predictions: {num_preds}")
                        print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
                        print(f"  Score mean: {scores.mean():.4f}")
                        
                        # Show score distribution
                        thresholds = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
                        for t in thresholds:
                            count = (scores >= t).sum()
                            print(f"  Scores >= {t}: {count}")
            
            print(f"\nBatch {batch_idx}: {batch_predictions} total predictions")
            total_predictions += batch_predictions
            batch_count += 1
    
    print(f"\n=== Summary ===")
    print(f"Total predictions across {batch_count} batches: {total_predictions}")
    print(f"Average predictions per batch: {total_predictions / batch_count:.1f}")
    
    # Try both training and inference mode
    print("\n=== Testing with model.predict() ===")
    model.train()  # Reset to training mode
    model.eval()   # Back to eval
    
    # Test predict function
    for batch_idx, (images, targets) in enumerate(val_loader):
        if batch_idx >= 1:
            break
            
        images = [img.to(device) for img in images]
        
        # Use predict function
        predictions = model.model.predict(images, score_threshold=0.05)
        
        for i, pred in enumerate(predictions):
            print(f"\nPredict function - Image {i}:")
            print(f"  Detections: {len(pred['boxes'])}")
            if len(pred['boxes']) > 0:
                scores = pred['scores'].cpu().numpy()
                print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")


if __name__ == "__main__":
    test_predictions()