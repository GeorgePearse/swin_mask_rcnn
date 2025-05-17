"""Debug script to understand why validation predictions are not showing up."""

import torch
import numpy as np
from pathlib import Path
import sys
import json
import logging
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.data.dataset import CocoDataset
from swin_maskrcnn.data.transforms_simple import get_transform_simple
from scripts.config.training_config import TrainingConfig
from torch.utils.data import DataLoader

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_validation_predictions():
    """Debug why we're not seeing predictions during validation."""
    # Load config
    config_path = "/home/georgepearse/swin_maskrcnn/scripts/config/train_with_fixed_biases.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = TrainingConfig(**config_dict)
    
    # Create model
    model = SwinMaskRCNN(num_classes=config.num_classes)
    
    # Load checkpoint
    checkpoint_path = config.checkpoint_path
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {len(missing_keys)}")
    print(f"Unexpected keys: {len(unexpected_keys)}")
    
    # Check biases
    cls_bias = model.roi_head.bbox_head.fc_cls.bias.detach().cpu().numpy()
    print(f"\nClassifier biases:")
    print(f"Background bias: {cls_bias[0]:.4f}")
    print(f"Object bias mean: {cls_bias[1:].mean():.4f}")
    print(f"Object bias range: [{cls_bias[1:].min():.4f}, {cls_bias[1:].max():.4f}]")
    
    # Create dataset
    val_dataset = CocoDataset(
        root_dir=str(config.img_root),
        annotation_file=str(config.val_ann),
        transforms=get_transform_simple(train=False),
        mode='train'  # Still use train mode to get targets
    )
    
    # Create dataloader with small batch
    from swin_maskrcnn.utils.collate import collate_fn
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Test on a few batches
    num_batches_to_test = 5
    total_predictions = 0
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            if batch_idx >= num_batches_to_test:
                break
                
            print(f"\nBatch {batch_idx + 1}:")
            
            # Move images to device
            images = [img.to(device) for img in images]
            
            # Get predictions
            outputs = model(images)
            
            # Debug outputs
            for i, output in enumerate(outputs):
                if output is None:
                    print(f"  Image {i}: No output (None)")
                    continue
                
                print(f"  Image {i}:")
                print(f"    Keys: {output.keys()}")
                
                if 'boxes' in output:
                    num_boxes = len(output['boxes'])
                    print(f"    Number of boxes: {num_boxes}")
                    
                    if num_boxes > 0:
                        scores = output['scores'].cpu().numpy()
                        labels = output['labels'].cpu().numpy()
                        
                        print(f"    Score range: [{scores.min():.4f}, {scores.max():.4f}]")
                        print(f"    Score mean: {scores.mean():.4f}")
                        print(f"    Labels: {np.unique(labels)}")
                        
                        # Look at score distribution
                        for threshold in [0.001, 0.01, 0.05, 0.1, 0.3, 0.5]:
                            count = (scores >= threshold).sum()
                            print(f"    Predictions with score >= {threshold}: {count}")
                        
                        # Show top predictions
                        if num_boxes > 0:
                            top_indices = np.argsort(scores)[-5:][::-1]
                            print(f"    Top 5 predictions:")
                            for idx in top_indices:
                                print(f"      Label: {labels[idx]}, Score: {scores[idx]:.4f}")
                        
                        total_predictions += num_boxes
                else:
                    print(f"    No boxes in output")
    
    print(f"\nTotal predictions across {num_batches_to_test} batches: {total_predictions}")
    
    # Let's also check RPN directly
    print("\n=== Checking RPN directly ===")
    with torch.no_grad():
        # Get a single image
        images, targets = next(iter(val_loader))
        images = [images[0].to(device)]
        
        # Get features
        features = model.backbone(images[0].unsqueeze(0))
        if model.neck is not None:
            features = model.neck(features)
        
        # Get RPN predictions
        rpn_outputs = model.rpn(features, targets=[targets[0]])
        
        print(f"RPN outputs: {type(rpn_outputs)}")
        if isinstance(rpn_outputs, tuple) and len(rpn_outputs) == 2:
            proposals, losses = rpn_outputs
            print(f"Number of proposals: {[len(p) for p in proposals]}")
        else:
            print(f"Unexpected RPN output format")
    
    # Let's also check ROI head predictions before filtering
    print("\n=== Checking ROI Head directly ===")
    with torch.no_grad():
        # Use the same features and proposals
        if isinstance(rpn_outputs, tuple) and len(rpn_outputs) == 2:
            proposals, _ = rpn_outputs
            
            # Pass through ROI head
            roi_outputs = model.roi_head(features, proposals, targets=None)
            
            print(f"ROI outputs: {type(roi_outputs)}")
            if isinstance(roi_outputs, list):
                for i, output in enumerate(roi_outputs):
                    print(f"  Image {i}: {output.keys() if output else 'None'}")
                    if output and 'scores' in output:
                        scores = output['scores'].cpu().numpy()
                        print(f"    Raw score range: [{scores.min():.4f}, {scores.max():.4f}]")
                        print(f"    Number of predictions: {len(scores)}")

if __name__ == "__main__":
    debug_validation_predictions()