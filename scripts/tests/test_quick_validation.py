"""Quick test to verify detection improvements with new bias initialization."""
import torch
import numpy as np
from pathlib import Path
import sys
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.data.dataset import CocoDataset
from swin_maskrcnn.data.transforms_simple import get_transform_simple
from scripts.config.training_config import TrainingConfig
from torch.utils.data import DataLoader
from swin_maskrcnn.utils.collate import collate_fn

def test_quick_validation():
    """Test model predictions with the new checkpoint."""
    # Load config
    config_path = "/home/georgepearse/swin_maskrcnn/scripts/config/train_with_fixed_biases.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = TrainingConfig(**config_dict)
    
    # Create model
    model = SwinMaskRCNN(
        num_classes=config.num_classes,
        roi_cls_pos_weight=config.roi_cls_pos_weight
    )
    
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
        mode='train'
    )
    
    # Create dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
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
    predictions_per_threshold = {0.1: 0, 0.3: 0, 0.5: 0}
    
    print("\nRunning quick validation...")
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            if batch_idx >= num_batches_to_test:
                break
                
            # Move images to device
            images = [img.to(device) for img in images]
            
            # Get predictions
            outputs = model(images)
            
            # Count predictions
            for output in outputs:
                if 'boxes' in output and len(output['boxes']) > 0:
                    scores = output['scores'].cpu().numpy()
                    total_predictions += len(scores)
                    
                    for threshold in predictions_per_threshold:
                        predictions_per_threshold[threshold] += (scores >= threshold).sum()
    
    print(f"\nTotal predictions in {num_batches_to_test} batches: {total_predictions}")
    for threshold, count in predictions_per_threshold.items():
        print(f"Predictions with score >= {threshold}: {count}")

if __name__ == "__main__":
    test_quick_validation()