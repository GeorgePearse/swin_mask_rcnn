"""Debug validation predictions with absolutely no thresholding."""
import sys
sys.path.append("/home/georgepearse/swin_maskrcnn")

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from swin_maskrcnn.data.dataset import CocoDataset
from swin_maskrcnn.data.transforms_simple import get_transform_simple
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.utils.collate import collate_fn
from swin_maskrcnn.utils.logging import setup_logger


def main():
    # Hardcoded paths
    checkpoint_path = Path("/home/georgepearse/swin_maskrcnn/test_checkpoints/run_20250517_180326/checkpoint-epoch=00-step=200.ckpt")
    img_root = Path("/home/georgepearse/data/images")
    val_ann = Path("/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:38.270134_val_ordered.json")
    
    logger = setup_logger(name="debug")
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract model state from Lightning checkpoint
    if 'state_dict' in checkpoint:
        # Lightning checkpoint format
        state_dict = checkpoint['state_dict']
        # Remove 'model.' prefix from keys
        model_state = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                model_state[k[6:]] = v
            else:
                model_state[k] = v
    else:
        model_state = checkpoint.get('model_state_dict', checkpoint)
    
    # Check classification biases
    fc_cls_bias = model_state.get('roi_head.bbox_head.fc_cls.bias')
    if fc_cls_bias is not None:
        fc_cls_bias_np = fc_cls_bias.numpy()
        logger.info(f"Classification bias shape: {fc_cls_bias_np.shape}")
        logger.info(f"Background bias: {fc_cls_bias_np[0]:.4f}")
        logger.info(f"Foreground biases - mean: {fc_cls_bias_np[1:].mean():.4f}, "
                   f"min: {fc_cls_bias_np[1:].min():.4f}, max: {fc_cls_bias_np[1:].max():.4f}")
    else:
        logger.info("Could not find roi_head.bbox_head.fc_cls.bias in checkpoint")
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = SwinMaskRCNN(num_classes=69).to(device)
    
    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
    logger.info(f"Missing keys: {len(missing_keys)}")
    logger.info(f"Unexpected keys: {len(unexpected_keys)}")
    
    model.eval()
    
    # Check biases after loading
    logger.info("\nBiases after loading:")
    with torch.no_grad():
        fc_cls_bias = model.roi_head.bbox_head.fc_cls.bias.cpu().numpy()
        logger.info(f"Background bias: {fc_cls_bias[0]:.4f}")
        logger.info(f"Foreground biases - mean: {fc_cls_bias[1:].mean():.4f}, "
                   f"min: {fc_cls_bias[1:].min():.4f}, max: {fc_cls_bias[1:].max():.4f}")
    
    # Create validation dataset
    val_dataset = CocoDataset(
        root_dir=str(img_root),
        annotation_file=str(val_ann),
        transforms=get_transform_simple(train=False),
        mode='train'  # To get targets for debugging
    )
    
    # Only load a few images for debugging
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False
    )
    
    # Test predictions on first batch
    logger.info("\nTesting predictions on first batch...")
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            if batch_idx >= 2:  # Test first 2 batches
                break
                
            logger.info(f"\nBatch {batch_idx}:")
            logger.info(f"Number of images: {len(images)}")
            
            # Move to device
            images = [img.to(device) for img in images]
            
            # Get predictions
            predictions = model(images)
            
            # Log detailed prediction info
            for i, pred in enumerate(predictions):
                logger.info(f"\nImage {i}:")
                logger.info(f"  Target boxes: {len(targets[i]['boxes'])}")
                
                if pred is None:
                    logger.info("  Prediction is None!")
                    continue
                
                if 'boxes' not in pred:
                    logger.info("  No 'boxes' key in prediction!")
                    logger.info(f"  Keys: {pred.keys()}")
                    continue
                
                num_preds = len(pred['boxes'])
                logger.info(f"  Number of predictions: {num_preds}")
                
                if num_preds > 0:
                    scores = pred['scores'].cpu().numpy()
                    labels = pred['labels'].cpu().numpy()
                    
                    # Score statistics
                    logger.info(f"  Score stats - min: {scores.min():.4f}, "
                               f"max: {scores.max():.4f}, mean: {scores.mean():.4f}")
                    
                    # Count predictions above various thresholds
                    thresholds = [0.0, 0.001, 0.01, 0.05, 0.1, 0.3, 0.5]
                    for thr in thresholds:
                        count = (scores >= thr).sum()
                        logger.info(f"  Predictions with score >= {thr}: {count}")
                    
                    # Label distribution
                    unique_labels, counts = np.unique(labels, return_counts=True)
                    logger.info(f"  Label distribution: {dict(zip(unique_labels, counts))}")
                    
                    # Show top 5 predictions
                    logger.info("  Top 5 predictions:")
                    top_idx = np.argsort(scores)[::-1][:5]
                    for idx in top_idx:
                        logger.info(f"    Label {labels[idx]}, Score: {scores[idx]:.4f}")
    
    logger.info("\nDebug complete!")


if __name__ == "__main__":
    main()