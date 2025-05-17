import torch
import torch.nn as nn
from typing import List, Dict, Any
import logging
import numpy as np
from datetime import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.models.swin import SwinConfig
from swin_maskrcnn.data.dataset import CocoDataset
from swin_maskrcnn.data.transforms_simple import get_transforms
from swin_maskrcnn.utils.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def inspect_prediction_head(model: nn.Module) -> None:
    """Inspect the prediction head of the model."""
    # Get the ROI head
    roi_head = model.roi_head
    
    # Check the classifier weights and biases
    classifier = roi_head.fc_cls
    logger.info(f"Classifier shape: {classifier.weight.shape}")
    logger.info(f"Classifier bias shape: {classifier.bias.shape}")
    
    # Analyze the bias values
    bias_values = classifier.bias.detach().cpu().numpy()
    logger.info(f"Background class bias: {bias_values[0]:.4f}")
    logger.info(f"Non-background bias mean: {bias_values[1:].mean():.4f}")
    logger.info(f"Non-background bias std: {bias_values[1:].std():.4f}")
    logger.info(f"Non-background bias min: {bias_values[1:].min():.4f}")
    logger.info(f"Non-background bias max: {bias_values[1:].max():.4f}")
    
    # Analyze weights
    weights = classifier.weight.detach().cpu().numpy()
    logger.info(f"Background weight norm: {np.linalg.norm(weights[0]):.4f}")
    logger.info(f"Non-background weight norm mean: {np.mean([np.linalg.norm(w) for w in weights[1:]]):.4f}")
    
    # Check bbox regressor
    bbox_pred = roi_head.fc_reg
    logger.info(f"BBox regressor shape: {bbox_pred.weight.shape}")
    logger.info(f"BBox regressor bias norm: {np.linalg.norm(bbox_pred.bias.detach().cpu().numpy()):.4f}")
    
    # Check mask predictor
    if hasattr(roi_head, 'mask_head'):
        mask_head = roi_head.mask_head
        last_conv = mask_head.convs[-1]
        logger.info(f"Mask predictor output channels: {last_conv.out_channels}")


def analyze_predictions(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> None:
    """Analyze model predictions in detail."""
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 2:  # Just analyze a few batches
                break
            
            images = batch['images'].to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in target.items()} 
                       for target in batch['targets']]
            
            # Get features from the backbone
            features = model.backbone(images)
            logger.info(f"Backbone features: {[f.shape for f in features]}")
            
            # Get FPN features
            fpn_features = model.neck(features)
            logger.info(f"FPN features: {[f.shape for f in fpn_features]}")
            
            # Get RPN proposals
            proposals = model.rpn(images, features, fpn_features, targets if model.training else None)
            logger.info(f"Number of proposals: {[len(p) for p in proposals]}")
            
            # Get ROI features and predictions
            if proposals[0].shape[0] > 0:
                # Extract ROI features directly
                roi_features = model.roi_head.box_roi_pool(fpn_features, proposals)
                logger.info(f"ROI features shape: {roi_features.shape}")
                
                # Process through box head
                box_features = model.roi_head.box_head(roi_features)
                logger.info(f"Box features shape: {box_features.shape}")
                
                # Get classification scores
                cls_scores = model.roi_head.fc_cls(box_features)
                logger.info(f"Classification scores shape: {cls_scores.shape}")
                
                # Apply softmax to see the actual probabilities
                probs = torch.softmax(cls_scores, dim=-1)
                background_probs = probs[:, 0]
                max_object_probs = probs[:, 1:].max(dim=-1)[0]
                
                logger.info(f"Background probability - mean: {background_probs.mean():.4f}, "
                          f"min: {background_probs.min():.4f}, max: {background_probs.max():.4f}")
                logger.info(f"Max object probability - mean: {max_object_probs.mean():.4f}, "
                          f"min: {max_object_probs.min():.4f}, max: {max_object_probs.max():.4f}")
                
                # Check raw scores before softmax
                logger.info(f"Raw classification scores - mean: {cls_scores.mean():.4f}, "
                          f"std: {cls_scores.std():.4f}, min: {cls_scores.min():.4f}, "
                          f"max: {cls_scores.max():.4f}")
                logger.info(f"Background raw scores - mean: {cls_scores[:, 0].mean():.4f}, "
                          f"std: {cls_scores[:, 0].std():.4f}")
                logger.info(f"Object raw scores - mean: {cls_scores[:, 1:].mean():.4f}, "
                          f"std: {cls_scores[:, 1:].std():.4f}")
                
                # Get predictions with lower threshold
                obj_logits = cls_scores[:, 1:]
                obj_labels = torch.argmax(obj_logits, dim=1) + 1
                obj_scores = torch.max(obj_logits, dim=1)[0]
                
                # Try with very low threshold
                for threshold in [0.5, 0.1, 0.01, 0.001]:
                    keep = obj_scores > threshold
                    logger.info(f"Detections with threshold {threshold}: {keep.sum().item()}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model with correct number of classes
    model = SwinMaskRCNN(num_classes=70).to(device)
    
    # Load checkpoint
    checkpoint_path = "/home/georgepearse/swin_maskrcnn/checkpoints/maskrcnn-swin-s-p4-w7/pytorch_model.bin"
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Missing keys: {len(missing_keys)}")
    logger.info(f"Unexpected keys: {len(unexpected_keys)}")
    
    # Inspect the prediction head
    logger.info("=== Inspecting Prediction Head ===")
    inspect_prediction_head(model)
    
    # Create a small dataset for testing
    val_ann_path = "/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:38.270134_val_ordered.json"
    val_dataset = CocoDataset(
        ann_file=val_ann_path,
        img_dir="/home/georgepearse/data/images",
        transforms=get_transforms(train=False),
        ignore_images_without_annotations=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=val_dataset.collate_fn
    )
    
    # Analyze predictions
    logger.info("\n=== Analyzing Predictions ===")
    analyze_predictions(model, val_loader, device)


if __name__ == "__main__":
    main()