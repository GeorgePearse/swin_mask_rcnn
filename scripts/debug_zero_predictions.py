"""Debug script to understand why zero predictions are generated during validation."""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.data.dataset import CocoDataset
from swin_maskrcnn.data.transforms_simple import get_transform_simple
from torch.utils.data import DataLoader
from swin_maskrcnn.utils.collate import collate_fn
from swin_maskrcnn.utils.logging import setup_logger
from swin_maskrcnn.utils.load_coco_weights import load_coco_weights


def analyze_layer_outputs(module, input, output):
    """Hook function to analyze layer outputs."""
    if isinstance(output, torch.Tensor):
        print(f"{module.__class__.__name__}: output shape={output.shape}, "
              f"min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
    elif isinstance(output, (list, tuple)):
        print(f"{module.__class__.__name__}: output is {type(output).__name__} with {len(output)} elements")
        for i, out in enumerate(output):
            if isinstance(out, torch.Tensor):
                print(f"  [{i}] shape={out.shape}, min={out.min():.4f}, max={out.max():.4f}, mean={out.mean():.4f}")


def trace_rpn_flow(model, features):
    """Trace the flow through RPN to understand where predictions fail."""
    print("\n=== RPN Flow Analysis ===")
    
    # Get RPN outputs
    rpn_cls_scores, rpn_bbox_preds = model.rpn_head(features)
    
    print("\nRPN outputs:")
    for i, (cls, bbox) in enumerate(zip(rpn_cls_scores, rpn_bbox_preds)):
        print(f"Level {i}:")
        print(f"  cls_scores: shape={cls.shape}, min={cls.min():.4f}, max={cls.max():.4f}, mean={cls.mean():.4f}")
        print(f"  bbox_preds: shape={bbox.shape}, min={bbox.min():.4f}, max={bbox.max():.4f}, mean={bbox.mean():.4f}")
    
    # Get proposals
    img_metas = [{'img_shape': (800, 1333), 'ori_shape': (800, 1333)}]
    proposals = model.rpn_head.get_proposals(
        rpn_cls_scores, rpn_bbox_preds,
        [(800, 1333)],  # img_shapes
        {'nms_pre': 1000, 'nms_thr': 0.7, 'max_per_img': 1000}
    )
    
    print(f"\nNumber of proposals generated: {[len(p) for p in proposals]}")
    
    if proposals and len(proposals[0]) > 0:
        print(f"Proposal shapes: {[p.shape for p in proposals]}")
        # Proposals are in format [x1, y1, x2, y2] without scores
        print(f"First few proposals:\n{proposals[0][:5]}")
    
    return proposals


def trace_roi_flow(model, features, proposals):
    """Trace the flow through ROI head to understand where predictions fail."""
    print("\n=== ROI Head Flow Analysis ===")
    
    if not proposals or len(proposals[0]) == 0:
        print("No proposals to process in ROI head")
        return None
    
    # ROI head forward for detection  
    detections = model.roi_head(features, proposals)
    
    print(f"\nNumber of detections: {[len(d.get('boxes', [])) for d in detections]}")
    
    for i, det in enumerate(detections):
        if det and 'scores' in det and len(det['scores']) > 0:
            print(f"Image {i} detections:")
            print(f"  Boxes: {det['boxes'].shape}")
            print(f"  Scores: min={det['scores'].min():.4f}, max={det['scores'].max():.4f}")
            print(f"  Labels: {det['labels'].unique().tolist()}")
        else:
            print(f"Image {i}: No detections")
    
    return detections


def debug_full_pipeline():
    """Debug the full prediction pipeline step by step."""
    
    # Setup logging
    logger = setup_logger(name="debug", log_dir=".", level="DEBUG")
    
    # Load config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = SwinMaskRCNN(num_classes=69)
    
    # Load COCO weights
    print("\nLoading COCO pretrained weights...")
    missing_keys, unexpected_keys = load_coco_weights(model, num_classes=69)
    print(f"Missing keys: {len(missing_keys)}")
    print(f"Unexpected keys: {len(unexpected_keys)}")
    
    # Check biases after loading
    if hasattr(model.roi_head, 'fc_cls'):
        cls_bias = model.roi_head.fc_cls.bias.detach().cpu().numpy()
        print(f"\nClassification biases after loading:")
        print(f"  Background bias: {cls_bias[0]:.4f}")
        print(f"  Object bias mean: {cls_bias[1:].mean():.4f}")
        print(f"  Object bias range: [{cls_bias[1:].min():.4f}, {cls_bias[1:].max():.4f}]")
    
    model = model.to(device)
    model.eval()
    
    # Create validation dataset
    print("\nLoading validation dataset...")
    val_dataset = CocoDataset(
        root_dir='/home/georgepearse/data/images',
        annotation_file='/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:38.270134_val_ordered.json',
        transforms=get_transform_simple(train=False),
        mode='train'
    )
    
    # Create dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Process first batch
    print("\nProcessing first batch...")
    for batch_idx, (images, targets) in enumerate(val_loader):
        if batch_idx > 0:
            break
            
        # Move to device
        images = [img.to(device) for img in images]
        
        print(f"\nImage shape: {images[0].shape}")
        print(f"Image range: [{images[0].min():.3f}, {images[0].max():.3f}]")
        
        # Get ground truth info
        print(f"\nGround truth:")
        print(f"  Number of objects: {len(targets[0]['boxes'])}")
        if len(targets[0]['boxes']) > 0:
            print(f"  Box areas: {(targets[0]['boxes'][:, 2] - targets[0]['boxes'][:, 0]) * (targets[0]['boxes'][:, 3] - targets[0]['boxes'][:, 1])}")
            print(f"  Labels: {targets[0]['labels'].tolist()}")
        
        with torch.no_grad():
            # Forward through backbone
            print("\n=== Backbone Analysis ===")
            features = model.backbone(torch.stack(images))
            for i, feat in enumerate(features):
                print(f"Feature level {i}: shape={feat.shape}, "
                      f"min={feat.min():.4f}, max={feat.max():.4f}, "
                      f"mean={feat.mean():.4f}, std={feat.std():.4f}")
            
            # Forward through FPN
            print("\n=== FPN Analysis ===")
            fpn_features = model.neck(features)
            for i, feat in enumerate(fpn_features):
                print(f"FPN level {i}: shape={feat.shape}, "
                      f"min={feat.min():.4f}, max={feat.max():.4f}, "
                      f"mean={feat.mean():.4f}, std={feat.std():.4f}")
            
            # Trace RPN
            proposals = trace_rpn_flow(model, fpn_features)
            
            # Trace ROI head
            detections = trace_roi_flow(model, fpn_features, proposals)
            
            # Full forward pass
            print("\n=== Full Forward Pass ===")
            full_detections = model(images)
            
            print(f"\nFinal detections: {[len(d.get('boxes', [])) for d in full_detections]}")
            
                    # Analyze score distributions
            all_scores = []
            for det in full_detections:
                if 'scores' in det and len(det['scores']) > 0:
                    all_scores.extend(det['scores'].cpu().numpy())
            
            if all_scores:
                all_scores = np.array(all_scores)
                print(f"\nScore distribution ({len(all_scores)} detections):")
                print(f"  Min: {all_scores.min():.4f}")
                print(f"  Max: {all_scores.max():.4f}")
                print(f"  Mean: {all_scores.mean():.4f}")
                print(f"  Std: {all_scores.std():.4f}")
                
                # Score histogram
                hist, bins = np.histogram(all_scores, bins=10)
                print("\n  Score histogram:")
                for i in range(len(hist)):
                    print(f"    [{bins[i]:.3f}-{bins[i+1]:.3f}]: {hist[i]}")
            else:
                print("\nNo scores found in detections!")
                print(f"Detection keys: {[list(d.keys()) for d in full_detections]}")
                
            # Debug ROI head internals
            print("\n=== ROI Head Debug ===")
            with torch.no_grad():
                # Let's manually trace through ROI head
                roi_head = model.roi_head
                
                # Extract RoI features  
                if proposals and len(proposals[0]) > 0:
                    print(f"\nExtracting ROI features for {len(proposals[0])} proposals...")
                    # The roi_head should have a roi_extractor
                    roi_extractor = roi_head.roi_extractor
                    
                    # Use all FPN levels for ROI extraction (typically first 5)
                    roi_feats = roi_extractor(
                        fpn_features[:5],  # Use all FPN levels
                        proposals
                    )
                    print(f"ROI features shape: {roi_feats.shape}")
                    print(f"ROI features stats: min={roi_feats.min():.4f}, max={roi_feats.max():.4f}, mean={roi_feats.mean():.4f}")
                    
                    # Forward through bbox head
                    bbox_head = roi_head.bbox_head
                    
                    # Use the forward function of bbox_head
                    cls_score, bbox_pred = bbox_head(roi_feats)
                    
                    print(f"\nClassification scores: shape={cls_score.shape}")
                    print(f"Classification scores stats: min={cls_score.min():.4f}, max={cls_score.max():.4f}, mean={cls_score.mean():.4f}")
                    
                    # Apply softmax
                    cls_prob = torch.softmax(cls_score, dim=1)
                    print(f"\nClass probabilities after softmax:")
                    print(f"  Background prob: min={cls_prob[:, 0].min():.4f}, max={cls_prob[:, 0].max():.4f}, mean={cls_prob[:, 0].mean():.4f}")
                    print(f"  Foreground prob: min={cls_prob[:, 1:].min():.4f}, max={cls_prob[:, 1:].max():.4f}, mean={cls_prob[:, 1:].mean():.4f}")
                    
                    # Get predictions
                    scores, pred_labels = cls_prob.max(dim=1)
                    print(f"\nPredicted labels: {torch.bincount(pred_labels).tolist()}")
                    print(f"Number of non-background predictions: {(pred_labels > 0).sum()}")
                    
                    # Check biases
                    print(f"\nClassifier biases:")
                    print(f"  Background bias: {bbox_head.fc_cls.bias[0].item():.4f}")
                    print(f"  Object bias mean: {bbox_head.fc_cls.bias[1:].mean().item():.4f}")
                    print(f"  Object bias std: {bbox_head.fc_cls.bias[1:].std().item():.4f}")
                    print(f"  Object bias range: [{bbox_head.fc_cls.bias[1:].min().item():.4f}, {bbox_head.fc_cls.bias[1:].max().item():.4f}]")
                    
                    # Force trace through inference path
                    print(f"\n=== Inference Path Debug ===")
                    results = roi_head.get_results(cls_score, bbox_pred, None, proposals, None)
                    print(f"Results from ROI head: {len(results)} images")
                    for i, res in enumerate(results):
                        print(f"  Image {i}: {len(res['boxes'])} detections")
                        if len(res['boxes']) > 0:
                            print(f"    Score range: [{res['scores'].min():.4f}, {res['scores'].max():.4f}]")
                            print(f"    Label distribution: {torch.bincount(res['labels']).tolist()}")
    
    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    debug_full_pipeline()