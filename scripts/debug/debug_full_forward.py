"""Debug the full forward pass to understand why scores are still low."""
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import yaml

from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.data.dataset import CocoDataset
from swin_maskrcnn.data.transforms_simple import get_transform_simple
from torch.utils.data import DataLoader
from swin_maskrcnn.utils.collate import collate_fn
from swin_maskrcnn.utils.load_coco_weights import load_coco_weights


def analyze_forward_pass():
    """Analyze the forward pass step by step."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = SwinMaskRCNN(num_classes=69)
    
    # Load COCO weights
    print("Loading COCO weights...")
    missing_keys, unexpected_keys = load_coco_weights(model, num_classes=69)
    
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
    
    # Get first batch
    for batch_idx, (images, targets) in enumerate(val_loader):
        if batch_idx > 0:
            break
        
        images = [img.to(device) for img in images]
        
        with torch.no_grad():
            # Forward through backbone and FPN
            features = model.backbone(torch.stack(images))
            fpn_features = model.neck(features)
            
            # RPN forward
            rpn_cls_scores, rpn_bbox_preds = model.rpn_head(fpn_features)
            
            # Get proposals
            proposals = model.rpn_head.get_proposals(
                rpn_cls_scores, rpn_bbox_preds,
                [(img.shape[-2], img.shape[-1]) for img in images],
                {'nms_pre': 1000, 'nms_thr': 0.7, 'max_per_img': 1000}
            )
            
            print(f"Number of proposals: {[len(p) for p in proposals]}")
            
            # Forward through ROI head manually
            roi_head = model.roi_head
            
            # Extract ROI features
            roi_feats = roi_head.bbox_roi_extractor(
                fpn_features[:5],  # Use first 5 FPN levels
                proposals
            )
            print(f"ROI features shape: {roi_feats.shape}")
            print(f"ROI features stats: min={roi_feats.min():.4f}, max={roi_feats.max():.4f}, mean={roi_feats.mean():.4f}, std={roi_feats.std():.4f}")
            
            # Forward through bbox head
            bbox_head = roi_head.bbox_head
            
            # Flatten features
            x = roi_feats.flatten(1)
            print(f"Flattened shape: {x.shape}")
            
            # FC layers
            x1 = F.relu(bbox_head.shared_fc1(x))
            print(f"After FC1: shape={x1.shape}, min={x1.min():.4f}, max={x1.max():.4f}, mean={x1.mean():.4f}, nonzero={(x1 > 0).float().mean():.4f}")
            
            x2 = F.relu(bbox_head.shared_fc2(x1))
            print(f"After FC2: shape={x2.shape}, min={x2.min():.4f}, max={x2.max():.4f}, mean={x2.mean():.4f}, nonzero={(x2 > 0).float().mean():.4f}")
            
            # Classification
            cls_score = bbox_head.fc_cls(x2)
            print(f"\nClassification scores: shape={cls_score.shape}")
            print(f"Raw logits stats: min={cls_score.min():.4f}, max={cls_score.max():.4f}, mean={cls_score.mean():.4f}, std={cls_score.std():.4f}")
            
            # Check logits per class
            print("\nLogits by class:")
            print(f"  Background logits: min={cls_score[:, 0].min():.4f}, max={cls_score[:, 0].max():.4f}, mean={cls_score[:, 0].mean():.4f}")
            print(f"  Foreground logits: min={cls_score[:, 1:].min():.4f}, max={cls_score[:, 1:].max():.4f}, mean={cls_score[:, 1:].mean():.4f}")
            
            # Apply softmax
            cls_prob = F.softmax(cls_score, dim=1)
            print(f"\nAfter softmax:")
            print(f"  Background prob: min={cls_prob[:, 0].min():.4f}, max={cls_prob[:, 0].max():.4f}, mean={cls_prob[:, 0].mean():.4f}")
            print(f"  Foreground prob: min={cls_prob[:, 1:].min():.4f}, max={cls_prob[:, 1:].max():.4f}, mean={cls_prob[:, 1:].mean():.4f}")
            
            # Check FC layer parameters
            print("\n=== FC Layer Analysis ===")
            print(f"FC1 weight: shape={bbox_head.shared_fc1.weight.shape}, norm={bbox_head.shared_fc1.weight.norm():.4f}")
            print(f"FC1 bias: norm={bbox_head.shared_fc1.bias.norm():.4f}, mean={bbox_head.shared_fc1.bias.mean():.4f}")
            
            print(f"FC2 weight: shape={bbox_head.shared_fc2.weight.shape}, norm={bbox_head.shared_fc2.weight.norm():.4f}")
            print(f"FC2 bias: norm={bbox_head.shared_fc2.bias.norm():.4f}, mean={bbox_head.shared_fc2.bias.mean():.4f}")
            
            print(f"FC_cls weight: shape={bbox_head.fc_cls.weight.shape}")
            print(f"  Background weight norm: {bbox_head.fc_cls.weight[0].norm():.4f}")
            print(f"  Foreground weight norm mean: {bbox_head.fc_cls.weight[1:].norm(dim=1).mean():.4f}")
            print(f"  Weight variance: {bbox_head.fc_cls.weight.var():.6f}")
            
            # Try manual forward with debugging
            print("\n=== Manual calculation ===")
            # Take first ROI feature
            first_roi = roi_feats[0].flatten()
            print(f"First ROI feature norm: {first_roi.norm():.4f}")
            
            # Through FC1
            fc1_out = F.relu(F.linear(first_roi, bbox_head.shared_fc1.weight, bbox_head.shared_fc1.bias))
            print(f"After FC1: norm={fc1_out.norm():.4f}, nonzero={(fc1_out > 0).float().mean():.4f}")
            
            # Through FC2
            fc2_out = F.relu(F.linear(fc1_out, bbox_head.shared_fc2.weight, bbox_head.shared_fc2.bias))
            print(f"After FC2: norm={fc2_out.norm():.4f}, nonzero={(fc2_out > 0).float().mean():.4f}")
            
            # Classification
            cls_out = F.linear(fc2_out, bbox_head.fc_cls.weight, bbox_head.fc_cls.bias)
            print(f"Classification output: {cls_out[:5]}")  # Show first 5 values
            
            # Softmax
            cls_prob_manual = F.softmax(cls_out, dim=0)
            print(f"Probabilities: background={cls_prob_manual[0]:.4f}, max_foreground={cls_prob_manual[1:].max():.4f}")
    
    print("\n=== Analysis complete ===")


if __name__ == "__main__":
    analyze_forward_pass()