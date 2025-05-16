# Loss Function Modifications for Partial Annotations

## Overview

This document outlines the modifications needed to loss functions to properly handle partial annotations with explicit background regions.

## Core Concept

Only compute losses within annotated regions, ignoring unannotated areas completely. This prevents the model from being penalized for predictions in regions that weren't annotated.

## Loss Components to Modify

### 1. RPN Loss Modifications

```python
class PartialAnnotationRPNLoss(nn.Module):
    """RPN loss that respects annotated regions."""
    
    def forward(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, annotated_masks):
        """
        Args:
            cls_scores: (N, A, H, W) objectness scores
            bbox_preds: (N, A*4, H, W) bbox predictions
            gt_bboxes: list of (K, 4) ground truth boxes
            gt_labels: list of (K,) ground truth labels
            annotated_masks: (N, H, W) binary masks of annotated regions
        """
        device = cls_scores.device
        batch_size = cls_scores.size(0)
        
        # Generate anchors
        anchors = self.anchor_generator(featmap_sizes, device)
        
        # Assign anchors to ground truth
        anchor_targets = []
        valid_masks = []
        
        for i in range(batch_size):
            # Get annotated region mask for this image
            ann_mask = annotated_masks[i]
            
            # Only consider anchors in annotated regions
            anchor_centers = self._get_anchor_centers(anchors)
            valid_anchor_mask = self._sample_locations(anchor_centers, ann_mask)
            
            # Assign valid anchors to GT
            targets = self._assign_anchors(
                anchors[valid_anchor_mask],
                gt_bboxes[i],
                gt_labels[i]
            )
            
            anchor_targets.append(targets)
            valid_masks.append(valid_anchor_mask)
        
        # Compute losses only on valid anchors
        cls_loss = self._cls_loss(cls_scores, anchor_targets, valid_masks)
        reg_loss = self._reg_loss(bbox_preds, anchor_targets, valid_masks)
        
        return cls_loss + reg_loss
```

### 2. ROI Head Loss Modifications

```python
class PartialAnnotationROILoss(nn.Module):
    """ROI head loss for partial annotations."""
    
    def forward(self, cls_scores, bbox_preds, mask_preds, 
                proposals, gt_bboxes, gt_labels, gt_masks,
                annotated_masks, background_masks):
        """Compute losses only for proposals in annotated regions."""
        
        # Filter proposals by annotated regions
        valid_proposals = []
        proposal_targets = []
        
        for i in range(batch_size):
            ann_mask = annotated_masks[i]
            bg_mask = background_masks[i]
            
            # Check which proposals are in annotated regions
            prop_centers = self._get_proposal_centers(proposals[i])
            in_annotated = self._sample_locations(prop_centers, ann_mask)
            
            valid_props = proposals[i][in_annotated]
            
            # Assign proposals to GT or background
            targets = self._assign_proposals(
                valid_props,
                gt_bboxes[i],
                gt_labels[i],
                bg_mask
            )
            
            valid_proposals.append(valid_props)
            proposal_targets.append(targets)
        
        # Compute losses
        cls_loss = self._classification_loss(cls_scores, proposal_targets)
        bbox_loss = self._bbox_regression_loss(bbox_preds, proposal_targets)
        mask_loss = self._mask_loss(mask_preds, proposal_targets, gt_masks)
        
        return {
            'loss_cls': cls_loss,
            'loss_bbox': bbox_loss,
            'loss_mask': mask_loss
        }
```

### 3. Mask Loss Modifications

```python
def compute_mask_loss_partial(mask_preds, mask_targets, annotated_masks):
    """Compute mask loss only within annotated regions."""
    
    if len(mask_targets) == 0:
        return mask_preds.sum() * 0
    
    # Stack targets and predictions
    mask_preds = mask_preds.view(-1, H, W)
    mask_targets = torch.cat(mask_targets, dim=0)
    roi_annotated_masks = extract_roi_masks(annotated_masks, proposals)
    
    # Binary cross entropy with annotation mask
    loss = F.binary_cross_entropy_with_logits(
        mask_preds,
        mask_targets,
        reduction='none'
    )
    
    # Apply annotation mask
    loss = loss * roi_annotated_masks
    
    # Average over valid pixels only
    valid_pixels = roi_annotated_masks.sum()
    if valid_pixels > 0:
        loss = loss.sum() / valid_pixels
    else:
        loss = loss.sum() * 0
    
    return loss
```

### 4. Background Handling

```python
class BackgroundAwareLoss(nn.Module):
    """Loss that uses explicit background annotations."""
    
    def _assign_proposals(self, proposals, gt_bboxes, gt_labels, background_mask):
        """Assign proposals considering explicit background."""
        
        # Standard positive assignment based on IoU
        ious = box_iou(proposals, gt_bboxes)
        max_ious, matched_gt = ious.max(dim=1)
        
        # Positive samples (high IoU with objects)
        pos_mask = max_ious >= self.pos_iou_thr
        
        # Explicit negative samples (in background regions)
        prop_centers = box_centers(proposals)
        in_background = sample_points(prop_centers, background_mask)
        explicit_neg_mask = in_background & (max_ious < self.neg_iou_thr)
        
        # Uncertain samples (not positive, not in explicit background)
        uncertain_mask = ~pos_mask & ~explicit_neg_mask
        
        # Build targets
        labels = torch.zeros_like(matched_gt)
        labels[pos_mask] = gt_labels[matched_gt[pos_mask]]
        labels[explicit_neg_mask] = 0  # Background class
        labels[uncertain_mask] = -1  # Ignore in loss
        
        return labels, matched_gt
```

### 5. Loss Aggregation

```python
class PartialAnnotationMaskRCNNLoss(nn.Module):
    """Complete loss for Mask R-CNN with partial annotations."""
    
    def forward(self, outputs, targets):
        # Extract components
        rpn_cls, rpn_bbox = outputs['rpn_cls'], outputs['rpn_bbox']
        roi_cls, roi_bbox, roi_mask = outputs['roi_cls'], outputs['roi_bbox'], outputs['roi_mask']
        
        # Extract annotation masks
        annotated_masks = torch.stack([t['annotated_mask'] for t in targets])
        background_masks = torch.stack([t['background_mask'] for t in targets])
        
        # Compute losses with partial annotation support
        rpn_loss = self.rpn_loss(
            rpn_cls, rpn_bbox,
            targets['gt_bboxes'], targets['gt_labels'],
            annotated_masks
        )
        
        roi_loss = self.roi_loss(
            roi_cls, roi_bbox, roi_mask,
            outputs['proposals'],
            targets['gt_bboxes'], targets['gt_labels'], targets['gt_masks'],
            annotated_masks, background_masks
        )
        
        total_loss = rpn_loss + roi_loss
        
        return {
            'loss': total_loss,
            'rpn_loss': rpn_loss,
            'roi_loss': roi_loss,
            'coverage': annotated_masks.mean()  # Track annotation coverage
        }
```

## Implementation Strategy

### 1. Gradual Integration

1. Start with RPN loss modifications
2. Add ROI head loss support
3. Integrate mask loss changes
4. Test end-to-end training

### 2. Validation Modes

```python
def validate_loss_computation(loss_with_partial, loss_standard, annotated_mask):
    """Ensure partial loss matches standard loss in fully annotated regions."""
    
    # In fully annotated images, losses should match
    if annotated_mask.all():
        assert torch.allclose(loss_with_partial, loss_standard, rtol=1e-5)
    
    # In partially annotated images, loss should be lower
    if not annotated_mask.all():
        assert loss_with_partial <= loss_standard
```

### 3. Debugging Tools

```python
class LossDebugger:
    """Utilities for debugging partial annotation losses."""
    
    def visualize_loss_mask(self, loss_map, annotated_mask):
        """Visualize where losses are being computed."""
        
        # Show loss heatmap
        plt.subplot(1, 3, 1)
        plt.imshow(loss_map.cpu())
        plt.title('Loss Map')
        
        # Show annotation mask
        plt.subplot(1, 3, 2)
        plt.imshow(annotated_mask.cpu())
        plt.title('Annotated Regions')
        
        # Show masked loss
        plt.subplot(1, 3, 3)
        plt.imshow((loss_map * annotated_mask).cpu())
        plt.title('Masked Loss')
        
        plt.show()
```

## Benefits

1. **Accurate Training**: No penalty for unannotated regions
2. **Better Convergence**: Cleaner gradients from annotated areas
3. **Flexibility**: Support various annotation strategies
4. **Quality**: Explicit background improves negative sampling

## Testing Plan

1. Unit tests for each loss component
2. Integration tests with partial data
3. Comparison with full annotation baseline
4. Convergence analysis on real data
5. Ablation studies on coverage levels