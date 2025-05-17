# Loss Fix Implementation Guide

Based on the comparison with MMDetection, here are the specific changes needed to fix the loss calculation issues that cause detections to drop to 0.

## 1. RPN Head Loss Fixes

### Current Issues:
- No avg_factor normalization
- No protection against empty batches
- Missing positive sample weighting

### Required Changes:

```python
class RPNHead(nn.Module):
    def __init__(
        self,
        ...,
        cls_pos_weight=1.0,  # New parameter
        loss_cls_weight=1.0,  # New parameter
        loss_bbox_weight=1.0,  # New parameter
    ):
        ...
        # Update loss functions
        self.loss_cls = nn.BCEWithLogitsLoss(reduction='none')
        self.loss_bbox = nn.SmoothL1Loss(reduction='none')
        
        # Loss weights
        self.cls_pos_weight = cls_pos_weight
        self.loss_cls_weight = loss_cls_weight
        self.loss_bbox_weight = loss_bbox_weight
        
    def loss(self, cls_scores, bbox_preds, gt_bboxes, img_sizes):
        """Calculate RPN loss with proper normalization."""
        device = cls_scores[0].device
        featmap_sizes = [score.shape[-2:] for score in cls_scores]
        anchors = self.get_anchors(featmap_sizes, device)
        
        # Count total samples for normalization
        total_pos = 0
        total_neg = 0
        
        # ... existing matching code ...
        
        # Count samples for avg_factor
        for img_targets in cls_targets:
            for level_targets in img_targets:
                valid_mask = level_targets >= 0
                pos_mask = level_targets == 1
                neg_mask = level_targets == 0
                total_pos += pos_mask.sum().item()
                total_neg += neg_mask.sum().item()
        
        # Calculate avg_factor (following MMDetection)
        avg_factor = max(total_pos + total_neg, 1.0)
        
        # Calculate losses with proper normalization
        cls_loss = self._cls_loss(cls_scores, cls_targets, avg_factor)
        bbox_loss = self._bbox_loss(bbox_preds, bbox_targets, pos_indices, max(total_pos, 1.0))
        
        return {
            'rpn_cls_loss': cls_loss * self.loss_cls_weight,
            'rpn_bbox_loss': bbox_loss * self.loss_bbox_weight
        }
    
    def _cls_loss(self, cls_scores, cls_targets, avg_factor):
        """Calculate classification loss with proper normalization."""
        loss = torch.tensor(0.0, device=cls_scores[0].device)
        eps = 1e-6  # For numerical stability
        
        batch_size = cls_scores[0].size(0)
        for batch_idx in range(batch_size):
            for level_idx, level_scores in enumerate(cls_scores):
                scores = level_scores[batch_idx].permute(1, 2, 0).reshape(-1)
                
                if batch_idx < len(cls_targets) and level_idx < len(cls_targets[batch_idx]):
                    targets = cls_targets[batch_idx][level_idx]
                    
                    # Only calculate loss for valid targets (exclude -1)
                    valid_mask = targets >= 0
                    if valid_mask.any():
                        valid_scores = scores[valid_mask]
                        valid_targets = targets[valid_mask].float()
                        
                        # Apply positive weight to foreground samples
                        weight = torch.ones_like(valid_targets)
                        weight[valid_targets == 1] = self.cls_pos_weight
                        
                        batch_loss = F.binary_cross_entropy_with_logits(
                            valid_scores, valid_targets, weight=weight, reduction='sum'
                        )
                        loss = loss + batch_loss
        
        # Normalize by avg_factor
        return loss / (avg_factor + eps)
    
    def _bbox_loss(self, bbox_preds, bbox_targets, pos_indices, avg_factor):
        """Calculate bounding box regression loss with proper normalization."""
        loss = torch.tensor(0.0, device=bbox_preds[0].device)
        eps = 1e-6
        
        # ... existing bbox loss calculation ...
        
        # Normalize by number of positive samples
        return loss / (avg_factor + eps)
```

## 2. ROI Head Loss Fixes

### Current Issues:
- No avg_factor in classification loss
- Missing positive sample weighting
- No loss weight configuration

### Required Changes:

```python
class StandardRoIHead(nn.Module):
    def __init__(
        self,
        ...,
        cls_pos_weight=1.0,  # New parameter
        loss_cls_weight=1.0,  # New parameter
        loss_bbox_weight=1.0,  # New parameter
        loss_mask_weight=1.0,  # New parameter
    ):
        ...
        self.cls_pos_weight = cls_pos_weight
        self.loss_cls_weight = loss_cls_weight
        self.loss_bbox_weight = loss_bbox_weight
        self.loss_mask_weight = loss_mask_weight
        
    def loss(self, cls_scores, bbox_preds, mask_preds, labels, bbox_targets, mask_targets):
        """Calculate losses with proper normalization."""
        losses = {}
        
        # Concatenate labels from all images
        all_labels = torch.cat(labels)
        
        # Calculate avg_factor for classification
        # Count valid samples (not ignored)
        avg_factor = max(len(all_labels), 1.0)
        
        # Classification loss with avg_factor
        if self.cls_pos_weight > 1.0:
            # Apply positive weight to foreground classes
            weight = torch.ones_like(all_labels, dtype=torch.float32)
            weight[all_labels > 0] = self.cls_pos_weight
            cls_loss = F.cross_entropy(cls_scores, all_labels, weight=weight)
        else:
            cls_loss = F.cross_entropy(cls_scores, all_labels)
            
        losses['cls_loss'] = cls_loss * self.loss_cls_weight
        
        # Bbox regression loss with normalization by positive samples
        pos_mask = all_labels > 0
        num_pos = max(pos_mask.sum().item(), 1.0)
        
        if pos_mask.any():
            # ... existing bbox loss calculation ...
            bbox_loss = F.smooth_l1_loss(pos_bbox_preds, bbox_targets_concat, reduction='sum')
            bbox_loss = bbox_loss / num_pos
        else:
            bbox_loss = torch.tensor(0.0, device=cls_scores.device)
            
        losses['bbox_loss'] = bbox_loss * self.loss_bbox_weight
        
        # Mask loss with normalization
        if mask_preds is not None and pos_mask.any():
            # ... existing mask loss calculation ...
            mask_loss = F.binary_cross_entropy_with_logits(
                selected_mask_preds, mask_targets_concat, reduction='sum'
            )
            mask_loss = mask_loss / num_pos
        else:
            mask_loss = torch.tensor(0.0, device=cls_scores.device)
            
        losses['mask_loss'] = mask_loss * self.loss_mask_weight
        
        return losses
```

## 3. Training Configuration Updates

### Add to config.yaml:

```yaml
model:
  rpn_head:
    cls_pos_weight: 1.0  # Increase to emphasize positive samples
    loss_cls_weight: 1.0
    loss_bbox_weight: 1.0
  
  roi_head:
    cls_pos_weight: 1.0  # Increase for foreground emphasis
    loss_cls_weight: 1.0
    loss_bbox_weight: 1.0
    loss_mask_weight: 1.0

training:
  gradient_clip:
    enabled: true
    max_norm: 35.0
    norm_type: 2
```

## 4. Model Initialization Updates

### Update mask_rcnn.py:

```python
class MaskRCNN(nn.Module):
    def __init__(self, cfg):
        ...
        # RPN with loss weights
        self.rpn_head = RPNHead(
            in_channels=backbone_channels[0],
            feat_channels=cfg.model.rpn_head.feat_channels,
            anchor_generator=anchor_generator,
            pos_iou_thr=cfg.model.rpn_head.pos_iou_thr,
            neg_iou_thr=cfg.model.rpn_head.neg_iou_thr,
            num_neg_ratio=cfg.model.rpn_head.num_neg_ratio,
            cls_pos_weight=cfg.model.rpn_head.get('cls_pos_weight', 1.0),
            loss_cls_weight=cfg.model.rpn_head.get('loss_cls_weight', 1.0),
            loss_bbox_weight=cfg.model.rpn_head.get('loss_bbox_weight', 1.0),
        )
        
        # ROI Head with loss weights
        self.roi_head = StandardRoIHead(
            num_classes=cfg.model.num_classes,
            pos_iou_thr=cfg.model.roi_head.pos_iou_thr,
            neg_iou_thr=cfg.model.roi_head.neg_iou_thr,
            pos_ratio=cfg.model.roi_head.pos_ratio,
            num_samples=cfg.model.roi_head.num_samples,
            cls_pos_weight=cfg.model.roi_head.get('cls_pos_weight', 1.0),
            loss_cls_weight=cfg.model.roi_head.get('loss_cls_weight', 1.0),
            loss_bbox_weight=cfg.model.roi_head.get('loss_bbox_weight', 1.0),
            loss_mask_weight=cfg.model.roi_head.get('loss_mask_weight', 1.0),
        )
```

## 5. Training Loop Updates

### Add gradient clipping to trainer.py:

```python
def training_step(self, batch, batch_idx):
    # ... existing code ...
    
    # Gradient clipping
    if self.cfg.training.get('gradient_clip', {}).get('enabled', False):
        torch.nn.utils.clip_grad_norm_(
            self.parameters(),
            max_norm=self.cfg.training.gradient_clip.max_norm,
            norm_type=self.cfg.training.gradient_clip.norm_type
        )
    
    return loss
```

## Key Benefits of These Changes:

1. **Proper Normalization**: Losses are normalized by the correct number of samples
2. **Positive Sample Weighting**: Can emphasize foreground samples to combat class imbalance
3. **Loss Component Balancing**: Can adjust relative importance of cls/bbox/mask losses
4. **Gradient Stability**: Protection against numerical issues and gradient explosion
5. **Empty Batch Handling**: Graceful handling of batches with no positive samples

## Testing the Fix:

1. Start with default weights (all 1.0)
2. Monitor loss values and gradient norms
3. If detections still drop to 0:
   - Increase `cls_pos_weight` to 2.0-3.0
   - Adjust loss component weights
   - Enable gradient clipping
4. Fine-tune weights based on validation performance