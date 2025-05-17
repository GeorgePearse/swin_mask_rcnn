# Loss Fixes Applied to Address Detection Drop Issue

## Summary of Changes

This document summarizes all the changes applied to fix the issue where detections drop to 0 during training.

## 1. RPN Loss Normalization

### Changes Made:
- Added `avg_factor` normalization to classification loss
- Normalized bbox loss by number of positive samples
- Added numerical stability with epsilon values
- Implemented proper sample counting across all levels

### Key Modifications:
```python
# In rpn.py
- Added total_pos and total_neg counting
- Calculate avg_factor = max(total_pos + total_neg, 1.0)
- Normalize cls_loss by avg_factor
- Normalize bbox_loss by number of positive samples
```

## 2. Positive Sample Weighting

### Changes Made:
- Added `cls_pos_weight` parameter to upweight foreground samples
- Implemented weighted binary cross-entropy for RPN
- Added positive weighting for ROI classification

### Parameters Added:
- `rpn_cls_pos_weight`: Weight for positive RPN samples
- `roi_cls_pos_weight`: Weight for foreground ROI classes

## 3. ROI Head Loss Normalization

### Changes Made:
- Added proper normalization for all three losses
- Classification: Uses avg_factor (total samples)
- Bbox: Normalized by number of positive samples
- Mask: Normalized by number of positive samples
- Changed reduction from 'mean' to 'sum' with manual normalization

## 4. Loss Weight Configuration

### Parameters Added:
- `rpn_loss_cls_weight`: Weight for RPN classification loss
- `rpn_loss_bbox_weight`: Weight for RPN bbox loss
- `roi_loss_cls_weight`: Weight for ROI classification loss
- `roi_loss_bbox_weight`: Weight for ROI bbox loss
- `roi_loss_mask_weight`: Weight for ROI mask loss

## 5. Model Initialization Updates

### Files Modified:
- `swin_maskrcnn/models/mask_rcnn.py`: Added all new parameters to constructor
- `scripts/train.py`: Updated model initialization with new parameters
- `scripts/config/training_config.py`: Added new configuration fields

## 6. Gradient Clipping

### Already Implemented:
- The trainer already had gradient clipping implemented
- Controlled by `clip_grad_norm` parameter in config

## 7. Configuration Updates

### Files Updated:
- `config.yaml`: Added all new loss parameters with default values
- `training_config.py`: Added Pydantic fields for all new parameters

## Implementation Details

### RPN Loss Function Changes:
```python
def _cls_loss(self, cls_scores, cls_targets, avg_factor):
    # Added avg_factor parameter
    # Apply positive weight: weight[valid_targets == 1] = self.cls_pos_weight
    # Use reduction='sum' and divide by avg_factor
    return loss / (avg_factor + eps)
```

### ROI Loss Function Changes:
```python
def loss(self, ...):
    # Calculate num_pos = max(pos_mask.sum().item(), 1.0)
    # Cls loss: Optional positive weighting
    # Bbox loss: reduction='sum', divide by num_pos
    # Mask loss: reduction='sum', divide by num_pos
    # Apply loss weights before returning
```

## Default Recommended Values

For training with severe class imbalance:
```yaml
rpn_cls_pos_weight: 2.0  # Upweight positive anchors
rpn_loss_cls_weight: 1.0
rpn_loss_bbox_weight: 1.0
roi_cls_pos_weight: 2.0  # Upweight foreground classes
roi_loss_cls_weight: 1.0
roi_loss_bbox_weight: 1.0
roi_loss_mask_weight: 1.0
```

## Expected Benefits

1. **Stable Gradients**: Proper normalization prevents gradient explosion/vanishing
2. **Better Positive/Negative Balance**: Upweighting combats class imbalance
3. **Consistent Loss Scales**: Loss components contribute equally to training
4. **Robustness**: Handles empty batches and edge cases gracefully
5. **Tunability**: Can adjust weights based on validation performance

## Monitoring During Training

Watch for:
- Loss values staying reasonable (not NaN/inf)
- Gradients having reasonable magnitudes
- Detections not dropping to 0 after initial steps
- Balanced contribution from different loss components
- Validation mAP improving over time

## Next Steps

1. Start training with default weights (all 1.0)
2. Monitor loss behavior in first few epochs
3. If detections still drop, increase positive weights to 2.0-3.0
4. Fine-tune based on validation performance
5. Consider adjusting individual loss component weights