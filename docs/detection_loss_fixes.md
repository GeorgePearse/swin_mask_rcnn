# Detection Loss Fixes

This document describes the fixes applied to address the issue of detections dropping to 0 during training.

## Issue Analysis

The main problems were:

1. **Class imbalance not addressed**: Default positive weight of 1.0 wasn't handling the severe background/foreground imbalance
2. **Loss normalization**: Classification loss wasn't using proper reduction
3. **No emphasis on detecting objects**: Equal weight for background and foreground led the model to suppress all detections

## Fixes Applied

### 1. Updated Positive Weights
- **RPN classification positive weight**: Increased from 1.0 to 5.0
- **ROI classification positive weight**: Increased from 3.0 to 3.0
  
These weights emphasize positive (foreground) samples during training, preventing the model from simply predicting all background.

### 2. Loss Normalization
- Added explicit `reduction='mean'` to cross-entropy loss in ROI head
- Ensures proper averaging across samples
- Already had proper normalization in RPN (with avg_factor)

### 3. Configuration Updates
Updated both `training_config.py` and `config.yaml` with:
```yaml
rpn_cls_pos_weight: 5.0  # Was 1.0
roi_cls_pos_weight: 3.0  # Was 1.0
```

### 4. Existing Protections
The code already had:
- Gradient clipping (`clip_grad_norm: 10.0`)
- Proper bbox and mask loss normalization
- Handling of empty batches
- Mixed precision training support

## Expected Impact

These changes should:
1. Prevent the model from converging to predicting all background
2. Maintain stable gradients during training
3. Keep detection scores reasonable throughout training
4. Better handle the natural class imbalance in detection tasks

## Monitoring

During training, monitor:
- Loss values (should decrease but not collapse)
- Number of predictions above various thresholds
- Validation mAP scores
- Background vs foreground prediction ratio

If issues persist, consider:
- Further increasing positive weights
- Adding focal loss for better hard example mining
- Implementing online hard negative mining
- Adjusting learning rate or gradient clipping

## Related Documentation

See `loss_fixes_applied.md` for a comprehensive overview of all loss fixes applied to the codebase, including the normalization strategies and configuration changes made throughout the project.