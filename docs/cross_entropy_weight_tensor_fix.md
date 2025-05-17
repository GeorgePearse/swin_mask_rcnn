# Cross Entropy Weight Tensor Fix

## Problem

The training was failing with the error:
```
weight tensor should be defined either for all 70 classes or no classes but got weight tensor of shape: [1024]
```

This error occurred during the ROI head loss calculation when using weighted cross entropy loss.

## Root Cause

In `swin_maskrcnn/models/roi_head.py`, the weight tensor was being created with shape matching the labels tensor:
```python
weight = torch.ones_like(all_labels, dtype=torch.float32)
weight[all_labels > 0] = self.cls_pos_weight
```

However, `F.cross_entropy` expects the weight parameter to have shape `[num_classes]`, not shape `[num_samples]`. The weight tensor should specify the weight for each class, not for each sample.

## Solution

Fixed the weight tensor creation to have the correct shape `[num_classes + 1]`:
```python
# Create weight tensor for each class (shape: [num_classes + 1])
# Apply positive weight to all foreground classes
num_total_classes = cls_scores.size(1)  # num_classes + 1
weight = torch.ones(num_total_classes, dtype=torch.float32, device=cls_scores.device)
weight[1:] = self.cls_pos_weight  # Apply to all foreground classes
cls_loss = F.cross_entropy(cls_scores, all_labels, weight=weight, reduction='mean')
```

The key changes:
1. Create weight tensor with shape `[num_total_classes]` instead of shape matching labels
2. Set weight[0] = 1.0 for background class
3. Set weight[1:] = cls_pos_weight for all foreground classes

This follows the PyTorch documentation for `F.cross_entropy` which states:
> weight (Tensor, optional) – a manual rescaling weight given to each class. If given, has to be a Tensor of size C

## Implementation Details

The fix correctly handles:
- Background class (index 0) gets weight 1.0
- All foreground classes (indices 1 through num_classes) get weight cls_pos_weight
- The weight tensor is created on the same device as the scores to avoid device mismatch

## Testing

After this fix, the model should train without the weight tensor shape error. The weighted loss will properly apply higher weight to foreground classes to help with class imbalance.