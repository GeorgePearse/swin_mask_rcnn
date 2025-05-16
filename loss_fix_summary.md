# Loss Fix Summary

## Problem
Initial training loss was extremely high (~1000s) compared to MMDetection's expected 6-10 range.

## Root Causes Identified
1. **Improper RPN Bias Initialization**: RPN classification bias was initialized to 0, causing sigmoid(0) = 0.5 predictions and high BCE loss
2. **Loss Accumulation**: RPN losses were using `.sum()` instead of `.mean()`, accumulating losses across all anchors
3. **Missing Weight Initialization**: Swin backbone wasn't properly initializing weights with truncated normal
4. **Drop Path Rate**: Was using 0.2 instead of MMDetection's default 0.1

## Solutions Implemented

### 1. RPN Bias Initialization
Changed RPN classification bias from 0 to -4.59 (log(0.01/0.99)), making initial predictions favor background (~99% background probability).

### 2. Loss Normalization
Changed RPN loss calculation from `.sum()` to `.mean()` for both classification and bbox regression losses.

### 3. Weight Initialization
Added `_init_weights` method to SwinTransformer:
- Linear layers: truncated normal with std=0.02
- LayerNorm: bias=0, weight=1.0
- Conv2d: Kaiming normal initialization

### 4. Architecture Parameters
- Changed drop_path_rate from 0.2 to 0.1
- Added ROI head bias initialization to favor background class

## Results

### Before Fix
- RPN Cls Loss: 13.36
- RPN Bbox Loss: 0.00
- ROI Cls Loss: 6.25
- Total Loss: 22.06

### After Fix
- RPN Cls Loss: 0.048 (278x reduction!)
- RPN Bbox Loss: 0.012
- ROI Cls Loss: 6.23
- Total Loss: 7.76 (within expected 6-10 range)

### With Pretrained Weights
- Total Loss: 3.92 (even better)

The loss is now in a reasonable range for training to begin!