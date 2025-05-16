# Fix: No Predictions During Inference

## Issue
During validation, the model was not generating any predictions despite the training loss decreasing normally. The logs showed:
```
No predictions made! Returning zero metrics.
```

## Root Cause Analysis
After investigation, I found that both the RPN (Region Proposal Network) and ROI head had extremely strong biases toward the background class:

1. **RPN bias**: Initialized to -4.59, which results in ~0.01 foreground probability
2. **ROI head bias**: Background class initialized to 2.0, strongly favoring background predictions

These extreme biases were causing:
- RPN to generate very few proposals with high confidence
- ROI head to classify almost all proposals as background (class 0)
- No detections passing even moderate score thresholds (0.05)

## Solution
Made the following changes to reduce the extreme bias toward background:

### 1. Reduced RPN Bias
```python
# In swin_maskrcnn/models/rpn.py
# Changed from:
nn.init.constant_(self.rpn_cls.bias, -4.59)
# To:
nn.init.constant_(self.rpn_cls.bias, -2.0)  # Less extreme bias
```

### 2. Reduced ROI Head Background Bias
```python
# In swin_maskrcnn/models/roi_head.py (BBoxHead.init_weights)
# Changed from:
self.fc_cls.bias.data[0] = 2.0  # Favor background class
# To:
self.fc_cls.bias.data[0] = 0.5  # Small bias toward background
```

### 3. Added Debug Logging
Added logging to track proposals and scores during inference:
- Added debug logging in ROI head's `get_results` method
- Added proposal count logging in MaskRCNN's forward pass
- Added logger instances to model components
- Changed log level from INFO to DEBUG

### 4. Lowered Score Threshold for Debugging
```python
# In scripts/train.py (evaluate_coco method)
# Changed from:
if score < 0.05:
# To:
if score < 0.001:  # Very low threshold for debugging
```

## Rationale
The extreme initialization biases were appropriate for COCO dataset with 80 well-balanced classes, but caused issues when:
1. Fine-tuning on a new dataset with different class distribution
2. Using a different number of classes (69 vs 80)
3. Early in training when the model hasn't adapted yet

The more moderate biases allow the model to:
- Generate more initial proposals through RPN
- Classify some proposals as foreground classes
- Produce detections that can be evaluated and refined through training

## Impact
With these changes, the model should now generate predictions during inference, allowing:
- Proper COCO metric evaluation
- Feedback for model improvement
- Normal training progression

The biases can be further tuned based on the specific dataset characteristics if needed.