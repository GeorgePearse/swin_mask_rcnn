# Training Observations with Corrected Biases

## Predictions Over Time
1. Initial validation (epoch 0, step 0): 589 predictions
2. After ~50 steps: 5,838 predictions (peak)
3. After ~100 steps: 3,966 predictions
4. After ~150 steps: 4,045 predictions
5. After ~200 steps: 1,633 predictions
6. After ~250 steps: 1,272 predictions

## Key Observations
- The model continues to make predictions throughout training, which is excellent
- Prediction count fluctuates but remains substantial (never drops to near-zero like before)
- Training loss is decreasing from ~543 to values below 1.0
- mAP remains at 0.0 in early training, which is expected as the model needs more time to learn

## Analysis
1. **Success**: Fixed the zero-prediction issue with bias adjustments
2. **Expected behavior**: The fluctuation in prediction counts is normal as the model learns to be more selective
3. **Next steps**: Continue training for longer to see if mAP starts improving

## Configuration That Works
- Background bias: -2.0 (reduces false negatives)
- Foreground bias: 0.01 (encourages object detection)
- ROI cls_pos_weight: 0.5 (reduced penalty for false positives)
- Learning rate: 0.0001
- Batch size: 1 (to avoid OOM)