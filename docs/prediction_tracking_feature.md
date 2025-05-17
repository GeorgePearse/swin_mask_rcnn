# Prediction Tracking Feature

This feature adds comprehensive logging of end-to-end predictions made by the model during training and validation.

## Changes Made

### 1. Training Step Prediction Tracking

In `scripts/train.py`, the `training_step` method now:
- Runs the model in inference mode to get actual predictions
- Counts the number of predictions per image
- Logs the following metrics:
  - `train/total_predictions`: Total number of predictions in the batch
  - `train/avg_predictions_per_image`: Average predictions per image
  - `train/images_with_predictions`: Number of images that have at least one prediction

### 2. Validation Step Enhanced Tracking

The `validation_step` method now tracks predictions by score threshold:
- Counts predictions above various thresholds (0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9)
- Logs `val/predictions_above_{threshold}` for each threshold

### 3. Model Forward Pass Debug Output

In `swin_maskrcnn/models/mask_rcnn.py`, the forward pass now:
- Logs total detections and per-image detection counts
- Computes and displays score statistics (min, max, mean, std)
- Shows a score histogram to visualize the distribution of confidence scores

### 4. ROI Head Logging Optimization

Removed verbose print statements from `roi_head.py` that would clutter the output during training, keeping only logger-based debugging.

## Benefits

This feature provides valuable insights into:
1. How many predictions the model is making during training
2. The confidence score distribution of predictions
3. How predictions evolve over training epochs
4. Which images are getting predictions and which aren't

## Usage

The tracking is automatic and will appear in:
- Console output (for debug prints during inference)
- TensorBoard logs (for metrics logged via Lightning)
- CSV error logs (for any errors during prediction)

## Monitoring

To monitor the prediction counts during training:
1. Open TensorBoard: `tensorboard --logdir checkpoints/tensorboard`
2. Look for the following metrics:
   - `train/total_predictions`
   - `train/avg_predictions_per_image`
   - `train/images_with_predictions`
   - `val/predictions_above_*` (various thresholds)

This helps diagnose issues like:
- Model not making any predictions early in training
- Confidence scores being too low
- Imbalanced predictions across images