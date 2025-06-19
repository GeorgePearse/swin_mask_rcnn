# Fast Training Loops for SWIN Mask R-CNN

This document describes the optimizations implemented for fast train/eval loops with proper class-level metrics.

## Overview

The implementation provides several levels of feedback mechanisms:

1. **Quick Evaluation** - Ultra-fast feedback every 10-50 steps
2. **Fast Validation** - Regular validation with small subset
3. **Full Validation** - Complete evaluation with per-class metrics
4. **Real-time Monitoring** - Live metrics tracking

## Features

### 1. Quick Evaluation System

- Runs on a tiny subset (10-50 images) every N steps
- Provides immediate feedback on model predictions
- Tracks top-K performing classes
- Minimal overhead (~2-5 seconds)

```yaml
# Enable in config
quick_eval_enabled: true
quick_eval_interval: 10  # Every 10 steps
quick_eval_samples: 10   # Use 10 samples
track_top_k_classes: 5   # Track top 5 classes
```

### 2. Enhanced Class-Level Metrics

The validation now provides:
- **Top Performing Classes**: Shows best AP scores with recall
- **Classes Needing Attention**: Highlights underperforming classes
- **Aggregate Metrics**: Top-10 vs Bottom-10 class performance
- **AP Spread**: Indicates training balance across classes

Example output:
```
================================================================================
PER-CLASS METRICS (Detection - AP@0.5:0.95) - Epoch 0, Step 100
================================================================================

                     TOP PERFORMING CLASSES                      
Class Name                           AP     AP50    Preds       GT   Recall
----------------------------------------------------------------------
person                            0.4521   0.6834      234      189     1.24
car                              0.3892   0.5921      156      201     0.78
bicycle                          0.3214   0.4832       89      112     0.79

                   CLASSES NEEDING ATTENTION                     
Class Name                           AP     AP50    Preds       GT   Recall
----------------------------------------------------------------------
skateboard                       0.0012   0.0034        2       45     0.04
surfboard                        0.0008   0.0021        1       38     0.03
tennis racket                    0.0000   0.0000        0       29     0.00
```

### 3. Real-time Metrics Tracking

The `MetricsTracker` callback provides:
- Moving averages for all losses
- Performance alerts when metrics degrade
- JSON export for external monitoring
- Time estimates for training completion

### 4. Monitoring Script

Run alongside training for live updates:
```bash
python scripts/monitor_training.py --metrics-dir ./fast_dev_checkpoints/run_*/metrics
```

## Configuration Examples

### Fast Development Loop
```yaml
# scripts/config/fast_dev_loop.yaml
train_batch_size: 2
val_batch_size: 4
steps_per_validation: 20  # Validate every 20 steps
validation_start_step: 20
max_val_images: 20       # Small validation set

# Quick eval for immediate feedback
quick_eval_enabled: true
quick_eval_interval: 10
quick_eval_samples: 10

# More frozen layers for speed
frozen_backbone_stages: 3
```

### Balanced Training Loop
```yaml
# For more thorough but still fast training
train_batch_size: 4
val_batch_size: 8
steps_per_validation: 100
validation_start_step: 200
max_val_images: 100

quick_eval_enabled: true
quick_eval_interval: 50
quick_eval_samples: 50
track_top_k_classes: 10
```

## Usage

1. **Start training with fast config**:
```bash
python scripts/train.py --config scripts/config/fast_dev_loop.yaml
```

2. **Monitor in another terminal**:
```bash
# Find the latest run directory
RUN_DIR=$(ls -td fast_dev_checkpoints/run_* | head -1)
python scripts/monitor_training.py --metrics-dir $RUN_DIR/metrics
```

3. **View in TensorBoard**:
```bash
tensorboard --logdir fast_dev_checkpoints/tensorboard
```

## Performance Tips

1. **GPU Memory**: Use smaller batch sizes and `use_amp: true`
2. **CPU Bottleneck**: Reduce `num_workers` or use `quick_eval_samples`
3. **Disk I/O**: Use local SSD for checkpoints and logs
4. **Network**: Download pretrained weights once and use `checkpoint_path`

## Interpreting Results

### Quick Evaluation Metrics
- `quick_eval/total_predictions`: Total detections in subset
- `quick_eval/categories_detected`: Number of unique classes found
- `quick_eval/top_N_cat_X_count`: Predictions for top classes

### Validation Metrics
- `val/top_10_classes_mAP`: Performance on best classes
- `val/bottom_10_classes_mAP`: Performance on worst classes  
- `val/ap_spread`: Difference between top and bottom (lower is better)

### Training Progress
- `train/avg_predictions_per_image`: Should increase early in training
- `train/images_with_predictions`: Should approach batch size
- Loss moving averages: Should generally decrease over time

## Troubleshooting

### No predictions during quick eval
- Model may need more warmup steps
- Try increasing `validation_start_step`
- Check if pretrained weights loaded correctly

### Class imbalance issues
- Monitor `val/ap_spread` metric
- Adjust class weights if needed
- Consider focal loss for extreme imbalance

### Slow validation
- Reduce `max_val_images`
- Increase `val_batch_size` if GPU memory allows
- Disable segmentation metrics if not needed