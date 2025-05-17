# Batch Size and Collate Function Improvements

## Issue

The training was experiencing significant performance variability with batches taking inconsistent amounts of time to process. This created a "jittery" training experience where some batches would process quickly while others would take much longer.

## Root Cause

The issue was caused by two factors:

1. **Inconsistent batch sizes**: Without `drop_last=True` in the DataLoader, the final batch in each epoch could have a different size than the rest, causing performance variations.

2. **Improper collate function**: The collate function wasn't properly handling the batching of images and targets, which could lead to inefficient memory access patterns.

## Solution

We implemented the following fixes:

### 1. Consistent Batch Sizes

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=config.train_batch_size,
    shuffle=True,
    num_workers=config.num_workers,
    collate_fn=collate_fn,
    pin_memory=True,
    drop_last=True  # Ensure consistent batch sizes
)
```

By adding `drop_last=True`, we ensure that all batches have the same size, eliminating the variability caused by smaller final batches.

### 2. Improved Collate Function

The collate function in `swin_maskrcnn/utils/collate.py` was updated to properly handle variable-sized targets while maintaining efficient tensor operations:

```python
def collate_fn(batch):
    """Custom collate function for handling variable-sized annotations."""
    images = []
    targets = []
    
    for img, target in batch:
        images.append(img)
        targets.append(target)
    
    # Stack images into a batch tensor (when possible)
    # This is more efficient than keeping them as a list
    if isinstance(images[0], torch.Tensor):
        images = torch.stack(images, dim=0)
    
    return images, targets
```

## Results

These improvements resulted in:

1. **Consistent training time per batch**: Each batch now takes approximately the same amount of time to process.

2. **Reduced GPU memory fragmentation**: Proper batching reduces memory allocation overhead.

3. **More predictable training behavior**: The elimination of batch size variations makes training progress more linear and predictable.

4. **Better GPU utilization**: Consistent batch sizes allow the GPU to optimize memory access patterns more effectively.

## Monitoring

The improvements can be monitored through:

- TensorBoard logs showing consistent step times
- GPU utilization metrics remaining stable throughout training
- Memory usage patterns showing less fragmentation

## Implementation Notes

When implementing similar fixes in other projects:

1. Always use `drop_last=True` for training dataloaders when consistent performance is important
2. Ensure collate functions properly handle batching without creating unnecessary copies
3. Monitor batch processing times during early training to catch any variability
4. Use PyTorch Lightning's built-in logging to track batch sizes: `self.log('batch_size', batch_size)`