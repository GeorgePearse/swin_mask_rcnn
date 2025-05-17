# Error Handling Implementation Summary

## Overview
This implementation adds robust error handling to the training pipeline to skip problematic instances and log them to a CSV file, allowing training to continue instead of crashing.

## Key Changes

### 1. Dataset Modification (`swin_maskrcnn/data/dataset.py`)
- Added `image_filename` to the target dictionary in `__getitem__`
- This allows tracking which images cause errors

### 2. Training Script Modifications (`scripts/train.py`)
- Added imports for CSV logging and datetime
- Modified `train_step` method to:
  - Catch exceptions during forward pass
  - Return error information instead of crashing
  - Preserve string fields (like image_filename) when moving to device
- Added `log_error_to_csv` method to write error details to CSV
- Modified training loop to:
  - Check if train_step returned an error
  - Log errors to CSV and skip problematic batches
  - Continue training with remaining data

### 3. CSV Error Logging
- Creates a timestamped CSV file for each training run
- Logs the following information for each error:
  - Epoch number
  - Batch index
  - Image filename
  - Image ID
  - Error type (e.g., AssertionError)
  - Error message (e.g., Shape mismatch)
- File location: `checkpoints/problematic_images_YYYYMMDD_HHMMSS.csv`

## Usage
The error handling is automatic - when training encounters a problematic instance:
1. The error is caught
2. Details are logged to the CSV file
3. A warning is logged to the console
4. Training continues with the next batch

## Benefits
- Training can continue even when encountering problematic data
- Problematic images are identified for later investigation
- No manual intervention required during training
- Easy to analyze error patterns from the CSV file

## Example CSV Output
```csv
epoch,batch_idx,image_filename,image_id,error_type,error_message
0,15,image_123.jpg,123,AssertionError,Shape mismatch: predictions 8 vs targets 520
1,42,image_456.jpg,456,RuntimeError,CUDA out of memory
```

This allows you to identify and fix problematic images after training completes.