# SWIN MaskRCNN Implementation Instructions

This document provides instructions for creating an isolated SWIN-based Mask R-CNN implementation without dependencies on mmcv, mmengine, or mmdetection packages.

## Project Structure

```
swin_maskrcnn/
├── swin_maskrcnn/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── swin.py          # SWIN Transformer backbone
│   │   ├── fpn.py           # Feature Pyramid Network
│   │   ├── rpn.py           # Region Proposal Network
│   │   ├── roi_head.py      # ROI head with mask prediction
│   │   └── mask_rcnn.py     # Main model combining all components
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py       # COCO dataset wrapper
│   │   ├── transforms.py    # Albumentations transforms
│   │   └── transforms_simple.py  # Simplified transforms
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py       # Training logic
│   └── utils/
│       ├── __init__.py
│       └── collate.py       # Custom collate function
├── pyproject.toml           # Package configuration
├── train_final.py          # Training script
└── instructions.md         # This file
```

## Key Implementation Details

### 1. SWIN Transformer Backbone

The SWIN transformer implementation includes:
- Window-based self-attention mechanism
- Shifted windows for cross-window connections
- Patch merging for hierarchical feature extraction
- Proper handling of odd dimensions through padding

Key fixes made:
- Fixed norm layer dimensions (channel dims instead of spatial dims)
- Added padding for PatchMerging to handle odd dimensions
- Ensured tensors are on the correct device

### 2. Feature Pyramid Network (FPN)

Implementation includes:
- Lateral connections from backbone features
- Top-down pathway with upsampling
- Output projections for RPN and ROI heads

### 3. Region Proposal Network (RPN)

Features:
- Multi-scale anchor generation
- Objectness classification and bbox regression
- Proposal filtering and NMS
- Proper loss calculation with positive/negative sampling

Key fixes:
- Device compatibility for anchor generation
- Proper matching of predictions to targets using positive indices

### 4. ROI Head

Implements:
- ROI Align for feature extraction
- Bbox classification and regression
- Mask prediction head
- Proper FPN level assignment based on proposal size

Key fixes:
- Full FPN level assignment implementation (not simplified)
- Proper mask encoding with grid sampling
- Device compatibility in label assignment

### 5. Data Loading

The dataset implementation:
- Uses torchvision's COCO tools
- Integrates with Albumentations for augmentation
- Handles mask processing and conversion
- Includes proper normalization

Key fixes:
- Made masks contiguous to avoid negative stride errors
- Fixed label key names for Albumentations compatibility
- Added ImageNet normalization in dataset

### 6. Training

The training setup includes:
- AdamW optimizer
- Loss calculation for all components
- Checkpoint saving
- Validation loop
- Proper batch handling for list format

Key fixes:
- Custom collate function to handle list format
- Trainer modifications to work with list of images
- Correct number of classes for CMR dataset (69 classes)

## Known Issues and Solutions

1. **Negative stride error with masks**: Fixed by ensuring masks are contiguous with `np.ascontiguousarray()`

2. **Device mismatch errors**: Fixed by properly moving tensors to device and ensuring anchor generator creates tensors on correct device

3. **Transform compatibility**: Created simplified transforms without ToTensorV2 to avoid issues

4. **Label range errors**: Fixed by using correct number of classes (69) for CMR dataset

5. **Batch format**: Model expects batched tensor but dataloader returns list - fixed by stacking in forward pass

## Training the Model

To train on the CMR dataset:

```bash
# Install dependencies
pip install torch torchvision pycocotools albumentations tqdm numpy

# Run training
python train_final.py
```

Configuration can be modified in `train_final.py`:
- `batch_size`: Batch size for training
- `num_workers`: Number of dataloader workers (set to 0 to avoid transform issues)
- `num_epochs`: Number of training epochs
- `lr`: Learning rate
- `checkpoint_dir`: Directory for saving checkpoints

## Testing

A minimal test script is available:
```bash
python test_minimal.py
```

This tests both training and inference modes with synthetic data.

## Future Improvements

1. Implement multi-GPU training support
2. Add more sophisticated augmentations
3. Implement proper evaluation metrics (mAP)
4. Add pretrained weight loading support
5. Optimize memory usage for larger batch sizes
6. Fix multiprocessing support for dataloaders