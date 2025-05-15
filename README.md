# SWIN Mask R-CNN

An isolated implementation of SWIN-based Mask R-CNN without mm* dependencies.

## Installation

```bash
uv pip install -e .
```

## Usage

### Training

To train the model on COCO dataset:

```bash
train-maskrcnn \
    --train-ann /path/to/annotations/instances_train2017.json \
    --val-ann /path/to/annotations/instances_val2017.json \
    --batch-size 2 \
    --num-epochs 12
```

The script will automatically locate the image directories based on the annotation paths. If your images are in a different location, use `--images-root`:

```bash
train-maskrcnn \
    --train-ann /path/to/annotations/instances_train2017.json \
    --val-ann /path/to/annotations/instances_val2017.json \
    --images-root /path/to/coco/images \
    --batch-size 2 \
    --num-epochs 12
```

### Python API

```python
from swin_maskrcnn import SwinMaskRCNN, create_dataloaders, train_mask_rcnn

# Create model
model = SwinMaskRCNN(num_classes=80)

# Create dataloaders
train_loader, val_loader = create_dataloaders(
    train_root="/path/to/train2017",
    train_ann_file="/path/to/instances_train2017.json",
    val_root="/path/to/val2017",
    val_ann_file="/path/to/instances_val2017.json",
    batch_size=2
)

# Train
config = {
    'num_epochs': 12,
    'learning_rate': 0.0001,
    'checkpoint_dir': './checkpoints'
}

trainer = train_mask_rcnn(model, train_loader, val_loader, config)
```

## Architecture

- **Backbone**: SWIN Transformer
- **Neck**: Feature Pyramid Network (FPN)
- **Head**: Region Proposal Network (RPN) + ROI Head with mask prediction

## Requirements

- PyTorch >= 2.0
- torchvision >= 0.15
- pycocotools
- albumentations
- tqdm
- einops

## Development

Install development dependencies:

```bash
uv pip install -e ".[dev]"
```

### Testing

Run tests:

```bash
pytest tests/
```

Run tests with coverage:

```bash
pytest tests/ --cov=swin_maskrcnn --cov-report=html
```

### Code Quality

Run linters and type checkers:

```bash
# Format code
black .

# Lint code
ruff check .

# Type check
mypy swin_maskrcnn/
```

### Contributing

This project uses GitHub Actions for continuous integration. All pull requests will automatically run tests against Python 3.9, 3.10, and 3.11.

To contribute:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request