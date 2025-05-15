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

### Inference

Run inference on images and visualize results in FiftyOne:

```bash
# Run inference from command line
python inference.py \
    --model-path /path/to/checkpoint.pth \
    --images /path/to/image1.jpg /path/to/image2.jpg \
    --score-threshold 0.5 \
    --nms-threshold 0.5
```

Using the Python API:

```python
from swin_maskrcnn import run_inference_pipeline

# Run inference and visualize
results = run_inference_pipeline(
    model_path="/path/to/checkpoint.pth",
    image_paths=["/path/to/image1.jpg", "/path/to/image2.jpg"],
    num_classes=80,
    score_threshold=0.5,
    visualize=True,  # Opens FiftyOne app
    dataset_name="my_predictions"
)

# Access predictions
for image_path, predictions in zip(results['image_paths'], results['predictions']):
    print(f"Image: {image_path}")
    print(f"Boxes: {predictions['boxes']}")
    print(f"Labels: {predictions['labels']}")
    print(f"Scores: {predictions['scores']}")
    print(f"Masks: {predictions['masks'].shape if predictions['masks'] is not None else None}")
```

FiftyOne visualization features:
- Interactive bounding box and mask visualization
- Confidence score filtering
- Class-based filtering
- Export to various formats
- Dataset versioning and management

### Example

See `examples/inference_example.py` for a complete working example:

```bash
python examples/inference_example.py
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