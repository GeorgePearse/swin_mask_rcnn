# ONNX Export Callback

The SWIN Mask R-CNN training framework includes a PyTorch Lightning callback for automatic ONNX export during validation.

## Overview

The `ONNXExportCallback` automatically exports the model to ONNX format at the end of each validation epoch. This is implemented as a proper Lightning callback in the `swin_maskrcnn/callbacks` directory.

## Features

- Exports model backbone to ONNX format (full model export is also supported)
- Saves raw PyTorch weights alongside ONNX files
- Integrates seamlessly with PyTorch Lightning training
- Configurable export directory and options
- Automatic filename generation with epoch and step information

## Usage

The callback is automatically included in the training script:

```python
from swin_maskrcnn.callbacks import ONNXExportCallback

# Setup callbacks
callbacks = [
    ModelCheckpoint(...),
    LearningRateMonitor(...),
    ONNXExportCallback(
        export_dir=run_dir,
        export_backbone_only=True,
        save_weights=True
    )
]

trainer = pl.Trainer(callbacks=callbacks, ...)
```

## Configuration Options

- `export_dir`: Directory to save ONNX files (defaults to checkpoint directory)
- `export_backbone_only`: If True, exports only backbone. If False, exports full model
- `save_weights`: If True, also saves raw PyTorch weights

## Output Files

The callback generates two files per validation epoch:
- `model_epoch{XXX}_step{YYYYY}.onnx`: ONNX model file
- `weights_epoch{XXX}_step{YYYYY}.pth`: Raw PyTorch weights

## Implementation Details

The callback is implemented in:
- `swin_maskrcnn/callbacks/onnx_export.py`: Main callback implementation
- `swin_maskrcnn/callbacks/__init__.py`: Exports the callback

The callback hooks into Lightning's `on_validation_epoch_end` method to perform exports.

## Testing

Comprehensive tests are available in `tests/test_onnx_export.py`, covering:
- Basic functionality
- Error handling
- Integration with PyTorch Lightning
- Filename formatting
- Configuration options