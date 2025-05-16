# Frozen Backbone Stages Configuration

The SWIN Mask R-CNN implementation now supports fine-grained control over which backbone stages to freeze during training. This is useful for transfer learning scenarios where you want to preserve pretrained features while fine-tuning specific layers.

## Configuration Parameter

The `frozen_backbone_stages` parameter in `TrainingConfig` controls how many stages of the SWIN backbone are frozen:

- `-1`: No freezing - all backbone parameters are trainable
- `2`: (default) Freeze patch embedding + stages 1-2
- `0`: Freeze only the patch embedding layer
- `1`: Freeze patch embedding + stage 1
- `2`: Freeze patch embedding + stages 1-2
- `3`: Freeze patch embedding + stages 1-3
- `4`: Freeze all backbone stages (equivalent to `freeze_backbone=True`)

## Usage Examples

### In Configuration YAML

```yaml
# Model settings
pretrained_backbone: true
frozen_backbone_stages: 2  # Freeze patch embedding and first 2 stages
```

### In Python Code

```python
from scripts.config.training_config import TrainingConfig

# Create config with partial backbone freezing
config = TrainingConfig(
    frozen_backbone_stages=2,  # Freeze first 2 stages
    num_classes=69,
    # ... other parameters
)
```

### Creating Model Directly

```python
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN

model = SwinMaskRCNN(
    num_classes=69,
    frozen_backbone_stages=3  # Freeze first 3 stages
)
```

## Freezing Strategy Guidelines

- **Full fine-tuning** (`frozen_backbone_stages=-1`): Best when you have a large dataset similar to your target domain
- **Partial freezing** (`frozen_backbone_stages=1-3`): Good for medium-sized datasets or when computational resources are limited
- **Full freezing** (`frozen_backbone_stages=4`): Useful for small datasets or when you only want to train the detection/segmentation heads

## Implementation Details

When stages are frozen:
- Their parameters have `requires_grad=False`
- They are set to `.eval()` mode during training to disable batch normalization updates
- Gradients are not computed for these layers, saving memory and computation

## Validation

The parameter is validated to ensure it's within the valid range:
- Minimum: `-1` (no freezing)
- Maximum: `4` (all stages frozen)

Any value outside this range will raise a validation error.