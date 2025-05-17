# Max Validation Images Feature

This feature allows you to limit the number of images used during validation for faster debugging and development.

## Configuration

Add the `max_val_images` parameter to your YAML configuration file:

```yaml
# Validation settings
validation_iou_thresh: 0.5
max_val_images: 50  # Limit validation to 50 images
```

## Usage

- Set `max_val_images` to any positive integer to limit the validation dataset
- Set to `None` or omit the parameter to use the entire validation dataset
- The parameter only affects validation, not training

## Example Configurations

### For quick debugging (very few images)
```yaml
max_val_images: 10
```

### For testing predictions (moderate amount)
```yaml 
max_val_images: 50
```

### For near-production testing
```yaml
max_val_images: 100
```

### For full validation
```yaml
max_val_images: null  # or just omit the parameter
```

## Implementation Details

When `max_val_images` is specified and less than the total validation dataset size:
1. A `Subset` of the validation dataset is created using the first N images
2. The indices used are `[0, 1, 2, ..., max_val_images-1]`
3. This ensures consistent validation across runs for reproducibility

## Default Configurations

- `config.yaml`: Set to 50 images for efficient prediction testing
- `test_lightning.yaml`: Set to 20 images for quick tests
- Other test configs: Not yet configured

This feature is particularly useful when debugging whether the model is producing predictions at all, as it significantly reduces validation time.