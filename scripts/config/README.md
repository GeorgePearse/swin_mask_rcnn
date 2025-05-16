# Training Configuration

This directory contains configuration files for training SWIN Mask R-CNN.

## Usage

### Using Default Configuration

To train with default configuration:

```bash
python scripts/train.py
```

### Using Custom Configuration

To train with a custom YAML configuration file:

```bash
python scripts/train.py --config scripts/config/sample_config.yaml
```

### Creating Custom Configuration

You can create your own configuration YAML file by copying `sample_config.yaml` and modifying the parameters. All parameters have defaults, so you only need to specify the values you want to change.

## Configuration Parameters

See `training_config.py` for all available parameters and their descriptions.

### Key Parameters

- `lr`: Learning rate (default: 1e-4)
- `momentum`: Momentum for SGD optimizer (default: 0.9)
- `steps_per_validation`: How often to run validation (default: 5)
- `optimizer`: Choose between 'adamw' or 'sgd' (default: 'adamw')
- `num_epochs`: Number of training epochs (default: 12)
- `batch_size`: Training batch size (default: 1)

## Example: High Learning Rate SGD

Create a file `high_lr_sgd.yaml`:

```yaml
lr: 0.001
momentum: 0.95
optimizer: sgd
steps_per_validation: 10
```

Then run:

```bash
python scripts/train.py --config scripts/config/high_lr_sgd.yaml
```