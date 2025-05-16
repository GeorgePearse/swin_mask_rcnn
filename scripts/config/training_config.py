"""Training configuration using Pydantic for validation and type safety."""
from pathlib import Path
from typing import Optional, Literal
from pydantic import BaseModel, Field, ConfigDict


class TrainingConfig(BaseModel):
    """Training configuration for SWIN Mask R-CNN."""
    
    # Dataset paths
    train_ann: Path = Field(
        default=Path('/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:23.077836_train_ordered.json'),
        description="Path to training annotations file"
    )
    val_ann: Path = Field(
        default=Path('/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:38.270134_val_ordered.json'),
        description="Path to validation annotations file"
    )
    img_root: Path = Field(
        default=Path('/home/georgepearse/data/images'),
        description="Root directory of images"
    )
    
    # Model parameters
    num_classes: int = Field(default=69, description="Number of classes in dataset")
    
    # Training parameters
    batch_size: int = Field(default=1, description="Batch size for training")
    num_workers: int = Field(default=0, description="Number of data loading workers")
    lr: float = Field(default=1e-4, description="Learning rate")
    momentum: float = Field(default=0.9, description="Momentum for SGD optimizer")
    weight_decay: float = Field(default=0.05, description="Weight decay (L2 regularization)")
    num_epochs: int = Field(default=12, description="Number of training epochs")
    
    # Iteration-based validation
    steps_per_validation: int = Field(
        default=5,
        description="Number of training steps between validation runs"
    )
    
    # Gradient clipping
    clip_grad_norm: float = Field(default=10.0, description="Gradient clipping norm")
    
    # Checkpointing and logging
    checkpoint_dir: Path = Field(
        default=Path('./test_checkpoints'),
        description="Directory to save checkpoints"
    )
    log_interval: int = Field(default=50, description="Steps between logging")
    save_interval: Optional[int] = Field(
        default=None,
        description="Steps between saving checkpoints (None for epoch-based saving)"
    )
    
    # Optimizer choice
    optimizer: Literal["adamw", "sgd"] = Field(
        default="adamw",
        description="Optimizer type: 'adamw' or 'sgd'"
    )
    
    # Learning rate scheduler
    use_scheduler: bool = Field(
        default=True,
        description="Whether to use OneCycleLR scheduler"
    )
    
    # Mixed precision training
    use_amp: bool = Field(
        default=False,
        description="Use automatic mixed precision training"
    )
    
    # Model settings
    pretrained_backbone: bool = Field(
        default=True,
        description="Use pretrained backbone weights"
    )
    
    # Validation settings
    validation_iou_thresh: float = Field(
        default=0.5,
        description="IoU threshold for validation metrics"
    )

    model_config = ConfigDict(validate_assignment=True)