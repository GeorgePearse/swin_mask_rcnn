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
    train_batch_size: int = Field(default=4, description="Batch size for training")
    val_batch_size: int = Field(default=8, description="Batch size for validation")
    num_workers: int = Field(default=0, description="Number of data loading workers")
    lr: float = Field(default=1e-4, description="Learning rate")
    momentum: float = Field(default=0.9, description="Momentum for SGD optimizer")
    weight_decay: float = Field(default=0.05, description="Weight decay (L2 regularization)")
    num_epochs: int = Field(default=12, description="Number of training epochs")
    
    # Iteration-based validation
    steps_per_validation: int = Field(
        default=200,
        description="Number of training steps between validation runs"
    )
    validation_start_step: int = Field(
        default=1000,
        description="Number of training steps before starting validation"
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
    pretrained_checkpoint_url: str = Field(
        default="https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth",
        description="URL for pretrained checkpoint (mmdetection format)"
    )
    frozen_backbone_stages: int = Field(
        default=2,
        ge=-1,
        le=4,
        description="Number of SWIN backbone stages to freeze (-1 for none, max 4 for all stages)"
    )
    
    # Validation settings
    validation_iou_thresh: float = Field(
        default=0.5,
        description="IoU threshold for validation metrics"
    )

    model_config = ConfigDict(validate_assignment=True)