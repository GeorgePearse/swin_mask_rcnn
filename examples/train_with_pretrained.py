"""Example training script using pretrained SWIN weights from mmdetection."""
from scripts.config.training_config import TrainingConfig
from scripts.train import main


# Example configuration using pretrained weights
config = TrainingConfig(
    # Dataset configuration
    train_ann='/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:23.077836_train_ordered.json',
    val_ann='/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:38.270134_val_ordered.json',
    img_root='/home/georgepearse/data/images',
    
    # Model configuration
    num_classes=69,
    pretrained_backbone=True,
    pretrained_checkpoint_url='https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth',
    frozen_backbone_stages=2,  # Freeze first 2 stages (default)
    
    # Training parameters
    train_batch_size=2,
    val_batch_size=4,
    lr=1e-4,
    num_epochs=12,
    
    # Iteration-based validation
    steps_per_validation=200,
    validation_start_step=1000,
    
    # Optimizer settings
    optimizer='adamw',
    use_scheduler=True,
    
    # Checkpoint directory
    checkpoint_dir='./pretrained_checkpoints'
)


if __name__ == "__main__":
    # Create config file for training
    import yaml
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config_dict = config.model_dump()
        yaml.dump(config_dict, f)
        config_path = f.name
    
    print("Training SWIN Mask R-CNN with pretrained weights")
    print(f"Pretrained URL: {config.pretrained_checkpoint_url}")
    print(f"Frozen stages: {config.frozen_backbone_stages}")
    print(f"Config saved to: {config_path}")
    
    # Start training
    main(config_path=config_path)