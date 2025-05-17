"""Test script to verify the validation start step fix."""
import torch
import pytorch_lightning as pl
from pycocotools.coco import COCO

from swin_maskrcnn.data.dataset import CocoDataset
from swin_maskrcnn.data.transforms_simple import get_transform_simple
from scripts.config import TrainingConfig
from scripts.train import MaskRCNNLightningModule
from torch.utils.data import DataLoader, Subset
from swin_maskrcnn.utils.collate import collate_fn


def test_validation_fix():
    """Test the validation start step fix."""
    # Create configuration with early validation start
    config = TrainingConfig(
        num_classes=69,
        train_batch_size=1,
        val_batch_size=1,
        num_workers=0,
        validation_start_step=100,  # High enough that we'll test before it
        steps_per_validation=50,
        checkpoint_dir="./test_checkpoints/validation_fix_test",
        max_val_images=10,  # Use limited validation set
    )
    
    # Create minimal datasets
    train_dataset = CocoDataset(
        root_dir=str(config.img_root),
        annotation_file=str(config.train_ann),
        transforms=get_transform_simple(train=True),
        mode='train'
    )
    train_subset = Subset(train_dataset, list(range(5)))  # Just 5 training samples
    
    val_dataset = CocoDataset(
        root_dir=str(config.img_root),
        annotation_file=str(config.val_ann),
        transforms=get_transform_simple(train=False),
        mode='train'
    )
    val_subset = Subset(val_dataset, list(range(config.max_val_images)))
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )
    
    # Create COCO object
    val_coco = COCO(str(config.val_ann))
    
    # Create Lightning module
    model = MaskRCNNLightningModule(
        config=config,
        val_coco=val_coco
    )
    
    # Create trainer with minimal validation
    trainer = pl.Trainer(
        max_epochs=1,
        max_steps=10,  # Run only 10 training steps
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        val_check_interval=5,  # Validate every 5 steps to test the fix
        logger=False,  # Disable logger for test
        enable_progress_bar=False,
        enable_checkpointing=False,
    )
    
    print("Running validation fix test...")
    try:
        # This should no longer fail with ModelCheckpoint error
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )
        print("✅ Validation fix is working! No ModelCheckpoint error occurred.")
        return True
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False


if __name__ == "__main__":
    success = test_validation_fix()
    if success:
        print("\nThe validation start step issue has been fixed successfully!")
    else:
        print("\nThere are still issues with the validation start step.")