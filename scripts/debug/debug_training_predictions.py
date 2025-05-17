"""Debug training predictions."""
import sys
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
from pathlib import Path

# Load PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from swin_maskrcnn.data.dataset import CocoDataset
from swin_maskrcnn.data.transforms_simple import get_transform_simple
from swin_maskrcnn.utils.collate import collate_fn
from swin_maskrcnn.utils.logging import setup_logger
from scripts.train import MaskRCNNLightningModule
from scripts.config import TrainingConfig
from pycocotools.coco import COCO

import yaml
from torch.utils.data import DataLoader


class DebugValidationCallback(Callback):
    """Custom callback to debug validation predictions."""
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Hook into validation batch end to check predictions."""
        if batch_idx < 3:  # Only log first few batches
            print(f"\nDebug Validation Batch {batch_idx}:")
            print(f"  Number of outputs: {len(outputs)}")
            if outputs and 'predictions' in outputs:
                print(f"  Predictions count: {outputs['predictions']}")
            
            # Access model directly
            images, targets = batch 
            images = [img.to(pl_module.device) for img in images]
            
            with torch.no_grad():
                predictions = pl_module.model(images)
                for i, pred in enumerate(predictions):
                    if pred and 'boxes' in pred:
                        print(f"  Image {i}: {len(pred['boxes'])} boxes")
                        if len(pred['boxes']) > 0:
                            scores = pred['scores'].cpu().numpy()
                            print(f"    Score range: [{scores.min():.4f}, {scores.max():.4f}]")
                    else:
                        print(f"  Image {i}: No predictions")


def main():
    logger = setup_logger()
    
    # Load config
    config_path = Path("/home/georgepearse/swin_maskrcnn/scripts/config/train_with_fixed_biases.yaml")
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = TrainingConfig(**config_dict)
    
    # Create datasets
    logger.info("Creating datasets...")
    val_dataset = CocoDataset(
        root_dir=str(config.img_root),
        annotation_file=str(config.val_ann),
        transforms=get_transform_simple(train=False),
        mode='train'
    )
    
    # Only use first 5 images for debugging
    from torch.utils.data import Subset
    val_dataset = Subset(val_dataset, indices=list(range(5)))
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False,
    )
    
    # Create COCO object for validation
    val_coco = COCO(str(config.val_ann))
    
    # Create Lightning module
    logger.info("Creating Lightning module...")
    lightning_model = MaskRCNNLightningModule(
        config=config,
        val_coco=val_coco
    )
    
    # Create trainer with our debug callback
    trainer = pl.Trainer(
        max_epochs=1,
        val_check_interval=1,  # Validate immediately
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[DebugValidationCallback()],
        num_sanity_val_steps=0,  # Skip sanity check
        enable_model_summary=False,
    )
    
    # Run validation only
    logger.info("Running validation...")
    trainer.validate(model=lightning_model, dataloaders=val_loader)
    
    logger.info("Debug complete!")


if __name__ == "__main__":
    main()