"""Training script with PyTorch Lightning and TensorBoard integration."""
import json
from pathlib import Path
from typing import Dict, Optional, Any, List

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.optim import AdamW, SGD, Optimizer
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
import time
import csv
from datetime import datetime

from swin_maskrcnn.data.dataset import CocoDataset
from swin_maskrcnn.data.transforms_simple import get_transform_simple
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.utils.collate import collate_fn
from scripts.config import TrainingConfig
from swin_maskrcnn.utils.pretrained_loader import load_pretrained_from_url
from swin_maskrcnn.utils.logging import setup_logger
from swin_maskrcnn.utils.load_coco_weights import load_coco_weights
from swin_maskrcnn.callbacks import ONNXExportCallback


def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def get_gpu_utilization():
    """Get current GPU utilization percentage."""
    if torch.cuda.is_available():
        try:
            # Try using pynvml if available
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return utilization.gpu
        except ImportError:
            # Fallback: Use nvidia-smi command
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_id = torch.cuda.current_device()
                lines = result.stdout.strip().split('\n')
                if gpu_id < len(lines):
                    return float(lines[gpu_id])
            # If nvidia-smi fails, return -1 to indicate unavailable
            return -1.0
        except Exception:
            return -1.0
    return 0.0


class MaskRCNNLightningModule(pl.LightningModule):
    """PyTorch Lightning Module for SWIN-based Mask R-CNN."""
    
    def __init__(
        self,
        config: TrainingConfig,
        val_coco,  # COCO object for validation
    ):
        super().__init__()
        self.config = config
        self.val_coco = val_coco
        
        # Initialize model
        self.model = SwinMaskRCNN(
            num_classes=config.num_classes,
            frozen_backbone_stages=config.frozen_backbone_stages,
            rpn_cls_pos_weight=config.rpn_cls_pos_weight,
            rpn_loss_cls_weight=config.rpn_loss_cls_weight,
            rpn_loss_bbox_weight=config.rpn_loss_bbox_weight,
            roi_cls_pos_weight=config.roi_cls_pos_weight,
            roi_loss_cls_weight=config.roi_loss_cls_weight,
            roi_loss_bbox_weight=config.roi_loss_bbox_weight,
            roi_loss_mask_weight=config.roi_loss_mask_weight,
        )
        
        # Load pretrained weights if specified
        if config.pretrained_backbone:
            # Always load COCO pretrained weights with proper initialization
            print("Loading COCO pretrained weights...")
            missing_keys, unexpected_keys = load_coco_weights(self.model, num_classes=config.num_classes)
            print(f"Missing keys: {len(missing_keys)}")
            print(f"Unexpected keys: {len(unexpected_keys)}")
            
            # Log some diagnostic info about the model
            if hasattr(self.model.roi_head, 'fc_cls'):
                cls_bias = self.model.roi_head.fc_cls.bias.detach().cpu().numpy()
                print(f"Background bias after loading: {cls_bias[0]:.4f}")
                print(f"Object bias mean: {cls_bias[1:].mean():.4f}")
                print(f"Object bias range: [{cls_bias[1:].min():.4f}, {cls_bias[1:].max():.4f}]")
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Initialize CSV logger for problematic images
        self.error_log_path = Path("problematic_images.csv")
        with open(self.error_log_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['epoch', 'batch_idx', 'image_filename', 'image_id', 'error_type', 'error_message'])
        
        # Validation predictions storage
        self.validation_outputs = []
        
        # Loss weights - adjust to help with detection stability
        self.loss_weights = {
            'rpn_cls_loss': 1.0,
            'rpn_bbox_loss': 1.0,
            'roi_cls_loss': 1.0,
            'roi_bbox_loss': 1.0,
            'roi_mask_loss': 1.0,
        }
    
    def forward(self, images, targets=None):
        return self.model(images, targets)
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        # Initialize optimizer
        if self.config.optimizer == "adamw":
            optimizer = AdamW(
                self.model.parameters(), 
                lr=self.config.lr,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "sgd":
            optimizer = SGD(
                self.model.parameters(),
                lr=self.config.lr,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        # Initialize scheduler if requested
        if self.config.use_scheduler:
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.config.lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.05,
                anneal_strategy='cos'
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                }
            }
        
        return optimizer
    
    def training_step(self, batch, batch_idx):
        """Lightning training step."""
        images, targets = batch
        batch_size = len(images)
        
        # Debug batch size issue
        if batch_idx < 5:  # Log first few batches
            print(f"Batch {batch_idx}: actual batch size = {batch_size}, expected = {self.config.train_batch_size}")
        
        try:
            # Move to device
            images = [img.to(self.device) for img in images]
            targets_device = []
            for t in targets:
                t_device = {}
                for k, v in t.items():
                    if isinstance(v, torch.Tensor):
                        t_device[k] = v.to(self.device)
                    else:
                        t_device[k] = v  # Keep string fields (like image_filename) as is
                targets_device.append(t_device)
            
            # Make predictions (inference mode) to count end-to-end predictions
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(images)
                
                # Count predictions per image
                pred_counts = []
                for pred in predictions:
                    if pred is not None and 'boxes' in pred:
                        num_preds = len(pred['boxes'])
                    else:
                        num_preds = 0
                    pred_counts.append(num_preds)
                
                total_predictions = sum(pred_counts)
                avg_predictions = total_predictions / len(predictions) if predictions else 0
                
                # Log prediction statistics
                self.log('train/total_predictions', total_predictions, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
                self.log('train/avg_predictions_per_image', avg_predictions, on_step=True, on_epoch=True, batch_size=batch_size)
                self.log('train/images_with_predictions', sum(1 for c in pred_counts if c > 0), on_step=True, on_epoch=True, batch_size=batch_size)
                
                # Log number of annotations per image for comparison
                annotation_counts = []
                for target in targets_device:
                    if 'boxes' in target:
                        annotation_counts.append(len(target['boxes']))
                    else:
                        annotation_counts.append(0)
                
                total_annotations = sum(annotation_counts)
                avg_annotations = total_annotations / len(annotation_counts) if annotation_counts else 0
                
                self.log('train/total_annotations', total_annotations, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
                self.log('train/avg_annotations_per_image', avg_annotations, on_step=True, on_epoch=True, batch_size=batch_size)
                self.log('train/images_with_annotations', sum(1 for c in annotation_counts if c > 0), on_step=True, on_epoch=True, batch_size=batch_size)
            
            # Back to training mode for loss calculation
            self.model.train()
            
            # Forward pass for training loss
            loss_dict = self.forward(images, targets_device)
            
            # Weight the losses
            weighted_losses = {}
            for k, v in loss_dict.items():
                weight = self.loss_weights.get(k, 1.0)
                weighted_losses[k] = v * weight
            
            total_loss = sum(weighted_losses.values())
            
            # Log losses to TensorBoard
            self.log('train/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
            # Log both raw and weighted losses for better debugging
            for k, v in loss_dict.items():
                self.log(f'train/{k}', v, on_step=True, on_epoch=True, batch_size=batch_size)
                weight = self.loss_weights.get(k, 1.0)
                if weight != 1.0:
                    self.log(f'train/{k}_weighted', v * weight, on_step=True, on_epoch=True, batch_size=batch_size)
            
            # Log memory usage
            memory_mb = get_gpu_memory_mb()
            gpu_util = get_gpu_utilization()
            self.log('train/memory_mb', memory_mb, on_step=True, on_epoch=False, batch_size=batch_size)
            self.log('train/gpu_utilization', gpu_util, on_step=True, on_epoch=False, batch_size=batch_size)
            
            return total_loss
            
        except Exception as e:
            # Log the error with information about the problematic images
            self.log_error_to_csv(self.current_epoch, batch_idx, {
                'error_type': type(e).__name__,
                'error_message': str(e),
            }, targets)
            
            # Skip this batch
            print(f"Skipping batch {batch_idx} due to error: {e}")
            return None
    
    def log_error_to_csv(self, epoch: int, batch_idx: int, error_info: Dict, targets: list):
        """Log error information to CSV file."""
        with open(self.error_log_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Log error for each image in the batch
            for target in targets:
                image_filename = target.get('image_filename', 'unknown')
                image_id = target.get('image_id', torch.tensor([0])).item() if isinstance(target.get('image_id'), torch.Tensor) else 0
                
                writer.writerow([
                    epoch,
                    batch_idx,
                    image_filename,
                    image_id,
                    error_info.get('error_type', 'Unknown'),
                    error_info.get('error_message', 'No message')
                ])
    
    def validation_step(self, batch, batch_idx):
        """Lightning validation step."""
        images, targets = batch
        batch_size = len(images)
        
        # Skip validation if we haven't reached the minimum step count
        skip_validation = False
        if hasattr(self.config, 'validation_start_step'):
            if self.global_step < self.config.validation_start_step:
                skip_validation = True
        
        # If we're skipping validation, return early with minimal computation
        if skip_validation:
            return {'predictions': 0}
        
        # Move to device
        images = [img.to(self.device) for img in images]
        
        # Get predictions (no targets for inference)
        outputs = self.forward(images)
        
        # Store predictions for COCO evaluation
        batch_predictions = []
        
        # Track prediction statistics
        score_thresholds = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
        threshold_counts = {t: 0 for t in score_thresholds}
        
        # Process each image
        for i, output in enumerate(outputs):
            # Get original image ID from target
            image_id = targets[i]['image_id'].item()
            
            # Skip if no predictions at all (no boxes)
            if len(output.get('boxes', [])) == 0:
                continue
                
            # Handle the case where masks might be None for 0 predictions
            if output.get('masks') is None:
                continue
            
            # Track predictions by score threshold
            scores = output['scores'].cpu().numpy()
            for threshold in score_thresholds:
                threshold_counts[threshold] += (scores >= threshold).sum()
            
            # Convert predictions to COCO format
            for j, (box, label, score, mask) in enumerate(zip(
                output['boxes'].cpu().numpy(),
                output['labels'].cpu().numpy(),
                output['scores'].cpu().numpy(),
                output['masks'].cpu().numpy()
            )):
                # Use a very low threshold for debugging
                if score < 0.001:  # Changed from 0.05 to 0.001 for debugging
                    continue
                
                # Convert mask to binary format and then to RLE
                mask_binary = (mask[0] > 0.5).astype(np.uint8)
                
                # Convert to RLE using COCO's mask utilities
                rle = maskUtils.encode(np.asfortranarray(mask_binary))
                if rle is not None and 'counts' in rle:
                    rle['counts'] = rle['counts'].decode('utf-8')  # Convert bytes to string
                else:
                    # Skip this prediction if mask encoding failed
                    continue
                
                # Get bounding box in COCO format [x, y, width, height]
                x1, y1, x2, y2 = box
                bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                
                batch_predictions.append({
                    'image_id': int(image_id),
                    'category_id': int(label),
                    'bbox': bbox,
                    'score': float(score),
                    'segmentation': rle  # RLE format
                })
        
        self.validation_outputs.extend(batch_predictions)
        
        # Log prediction statistics by threshold
        for threshold, count in threshold_counts.items():
            self.log(f'val/predictions_above_{threshold}', count, on_step=True, on_epoch=True, batch_size=batch_size)
        
        # Log number of predictions made this batch
        self.log('val/batch_predictions', len(batch_predictions), on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('val/batch_images_with_predictions', len([o for o in outputs if len(o.get('boxes', [])) > 0]), on_step=True, on_epoch=True, batch_size=batch_size)
        
        # Return a dummy value for Lightning
        return {'predictions': len(batch_predictions)}
    
    @property
    def train_batch_size(self) -> int:
        """Return train batch size for Lightning."""
        return self.config.train_batch_size
    
    @property
    def val_batch_size(self) -> int:
        """Return validation batch size for Lightning."""
        return self.config.val_batch_size
    
    def on_sanity_check_start(self):
        """Called at the start of the sanity check."""
        # Log default metric to prevent ModelCheckpoint errors during sanity check
        self.log('val/mAP', 0.0, on_epoch=True, sync_dist=True, batch_size=self.config.val_batch_size)
        
    def on_sanity_check_end(self):
        """Called at the end of the sanity check."""
        # Log default metric to prevent ModelCheckpoint errors
        self.log('val/mAP', 0.0, on_epoch=True, sync_dist=True, batch_size=self.config.val_batch_size)
    
    
    def on_validation_epoch_end(self):
        """Run COCO evaluation at the end of validation epoch."""
        # If validation hasn't run yet, log default metric
        if hasattr(self.config, 'validation_start_step') and self.global_step < self.config.validation_start_step:
            self.log('val/mAP', 0.0, on_epoch=True, sync_dist=True, rank_zero_only=False)
            self.log('val/mAP50', 0.0, on_epoch=True, sync_dist=True, rank_zero_only=False)
            return
            
        if not self.validation_outputs:
            print("No validation predictions to evaluate")
            print(f"Total validation outputs: {len(self.validation_outputs)}")
            print(f"This can happen if all predictions have scores below the threshold or no masks")
            # Log zero metric so ModelCheckpoint doesn't fail
            self.log('val/mAP', 0.0, on_epoch=True, sync_dist=True, rank_zero_only=False)
            self.log('val/mAP50', 0.0, on_epoch=True, sync_dist=True, rank_zero_only=False)
            return
        
        # Save predictions for evaluation
        predictions = self.validation_outputs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pred_file = Path(f'predictions_epoch_{self.current_epoch}_{timestamp}.json')
        
        print(f"Saving {len(predictions)} predictions to {pred_file}")
        with open(pred_file, 'w') as f:
            json.dump(predictions, f)
        
        # If no predictions, return zeros
        if not predictions:
            print("No predictions made! Returning zero metrics.")
            metrics = {
                'mAP': 0.0,
                'mAP50': 0.0,
                'mAP75': 0.0,
                'mAP_small': 0.0,
                'mAP_medium': 0.0,
                'mAP_large': 0.0
            }
        else:
            print(f"Running COCO evaluation with {len(predictions)} predictions...")
            
            # Load predictions as COCO result
            coco_dt = self.val_coco.loadRes(str(pred_file))
            
            # Run COCO evaluation
            coco_eval = COCOeval(self.val_coco, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Extract metrics
            metrics = {
                'mAP': coco_eval.stats[0],
                'mAP50': coco_eval.stats[1],
                'mAP75': coco_eval.stats[2],
                'mAP_small': coco_eval.stats[3],
                'mAP_medium': coco_eval.stats[4],
                'mAP_large': coco_eval.stats[5],
            }
            
            # Also run segmentation evaluation if available
            try:
                coco_eval_seg = COCOeval(self.val_coco, coco_dt, 'segm')
                coco_eval_seg.evaluate()
                coco_eval_seg.accumulate()
                coco_eval_seg.summarize()
                
                metrics.update({
                    'mAP_seg': coco_eval_seg.stats[0],
                    'mAP50_seg': coco_eval_seg.stats[1],
                    'mAP75_seg': coco_eval_seg.stats[2],
                })
            except Exception as e:
                print(f"Segmentation evaluation failed: {e}")
        
        # Log metrics to TensorBoard
        # Important: log mAP first and sync across all GPUs
        self.log('val/mAP', metrics['mAP'], on_epoch=True, sync_dist=True, rank_zero_only=False)
        
        # Log all other metrics
        for metric_name, metric_value in metrics.items():
            self.log(f'val/{metric_name}', metric_value, on_epoch=True, sync_dist=True, rank_zero_only=False)
        
        # Clear validation outputs for next epoch
        self.validation_outputs = []
        
        # Print metrics
        print("\nValidation Results Summary:")
        print(f"  Detection mAP: {metrics['mAP']:.4f}")
        print(f"  Detection mAP50: {metrics['mAP50']:.4f}")
        print(f"  Detection mAP75: {metrics['mAP75']:.4f}")
        print(f"  mAP small: {metrics['mAP_small']:.4f}")
        print(f"  mAP medium: {metrics['mAP_medium']:.4f}")
        print(f"  mAP large: {metrics['mAP_large']:.4f}")
        if 'mAP_seg' in metrics:
            print(f"  Segmentation mAP: {metrics['mAP_seg']:.4f}")
            print(f"  Segmentation mAP50: {metrics['mAP50_seg']:.4f}")
            print(f"  Segmentation mAP75: {metrics['mAP75_seg']:.4f}")
    
    


def main(config_path: Optional[str] = None):
    # Load configuration
    if config_path is None:
        # Default to config.yaml in the same directory as this script
        default_config = Path(__file__).parent / "config" / "config.yaml"
        if default_config.exists():
            config_path = str(default_config)
    
    if config_path:
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = TrainingConfig(**config_dict)
    else:
        # Fallback to default values from TrainingConfig
        config = TrainingConfig()
    
    # Setup logging
    log_dir = config.checkpoint_dir / "logs"
    logger = setup_logger(
        name="train",
        log_dir=str(log_dir),
        level="DEBUG"
    )
    
    logger.info(f"Using PyTorch Lightning + TensorBoard")
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = CocoDataset(
        root_dir=str(config.img_root),
        annotation_file=str(config.train_ann),
        transforms=get_transform_simple(train=True),
        mode='train'
    )
    
    val_dataset = CocoDataset(
        root_dir=str(config.img_root),
        annotation_file=str(config.val_ann),
        transforms=get_transform_simple(train=False),
        mode='train'  # Still use train mode to get targets
    )
    
    # Create subset of validation dataset if max_val_images is specified
    if config.max_val_images is not None and config.max_val_images < len(val_dataset):
        from torch.utils.data import Subset
        indices = list(range(config.max_val_images))
        val_dataset = Subset(val_dataset, indices)
        logger.info(f"Using subset of validation dataset: {config.max_val_images} images")
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True  # Ensure consistent batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False  # Keep all validation samples
    )
    
    # Create COCO object for validation
    val_coco = COCO(str(config.val_ann))
    
    # Create Lightning module
    logger.info("Creating Lightning module...")
    lightning_model = MaskRCNNLightningModule(
        config=config,
        val_coco=val_coco
    )
    
    # Create run-specific directory with timestamp
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.checkpoint_dir / f"run_{run_timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create empty marker file to avoid Lightning warning
    (run_dir / ".lightning_save_marker").touch()
    
    # Setup TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=str(config.checkpoint_dir),
        name="tensorboard",
        version=run_timestamp
    )
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=str(run_dir),
            filename='checkpoint-{epoch:02d}-{step}',
            monitor='val/mAP',
            mode='max',
            save_top_k=3,
            save_last=True
        ),
        LearningRateMonitor(logging_interval='step'),
        ONNXExportCallback(
            export_dir=run_dir,
            export_backbone_only=True,
            save_weights=True
        )
    ]
    
    # Create trainer with validation interval
    val_check_interval = config.steps_per_validation if hasattr(config, 'steps_per_validation') else 1.0
    
    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=tb_logger,
        callbacks=callbacks,
        val_check_interval=val_check_interval,
        log_every_n_steps=config.log_interval,
        gradient_clip_val=config.clip_grad_norm if config.clip_grad_norm > 0 else None,
        precision="16-mixed" if config.use_amp else 32,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Save run configuration
    config_dict = config.model_dump()
    for key, value in config_dict.items():
        if isinstance(value, Path):
            config_dict[key] = str(value)
    
    run_info = {
        "start_timestamp": run_timestamp,
        "config": config_dict,
        "num_train_samples": len(train_dataset),
        "num_val_samples": len(val_dataset),
        "device": str(trainer.device_ids),
        "tensorboard_dir": tb_logger.log_dir
    }
    with open(run_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)
    
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"TensorBoard logs: {tb_logger.log_dir}")
    logger.info(f"To monitor training: tensorboard --logdir {config.checkpoint_dir / 'tensorboard'}")
    
    # Train
    logger.info("Starting training...")
    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    
    logger.info("Training completed!")
    
    # Save final model weights
    final_path = run_dir / "final_model_weights.pth"
    torch.save(lightning_model.model.state_dict(), final_path)
    logger.info(f"Saved final model weights to {final_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train SWIN Mask R-CNN')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (defaults to scripts/config/config.yaml)')
    args = parser.parse_args()
    
    main(config_path=args.config)
