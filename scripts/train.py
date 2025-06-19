"""Training script with PyTorch Lightning and TensorBoard integration."""
import json
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.optim import AdamW, SGD, Optimizer
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Subset
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
from swin_maskrcnn.callbacks import ONNXExportCallback, MetricsTracker


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


class MetricsTracker(Callback):
    """Callback to track and export metrics for real-time monitoring."""
    
    def __init__(self, export_dir: Path, window_size: int = 100):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.window_size = window_size
        
        # Moving averages for losses
        self.loss_windows = defaultdict(lambda: deque(maxlen=window_size))
        self.metric_history = []
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Track training metrics."""
        if outputs is None:
            return
            
        # Get logged metrics (use callback_metrics for more complete data)
        metrics = trainer.callback_metrics
        
        # Update moving averages
        for key, value in metrics.items():
            if 'train/' in key and isinstance(value, (int, float)):
                self.loss_windows[key].append(float(value))
        
        # Export current state
        self._export_metrics(trainer, pl_module)
        
    def on_validation_end(self, trainer, pl_module):
        """Track validation metrics."""
        self._export_metrics(trainer, pl_module)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Track validation metrics after epoch end (includes per-class metrics)."""
        self._export_metrics(trainer, pl_module)
        
    def _export_metrics(self, trainer, pl_module):
        """Export current metrics to JSON."""
        current_metrics = {
            'step': trainer.global_step,
            'epoch': trainer.current_epoch,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Add moving averages
        for key, window in self.loss_windows.items():
            if len(window) > 0:
                current_metrics[f'{key}_avg'] = np.mean(list(window))
                current_metrics[f'{key}_std'] = np.std(list(window))
                
                # Calculate trend (positive = increasing, negative = decreasing)
                if len(window) >= 10:
                    recent = list(window)[-10:]
                    older = list(window)[-20:-10] if len(window) >= 20 else list(window)[:10]
                    trend = np.mean(recent) - np.mean(older)
                    current_metrics[f'{key}_trend'] = trend
        
        # Add current logged metrics (use callback_metrics for more complete data)
        for key, value in trainer.callback_metrics.items():
            if isinstance(value, (int, float)):
                current_metrics[key] = float(value)
        
        # Add class-level metrics from the module if available
        if hasattr(trainer.lightning_module, 'latest_class_metrics'):
            for key, value in trainer.lightning_module.latest_class_metrics.items():
                if isinstance(value, (int, float)):
                    current_metrics[key] = float(value)
        
        # Add quick eval metrics from the module if available
        if hasattr(trainer.lightning_module, 'latest_quick_eval_metrics'):
            for key, value in trainer.lightning_module.latest_quick_eval_metrics.items():
                if isinstance(value, (int, float)):
                    current_metrics[key] = float(value)
        
        # Save to file
        metrics_file = self.export_dir / 'metrics.json'
        self.metric_history.append(current_metrics)
        
        # Keep only last 1000 entries to prevent file from growing too large
        if len(self.metric_history) > 1000:
            self.metric_history = self.metric_history[-1000:]
            
        with open(metrics_file, 'w') as f:
            json.dump(self.metric_history, f, indent=2)


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
        
        # Quick eval settings
        self.quick_eval_enabled = getattr(config, 'quick_eval_enabled', False)
        self.quick_eval_interval = getattr(config, 'quick_eval_interval', 50)
        self.quick_eval_samples = getattr(config, 'quick_eval_samples', 50)
        self.track_top_k_classes = getattr(config, 'track_top_k_classes', 10)
        
        # Class-level metrics tracking
        self.class_predictions_buffer = {}
        self.class_metrics_history = []
        self.latest_class_metrics = {}  # Store latest class metrics for export
        self.latest_quick_eval_metrics = {}  # Store latest quick eval metrics
        
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
            # Check if we have a specific checkpoint to load
            if hasattr(config, 'checkpoint_path') and config.checkpoint_path:
                print(f"Loading checkpoint from {config.checkpoint_path}")
                checkpoint = torch.load(config.checkpoint_path, map_location='cpu', weights_only=False)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                print(f"Missing keys: {len(missing_keys)}")
                print(f"Unexpected keys: {len(unexpected_keys)}")
            else:
                # Load COCO pretrained weights with proper initialization
                print("Loading COCO pretrained weights...")
                missing_keys, unexpected_keys = load_coco_weights(self.model, num_classes=config.num_classes)
                print(f"Missing keys: {len(missing_keys)}")
                print(f"Unexpected keys: {len(unexpected_keys)}")
            
            # Log some diagnostic info about the model
            if hasattr(self.model.roi_head.bbox_head, 'fc_cls'):
                cls_bias = self.model.roi_head.bbox_head.fc_cls.bias.detach().cpu().numpy()
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
        
        # Quick evaluation setup
        self.quick_eval_enabled = config.quick_eval_enabled if hasattr(config, 'quick_eval_enabled') else False
        self.quick_eval_interval = config.quick_eval_interval if hasattr(config, 'quick_eval_interval') else 50
        self.quick_eval_samples = config.quick_eval_samples if hasattr(config, 'quick_eval_samples') else 50
        self.track_top_k_classes = config.track_top_k_classes if hasattr(config, 'track_top_k_classes') else 10
        self.last_quick_eval_step = 0
        self.quick_eval_outputs = []
        
        # Class performance tracking
        self.class_performance = defaultdict(lambda: {'predictions': 0, 'correct': 0, 'total_gt': 0})
        self.recent_class_aps = defaultdict(list)  # Track recent APs for each class
        
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
            
            # Run quick evaluation if enabled
            if self.quick_eval_enabled and (self.global_step - self.last_quick_eval_step) >= self.quick_eval_interval:
                self._run_quick_evaluation()
                self.last_quick_eval_step = self.global_step
            
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
    
    def _run_quick_evaluation(self):
        """Run a quick evaluation on a subset of validation data for fast feedback."""
        if not hasattr(self, 'quick_eval_loader'):
            # Create a small subset loader for quick evaluation
            from torch.utils.data import Subset, DataLoader
            from swin_maskrcnn.data.dataset import CocoDataset
            from swin_maskrcnn.data.transforms_simple import get_transform_simple
            
            # Create a small validation dataset
            val_dataset = CocoDataset(
                root_dir=str(self.config.img_root),
                annotation_file=str(self.config.val_ann),
                transforms=get_transform_simple(train=False),
                mode='train'  # Still use train mode to get targets
            )
                
            # Create subset
            indices = list(range(min(self.quick_eval_samples, len(val_dataset))))
            quick_dataset = Subset(val_dataset, indices)
            
            self.quick_eval_loader = DataLoader(
                quick_dataset,
                batch_size=self.config.val_batch_size,
                shuffle=False,
                num_workers=0,  # Use 0 workers for speed
                collate_fn=collate_fn,
                pin_memory=True
            )
        
        # Run quick evaluation
        self.eval()
        quick_predictions = []
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(self.quick_eval_loader):
                if batch_idx >= 5:  # Limit batches for speed
                    break
                    
                images = [img.to(self.device) for img in images]
                outputs = self.forward(images)
                
                # Process predictions
                for i, output in enumerate(outputs):
                    if len(output.get('boxes', [])) == 0:
                        continue
                        
                    image_id = targets[i]['image_id'].item()
                    
                    for box, label, score in zip(
                        output['boxes'].cpu().numpy(),
                        output['labels'].cpu().numpy(),
                        output['scores'].cpu().numpy()
                    ):
                        if score > 0.05:  # Higher threshold for quick eval
                            quick_predictions.append({
                                'image_id': int(image_id),
                                'category_id': int(label),
                                'score': float(score)
                            })
        
        self.train()
        
        # Compute quick metrics by category
        cat_predictions = {}
        for pred in quick_predictions:
            cat_id = pred['category_id']
            if cat_id not in cat_predictions:
                cat_predictions[cat_id] = []
            cat_predictions[cat_id].append(pred['score'])
        
        # Log top performing categories
        sorted_cats = sorted(cat_predictions.items(), 
                           key=lambda x: (len(x[1]), np.mean(x[1])), 
                           reverse=True)
        
        # Log quick metrics
        self.latest_quick_eval_metrics = {}  # Reset quick eval metrics
        
        for idx, (cat_id, scores) in enumerate(sorted_cats[:self.track_top_k_classes]):
            count_key = f'quick_eval/top_{idx}_cat_{cat_id}_count'
            score_key = f'quick_eval/top_{idx}_cat_{cat_id}_avg_score'
            
            self.log(count_key, len(scores), 
                    on_step=True, batch_size=self.config.train_batch_size)
            self.log(score_key, np.mean(scores), 
                    on_step=True, batch_size=self.config.train_batch_size)
            
            # Store for export
            self.latest_quick_eval_metrics[count_key] = len(scores)
            self.latest_quick_eval_metrics[score_key] = float(np.mean(scores))
        
        total_quick_preds = len(quick_predictions)
        self.log('quick_eval/total_predictions', total_quick_preds, 
                on_step=True, prog_bar=True, batch_size=self.config.train_batch_size)
        self.log('quick_eval/categories_detected', len(cat_predictions), 
                on_step=True, batch_size=self.config.train_batch_size)
        
        # Store for export
        self.latest_quick_eval_metrics['quick_eval/total_predictions'] = total_quick_preds
        self.latest_quick_eval_metrics['quick_eval/categories_detected'] = len(cat_predictions)
    
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
                # Comment out threshold for debugging
                # if score < 0.001:  # Using very low threshold
                #     continue
                
                # Convert mask to binary format and then to RLE
                # mask already should be shape [H, W]
                if mask.ndim == 3:
                    mask_binary = (mask[0] > 0.5).astype(np.uint8)
                else:
                    mask_binary = (mask > 0.5).astype(np.uint8)
                
                # For untrained models, masks might be all zeros - check this
                if mask_binary.sum() == 0:
                    # Create a small box mask based on the bounding box
                    if mask_binary.ndim == 2:
                        h, w = mask_binary.shape
                    else:
                        # Should not happen, but handle edge case
                        h, w = mask_binary.shape[-2:]
                    x1n, y1n, x2n, y2n = box.astype(int)
                    x1n = max(0, min(x1n, w-1))
                    y1n = max(0, min(y1n, h-1))
                    x2n = max(0, min(x2n, w))
                    y2n = max(0, min(y2n, h))
                    mask_binary[y1n:y2n, x1n:x2n] = 1
                
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
            
            # Compute per-class metrics with enhanced display
            print("\n" + "="*80)
            print(f"PER-CLASS METRICS (Detection - AP@0.5:0.95) - Epoch {self.current_epoch}, Step {self.global_step}")
            print("="*80)
            
            # Get category info
            cat_ids = self.val_coco.getCatIds()
            cat_names = {cat['id']: cat['name'] for cat in self.val_coco.loadCats(cat_ids)}
            
            # Compute per-class AP for detection
            class_metrics = {}
            for cat_id in cat_ids:
                # Filter predictions for this category
                cat_predictions = [p for p in predictions if p['category_id'] == cat_id]
                
                if not cat_predictions:
                    class_metrics[cat_id] = {
                        'name': cat_names[cat_id],
                        'ap': 0.0,
                        'ap50': 0.0,
                        'num_predictions': 0,
                        'num_gt': len(self.val_coco.getAnnIds(catIds=[cat_id]))
                    }
                else:
                    # Create temporary prediction file for this category
                    temp_pred_file = Path(f'temp_cat_{cat_id}_predictions.json')
                    with open(temp_pred_file, 'w') as f:
                        json.dump(cat_predictions, f)
                    
                    try:
                        # Load predictions for this category
                        coco_dt_cat = self.val_coco.loadRes(str(temp_pred_file))
                        
                        # Run evaluation for this category only
                        coco_eval_cat = COCOeval(self.val_coco, coco_dt_cat, 'bbox')
                        coco_eval_cat.params.catIds = [cat_id]
                        coco_eval_cat.evaluate()
                        coco_eval_cat.accumulate()
                        coco_eval_cat.summarize()
                        
                        class_metrics[cat_id] = {
                            'name': cat_names[cat_id],
                            'ap': coco_eval_cat.stats[0],
                            'ap50': coco_eval_cat.stats[1],
                            'num_predictions': len(cat_predictions),
                            'num_gt': len(self.val_coco.getAnnIds(catIds=[cat_id]))
                        }
                        
                        # Clean up temp file
                        temp_pred_file.unlink(missing_ok=True)
                    except Exception as e:
                        print(f"Error computing metrics for category {cat_names[cat_id]}: {e}")
                        class_metrics[cat_id] = {
                            'name': cat_names[cat_id],
                            'ap': 0.0,
                            'ap50': 0.0,
                            'num_predictions': len(cat_predictions),
                            'num_gt': len(self.val_coco.getAnnIds(catIds=[cat_id]))
                        }
            
            # Sort by AP score and print with enhanced formatting
            sorted_classes = sorted(class_metrics.items(), key=lambda x: x[1]['ap'], reverse=True)
            
            # Print top performing classes first
            print(f"\n{'TOP PERFORMING CLASSES':^62}")
            print(f"{'Class Name':<30} {'AP':>8} {'AP50':>8} {'Preds':>8} {'GT':>8} {'Recall':>8}")
            print("-" * 70)
            
            top_k = min(10, len(sorted_classes))
            for cat_id, metrics_dict in sorted_classes[:top_k]:
                recall = metrics_dict['num_predictions'] / max(1, metrics_dict['num_gt'])
                print(f"{metrics_dict['name']:<30} {metrics_dict['ap']:>8.4f} {metrics_dict['ap50']:>8.4f} "
                      f"{metrics_dict['num_predictions']:>8} {metrics_dict['num_gt']:>8} {recall:>8.2f}")
            
            # Print struggling classes
            print(f"\n{'CLASSES NEEDING ATTENTION':^62}")
            print(f"{'Class Name':<30} {'AP':>8} {'AP50':>8} {'Preds':>8} {'GT':>8} {'Recall':>8}")
            print("-" * 70)
            
            # Show bottom performing classes with GT annotations
            bottom_classes = [x for x in sorted_classes if x[1]['num_gt'] > 0][-10:]
            for cat_id, metrics_dict in bottom_classes:
                recall = metrics_dict['num_predictions'] / max(1, metrics_dict['num_gt'])
                print(f"{metrics_dict['name']:<30} {metrics_dict['ap']:>8.4f} {metrics_dict['ap50']:>8.4f} "
                      f"{metrics_dict['num_predictions']:>8} {metrics_dict['num_gt']:>8} {recall:>8.2f}")
            
            # Print summary statistics
            print("-" * 62)
            classes_with_predictions = sum(1 for m in class_metrics.values() if m['num_predictions'] > 0)
            classes_with_ap = sum(1 for m in class_metrics.values() if m['ap'] > 0.0)
            avg_ap_all = np.mean([m['ap'] for m in class_metrics.values()])
            avg_ap50_all = np.mean([m['ap50'] for m in class_metrics.values()])
            
            print(f"Classes with predictions: {classes_with_predictions}/{len(class_metrics)}")
            print(f"Classes with AP > 0: {classes_with_ap}/{len(class_metrics)}")
            print(f"Average AP (all classes): {avg_ap_all:.4f}")
            print(f"Average AP50 (all classes): {avg_ap50_all:.4f}")
            print("="*80 + "\n")
            
            # Log per-class metrics to TensorBoard (only top/bottom K to avoid clutter)
            for idx, (cat_id, metrics_dict) in enumerate(sorted_classes[:self.track_top_k_classes]):
                self.log(f'val/top_{idx}_{metrics_dict["name"]}_ap', metrics_dict['ap'], 
                        on_epoch=True, sync_dist=True)
                self.log(f'val/top_{idx}_{metrics_dict["name"]}_ap50', metrics_dict['ap50'], 
                        on_epoch=True, sync_dist=True)
                
            # Log aggregate metrics by performance tier
            top_10_ap = np.mean([m[1]['ap'] for m in sorted_classes[:10]])
            bottom_10_ap = np.mean([m[1]['ap'] for m in sorted_classes[-10:] if m[1]['num_gt'] > 0])
            
            self.log('val/top_10_classes_mAP', top_10_ap, on_epoch=True, sync_dist=True)
            self.log('val/bottom_10_classes_mAP', bottom_10_ap, on_epoch=True, sync_dist=True)
            self.log('val/ap_spread', top_10_ap - bottom_10_ap, on_epoch=True, sync_dist=True)
            
            # Store class metrics for export
            self.latest_class_metrics = {
                'val/top_10_classes_mAP': top_10_ap,
                'val/bottom_10_classes_mAP': bottom_10_ap,
                'val/ap_spread': top_10_ap - bottom_10_ap
            }
            
            # Add per-class metrics to storage
            for idx, (cat_id, metrics_dict) in enumerate(sorted_classes[:self.track_top_k_classes]):
                self.latest_class_metrics[f'val/top_{idx}_{metrics_dict["name"]}_ap'] = metrics_dict['ap']
                self.latest_class_metrics[f'val/top_{idx}_{metrics_dict["name"]}_ap50'] = metrics_dict['ap50']
            
            # Save detailed class metrics to JSON for external monitoring
            class_metrics_file = Path(f'class_metrics_epoch_{self.current_epoch}_step_{self.global_step}.json')
            class_metrics_export = {
                'epoch': self.current_epoch,
                'step': self.global_step,
                'timestamp': datetime.now().isoformat(),
                'overall_metrics': {
                    'mAP': metrics['mAP'],
                    'mAP50': metrics['mAP50'],
                    'classes_with_predictions': classes_with_predictions,
                    'classes_with_ap': classes_with_ap,
                    'total_classes': len(class_metrics)
                },
                'per_class_metrics': class_metrics,
                'sorted_by_ap': [
                    {
                        'rank': idx + 1,
                        'cat_id': cat_id,
                        'name': m['name'],
                        'ap': m['ap'],
                        'ap50': m['ap50'],
                        'num_predictions': m['num_predictions'],
                        'num_gt': m['num_gt'],
                        'recall': m['num_predictions'] / max(1, m['num_gt'])
                    }
                    for idx, (cat_id, m) in enumerate(sorted_classes)
                ]
            }
            
            with open(class_metrics_file, 'w') as f:
                json.dump(class_metrics_export, f, indent=2)
            print(f"Saved detailed class metrics to: {class_metrics_file}")
        
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
        ),
        MetricsTracker(
            export_dir=run_dir / "metrics",
            window_size=100
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
