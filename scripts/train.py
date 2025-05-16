"""Training script with iteration-based validation and COCO metrics."""
import json
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.optim import AdamW, SGD, Optimizer
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm
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


class IterationBasedTrainer:
    """Trainer with iteration-based validation and COCO evaluation."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        val_coco,  # COCO object for validation
        config: TrainingConfig,
        device: Optional[torch.device] = None,
        logger = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_coco = val_coco
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        
        # Move model to device
        self.model.to(self.device)
        
        # Create run-specific directory with timestamp
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.config.checkpoint_dir / f"run_{self.run_timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Log the run directory
        if self.logger:
            self.logger.info(f"Created run directory: {self.run_dir}")
        
        # Initialize optimizer
        self.optimizer: Optimizer
        if self.config.optimizer == "adamw":
            self.optimizer = AdamW(
                self.model.parameters(), 
                lr=self.config.lr,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "sgd":
            self.optimizer = SGD(
                self.model.parameters(),
                lr=self.config.lr,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        # Initialize scheduler
        self.scheduler: Optional[OneCycleLR] = None
        
        # Initialize CSV logger for problematic images
        self.error_log_path = self.run_dir / "problematic_images.csv"
        
        # Create CSV file with headers
        with open(self.error_log_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['epoch', 'batch_idx', 'image_filename', 'image_id', 'error_type', 'error_message'])
        if self.config.use_scheduler:
            total_steps = len(train_loader) * self.config.num_epochs
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.lr,
                total_steps=total_steps,
                pct_start=0.05,
                anneal_strategy='cos'
            )
        
        # Training history
        self.train_history: Dict[str, list] = {
            'loss': [],
            'rpn_cls_loss': [],
            'rpn_bbox_loss': [],
            'roi_cls_loss': [],
            'roi_bbox_loss': [],
            'roi_mask_loss': [],
            'memory_mb': [],
            'gpu_utilization': []
        }
        
        self.val_history: Dict[str, list] = {
            'mAP': [],
            'mAP50': [],
            'mAP75': [],
            'mAP_small': [],
            'mAP_medium': [],
            'mAP_large': [],
            'mAP_seg': [],
            'mAP50_seg': [],
            'mAP75_seg': []
        }
        
        self.global_step = 0
        self.best_mAP = 0.0
    
    def train_step(self, images, targets) -> Optional[Dict[str, float]]:
        """Single training step. Returns None if an error occurs."""
        try:
            self.model.train()
            
            # Move to device, preserving string fields
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
            
            # Forward pass
            loss_dict = self.model(images, targets_device)
            
            # Apply loss weights (similar to MMDetection)
            loss_weights = {
                'rpn_cls_loss': 1.0,
                'rpn_bbox_loss': 1.0,
                'roi_cls_loss': 1.0,
                'roi_bbox_loss': 1.0,
                'roi_mask_loss': 1.0,
            }
            
            # Weight the losses
            weighted_losses = {}
            for k, v in loss_dict.items():
                weight = loss_weights.get(k, 1.0)
                weighted_losses[k] = v * weight
            
            total_loss = sum(weighted_losses.values())
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            if self.config.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
            
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            
            # Convert losses to dict
            loss_values = {k: v.item() for k, v in loss_dict.items()}
            loss_values['total'] = total_loss.item()
            
            # Add memory usage and GPU utilization
            loss_values['memory_mb'] = get_gpu_memory_mb()
            loss_values['gpu_utilization'] = get_gpu_utilization()
            
            return loss_values
        
        except Exception as e:
            # Log the error with information about the problematic images
            error_info = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'targets': targets  # Keep the original targets with filenames
            }
            
            if self.logger:
                self.logger.warning(f"Error during training step: {e}")
            
            return error_info
    
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
    
    @torch.no_grad()
    def evaluate_coco(self) -> Dict[str, float]:
        """Evaluate model using COCO metrics."""
        self.logger.info("Starting COCO evaluation...")
        self.model.eval()
        
        predictions: list[Dict[str, Any]] = []
        total_images = 0
        
        pbar = tqdm(self.val_loader, desc="Evaluating (inference)")
        for batch_idx, (images, targets) in enumerate(pbar):
            # Move to device
            images = [img.to(self.device) for img in images]
            
            # Get predictions
            outputs = self.model(images)
            
            batch_size = len(images)
            total_images += batch_size
            pbar.set_postfix({
                'images': f'{total_images}',
                'predictions': f'{len(predictions)}'
            })
            
            # Process each image
            for i, output in enumerate(outputs):
                # Get original image ID from target
                image_id = targets[i]['image_id'].item()
                
                # Count predictions per image
                num_preds = len(output['boxes'])
                
                # Debug logging for first few batches
                if batch_idx < 5:
                    if num_preds > 0:
                        max_score = output['scores'].max().item() if 'scores' in output else 0
                        label_dist = torch.bincount(output['labels']).tolist() if 'labels' in output else []
                        self.logger.debug(f"Image {image_id}: {num_preds} detections, max_score: {max_score:.4f}, labels: {label_dist}")
                    else:
                        self.logger.debug(f"Image {image_id}: No detections!")
                
                # Skip if no predictions or no masks
                if output['masks'] is None:
                    continue
                    
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
                    
                    predictions.append({
                        'image_id': int(image_id),
                        'category_id': int(label),
                        'bbox': bbox,
                        'score': float(score),
                        'segmentation': rle  # RLE format
                    })
        
        # If no predictions, return zeros
        if not predictions:
            self.logger.warning("No predictions made! Returning zero metrics.")
            return {
                'mAP': 0.0,
                'mAP50': 0.0,
                'mAP75': 0.0,
                'mAP_small': 0.0,
                'mAP_medium': 0.0,
                'mAP_large': 0.0
            }
        
        self.logger.info(f"Collected {len(predictions)} predictions across {total_images} images")
        
        # Save predictions for evaluation
        pred_file = self.run_dir / f'predictions_step_{self.global_step}.json'
        self.logger.info(f"Saving predictions to {pred_file}")
        with open(pred_file, 'w') as f:
            json.dump(predictions, f)
        
        # Load predictions as COCO result
        self.logger.info("Loading predictions for COCO evaluation...")
        coco_dt = self.val_coco.loadRes(str(pred_file))
        
        # Run COCO evaluation
        self.logger.info("Running COCO bounding box evaluation...")
        coco_eval = COCOeval(self.val_coco, coco_dt, 'bbox')
        self.logger.info("  - Evaluating detections...")
        coco_eval.evaluate()
        self.logger.info("  - Accumulating results...")
        coco_eval.accumulate()
        self.logger.info("  - Computing metrics...")
        coco_eval.summarize()
        
        # Extract metrics
        metrics = {
            'mAP': coco_eval.stats[0],  # AP @ IoU=0.50:0.95
            'mAP50': coco_eval.stats[1],  # AP @ IoU=0.50
            'mAP75': coco_eval.stats[2],  # AP @ IoU=0.75
            'mAP_small': coco_eval.stats[3],  # AP for small objects
            'mAP_medium': coco_eval.stats[4],  # AP for medium objects
            'mAP_large': coco_eval.stats[5],  # AP for large objects
        }
        
        # Also run segmentation evaluation if available
        try:
            self.logger.info("Running COCO segmentation evaluation...")
            coco_eval_seg = COCOeval(self.val_coco, coco_dt, 'segm')
            self.logger.info("  - Evaluating segmentations...")
            coco_eval_seg.evaluate()
            self.logger.info("  - Accumulating results...")
            coco_eval_seg.accumulate()
            self.logger.info("  - Computing metrics...")
            coco_eval_seg.summarize()
            
            metrics.update({
                'mAP_seg': coco_eval_seg.stats[0],
                'mAP50_seg': coco_eval_seg.stats[1],
                'mAP75_seg': coco_eval_seg.stats[2],
            })
        except Exception as e:
            # If segmentation evaluation fails, just skip it
            self.logger.warning(f"Segmentation evaluation failed: {e}")
            pass
        
        return metrics
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_mAP': self.best_mAP,
            'metrics': metrics
        }
        
        # Include mAP50 in filename if available
        map50_str = ""
        if metrics and 'mAP50' in metrics:
            map50_str = f"_map50_{metrics['mAP50']:.4f}"
        
        # Save regular checkpoint
        checkpoint_path = self.run_dir / f"checkpoint_step_{self.global_step}{map50_str}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Also save just the model weights separately (for easier inference loading)
        weights_path = self.run_dir / f"model_weights_step_{self.global_step}{map50_str}.pth"
        torch.save(self.model.state_dict(), weights_path)
        self.logger.info(f"Saved checkpoint and model weights at step {self.global_step} with mAP50: {metrics.get('mAP50', 0.0):.4f}")
        
        # Save best checkpoint
        if is_best:
            best_path = self.run_dir / f"best{map50_str}.pth"
            torch.save(checkpoint, best_path)
            
            # Also save best model weights separately
            best_weights_path = self.run_dir / f"best_model_weights{map50_str}.pth"
            torch.save(self.model.state_dict(), best_weights_path)
            self.logger.info(f"Saved best model with mAP: {self.best_mAP:.4f}, mAP50: {metrics.get('mAP50', 0.0):.4f}")
    
    def train(self):
        """Run the complete training loop."""
        self.logger.info(f"Starting training on device: {self.device}")
        self.logger.info(f"Run directory: {self.run_dir}")
        self.logger.info(f"Number of training samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Number of validation samples: {len(self.val_loader.dataset)}")
        self.logger.info(f"Validation starts after {self.config.validation_start_step} training steps")
        self.logger.info(f"Running validation every {self.config.steps_per_validation} steps thereafter")
        
        # Save run configuration
        # Convert Path objects to strings for JSON serialization
        config_dict = self.config.model_dump()
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
        
        run_info = {
            "start_timestamp": self.run_timestamp,
            "config": config_dict,
            "num_train_samples": len(self.train_loader.dataset),
            "num_val_samples": len(self.val_loader.dataset),
            "device": str(self.device)
        }
        with open(self.run_dir / "run_info.json", "w") as f:
            json.dump(run_info, f, indent=2)
        
        epoch = 0
        
        while epoch < self.config.num_epochs:
            epoch_start_time = time.time()
            epoch_losses = {
                'loss': [],
                'rpn_cls_loss': [],
                'rpn_bbox_loss': [],
                'roi_cls_loss': [],
                'roi_bbox_loss': [],
                'roi_mask_loss': [],
                'memory_mb': [],
                'gpu_utilization': []
            }
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            for batch_idx, (images, targets) in enumerate(pbar):
                # Training step
                loss_values = self.train_step(images, targets)
                
                # Check if this is an error or successful training step
                if 'error_type' in loss_values:
                    # This is an error - log it and skip this batch
                    self.log_error_to_csv(epoch, batch_idx, loss_values, targets)
                    self.logger.warning(f"Skipping batch {batch_idx} due to error: {loss_values['error_message']}")
                    continue
                
                # Record losses (only for successful steps)
                for key in epoch_losses:
                    if key == 'loss':
                        epoch_losses[key].append(loss_values['total'])
                    elif key in ['memory_mb', 'gpu_utilization']:
                        epoch_losses[key].append(loss_values.get(key, 0.0))
                    else:
                        epoch_losses[key].append(loss_values.get(key, 0.0))
                
                # Update progress bar (only if we have valid losses)
                if epoch_losses['loss']:  # Check if we have any successful steps
                    latest_loss = epoch_losses['loss'][-1]
                    mem_mb = epoch_losses['memory_mb'][-1] if epoch_losses['memory_mb'] else 0
                    gpu_util = epoch_losses['gpu_utilization'][-1] if epoch_losses['gpu_utilization'] else -1
                    gpu_str = f'{gpu_util:.0f}%' if gpu_util >= 0 else 'N/A'
                    
                    if self.scheduler:
                        pbar.set_postfix({
                            'loss': f'{latest_loss:.4f}',
                            'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
                            'mem': f'{mem_mb:.0f}MB',
                            'gpu': gpu_str
                        })
                    else:
                        pbar.set_postfix({
                            'loss': f'{latest_loss:.4f}',
                            'lr': f'{self.config.lr:.2e}',
                            'mem': f'{mem_mb:.0f}MB',
                        'gpu': gpu_str
                    })
                
                # Log every N steps
                if self.global_step % self.config.log_interval == 0:
                    avg_loss = np.mean(epoch_losses['loss'][-self.config.log_interval:])
                    # Calculate average GPU utilization over last N steps
                    recent_gpu = epoch_losses['gpu_utilization'][-self.config.log_interval:]
                    avg_gpu = np.mean([g for g in recent_gpu if g >= 0]) if recent_gpu else -1
                    
                    gpu_util = loss_values.get('gpu_utilization', -1)
                    gpu_str = f'{gpu_util:.0f}%' if gpu_util >= 0 else 'N/A'
                    avg_gpu_str = f'{avg_gpu:.0f}%' if avg_gpu >= 0 else 'N/A'
                    
                    self.logger.info(f"Step {self.global_step}, Loss: {avg_loss:.4f}, Memory: {loss_values['memory_mb']:.0f}MB, " +
                          f"GPU: {gpu_str} (avg: {avg_gpu_str})")
                
                # Run validation every N steps after initial training period
                if (self.global_step >= self.config.validation_start_step and 
                    self.global_step % self.config.steps_per_validation == 0):
                    self.logger.info(f"{'='*60}")
                    self.logger.info(f"VALIDATION at step {self.global_step}")
                    self.logger.info(f"{'='*60}")
                    
                    val_start_time = time.time()
                    
                    # Run COCO evaluation
                    eval_start_time = time.time()
                    coco_metrics = self.evaluate_coco()
                    eval_end_time = time.time()
                    self.logger.info(f"COCO evaluation took {eval_end_time - eval_start_time:.1f}s")
                    
                    # Print metrics
                    self.logger.info("\nValidation Results Summary:")
                    self.logger.info(f"  Detection mAP: {coco_metrics['mAP']:.4f}")
                    self.logger.info(f"  Detection mAP50: {coco_metrics['mAP50']:.4f}")
                    self.logger.info(f"  Detection mAP75: {coco_metrics['mAP75']:.4f}")
                    self.logger.info(f"  mAP small: {coco_metrics['mAP_small']:.4f}")
                    self.logger.info(f"  mAP medium: {coco_metrics['mAP_medium']:.4f}")
                    self.logger.info(f"  mAP large: {coco_metrics['mAP_large']:.4f}")
                    if 'mAP_seg' in coco_metrics:
                        self.logger.info(f"  Segmentation mAP: {coco_metrics['mAP_seg']:.4f}")
                        self.logger.info(f"  Segmentation mAP50: {coco_metrics['mAP50_seg']:.4f}")
                        self.logger.info(f"  Segmentation mAP75: {coco_metrics['mAP75_seg']:.4f}")
                    
                    # Update history
                    for key, value in coco_metrics.items():
                        if key in self.val_history:
                            self.val_history[key].append(value)
                    
                    # Save checkpoint
                    is_best = coco_metrics['mAP'] > self.best_mAP
                    if is_best:
                        self.best_mAP = coco_metrics['mAP']
                        self.logger.info(f"New best mAP: {self.best_mAP:.4f}")
                    
                    self.save_checkpoint(coco_metrics, is_best)
                    
                    val_end_time = time.time()
                    self.logger.info(f"Total validation time: {val_end_time - val_start_time:.1f}s")
                    self.logger.info(f"{'='*60}")
                
                self.global_step += 1
            
            # Update epoch-level history
            avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
            for key, value in avg_losses.items():
                self.train_history[key].append(value)
            
            # End of epoch summary
            epoch_time = time.time() - epoch_start_time
            avg_gpu = np.mean([g for g in epoch_losses['gpu_utilization'] if g >= 0])
            self.logger.info(f"Epoch {epoch+1} Summary:")
            self.logger.info(f"  Time: {epoch_time:.1f}s")
            self.logger.info(f"  Steps: {len(epoch_losses['loss'])}")
            self.logger.info(f"  Avg Loss: {avg_losses['loss']:.4f}")
            self.logger.info(f"  Avg GPU: {avg_gpu:.0f}%")
            self.logger.info(f"  Avg Memory: {avg_losses['memory_mb']:.0f}MB")
            self.logger.info(f"  Steps/sec: {len(epoch_losses['loss'])/epoch_time:.2f}")
            
            epoch += 1
        
        self.logger.info("Training completed!")
        self.logger.info(f"Best mAP: {self.best_mAP:.4f}")
        
        # Save final model
        final_path = self.config.checkpoint_dir / "final.pth"
        torch.save(self.model.state_dict(), final_path)
        self.logger.info(f"Saved final model to {final_path}")
    


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
        level="DEBUG"  # Changed from INFO to DEBUG to see debug messages
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create datasets
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
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn
    )
    
    # Create COCO object for validation
    val_coco = COCO(str(config.val_ann))
    
    # Create model
    model = SwinMaskRCNN(
        num_classes=config.num_classes,
        frozen_backbone_stages=config.frozen_backbone_stages  # Use the frozen stages from config
    )
    model.logger = logger  # Add logger to model for debug output
    model.roi_head.logger = logger  # Add logger to ROI head for debug output
    
    # Load pretrained weights if specified
    if config.pretrained_backbone and config.pretrained_checkpoint_url:
        logger.info(f"Loading pretrained weights from {config.pretrained_checkpoint_url}")
        load_pretrained_from_url(model, config.pretrained_checkpoint_url, strict=False)
    
    model = model.to(device)
    
    # Create trainer
    trainer = IterationBasedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        val_coco=val_coco,
        config=config,
        logger=logger
    )
    
    # Train
    trainer.train()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train SWIN Mask R-CNN')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (defaults to scripts/config/config.yaml)')
    args = parser.parse_args()
    
    main(config_path=args.config)
