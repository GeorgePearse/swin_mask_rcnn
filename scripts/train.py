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

from swin_maskrcnn.data.dataset import CocoDataset
from swin_maskrcnn.data.transforms_simple import get_transform_simple
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.utils.collate import collate_fn
from scripts.config import TrainingConfig
from swin_maskrcnn.utils.pretrained_loader import load_pretrained_from_url


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
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_coco = val_coco
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        self.model.to(self.device)
        
        # Create checkpoint directory
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
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
    
    def train_step(self, images, targets) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        # Move to device
        images = [img.to(self.device) for img in images]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = self.model(images, targets)
        
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
    
    @torch.no_grad()
    def evaluate_coco(self) -> Dict[str, float]:
        """Evaluate model using COCO metrics."""
        print("\nStarting COCO evaluation...")
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
                if num_preds > 0:
                    pbar.set_description(f"Evaluating (image {image_id}, {num_preds} detections)")
                
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
                    # Only keep predictions with score > 0.05
                    if score < 0.05:
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
            print("Warning: No predictions made! Returning zero metrics.")
            return {
                'mAP': 0.0,
                'mAP50': 0.0,
                'mAP75': 0.0,
                'mAP_small': 0.0,
                'mAP_medium': 0.0,
                'mAP_large': 0.0
            }
        
        print(f"\nCollected {len(predictions)} predictions across {total_images} images")
        
        # Save predictions for evaluation
        pred_file = self.config.checkpoint_dir / f'predictions_step_{self.global_step}.json'
        print(f"Saving predictions to {pred_file}")
        with open(pred_file, 'w') as f:
            json.dump(predictions, f)
        
        # Load predictions as COCO result
        print("Loading predictions for COCO evaluation...")
        coco_dt = self.val_coco.loadRes(str(pred_file))
        
        # Run COCO evaluation
        print("Running COCO bounding box evaluation...")
        coco_eval = COCOeval(self.val_coco, coco_dt, 'bbox')
        print("  - Evaluating detections...")
        coco_eval.evaluate()
        print("  - Accumulating results...")
        coco_eval.accumulate()
        print("  - Computing metrics...")
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
            print("\nRunning COCO segmentation evaluation...")
            coco_eval_seg = COCOeval(self.val_coco, coco_dt, 'segm')
            print("  - Evaluating segmentations...")
            coco_eval_seg.evaluate()
            print("  - Accumulating results...")
            coco_eval_seg.accumulate()
            print("  - Computing metrics...")
            coco_eval_seg.summarize()
            
            metrics.update({
                'mAP_seg': coco_eval_seg.stats[0],
                'mAP50_seg': coco_eval_seg.stats[1],
                'mAP75_seg': coco_eval_seg.stats[2],
            })
        except Exception as e:
            # If segmentation evaluation fails, just skip it
            print(f"Segmentation evaluation failed: {e}")
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
        
        # Save regular checkpoint
        checkpoint_path = self.config.checkpoint_dir / f"checkpoint_step_{self.global_step}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.config.checkpoint_dir / "best.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best model with mAP: {self.best_mAP:.4f}")
    
    def train(self):
        """Run the complete training loop."""
        print(f"Starting training on device: {self.device}")
        print(f"Number of training samples: {len(self.train_loader.dataset)}")
        print(f"Number of validation samples: {len(self.val_loader.dataset)}")
        print(f"Validation starts after {self.config.validation_start_step} training steps")
        print(f"Running validation every {self.config.steps_per_validation} steps thereafter")
        
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
                
                # Record losses
                for key in epoch_losses:
                    if key == 'loss':
                        epoch_losses[key].append(loss_values['total'])
                    elif key in ['memory_mb', 'gpu_utilization']:
                        epoch_losses[key].append(loss_values.get(key, 0.0))
                    else:
                        epoch_losses[key].append(loss_values.get(key, 0.0))
                
                # Update progress bar
                gpu_util = loss_values.get('gpu_utilization', -1)
                gpu_str = f'{gpu_util:.0f}%' if gpu_util >= 0 else 'N/A'
                
                if self.scheduler:
                    pbar.set_postfix({
                        'loss': f'{loss_values["total"]:.4f}',
                        'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
                        'mem': f'{loss_values["memory_mb"]:.0f}MB',
                        'gpu': gpu_str
                    })
                else:
                    pbar.set_postfix({
                        'loss': f'{loss_values["total"]:.4f}',
                        'lr': f'{self.config.lr:.2e}',
                        'mem': f'{loss_values["memory_mb"]:.0f}MB',
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
                    
                    print(f"\nStep {self.global_step}, Loss: {avg_loss:.4f}, Memory: {loss_values['memory_mb']:.0f}MB, " +
                          f"GPU: {gpu_str} (avg: {avg_gpu_str})")
                
                # Run validation every N steps after initial training period
                if (self.global_step >= self.config.validation_start_step and 
                    self.global_step % self.config.steps_per_validation == 0):
                    print(f"\n{'='*60}")
                    print(f"VALIDATION at step {self.global_step}")
                    print(f"{'='*60}")
                    
                    val_start_time = time.time()
                    
                    # Run COCO evaluation
                    eval_start_time = time.time()
                    coco_metrics = self.evaluate_coco()
                    eval_end_time = time.time()
                    print(f"COCO evaluation took {eval_end_time - eval_start_time:.1f}s")
                    
                    # Print metrics
                    print("\nValidation Results Summary:")
                    print(f"  Detection mAP: {coco_metrics['mAP']:.4f}")
                    print(f"  Detection mAP50: {coco_metrics['mAP50']:.4f}")
                    print(f"  Detection mAP75: {coco_metrics['mAP75']:.4f}")
                    print(f"  mAP small: {coco_metrics['mAP_small']:.4f}")
                    print(f"  mAP medium: {coco_metrics['mAP_medium']:.4f}")
                    print(f"  mAP large: {coco_metrics['mAP_large']:.4f}")
                    if 'mAP_seg' in coco_metrics:
                        print(f"  Segmentation mAP: {coco_metrics['mAP_seg']:.4f}")
                        print(f"  Segmentation mAP50: {coco_metrics['mAP50_seg']:.4f}")
                        print(f"  Segmentation mAP75: {coco_metrics['mAP75_seg']:.4f}")
                    
                    # Update history
                    for key, value in coco_metrics.items():
                        if key in self.val_history:
                            self.val_history[key].append(value)
                    
                    # Save checkpoint
                    is_best = coco_metrics['mAP'] > self.best_mAP
                    if is_best:
                        self.best_mAP = coco_metrics['mAP']
                        print(f"New best mAP: {self.best_mAP:.4f}")
                    
                    self.save_checkpoint(coco_metrics, is_best)
                    
                    val_end_time = time.time()
                    print(f"\nTotal validation time: {val_end_time - val_start_time:.1f}s")
                    print(f"{'='*60}\n")
                
                self.global_step += 1
            
            # Update epoch-level history
            avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
            for key, value in avg_losses.items():
                self.train_history[key].append(value)
            
            # End of epoch summary
            epoch_time = time.time() - epoch_start_time
            avg_gpu = np.mean([g for g in epoch_losses['gpu_utilization'] if g >= 0])
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Time: {epoch_time:.1f}s")
            print(f"  Steps: {len(epoch_losses['loss'])}")
            print(f"  Avg Loss: {avg_losses['loss']:.4f}")
            print(f"  Avg GPU: {avg_gpu:.0f}%")
            print(f"  Avg Memory: {avg_losses['memory_mb']:.0f}MB")
            print(f"  Steps/sec: {len(epoch_losses['loss'])/epoch_time:.2f}")
            
            epoch += 1
        
        print("\nTraining completed!")
        print(f"Best mAP: {self.best_mAP:.4f}")
        
        # Save final model
        final_path = self.checkpoint_dir / "final.pth"
        torch.save(self.model.state_dict(), final_path)
        print(f"Saved final model to {final_path}")
    


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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
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
    
    # Load pretrained weights if specified
    if config.pretrained_backbone and config.pretrained_checkpoint_url:
        print(f"Loading pretrained weights from {config.pretrained_checkpoint_url}")
        load_pretrained_from_url(model, config.pretrained_checkpoint_url, strict=False)
    
    model = model.to(device)
    
    # Create trainer
    trainer = IterationBasedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        val_coco=val_coco,
        config=config
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
