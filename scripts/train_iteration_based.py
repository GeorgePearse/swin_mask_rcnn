"""Training script with iteration-based validation and COCO metrics."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from typing import Dict, Any, Optional
import json
import time

from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.data.dataset import CocoDataset
from swin_maskrcnn.data.transforms_simple import get_transform_simple
from swin_maskrcnn.utils.collate import collate_fn

# Configuration
config = {
    'train_ann': '/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:23.077836_train_ordered.json',
    'val_ann': '/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:38.270134_val_ordered.json',
    'img_root': '/home/georgepearse/data/images',
    'num_classes': 69,
    'batch_size': 1,
    'num_workers': 0,
    'lr': 1e-4,
    'num_epochs': 12,
    'steps_per_validation': 5,  # Run validation every N iterations
    'weight_decay': 0.05,
    'clip_grad_norm': 10.0,
    'checkpoint_dir': './test_checkpoints',
    'log_interval': 50
}


class IterationBasedTrainer:
    """Trainer with iteration-based validation and COCO evaluation."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        val_coco,  # COCO object for validation
        num_epochs: int = 12,
        steps_per_validation: int = 500,
        learning_rate: float = 0.0001,
        weight_decay: float = 0.05,
        clip_grad_norm: float = 10.0,
        checkpoint_dir: str = "./checkpoints",
        log_interval: int = 50,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_coco = val_coco
        self.num_epochs = num_epochs
        self.steps_per_validation = steps_per_validation
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.clip_grad_norm = clip_grad_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        self.model.to(self.device)
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Initialize scheduler
        total_steps = len(train_loader) * num_epochs
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            total_steps=total_steps,
            pct_start=0.05,
            anneal_strategy='cos'
        )
        
        # Training history
        self.train_history = {
            'loss': [],
            'rpn_cls_loss': [],
            'rpn_bbox_loss': [],
            'roi_cls_loss': [],
            'roi_bbox_loss': [],
            'roi_mask_loss': []
        }
        
        self.val_history = {
            'loss': [],
            'mAP': [],
            'mAP50': [],
            'mAP75': [],
            'mAP_small': [],
            'mAP_medium': [],
            'mAP_large': []
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
        total_loss = sum(loss_dict.values())
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        if self.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Convert losses to dict
        loss_values = {k: v.item() for k, v in loss_dict.items()}
        loss_values['total'] = total_loss.item()
        
        return loss_values
    
    @torch.no_grad()
    def evaluate_coco(self) -> Dict[str, float]:
        """Evaluate model using COCO metrics."""
        self.model.eval()
        
        predictions = []
        
        pbar = tqdm(self.val_loader, desc="Evaluating")
        for images, targets in pbar:
            # Move to device
            images = [img.to(self.device) for img in images]
            
            # Get predictions
            outputs = self.model(images)
            
            # Process each image
            for i, output in enumerate(outputs):
                # Get original image ID from target
                image_id = targets[i]['image_id'].item()
                
                # Convert predictions to COCO format
                for box, label, score, mask in zip(
                    output['boxes'].cpu().numpy(),
                    output['labels'].cpu().numpy(),
                    output['scores'].cpu().numpy(),
                    output['masks'].cpu().numpy()
                ):
                    # Only keep predictions with score > 0.05
                    if score < 0.05:
                        continue
                    
                    # Convert mask to RLE format
                    mask_binary = (mask[0] > 0.5).astype(np.uint8)
                    
                    # Get bounding box in COCO format [x, y, width, height]
                    x1, y1, x2, y2 = box
                    bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                    
                    predictions.append({
                        'image_id': int(image_id),
                        'category_id': int(label),
                        'bbox': bbox,
                        'score': float(score),
                        'segmentation': mask_binary.tolist()  # Simplified format
                    })
        
        # If no predictions, return zeros
        if not predictions:
            return {
                'mAP': 0.0,
                'mAP50': 0.0,
                'mAP75': 0.0,
                'mAP_small': 0.0,
                'mAP_medium': 0.0,
                'mAP_large': 0.0
            }
        
        # Save predictions for evaluation
        pred_file = self.checkpoint_dir / f'predictions_step_{self.global_step}.json'
        with open(pred_file, 'w') as f:
            json.dump(predictions, f)
        
        # Load predictions as COCO result
        coco_dt = self.val_coco.loadRes(str(pred_file))
        
        # Run COCO evaluation
        coco_eval = COCOeval(self.val_coco, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
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
            coco_eval_seg = COCOeval(self.val_coco, coco_dt, 'segm')
            coco_eval_seg.evaluate()
            coco_eval_seg.accumulate()
            coco_eval_seg.summarize()
            
            metrics.update({
                'mAP_seg': coco_eval_seg.stats[0],
                'mAP50_seg': coco_eval_seg.stats[1],
                'mAP75_seg': coco_eval_seg.stats[2],
            })
        except:
            # If segmentation evaluation fails, just skip it
            pass
        
        return metrics
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_mAP': self.best_mAP,
            'metrics': metrics
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{self.global_step}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best model with mAP: {self.best_mAP:.4f}")
    
    def train(self):
        """Run the complete training loop."""
        print(f"Starting training on device: {self.device}")
        print(f"Number of training samples: {len(self.train_loader.dataset)}")
        print(f"Number of validation samples: {len(self.val_loader.dataset)}")
        print(f"Running validation every {self.steps_per_validation} steps")
        
        epoch = 0
        
        while epoch < self.num_epochs:
            epoch_losses = {
                'loss': [],
                'rpn_cls_loss': [],
                'rpn_bbox_loss': [],
                'roi_cls_loss': [],
                'roi_bbox_loss': [],
                'roi_mask_loss': []
            }
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            
            for batch_idx, (images, targets) in enumerate(pbar):
                # Training step
                loss_values = self.train_step(images, targets)
                
                # Record losses
                for key in epoch_losses:
                    if key == 'loss':
                        epoch_losses[key].append(loss_values['total'])
                    else:
                        epoch_losses[key].append(loss_values.get(key, 0.0))
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss_values["total"]:.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
                })
                
                # Log every N steps
                if self.global_step % self.log_interval == 0:
                    avg_loss = np.mean(epoch_losses['loss'][-self.log_interval:])
                    print(f"\nStep {self.global_step}, Loss: {avg_loss:.4f}")
                
                # Run validation every N steps
                if self.global_step > 0 and self.global_step % self.steps_per_validation == 0:
                    print(f"\nRunning validation at step {self.global_step}")
                    
                    # Calculate validation loss
                    val_loss = self.validate_loss()
                    
                    # Run COCO evaluation
                    coco_metrics = self.evaluate_coco()
                    
                    # Combine metrics
                    all_metrics = {'val_loss': val_loss}
                    all_metrics.update(coco_metrics)
                    
                    # Print metrics
                    print(f"Validation Results at Step {self.global_step}:")
                    print(f"  Val Loss: {val_loss:.4f}")
                    print(f"  mAP: {coco_metrics['mAP']:.4f}")
                    print(f"  mAP50: {coco_metrics['mAP50']:.4f}")
                    print(f"  mAP75: {coco_metrics['mAP75']:.4f}")
                    
                    # Update history
                    self.val_history['loss'].append(val_loss)
                    for key, value in coco_metrics.items():
                        if key in self.val_history:
                            self.val_history[key].append(value)
                    
                    # Save checkpoint
                    is_best = coco_metrics['mAP'] > self.best_mAP
                    if is_best:
                        self.best_mAP = coco_metrics['mAP']
                    
                    self.save_checkpoint(all_metrics, is_best)
                
                self.global_step += 1
            
            # Update epoch-level history
            avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
            for key, value in avg_losses.items():
                self.train_history[key].append(value)
            
            epoch += 1
        
        print("\nTraining completed!")
        print(f"Best mAP: {self.best_mAP:.4f}")
        
        # Save final model
        final_path = self.checkpoint_dir / "final.pth"
        torch.save(self.model.state_dict(), final_path)
        print(f"Saved final model to {final_path}")
    
    @torch.no_grad()
    def validate_loss(self) -> float:
        """Calculate validation loss."""
        self.model.train()  # Keep in train mode to get losses
        val_losses = []
        
        for images, targets in self.val_loader:
            # Move to device
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            with torch.amp.autocast('cuda', enabled=False):  # Disable mixed precision for validation
                loss_dict = self.model(images, targets)
                total_loss = sum(loss_dict.values())
            
            val_losses.append(total_loss.item())
        
        return np.mean(val_losses)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = CocoDataset(
        root_dir=config['img_root'],
        annotation_file=config['train_ann'],
        transforms=get_transform_simple(train=True),
        mode='train'
    )
    
    val_dataset = CocoDataset(
        root_dir=config['img_root'],
        annotation_file=config['val_ann'],
        transforms=get_transform_simple(train=False),
        mode='train'  # Still use train mode to get targets
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn
    )
    
    # Create COCO object for validation
    val_coco = COCO(config['val_ann'])
    
    # Create model
    model = SwinMaskRCNN(num_classes=config['num_classes'])
    model = model.to(device)
    
    # Create trainer
    trainer = IterationBasedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        val_coco=val_coco,
        num_epochs=config['num_epochs'],
        steps_per_validation=config['steps_per_validation'],
        learning_rate=config['lr'],
        weight_decay=config['weight_decay'],
        clip_grad_norm=config['clip_grad_norm'],
        checkpoint_dir=config['checkpoint_dir'],
        log_interval=config['log_interval']
    )
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()