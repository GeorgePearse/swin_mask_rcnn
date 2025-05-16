"""
Training loop for SWIN-based Mask R-CNN.
"""
import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any


class MaskRCNNTrainer:
    """Trainer class for SWIN-based Mask R-CNN."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        num_epochs: int = 12,
        learning_rate: float = 0.0001,
        weight_decay: float = 0.05,
        clip_grad_norm: float = 10.0,
        checkpoint_dir: str = "./checkpoints",
        log_interval: int = 50,
        device: Optional[torch.device] = None
    ):
        """Initialize the trainer.
        
        Args:
            model: The Mask R-CNN model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of training epochs
            learning_rate: Initial learning rate
            weight_decay: Weight decay for AdamW optimizer
            clip_grad_norm: Maximum gradient norm for clipping
            checkpoint_dir: Directory to save checkpoints
            log_interval: Steps between logging
            device: Training device
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
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
            'loss': []
        }
        
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dict containing average losses for the epoch
        """
        self.model.train()
        
        epoch_losses = {
            'loss': [],
            'rpn_cls_loss': [],
            'rpn_bbox_loss': [],
            'roi_cls_loss': [],
            'roi_bbox_loss': [],
            'roi_mask_loss': []
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            # Move to device
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = self.model(images, targets)
            
            # Calculate total loss
            total_loss = sum(loss_dict.values())
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Record losses
            for key, value in loss_dict.items():
                epoch_losses[key.replace('rpn_', 'rpn_').replace('roi_', 'roi_')].append(value.item())
            epoch_losses['loss'].append(total_loss.item())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Log every N steps
            if self.global_step % self.log_interval == 0:
                avg_loss = np.mean(epoch_losses['loss'][-self.log_interval:])
                print(f"Step {self.global_step}, Loss: {avg_loss:.4f}")
            
            self.global_step += 1
        
        # Calculate epoch averages
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        
        # Update history
        for key, value in avg_losses.items():
            self.train_history[key].append(value)
        
        return avg_losses
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dict containing validation metrics
        """
        print(f"\nStarting validation for epoch {epoch+1}...")
        self.model.eval()
        
        val_losses = []
        detailed_losses = defaultdict(list)
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Validation]")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            # Move to device
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Forward pass (for validation, we still compute losses)
            loss_dict = self.model(images, targets)
            total_loss = sum(loss_dict.values())
            
            val_losses.append(total_loss.item())
            
            # Track individual loss components
            for key, value in loss_dict.items():
                detailed_losses[key].append(value.item())
            
            pbar.set_postfix({
                'batch': f'{batch_idx+1}',
                'loss': f'{total_loss.item():.4f}'
            })
        
        # Calculate averages
        avg_val_loss = np.mean(val_losses)
        self.val_history['loss'].append(avg_val_loss)
        
        # Print detailed loss breakdown
        print(f"\nValidation results for epoch {epoch+1}:")
        print(f"  Average loss: {avg_val_loss:.4f}")
        print(f"  Loss breakdown:")
        for key, values in detailed_losses.items():
            avg_loss = np.mean(values)
            print(f"    {key}: {avg_loss:.4f}")
        
        return {'loss': avg_val_loss}
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best model with val_loss: {self.best_val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_history = checkpoint['train_history']
        self.val_history = checkpoint['val_history']
        self.best_val_loss = checkpoint['best_val_loss']
        self.global_step = checkpoint['global_step']
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        return checkpoint['epoch']
    
    def train(self):
        """Run the complete training loop."""
        print(f"Starting training on device: {self.device}")
        print(f"Number of training samples: {len(self.train_loader.dataset)}")
        print(f"Number of validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(self.num_epochs):
            # Train for one epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Print epoch summary
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch+1}/{self.num_epochs} SUMMARY")
            print(f"{'='*60}")
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.2e}")
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                print(f">>> NEW BEST VALIDATION LOSS! <<<")
            
            self.save_checkpoint(epoch, is_best)
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch)
        
        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Save final model
        final_path = self.checkpoint_dir / "final.pth"
        torch.save(self.model.state_dict(), final_path)
        print(f"Saved final model to {final_path}")
    
    def plot_history(self, save_path: Optional[str] = None):
        """Plot training history.
        
        Args:
            save_path: Optional path to save the plot
        """
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot total loss
        ax1.plot(self.train_history['loss'], label='Train')
        ax1.plot(self.val_history['loss'], label='Val')
        ax1.set_title('Total Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot individual losses
        ax2.plot(self.train_history['rpn_cls_loss'], label='RPN Cls')
        ax2.plot(self.train_history['rpn_bbox_loss'], label='RPN BBox')
        ax2.plot(self.train_history['roi_cls_loss'], label='ROI Cls')
        ax2.plot(self.train_history['roi_bbox_loss'], label='ROI BBox')
        ax2.plot(self.train_history['roi_mask_loss'], label='ROI Mask')
        ax2.set_title('Individual Losses')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


def train_mask_rcnn(
    model,
    train_loader,
    val_loader,
    config: Dict[str, Any]
):
    """Convenience function to train Mask R-CNN.
    
    Args:
        model: The Mask R-CNN model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dictionary
    """
    trainer = MaskRCNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        **config
    )
    
    trainer.train()
    return trainer