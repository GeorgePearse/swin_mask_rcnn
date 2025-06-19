"""Real-time metrics tracking callback for PyTorch Lightning."""
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import numpy as np
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
from collections import defaultdict, deque


class MetricsTracker(Callback):
    """
    Track and display real-time metrics during training for quick feedback.
    
    Features:
    - Running averages of losses
    - Per-class prediction tracking
    - Performance alerts when metrics drop
    - Export metrics to JSON for external monitoring
    """
    
    def __init__(
        self,
        window_size: int = 100,
        export_interval: int = 50,
        export_dir: Optional[Path] = None,
        alert_threshold: float = 0.2,
        track_classes: bool = True,
        top_k_classes: int = 10
    ):
        """
        Initialize the metrics tracker.
        
        Args:
            window_size: Size of the moving average window
            export_interval: Steps between metric exports
            export_dir: Directory to export metrics (creates if not exists)
            alert_threshold: Relative drop threshold for performance alerts
            track_classes: Whether to track per-class metrics
            top_k_classes: Number of top/bottom classes to track
        """
        self.window_size = window_size
        self.export_interval = export_interval
        self.export_dir = Path(export_dir) if export_dir else Path("./metrics_exports")
        self.export_dir.mkdir(exist_ok=True)
        self.alert_threshold = alert_threshold
        self.track_classes = track_classes
        self.top_k_classes = top_k_classes
        
        # Metrics storage
        self.loss_windows = defaultdict(lambda: deque(maxlen=window_size))
        self.metric_history = defaultdict(list)
        self.class_predictions = defaultdict(lambda: defaultdict(int))
        self.last_export_step = 0
        self.start_time = time.time()
        
        # Performance tracking
        self.best_metrics = {}
        self.performance_alerts = []
    
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int
    ):
        """Track metrics after each training batch."""
        # Skip if outputs is None (error batch)
        if outputs is None:
            return
            
        # Get logged metrics
        metrics = trainer.callback_metrics
        
        # Track losses with moving averages
        for key, value in metrics.items():
            if 'loss' in key and isinstance(value, (int, float)):
                self.loss_windows[key].append(float(value))
                
                # Calculate moving average
                if len(self.loss_windows[key]) > 0:
                    moving_avg = np.mean(list(self.loss_windows[key]))
                    self.metric_history[f"{key}_ma"].append(moving_avg)
                    
                    # Check for performance alerts
                    if key in self.best_metrics:
                        if moving_avg > self.best_metrics[key] * (1 + self.alert_threshold):
                            alert = {
                                'step': trainer.global_step,
                                'metric': key,
                                'current': moving_avg,
                                'best': self.best_metrics[key],
                                'increase_pct': (moving_avg / self.best_metrics[key] - 1) * 100
                            }
                            self.performance_alerts.append(alert)
                            print(f"\n⚠️  Performance Alert: {key} increased by {alert['increase_pct']:.1f}%")
                    else:
                        self.best_metrics[key] = moving_avg
                    
                    # Update best metric if improved
                    if moving_avg < self.best_metrics.get(key, float('inf')):
                        self.best_metrics[key] = moving_avg
        
        # Track prediction statistics
        if 'train/avg_predictions_per_image' in metrics:
            pred_avg = float(metrics['train/avg_predictions_per_image'])
            self.metric_history['prediction_rate'].append(pred_avg)
            
        # Export metrics periodically
        if trainer.global_step - self.last_export_step >= self.export_interval:
            self._export_metrics(trainer.global_step)
            self.last_export_step = trainer.global_step
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Track validation metrics and class-level performance."""
        metrics = trainer.callback_metrics
        
        # Track overall validation metrics
        val_metrics = {
            'step': trainer.global_step,
            'epoch': trainer.current_epoch,
            'time_elapsed': time.time() - self.start_time
        }
        
        for key, value in metrics.items():
            if key.startswith('val/') and isinstance(value, (int, float)):
                val_metrics[key] = float(value)
        
        self.metric_history['validation_checkpoints'].append(val_metrics)
        
        # Display quick summary
        if 'val/mAP' in metrics:
            print(f"\n📊 Quick Validation Summary (Step {trainer.global_step}):")
            print(f"   mAP: {metrics.get('val/mAP', 0):.4f}")
            print(f"   mAP50: {metrics.get('val/mAP50', 0):.4f}")
            print(f"   Top 10 Classes mAP: {metrics.get('val/top_10_classes_mAP', 0):.4f}")
            print(f"   Bottom 10 Classes mAP: {metrics.get('val/bottom_10_classes_mAP', 0):.4f}")
            print(f"   AP Spread: {metrics.get('val/ap_spread', 0):.4f}")
            
            # Show time efficiency
            time_per_epoch = (time.time() - self.start_time) / max(1, trainer.current_epoch)
            print(f"   Time per epoch: {time_per_epoch:.1f}s")
            print(f"   Estimated time to completion: {time_per_epoch * (trainer.max_epochs - trainer.current_epoch):.1f}s")
    
    def _export_metrics(self, step: int):
        """Export current metrics to JSON for external monitoring."""
        export_data = {
            'step': step,
            'timestamp': time.time(),
            'moving_averages': {},
            'best_metrics': self.best_metrics,
            'recent_alerts': self.performance_alerts[-10:],  # Last 10 alerts
            'training_speed': {
                'steps_per_second': step / (time.time() - self.start_time),
                'time_elapsed': time.time() - self.start_time
            }
        }
        
        # Add moving averages
        for key, window in self.loss_windows.items():
            if len(window) > 0:
                export_data['moving_averages'][key] = {
                    'current': float(list(window)[-1]),
                    'average': float(np.mean(list(window))),
                    'std': float(np.std(list(window))),
                    'min': float(np.min(list(window))),
                    'max': float(np.max(list(window)))
                }
        
        # Save to file
        export_file = self.export_dir / f"metrics_step_{step}.json"
        with open(export_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        # Also save a "latest" file for easy access
        latest_file = self.export_dir / "latest_metrics.json"
        with open(latest_file, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Save final metrics summary."""
        summary = {
            'total_steps': trainer.global_step,
            'total_time': time.time() - self.start_time,
            'best_metrics': self.best_metrics,
            'total_alerts': len(self.performance_alerts),
            'metric_history_lengths': {k: len(v) for k, v in self.metric_history.items()},
            'final_moving_averages': {}
        }
        
        # Add final moving averages
        for key, window in self.loss_windows.items():
            if len(window) > 0:
                summary['final_moving_averages'][key] = float(np.mean(list(window)))
        
        # Save summary
        summary_file = self.export_dir / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Training completed! Metrics exported to: {self.export_dir}")
        print(f"   Total time: {summary['total_time']:.1f}s")
        print(f"   Total steps: {summary['total_steps']}")
        print(f"   Performance alerts: {summary['total_alerts']}")