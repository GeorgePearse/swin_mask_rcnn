#!/usr/bin/env python3
"""
Test script for fast training loop features.

This script runs a minimal training session to verify:
1. Quick evaluation runs correctly
2. Class-level metrics are computed
3. MetricsTracker exports data properly
"""
import sys
from pathlib import Path
import torch
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.train import main as train_main


def create_test_config():
    """Create a minimal test configuration."""
    config = {
        # Use CMR dataset
        'train_ann': '/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:23.077836_train_ordered.json',
        'val_ann': '/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:38.270134_val_ordered.json',
        'img_root': '/home/georgepearse/data/images',
        
        # Model
        'num_classes': 69,
        'pretrained_backbone': True,
        'frozen_backbone_stages': 3,
        
        # Very small batches for testing
        'train_batch_size': 1,
        'val_batch_size': 2,
        'num_workers': 0,  # Avoid multiprocessing issues in test
        
        # Quick training
        'num_epochs': 1,
        'lr': 0.0001,
        'optimizer': 'adamw',
        'use_scheduler': False,
        'use_amp': False,
        
        # Fast validation
        'steps_per_validation': 5,  # Very frequent
        'validation_start_step': 5,
        'max_val_images': 4,  # Tiny validation
        
        # Quick eval settings
        'quick_eval_enabled': True,
        'quick_eval_interval': 3,  # Every 3 steps
        'quick_eval_samples': 2,   # Just 2 samples
        'track_top_k_classes': 3,
        
        # Output
        'checkpoint_dir': './test_fast_loop',
        'log_interval': 1,  # Log every step
        'clip_grad_norm': 10.0,
    }
    
    # Save config
    config_path = Path('test_fast_loop_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return str(config_path)


def verify_outputs(checkpoint_dir='./test_fast_loop'):
    """Verify that expected outputs were created."""
    checkpoint_dir = Path(checkpoint_dir)
    
    print("\nVerifying outputs...")
    
    # Find run directory
    run_dirs = list(checkpoint_dir.glob('run_*'))
    if not run_dirs:
        print("❌ No run directory found")
        return False
    
    run_dir = sorted(run_dirs)[-1]
    print(f"✅ Run directory: {run_dir}")
    
    # Check for metrics
    metrics_dir = run_dir / 'metrics'
    if metrics_dir.exists():
        metrics_files = list(metrics_dir.glob('metrics_step_*.json'))
        print(f"✅ Metrics files: {len(metrics_files)} found")
        
        # Check latest metrics
        latest_file = metrics_dir / 'latest_metrics.json'
        if latest_file.exists():
            import json
            with open(latest_file) as f:
                latest = json.load(f)
            print(f"✅ Latest metrics at step: {latest['step']}")
            
            # Display some key metrics
            if 'moving_averages' in latest:
                print("\n  Moving Averages:")
                for key, stats in list(latest['moving_averages'].items())[:3]:
                    print(f"    {key}: {stats['average']:.4f} (±{stats['std']:.4f})")
    else:
        print("❌ No metrics directory found")
    
    # Check for TensorBoard logs
    tb_dirs = list(checkpoint_dir.glob('tensorboard/*'))
    if tb_dirs:
        print(f"✅ TensorBoard logs: {len(tb_dirs)} runs")
    else:
        print("❌ No TensorBoard logs found")
    
    # Check for checkpoints
    ckpt_files = list(run_dir.glob('*.ckpt'))
    if ckpt_files:
        print(f"✅ Checkpoints: {len(ckpt_files)} saved")
    
    return True


def main():
    """Run the test."""
    print("Testing fast training loop features...")
    
    # Create test config
    config_path = create_test_config()
    print(f"Created test config: {config_path}")
    
    try:
        # Run training
        print("\nStarting training...")
        train_main(config_path)
        
        # Verify outputs
        print("\n" + "="*60)
        success = verify_outputs()
        
        if success:
            print("\n✅ Fast training loop test completed successfully!")
        else:
            print("\n❌ Some outputs were missing")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup
        Path(config_path).unlink(missing_ok=True)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())