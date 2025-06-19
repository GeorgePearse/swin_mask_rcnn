#!/usr/bin/env python3
"""Test script to verify metrics export is working correctly."""
import json
from pathlib import Path
import time

def check_metrics_file(metrics_path: Path):
    """Check what's in the metrics file."""
    if not metrics_path.exists():
        print(f"Metrics file not found: {metrics_path}")
        return
    
    with open(metrics_path, 'r') as f:
        data = json.load(f)
    
    print(f"Total entries: {len(data)}")
    
    if data:
        latest = data[-1]
        print(f"\nLatest entry (step {latest.get('step', 'N/A')}):")
        print(f"Keys: {sorted(latest.keys())}")
        
        # Check for different types of metrics
        train_metrics = [k for k in latest.keys() if k.startswith('train/')]
        val_metrics = [k for k in latest.keys() if k.startswith('val/')]
        quick_eval_metrics = [k for k in latest.keys() if k.startswith('quick_eval/')]
        
        print(f"\nTrain metrics: {len(train_metrics)}")
        if train_metrics:
            print(f"  Sample: {train_metrics[:3]}")
            
        print(f"\nValidation metrics: {len(val_metrics)}")
        if val_metrics:
            print(f"  Sample: {val_metrics[:3]}")
            # Check for class metrics
            class_metrics = [k for k in val_metrics if 'class' in k or 'top_' in k]
            if class_metrics:
                print(f"  Class metrics found: {class_metrics[:5]}")
            
        print(f"\nQuick eval metrics: {len(quick_eval_metrics)}")
        if quick_eval_metrics:
            print(f"  Sample: {quick_eval_metrics[:3]}")

def check_class_metrics_files():
    """Check for separate class metrics files."""
    class_files = list(Path('.').glob('class_metrics_epoch_*.json'))
    
    if class_files:
        print(f"\nFound {len(class_files)} class metrics files:")
        for f in sorted(class_files)[-3:]:  # Show last 3
            print(f"  {f}")
            with open(f, 'r') as fp:
                data = json.load(fp)
                print(f"    - Epoch: {data.get('epoch', 'N/A')}, Step: {data.get('step', 'N/A')}")
                if 'overall_metrics' in data:
                    om = data['overall_metrics']
                    print(f"    - mAP: {om.get('mAP', 0):.4f}, Classes with predictions: {om.get('classes_with_predictions', 0)}")
    else:
        print("\nNo class metrics files found (class_metrics_epoch_*.json)")

def main():
    # Check the main metrics file
    metrics_dir = Path('./fast_dev_checkpoints/run_20250619_163750/metrics')
    metrics_file = metrics_dir / 'metrics.json'
    
    print("=" * 60)
    print("METRICS EXPORT DIAGNOSTIC")
    print("=" * 60)
    
    check_metrics_file(metrics_file)
    check_class_metrics_files()
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("=" * 60)
    print("1. If no metrics are being exported, check that trainer.callback_metrics contains data")
    print("2. If validation hasn't run yet, wait until step 20+ (based on config)")
    print("3. The updated MetricsTracker should capture metrics from trainer.callback_metrics")
    print("4. Class metrics should appear after validation epochs complete")

if __name__ == '__main__':
    main()