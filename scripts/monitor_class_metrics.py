#!/usr/bin/env python3
"""Monitor per-class metrics during Swin Mask R-CNN training."""
import json
import argparse
from pathlib import Path
from datetime import datetime
import time
import os
from typing import Dict, List, Optional
import glob


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def load_latest_class_metrics(pattern: str = "class_metrics_*.json") -> Optional[Dict]:
    """Load the most recent class metrics file."""
    files = glob.glob(pattern)
    if not files:
        return None
    
    # Sort by modification time and get the latest
    latest_file = max(files, key=os.path.getmtime)
    
    try:
        with open(latest_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {latest_file}: {e}")
        return None


def display_class_metrics(refresh_interval: int = 10):
    """Display class metrics with live updates."""
    last_step = -1
    
    while True:
        try:
            # Load latest metrics
            metrics = load_latest_class_metrics()
            
            if not metrics:
                print("Waiting for class metrics file...")
                time.sleep(refresh_interval)
                continue
            
            # Check if this is new data
            current_step = metrics.get('step', 0)
            if current_step == last_step:
                time.sleep(refresh_interval)
                continue
            
            last_step = current_step
            
            # Clear screen and display header
            clear_screen()
            
            print("=" * 100)
            print(f"SWIN MASK R-CNN - PER-CLASS METRICS MONITOR".center(100))
            print(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(100))
            print("=" * 100)
            
            # Display overall metrics
            overall = metrics.get('overall_metrics', {})
            print(f"\nOverall Performance (Epoch {metrics.get('epoch', 0)}, Step {metrics.get('step', 0)}):")
            print(f"  mAP: {overall.get('mAP', 0):.4f}")
            print(f"  mAP50: {overall.get('mAP50', 0):.4f}")
            print(f"  Classes with predictions: {overall.get('classes_with_predictions', 0)}/{overall.get('total_classes', 0)}")
            print(f"  Classes with AP > 0: {overall.get('classes_with_ap', 0)}/{overall.get('total_classes', 0)}")
            
            # Display top performing classes
            sorted_classes = metrics.get('sorted_by_ap', [])
            if sorted_classes:
                print(f"\n{'TOP 15 PERFORMING CLASSES':^100}")
                print(f"{'Rank':<6} {'Class Name':<30} {'AP':>8} {'AP50':>8} {'Preds':>8} {'GT':>8} {'Recall':>8} {'Status':<15}")
                print("-" * 100)
                
                for cls in sorted_classes[:15]:
                    status = ""
                    if cls['ap'] > 0.5:
                        status = "🟢 Excellent"
                    elif cls['ap'] > 0.3:
                        status = "🟡 Good"
                    elif cls['ap'] > 0.1:
                        status = "🟠 Fair"
                    elif cls['ap'] > 0:
                        status = "🔴 Poor"
                    else:
                        status = "⚫ No AP"
                    
                    print(f"{cls['rank']:<6} {cls['name']:<30} {cls['ap']:>8.4f} {cls['ap50']:>8.4f} "
                          f"{cls['num_predictions']:>8} {cls['num_gt']:>8} {cls['recall']:>8.2f} {status:<15}")
                
                # Display bottom performing classes with GT > 0
                bottom_classes = [c for c in sorted_classes if c['num_gt'] > 0][-15:]
                if bottom_classes and len(bottom_classes) > 5:
                    print(f"\n{'BOTTOM 15 CLASSES (WITH GT ANNOTATIONS)':^100}")
                    print(f"{'Rank':<6} {'Class Name':<30} {'AP':>8} {'AP50':>8} {'Preds':>8} {'GT':>8} {'Recall':>8} {'Status':<15}")
                    print("-" * 100)
                    
                    for cls in bottom_classes:
                        status = "🆘 Needs work" if cls['ap'] == 0 else "🔴 Poor"
                        print(f"{cls['rank']:<6} {cls['name']:<30} {cls['ap']:>8.4f} {cls['ap50']:>8.4f} "
                              f"{cls['num_predictions']:>8} {cls['num_gt']:>8} {cls['recall']:>8.2f} {status:<15}")
                
                # Summary statistics
                print("\n" + "-" * 100)
                print("Summary:")
                zero_ap_classes = [c for c in sorted_classes if c['ap'] == 0 and c['num_gt'] > 0]
                if zero_ap_classes:
                    print(f"  Classes with 0 AP (but have GT): {len(zero_ap_classes)}")
                    print(f"    Examples: {', '.join([c['name'] for c in zero_ap_classes[:5]])}")
                
                no_pred_classes = [c for c in sorted_classes if c['num_predictions'] == 0 and c['num_gt'] > 0]
                if no_pred_classes:
                    print(f"  Classes with no predictions (but have GT): {len(no_pred_classes)}")
                    print(f"    Examples: {', '.join([c['name'] for c in no_pred_classes[:5]])}")
            
            # Display footer
            print("\n" + "-" * 100)
            print(f"Refreshing every {refresh_interval}s | Press Ctrl+C to exit")
            print("Hint: Class metrics are updated after each validation epoch")
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
        
        time.sleep(refresh_interval)


def main():
    parser = argparse.ArgumentParser(description="Monitor per-class metrics for Swin Mask R-CNN training")
    parser.add_argument('--refresh', type=int, default=10,
                       help='Refresh interval in seconds (default: 10)')
    
    args = parser.parse_args()
    
    print("Starting class metrics monitor...")
    print("Looking for class_metrics_*.json files in current directory")
    display_class_metrics(args.refresh)


if __name__ == '__main__':
    main()