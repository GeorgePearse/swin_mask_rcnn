#!/usr/bin/env python3
"""Real-time training monitor for Swin Mask R-CNN."""
import json
import argparse
from pathlib import Path
from datetime import datetime
import time
import os
import sys
from typing import Dict, List, Optional


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def format_time_elapsed(seconds: int) -> str:
    """Format elapsed time as HH:MM:SS."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_eta(step: int, total_steps: int, elapsed: int) -> str:
    """Calculate and format ETA."""
    if step == 0:
        return "N/A"
    
    steps_per_second = step / elapsed
    remaining_steps = total_steps - step
    eta_seconds = int(remaining_steps / steps_per_second)
    
    return format_time_elapsed(eta_seconds)


def get_trend_symbol(trend: float) -> str:
    """Get symbol for trend direction."""
    if abs(trend) < 1e-6:
        return "→"
    elif trend > 0:
        return "↑"
    else:
        return "↓"


def display_metrics(metrics_path: Path, refresh_interval: int = 5):
    """Display metrics with live updates."""
    start_time = time.time()
    
    while True:
        try:
            # Load metrics
            if not metrics_path.exists():
                print(f"Waiting for metrics file: {metrics_path}")
                time.sleep(refresh_interval)
                continue
            
            with open(metrics_path, 'r') as f:
                metrics_history = json.load(f)
            
            if not metrics_history:
                print("No metrics available yet...")
                time.sleep(refresh_interval)
                continue
            
            # Get latest metrics
            latest = metrics_history[-1]
            
            # Clear screen and display header
            clear_screen()
            elapsed = int(time.time() - start_time)
            
            print("=" * 80)
            print(f"SWIN MASK R-CNN TRAINING MONITOR".center(80))
            print(f"Elapsed: {format_time_elapsed(elapsed)} | Updated: {datetime.now().strftime('%H:%M:%S')}".center(80))
            print("=" * 80)
            
            # Display basic info
            step = latest.get('step', 0)
            epoch = latest.get('epoch', 0)
            
            print(f"\nTraining Progress:")
            print(f"  Step: {step:,}")
            print(f"  Epoch: {epoch}")
            
            # Display losses with trends
            print(f"\nLosses (with trends):")
            loss_keys = ['train/loss', 'train/rpn_cls_loss', 'train/rpn_bbox_loss', 
                        'train/roi_cls_loss', 'train/roi_bbox_loss', 'train/roi_mask_loss']
            
            for key in loss_keys:
                if f'{key}_avg' in latest:
                    avg = latest[f'{key}_avg']
                    trend = latest.get(f'{key}_trend', 0)
                    symbol = get_trend_symbol(trend)
                    
                    name = key.replace('train/', '').replace('_', ' ').title()
                    print(f"  {name:<20}: {avg:>8.4f} {symbol} ({trend:+.4f})")
            
            # Display detection metrics
            print(f"\nDetection Metrics:")
            if 'train/total_predictions' in latest:
                total_preds = latest.get('train/total_predictions', 0)
                avg_preds = latest.get('train/avg_predictions_per_image', 0)
                total_anns = latest.get('train/total_annotations', 0)
                avg_anns = latest.get('train/avg_annotations_per_image', 0)
                
                print(f"  Predictions per batch: {total_preds}")
                print(f"  Avg predictions/image: {avg_preds:.1f}")
                print(f"  Annotations per batch: {total_anns}")
                print(f"  Avg annotations/image: {avg_anns:.1f}")
            
            # Display quick evaluation results if available
            quick_eval_keys = [k for k in latest.keys() if 'quick_eval/' in k]
            if quick_eval_keys:
                print(f"\nQuick Evaluation (last at step {step}):")
                
                total_quick_preds = latest.get('quick_eval/total_predictions', 0)
                cats_detected = latest.get('quick_eval/categories_detected', 0)
                
                print(f"  Total predictions: {total_quick_preds}")
                print(f"  Categories detected: {cats_detected}")
                
                # Show top categories
                print(f"\n  Top Performing Categories:")
                for i in range(5):
                    count_key = f'quick_eval/top_{i}_cat_*_count'
                    score_key = f'quick_eval/top_{i}_cat_*_avg_score'
                    
                    # Find matching keys
                    count_keys = [k for k in latest.keys() if k.startswith(f'quick_eval/top_{i}_cat_') and k.endswith('_count')]
                    score_keys = [k for k in latest.keys() if k.startswith(f'quick_eval/top_{i}_cat_') and k.endswith('_avg_score')]
                    
                    if count_keys and score_keys:
                        cat_id = count_keys[0].split('_cat_')[1].split('_count')[0]
                        count = latest.get(count_keys[0], 0)
                        score = latest.get(score_keys[0], 0)
                        print(f"    #{i+1} Category {cat_id}: {count} predictions, avg score {score:.3f}")
            
            # Display validation metrics if available
            val_keys = [k for k in latest.keys() if 'val/' in k]
            if val_keys:
                print(f"\nValidation Metrics:")
                if 'val/mAP' in latest:
                    print(f"  mAP: {latest['val/mAP']:.4f}")
                if 'val/mAP50' in latest:
                    print(f"  mAP50: {latest['val/mAP50']:.4f}")
                if 'val/top_10_classes_mAP' in latest:
                    print(f"  Top 10 classes mAP: {latest['val/top_10_classes_mAP']:.4f}")
                if 'val/bottom_10_classes_mAP' in latest:
                    print(f"  Bottom 10 classes mAP: {latest['val/bottom_10_classes_mAP']:.4f}")
                if 'val/ap_spread' in latest:
                    print(f"  AP Spread: {latest['val/ap_spread']:.4f}")
                
                # Display per-class metrics if available
                class_ap_keys = [k for k in latest.keys() if k.startswith('val/top_') and '_ap' in k and not k.endswith('_ap50')]
                if class_ap_keys:
                    print(f"\n  Top Performing Classes:")
                    # Sort and display top classes
                    for i in range(min(5, len(class_ap_keys))):
                        key = f'val/top_{i}_'
                        matching_keys = [k for k in latest.keys() if k.startswith(key) and k.endswith('_ap')]
                        if matching_keys:
                            full_key = matching_keys[0]
                            # Extract class name from key
                            class_name = full_key.replace(f'val/top_{i}_', '').replace('_ap', '')
                            ap = latest.get(full_key, 0)
                            ap50_key = full_key.replace('_ap', '_ap50')
                            ap50 = latest.get(ap50_key, 0)
                            print(f"    #{i+1} {class_name}: AP={ap:.3f}, AP50={ap50:.3f}")
            
            # Display system metrics
            print(f"\nSystem Metrics:")
            if 'train/memory_mb' in latest:
                memory_mb = latest['train/memory_mb']
                memory_gb = memory_mb / 1024
                print(f"  GPU Memory: {memory_gb:.2f} GB ({memory_mb:.0f} MB)")
            if 'train/gpu_utilization' in latest:
                gpu_util = latest['train/gpu_utilization']
                if gpu_util >= 0:
                    print(f"  GPU Utilization: {gpu_util:.1f}%")
            
            # Display footer with hints
            print("\n" + "-" * 80)
            print(f"Refreshing every {refresh_interval}s | Press Ctrl+C to exit")
            
        except json.JSONDecodeError:
            print("Error reading metrics file, retrying...")
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
        
        time.sleep(refresh_interval)


def main():
    parser = argparse.ArgumentParser(description="Monitor Swin Mask R-CNN training in real-time")
    parser.add_argument('--metrics-dir', type=str, required=True,
                       help='Path to metrics directory (e.g., ./checkpoints/run_*/metrics)')
    parser.add_argument('--refresh', type=int, default=5,
                       help='Refresh interval in seconds (default: 5)')
    
    args = parser.parse_args()
    
    # Find metrics.json file
    metrics_dir = Path(args.metrics_dir)
    
    # Handle wildcards in path
    if '*' in str(metrics_dir):
        # Find matching directories
        parent = metrics_dir.parent
        pattern = metrics_dir.name
        
        matching_dirs = list(parent.glob(pattern))
        if not matching_dirs:
            print(f"No directories matching pattern: {metrics_dir}")
            sys.exit(1)
        
        # Use the most recent one
        metrics_dir = sorted(matching_dirs, key=lambda p: p.stat().st_mtime)[-1]
    
    metrics_path = metrics_dir / 'metrics.json'
    
    print(f"Monitoring metrics from: {metrics_path}")
    display_metrics(metrics_path, args.refresh)


if __name__ == '__main__':
    main()