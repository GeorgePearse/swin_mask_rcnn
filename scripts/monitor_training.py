#!/usr/bin/env python3
"""
Real-time training monitor for SWIN Mask R-CNN.

This script monitors the metrics exported by MetricsTracker callback
and displays them in a terminal-friendly format.

Usage:
    python scripts/monitor_training.py --metrics-dir ./fast_dev_checkpoints/run_*/metrics
"""
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import sys
import os


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def format_time(seconds):
    """Format seconds into human-readable time."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def load_latest_metrics(metrics_dir):
    """Load the latest metrics from the export directory."""
    latest_file = metrics_dir / "latest_metrics.json"
    if latest_file.exists():
        with open(latest_file) as f:
            return json.load(f)
    return None


def display_metrics(metrics, prev_metrics=None):
    """Display metrics in a formatted way."""
    clear_screen()
    
    print("=" * 80)
    print(f"{'SWIN Mask R-CNN Training Monitor':^80}")
    print("=" * 80)
    print(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Step: {metrics['step']} | Time Elapsed: {format_time(metrics['training_speed']['time_elapsed'])}")
    print(f"Training Speed: {metrics['training_speed']['steps_per_second']:.2f} steps/sec")
    print()
    
    # Display moving averages
    print("LOSS METRICS (Moving Averages)")
    print("-" * 80)
    print(f"{'Metric':<30} {'Current':>12} {'Average':>12} {'Std Dev':>12} {'Trend':>12}")
    print("-" * 80)
    
    for key in sorted(metrics['moving_averages'].keys()):
        stats = metrics['moving_averages'][key]
        current = stats['current']
        average = stats['average']
        std = stats['std']
        
        # Calculate trend if we have previous metrics
        trend = ""
        if prev_metrics and key in prev_metrics.get('moving_averages', {}):
            prev_avg = prev_metrics['moving_averages'][key]['average']
            if prev_avg > 0:
                change = (average - prev_avg) / prev_avg * 100
                if abs(change) > 0.1:  # Only show significant changes
                    trend = f"{change:+.1f}%"
        
        print(f"{key:<30} {current:>12.4f} {average:>12.4f} {std:>12.4f} {trend:>12}")
    
    # Display best metrics
    print("\nBEST METRICS")
    print("-" * 80)
    for key, value in sorted(metrics['best_metrics'].items()):
        print(f"{key:<30} {value:>12.4f}")
    
    # Display recent alerts
    if metrics['recent_alerts']:
        print("\nRECENT PERFORMANCE ALERTS")
        print("-" * 80)
        for alert in metrics['recent_alerts'][-5:]:  # Show last 5 alerts
            print(f"Step {alert['step']}: {alert['metric']} increased by {alert['increase_pct']:.1f}%")
    
    print("\n" + "=" * 80)
    print("Press Ctrl+C to exit")


def monitor_loop(metrics_dir, refresh_interval=5):
    """Main monitoring loop."""
    prev_metrics = None
    
    try:
        while True:
            metrics = load_latest_metrics(metrics_dir)
            
            if metrics:
                display_metrics(metrics, prev_metrics)
                prev_metrics = metrics
            else:
                print(f"Waiting for metrics in {metrics_dir}...")
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Monitor SWIN Mask R-CNN training")
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        required=True,
        help="Path to metrics export directory"
    )
    parser.add_argument(
        "--refresh",
        type=int,
        default=5,
        help="Refresh interval in seconds (default: 5)"
    )
    
    args = parser.parse_args()
    
    if not args.metrics_dir.exists():
        print(f"Error: Metrics directory {args.metrics_dir} does not exist")
        sys.exit(1)
    
    print(f"Starting monitor for: {args.metrics_dir}")
    monitor_loop(args.metrics_dir, args.refresh)


if __name__ == "__main__":
    main()