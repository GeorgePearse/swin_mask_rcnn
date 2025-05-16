#!/usr/bin/env python3
"""Standalone demo of the logging functionality."""
import logging
import sys
from datetime import datetime
from pathlib import Path

# This is a simplified version of setup_logger for demonstration
def demo_setup_logger(name="demo", log_dir="./demo_logs", level="INFO"):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    if logger.handlers:
        return logger
    
    format_string = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    info_file = log_dir / f"train_{timestamp}.log"
    info_handler = logging.FileHandler(info_file)
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)
    logger.addHandler(info_handler)
    
    return logger


def main():
    logger = demo_setup_logger()
    
    logger.info("=== Starting Training Session ===")
    logger.info("Loading configuration...")
    logger.info("Model: SWIN Mask R-CNN")
    logger.info("Dataset: CMR (69 classes)")
    logger.info("Device: cuda")
    
    # Simulate training progress
    logger.info("Beginning training loop...")
    for epoch in range(1, 4):
        logger.info(f"Epoch {epoch}/3")
        for step in range(1, 6):
            loss = 10.0 - (epoch - 1) * 2.0 - step * 0.1
            logger.info(f"  Step {step}/5, Loss: {loss:.4f}")
        
        # Validation
        logger.info(f"  Running validation...")
        mAP = 0.1 * epoch
        logger.info(f"  Validation mAP: {mAP:.4f}")
        
        if epoch == 2:
            logger.warning("  GPU memory usage high: 95%")
    
    # Simulate an error
    try:
        result = 1 / 0
    except ZeroDivisionError:
        logger.error("Division by zero error occurred during metric calculation")
    
    logger.info("Training completed successfully!")
    logger.info("Best mAP: 0.3000")
    logger.info("Checkpoints saved to ./checkpoints/")
    
    # Show where logs are saved
    print(f"\n[Console] Log files saved to: {Path('./demo_logs').absolute()}")


if __name__ == "__main__":
    main()