#!/usr/bin/env python3
"""Test script to demonstrate logging functionality."""
import torch
import time
from pathlib import Path
from swin_maskrcnn.utils.logging import setup_logger


def test_logging():
    """Test different logging scenarios."""
    # Setup logger with different configurations
    log_dir = Path("./test_logs")
    
    # Test INFO level logger
    logger_info = setup_logger(
        name="test_info",
        log_dir=str(log_dir),
        level="INFO"
    )
    
    logger_info.info("=== Starting Logging Test ===")
    logger_info.info("This is an INFO message")
    logger_info.warning("This is a WARNING message")
    logger_info.error("This is an ERROR message")
    logger_info.debug("This DEBUG message won't appear in INFO level")
    
    # Test DEBUG level logger
    logger_debug = setup_logger(
        name="test_debug",
        log_dir=str(log_dir),
        level="DEBUG"
    )
    
    logger_debug.debug("This DEBUG message will appear in DEBUG level")
    logger_debug.info("Processing batch 1/100...")
    
    # Test exception logging
    try:
        x = 1 / 0
    except ZeroDivisionError:
        logger_info.exception("Caught an exception:")
    
    # Test formatting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger_info.info(f"Using device: {device}")
    
    # Test performance logging
    start_time = time.time()
    time.sleep(0.1)  # Simulate some work
    elapsed = time.time() - start_time
    logger_info.info(f"Operation took {elapsed:.3f} seconds")
    
    # Test structured logging
    metrics = {
        "loss": 1.234,
        "accuracy": 0.956,
        "lr": 0.001
    }
    logger_info.info(f"Training metrics: {metrics}")
    
    # Test multiline logging
    multiline_msg = """
    Model Configuration:
    - Architecture: SWIN-Transformer
    - Backbone: SWIN-Small
    - Classes: 69
    - Image Size: 800x800
    """
    logger_info.info(multiline_msg)
    
    logger_info.info("=== Logging Test Complete ===")
    logger_info.info(f"Check log files in: {log_dir}")


if __name__ == "__main__":
    test_logging()