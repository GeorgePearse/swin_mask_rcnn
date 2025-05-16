# Logging Setup Summary

This document summarizes the logging setup implementation for the SWIN Mask R-CNN project.

## Changes Made

1. **Created Logging Module**: `/swin_maskrcnn/utils/logging.py`
   - Setup function to configure loggers with console and file handlers
   - Separate log files for different log levels (info, error, debug)
   - Timestamps included in log filenames
   - Customizable log levels and formats

2. **Updated Training Scripts**:
   - `/scripts/train.py`: Converted all print statements to logger calls
   - `/scripts/check_initial_loss.py`: Converted all print statements to logger calls
   - `/swin_maskrcnn/training/trainer.py`: Added logger support

3. **Log Directory Structure**:
   - Log files are saved to `<checkpoint_dir>/logs/`
   - Different files for different levels:
     - `train_<timestamp>.log`: All INFO level messages
     - `error_<timestamp>.log`: ERROR and CRITICAL messages only
     - `debug_<timestamp>.log`: DEBUG messages (when debug level is enabled)

## Usage

### Basic Setup
```python
from swin_maskrcnn.utils.logging import setup_logger

# Setup logger
logger = setup_logger(
    name="my_module",
    log_dir="./logs",
    level="INFO"
)

# Use logger
logger.info("Training started")
logger.warning("GPU memory low")
logger.error("Model failed to converge")
```

### Log Levels
- **DEBUG**: Detailed information, typically of interest only when diagnosing problems
- **INFO**: Confirmation that things are working as expected
- **WARNING**: An indication that something unexpected happened
- **ERROR**: Due to a more serious problem, the software has not been able to perform some function
- **CRITICAL**: A serious error, indicating that the program itself may be unable to continue running

## Benefits

1. **Better Debugging**: All output is saved to files for later analysis
2. **Structured Logging**: Different log levels for different types of messages
3. **Production Ready**: No more print statements in production code
4. **Timestamps**: All log entries include timestamps
5. **Flexible**: Easy to change log levels without modifying code

## Configuration

Log level can be changed via:
1. Environment variable: `LOG_LEVEL=DEBUG python train.py`
2. Config file: Add `log_level: DEBUG` to your config.yaml
3. Code: Pass `level="DEBUG"` to setup_logger()

## Next Steps

1. Update remaining modules to use logging
2. Add log rotation to prevent disk space issues
3. Consider structured logging (JSON format) for better parsing
4. Add remote logging capabilities for distributed training