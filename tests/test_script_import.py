"""Test that the training script can be imported without errors."""
import sys
sys.path.append('/home/georgepearse/swin_maskrcnn')

try:
    from scripts.train import IterationBasedTrainer, get_gpu_memory_mb
    from scripts.config import TrainingConfig
    print("✓ Successfully imported IterationBasedTrainer")
    print("✓ Successfully imported get_gpu_memory_mb")
    print("✓ Successfully imported TrainingConfig")
    
    # Test config loading
    config = TrainingConfig()
    print(f"✓ Train batch size: {config.train_batch_size}")
    print(f"✓ Validation batch size: {config.val_batch_size}")
    
    # Test memory function
    memory = get_gpu_memory_mb()
    print(f"✓ GPU memory function works: {memory:.2f} MB")
    
except Exception as e:
    print(f"✗ Import error: {e}")
    import traceback
    traceback.print_exc()