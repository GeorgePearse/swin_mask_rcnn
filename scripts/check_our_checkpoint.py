"""Check our checkpoint to see training progress."""
import torch
import sys
from pathlib import Path

def inspect_checkpoint(checkpoint_path: str):
    """Inspect checkpoint structure and training state."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("\nCheckpoint keys:")
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], dict):
            print(f"  {key}: dict with {len(checkpoint[key])} keys")
        else:
            print(f"  {key}: {type(checkpoint[key])}")
    
    # Check optimizer state
    if 'optimizer_state_dict' in checkpoint:
        print("\nOptimizer state available")
        opt_state = checkpoint['optimizer_state_dict']
        if 'state' in opt_state and len(opt_state['state']) > 0:
            # Get first parameter's state
            first_param_state = list(opt_state['state'].values())[0]
            if 'step' in first_param_state:
                print(f"  Steps: {first_param_state['step']}")
    
    # Check training info
    if 'epoch' in checkpoint:
        print(f"\nTraining epoch: {checkpoint['epoch']}")
    if 'step' in checkpoint:
        print(f"Training step: {checkpoint['step']}")
    if 'loss' in checkpoint:
        print(f"Last loss: {checkpoint['loss']}")
    
    # Check training history
    if 'train_history' in checkpoint:
        print("\nTraining history:")
        train_hist = checkpoint['train_history']
        for key, values in train_hist.items():
            if isinstance(values, list) and len(values) > 0:
                print(f"  {key}: {len(values)} entries, last 5: {values[-5:]}")
    
    if 'val_history' in checkpoint:
        print("\nValidation history:")
        val_hist = checkpoint['val_history']
        for key, values in val_hist.items():
            if isinstance(values, list) and len(values) > 0:
                print(f"  {key}: {len(values)} entries, last 5: {values[-5:]}")
    
    # Check model state
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"\nModel state dict: {len(state_dict)} parameters")
        
        # Check key layers
        key_patterns = ['roi_head.bbox_head.fc_cls', 'roi_head.bbox_head.fc_reg', 
                       'rpn_head.conv_cls', 'rpn_head.conv_bbox']
        
        for pattern in key_patterns:
            matching_keys = [k for k in state_dict.keys() if pattern in k]
            if matching_keys:
                print(f"\n{pattern}:")
                for key in matching_keys:
                    tensor = state_dict[key]
                    print(f"  {key}: shape={tensor.shape}, mean={tensor.mean():.6f}, std={tensor.std():.6f}")

def check_model_predictions():
    """Quick check of model predictions on a dummy input."""
    from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
    
    # Load model
    model = SwinMaskRCNN(num_classes=69)
    checkpoint = torch.load(sys.argv[1], map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        try:
            output = model(dummy_input)
            print(f"\nModel output on dummy input: {type(output)}")
            if isinstance(output, list) and len(output) > 0:
                print(f"First detection: {output[0].keys()}")
        except Exception as e:
            print(f"\nError during forward pass: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_our_checkpoint.py <checkpoint_path>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    inspect_checkpoint(checkpoint_path)
    
    print("\n" + "="*50)
    print("Checking model predictions...")
    check_model_predictions()