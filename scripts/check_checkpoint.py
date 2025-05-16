"""Inspect our checkpoint to check predictions."""
import torch
import sys
from pathlib import Path


def inspect_checkpoint(checkpoint_path: str):
    """Inspect checkpoint contents."""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    print("==== Checkpoint Contents ====")
    for key in ckpt.keys():
        print(f"{key}: {type(ckpt[key])}")
    
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        print("\n==== Model State Dict Keys ====")
        print(f"Total keys: {len(state_dict)}")
        
        # Look for specific layers that might indicate issues
        relevant_keys = []
        for key in state_dict.keys():
            if any(pattern in key for pattern in ["rpn", "roi_head", "fc_cls", "fc_reg", "conv_logits"]):
                relevant_keys.append(key)
        
        print("\n==== Relevant Output Layer Keys ====")
        for key in sorted(relevant_keys):
            value = state_dict[key]
            print(f"{key}: {value.shape} (min={value.min():.4f}, max={value.max():.4f}, mean={value.mean():.4f})")
    
    # Check if optimizer state is included
    if "optimizer" in ckpt:
        opt_state = ckpt["optimizer"]
        print(f"\n==== Optimizer State ====")
        print(f"Type: {type(opt_state)}")
        if "state" in opt_state:
            print(f"Number of parameter groups: {len(opt_state['state'])}")
    
    # Check training metadata
    for key in ["epoch", "iteration", "global_step"]:
        if key in ckpt:
            print(f"{key}: {ckpt[key]}")
    
    return ckpt


if __name__ == "__main__":
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        checkpoint_path = "test_checkpoints/checkpoint_step_200.pth"
    
    inspect_checkpoint(checkpoint_path)