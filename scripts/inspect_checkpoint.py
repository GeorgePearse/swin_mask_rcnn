"""Script to inspect mmdetection checkpoint structure for key mapping."""
import torch
from pathlib import Path
from swin_maskrcnn.utils.pretrained_loader import download_checkpoint


def inspect_checkpoint(url: str):
    """Download and inspect checkpoint structure."""
    # Download checkpoint
    checkpoint_path = download_checkpoint(url)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("Checkpoint keys:")
    for key in checkpoint.keys():
        print(f"  {key}")
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("\nState dict structure:")
        
        # Group keys by prefix
        prefixes = {}
        for key in sorted(state_dict.keys()):
            prefix = key.split('.')[0]
            if prefix not in prefixes:
                prefixes[prefix] = []
            prefixes[prefix].append(key)
        
        # Print grouped keys
        for prefix, keys in prefixes.items():
            print(f"\n{prefix}:")
            for key in keys[:10]:  # Show first 10 keys
                print(f"  {key} -> {state_dict[key].shape}")
            if len(keys) > 10:
                print(f"  ... and {len(keys) - 10} more")


if __name__ == "__main__":
    url = "https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth"
    inspect_checkpoint(url)