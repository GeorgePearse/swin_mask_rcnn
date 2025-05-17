import torch
from pathlib import Path
import os
from pathlib import Path
import urllib.request
import tempfile

def download_coco_checkpoint():
    """Download the COCO pretrained checkpoint."""
    # URL for SWIN Mask R-CNN checkpoint
    url = "https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth"
    
    cache_dir = Path.home() / ".cache" / "swin_maskrcnn"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    filename = url.split("/")[-1]
    cache_path = cache_dir / filename
    
    if not cache_path.exists():
        print(f"Downloading checkpoint from {url}")
        urllib.request.urlretrieve(url, cache_path)
        print(f"Saved to {cache_path}")
    else:
        print(f"Using cached checkpoint from {cache_path}")
    
    return cache_path

def analyze_checkpoint(path):
    """Analyze the checkpoint structure."""
    print(f"\nAnalyzing checkpoint: {path}")
    
    ckpt = torch.load(path, map_location='cpu')
    
    print(f"Checkpoint type: {type(ckpt)}")
    
    if isinstance(ckpt, dict):
        print(f"Keys: {list(ckpt.keys())}")
        
        # Check state_dict
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
            print(f"\nState dict keys sample: {list(state_dict.keys())[:10]}")
            
            # Check for classifier
            cls_keys = [k for k in state_dict.keys() if 'fc_cls' in k]
            print(f"\nClassifier keys: {cls_keys}")
            
            if cls_keys:
                for key in cls_keys:
                    if 'weight' in key:
                        print(f"{key} shape: {state_dict[key].shape}")
                    if 'bias' in key:
                        print(f"{key} shape: {state_dict[key].shape}")
                        print(f"Background bias: {state_dict[key][0].item():.4f}")
                        print(f"Object bias mean: {state_dict[key][1:].mean().item():.4f}")
                        print(f"Object bias range: [{state_dict[key][1:].min().item():.4f}, {state_dict[key][1:].max().item():.4f}]")
            
            # Check for bbox head
            bbox_keys = [k for k in state_dict.keys() if 'bbox_head' in k]
            print(f"\nBbox head keys sample: {bbox_keys[:5]}")
    else:
        print(f"Direct tensor checkpoint: {ckpt.shape}")


def main():
    # Download and analyze COCO checkpoint
    coco_path = download_coco_checkpoint()
    analyze_checkpoint(coco_path)
    
    # Also analyze our test checkpoint for comparison
    test_path = "/home/georgepearse/swin_maskrcnn/test_checkpoints/checkpoint_step_200.pth"
    if Path(test_path).exists():
        analyze_checkpoint(test_path)


if __name__ == "__main__":
    main()