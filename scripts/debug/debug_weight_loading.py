"""Debug the weight loading process."""
import torch
import numpy as np
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.utils.load_coco_weights import download_coco_checkpoint, convert_coco_weights_to_swin


def debug_weight_loading():
    """Debug weight loading process."""
    
    # Create model
    model = SwinMaskRCNN(num_classes=69)
    
    # Get initial weights
    initial_fc_cls_weight = model.roi_head.bbox_head.fc_cls.weight.data.clone()
    initial_fc_cls_bias = model.roi_head.bbox_head.fc_cls.bias.data.clone()
    
    print("Initial classifier stats:")
    print(f"  Weight norm: {initial_fc_cls_weight.norm():.4f}")
    print(f"  Bias: {initial_fc_cls_bias[:5]}")  # First 5 values
    
    # Load checkpoint
    checkpoint_path = download_coco_checkpoint() 
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Look for classifier keys in checkpoint
    print("\nClassifier keys in checkpoint:")
    for key in state_dict.keys():
        if 'fc_cls' in key or 'classifier' in key:
            print(f"  {key}: shape={state_dict[key].shape}")
    
    # Convert keys
    converted_state_dict = convert_coco_weights_to_swin(state_dict)
    
    # Check converted keys
    print("\nConverted classifier keys:")
    for key in converted_state_dict.keys():
        if 'fc_cls' in key:
            print(f"  {key}: shape={converted_state_dict[key].shape}")
            if 'weight' in key:
                print(f"    Weight stats: min={converted_state_dict[key].min():.4f}, max={converted_state_dict[key].max():.4f}, norm={converted_state_dict[key].norm():.4f}")
            if 'bias' in key:
                print(f"    Bias stats: min={converted_state_dict[key].min():.4f}, max={converted_state_dict[key].max():.4f}")
                print(f"    First 5 values: {converted_state_dict[key][:5]}")
    
    # Check if the weight key exists and what it looks like
    cls_weight_key = 'roi_head.bbox_head.fc_cls.weight'
    if cls_weight_key in converted_state_dict:
        coco_weight = converted_state_dict[cls_weight_key]
        print(f"\nCOCO classifier weight shape: {coco_weight.shape}")
        print(f"  Norm: {coco_weight.norm():.4f}")
        print(f"  Background weight norm: {coco_weight[0].norm():.4f}")
        print(f"  Foreground weight norm mean: {coco_weight[1:].norm(dim=1).mean():.4f}")
        
        # Check if weights are mostly zero
        zero_ratio = (coco_weight.abs() < 1e-6).float().mean()
        print(f"  Ratio of near-zero weights: {zero_ratio:.4f}")
    
    print("\nDone!")


if __name__ == "__main__":
    debug_weight_loading()