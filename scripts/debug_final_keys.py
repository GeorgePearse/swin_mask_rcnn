"""Debug the final key mismatches."""
from swin_maskrcnn.utils.load_coco_weights import download_coco_checkpoint, convert_coco_weights_to_swin
import torch

# Load checkpoint
checkpoint_path = download_coco_checkpoint()
checkpoint = torch.load(checkpoint_path, map_location='cpu')
state_dict = checkpoint.get('state_dict', checkpoint)

# Test norm conversion
test_norm_keys = ['backbone.norm0.weight', 'backbone.norm1.weight', 'backbone.norm2.weight', 'backbone.norm3.weight']
for key in test_norm_keys:
    if key in state_dict:
        converted = convert_coco_weights_to_swin({key: state_dict[key]})
        print(f"Original: {key}")
        print(f"Converted: {list(converted.keys())}")
        print()

# Check shared FC issue
shared_fc_keys = [k for k in state_dict.keys() if 'shared_fcs' in k]
print("Shared FC keys in checkpoint:")
for key in shared_fc_keys[:5]:
    print(f"  {key}")
    converted = convert_coco_weights_to_swin({key: state_dict[key]})
    print(f"  -> {list(converted.keys())}")
    
# Check FPN convs
fpn_keys = [k for k in state_dict.keys() if 'fpn_convs' in k]
print("\nFPN conv keys in checkpoint:")
for key in sorted(fpn_keys)[-5:]:
    print(f"  {key}")
    converted = convert_coco_weights_to_swin({key: state_dict[key]})
    print(f"  -> {list(converted.keys())}")