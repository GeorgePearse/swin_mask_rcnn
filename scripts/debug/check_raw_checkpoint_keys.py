"""Check raw checkpoint keys before conversion."""
import torch
from swin_maskrcnn.utils.load_coco_weights import download_coco_checkpoint

# Load checkpoint
checkpoint_path = download_coco_checkpoint()
checkpoint = torch.load(checkpoint_path, map_location='cpu')
state_dict = checkpoint.get('state_dict', checkpoint)

print(f"Total keys in checkpoint: {len(state_dict)}")

# Check MLP keys
print("\nMLP keys in checkpoint (raw):")
mlp_keys = [k for k in state_dict.keys() if 'ffn' in k and 'stages.0.blocks.0' in k]
for key in sorted(mlp_keys)[:10]:
    print(f"  {key}: shape={state_dict[key].shape}")

# Check converted mlp keys  
converted_keys = [k.replace('ffn', 'mlp').replace('stages', 'layers') for k in mlp_keys]
print("\nExpected converted keys:")
for key in converted_keys[:10]:
    print(f"  {key}")

# Check norm layer keys
print("\nNorm layer keys:")
norm_keys = [k for k in state_dict.keys() if 'norm' in k and 'backbone' in k]
for key in sorted(norm_keys)[:20]:
    print(f"  {key}")