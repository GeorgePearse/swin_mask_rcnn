"""Debug the final conversion issues."""
import torch
from swin_maskrcnn.utils.load_coco_weights import download_coco_checkpoint
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN

# Load checkpoint
checkpoint_path = download_coco_checkpoint()
checkpoint = torch.load(checkpoint_path, map_location='cpu')
state_dict = checkpoint.get('state_dict', checkpoint)

# Test key conversion 
test_keys = [
    'backbone.norm0.weight',
    'backbone.norm0.bias',
    'backbone.stages.0.blocks.0.ffn.layers.1.weight',
    'backbone.stages.0.blocks.0.ffn.layers.1.bias',
]

print("Key conversion test:")
for key in test_keys:
    if key in state_dict:
        new_key = key
        new_key = new_key.replace('backbone.stages', 'backbone.layers')
        new_key = new_key.replace('ffn', 'mlp')
        
        # First handle the specific norm case
        if 'backbone.norm' in new_key and new_key[13].isdigit():  # 13 is position after 'backbone.norm'
            new_key = new_key.replace('backbone.norm', 'backbone.norm_layer')
        
        # Handle MLP layer mapping
        if 'mlp.layers.' in new_key:
            print(f"  Found MLP layer: {key}")
            parts = new_key.split('.')
            mlp_idx = parts.index('mlp')
            if mlp_idx + 3 < len(parts):
                layer_idx = parts[mlp_idx + 2]
                sublayer_idx = parts[mlp_idx + 3]
                print(f"    Layer indices: {layer_idx}.{sublayer_idx}")
                # Should be 1 -> 3 for the second linear layer
                if layer_idx == '1':
                    new_idx = '3'
                    new_parts = parts[:mlp_idx + 1] + [new_idx] + parts[mlp_idx + 4:]
                    new_key = '.'.join(new_parts)
        
        print(f"  {key} -> {new_key}")

# Check what actual keys exist in model for norm layers
model = SwinMaskRCNN(num_classes=69)
model_keys = list(model.state_dict().keys())

print("\nModel norm layer keys:")
norm_keys = [k for k in model_keys if 'norm_layer' in k and 'backbone' in k]
for key in sorted(norm_keys)[:10]:
    print(f"  {key}: shape={model.state_dict()[key].shape}")