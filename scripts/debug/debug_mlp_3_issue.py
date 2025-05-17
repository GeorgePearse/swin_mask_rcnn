"""Debug why MLP layer 3 weights are not being converted correctly."""
from swin_maskrcnn.utils.load_coco_weights import convert_coco_weights_to_swin, download_coco_checkpoint
import torch

# Load checkpoint
checkpoint_path = download_coco_checkpoint()
checkpoint = torch.load(checkpoint_path, map_location='cpu')
state_dict = checkpoint.get('state_dict', checkpoint)

# Test the conversion for a specific MLP key
test_key = 'backbone.stages.0.blocks.0.ffn.layers.1.weight'
print(f"Original key: {test_key}")
converted = convert_coco_weights_to_swin({test_key: state_dict[test_key]})
print(f"Converted keys: {list(converted.keys())}")

# Let's manually trace through the conversion
key = test_key
key = key.replace('backbone.stages', 'backbone.layers')
key = key.replace('ffn', 'mlp')
print(f"After basic replacements: {key}")

# Check the MLP layer handling
if 'mlp.layers.' in key:
    parts = key.split('.')
    mlp_idx = parts.index('mlp')
    if mlp_idx + 3 < len(parts) and parts[mlp_idx + 1] == 'layers':
        layer_idx = parts[mlp_idx + 2]
        sublayer_idx = parts[mlp_idx + 3]
        print(f"Indices: layer={layer_idx}, sublayer={sublayer_idx}")
        
        # The current logic
        if layer_idx == '0' and sublayer_idx == '0':
            new_idx = '0'
        elif layer_idx == '1':
            new_idx = '3'
        else:
            new_idx = sublayer_idx
        print(f"New index would be: {new_idx}")
        
        # Problem: we're looking at index 3, but the actual part is 'weight' not '3'
        # The sublayer_idx is 'weight', not '3'
        print(f"Parts at mlp_idx + 3: {parts[mlp_idx + 3]}")  # This is 'weight'