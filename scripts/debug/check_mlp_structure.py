"""Check the MLP structure differences between checkpoint and model."""
import torch
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.utils.load_coco_weights import download_coco_checkpoint

# Load checkpoint
checkpoint_path = download_coco_checkpoint()
checkpoint = torch.load(checkpoint_path, map_location='cpu')
state_dict = checkpoint.get('state_dict', checkpoint)

# Create model
model = SwinMaskRCNN(num_classes=69)

print("Model MLP structure:")
mlp = model.backbone.layers[0].blocks[0].mlp
print(f"MLP type: {type(mlp)}")
print("MLP layers:")
for i, layer in enumerate(mlp):
    print(f"  [{i}]: {type(layer)}")
    if hasattr(layer, 'weight'):
        print(f"       weight shape: {layer.weight.shape}")

print("\nCheckpoint MLP keys (first block):")
mlp_keys = [k for k in state_dict.keys() if 'layers.0.blocks.0.mlp' in k]
for key in sorted(mlp_keys)[:10]:
    print(f"  {key}: shape={state_dict[key].shape}")

print("\nCheckpoint norm layer keys:")
norm_keys = [k for k in state_dict.keys() if 'norm_layer' in k]
for key in sorted(norm_keys):
    print(f"  {key}: shape={state_dict[key].shape}")