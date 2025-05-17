"""Debug the mismatch between checkpoint and model keys."""
import torch
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.utils.load_coco_weights import download_coco_checkpoint, convert_coco_weights_to_swin

# Create model
model = SwinMaskRCNN(num_classes=69)
model_keys = set(model.state_dict().keys())

# Load COCO checkpoint
checkpoint_path = download_coco_checkpoint()
checkpoint = torch.load(checkpoint_path, map_location='cpu')
state_dict = checkpoint.get('state_dict', checkpoint)

# Convert keys
converted_state_dict = convert_coco_weights_to_swin(state_dict)
checkpoint_keys = set(converted_state_dict.keys())

# Find mismatches
missing_in_checkpoint = model_keys - checkpoint_keys
extra_in_checkpoint = checkpoint_keys - model_keys

print(f"Model has {len(model_keys)} keys")
print(f"Checkpoint has {len(checkpoint_keys)} keys")
print(f"Missing in checkpoint: {len(missing_in_checkpoint)} keys")
print(f"Extra in checkpoint: {len(extra_in_checkpoint)} keys")

# Analyze missing keys by component
missing_by_component = {}
for key in missing_in_checkpoint:
    component = key.split('.')[0]
    if component not in missing_by_component:
        missing_by_component[component] = []
    missing_by_component[component].append(key)

print("\nMissing keys by component:")
for component, keys in missing_by_component.items():
    print(f"  {component}: {len(keys)} keys")
    if len(keys) <= 5:
        for key in keys:
            print(f"    - {key}")
    else:
        for key in keys[:3]:
            print(f"    - {key}")
        print(f"    ... and {len(keys) - 3} more")

# Analyze extra keys by component
extra_by_component = {}
for key in extra_in_checkpoint:
    component = key.split('.')[0]
    if component not in extra_by_component:
        extra_by_component[component] = []
    extra_by_component[component].append(key)

print("\nExtra keys in checkpoint:")
for component, keys in extra_by_component.items():
    print(f"  {component}: {len(keys)} keys")
    if len(keys) <= 5:
        for key in keys:
            print(f"    - {key}")
    else:
        for key in keys[:3]:
            print(f"    - {key}")
        print(f"    ... and {len(keys) - 3} more")

# Check specific important keys
print("\nChecking specific important keys:")
important_keys = [
    'roi_head.fc_cls.weight',
    'roi_head.fc_cls.bias',
    'roi_head.fc_reg.weight',
    'roi_head.fc_reg.bias',
    'roi_head.bbox_head.fc_cls.weight',
    'roi_head.bbox_head.fc_cls.bias',
    'backbone.patch_embed.proj.weight',
    'backbone.layers.0.blocks.0.attn.qkv.weight',
]

for key in important_keys:
    in_model = key in model_keys
    in_checkpoint = key in checkpoint_keys
    print(f"  {key}: model={in_model}, checkpoint={in_checkpoint}")

# Check what's actually in roi_head
print("\nROI head keys in model:")
roi_keys = [k for k in model_keys if k.startswith('roi_head')]
for key in sorted(roi_keys)[:10]:
    print(f"  {key}")
if len(roi_keys) > 10:
    print(f"  ... and {len(roi_keys) - 10} more")

print("\nROI head keys in checkpoint:")
roi_keys_ckpt = [k for k in checkpoint_keys if k.startswith('roi_head')]
for key in sorted(roi_keys_ckpt)[:10]:
    print(f"  {key}")
if len(roi_keys_ckpt) > 10:
    print(f"  ... and {len(roi_keys_ckpt) - 10} more")