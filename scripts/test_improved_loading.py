"""Test the improved weight loading to see if we reduced mismatches."""
import torch
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.utils.load_coco_weights import load_coco_weights

# Create model with 69 classes
model = SwinMaskRCNN(num_classes=69)

print("Loading COCO pretrained weights...")
missing_keys, unexpected_keys = load_coco_weights(model, num_classes=69)

print(f"Missing keys: {len(missing_keys)}")
print(f"Unexpected keys: {len(unexpected_keys)}")

if len(missing_keys) > 0:
    print("\nFirst 10 missing keys:")
    for key in missing_keys[:10]:
        print(f"  {key}")
    if len(missing_keys) > 10:
        print(f"  ... and {len(missing_keys) - 10} more")

if len(unexpected_keys) > 0:
    print("\nFirst 10 unexpected keys:")
    for key in unexpected_keys[:10]:
        print(f"  {key}")
    if len(unexpected_keys) > 10:
        print(f"  ... and {len(unexpected_keys) - 10} more")

# Check if important layers are loaded
print("\nChecking key layers:")
key_checks = [
    ('roi_head.bbox_head.fc_cls', model.roi_head.bbox_head.fc_cls.weight),
    ('roi_head.bbox_head.fc_reg', model.roi_head.bbox_head.fc_reg.weight),
    ('rpn_head.rpn_conv', model.rpn_head.rpn_conv.weight),
    ('backbone.patch_embed.proj', model.backbone.patch_embed.proj.weight),
]

for name, param in key_checks:
    if param is not None:
        print(f"  {name}: shape={param.shape}, mean={param.mean():.6f}, std={param.std():.6f}")
    else:
        print(f"  {name}: None")

# Test forward pass
print("\nTesting forward pass...")
try:
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        outputs = model(dummy_input)
    print("Forward pass successful!")
    if isinstance(outputs, list) and len(outputs) > 0:
        print(f"Output type: {type(outputs[0])}")
except Exception as e:
    print(f"Forward pass failed: {e}")