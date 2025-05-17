"""Check the actual norm layer structure in the model."""
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN

model = SwinMaskRCNN(num_classes=69)

print("Backbone attributes:")
for attr in dir(model.backbone):
    if 'norm' in attr and not attr.startswith('_'):
        print(f"  {attr}: {type(getattr(model.backbone, attr))}")

print("\nModel state dict norm-related keys:")
for key in sorted(model.state_dict().keys()):
    if 'norm' in key and 'backbone' in key:
        print(f"  {key}: shape={model.state_dict()[key].shape}")