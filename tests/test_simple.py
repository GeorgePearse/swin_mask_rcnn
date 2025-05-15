"""Simple test to validate model dimensions."""
import torch
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN

# Create model
model = SwinMaskRCNN(num_classes=10)

# Create small dummy input
batch_size = 1
img_size = 224  # Must be divisible by patch_size
images = torch.randn(batch_size, 3, img_size, img_size)

# Test backbone only
backbone = model.backbone
features = backbone(images)

print(f"Input shape: {images.shape}")
for i, feat in enumerate(features):
    print(f"Feature {i} shape: {feat.shape}")

# Test full model
model.eval()
with torch.no_grad():
    outputs = model(images)
    print(f"Outputs: {outputs}")