"""Debug model dimensions."""
import torch
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN

# Create model with debug
model = SwinMaskRCNN(num_classes=10)

print("Model created")
print(f"Backbone embed_dims: {model.backbone.embed_dims}")
print(f"Backbone num_features: {model.backbone.num_features}")
print(f"Backbone out_indices: {model.backbone.out_indices}")

# Check norm layers
for i, norm in enumerate(model.backbone.norms):
    print(f"Norm layer {i}: {norm}")

# Create image
batch_size = 1
img_size = 224  
images = torch.randn(batch_size, 3, img_size, img_size)

# Test backbone only
backbone = model.backbone

# Let's test step by step
x = backbone.patch_embed(images)
print(f"After patch_embed: {x.shape}")

H = W = int(x.shape[1] ** 0.5)
print(f"H={H}, W={W}")

# Test first layer
layer0 = backbone.layers[0]
x_out, H_out, W_out, x, Hd, Wd = layer0(x, H, W)
print(f"After layer 0: x_out.shape={x_out.shape}, H={H_out}, W={W_out}")

# Check if layer 0 is in out_indices
if 0 in backbone.out_indices:
    norm_idx = backbone.out_indices.index(0)
    print(f"Applying norm layer {norm_idx}")
    norm_layer = backbone.norms[norm_idx]
    print(f"Norm layer expects {norm_layer.normalized_shape}")
    print(f"Got shape {x_out.shape}")
    # This should fail
    x_out_normed = norm_layer(x_out)