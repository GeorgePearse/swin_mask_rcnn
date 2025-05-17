"""Check the actual model structure to understand key naming."""
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
import torch

# Create model
model = SwinMaskRCNN(num_classes=69)

print("Model structure inspection:")
print("\nBackbone attention module structure:")
first_block = model.backbone.layers[0].blocks[0]
print(f"First block type: {type(first_block)}")
print(f"First block attention: {type(first_block.attn)}")

# Check attention module attributes
print("\nAttention module attributes:")
for attr in dir(first_block.attn):
    if not attr.startswith('_') and hasattr(getattr(first_block.attn, attr), 'weight'):
        print(f"  {attr}: {type(getattr(first_block.attn, attr))}")

print("\nROI Head structure:")
print(f"ROI Head type: {type(model.roi_head)}")
print(f"Has bbox_head: {hasattr(model.roi_head, 'bbox_head')}")
print(f"Has fc_cls directly: {hasattr(model.roi_head, 'fc_cls')}")

# Check what's in bbox_head
if hasattr(model.roi_head, 'bbox_head'):
    print(f"Bbox head type: {type(model.roi_head.bbox_head)}")
    print("Bbox head attributes:")
    for attr in dir(model.roi_head.bbox_head):
        if not attr.startswith('_') and hasattr(getattr(model.roi_head.bbox_head, attr), 'weight'):
            print(f"  {attr}: {type(getattr(model.roi_head.bbox_head, attr))}")

print("\nFPN/Neck structure:")
print(f"Neck type: {type(model.neck)}")
print("Neck lateral convs:")
for i, conv in enumerate(model.neck.lateral_convs):
    print(f"  lateral_convs[{i}]: {type(conv)}")
    if hasattr(conv, 'conv'):
        print(f"    has .conv attribute")
    
print("\nRPN structure:")
print(f"RPN type: {type(model.rpn_head)}")
print("RPN attributes:")
for attr in dir(model.rpn_head):
    if not attr.startswith('_') and hasattr(getattr(model.rpn_head, attr), 'weight'):
        print(f"  {attr}: {type(getattr(model.rpn_head, attr))}")