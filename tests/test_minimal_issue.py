import torch
from torchvision.ops import box_iou

# Reproduce the minimal issue
device = torch.device('cpu')

# Create a ground truth bbox tensor
gt_bbox = torch.tensor([[100, 100, 200, 200]], dtype=torch.float32)  # Shape: [1, 4]
print(f"Original gt_bbox shape: {gt_bbox.shape}")

# Let's see what happens when we try to access it as if it were 1D
print(f"gt_bbox[0] shape: {gt_bbox[0].shape}")  # This should work
print(f"gt_bbox[0] value: {gt_bbox[0]}")

# Now let's try what might be causing the error
try:
    # If the code somewhere expects gt_bbox to be a list of 1D tensors
    # instead of a 2D tensor, this could cause issues
    print(f"Trying to access with too many indices...")
    print(f"gt_bbox[0, 1]: {gt_bbox[0, 1]}")  # This works
    print(f"gt_bbox[:, 2]: {gt_bbox[:, 2]}")  # This works too
except Exception as e:
    print(f"Error: {e}")

# Test with box_area which is mentioned in the error
from torchvision.ops.boxes import box_area

print(f"\nTesting box_area:")
print(f"box_area(gt_bbox): {box_area(gt_bbox)}")

# Test what happens if gt_bbox is somehow converted to 1D
gt_bbox_1d = torch.tensor([100, 100, 200, 200], dtype=torch.float32)  # Shape: [4]
print(f"\n1D tensor shape: {gt_bbox_1d.shape}")

try:
    print(f"box_area on 1D tensor...")
    result = box_area(gt_bbox_1d)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error with 1D tensor: {e}")
    print(f"This matches our error!")

# Let's test with an unsqueezed version
gt_bbox_unsqueezed = gt_bbox_1d.unsqueeze(0)
print(f"\nUnsqueezed shape: {gt_bbox_unsqueezed.shape}")
print(f"box_area(unsqueezed): {box_area(gt_bbox_unsqueezed)}")