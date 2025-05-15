import torch
from torchvision.ops import box_iou

# Let's trace the actual issue based on the error stack
device = torch.device('cpu')

# Create test data similar to what would be in the RPN
level_anchors = torch.tensor([
    [0, 0, 32, 32],
    [32, 0, 64, 32],
    [0, 32, 32, 64]
], dtype=torch.float32)

# Case 1: Normal 2D ground truth (what it should be)
gt_bbox_2d = torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]], dtype=torch.float32)
print(f"Case 1 - Normal 2D gt_bbox shape: {gt_bbox_2d.shape}")

try:
    ious_1 = box_iou(level_anchors, gt_bbox_2d)
    print(f"IoU computation works: {ious_1.shape}")
except Exception as e:
    print(f"Error: {e}")

# Case 2: Check what happens if gt_bbox somehow becomes 1D
# This might happen if gt_bbox is indexed incorrectly
print(f"\nCase 2 - Testing potential 1D issue")

# Let's simulate if gt_bbox was incorrectly sliced
gt_bbox_slice = gt_bbox_2d[0]  # This creates a 1D tensor!
print(f"gt_bbox[0] shape: {gt_bbox_slice.shape}")

try:
    ious_2 = box_iou(level_anchors, gt_bbox_slice)
    print(f"IoU computation result: {ious_2.shape}")
except Exception as e:
    print(f"Error with 1D tensor: {e}")
    print("This matches our error!")

# Case 3: Let's check if the issue is in matched_gt indexing
print(f"\nCase 3 - Testing matched_gt indexing")
max_ious, matched_gt_idx = torch.tensor([0.8, 0.2, 0.9]), torch.tensor([0, 1, 0])
pos_mask = torch.tensor([True, False, True])

# Correct way to index
matched_gt_correct = gt_bbox_2d[matched_gt_idx[pos_mask]]
print(f"Correct indexing result shape: {matched_gt_correct.shape}")

# Wrong way that could cause 1D issue
if len(gt_bbox_2d) == 1:
    # If gt_bbox has only one box, direct indexing might cause issues
    gt_bbox_single = torch.tensor([[100, 100, 200, 200]], dtype=torch.float32)
    print(f"\nWith single gt_bbox shape: {gt_bbox_single.shape}")
    
    # This would work
    matched_gt_single = gt_bbox_single[matched_gt_idx[pos_mask]]
    print(f"Indexing single bbox result shape: {matched_gt_single.shape}")