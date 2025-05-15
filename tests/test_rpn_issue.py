import torch
import numpy as np

# Simulate the data structure causing the issue
def test_rpn_input_issue():
    # Create sample data as it comes from the dataset
    target1 = {
        'boxes': torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100]]),
        'labels': torch.tensor([1, 2])
    }
    target2 = {
        'boxes': torch.tensor([[20, 20, 40, 40]]),
        'labels': torch.tensor([0])
    }
    
    targets = [target1, target2]
    
    # This is what's being passed to RPN loss
    gt_bboxes = [t['boxes'] for t in targets]
    
    print("gt_bboxes structure:")
    for i, gt_bbox in enumerate(gt_bboxes):
        print(f"  Image {i}: shape={gt_bbox.shape}, type={type(gt_bbox)}")
    
    # Test what happens when we try to index into a bbox
    print("\nAttempting to index as in RPN loss:")
    for img_idx, gt_bbox in enumerate(gt_bboxes):
        print(f"  Image {img_idx}: gt_bbox shape={gt_bbox.shape}")
        # This should work - it's already a 2D tensor
        if len(gt_bbox.shape) == 2:
            print(f"    Box 0: {gt_bbox[0]}")
        else:
            print(f"    ERROR: Expected 2D tensor but got shape {gt_bbox.shape}")

    # Test IoU computation
    from torchvision.ops import box_iou
    
    # Create some fake anchors
    anchors = torch.tensor([
        [0, 0, 32, 32],
        [16, 16, 48, 48],
        [32, 32, 64, 64]
    ])
    
    print("\nTesting IoU computation:")
    for i, gt_bbox in enumerate(gt_bboxes):
        try:
            ious = box_iou(anchors, gt_bbox)
            print(f"  Image {i}: IoU shape={ious.shape} - SUCCESS")
        except Exception as e:
            print(f"  Image {i}: ERROR - {str(e)}")

if __name__ == "__main__":
    test_rpn_input_issue()