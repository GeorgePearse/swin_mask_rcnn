import torch
from swin_maskrcnn.models.rpn import RPNHead

def test_anchor_structure_fix():
    device = torch.device('cpu')
    
    # Initialize RPN with correct parameters
    rpn = RPNHead(
        in_channels=256,
        feat_channels=256,
    )
    
    # Feature map sizes (simulating multi-level features)
    featmap_sizes = [
        (100, 100),  # P3
        (50, 50),    # P4
        (25, 25)     # P5
    ]
    
    # Generate anchors using the actual method
    anchors = rpn.get_anchors(featmap_sizes, device)
    
    print("Anchor structure from RPN:")
    print(f"Type: {type(anchors)}")
    print(f"Length: {len(anchors)}")
    for i, level_anchors in enumerate(anchors):
        print(f"  Level {i}: shape={level_anchors.shape}")
    
    # Now let's see what the loss function expects vs what it gets
    gt_bboxes = [
        torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]], dtype=torch.float32),  # Image 1
        torch.tensor([[50, 50, 150, 150]], dtype=torch.float32)  # Image 2
    ]
    
    print("\nShould be iterating through images, but anchors are per-level!")
    print("This is the bug - we need anchors to be per-image, per-level")
    
    # What the code currently does (WRONG):
    print("\nCurrent (WRONG) iteration:")
    for img_idx, gt_bbox in enumerate(gt_bboxes):
        print(f"Image {img_idx}: gt_bbox shape={gt_bbox.shape}")
        for level_idx, level_anchors in enumerate(anchors):
            print(f"  ERROR: This treats anchors as per-image but they're per-level!")
            print(f"  Level {level_idx}: shape={level_anchors.shape}")
            if img_idx > 0:  # This would cause the error
                break
    
    # What it should do (CORRECT):
    print("\nCorrect iteration:")
    for img_idx, gt_bbox in enumerate(gt_bboxes):
        print(f"Image {img_idx}: gt_bbox shape={gt_bbox.shape}")
        # Anchors are the same for all images, just iterate through levels
        for level_idx, level_anchors in enumerate(anchors):
            print(f"  Level {level_idx}: shape={level_anchors.shape}")

if __name__ == "__main__":
    test_anchor_structure_fix()