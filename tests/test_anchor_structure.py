import torch
from swin_maskrcnn.models.rpn import AnchorGenerator

def test_anchor_structure():
    device = torch.device('cpu')
    
    # Initialize anchor generator
    anchor_gen = AnchorGenerator(
        strides=[8, 16, 32],
        ratios=[0.5, 1.0, 2.0],
        scales=[8]
    )
    
    # Feature map sizes (simulating multi-level features)
    featmap_sizes = [
        (100, 100),  # P3
        (50, 50),    # P4 
        (25, 25)     # P5
    ]
    
    # Generate anchors for a single image
    anchors = []
    for i, size in enumerate(featmap_sizes):
        stride = anchor_gen.strides[i]
        anchor = anchor_gen.generate_anchors(size, stride, device)
        anchors.append(anchor)
    
    print("Anchor structure for single image:")
    for i, level_anchors in enumerate(anchors):
        print(f"  Level {i}: shape={level_anchors.shape}")
    
    # Now simulate what should happen in loss function
    gt_bboxes = [
        torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]], dtype=torch.float32),  # Image 1
        torch.tensor([[50, 50, 150, 150]], dtype=torch.float32)  # Image 2
    ]
    
    print("\nGround truth structure:")
    for i, gt_bbox in enumerate(gt_bboxes):
        print(f"  Image {i}: shape={gt_bbox.shape}")
    
    # Test the expected structure in loss function  
    print("\nWhat the loss function expects:")
    
    # This is what's likely happening - mismatch between per-image anchors and per-image gt
    for img_idx, gt_bbox in enumerate(gt_bboxes):
        print(f"\nProcessing Image {img_idx}:")
        for level_idx, level_anchors in enumerate(anchors):
            # This is where the bug is - using same anchors for all images
            print(f"  Level {level_idx}: anchors shape={level_anchors.shape}, gt shape={gt_bbox.shape}")
            
            # Test IoU computation
            from torchvision.ops import box_iou
            try:
                ious = box_iou(level_anchors, gt_bbox)
                print(f"    IoU shape: {ious.shape} - SUCCESS")
            except Exception as e:
                print(f"    ERROR: {str(e)}")

if __name__ == "__main__":
    test_anchor_structure()