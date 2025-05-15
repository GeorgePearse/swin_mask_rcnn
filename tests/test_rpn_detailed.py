import torch
from torchvision.ops import box_iou
from swin_maskrcnn.models.rpn import RPNHead
import torch.nn as nn

def test_rpn_loss():
    # Create a simplified RPN to debug
    device = torch.device('cpu')
    
    # Create some dummy data
    # Feature maps from FPN
    features = [
        torch.randn(2, 256, 100, 100),  # P3
        torch.randn(2, 256, 50, 50),    # P4
        torch.randn(2, 256, 25, 25),    # P5
    ]
    
    # Ground truth 
    gt_bboxes = [
        torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]], dtype=torch.float32),  # Image 1
        torch.tensor([[50, 50, 150, 150]], dtype=torch.float32)  # Image 2
    ]
    
    img_shapes = [(800, 800), (800, 800)]
    
    # Initialize RPN
    from swin_maskrcnn.models.rpn import AnchorGenerator
    anchor_generator = AnchorGenerator(
        strides=[8, 16, 32],
        ratios=[0.5, 1.0, 2.0],
        scales=[8]
    )
    rpn = RPNHead(
        in_channels=256,
        feat_channels=256,
        anchor_generator=anchor_generator
    )
    
    # Forward pass
    cls_scores, bbox_preds = rpn(features)
    
    print("Number of feature levels:", len(features))
    print("cls_scores shapes:", [c.shape for c in cls_scores])
    print("bbox_preds shapes:", [b.shape for b in bbox_preds])
    
    # Generate anchors for debugging
    featmap_sizes = [feat.shape[-2:] for feat in features]
    anchors = rpn.get_anchors(featmap_sizes, device)
    
    print("\nAnchors structure:")
    print(f"Number of levels: {len(anchors)}")
    for level_idx, level_anchors in enumerate(anchors):
        print(f"  Level {level_idx}: shape={level_anchors.shape}")
    
    # Try to reproduce the error
    print("\nTesting IoU computation in loss function:")
    
    for img_idx, gt_bbox in enumerate(gt_bboxes):
        print(f"\nImage {img_idx}: gt_bbox shape={gt_bbox.shape}")
        for level_idx, level_anchors in enumerate(anchors):
            print(f"  Level {level_idx}: anchor shape={level_anchors.shape}")
            try:
                ious = box_iou(level_anchors, gt_bbox)
                print(f"    IoU computed successfully: shape={ious.shape}")
            except Exception as e:
                print(f"    ERROR: {str(e)}")
                print(f"    gt_bbox type: {type(gt_bbox)}")
                print(f"    gt_bbox.shape: {gt_bbox.shape}")
                print(f"    gt_bbox.ndim: {gt_bbox.ndim}")
                if hasattr(gt_bbox, 'size'):
                    print(f"    gt_bbox.size(): {gt_bbox.size()}")
    
    # Actually call the loss function
    try:
        print("\nCalling RPN loss function...")
        losses = rpn.loss(cls_scores, bbox_preds, gt_bboxes, img_shapes)
        print("Loss computed successfully:", losses)
    except Exception as e:
        print(f"ERROR in loss function: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rpn_loss()