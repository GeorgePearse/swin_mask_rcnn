"""Debug RPN outputs to see where predictions are failing."""
import torch
import numpy as np
from PIL import Image
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from torchvision.transforms import Compose, ToTensor, Normalize


def debug_rpn(checkpoint_path, image_path, num_classes=69):
    """Debug RPN outputs."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = SwinMaskRCNN(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Hook to capture RPN outputs
    rpn_outputs = {}
    
    def rpn_forward_hook(module, input, output):
        rpn_outputs['cls_scores'] = output[0]
        rpn_outputs['bbox_preds'] = output[1]
    
    def rpn_proposals_hook(module, input, output):
        rpn_outputs['proposals'] = output
    
    # Register hooks
    model.rpn_head.register_forward_hook(rpn_forward_hook)
    
    # Manual forward pass to debug
    with torch.no_grad():
        # Extract features
        features = model.backbone(img_tensor)
        features = model.neck(features)
        
        # RPN forward
        rpn_cls_scores, rpn_bbox_preds = model.rpn_head(features)
        
        print("=== RPN scores ===")
        for i, score in enumerate(rpn_cls_scores):
            print(f"Level {i}: shape={score.shape}")
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(score)
            print(f"  Foreground probs > 0.5: {(probs > 0.5).sum()}")
            print(f"  Foreground probs > 0.3: {(probs > 0.3).sum()}")
            print(f"  Max foreground prob: {probs.max():.4f}")
            print(f"  Mean foreground prob: {probs.mean():.4f}")
        
        print("\n=== Getting proposals ===")
        proposals = model.rpn_head.get_proposals(
            rpn_cls_scores, rpn_bbox_preds,
            [(img.shape[-2], img.shape[-1]) for img in img_tensor],
            {'nms_pre': 1000, 'nms_thr': 0.7, 'max_per_img': 1000}
        )
        
        print(f"Number of proposals: {[len(p) for p in proposals]}")
        for i, prop in enumerate(proposals):
            print(f"Image {i}: {len(prop)} proposals")
            if len(prop) > 0:
                print(f"  Proposal shape: {prop.shape}")
                print(f"  First 5 proposals: {prop[:5]}")
        
        # Continue with ROI head
        if sum(len(p) for p in proposals) > 0:
            print("\n=== ROI head forward ===")
            detections = model.roi_head(features, proposals)
            for i, det in enumerate(detections):
                print(f"Image {i} detections:")
                for k, v in det.items():
                    if isinstance(v, torch.Tensor):
                        print(f"  {k}: shape={v.shape}")
                        if k == 'scores' and v.numel() > 0:
                            print(f"    Max score: {v.max():.4f}")
                            print(f"    Top 5 scores: {v[:5].tolist()}")
    
    return rpn_outputs


if __name__ == "__main__":
    checkpoint_path = "test_checkpoints/checkpoint_step_200.pth"
    image_path = "/home/georgepearse/data/images/2024-04-11T10:13:35.128372706Z-53.jpg"
    
    debug_rpn(checkpoint_path, image_path)