"""Create a correctly initialized checkpoint with proper bias values."""
import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.utils.load_coco_weights import load_coco_weights

def create_corrected_checkpoint():
    """Create checkpoint with corrected classification biases."""
    # Create model
    model = SwinMaskRCNN(num_classes=69)
    
    # Load COCO pretrained weights with proper initialization
    missing_keys, unexpected_keys = load_coco_weights(model, num_classes=69)
    print(f"Missing keys: {len(missing_keys)}")
    print(f"Unexpected keys: {len(unexpected_keys)}")
    
    # Fix the classification biases with new initialization
    cls_bias = model.roi_head.bbox_head.fc_cls.bias
    cls_bias.data[0] = -2.0  # Background bias negative to reduce false negatives
    cls_bias.data[1:] = 0.01  # Slightly positive object biases to encourage detections
    
    # Print some stats
    print(f"Background bias: {cls_bias.data[0].item():.4f}")
    print(f"Object bias mean: {cls_bias.data[1:].mean().item():.4f}")
    print(f"Object bias range: [{cls_bias.data[1:].min().item():.4f}, {cls_bias.data[1:].max().item():.4f}]")
    
    # Save corrected checkpoint
    output_path = "/home/georgepearse/swin_maskrcnn/checkpoints/coco_initialized_corrected_biases_v2.pth"
    torch.save(model.state_dict(), output_path)
    print(f"Saved corrected checkpoint to: {output_path}")

if __name__ == "__main__":
    create_corrected_checkpoint()