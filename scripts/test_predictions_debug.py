"""Quick test to check if model can make predictions."""
import torch
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.utils.load_coco_weights import load_coco_weights

# Create model
model = SwinMaskRCNN(num_classes=69)

# Load pretrained weights
print("Loading COCO pretrained weights...")
missing_keys, unexpected_keys = load_coco_weights(model, num_classes=69)
print(f"Missing keys: {len(missing_keys)}")
print(f"Unexpected keys: {len(unexpected_keys)}")

# Check bias initialization
if hasattr(model.roi_head.bbox_head, 'fc_cls'):
    cls_bias = model.roi_head.bbox_head.fc_cls.bias.detach().cpu().numpy()
    print(f"\nClassifier biases:")
    print(f"  Background bias: {cls_bias[0]:.4f}")
    print(f"  Object bias mean: {cls_bias[1:].mean():.4f}")
    print(f"  Object bias range: [{cls_bias[1:].min():.4f}, {cls_bias[1:].max():.4f}]")

# Test with a dummy image
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Create a dummy image
dummy_img = torch.randn(3, 800, 800).to(device)

with torch.no_grad():
    outputs = model([dummy_img])
    
if outputs and len(outputs) > 0:
    output = outputs[0]
    if 'boxes' in output:
        num_preds = len(output['boxes'])
        print(f"\nModel made {num_preds} predictions!")
        if num_preds > 0:
            print(f"Score range: [{output['scores'].min():.4f}, {output['scores'].max():.4f}]")
            print(f"Unique labels predicted: {output['labels'].unique().tolist()}")
    else:
        print("\nNo boxes in output!")
else:
    print("\nNo outputs from model!")