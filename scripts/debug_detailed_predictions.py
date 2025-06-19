"""Debug predictions in detail."""
import torch
import torch.nn.functional as F
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.utils.load_coco_weights import load_coco_weights

# Create model
model = SwinMaskRCNN(num_classes=69)

# Load pretrained weights
print("Loading COCO pretrained weights...")
missing_keys, unexpected_keys = load_coco_weights(model, num_classes=69)
print(f"Missing keys: {len(missing_keys)}")

# Check bias initialization
if hasattr(model.roi_head.bbox_head, 'fc_cls'):
    cls_bias = model.roi_head.bbox_head.fc_cls.bias.detach().cpu().numpy()
    print(f"\nClassifier biases after loading:")
    print(f"  Background bias: {cls_bias[0]:.4f}")
    print(f"  Object bias mean: {cls_bias[1:].mean():.4f}")
    print(f"  Object bias range: [{cls_bias[1:].min():.4f}, {cls_bias[1:].max():.4f}]")

# Test with a dummy image
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Create a dummy batch
dummy_imgs = [torch.randn(3, 800, 800).to(device) for _ in range(2)]

# Hook to capture intermediate outputs
roi_outputs = []
def roi_hook(module, input, output):
    if isinstance(output, tuple) and len(output) == 2:  # cls_score, bbox_pred
        roi_outputs.append({
            'cls_scores': output[0].detach().cpu(),
            'bbox_preds': output[1].detach().cpu()
        })

# Register hook
hook = model.roi_head.bbox_head.register_forward_hook(roi_hook)

with torch.no_grad():
    outputs = model(dummy_imgs)
    
print(f"\n=== Batch Predictions ===")
for i, output in enumerate(outputs):
    if 'boxes' in output:
        num_preds = len(output['boxes'])
        print(f"\nImage {i}: {num_preds} predictions")
        if num_preds > 0:
            scores = output['scores']
            labels = output['labels']
            print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
            print(f"  Score mean: {scores.mean():.4f}")
            print(f"  Unique labels: {labels.unique().tolist()}")
            
            # Show top 5 predictions
            top_k = min(5, num_preds)
            top_scores, top_indices = scores.topk(top_k)
            print(f"  Top {top_k} predictions:")
            for j, (score, idx) in enumerate(zip(top_scores, top_indices)):
                label = labels[idx]
                print(f"    {j+1}. Label: {label}, Score: {score:.4f}")

# Analyze ROI head outputs
if roi_outputs:
    print(f"\n=== ROI Head Analysis ===")
    print(f"Number of ROI forward passes: {len(roi_outputs)}")
    
    for i, roi_out in enumerate(roi_outputs[:3]):  # Show first 3
        cls_scores = roi_out['cls_scores']
        print(f"\nROI batch {i}:")
        print(f"  Classification scores shape: {cls_scores.shape}")
        
        # Apply softmax to get probabilities
        cls_probs = F.softmax(cls_scores, dim=1)
        
        # Background vs foreground analysis
        bg_probs = cls_probs[:, 0]
        fg_probs = cls_probs[:, 1:]
        
        print(f"  Background probability - mean: {bg_probs.mean():.4f}, min: {bg_probs.min():.4f}, max: {bg_probs.max():.4f}")
        print(f"  Best foreground probability per ROI:")
        for j in range(min(5, len(cls_probs))):
            best_fg_prob = fg_probs[j].max()
            best_fg_class = fg_probs[j].argmax() + 1  # +1 for background offset
            print(f"    ROI {j}: class {best_fg_class} with prob {best_fg_prob:.4f}")

# Remove hook
hook.remove()