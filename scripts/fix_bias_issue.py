"""Script to debug and fix the bias issue causing all detections to be classified as background."""
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.utils.load_coco_weights import load_coco_weights


def diagnose_biases():
    """Diagnose the bias issue."""
    
    # Create model
    model = SwinMaskRCNN(num_classes=69)
    
    # Check biases before loading weights
    print("=== Initial biases (before loading weights) ===")
    fc_cls = model.roi_head.bbox_head.fc_cls
    bias = fc_cls.bias.detach().cpu().numpy()
    print(f"Background bias: {bias[0]:.4f}")
    print(f"Object bias mean: {bias[1:].mean():.4f}")
    print(f"Object bias range: [{bias[1:].min():.4f}, {bias[1:].max():.4f}]")
    
    # Simulate what happens with these biases
    fake_logits = torch.randn(100, 70)  # 100 proposals, 70 classes
    # Add our biases
    fake_logits = fake_logits + fc_cls.bias
    
    probs = F.softmax(fake_logits, dim=1)
    print(f"\nSimulated background probability: mean={probs[:, 0].mean():.4f}")
    print(f"Simulated foreground probability: mean={probs[:, 1:].mean():.4f}")
    
    # Load COCO weights
    print("\n=== After loading COCO weights ===")
    missing_keys, unexpected_keys = load_coco_weights(model, num_classes=69)
    
    bias = model.roi_head.bbox_head.fc_cls.bias.detach().cpu().numpy()
    print(f"Background bias: {bias[0]:.4f}")
    print(f"Object bias mean: {bias[1:].mean():.4f}")
    print(f"Object bias range: [{bias[1:].min():.4f}, {bias[1:].max():.4f}]")
    
    # Simulate what happens with these biases
    fake_logits = torch.randn(100, 70)  # 100 proposals, 70 classes
    # Add our biases
    fake_logits = fake_logits + model.roi_head.bbox_head.fc_cls.bias
    
    probs = F.softmax(fake_logits, dim=1)
    print(f"\nSimulated background probability: mean={probs[:, 0].mean():.4f}")
    print(f"Simulated foreground probability: mean={probs[:, 1:].mean():.4f}")
    
    # Check if the issue is with weight initialization too
    weight = fc_cls.weight.detach().cpu().numpy()
    print(f"\nClassifier weight stats:")
    print(f"  Weight shape: {weight.shape}")
    print(f"  Weight mean: {weight.mean():.4f}")
    print(f"  Weight std: {weight.std():.4f}")
    print(f"  Background weight norm: {np.linalg.norm(weight[0]):.4f}")
    print(f"  Object weight norm mean: {np.linalg.norm(weight[1:], axis=1).mean():.4f}")


def test_bias_fix():
    """Test a fix for the bias issue."""
    print("\n=== Testing bias fix ===")
    
    model = SwinMaskRCNN(num_classes=69)
    load_coco_weights(model, num_classes=69)
    
    # Fix the biases manually
    fc_cls = model.roi_head.bbox_head.fc_cls
    
    # Use more balanced bias initialization
    # Instead of extreme focal loss initialization, use more moderate values
    prior_prob = 0.01  # Similar to focal loss but with different calculation
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))  # ~-4.595
    
    with torch.no_grad():
        # Set background bias slightly positive to reduce false positives
        fc_cls.bias[0] = 0.0  # Neutral background bias
        # Set foreground biases negative but not too extreme
        fc_cls.bias[1:] = -2.0  # Moderate negative bias for foreground
    
    print(f"Fixed background bias: {fc_cls.bias[0].item():.4f}")
    print(f"Fixed object bias: {fc_cls.bias[1].item():.4f}")
    
    # Test with fake logits
    fake_logits = torch.randn(100, 70)
    fake_logits = fake_logits + fc_cls.bias
    
    probs = F.softmax(fake_logits, dim=1)
    scores, preds = probs.max(dim=1)
    
    print(f"\nAfter fix:")
    print(f"  Background probability: mean={probs[:, 0].mean():.4f}")
    print(f"  Foreground probability: mean={probs[:, 1:].mean():.4f}")
    print(f"  Predicted as foreground: {(preds > 0).sum()} / 100")
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    
    # Save fixed checkpoint
    checkpoint_path = "fixed_biases_checkpoint.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nSaved fixed model to {checkpoint_path}")


if __name__ == "__main__":
    print("=== Diagnosing bias issue ===")
    diagnose_biases()
    
    print("\n" + "="*50 + "\n")
    test_bias_fix()