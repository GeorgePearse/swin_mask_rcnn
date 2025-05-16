"""Test raw predictions from model."""
import torch
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.data.transforms_simple import get_inference_transform


def test_raw_predictions(checkpoint_path, image_path, num_classes=69):
    """Test raw model predictions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = SwinMaskRCNN(num_classes=num_classes)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transform = get_inference_transform()
    transformed = transform(image=image)
    img_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Get raw predictions
    with torch.no_grad():
        outputs = model(img_tensor)
    
    print(f"Raw model outputs:")
    for i, output in enumerate(outputs):
        print(f"\nImage {i}:")
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                if v.numel() > 0:
                    print(f"    min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}")
                    if k == 'scores':
                        # Show score distribution
                        print(f"    Scores > 0.1: {(v > 0.1).sum()}")
                        print(f"    Scores > 0.5: {(v > 0.5).sum()}")
                        print(f"    Top 5 scores: {v[:5].tolist()}")
            else:
                print(f"  {k}: {v}")
    
    return outputs


if __name__ == "__main__":
    checkpoint_path = "test_checkpoints/checkpoint_step_200.pth"
    image_path = "/home/georgepearse/data/images/2024-04-11T10:13:35.128372706Z-53.jpg"
    
    outputs = test_raw_predictions(checkpoint_path, image_path)