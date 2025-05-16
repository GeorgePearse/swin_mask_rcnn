"""Test simple inference without the full pipeline."""
import torch
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from torchvision.transforms import Compose, ToTensor, Normalize


def simple_inference(checkpoint_path, image_path, num_classes=69):
    """Run simple inference test."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = SwinMaskRCNN(num_classes=num_classes)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    
    # Simple transform - just convert to tensor and normalize
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    print(f"Image tensor shape: {img_tensor.shape}")
    
    # First test - raw forward pass
    print("\n=== Testing raw forward pass ===")
    with torch.no_grad():
        outputs = model(img_tensor)
    
    print(f"Number of outputs: {len(outputs)}")
    for i, output in enumerate(outputs):
        print(f"\nImage {i} predictions:")
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={v.shape}")
                if v.numel() > 0:
                    print(f"    min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}")
                    if k == 'scores' and v.numel() > 0:
                        print(f"    Top 5 scores: {v[:5].tolist()}")
                        print(f"    Number of predictions: {len(v)}")
            else:
                print(f"  {k}: {v}")
    
    # Second test - predict method
    print("\n=== Testing predict method ===")
    predictions = model.predict(img_tensor, score_threshold=0.01, nms_threshold=0.5)
    
    for i, pred in enumerate(predictions):
        print(f"\nImage {i} filtered predictions:")
        for k, v in pred.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={v.shape}")
                if k == 'scores' and v.numel() > 0:
                    print(f"    Top 5 scores: {v[:5].tolist()}")
                    print(f"    Scores > 0.1: {(v > 0.1).sum()}")
                    print(f"    Scores > 0.5: {(v > 0.5).sum()}")
    
    return outputs, predictions


if __name__ == "__main__":
    checkpoint_path = "test_checkpoints/checkpoint_step_200.pth"
    image_path = "/home/georgepearse/data/images/2024-04-11T10:13:35.128372706Z-53.jpg"
    
    outputs, predictions = simple_inference(checkpoint_path, image_path)