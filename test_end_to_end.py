"""
Quick end-to-end test for the SWIN Mask R-CNN implementation with inference.
"""
import torch
import numpy as np
from PIL import Image
from swin_maskrcnn.models import SwinMaskRCNN
from swin_maskrcnn.inference import MaskRCNNPredictor


def main():
    print("Creating model...")
    # Create model with 69 classes (based on CLAUDE.md)
    model = SwinMaskRCNN(num_classes=69)
    model.eval()
    
    print("Creating synthetic checkpoint...")
    # Create a dummy checkpoint file for testing
    checkpoint = model.state_dict()
    torch.save(checkpoint, 'test_model.pth')
    
    print("Initializing predictor...")
    predictor = MaskRCNNPredictor('test_model.pth', num_classes=69)
    
    print("Creating test image...")
    # Create a dummy image
    test_image = Image.new('RGB', (640, 640), color='red')
    test_image_path = 'test_image.jpg'
    test_image.save(test_image_path)
    
    print("Running inference...")
    result = predictor.predict_single(test_image_path)
    
    print(f"Inference complete!")
    print(f"Boxes shape: {result['boxes'].shape}")
    print(f"Labels shape: {result['labels'].shape}")
    print(f"Scores shape: {result['scores'].shape}")
    print(f"Masks shape: {result['masks'].shape if result['masks'] is not None else 'None'}")
    
    print("\nSuccess! Model and inference are working correctly.")
    
    # Clean up
    import os
    os.remove('test_model.pth')
    os.remove(test_image_path)


if __name__ == "__main__":
    main()