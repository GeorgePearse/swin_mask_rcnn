"""Example script for running inference with SWIN Mask R-CNN."""
from pathlib import Path
import numpy as np
from PIL import Image
import tempfile

from swin_maskrcnn import MaskRCNNPredictor, run_inference_pipeline


def create_dummy_image(size=(640, 640)):
    """Create a dummy test image."""
    # Create an image with some shapes
    img = Image.new('RGB', size, color='white')
    pixels = img.load()
    
    # Draw a red rectangle
    for x in range(100, 200):
        for y in range(100, 200):
            pixels[x, y] = (255, 0, 0)
    
    # Draw a blue circle
    center_x, center_y = 300, 300
    radius = 50
    for x in range(center_x - radius, center_x + radius):
        for y in range(center_y - radius, center_y + radius):
            if (x - center_x)**2 + (y - center_y)**2 <= radius**2:
                if 0 <= x < size[0] and 0 <= y < size[1]:
                    pixels[x, y] = (0, 0, 255)
    
    return img


def main():
    """Main example function."""
    # Create a dummy model checkpoint
    print("Creating dummy model checkpoint...")
    from swin_maskrcnn import SwinMaskRCNN
    import torch
    
    model = SwinMaskRCNN(num_classes=10)
    model_path = "dummy_model.pth"
    torch.save(model.state_dict(), model_path)
    
    # Create dummy test images
    print("Creating test images...")
    test_images = []
    for i in range(3):
        img = create_dummy_image()
        img_path = f"test_image_{i}.jpg"
        img.save(img_path)
        test_images.append(img_path)
    
    print(f"Created {len(test_images)} test images")
    
    # Run inference
    print("\nRunning inference...")
    try:
        results = run_inference_pipeline(
            model_path=model_path,
            image_paths=test_images,
            num_classes=10,
            score_threshold=0.3,
            nms_threshold=0.5,
            visualize=True,
            dataset_name="example_predictions"
        )
        
        # Print results
        print("\nInference Results:")
        for i, (path, preds) in enumerate(zip(results['image_paths'], results['predictions'])):
            print(f"\nImage {i+1}: {path}")
            print(f"  Number of detections: {len(preds['boxes'])}")
            if len(preds['boxes']) > 0:
                print(f"  Confidence scores: {preds['scores']}")
                print(f"  Predicted classes: {preds['labels']}")
        
        print("\nFiftyOne visualization is available at http://localhost:5151")
        print("Press Ctrl+C to exit")
        
        try:
            input("Press Enter to exit...")
        except KeyboardInterrupt:
            pass
            
    finally:
        # Clean up
        print("\nCleaning up...")
        Path(model_path).unlink(missing_ok=True)
        for img_path in test_images:
            Path(img_path).unlink(missing_ok=True)
        
        # Clean up FiftyOne dataset
        import fiftyone as fo
        try:
            dataset = fo.load_dataset("example_predictions")
            dataset.delete()
        except:
            pass


if __name__ == "__main__":
    main()