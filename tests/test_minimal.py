"""
Minimal test to check if model runs with synthetic data.
"""
import torch
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN


def test_model_forward():
    """Test basic model forward pass with synthetic data."""
    # Create model
    model = SwinMaskRCNN(num_classes=10)
    model.eval()
    
    # Create synthetic data
    batch_size = 1
    img_size = 224
    images = torch.randn(batch_size, 3, img_size, img_size)
    
    # Create synthetic targets for training mode
    targets = []
    for i in range(batch_size):
        target = {
            'boxes': torch.tensor([[10, 10, 100, 100], [50, 50, 150, 150]], dtype=torch.float32),
            'labels': torch.tensor([1, 2], dtype=torch.int64),
            'masks': torch.randint(0, 2, (2, img_size, img_size), dtype=torch.uint8)
        }
        targets.append(target)
    
    # Test training mode
    model.train()
    print("Testing training mode...")
    losses = model(images, targets)
    print(f"Training losses: {losses.keys()}")
    for k, v in losses.items():
        if torch.is_tensor(v):
            print(f"  {k}: {v.item():.4f}")
        else:
            print(f"  {k}: {v}")
    
    # Test inference mode
    model.eval()
    print("\nTesting inference mode...")
    with torch.no_grad():
        outputs = model(images)
        print(f"Inference output type: {type(outputs)}")
        if isinstance(outputs, dict):
            print(f"Output keys: {outputs.keys()}")
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: shape {v.shape}")
                else:
                    print(f"  {k}: {type(v)}")
    
    print("\nModel test passed!")


if __name__ == '__main__':
    test_model_forward()