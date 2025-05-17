"""Simple test to verify ONNX export functionality."""
import torch
from pathlib import Path
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN

# Create a model
model = SwinMaskRCNN(num_classes=69)

# Test ONNX export with dummy input
dummy_input = torch.randn(1, 3, 800, 800)
onnx_path = Path("test_model.onnx")

try:
    # Export backbone only (as full model is complex)
    torch.onnx.export(
        model.backbone,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Successfully exported model backbone to: {onnx_path}")
    
    # Also test saving raw weights
    weights_path = Path("test_weights.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"Successfully saved model weights to: {weights_path}")
    
    # Verify files exist
    if onnx_path.exists():
        print(f"ONNX file size: {onnx_path.stat().st_size / 1024 / 1024:.2f} MB")
    if weights_path.exists():
        print(f"Weights file size: {weights_path.stat().st_size / 1024 / 1024:.2f} MB")
        
except Exception as e:
    print(f"Export failed: {e}")
    import traceback
    traceback.print_exc()