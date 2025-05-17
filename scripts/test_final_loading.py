"""Test final weight loading and inference."""
import torch
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.utils.load_coco_weights import load_coco_weights

# Create model with 69 classes
model = SwinMaskRCNN(num_classes=69)

print("Loading COCO pretrained weights...")
missing_keys, unexpected_keys = load_coco_weights(model, num_classes=69)

print(f"\nMissing keys: {len(missing_keys)}")
print(f"Unexpected keys: {len(unexpected_keys)}")

# Categorize the remaining issues
if missing_keys:
    missing_by_component = {}
    for key in missing_keys:
        component = key.split('.')[0]
        if component not in missing_by_component:
            missing_by_component[component] = []
        missing_by_component[component].append(key)
    
    print("\nRemaining missing keys by component:")
    for component, keys in missing_by_component.items():
        print(f"  {component}: {len(keys)} keys")
        if len(keys) <= 5:
            for key in keys:
                print(f"    - {key}")

if unexpected_keys:
    unexpected_by_component = {}
    for key in unexpected_keys:
        component = key.split('.')[0]
        if component not in unexpected_by_component:
            unexpected_by_component[component] = []
        unexpected_by_component[component].append(key)
    
    print("\nRemaining unexpected keys by component:")
    for component, keys in unexpected_by_component.items():
        print(f"  {component}: {len(keys)} keys")
        if len(keys) <= 5:
            for key in keys:
                print(f"    - {key}")

# Test forward pass
print("\nTesting forward pass...")
try:
    dummy_input = torch.randn(1, 3, 640, 640)
    model.eval()
    with torch.no_grad():
        outputs = model(dummy_input)
    print("Forward pass successful!")
    if isinstance(outputs, list) and len(outputs) > 0:
        print(f"Output type: {type(outputs[0])}")
        if hasattr(outputs[0], 'keys'):
            print(f"Output keys: {outputs[0].keys()}")
except Exception as e:
    print(f"Forward pass failed: {e}")
    import traceback
    traceback.print_exc()

# Test on GPU if available
if torch.cuda.is_available():
    print("\nTesting on GPU...")
    model = model.cuda()
    dummy_input = dummy_input.cuda()
    try:
        with torch.no_grad():
            outputs = model(dummy_input)
        print("GPU forward pass successful!")
    except Exception as e:
        print(f"GPU forward pass failed: {e}")