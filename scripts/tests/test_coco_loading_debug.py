import torch
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.utils.load_coco_weights import load_coco_weights
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create model with CMR dataset classes
model = SwinMaskRCNN(num_classes=70)

# Load COCO weights
missing_keys, unexpected_keys = load_coco_weights(model, num_classes=70, strict=False)

logger.info(f"Number of missing keys: {len(missing_keys)}")
logger.info(f"Number of unexpected keys: {len(unexpected_keys)}")

if missing_keys:
    # Filter for important missing keys
    important_missing = []
    for key in missing_keys:
        # These are often initialized separately and are OK to be missing
        if any(x in key for x in ['mask_head.conv_logits', 'rpn.anchor_generator', 'num_batches_tracked']):
            continue
        important_missing.append(key)
    
    if important_missing:
        logger.warning("Important missing keys:")
        for key in important_missing[:10]:  # Show first 10
            logger.warning(f"  {key}")
    else:
        logger.info("All important weights loaded successfully")

if unexpected_keys:
    logger.warning("Unexpected keys (first 10):")
    for key in unexpected_keys[:10]:
        logger.warning(f"  {key}")

# Check specific loaded weights
def check_loaded_param(name, param):
    if param is not None:
        logger.info(f"{name} shape: {param.shape}, mean: {param.mean().item():.6f}, std: {param.std().item():.6f}")
    else:
        logger.warning(f"{name} is None")

logger.info("\nChecking classifier weights:")
check_loaded_param("fc_cls.weight", model.roi_head.fc_cls.weight)
check_loaded_param("fc_cls.bias", model.roi_head.fc_cls.bias)

logger.info("\nChecking regression weights:")  
check_loaded_param("fc_reg.weight", model.roi_head.fc_reg.weight)
check_loaded_param("fc_reg.bias", model.roi_head.fc_reg.bias)

# Check backbone
logger.info("\nChecking backbone patch embed:")
check_loaded_param("patch_embed.proj.weight", model.backbone.patch_embed.proj.weight)

# Test forward pass
model.eval()
test_input = torch.randn(1, 3, 224, 224)
targets = [{
    'boxes': torch.tensor([[10, 10, 100, 100]], dtype=torch.float32),
    'labels': torch.tensor([1], dtype=torch.int64)
}]

try:
    with torch.no_grad():
        outputs = model([test_input], targets)
    logger.info("Forward pass successful")
except Exception as e:
    logger.error(f"Forward pass failed: {e}")