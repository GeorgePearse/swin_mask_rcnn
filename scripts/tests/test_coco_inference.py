import torch
import torch.nn as nn
from typing import List, Dict, Any
import numpy as np
from pathlib import Path
import urllib.request

from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.utils.logging import get_logger, setup_logger
from swin_maskrcnn.data.dataset import CocoDataset
from swin_maskrcnn.data.transforms_simple import get_transform_simple

# Setup logging
setup_logger(name=__name__, level="DEBUG")
logger = get_logger(__name__)


def download_coco_checkpoint():
    """Download the COCO pretrained checkpoint."""
    # URL for SWIN Mask R-CNN checkpoint
    url = "https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth"
    
    cache_dir = Path.home() / ".cache" / "swin_maskrcnn"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    filename = url.split("/")[-1]
    cache_path = cache_dir / filename
    
    if not cache_path.exists():
        logger.info(f"Downloading checkpoint from {url}")
        urllib.request.urlretrieve(url, cache_path)
        logger.info(f"Saved to {cache_path}")
    else:
        logger.info(f"Using cached checkpoint from {cache_path}")
    
    return cache_path


def convert_coco_weights(state_dict):
    """Convert COCO weights to our model format."""
    converted = {}
    
    # Mapping between mmdet and our naming
    mappings = {
        'backbone.patch_embed.projection': 'backbone.patch_embed.proj',
        'backbone.stages': 'backbone.layers',
        'backbone.norm': 'backbone.norm_layer',
        'w_msa': 'attn',
        'ffn': 'mlp',
        'roi_head.bbox_head': 'roi_head',
    }
    
    for key, value in state_dict.items():
        new_key = key
        
        # Apply mappings
        for old, new in mappings.items():
            new_key = new_key.replace(old, new)
        
        # Special handling for specific layers
        if 'relative_position_bias_table' in new_key:
            new_key = new_key.replace('attn.attn', 'attn')
        
        converted[new_key] = value
    
    return converted


def analyze_model_with_coco_weights(model, device):
    """Analyze model after loading COCO weights."""
    model.eval()
    
    logger.info("=== Model Analysis with COCO Weights ===")
    
    # Check classifier bias
    if hasattr(model.roi_head, 'fc_cls'):
        cls_bias = model.roi_head.fc_cls.bias.detach().cpu().numpy()
        logger.info(f"Classifier bias shape: {cls_bias.shape}")
        logger.info(f"Background bias: {cls_bias[0]:.4f}")
        logger.info(f"Object bias mean: {cls_bias[1:].mean():.4f}")
        logger.info(f"Object bias range: [{cls_bias[1:].min():.4f}, {cls_bias[1:].max():.4f}]")
    
    # Test on dummy input
    with torch.no_grad():
        dummy_image = torch.randn(1, 3, 800, 800).to(device)
        outputs = model(dummy_image, targets=None)
        
        if outputs and len(outputs[0]) > 0:
            output = outputs[0]
            if 'scores' in output:
                scores = output['scores']
                labels = output['labels']
                logger.info(f"Number of predictions: {len(scores)}")
                if len(scores) > 0:
                    logger.info(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
                    logger.info(f"Unique labels: {torch.unique(labels).tolist()}")
                    logger.info(f"Top 5 scores: {scores[:5].tolist()}")
                    logger.info(f"Top 5 labels: {labels[:5].tolist()}")
                else:
                    logger.info("No predictions above threshold")
        else:
            logger.info("No predictions")


def test_on_real_data(model, device):
    """Test model on real validation data."""
    val_ann_path = "/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:38.270134_val_ordered.json"
    val_dataset = CocoDataset(
        root="/home/georgepearse/data/images",
        annFile=val_ann_path,
        transform=get_transform_simple(train=False),
        remove_images_without_annotations=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=val_dataset.collate_fn
    )
    
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 2:  # Test just a few batches
                break
            
            images = batch['images'].to(device)
            
            outputs = model(images, targets=None)
            
            logger.info(f"\nBatch {i}:")
            for j, output in enumerate(outputs):
                if 'scores' in output:
                    scores = output['scores']
                    labels = output['labels']
                    logger.info(f"  Image {j}: {len(scores)} predictions")
                    if len(scores) > 0:
                        logger.info(f"    Score range: [{scores.min():.4f}, {scores.max():.4f}]")
                        logger.info(f"    Top 5 scores: {scores[:5].tolist()}")
                        logger.info(f"    Top 5 labels: {labels[:5].tolist()}")
                else:
                    logger.info(f"  Image {j}: No predictions")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model for COCO (81 classes including background)
    model = SwinMaskRCNN(num_classes=81).to(device)
    
    # Download and load COCO checkpoint
    coco_path = download_coco_checkpoint()
    checkpoint = torch.load(coco_path, map_location=device)
    state_dict = checkpoint['state_dict']
    
    # Convert weights
    converted_state_dict = convert_coco_weights(state_dict)
    
    # Special handling for mask head - COCO has 80 classes for masks (no background)
    # Remove mask head weights from conversion
    mask_keys_to_remove = [k for k in converted_state_dict.keys() if 'mask_head.conv_logits' in k]
    for key in mask_keys_to_remove:
        del converted_state_dict[key]
    
    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(converted_state_dict, strict=False)
    logger.info(f"Missing keys: {len(missing_keys)}")
    logger.info(f"Unexpected keys: {len(unexpected_keys)}")
    
    if missing_keys:
        logger.info(f"First 10 missing keys: {missing_keys[:10]}")
    
    # Initialize mask head properly
    if hasattr(model.roi_head, 'mask_head'):
        # The bias initialization is important for predictions
        nn.init.normal_(model.roi_head.mask_head.conv_logits.weight, mean=0, std=0.001)
        nn.init.constant_(model.roi_head.mask_head.conv_logits.bias, 0)
    
    # Analyze model
    analyze_model_with_coco_weights(model, device)
    
    # Test on real data
    logger.info("\n=== Testing on Real Data ===")
    test_on_real_data(model, device)


if __name__ == "__main__":
    main()