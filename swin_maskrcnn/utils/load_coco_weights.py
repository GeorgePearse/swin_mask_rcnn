"""Load COCO pre-trained weights for SWIN Mask R-CNN."""
import torch
import torch.nn as nn
from pathlib import Path
import urllib.request
from typing import Dict, Tuple


def download_coco_checkpoint():
    """Download the COCO pretrained checkpoint."""
    # URL for SWIN Mask R-CNN checkpoint
    url = "https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth"
    
    cache_dir = Path.home() / ".cache" / "swin_maskrcnn"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    filename = url.split("/")[-1]
    cache_path = cache_dir / filename
    
    if not cache_path.exists():
        print(f"Downloading checkpoint from {url}")
        urllib.request.urlretrieve(url, cache_path)
        print(f"Saved to {cache_path}")
    else:
        print(f"Using cached checkpoint from {cache_path}")
    
    return cache_path


def convert_coco_weights_to_swin(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert COCO weights to our SWIN model format."""
    converted = {}
    
    for key, value in state_dict.items():
        new_key = key
        
        # Backbone mappings
        new_key = new_key.replace('backbone.patch_embed.projection', 'backbone.patch_embed.proj')
        new_key = new_key.replace('backbone.stages', 'backbone.layers')
        new_key = new_key.replace('w_msa', 'attn') 
        new_key = new_key.replace('ffn', 'mlp')
        new_key = new_key.replace('ln1', 'norm1')
        new_key = new_key.replace('ln2', 'norm2')
        
        # Handle backbone norm layers (norm0 -> norms.0, etc.)
        if 'backbone.norm' in new_key and len(new_key) > 13 and new_key[13].isdigit():
            # Extract the digit and convert norm0 -> norms.0
            digit = new_key[13]  # Position after 'backbone.norm'
            new_key = new_key.replace(f'backbone.norm{digit}', f'backbone.norms.{digit}')
        
        # Handle MLP layer flattening 
        # Convert mlp.layers.0.0.weight -> mlp.0.weight
        # Convert mlp.layers.1.weight -> mlp.3.weight  
        if 'mlp.layers.' in new_key:
            # Extract the layer indices
            parts = new_key.split('.')
            mlp_idx = parts.index('mlp')
            if mlp_idx + 3 < len(parts) and parts[mlp_idx + 1] == 'layers':
                layer_idx = parts[mlp_idx + 2]
                sublayer_idx = parts[mlp_idx + 3]
                
                # Check if sublayer_idx is a digit (for nested structure) or parameter name
                if sublayer_idx.isdigit():
                    # Handle nested structure: layers.0.0.weight
                    if layer_idx == '0' and sublayer_idx == '0':
                        new_idx = '0'
                    elif layer_idx == '1':
                        new_idx = '3'
                    else:
                        new_idx = sublayer_idx
                    # Reconstruct: mlp.new_idx.weight
                    new_parts = parts[:mlp_idx + 1] + [new_idx] + parts[mlp_idx + 4:]
                else:
                    # Handle direct structure: layers.1.weight 
                    if layer_idx == '0':
                        new_idx = '0'
                    elif layer_idx == '1':
                        new_idx = '3'
                    else:
                        new_idx = layer_idx
                    # Reconstruct: mlp.new_idx.weight
                    new_parts = parts[:mlp_idx + 1] + [new_idx] + parts[mlp_idx + 3:]
                
                new_key = '.'.join(new_parts)
        
        # RPN mappings - rpn_head in model, rpn in checkpoint
        if new_key.startswith('rpn.'):
            new_key = new_key.replace('rpn.', 'rpn_head.')
        
        # ROI Head mappings - need to add bbox_head
        if new_key.startswith('roi_head.'):
            # fc_cls and fc_reg go under bbox_head
            if 'roi_head.fc_cls' in new_key or 'roi_head.fc_reg' in new_key:
                new_key = new_key.replace('roi_head.fc_', 'roi_head.bbox_head.fc_')
            elif 'roi_head.bbox_head.shared_fcs' in new_key:
                # Map bbox_head.shared_fcs.0 -> bbox_head.shared_fc1
                if 'shared_fcs.0' in new_key:
                    new_key = new_key.replace('shared_fcs.0', 'shared_fc1')
                elif 'shared_fcs.1' in new_key:
                    new_key = new_key.replace('shared_fcs.1', 'shared_fc2')
        
        # Handle attention double nesting
        if 'attn.attn.' in new_key:
            new_key = new_key.replace('attn.attn.', 'attn.')
        
        # Handle conv wrappers in FPN and mask head
        if '.conv.' in new_key and ('lateral_convs' in new_key or 'fpn_convs' in new_key):
            new_key = new_key.replace('.conv.', '.')
        if 'mask_head.convs.' in new_key and '.conv.' in new_key:
            new_key = new_key.replace('.conv.', '.')
        
        # Handle mask deconv
        if 'mask_head.conv_naive' in new_key:
            new_key = new_key.replace('conv_naive', 'deconv')
            
        # Skip mask head logits if present (different number of classes)
        if 'mask_head.conv_logits' in new_key:
            continue
            
        converted[new_key] = value
    
    return converted


def load_coco_weights(model: nn.Module, num_classes: int = 70, strict: bool = False) -> Tuple[list, list]:
    """Load COCO pretrained weights into model.
    
    Args:
        model: SWIN Mask R-CNN model
        num_classes: Target number of classes (including background)
        strict: Whether to require all keys to match
    
    Returns:
        missing_keys: List of keys not found in checkpoint
        unexpected_keys: List of keys not found in model
    """
    # Download checkpoint if needed
    checkpoint_path = download_coco_checkpoint()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Convert keys
    converted_state_dict = convert_coco_weights_to_swin(state_dict)
    
    # Handle classifier layer - resize if needed
    if num_classes != 81:  # COCO has 81 classes
        # Classification layer - now under bbox_head
        if 'roi_head.bbox_head.fc_cls.weight' in converted_state_dict:
            coco_cls_weight = converted_state_dict['roi_head.bbox_head.fc_cls.weight']
            coco_cls_bias = converted_state_dict['roi_head.bbox_head.fc_cls.bias']
            
            # Initialize new classifier weights (num_classes + 1 for background)
            new_cls_weight = torch.randn(num_classes + 1, coco_cls_weight.shape[1]) * 0.001
            new_cls_bias = torch.zeros(num_classes + 1)
            
            # Copy COCO weights for background class
            new_cls_weight[0] = coco_cls_weight[0]
            new_cls_bias[0] = coco_cls_bias[0]
            
            # For remaining classes, use random initialization but with similar scale
            if num_classes > 0:
                nn.init.normal_(new_cls_weight[1:], mean=0, std=0.001)
                nn.init.constant_(new_cls_bias[1:], 0)
            
            converted_state_dict['roi_head.bbox_head.fc_cls.weight'] = new_cls_weight
            converted_state_dict['roi_head.bbox_head.fc_cls.bias'] = new_cls_bias
        
        # Bbox regression layer - also needs resizing
        if 'roi_head.bbox_head.fc_reg.weight' in converted_state_dict:
            coco_reg_weight = converted_state_dict['roi_head.bbox_head.fc_reg.weight']
            coco_reg_bias = converted_state_dict['roi_head.bbox_head.fc_reg.bias']
            
            # Initialize new regression weights (num_classes * 4, without background)
            new_reg_weight = torch.randn(num_classes * 4, coco_reg_weight.shape[1]) * 0.001
            new_reg_bias = torch.zeros(num_classes * 4)
            
            # Copy first set of regression weights 
            new_reg_weight[:4] = coco_reg_weight[:4]
            new_reg_bias[:4] = coco_reg_bias[:4]
            
            converted_state_dict['roi_head.bbox_head.fc_reg.weight'] = new_reg_weight
            converted_state_dict['roi_head.bbox_head.fc_reg.bias'] = new_reg_bias
    
    # Load converted weights
    missing_keys, unexpected_keys = model.load_state_dict(converted_state_dict, strict=strict)
    
    # Initialize mask head if needed
    if hasattr(model.roi_head, 'mask_head'):
        nn.init.normal_(model.roi_head.mask_head.conv_logits.weight, mean=0, std=0.001)
        nn.init.constant_(model.roi_head.mask_head.conv_logits.bias, 0)
    
    return missing_keys, unexpected_keys