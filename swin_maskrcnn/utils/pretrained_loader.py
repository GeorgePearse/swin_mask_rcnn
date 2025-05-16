"""Utilities for loading pretrained weights from mmdetection format."""
import torch
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict, Optional


def download_checkpoint(url: str, cache_dir: Optional[Path] = None) -> Path:
    """Download checkpoint from URL to local cache."""
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "swin_maskrcnn"
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract filename from URL
    filename = url.split("/")[-1]
    cache_path = cache_dir / filename
    
    # Download if not already cached
    if not cache_path.exists():
        print(f"Downloading checkpoint from {url}")
        urllib.request.urlretrieve(url, cache_path)
        print(f"Saved to {cache_path}")
    else:
        print(f"Using cached checkpoint from {cache_path}")
    
    return cache_path


def convert_mmdet_to_swin(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    """Convert mmdetection checkpoint to our SWIN format.
    
    mmdetection format uses different key names:
    - 'state_dict' contains the model weights
    - Keys are prefixed with 'backbone.', 'neck.', etc.
    - Some layer names may differ
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Convert keys to match our implementation
    converted_state_dict = {}
    
    for key, value in state_dict.items():
        # Skip non-backbone weights for now
        if not key.startswith('backbone.'):
            continue
            
        # Remove 'backbone.' prefix
        new_key = key.replace('backbone.', '')
        
        # Key mappings from mmdetection to our implementation
        key_mappings = {
            # Patch embedding
            'patch_embed.projection': 'patch_embed.proj',
            # Stages to layers
            'stages': 'layers',
            # Attention module names
            'attn.w_msa': 'attn',
            # FFN to MLP
            'ffn.layers.0.0': 'mlp.0',  # First linear
            'ffn.layers.1': 'mlp.3',    # Second linear
            # Downsample names
            'downsample.reduction': 'downsample.reduction',
            'downsample.norm': 'downsample.norm',
        }
        
        # Apply mappings
        for old_pattern, new_pattern in key_mappings.items():
            if old_pattern in new_key:
                new_key = new_key.replace(old_pattern, new_pattern)
        
        # Handle norm layers (norm0, norm1, norm2, norm3 -> norms.0, norms.1, etc.)
        if new_key.startswith('norm') and new_key[4:].split('.')[0].isdigit():
            parts = new_key.split('.')
            norm_idx = parts[0][4]  # Get the digit after 'norm'
            suffix = '.'.join(parts[1:]) if len(parts) > 1 else ''
            new_key = f'norms.{norm_idx}'
            if suffix:
                new_key += f'.{suffix}'
        
        converted_state_dict[new_key] = value
    
    return converted_state_dict


def load_pretrained_from_url(model: torch.nn.Module, url: str, strict: bool = False) -> None:
    """Load pretrained weights from URL into model.
    
    Args:
        model: The model to load weights into
        url: URL to download checkpoint from
        strict: Whether to enforce strict key matching
    """
    # Download checkpoint
    checkpoint_path = download_checkpoint(url)
    
    # Convert to our format
    converted_state_dict = convert_mmdet_to_swin(checkpoint_path)
    
    # Filter for backbone weights only
    backbone_state_dict = {
        k: v for k, v in converted_state_dict.items()
        if not k.startswith(('neck.', 'rpn_head.', 'roi_head.'))
    }
    
    # Load into backbone
    missing_keys, unexpected_keys = model.backbone.load_state_dict(
        backbone_state_dict, strict=strict
    )
    
    if missing_keys:
        print(f"Missing keys in backbone: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys in backbone: {unexpected_keys}")
    
    print(f"Successfully loaded pretrained backbone weights from {url}")