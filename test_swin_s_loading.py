#!/usr/bin/env python
"""Test loading pretrained Swin-S weights."""
import torch
from swin_maskrcnn.models.swin import SwinTransformer

def test_swin_s_loading():
    # Create model with Swin-S configuration
    model = SwinTransformer(
        embed_dims=96,
        depths=[2, 2, 18, 2],  # Swin-S configuration
        num_heads=[3, 6, 12, 24],
        window_size=7,
        out_indices=(0, 1, 2, 3),
    )
    
    # Create a dummy checkpoint with keys matching the pretrained model
    # Note: In real usage, you'd load an actual pretrained checkpoint
    checkpoint = {
        'state_dict': {
            'patch_embed.proj.weight': torch.randn(96, 3, 4, 4),
            'patch_embed.proj.bias': torch.randn(96),
            'patch_embed.norm.weight': torch.randn(96),
            'patch_embed.norm.bias': torch.randn(96),
        }
    }
    
    # Add layer weights for all 24 blocks (2 + 2 + 18 + 2)
    # Layer 0: 2 blocks
    for i in range(2):
        prefix = f'layers.0.blocks.{i}'
        checkpoint['state_dict'][f'{prefix}.norm1.weight'] = torch.randn(96)
        checkpoint['state_dict'][f'{prefix}.norm1.bias'] = torch.randn(96)
        checkpoint['state_dict'][f'{prefix}.attn.qkv.weight'] = torch.randn(288, 96)
        checkpoint['state_dict'][f'{prefix}.attn.qkv.bias'] = torch.randn(288)
        checkpoint['state_dict'][f'{prefix}.attn.proj.weight'] = torch.randn(96, 96)
        checkpoint['state_dict'][f'{prefix}.attn.proj.bias'] = torch.randn(96)
        # Add more weights as needed...
    
    # Layer 1: 2 blocks
    for i in range(2):
        prefix = f'layers.1.blocks.{i}'
        checkpoint['state_dict'][f'{prefix}.norm1.weight'] = torch.randn(192)
        checkpoint['state_dict'][f'{prefix}.norm1.bias'] = torch.randn(192)
        # Add more weights as needed...
    
    # Layer 2: 18 blocks (Swin-S has 18 blocks here)
    for i in range(18):
        prefix = f'layers.2.blocks.{i}'
        checkpoint['state_dict'][f'{prefix}.norm1.weight'] = torch.randn(384)
        checkpoint['state_dict'][f'{prefix}.norm1.bias'] = torch.randn(384)
        # Add more weights as needed...
    
    # Layer 3: 2 blocks
    for i in range(2):
        prefix = f'layers.3.blocks.{i}'
        checkpoint['state_dict'][f'{prefix}.norm1.weight'] = torch.randn(768)
        checkpoint['state_dict'][f'{prefix}.norm1.bias'] = torch.randn(768)
        # Add more weights as needed...
    
    # Filter backbone weights
    backbone_state_dict = {}
    for k, v in checkpoint['state_dict'].items():
            if k.startswith('backbone.'):
                backbone_key = k.replace('backbone.', '')
                backbone_state_dict[backbone_key] = v
            elif not k.startswith(('head.', 'fc.', 'classifier.')):
                # Direct mapping for patch_embed and layer weights
                backbone_state_dict[k] = v
    
    # Load weights
    msg = model.load_state_dict(backbone_state_dict, strict=False)
    print(f"Missing keys: {len(msg.missing_keys)}")
    print(f"Unexpected keys: {len(msg.unexpected_keys)}")
    
    # Show first few missing/unexpected keys
    if msg.missing_keys:
        print(f"\nFirst 5 missing keys: {msg.missing_keys[:5]}")
    if msg.unexpected_keys:
        print(f"\nFirst 5 unexpected keys: {msg.unexpected_keys[:5]}")
    
    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    outputs = model(x)
    print(f"\nOutput shapes: {[o.shape for o in outputs]}")
    
    print("\nSwin-S model loaded successfully!")

if __name__ == "__main__":
    test_swin_s_loading()