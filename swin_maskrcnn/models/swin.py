"""
Simplified SWIN Transformer implementation without mm* dependencies.
"""
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.randn(shape, dtype=x.dtype, device=x.device) < keep_prob
        return x * mask.float() / keep_prob


class WindowMSA(nn.Module):
    """Window based multi-head self-attention (W-MSA) module with relative position bias."""
    
    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        window_size: Tuple[int, int],
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop_rate: float = 0.,
        proj_drop_rate: float = 0.
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims ** -0.5
        
        # Define relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        
        # Create relative position index
        Wh, Ww = self.window_size
        coords_h = torch.arange(Wh)
        coords_w = torch.arange(Ww)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += Wh - 1
        relative_coords[:, :, 1] += Ww - 1
        relative_coords[:, :, 0] *= 2 * Ww - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        
    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size: window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    """SWIN Transformer Block."""
    
    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = norm_layer(embed_dims)
        self.attn = WindowMSA(
            embed_dims, num_heads, window_size=(window_size, window_size),
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop_rate=attn_drop,
            proj_drop_rate=drop
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(embed_dims)
        mlp_hidden_dim = int(embed_dims * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dims, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, embed_dims),
            nn.Dropout(drop)
        )
        
    def forward(self, x):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size: L={L}, H={H}, W={W}, H*W={H*W}"
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = H + pad_b, W + pad_r
        
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            
            # calculate attention mask
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=x.device)
            h_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            shifted_x = x
            attn_mask = None
        
        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)
        
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H_pad, W_pad)
        
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        # remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        
        x = x.view(B, H * W, C)
        
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage."""
    
    def __init__(
        self,
        embed_dims: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        norm_layer=nn.LayerNorm,
        downsample=None
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.depth = depth
        
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            )
            for i in range(depth)])
        
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(embed_dims, norm_layer=norm_layer)
        else:
            self.downsample = None
            
    def forward(self, x, H, W):
        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x)
        
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchMerging(nn.Module):
    """Patch Merging Layer."""
    
    def __init__(self, embed_dims: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.embed_dims = embed_dims
        self.reduction = nn.Linear(4 * embed_dims, 2 * embed_dims, bias=False)
        self.norm = norm_layer(4 * embed_dims)
        
    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        # Handle odd dimensions by padding
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = x.view(B, H, W, C)
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H = H + pad_h
            W = W + pad_w
            x = x.view(B, H * W, C)
        
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        
        x = self.norm(x)
        x = self.reduction(x)
        
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""
    
    def __init__(self, patch_size: int = 4, in_channels: int = 3, embed_dims: int = 96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        
        self.proj = nn.Conv2d(in_channels, embed_dims, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dims)
        else:
            self.norm = None
            
    def forward(self, x):
        B, C, H, W = x.shape
        # Pad if necessary
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x
        

class SwinTransformer(nn.Module):
    """Swin Transformer backbone for instance segmentation."""
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dims: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: int = 7,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.1,
        norm_layer=nn.LayerNorm,
        ape: bool = False,
        patch_norm: bool = True,
        out_indices: Tuple[int, ...] = (0, 1, 2, 3),
        frozen_stages: int = -1,
        use_checkpoint: bool = False
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dims = embed_dims
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        
        # Split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_channels=in_channels,
            embed_dims=embed_dims, norm_layer=norm_layer if self.patch_norm else None
        )
        
        patches_resolution = img_size // patch_size
        self.patches_resolution = patches_resolution
        
        # absolute position embedding
        if self.ape:
            num_patches = (img_size // patch_size) ** 2
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims))
            nn.init.trunc_normal_(self.absolute_pos_embed, std=.02)
            
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer_embed_dims = int(embed_dims * 2 ** i_layer)
            layer = BasicLayer(
                embed_dims=layer_embed_dims,  # Use the correct dimension for this layer
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None
            )
            self.layers.append(layer)
            
        # Feature dimensions for each layer (channel dimensions)
        num_features = [int(embed_dims * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features
        
        # Add norm layers for each output
        self.norms = nn.ModuleList()
        for i_layer in out_indices:
            # Norm layer uses channel dimension
            layer = norm_layer(num_features[i_layer])
            self.norms.append(layer)
            
        self._freeze_stages()
        
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
                
        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False
            
        if self.frozen_stages >= 2:
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
                    
    def forward(self, x):
        B, C, H_img, W_img = x.shape
        x = self.patch_embed(x)
        B, L, C = x.shape
        
        # Calculate actual patch resolution
        H = W = int(math.sqrt(L))
        assert H * W == L, f"L ({L}) must be a perfect square"
        
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        
        outs = []
        
        for i, layer in enumerate(self.layers):
            x_out, H, W, x, Hd, Wd = layer(x, H, W)
            
            if i in self.out_indices:
                norm_idx = self.out_indices.index(i)
                norm_layer = self.norms[norm_idx]
                x_out = norm_layer(x_out)
                
                # Reshape to image format
                B, L, C = x_out.shape
                x_out = x_out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                outs.append(x_out)
            
            H, W = Hd, Wd
            
        return outs