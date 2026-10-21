---
upstream:
  - repo: open-mmlab/mmdetection
    paths:
      - mmdet/models/backbones/swin.py
    ref: main
    last_seen: 8405ad9bfce4867f552f2f7a643c9e78a97eb0b6
local:
  paths:
    - swin_maskrcnn/models/swin.py
---

# Swin Transformer backbone (mmdetection flavour)

A from-scratch PyTorch re-implementation of mmdetection's `SwinTransformer` detection backbone with no
mmcv / mmengine dependency: `PatchEmbed` (4x4 conv), `WindowMSA` with the relative-position-bias table,
`window_partition` / `window_reverse`, shifted-window `SwinTransformerBlock` (cyclic `torch.roll` plus
attention mask, padding to a multiple of the window size), `PatchMerging`, per-stage `BasicLayer`,
stochastic depth (`DropPath`), output norms per stage and `_freeze_stages`. It exists because the project
goal is a standalone Swin Mask R-CNN that can be read, modified and exported (ONNX) without installing
the OpenMMLab stack, while still loading mmdetection's released Swin-S COCO weights.

Adaptation notes:

- Constructor arguments follow mmdetection names (`embed_dims`, `depths`, `num_heads`, `window_size`,
  `mlp_ratio`, `drop_path_rate`, `out_indices`, `frozen_stages`, `patch_norm`) so the mmdet config can be
  copied across. Keep them stable; do not drift toward timm's `embed_dim` / `depths` spelling.
- Module naming intentionally differs from mmdet (`layers` not `stages`, `attn` not `attn.w_msa`,
  `mlp` as an `nn.Sequential` instead of `ffn.layers`, `patch_embed.proj` not `patch_embed.projection`,
  `norms.N` instead of `normN`). The weight converter in `swin_maskrcnn/utils/pretrained_loader.py`
  depends on this mapping, so any renaming here must be mirrored there.
- Only the Swin v1 (p4-w7) backbone is supported: optional `ape` absolute position embedding is kept
  but off by default, there is no pretrain-window resizing, no Swin v2 post-norm or cosine attention (see `plans/swin_v2_and_modern_attention_upgrade.md`
  for the intended future work). Upstream changes to the attention-mask construction, relative position
  index computation or the `_freeze_stages` semantics are the drift worth porting.
- `_init_weights` mirrors mmdet's trunc-normal(std=0.02) / zero-bias scheme for linear layers (plus
  kaiming-normal for the patch-embed conv).
