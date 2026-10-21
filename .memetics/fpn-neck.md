---
upstream:
  - repo: open-mmlab/mmdetection
    paths:
      - mmdet/models/necks/fpn.py
    ref: main
    last_seen: fa77be290460e84ce7da975831cb7e687a419177
local:
  paths:
    - swin_maskrcnn/models/fpn.py
---

# Feature Pyramid Network neck

Plain-PyTorch port of mmdetection's `FPN` neck: 1x1 lateral convs over the selected backbone levels,
top-down nearest-neighbour upsampling and addition, 3x3 output convs, and extra levels via max-pooling
to reach `num_outs` (5 levels for Mask R-CNN: strides 4-64). Reimplemented to drop the mmcv `ConvModule`
and registry machinery while keeping the exact topology the released Swin-S Mask R-CNN checkpoint expects
(`in_channels=[96, 192, 384, 768]`, `out_channels=256`, `num_outs=5`).

Adaptation notes:

- Argument names (`in_channels_list`, `out_channels`, `num_outs`, `start_level`, `end_level`,
  `upsample_cfg`) follow upstream so configs translate directly; `in_channels_list` is the one deliberate
  rename.
- Only the default upstream behaviour is implemented: no `add_extra_convs` ('on_input' / 'on_lateral' /
  'on_output'), no `relu_before_extra_convs`, no norm layers in the lateral/fpn convs. Extra outputs are
  always max-pool of the last output. Do not add those variants unless a config needs them.
- Weight init is Xavier-uniform on convs as in mmdet's `init_weights`. Convs are bare `nn.Conv2d`
  modules in `lateral_convs` / `fpn_convs` (no `.conv` sub-module as in mmcv's `ConvModule`), so the
  checkpoint converter strips that level; keep the `ModuleList` names.
