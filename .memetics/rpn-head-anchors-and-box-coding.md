---
upstream:
  - repo: open-mmlab/mmdetection
    paths:
      - mmdet/models/dense_heads/rpn_head.py
      - mmdet/models/dense_heads/anchor_head.py
      - mmdet/models/task_modules/prior_generators/anchor_generator.py
      - mmdet/models/task_modules/coders/delta_xywh_bbox_coder.py
    ref: main
    last_seen: 130d1d75a9ce93081d694febd6b38451a0c430be
local:
  paths:
    - swin_maskrcnn/models/rpn.py
---

# RPN head with anchor generation, IoU assignment and delta-xywh box coding

`swin_maskrcnn/models/rpn.py` collapses four mmdetection components into one file: the
`AnchorGenerator` (per-level base anchors from `strides`, `ratios`, `scales=[8]`, shifted over the
feature grid), the `RPNHead` conv tower (3x3 shared conv, 1x1 objectness and 4x delta convs, normal(0.01)
init), the `AnchorHead` training logic (max-IoU assignment with `pos_iou_thr=0.7` / `neg_iou_thr=0.3`,
ignore band in between, random sampling of 256 anchors at 0.5 positive fraction, loss normalised by an
`avg_factor` "following MMDetection"), and the `DeltaXYWHBBoxCoder` (`encode_bbox` / `decode_bbox` with
the standard means/stds and clipping). Proposals are decoded, filtered with `nms_pre`, NMS'd with
`torchvision.ops.batched_nms` and truncated to `max_per_img`. It was rewritten rather than imported so
the whole proposal path is readable in one place and runs on plain torch + torchvision.

Adaptation notes:

- Config knobs on `RPNHead` / `SwinMaskRCNN` keep mmdet's names and defaults
  (`rpn_pos_threshold`, `rpn_neg_threshold`, `rpn_batch_size`, `rpn_positive_fraction`,
  `nms_pre` / `nms_thr` / `max_per_img` in the proposal cfg dict). Keep these so mmdet configs map 1:1.
- Loss: BCE-with-logits for objectness (optional positive weight) and L1 / smooth-L1 for deltas,
  normalised by `avg_factor = pos + neg` (cls) and `num_pos` (bbox) as upstream does. The previously
  observed "zero predictions" bugs came from getting this normalisation wrong, so compare against
  upstream `AnchorHead.loss_by_feat` when debugging.
- Not ported: `allowed_border` filtering of anchors, `GA`/`Cascade` RPN variants, the `level_wise`
  sampling, `ignore` gt handling, and mmdet's `BaseBoxes` abstraction; the coder is the default
  means (0,0,0,0) / stds (1,1,1,1) of the RPN stage with `clip_border=True`.
- `tests/test_anchor_fix.py` pins the anchor layout; change anchors only with that test.
