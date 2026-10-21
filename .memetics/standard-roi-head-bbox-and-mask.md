---
upstream:
  - repo: open-mmlab/mmdetection
    paths:
      - mmdet/models/roi_heads/standard_roi_head.py
      - mmdet/models/roi_heads/bbox_heads/bbox_head.py
      - mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py
      - mmdet/models/roi_heads/mask_heads/fcn_mask_head.py
      - mmdet/models/roi_heads/roi_extractors/single_level_roi_extractor.py
    ref: main
    last_seen: 3acc46bb346165205a72356e6bfb598b1073b6b6
local:
  paths:
    - swin_maskrcnn/models/roi_head.py
---

# Standard RoI head: Shared2FC bbox head + FCN mask head

Single-file port of mmdetection's Mask R-CNN second stage. `StandardRoIHead` does proposal sampling
(max-IoU assignment at 0.5 / 0.5, 512 RoIs at 0.25 positive ratio), FPN-level assignment by RoI scale
(the FPN-paper rule `floor(k0 + log2(sqrt(wh) / 224))` with `k0=4`, clamped to the available levels,
where mmdet's `SingleRoIExtractor.map_roi_levels` uses `finest_scale=56`), `torchvision.ops.RoIAlign`
pooling at 7x7 for boxes and 14x14 for masks, the `Shared2FCBBoxHead` (two 1024-d FCs,
`num_classes + 1` softmax classifier, per-class 4-d regression `fc_reg`), the
`FCNMaskHead` (4x conv3x3 -> deconv x2 -> 1x1 to `num_classes` 28x28 logits), target encoding (delta-xywh
box targets, mask targets cropped/resized via `grid_sample`), the losses (cross-entropy with optional
foreground weighting, smooth-L1 on positives, BCE-with-logits on the positive-class mask) and
`get_results` post-processing. Written locally to avoid mmcv ops and registries while keeping parameter
shapes identical to the released Swin-S COCO checkpoint.

Adaptation notes:

- Keep hyper-parameter names mmdet-compatible (`roi_feat_size=7`, `fc_channels=1024`, mask size 28,
  `pos_iou_thr` / `neg_iou_thr` / `num_samples` / `pos_ratio`). Background is index 0 locally
  (mmdet puts background last); this affects the converter and class-weight tensors, see
  `docs/cross_entropy_weight_tensor_fix.md`.
- Mask RoIs are extracted from the same FPN levels as bbox RoIs; there is no separate mask RoI
  extractor config as in mmdet.
- Not ported: `reg_decoded_bbox`, `loss_bbox` variants other than smooth-L1 / L1, `CascadeRoIHead`,
  `mask_iou_head`, test-time augmentation, `rcnn_train_cfg.mask_size` other than 28, and mmdet's
  `SamplingResult` abstraction (sampling is done inline per image).
- When upstream changes `fcn_mask_head.py`'s `_do_paste_mask` / `get_seg_masks` or the bbox head's
  `_get_targets_single`, re-check `encode_mask_targets` and `get_results` here.
