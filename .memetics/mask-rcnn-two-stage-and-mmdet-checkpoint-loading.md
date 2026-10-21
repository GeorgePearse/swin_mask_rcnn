---
upstream:
  - repo: open-mmlab/mmdetection
    paths:
      - mmdet/models/detectors/mask_rcnn.py
      - mmdet/models/detectors/two_stage.py
      - configs/_base_/models/mask-rcnn_r50_fpn.py
      - configs/swin/mask-rcnn_swin-s-p4-w7_fpn_amp-ms-crop-3x_coco.py
    ref: main
    last_seen: d2a3cbbfcf6ed8e631559d46f3dfd69859d2b6a5
local:
  paths:
    - swin_maskrcnn/models/mask_rcnn.py
    - swin_maskrcnn/utils/pretrained_loader.py
    - swin_maskrcnn/utils/load_coco_weights.py
---

# Two-stage Mask R-CNN assembly and mmdetection checkpoint conversion

`SwinMaskRCNN` is the plain-PyTorch equivalent of mmdet's `MaskRCNN(TwoStageDetector)`: backbone ->
FPN -> RPN (train/test proposal cfgs `nms_pre=2000/1000`, `nms_thr=0.7`, `max_per_img=1000`) -> RoI head,
returning a loss dict in training (`rpn_cls_loss`, `rpn_bbox_loss`, `roi_cls_loss`, `roi_bbox_loss`,
`roi_mask_loss`) and per-image detections in `predict` (score threshold + `batched_nms`). The defaults
(Swin-S depths `[2, 2, 18, 2]`, `frozen_backbone_stages`, drop-path 0.2, FPN 256, anchor/sampler numbers)
are lifted from mmdet's `mask-rcnn_r50_fpn.py` base and the `mask-rcnn_swin-s-p4-w7_fpn_amp-ms-crop-3x_coco`
config. `pretrained_loader.py` / `load_coco_weights.py` download the released Swin-S COCO checkpoint from
`download.openmmlab.com` and remap its `state_dict` keys (`backbone.stages.N` -> `layers.N`,
`attn.w_msa` -> `attn`, `ffn.layers.0.0` / `ffn.layers.1` -> `mlp.0` / `mlp.3`, `ln1` / `ln2` ->
`norm1` / `norm2`, `patch_embed.projection` -> `patch_embed.proj`, `normN` -> `norms.N`, neck/head
prefixes) onto the local module names, re-initialising `roi_head.bbox_head.fc_cls` when `num_classes`
differs from COCO's 80.

Adaptation notes:

- The key-mapping table is the contract between this repo and mmdet's module naming: any rename in
  upstream `swin.py`, `fpn.py`, `rpn_head.py` or the RoI heads breaks checkpoint loading silently
  (`strict=False`), so the loader should keep reporting missing / unexpected keys and the debug scripts
  under `scripts/debug/` compare against the raw checkpoint.
- The default checkpoint URL is the mmdet v2.0 Swin-S 3x ms-crop fp16 release; if upstream republishes
  weights under a new config name, update both the URL and the mapping, do not change the target layout.
- Not ported from `two_stage.py`: `with_rpn=False` mode, `rpn_head` sharing with `roi_head` training
  cfgs via `train_cfg`/`test_cfg` dicts (values are constructor kwargs here), `_forward` tensor mode and
  TTA. Loss key naming (`rpn_*`, `roi_*`) is the local convention consumed by the Lightning trainer and
  must stay.
