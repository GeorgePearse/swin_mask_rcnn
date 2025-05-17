# MMDetection Reference Paths

This document contains the most relevant code paths from the mmdetection repository for reference when working on the Swin Mask R-CNN implementation.

## Main Components

### Backbones
- **Swin Transformer Backbone**: `https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/backbones/swin.py`

### Detectors
- **Mask R-CNN Detector**: `https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/detectors/mask_rcnn.py`

### Dense Heads (RPN)
- **RPN Head**: `https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/dense_heads/rpn_head.py`
- **GA-RPN Head**: `https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/dense_heads/ga_rpn_head.py`
- **Cascade RPN Head**: `https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/dense_heads/cascade_rpn_head.py`

### ROI Heads
- **Base ROI Head**: `https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/roi_heads/base_roi_head.py`
- **Standard ROI Head**: `https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/roi_heads/standard_roi_head.py`
- **Cascade ROI Head**: `https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/roi_heads/cascade_roi_head.py`

### Necks (Feature Pyramid)
- **FPN**: `https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/necks/fpn.py`
- **FPN with CARAFE**: `https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/necks/fpn_carafe.py`
- **HR-FPN**: `https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/necks/hrfpn.py`

### Additional References

#### Directory Structure
- **All Dense Heads**: `https://github.com/open-mmlab/mmdetection/tree/main/mmdet/models/dense_heads`
- **All ROI Heads**: `https://github.com/open-mmlab/mmdetection/tree/main/mmdet/models/roi_heads`
- **All Necks**: `https://github.com/open-mmlab/mmdetection/tree/main/mmdet/models/necks`
- **All Backbones**: `https://github.com/open-mmlab/mmdetection/tree/main/mmdet/models/backbones`
- **All Detectors**: `https://github.com/open-mmlab/mmdetection/tree/main/mmdet/models/detectors`

#### Task Modules (Utilities)
- **Task Modules Directory**: `https://github.com/open-mmlab/mmdetection/tree/main/mmdet/models/task_modules`

#### Layers
- **Model Layers**: `https://github.com/open-mmlab/mmdetection/tree/main/mmdet/models/layers`

## Most Relevant Files for Swin Mask R-CNN

1. **Swin Backbone**: `mmdet/models/backbones/swin.py`
2. **Mask R-CNN Detector**: `mmdet/models/detectors/mask_rcnn.py`
3. **Standard ROI Head**: `mmdet/models/roi_heads/standard_roi_head.py`
4. **RPN Head**: `mmdet/models/dense_heads/rpn_head.py`
5. **FPN Neck**: `mmdet/models/necks/fpn.py`

## Usage

When implementing or debugging features in the Swin Mask R-CNN codebase, refer to these paths for:
- Understanding the expected architecture and implementation patterns
- Comparing loss calculations and forward pass implementations
- Debugging issues by cross-referencing with the official implementation
- Finding solutions to specific problems or edge cases

Note: Always check the specific implementation in mmdetection as it may have updates or specific handling that could be relevant to your implementation.