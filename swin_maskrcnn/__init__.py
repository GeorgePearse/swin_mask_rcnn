"""Isolated SWIN-based Mask R-CNN implementation."""

from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.data.dataset import CocoDataset, create_dataloaders
from swin_maskrcnn.training.trainer import MaskRCNNTrainer, train_mask_rcnn
from swin_maskrcnn.inference.predictor import MaskRCNNPredictor, run_inference_pipeline

__version__ = "0.1.0"
__all__ = [
    "SwinMaskRCNN",
    "CocoDataset",
    "create_dataloaders",
    "MaskRCNNTrainer",
    "train_mask_rcnn",
    "MaskRCNNPredictor",
    "run_inference_pipeline",
]