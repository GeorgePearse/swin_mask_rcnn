import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any
import logging
import numpy as np
from datetime import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.data.dataset import CocoDataset
from swin_maskrcnn.data.transforms_simple import get_transform_simple
from swin_maskrcnn.utils.logging import get_logger, setup_logger

# Setup logging
setup_logger(name=__name__, level="DEBUG")
logger = get_logger(__name__)


def inspect_prediction_head(model: nn.Module) -> None:
    """Inspect the prediction head of the model."""
    # Get the ROI head
    roi_head = model.roi_head
    
    # Check the classifier weights and biases
    logger.info("=== Classifier Layer Analysis ===")
    logger.info(f"ROI head type: {type(roi_head)}")
    logger.info(f"Number of classes: {model.num_classes}")
    
    # Check the box head structure
    if hasattr(roi_head, 'box_head'):
        box_head = roi_head.box_head
        logger.info(f"Box head type: {type(box_head)}")
        if hasattr(box_head, 'fc_layers'):
            logger.info(f"Number of FC layers in box head: {len(box_head.fc_layers)}")
    
    # Get the classifier layer
    if hasattr(roi_head, 'shared_fcs'):
        logger.info(f"Has shared FCs: {len(roi_head.shared_fcs) if roi_head.shared_fcs else 0}")
    
    if hasattr(roi_head, 'fc_cls'):
        classifier = roi_head.fc_cls
        logger.info(f"Classifier shape: {classifier.weight.shape}")
        logger.info(f"Classifier bias shape: {classifier.bias.shape}")
        
        # Analyze the bias values
        bias_values = classifier.bias.detach().cpu().numpy()
        logger.info(f"Background class bias: {bias_values[0]:.4f}")
        logger.info(f"Non-background bias mean: {bias_values[1:].mean():.4f}")
        logger.info(f"Non-background bias std: {bias_values[1:].std():.4f}")
        logger.info(f"Non-background bias range: [{bias_values[1:].min():.4f}, {bias_values[1:].max():.4f}]")
        
        # Analyze weights
        weights = classifier.weight.detach().cpu().numpy()
        logger.info(f"Background weight norm: {np.linalg.norm(weights[0]):.4f}")
        logger.info(f"Non-background weight norm mean: {np.mean([np.linalg.norm(w) for w in weights[1:]]):.4f}")
        
        # Show individual class biases for first few classes
        logger.info("First 10 class biases:")
        for i in range(min(10, len(bias_values))):
            logger.info(f"  Class {i}: {bias_values[i]:.4f}")


def analyze_model_predictions(model: nn.Module, device: torch.device) -> None:
    """Test model predictions on a dummy input."""
    model.eval()
    
    # Create dummy data
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 800, 800).to(device)
    
    # Create dummy targets (not used in eval mode, but needed for API)
    dummy_targets = []
    for i in range(batch_size):
        target = {
            'boxes': torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]], dtype=torch.float32).to(device),
            'labels': torch.tensor([1, 2], dtype=torch.int64).to(device),
            'masks': torch.ones((2, 800, 800), dtype=torch.uint8).to(device)
        }
        dummy_targets.append(target)
    
    logger.info("=== Running Inference ===")
    with torch.no_grad():
        try:
            outputs = model(dummy_images, targets=None)
            logger.info(f"Number of outputs: {len(outputs)}")
            
            for i, output in enumerate(outputs):
                if 'scores' in output:
                    scores = output['scores']
                    labels = output['labels']
                    logger.info(f"Image {i}:")
                    logger.info(f"  Number of predictions: {len(scores)}")
                    if len(scores) > 0:
                        logger.info(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
                        logger.info(f"  Unique labels: {torch.unique(labels).tolist()}")
                        logger.info(f"  Top 5 scores: {scores[:5].tolist()}")
                        logger.info(f"  Top 5 labels: {labels[:5].tolist()}")
                else:
                    logger.info(f"Image {i}: No predictions (empty output)")
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
    
    # Also test the classifier directly
    logger.info("\n=== Testing Classifier Directly ===")
    with torch.no_grad():
        # Create dummy features
        num_rois = 100
        feature_dim = 1024
        dummy_features = torch.randn(num_rois, feature_dim).to(device)
        
        if hasattr(model.roi_head, 'fc_cls'):
            # If there are shared FCs, pass through them first
            if hasattr(model.roi_head, 'shared_fcs') and model.roi_head.shared_fcs:
                for fc in model.roi_head.shared_fcs:
                    dummy_features = F.relu(fc(dummy_features))
            
            # Get classifier output
            cls_scores = model.roi_head.fc_cls(dummy_features)
            logger.info(f"Classifier output shape: {cls_scores.shape}")
            
            # Apply softmax to see probabilities
            probs = torch.softmax(cls_scores, dim=-1)
            
            # Background probabilities
            bg_probs = probs[:, 0]
            logger.info(f"Background probabilities - mean: {bg_probs.mean():.4f}, "
                       f"min: {bg_probs.min():.4f}, max: {bg_probs.max():.4f}")
            
            # Get the highest non-background probability for each ROI
            obj_probs, obj_labels = probs[:, 1:].max(dim=-1)
            logger.info(f"Best object probabilities - mean: {obj_probs.mean():.4f}, "
                       f"min: {obj_probs.min():.4f}, max: {obj_probs.max():.4f}")
            
            # Raw scores analysis
            logger.info(f"Raw scores - mean: {cls_scores.mean():.4f}, std: {cls_scores.std():.4f}")
            logger.info(f"Background raw scores - mean: {cls_scores[:, 0].mean():.4f}, "
                       f"std: {cls_scores[:, 0].std():.4f}")
            logger.info(f"Object raw scores - mean: {cls_scores[:, 1:].mean():.4f}, "
                       f"std: {cls_scores[:, 1:].std():.4f}")


def load_coco_weights(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    """Load COCO pre-trained weights."""
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Check the structure of the checkpoint
    if 'model' in state_dict:
        state_dict = state_dict['model']
    
    # Log some key information about the checkpoint
    logger.info(f"Checkpoint keys sample: {list(state_dict.keys())[:10]}")
    
    # Check classifier layer shape in checkpoint
    classifier_key = None
    for key in state_dict.keys():
        if 'fc_cls' in key and 'weight' in key:
            classifier_key = key
            break
    
    if classifier_key:
        logger.info(f"Found classifier key: {classifier_key}")
        logger.info(f"Classifier weight shape in checkpoint: {state_dict[classifier_key].shape}")
    
    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Missing keys: {len(missing_keys)}")
    logger.info(f"Unexpected keys: {len(unexpected_keys)}")
    
    if missing_keys:
        logger.info(f"First 10 missing keys: {missing_keys[:10]}")
    if unexpected_keys:
        logger.info(f"First 10 unexpected keys: {unexpected_keys[:10]}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model with correct number of classes (69 + 1 background = 70)
    model = SwinMaskRCNN(num_classes=70).to(device)
    logger.info(f"Created model with {model.num_classes} classes")
    
    # First inspect the model before loading weights
    logger.info("\n=== Model Structure (Before Loading Weights) ===")
    inspect_prediction_head(model)
    
    # Load checkpoint (use a test checkpoint for now)
    checkpoint_path = "/home/georgepearse/swin_maskrcnn/test_checkpoints/checkpoint_step_200.pth"
    load_coco_weights(model, checkpoint_path, device)
    
    # Inspect again after loading weights
    logger.info("\n=== Model Structure (After Loading Weights) ===")
    inspect_prediction_head(model)
    
    # Analyze predictions
    analyze_model_predictions(model, device)


if __name__ == "__main__":
    main()