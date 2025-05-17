"""Debug the forward pass to understand why no predictions are made."""
import sys
sys.path.append("/home/georgepearse/swin_maskrcnn")

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from swin_maskrcnn.data.dataset import CocoDataset
from swin_maskrcnn.data.transforms_simple import get_transform_simple
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.utils.collate import collate_fn
from swin_maskrcnn.utils.logging import setup_logger, get_logger


def main():
    # Hardcoded paths - using fixed biases checkpoint
    checkpoint_path = Path("/home/georgepearse/swin_maskrcnn/fixed_biases_checkpoint.pth")
    img_root = Path("/home/georgepearse/data/images")
    val_ann = Path("/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:38.270134_val_ordered.json")
    
    logger = setup_logger(name="debug", level="DEBUG")
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = SwinMaskRCNN(num_classes=69).to(device)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        # Lightning checkpoint format
        state_dict = checkpoint['state_dict']
        model_state = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                model_state[k[6:]] = v
            else:
                model_state[k] = v
    elif 'model_state_dict' in checkpoint:
        # Standard PyTorch checkpoint
        model_state = checkpoint['model_state_dict']
    else:
        # Direct state dict
        model_state = checkpoint
    
    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
    logger.info(f"Missing keys: {len(missing_keys)}")
    logger.info(f"Unexpected keys: {len(unexpected_keys)}")
    
    model.eval()
    
    # Create validation dataset
    val_dataset = CocoDataset(
        root_dir=str(img_root),
        annotation_file=str(val_ann),
        transforms=get_transform_simple(train=False),
        mode='train'  # To get targets for debugging
    )
    
    # Only load one image for detailed debugging
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False
    )
    
    # Test on first image with detailed logging
    logger.info("\nDetailed forward pass debugging...")
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            if batch_idx >= 1:  # Only test first image
                break
            
            logger.info(f"\nProcessing first image...")
            logger.info(f"Image shape: {images[0].shape}")
            logger.info(f"Number of ground truth boxes: {len(targets[0]['boxes'])}")
            
            # Move to device
            images = [img.to(device) for img in images]
            
            # Stack images for backbone
            images_stacked = torch.stack(images)
            
            # Step through the forward pass
            logger.info("\n1. Backbone forward pass...")
            features = model.backbone(images_stacked)
            logger.info(f"Backbone output shapes: {[f.shape for f in features]}")
            
            logger.info("\n2. FPN forward pass...")
            features = model.neck(features)
            logger.info(f"FPN output shapes: {[f.shape for f in features]}")
            
            logger.info("\n3. RPN forward pass...")
            rpn_cls_scores, rpn_bbox_preds = model.rpn_head(features)
            logger.info(f"RPN cls scores shapes: {[s.shape for s in rpn_cls_scores]}")
            logger.info(f"RPN bbox preds shapes: {[b.shape for b in rpn_bbox_preds]}")
            
            # Get proposals
            logger.info("\n4. Getting RPN proposals...")
            proposals = model.rpn_head.get_proposals(
                rpn_cls_scores, rpn_bbox_preds,
                [(img.shape[-2], img.shape[-1]) for img in images],
                {'nms_pre': 1000, 'nms_thr': 0.7, 'max_per_img': 1000}
            )
            
            if proposals is None:
                logger.info("Proposals is None!")
            else:
                logger.info(f"Number of proposals per image: {[len(p) for p in proposals]}")
                if len(proposals) > 0 and len(proposals[0]) > 0:
                    logger.info(f"First few proposals: {proposals[0][:5]}")
                    
                    # Check proposal scores
                    logger.info("\n5. Checking RPN scores...")
                    # Get objectness scores
                    objectness = rpn_cls_scores[0].sigmoid()
                    logger.info(f"Objectness score stats - min: {objectness.min():.4f}, "
                               f"max: {objectness.max():.4f}, mean: {objectness.mean():.4f}")
                    
                    # Count high confidence predictions
                    high_conf = (objectness > 0.5).sum()
                    logger.info(f"Number of locations with objectness > 0.5: {high_conf}")
            
            # Try ROI head
            logger.info("\n6. ROI head forward pass...")
            detections = model.roi_head(features, proposals)
            
            if detections is None:
                logger.info("Detections is None!")
            else:
                logger.info(f"Number of detections: {len(detections)}")
                for i, det in enumerate(detections):
                    logger.info(f"Image {i} detections: {len(det.get('boxes', []))} boxes")
                    if 'scores' in det and len(det['scores']) > 0:
                        scores = det['scores'].cpu().numpy()
                        logger.info(f"Score stats - min: {scores.min():.4f}, "
                                   f"max: {scores.max():.4f}, mean: {scores.mean():.4f}")
    
    logger.info("\nDebug complete!")


if __name__ == "__main__":
    main()