"""Test that COCO weight loading fixes the prediction issue."""
import torch
import numpy as np
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.utils.load_coco_weights import load_coco_weights
from swin_maskrcnn.data.dataset import CocoDataset
from swin_maskrcnn.data.transforms_simple import get_transform_simple
from swin_maskrcnn.utils.collate import collate_fn
from swin_maskrcnn.utils.logging import setup_logger, get_logger

setup_logger(name=__name__, level="INFO")
logger = get_logger(__name__)


def test_model_predictions():
    """Test that model makes predictions after loading COCO weights."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model with 70 classes (69 + background)
    model = SwinMaskRCNN(num_classes=70).to(device)
    
    # Load COCO weights
    logger.info("Loading COCO weights...")
    missing_keys, unexpected_keys = load_coco_weights(model, num_classes=70)
    logger.info(f"Missing keys: {len(missing_keys)}")
    logger.info(f"Unexpected keys: {len(unexpected_keys)}")
    
    # Check classifier bias
    if hasattr(model.roi_head, 'fc_cls'):
        cls_bias = model.roi_head.fc_cls.bias.detach().cpu().numpy()
        logger.info(f"Background bias: {cls_bias[0]:.4f}")
        logger.info(f"Object bias mean: {cls_bias[1:].mean():.4f}")
        logger.info(f"Object bias range: [{cls_bias[1:].min():.4f}, {cls_bias[1:].max():.4f}]")
    
    # Test on real data
    val_ann_path = "/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:38.270134_val_ordered.json"
    val_dataset = CocoDataset(
        root_dir="/home/georgepearse/data/images",
        annotation_file=val_ann_path,
        transforms=get_transform_simple(train=False),
        mode='val'
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 3:  # Test a few batches
                break
            
            images, targets = batch
            images = [img.to(device) for img in images]
            outputs = model(images, targets=None)
            
            logger.info(f"\nBatch {i}:")
            total_predictions = 0
            for j, output in enumerate(outputs):
                if 'scores' in output:
                    scores = output['scores']
                    labels = output['labels']
                    num_preds = len(scores)
                    total_predictions += num_preds
                    
                    logger.info(f"  Image {j}: {num_preds} predictions")
                    if num_preds > 0:
                        # Show distribution of predicted classes
                        unique_labels, counts = torch.unique(labels, return_counts=True)
                        logger.info(f"    Score range: [{scores.min():.4f}, {scores.max():.4f}]")
                        logger.info(f"    Label distribution: {dict(zip(unique_labels.tolist(), counts.tolist()))}")
                        
                        # Show top predictions
                        top_k = min(5, num_preds)
                        top_scores, top_indices = scores.topk(top_k)
                        top_labels = labels[top_indices]
                        logger.info(f"    Top {top_k} scores: {top_scores.tolist()}")
                        logger.info(f"    Top {top_k} labels: {top_labels.tolist()}")
                else:
                    logger.info(f"  Image {j}: No predictions")
            
            logger.info(f"Total predictions in batch: {total_predictions}")
            
            # Debug info about classifier scores
            if i == 0:  # Only for first batch
                # Get some intermediate features to debug
                with torch.no_grad():
                    features = model.backbone(images)
                    fpn_features = model.neck(features)
                    proposals = model.rpn(images, features, fpn_features, None)
                    
                    logger.info(f"\nDebug info:")
                    logger.info(f"Number of proposals per image: {[len(p) for p in proposals]}")
                    
                    # Extract ROI features
                    if proposals[0].shape[0] > 0:
                        roi_features = model.roi_head.box_roi_pool(fpn_features, proposals)
                        box_features = model.roi_head.box_head(roi_features)
                        cls_scores = model.roi_head.fc_cls(box_features)
                        
                        # Check score distribution
                        probs = torch.softmax(cls_scores, dim=-1)
                        bg_probs = probs[:, 0]
                        obj_probs = probs[:, 1:].max(dim=-1)[0]
                        
                        logger.info(f"Background probability - mean: {bg_probs.mean():.4f}, "
                                  f"min: {bg_probs.min():.4f}, max: {bg_probs.max():.4f}")
                        logger.info(f"Best object probability - mean: {obj_probs.mean():.4f}, "
                                  f"min: {obj_probs.min():.4f}, max: {obj_probs.max():.4f}")


if __name__ == "__main__":
    test_model_predictions()