"""Script to check initial loss values."""
import torch
import numpy as np
from swin_maskrcnn.data.dataset import CocoDataset
from swin_maskrcnn.data.transforms_simple import get_transform_simple
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.utils.collate import collate_fn
from torch.utils.data import DataLoader
from swin_maskrcnn.utils.pretrained_loader import load_pretrained_from_url
from scripts.config import TrainingConfig
from swin_maskrcnn.utils.logging import setup_logger


def check_initial_loss():
    """Check initial loss values with proper initialization."""
    # Setup logger
    logger = setup_logger(
        name="check_initial_loss",
        level="INFO"
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config
    config = TrainingConfig()
    
    # Create dataset
    train_dataset = CocoDataset(
        root_dir=str(config.img_root),
        annotation_file=str(config.train_ann),
        transforms=get_transform_simple(train=True),
        mode='train'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # Small batch size to avoid OOM
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=False  # Disable to save memory
    )
    
    # Test model initialization
    logger.info("Testing model initialization...")
    
    # Model without pretrained weights
    model_scratch = SwinMaskRCNN(num_classes=config.num_classes)
    model_scratch = model_scratch.to(device)
    model_scratch.train()
    
    # Model with pretrained weights
    model_pretrained = SwinMaskRCNN(num_classes=config.num_classes)
    if hasattr(config, 'pretrained_checkpoint_url') and config.pretrained_checkpoint_url:
        load_pretrained_from_url(model_pretrained, config.pretrained_checkpoint_url, strict=False)
    model_pretrained = model_pretrained.to(device)
    model_pretrained.train()
    
    # Test both models on first batch
    logger.info("Checking initial losses...")
    batch = next(iter(train_loader))
    images, targets = batch  # unpacking a tuple
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    
    # Test scratch model
    logger.info("Model from scratch:")
    losses_scratch = model_scratch(images, targets)
    total_loss_scratch = sum(losses_scratch.values())
    logger.info(f"  RPN Cls Loss: {losses_scratch.get('rpn_cls_loss', 0):.4f}")
    logger.info(f"  RPN Bbox Loss: {losses_scratch.get('rpn_bbox_loss', 0):.4f}")
    logger.info(f"  ROI Cls Loss: {losses_scratch.get('roi_cls_loss', 0):.4f}")
    logger.info(f"  ROI Bbox Loss: {losses_scratch.get('roi_bbox_loss', 0):.4f}")
    logger.info(f"  ROI Mask Loss: {losses_scratch.get('roi_mask_loss', 0):.4f}")
    logger.info(f"  Total Loss: {total_loss_scratch:.4f}")
    
    # Test pretrained model
    logger.info("Model with pretrained weights:")
    losses_pretrained = model_pretrained(images, targets)
    total_loss_pretrained = sum(losses_pretrained.values())
    logger.info(f"  RPN Cls Loss: {losses_pretrained.get('rpn_cls_loss', 0):.4f}")
    logger.info(f"  RPN Bbox Loss: {losses_pretrained.get('rpn_bbox_loss', 0):.4f}")
    logger.info(f"  ROI Cls Loss: {losses_pretrained.get('roi_cls_loss', 0):.4f}")
    logger.info(f"  ROI Bbox Loss: {losses_pretrained.get('roi_bbox_loss', 0):.4f}")
    logger.info(f"  ROI Mask Loss: {losses_pretrained.get('roi_mask_loss', 0):.4f}")
    logger.info(f"  Total Loss: {total_loss_pretrained:.4f}")
    
    # Check parameter statistics
    logger.info("Parameter statistics (scratch model):")
    for name, param in model_scratch.named_parameters():
        if 'weight' in name and len(param.shape) >= 2:
            logger.info(f"  {name}: mean={param.mean():.6f}, std={param.std():.6f}")
    
    # Compare with MMDetection's expected ranges
    logger.info("Expected initial loss ranges from MMDetection:")
    logger.info("  RPN Cls Loss: ~0.7 (binary cross-entropy)")
    logger.info("  RPN Bbox Loss: ~1-3")
    logger.info("  ROI Cls Loss: ~3-4 (depends on num_classes)")
    logger.info("  ROI Bbox Loss: ~1-2")
    logger.info("  ROI Mask Loss: ~0.7")
    logger.info("  Total Loss: ~6-10")


if __name__ == '__main__':
    check_initial_loss()