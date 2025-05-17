"""Debug mask prediction issue."""
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np

from swin_maskrcnn.data.dataset import CocoDataset  
from swin_maskrcnn.data.transforms_simple import get_transform_simple
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.utils.collate import collate_fn
from swin_maskrcnn.utils.logging import setup_logger


def main():
    logger = setup_logger(level="DEBUG")
    
    # Paths 
    checkpoint_path = Path("/home/georgepearse/swin_maskrcnn/checkpoints/coco_initialized_corrected_biases.pth")
    img_root = Path("/home/georgepearse/data/images")
    val_ann = Path("/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:38.270134_val_ordered.json")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    model = SwinMaskRCNN(num_classes=69).to(device)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    # Create dataset
    val_dataset = CocoDataset(
        root_dir=str(img_root),
        annotation_file=str(val_ann),
        transforms=get_transform_simple(train=False),
        mode='train'
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    
    # Test on first image with detailed debugging
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            if i >= 1:  # Only test first image
                break
            
            logger.info(f"\nProcessing image {i}...")
            images = [img.to(device) for img in images]
            
            # Get predictions
            predictions = model(images)
            
            for j, pred in enumerate(predictions):
                logger.info(f"Image {j} results:")
                logger.info(f"  Keys in prediction: {pred.keys()}")
                
                if 'boxes' in pred:
                    num_boxes = len(pred['boxes'])
                    logger.info(f"  Number of boxes: {num_boxes}")
                    
                    if num_boxes > 0:
                        scores = pred['scores'].cpu().numpy()
                        labels = pred['labels'].cpu().numpy()
                        
                        logger.info(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
                        logger.info(f"  Labels: {labels}")
                        
                        # Check masks
                        if 'masks' in pred:
                            masks = pred['masks']
                            if masks is not None:
                                logger.info(f"  Masks shape: {masks.shape}")
                                mask_vals = masks.cpu().numpy()
                                logger.info(f"  Mask value range: [{mask_vals.min():.4f}, {mask_vals.max():.4f}]")
                            else:
                                logger.info("  Masks are None!")
                        else:
                            logger.info("  No 'masks' key in prediction!")
    
    logger.info("\nDebug complete!")


if __name__ == "__main__":
    main()