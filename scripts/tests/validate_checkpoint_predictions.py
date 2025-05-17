"""Validate checkpoint produces predictions."""
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from swin_maskrcnn.data.dataset import CocoDataset
from swin_maskrcnn.data.transforms_simple import get_transform_simple
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.utils.collate import collate_fn
from swin_maskrcnn.utils.logging import setup_logger


def main():
    logger = setup_logger()
    
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
    
    # Test on first few images
    total_predictions = 0
    images_with_predictions = 0
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            if i >= 5:  # Test first 5 images
                break
                
            images = [img.to(device) for img in images]
            predictions = model(images)
            
            for j, pred in enumerate(predictions):
                num_boxes = len(pred.get('boxes', []))
                logger.info(f"Image {i}, batch {j}: {num_boxes} predictions")
                
                if num_boxes > 0:
                    images_with_predictions += 1
                    total_predictions += num_boxes
                    
                    # Show score distribution
                    scores = pred['scores'].cpu().numpy()
                    logger.info(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
                    logger.info(f"  Scores > 0.01: {(scores > 0.01).sum()}")
    
    logger.info(f"\nSummary:")
    logger.info(f"Total predictions: {total_predictions}")
    logger.info(f"Images with predictions: {images_with_predictions}/5")
    
    if total_predictions == 0:
        logger.error("No predictions made! There's still an issue with the model.")
    else:
        logger.info("Model is making predictions successfully!")


if __name__ == "__main__":
    main()