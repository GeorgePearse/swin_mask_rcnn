"""Debug validation predictions issue."""
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.config import TrainingConfig
from swin_maskrcnn.data.dataset import CocoDataset
from swin_maskrcnn.data.transforms_simple import get_transform_simple
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.utils.load_coco_weights import load_coco_weights
from swin_maskrcnn.utils.collate import collate_fn
from torch.utils.data import DataLoader
from pycocotools.coco import COCO


def debug_validation_predictions():
    """Debug why there are no validation predictions."""
    
    # Load configuration
    config = TrainingConfig()
    
    # Load validation dataset
    val_dataset = CocoDataset(
        root_dir=str(config.img_root),
        annotation_file=str(config.val_ann),
        transforms=get_transform_simple(train=False),
        mode='train'  # As in main training script
    )
    
    # Create small validation dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Small batch for debugging
        shuffle=False,
        num_workers=0,  # Debug mode
        collate_fn=collate_fn,
        pin_memory=False,
    )
    
    # Initialize model
    model = SwinMaskRCNN(
        num_classes=config.num_classes,
        rpn_cls_pos_weight=config.rpn_cls_pos_weight,
        rpn_loss_cls_weight=config.rpn_loss_cls_weight,
        rpn_loss_bbox_weight=config.rpn_loss_bbox_weight,
        roi_cls_pos_weight=config.roi_cls_pos_weight,
        roi_loss_cls_weight=config.roi_loss_cls_weight,
        roi_loss_bbox_weight=config.roi_loss_bbox_weight,
        roi_loss_mask_weight=config.roi_loss_mask_weight,
    )
    
    # Load COCO pretrained weights
    print("Loading COCO pretrained weights...")
    missing_keys, unexpected_keys = load_coco_weights(model, num_classes=config.num_classes)
    print(f"Missing keys: {len(missing_keys)}")
    print(f"Unexpected keys: {len(unexpected_keys)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Debug first batch
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            if batch_idx >= 3:  # Only check first 3 batches
                break
                
            print(f"\n=== Batch {batch_idx} ===")
            print(f"Number of images: {len(images)}")
            print(f"Image shapes: {[img.shape for img in images]}")
            
            # Move to device
            images = [img.to(device) for img in images]
            
            # Get predictions
            outputs = model(images)
            
            print(f"Number of outputs: {len(outputs)}")
            
            for i, output in enumerate(outputs):
                print(f"\nImage {i}:")
                print(f"  Output type: {type(output)}")
                
                if output is None:
                    print("  WARNING: Output is None!")
                    continue
                    
                for key in output.keys():
                    if isinstance(output[key], torch.Tensor):
                        print(f"  {key}: shape={output[key].shape}")
                    else:
                        print(f"  {key}: {output[key]}")
                
                # Check predictions in detail
                if 'boxes' in output and len(output['boxes']) > 0:
                    scores = output['scores'].cpu().numpy()
                    print(f"  Score stats - min: {scores.min():.4f}, max: {scores.max():.4f}, mean: {scores.mean():.4f}")
                    print(f"  Score distribution:")
                    thresholds = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
                    for t in thresholds:
                        count = (scores >= t).sum()
                        print(f"    >= {t}: {count} predictions")
                else:
                    print("  No boxes in output!")
                
                # Check if masks exist
                if 'masks' in output:
                    if output['masks'] is None:
                        print("  WARNING: masks is None!")
                    elif len(output['masks']) == 0:
                        print("  WARNING: masks is empty!")
                else:
                    print("  WARNING: No masks key in output!")


if __name__ == "__main__":
    debug_validation_predictions()