"""Validation script to test with fixed biases."""
import torch
import yaml
from pathlib import Path
from pycocotools.coco import COCO

from scripts.train import MaskRCNNLightningModule
from scripts.config.training_config import TrainingConfig
from swin_maskrcnn.data.dataset import CocoDataset
from swin_maskrcnn.data.transforms_simple import get_transform_simple
from torch.utils.data import DataLoader
from swin_maskrcnn.utils.collate import collate_fn


def run_validation():
    """Run validation with fixed biases."""
    
    # Load config
    config_path = Path("scripts/config/config.yaml")
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = TrainingConfig(**config_dict)
    
    # Create COCO object
    val_coco = COCO(str(config.val_ann))
    
    # Create model
    model = MaskRCNNLightningModule(
        config=config,
        val_coco=val_coco
    )
    
    # Check biases
    fc_cls = model.model.roi_head.bbox_head.fc_cls
    print(f"Biases: background={fc_cls.bias[0].item():.4f}, foreground={fc_cls.bias[1].item():.4f}")
    
    # Move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create validation loader
    val_dataset = CocoDataset(
        root_dir=str(config.img_root),
        annotation_file=str(config.val_ann),
        transforms=get_transform_simple(train=False),
        mode='train'
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Set threshold to very low value for debugging
    score_threshold = 0.001
    
    # Run validation
    print(f"\nRunning validation with score threshold {score_threshold}...")
    
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            if batch_idx >= 10:  # Test on 10 batches
                break
            
            # Move to device
            images = [img.to(device) for img in images]
            
            # Get predictions
            outputs = model.model(images)
            
            # Process predictions
            for i, output in enumerate(outputs):
                image_id = targets[i]['image_id'].item()
                
                if output is not None and 'boxes' in output and len(output['boxes']) > 0:
                    # Filter by score threshold
                    scores = output['scores']
                    keep = scores >= score_threshold
                    
                    if keep.any():
                        boxes = output['boxes'][keep]
                        labels = output['labels'][keep]
                        scores = scores[keep]
                        
                        print(f"Batch {batch_idx}, Image {i} (ID: {image_id}): {keep.sum()} detections")
                        print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
                        
                        # Save predictions for COCO evaluation
                        for j in range(len(boxes)):
                            box = boxes[j].cpu().numpy()
                            all_predictions.append({
                                'image_id': int(image_id),
                                'category_id': int(labels[j]),
                                'bbox': [float(box[0]), float(box[1]), 
                                        float(box[2] - box[0]), float(box[3] - box[1])],
                                'score': float(scores[j])
                            })
                else:
                    print(f"Batch {batch_idx}, Image {i}: No detections")
    
    print(f"\nTotal predictions: {len(all_predictions)}")
    
    if all_predictions:
        # Run COCO evaluation
        import json
        from pycocotools.cocoeval import COCOeval
        
        # Save predictions
        pred_file = 'temp_predictions.json'
        with open(pred_file, 'w') as f:
            json.dump(all_predictions, f)
        
        # Load predictions
        coco_dt = val_coco.loadRes(pred_file)
        
        # Run evaluation
        coco_eval = COCOeval(val_coco, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        print(f"\nmAP: {coco_eval.stats[0]:.4f}")
        print(f"mAP50: {coco_eval.stats[1]:.4f}")


if __name__ == "__main__":
    run_validation()