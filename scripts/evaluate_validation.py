"""Run predictions on validation set using trained model checkpoints."""
import argparse
import json
import logging
import torch
from pathlib import Path
from typing import List, Dict, Any

from swin_maskrcnn.inference.predictor import MaskRCNNPredictor
from swin_maskrcnn.data.dataset import CocoDataset
from swin_maskrcnn.utils.logging import setup_logger
from tqdm import tqdm


logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate model on validation set')
    
    parser.add_argument('--checkpoint-path', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--val-json', type=str, 
                      default='/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:38.270134_val_ordered.json',
                      help='Path to validation annotation JSON')
    parser.add_argument('--image-dir', type=str,
                      default='/home/georgepearse/data/images',
                      help='Directory containing images')
    parser.add_argument('--num-classes', type=int, default=69,
                      help='Number of classes')
    parser.add_argument('--score-threshold', type=float, default=0.05,
                      help='Score threshold for predictions')
    parser.add_argument('--nms-threshold', type=float, default=0.5,
                      help='NMS threshold')
    parser.add_argument('--batch-size', type=int, default=1,
                      help='Batch size for inference')
    parser.add_argument('--output-json', type=str, default='predictions.json',
                      help='Output JSON file for predictions')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to run inference on')
    parser.add_argument('--max-images', type=int, default=None,
                      help='Maximum number of images to process (for testing)')
    
    return parser.parse_args()


def coco_format_to_dict(coco_results: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """Convert COCO format results to dictionary grouped by image_id."""
    predictions_by_image = {}
    
    for result in coco_results:
        image_id = result['image_id']
        if image_id not in predictions_by_image:
            predictions_by_image[image_id] = []
        predictions_by_image[image_id].append(result)
    
    return predictions_by_image


def predictions_to_coco_format(predictions: Dict[str, Any], image_id: int) -> List[Dict[str, Any]]:
    """Convert model predictions to COCO format."""
    coco_results = []
    
    boxes = predictions['boxes']
    labels = predictions['labels']
    scores = predictions['scores']
    masks = predictions.get('masks', None)
    
    for i in range(len(boxes)):
        # Convert box from [x1, y1, x2, y2] to [x, y, width, height]
        box = boxes[i]
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        result = {
            'image_id': image_id,
            'category_id': int(labels[i]),
            'bbox': [float(x1), float(y1), float(width), float(height)],
            'score': float(scores[i])
        }
        
        # Add segmentation if available
        if masks is not None:
            # Convert mask to RLE format
            from pycocotools import mask as mask_utils
            mask_binary = masks[i].squeeze().astype('uint8')
            rle = mask_utils.encode(mask_binary.astype('uint8', order='F'))
            result['segmentation'] = {
                'size': rle['size'],
                'counts': rle['counts'].decode('utf-8')
            }
        
        coco_results.append(result)
    
    return coco_results


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup logging
    log_dir = Path(args.checkpoint_path).parent.parent / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(name='swin_maskrcnn', log_dir=str(log_dir), level='INFO')
    logger = logging.getLogger('swin_maskrcnn')
    logger.info(f"Starting evaluation with checkpoint: {args.checkpoint_path}")
    
    # Initialize predictor
    logger.info("Loading model...")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    predictor = MaskRCNNPredictor(
        model_path=args.checkpoint_path,
        num_classes=args.num_classes,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        device=device
    )
    
    # Load validation dataset
    logger.info("Loading validation dataset...")
    val_dataset = CocoDataset(
        root_dir=args.image_dir,
        annotation_file=args.val_json,
        transforms=None,  # No transforms for evaluation
        mode='val'
    )
    
    # Prepare to collect results
    all_predictions = []
    
    # Process images
    num_images = len(val_dataset)
    if args.max_images:
        num_images = min(num_images, args.max_images)
    
    logger.info(f"Processing {num_images} images...")
    
    for idx in tqdm(range(num_images), desc="Processing images"):
        # Get image info
        real_idx = val_dataset.valid_ids[idx]
        img_id = val_dataset.coco.ids[real_idx]
        img_info = val_dataset.coco.coco.loadImgs(img_id)[0]
        img_filename = img_info.get('file_name', f'image_{img_id}')
        image_path = Path(args.image_dir) / img_filename
        
        # Run prediction
        try:
            predictions = predictor.predict_single(str(image_path))
            
            # Debug: log prediction counts
            if idx == 0:  # First image only
                logger.info(f"Sample predictions for {image_path.name}:")
                logger.info(f"  Boxes: {len(predictions['boxes'])}")
                if len(predictions['boxes']) > 0:
                    logger.info(f"  Max score: {predictions['scores'].max():.4f}")
                    logger.info(f"  Min score: {predictions['scores'].min():.4f}")
                    logger.info(f"  Mean score: {predictions['scores'].mean():.4f}")
            
            # Convert to COCO format
            coco_results = predictions_to_coco_format(predictions, img_id)
            all_predictions.extend(coco_results)
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            continue
    
    # Save predictions
    output_path = Path(args.output_json)
    logger.info(f"Saving {len(all_predictions)} predictions to {output_path}")
    
    with open(output_path, 'w') as f:
        json.dump(all_predictions, f)
    
    # Print summary statistics
    predictions_by_image = coco_format_to_dict(all_predictions)
    logger.info(f"Total predictions: {len(all_predictions)}")
    logger.info(f"Images with predictions: {len(predictions_by_image)}")
    logger.info(f"Average predictions per image: {len(all_predictions) / num_images:.2f}")
    
    # Category distribution
    category_counts = {}
    for pred in all_predictions:
        cat_id = pred['category_id']
        category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
    
    logger.info("\nTop 10 predicted categories:")
    sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for cat_id, count in sorted_categories:
        logger.info(f"  Category {cat_id}: {count} predictions")
    
    logger.info("\nEvaluation complete!")


if __name__ == '__main__':
    main()