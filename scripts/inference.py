"""Run inference with trained SWIN Mask R-CNN model and visualize in FiftyOne."""
import argparse
from pathlib import Path
from typing import List

from swin_maskrcnn.inference.predictor import run_inference_pipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inference with SWIN Mask R-CNN')
    
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--images', nargs='+', required=True,
                        help='Path(s) to input images')
    parser.add_argument('--num-classes', type=int, default=69,
                        help='Number of classes')
    parser.add_argument('--score-threshold', type=float, default=0.5,
                        help='Score threshold for predictions')
    parser.add_argument('--nms-threshold', type=float, default=0.5,
                        help='NMS threshold')
    parser.add_argument('--no-visualize', action='store_true',
                        help='Skip FiftyOne visualization')
    parser.add_argument('--dataset-name', type=str, default='maskrcnn_predictions',
                        help='Name for FiftyOne dataset')
    
    return parser.parse_args()


def main():
    """Main inference function."""
    args = parse_args()
    
    # Verify model exists
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    
    # Verify images exist
    valid_images = []
    for image_path in args.images:
        if Path(image_path).exists():
            valid_images.append(image_path)
        else:
            print(f"Warning: Image not found: {image_path}")
    
    if not valid_images:
        raise ValueError("No valid images found")
    
    print(f"Running inference on {len(valid_images)} images...")
    
    # Run inference pipeline
    results = run_inference_pipeline(
        model_path=args.model_path,
        image_paths=valid_images,
        num_classes=args.num_classes,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        visualize=not args.no_visualize,
        dataset_name=args.dataset_name
    )
    
    # Print summary
    for i, (image_path, preds) in enumerate(zip(results['image_paths'], results['predictions'])):
        print(f"\n[{i+1}] {image_path}")
        print(f"  Detections: {len(preds['boxes'])}")
        if len(preds['boxes']) > 0:
            print(f"  Scores: {preds['scores'][:5]}...")  # Show first 5
            print(f"  Labels: {preds['labels'][:5]}...")
    
    if 'fiftyone_dataset' in results:
        print(f"\nVisualization available at http://localhost:5151")
        print("Press Ctrl+C to exit")
        
        try:
            # Keep the app running
            input("Press Enter to exit...")
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()