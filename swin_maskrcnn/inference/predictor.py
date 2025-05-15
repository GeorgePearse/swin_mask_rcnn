"""Inference predictor for SWIN Mask R-CNN."""
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import torchvision.transforms.functional as F
import fiftyone as fo
import fiftyone.types as fot
from tqdm import tqdm

from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.data.dataset import CocoDataset


class MaskRCNNPredictor:
    """Predictor class for SWIN Mask R-CNN inference."""
    
    def __init__(
        self,
        model_path: str,
        num_classes: int = 69,
        device: Optional[torch.device] = None,
        score_threshold: float = 0.5,
        nms_threshold: float = 0.5
    ):
        """Initialize predictor.
        
        Args:
            model_path: Path to model checkpoint
            num_classes: Number of classes (including background)
            device: Device to run inference on
            score_threshold: Minimum score for detections
            nms_threshold: NMS threshold
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        
        # Load model
        self.model = SwinMaskRCNN(num_classes=num_classes)
        self.load_checkpoint(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load class names if available
        self.class_names = self._get_class_names()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        self.model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def _get_class_names(self) -> List[str]:
        """Get class names (can be customized based on dataset)."""
        # For CMR dataset, we'd load actual class names from annotations
        # For now, use generic names
        return [f"class_{i}" for i in range(self.model.num_classes)]
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for inference.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Preprocessed image tensor
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Convert to tensor and normalize
        image_tensor = F.to_tensor(image)
        image_tensor = F.normalize(
            image_tensor, 
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        
        return image_tensor
    
    def postprocess_predictions(
        self, 
        predictions: Dict[str, torch.Tensor],
        original_size: tuple
    ) -> Dict[str, np.ndarray]:
        """Postprocess model predictions.
        
        Args:
            predictions: Raw model predictions
            original_size: Original image size (width, height)
            
        Returns:
            Processed predictions
        """
        # Extract predictions
        boxes = predictions['boxes'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        masks = predictions.get('masks')
        
        if masks is not None:
            masks = masks.cpu().numpy()
            # Resize masks to original image size
            h, w = original_size[1], original_size[0]
            resized_masks = []
            for mask in masks:
                # Threshold mask
                mask = (mask > 0.5).astype(np.uint8)
                # Resize to original size
                mask_pil = Image.fromarray(mask.squeeze())
                mask_resized = mask_pil.resize((w, h), Image.NEAREST)
                resized_masks.append(np.array(mask_resized))
            masks = np.array(resized_masks)
        
        return {
            'boxes': boxes,
            'labels': labels,
            'scores': scores,
            'masks': masks
        }
    
    def predict_single(self, image_path: str) -> Dict[str, np.ndarray]:
        """Run inference on a single image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Predictions dictionary
        """
        # Load and preprocess image
        image_tensor = self.preprocess_image(image_path)
        original_size = Image.open(image_path).size
        
        # Add batch dimension
        batch = image_tensor.unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model.predict(
                batch,
                score_threshold=self.score_threshold,
                nms_threshold=self.nms_threshold
            )[0]
        
        # Postprocess predictions
        predictions = self.postprocess_predictions(predictions, original_size)
        
        return predictions
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict[str, np.ndarray]]:
        """Run inference on multiple images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of prediction dictionaries
        """
        all_predictions = []
        
        for image_path in tqdm(image_paths, desc="Running inference"):
            predictions = self.predict_single(image_path)
            all_predictions.append(predictions)
        
        return all_predictions
    
    def visualize_in_fiftyone(
        self,
        image_paths: List[str],
        predictions: List[Dict[str, np.ndarray]],
        dataset_name: str = "maskrcnn_predictions",
        launch_app: bool = True
    ) -> fo.Dataset:
        """Visualize predictions in FiftyOne.
        
        Args:
            image_paths: List of image paths
            predictions: List of prediction dictionaries
            dataset_name: Name for FiftyOne dataset
            launch_app: Whether to launch FiftyOne app
            
        Returns:
            FiftyOne dataset
        """
        # Create dataset
        dataset = fo.Dataset(dataset_name)
        dataset.persistent = True
        
        # Add samples
        samples = []
        for image_path, preds in zip(image_paths, predictions):
            sample = fo.Sample(filepath=str(image_path))
            
            # Add detections
            detections = []
            for i in range(len(preds['boxes'])):
                box = preds['boxes'][i]
                label = preds['labels'][i]
                score = preds['scores'][i]
                
                # Convert box format [x1, y1, x2, y2] to relative [x, y, w, h]
                img = Image.open(image_path)
                w, h = img.size
                rel_box = [
                    box[0] / w,  # x
                    box[1] / h,  # y
                    (box[2] - box[0]) / w,  # width
                    (box[3] - box[1]) / h   # height
                ]
                
                # Create detection
                detection = fo.Detection(
                    label=self.class_names[label] if label < len(self.class_names) else f"class_{label}",
                    bounding_box=rel_box,
                    confidence=score
                )
                
                # Add mask if available
                if preds['masks'] is not None and i < len(preds['masks']):
                    mask = preds['masks'][i]
                    detection.mask = mask
                
                detections.append(detection)
            
            sample["predictions"] = fo.Detections(detections=detections)
            samples.append(sample)
        
        dataset.add_samples(samples)
        
        # Launch app if requested
        if launch_app:
            session = fo.launch_app(dataset, port=5151)
            print(f"FiftyOne app launched at http://localhost:5151")
            return dataset, session
        
        return dataset


def run_inference_pipeline(
    model_path: str,
    image_paths: Union[str, List[str]],
    num_classes: int = 69,
    score_threshold: float = 0.5,
    nms_threshold: float = 0.5,
    visualize: bool = True,
    dataset_name: str = "maskrcnn_predictions"
) -> Dict[str, Any]:
    """Run complete inference pipeline.
    
    Args:
        model_path: Path to model checkpoint
        image_paths: Single image path or list of paths
        num_classes: Number of classes
        score_threshold: Minimum score threshold
        nms_threshold: NMS threshold
        visualize: Whether to visualize in FiftyOne
        dataset_name: Name for FiftyOne dataset
        
    Returns:
        Dictionary with predictions and optional FiftyOne dataset
    """
    # Ensure image_paths is a list
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    
    # Initialize predictor
    predictor = MaskRCNNPredictor(
        model_path=model_path,
        num_classes=num_classes,
        score_threshold=score_threshold,
        nms_threshold=nms_threshold
    )
    
    # Run predictions
    predictions = predictor.predict_batch(image_paths)
    
    result = {
        'predictions': predictions,
        'image_paths': image_paths
    }
    
    # Visualize if requested
    if visualize:
        dataset = predictor.visualize_in_fiftyone(
            image_paths,
            predictions,
            dataset_name=dataset_name,
            launch_app=True
        )
        result['fiftyone_dataset'] = dataset
    
    return result