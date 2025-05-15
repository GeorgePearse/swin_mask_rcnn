"""
Custom collate function for Mask R-CNN training.
"""
import torch
from typing import List, Tuple, Dict


def collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[List[torch.Tensor], List[Dict]]:
    """
    Custom collate function for Mask R-CNN.
    
    Args:
        batch: List of (image, target) tuples
        
    Returns:
        images: List of image tensors
        targets: List of target dictionaries
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    return images, targets