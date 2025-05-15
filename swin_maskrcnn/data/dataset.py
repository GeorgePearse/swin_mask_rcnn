"""
COCO dataset wrapper with custom transforms.
"""
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoDetection
from pycocotools import mask as mask_utils
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CocoDataset(Dataset):
    """COCO dataset wrapper with Albumentations transforms."""
    
    def __init__(
        self,
        root_dir,
        annotation_file,
        transforms=None,
        mode='train'
    ):
        self.coco = CocoDetection(root=root_dir, annFile=annotation_file)
        self.transforms = transforms
        self.mode = mode
        
        # Get number of classes from the dataset
        cats = self.coco.coco.dataset.get('categories', [])
        self.num_classes = max([cat['id'] for cat in cats]) + 1
        
        # Filter images with valid annotations
        self.valid_ids = []
        for idx in range(len(self.coco)):
            try:
                img_id = self.coco.ids[idx]
                ann_ids = self.coco.coco.getAnnIds(imgIds=img_id)
                anns = self.coco.coco.loadAnns(ann_ids)
                
                # Check if there are valid annotations
                valid_anns = [ann for ann in anns 
                             if 'bbox' in ann and ann['area'] > 0 
                             and ann['bbox'][2] > 0 and ann['bbox'][3] > 0]
                
                if len(valid_anns) > 0:
                    self.valid_ids.append(idx)
            except Exception as e:
                print(f"Warning: Skipping image {idx} due to error: {e}")
                continue
    
    def __len__(self):
        return len(self.valid_ids)
    
    def __getitem__(self, idx):
        """Get item with image and annotations."""
        real_idx = self.valid_ids[idx]
        img, annotations = self.coco[real_idx]
        
        # Convert PIL to numpy
        img = np.array(img)
        
        # Extract bounding boxes, labels, and masks
        boxes = []
        labels = []
        masks = []
        
        for ann in annotations:
            try:
                # Skip invalid annotations
                if 'bbox' not in ann or ann['area'] <= 0:
                    continue
                    
                # Get bbox in COCO format [x, y, width, height]
                x, y, w, h = ann['bbox']
                
                # Skip invalid bounding boxes
                if w <= 0 or h <= 0:
                    continue
                
                # Convert to [x1, y1, x2, y2]
                x1, y1, x2, y2 = x, y, x + w, y + h
                
                # Ensure bbox is within image bounds
                x1 = max(0, min(x1, img.shape[1] - 1))
                y1 = max(0, min(y1, img.shape[0] - 1))
                x2 = max(0, min(x2, img.shape[1] - 1))
                y2 = max(0, min(y2, img.shape[0] - 1))
                
                # Skip if bbox is too small after clipping
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                
                boxes.append([x1, y1, x2, y2])
                labels.append(ann['category_id'])
            except Exception as e:
                print(f"Warning: Skipping annotation due to error: {e}")
                continue
            
            # Get segmentation mask
            try:
                if 'segmentation' in ann:
                    if isinstance(ann['segmentation'], list):
                        # Polygon format
                        rle = mask_utils.frPyObjects(
                            ann['segmentation'], 
                            img.shape[0], 
                            img.shape[1]
                        )
                        mask = mask_utils.decode(rle)
                        if len(mask.shape) > 2:
                            mask = mask.sum(axis=2) > 0
                    else:
                        # RLE format
                        mask = mask_utils.decode(ann['segmentation'])
                    
                    masks.append(mask.astype(np.uint8).copy())
                else:
                    # Create dummy mask if segmentation is missing
                    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                    masks.append(mask)
            except Exception as e:
                print(f"Warning: Error processing mask: {e}")
                # Create dummy mask on error
                mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                masks.append(mask)
        
        # Skip images with no valid annotations
        if len(boxes) == 0:
            # Return a dummy sample
            return self.__getitem__((idx + 1) % len(self))
        
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        # Apply transforms
        if self.transforms:
            transformed = self.transforms(
                image=img,
                bboxes=boxes,
                masks=masks,
                labels=labels
            )
            
            img = transformed['image']
            boxes = transformed['bboxes']
            # Ensure masks are contiguous
            masks = [np.ascontiguousarray(m) for m in transformed['masks']]
            labels = transformed['labels']
        
        # Convert to tensors
        if not isinstance(img, torch.Tensor):
            # Normalize and convert
            img = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
            # Apply ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = (img - mean) / std
        
        # Ensure boxes is always 2D even when empty
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        if len(masks) > 0:
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
        else:
            masks = torch.zeros((len(boxes), img.shape[1], img.shape[2]), dtype=torch.uint8)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([self.coco.ids[real_idx]]),
        }
        
        return img, target


def get_train_transforms(img_size=800):
    """Get training transforms using Albumentations."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc', 
        label_fields=['category_ids'],
        min_visibility=0.1
    ))


def get_val_transforms(img_size=800):
    """Get validation transforms using Albumentations."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc', 
        label_fields=['category_ids']
    ))


def collate_fn(batch):
    """Custom collate function for handling variable-sized annotations."""
    images = []
    targets = []
    
    for img, target in batch:
        images.append(img)
        targets.append(target)
    
    # Stack images into a batch
    images = torch.stack(images, dim=0)
    
    return images, targets


def create_dataloaders(
    train_root,
    train_ann_file,
    val_root,
    val_ann_file,
    batch_size=2,
    num_workers=4,
    img_size=800
):
    """Create train and validation dataloaders."""
    # Create datasets
    train_dataset = CocoDataset(
        root_dir=train_root,
        annotation_file=train_ann_file,
        transforms=get_train_transforms(img_size),
        mode='train'
    )
    
    val_dataset = CocoDataset(
        root_dir=val_root,
        annotation_file=val_ann_file,
        transforms=get_val_transforms(img_size),
        mode='val'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader