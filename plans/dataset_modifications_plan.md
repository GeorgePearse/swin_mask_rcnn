# Dataset Modifications for Explicit Background Annotations

## Overview

This document outlines the necessary modifications to the dataset classes to support explicit background annotations and partial frame annotations.

## Key Components

### 1. Extended COCODataset Class

```python
class COCODatasetWithBackground(COCODataset):
    """Extended COCO dataset supporting explicit background annotations."""
    
    def __init__(self, *args, background_category_id=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.background_category_id = background_category_id
        self.annotated_regions = self._load_annotated_regions()
    
    def _load_annotated_regions(self):
        """Load annotated region metadata from COCO annotations."""
        if 'annotated_regions' not in self.coco.dataset:
            # Backward compatibility: treat entire image as annotated
            return None
        
        regions = {}
        for region_info in self.coco.dataset['annotated_regions']:
            image_id = region_info['image_id']
            regions[image_id] = {
                'regions': region_info['regions'],
                'coverage': region_info['coverage_percentage']
            }
        return regions
```

### 2. Annotation Processing

```python
def _process_annotations(self, ann_info):
    """Process annotations including explicit background."""
    
    # Separate object and background annotations
    object_anns = []
    background_anns = []
    
    for ann in ann_info:
        if ann.get('category_id') == self.background_category_id:
            background_anns.append(ann)
        else:
            object_anns.append(ann)
    
    # Process object annotations (standard)
    gt_bboxes, gt_labels, gt_masks = self._parse_objects(object_anns)
    
    # Process background annotations
    background_mask = self._create_background_mask(background_anns)
    
    # Create annotated region mask
    annotated_mask = self._create_annotated_mask(img_info['id'])
    
    return {
        'gt_bboxes': gt_bboxes,
        'gt_labels': gt_labels,
        'gt_masks': gt_masks,
        'background_mask': background_mask,
        'annotated_mask': annotated_mask
    }
```

### 3. Region Mask Generation

```python
def _create_annotated_mask(self, image_id):
    """Create mask indicating which regions are annotated."""
    
    if self.annotated_regions is None:
        # Full image is annotated
        return np.ones((img_h, img_w), dtype=np.uint8)
    
    if image_id not in self.annotated_regions:
        # No regions specified, assume full annotation
        return np.ones((img_h, img_w), dtype=np.uint8)
    
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    regions = self.annotated_regions[image_id]['regions']
    
    for region in regions:
        if region['type'] == 'polygon':
            # Fill polygon region
            poly = np.array(region['coordinates'])
            cv2.fillPoly(mask, [poly], 1)
        elif region['type'] == 'rectangle':
            # Fill rectangle region
            x, y, w, h = region['bbox']
            mask[y:y+h, x:x+w] = 1
    
    return mask
```

### 4. Transform Pipeline Modifications

```python
class PartialAnnotationTransform:
    """Transform that respects partial annotations."""
    
    def __call__(self, image, target):
        # Extract annotation masks
        annotated_mask = target.get('annotated_mask')
        background_mask = target.get('background_mask')
        
        # Apply transforms while preserving annotation info
        if self.crop:
            image, target = self._crop_with_masks(image, target)
        
        if self.resize:
            image, target = self._resize_with_masks(image, target)
        
        return image, target
    
    def _crop_with_masks(self, image, target):
        """Crop image while updating annotation masks."""
        # Get crop parameters
        crop_box = self._get_crop_box(image, target)
        
        # Crop image and annotations
        image = crop_image(image, crop_box)
        target['gt_bboxes'] = crop_bboxes(target['gt_bboxes'], crop_box)
        target['gt_masks'] = crop_masks(target['gt_masks'], crop_box)
        
        # Crop special masks
        if 'annotated_mask' in target:
            target['annotated_mask'] = crop_mask(target['annotated_mask'], crop_box)
        
        if 'background_mask' in target:
            target['background_mask'] = crop_mask(target['background_mask'], crop_box)
        
        return image, target
```

### 5. Data Collation

```python
def collate_fn_with_background(batch):
    """Custom collate function for partial annotations."""
    
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        
        # Ensure all targets have required fields
        if 'annotated_mask' not in target:
            # Full annotation assumed
            h, w = image.shape[-2:]
            target['annotated_mask'] = torch.ones((h, w), dtype=torch.uint8)
        
        if 'background_mask' not in target:
            # No explicit background
            h, w = image.shape[-2:]
            target['background_mask'] = torch.zeros((h, w), dtype=torch.uint8)
        
        targets.append(target)
    
    return torch.stack(images), targets
```

### 6. Validation and Error Checking

```python
def validate_partial_annotations(self, target):
    """Validate that partial annotations are consistent."""
    
    annotated_mask = target['annotated_mask']
    gt_masks = target['gt_masks']
    background_mask = target['background_mask']
    
    # Check all object masks are within annotated regions
    for mask in gt_masks:
        if not np.all(mask <= annotated_mask):
            raise ValueError("Object annotation outside annotated region")
    
    # Check background is only within annotated regions
    if not np.all(background_mask <= annotated_mask):
        raise ValueError("Background annotation outside annotated region")
    
    # Check no overlap between objects and background
    object_union = np.any(gt_masks, axis=0)
    if np.any(object_union & background_mask):
        raise ValueError("Object and background annotations overlap")
```

### 7. Integration with Existing Pipeline

```python
# In train.py or dataset initialization
def create_dataset(cfg):
    if cfg.use_partial_annotations:
        dataset = COCODatasetWithBackground(
            ann_file=cfg.ann_file,
            data_prefix=cfg.data_prefix,
            background_category_id=0,
            transforms=create_partial_annotation_transforms(cfg)
        )
    else:
        dataset = COCODataset(
            ann_file=cfg.ann_file,
            data_prefix=cfg.data_prefix,
            transforms=create_standard_transforms(cfg)
        )
    
    return dataset
```

## Benefits

1. **Backward Compatible**: Works with standard COCO format
2. **Flexible**: Supports various region shapes
3. **Efficient**: Only processes annotated regions
4. **Robust**: Validates annotation consistency

## Implementation Priority

1. Core dataset class with background support
2. Region mask generation
3. Transform pipeline updates
4. Validation functions
5. Integration with training pipeline

## Testing Strategy

1. Unit tests for mask generation
2. Transform consistency tests
3. End-to-end data loading tests
4. Performance benchmarks
5. Backward compatibility tests