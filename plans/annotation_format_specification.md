# Annotation Format Specification for Partial Frame Annotation

## Overview

This document specifies the data format for partial frame annotations with explicit background regions.

## Format Extension

### 1. Categories Definition

```json
{
  "categories": [
    {
      "id": 0,
      "name": "background",
      "supercategory": "meta",
      "is_background": true
    },
    {
      "id": 1,
      "name": "person",
      "supercategory": "human"
    },
    // ... other object categories
  ]
}
```

### 2. Annotation Format

#### A. Object Annotation (Standard)
```json
{
  "id": 1001,
  "image_id": 100,
  "category_id": 1,
  "segmentation": [[x1, y1, x2, y2, ...]],
  "bbox": [x, y, width, height],
  "area": 1500.5,
  "iscrowd": 0
}
```

#### B. Background Annotation (Extended)
```json
{
  "id": 1002,
  "image_id": 100,
  "category_id": 0,
  "segmentation": [[x1, y1, x2, y2, ...]],
  "bbox": [x, y, width, height],
  "area": 2500.0,
  "iscrowd": 0,
  "annotation_type": "background"
}
```

### 3. Annotated Regions Metadata

```json
{
  "annotated_regions": [
    {
      "id": 1,
      "image_id": 100,
      "regions": [
        {
          "type": "polygon",
          "coordinates": [[x1, y1], [x2, y2], ...],
          "fully_annotated": true,
          "description": "Main subject area"
        },
        {
          "type": "rectangle",
          "bbox": [x, y, width, height],
          "fully_annotated": true,
          "description": "Secondary region"
        }
      ],
      "coverage_percentage": 45.5,
      "annotation_strategy": "information_dense"
    }
  ]
}
```

### 4. Complete Example

```json
{
  "images": [...],
  "categories": [
    {
      "id": 0,
      "name": "background",
      "supercategory": "meta",
      "is_background": true
    },
    {
      "id": 1,
      "name": "person",
      "supercategory": "human"
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "segmentation": [[100, 100, 200, 100, 200, 200, 100, 200]],
      "bbox": [100, 100, 100, 100],
      "area": 10000,
      "iscrowd": 0
    },
    {
      "id": 2,
      "image_id": 1,
      "category_id": 0,
      "segmentation": [[0, 0, 300, 0, 300, 100, 0, 100]],
      "bbox": [0, 0, 300, 100],
      "area": 30000,
      "iscrowd": 0,
      "annotation_type": "background"
    }
  ],
  "annotated_regions": [
    {
      "id": 1,
      "image_id": 1,
      "regions": [
        {
          "type": "rectangle",
          "bbox": [0, 0, 300, 300],
          "fully_annotated": true,
          "description": "Upper left quadrant"
        }
      ],
      "coverage_percentage": 25.0,
      "annotation_strategy": "information_dense"
    }
  ]
}
```

## Semantic Rules

1. **Background Category**: Always use category_id=0 for background
2. **Coverage**: Annotated regions should fully contain all annotations
3. **Completeness**: Within annotated regions, all visible objects must be labeled
4. **Background**: Explicit background annotations required for all non-object areas within annotated regions

## Backward Compatibility

- Systems not supporting partial annotations treat all areas as annotated
- Background annotations can be filtered out for standard processing
- `annotated_regions` field is optional for full-frame annotations

## Validation Rules

1. All annotations must fall within declared annotated regions
2. Background annotations must have `annotation_type: "background"`
3. No object annotations allowed outside annotated regions
4. Coverage percentage must match actual annotated area

## Benefits

1. **Precise Training**: Only compute losses on annotated regions
2. **Efficient Annotation**: Focus effort on information-dense areas
3. **Quality Control**: Explicit background prevents false negatives
4. **Flexible Strategy**: Support various annotation strategies

## Implementation Notes

1. Parser must handle both standard and extended formats
2. Visualization tools should show annotated vs ignored regions
3. Training code must mask losses outside annotated regions
4. Evaluation metrics computed only on annotated areas