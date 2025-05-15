"""Check label range in dataset."""
from pycocotools.coco import COCO
import numpy as np

def check_labels():
    ann_file = '/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:23.077836_train_ordered.json'
    coco = COCO(ann_file)
    
    # Get all category IDs
    cat_ids = coco.getCatIds()
    print(f"Category IDs: {cat_ids}")
    print(f"Min category ID: {min(cat_ids)}")
    print(f"Max category ID: {max(cat_ids)}")
    print(f"Number of categories: {len(cat_ids)}")
    
    # Get category names
    cats = coco.loadCats(cat_ids)
    print("\nCategory names:")
    for cat in cats:
        print(f"  {cat['id']}: {cat['name']}")
    
    # Check if categories are contiguous
    print(f"\nCategories contiguous: {cat_ids == list(range(min(cat_ids), max(cat_ids)+1))}")
    
if __name__ == '__main__':
    check_labels()