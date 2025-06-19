"""Check CMR dataset categories."""
import json
from pycocotools.coco import COCO

# Load validation annotations
val_ann_path = "/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:38.270134_val_ordered.json"
coco = COCO(val_ann_path)

# Get category info
cat_ids = coco.getCatIds()
cats = coco.loadCats(cat_ids)

print(f"Number of categories: {len(cats)}")
print("\nFirst 10 categories:")
for i, cat in enumerate(cats[:10]):
    print(f"  ID: {cat['id']}, Name: {cat['name']}")

print(f"\nCategory ID range: [{min(cat_ids)}, {max(cat_ids)}]")

# Check if categories are 0-indexed or 1-indexed
if 0 in cat_ids:
    print("Categories include 0 (0-indexed)")
else:
    print("Categories start from 1 (1-indexed)")

# Check a prediction file
import glob
pred_files = glob.glob("predictions_epoch_*.json")
if pred_files:
    latest_pred = sorted(pred_files)[-1]
    print(f"\nChecking predictions from: {latest_pred}")
    with open(latest_pred) as f:
        preds = json.load(f)
    
    if preds:
        pred_cats = set(p['category_id'] for p in preds)
        print(f"Predicted category IDs: {sorted(pred_cats)}")
        print(f"Valid predictions: {len([p for p in preds if p['category_id'] in cat_ids])}/{len(preds)}")