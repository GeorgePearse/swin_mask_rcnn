"""Test dataset loading to debug issues."""
import torch
from swin_maskrcnn.data.dataset import CocoDataset
from swin_maskrcnn.data.transforms import get_transform

def test_dataset():
    print("Testing dataset loading...")
    
    # Create dataset without transforms first
    dataset = CocoDataset(
        root_dir='/home/georgepearse/data/images',
        annotation_file='/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:23.077836_train_ordered.json',
        transforms=None,
        mode='train'
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a sample
    print("\nTesting sample without transforms...")
    img, target = dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Number of boxes: {len(target['boxes'])}")
    print(f"Number of masks: {len(target['masks'])}")
    print("Success!")
    
    # Now test with transforms
    print("\nTesting sample with transforms...")
    dataset.transforms = get_transform(train=True)
    try:
        img, target = dataset[0]
        print(f"Image shape: {img.shape}")
        print(f"Number of boxes: {len(target['boxes'])}")
        print(f"Number of masks: {len(target['masks'])}")
        print("Success!")
    except Exception as e:
        print(f"Error: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_dataset()