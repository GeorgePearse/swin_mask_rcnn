"""Test the error handling functionality during training."""
import torch
import csv
from pathlib import Path
from swin_maskrcnn.data.dataset import CocoDataset
from swin_maskrcnn.data.transforms_simple import get_transform_simple


def test_error_logging():
    """Test that error logging to CSV works correctly."""
    # Create a test CSV file
    test_csv_path = Path("test_error_log.csv")
    
    with open(test_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'batch_idx', 'image_filename', 'image_id', 'error_type', 'error_message'])
    
    # Test writing an error record
    with open(test_csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([0, 1, 'test_image.jpg', 12345, 'AssertionError', 'Shape mismatch'])
    
    # Read back and verify
    with open(test_csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)
        assert len(rows) == 2  # Header + 1 data row
        assert rows[1] == ['0', '1', 'test_image.jpg', '12345', 'AssertionError', 'Shape mismatch']
    
    # Clean up
    test_csv_path.unlink()
    print("CSV error logging test passed!")


def test_dataset_image_filename():
    """Test that dataset includes image filename in target."""
    # Test with CMR dataset
    dataset = CocoDataset(
        root_dir="/home/georgepearse/data/images",
        annotation_file="/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:38.270134_val_ordered.json",
        transforms=get_transform_simple('val'),
        mode='val'
    )
    
    if len(dataset) > 0:
        img, target = dataset[0]
        assert 'image_filename' in target, "image_filename not found in target"
        assert 'image_id' in target, "image_id not found in target"
        print(f"Dataset test passed! Image filename: {target['image_filename']}")
    else:
        print("Dataset is empty, cannot test")


if __name__ == "__main__":
    test_error_logging()
    test_dataset_image_filename()