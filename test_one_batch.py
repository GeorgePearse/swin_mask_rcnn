"""Test just one batch with normalization."""
import torch
from swin_maskrcnn.data.dataset import CocoDataset
from swin_maskrcnn.data.transforms_simple import get_transform_simple
from swin_maskrcnn.utils.collate import collate_fn
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from torch.utils.data import DataLoader


def main():
    # Create dataset
    dataset = CocoDataset(
        root_dir='/home/georgepearse/data/images',
        annotation_file='/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:23.077836_train_ordered.json',
        transforms=get_transform_simple(train=True),
        mode='train'
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Small batch
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Create model
    model = SwinMaskRCNN(num_classes=69)
    model.eval()  # Set to eval to avoid dropout
    
    # Get one batch
    images, targets = next(iter(dataloader))
    
    print(f"Batch size: {len(images)}")
    print(f"Image shape: {images[0].shape}")
    print(f"Image mean: {images[0].mean():.3f}")
    print(f"Image std: {images[0].std():.3f}")
    print(f"Image range: [{images[0].min():.3f}, {images[0].max():.3f}]")
    
    # Check if normalized (should have mean close to 0 and std close to 1)
    mean_val = images[0].mean().item()
    std_val = images[0].std().item()
    
    is_normalized = abs(mean_val) < 1.0 and abs(std_val - 1.0) < 1.0
    print(f"\nImage is normalized: {is_normalized}")
    
    # Try forward pass
    try:
        with torch.no_grad():
            outputs = model(images, targets)
        print("\nForward pass successful with normalization!")
        
        # Training mode test (should return losses)
        model.train()
        loss_dict = model(images, targets)
        print(f"Loss keys: {list(loss_dict.keys())}")
        total_loss = sum(loss_dict.values())
        print(f"Total loss: {total_loss.item():.3f}")
        
    except Exception as e:
        print(f"Error during forward pass: {e}")


if __name__ == '__main__':
    main()