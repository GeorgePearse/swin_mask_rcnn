"""Quick test with learning rate warmup to monitor detection counts during training."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from swin_maskrcnn.data.dataset import CocoDataset
from swin_maskrcnn.data.transforms_simple import get_transform_simple
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.utils.collate import collate_fn
from swin_maskrcnn.utils.load_coco_weights import load_coco_weights

# Configuration
img_root = "/home/georgepearse/data/images"
train_ann = "/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:23.077836_train_ordered.json"
num_classes = 69
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create dataset
train_dataset = CocoDataset(
    root_dir=img_root,
    annotation_file=train_ann,
    transforms=get_transform_simple(train=True),
    mode='train'
)

# Create dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)

# Create model
model = SwinMaskRCNN(num_classes=num_classes).to(device)

# Load pretrained weights
print("Loading COCO pretrained weights...")
missing_keys, unexpected_keys = load_coco_weights(model, num_classes=num_classes)
print(f"Missing keys: {len(missing_keys)}")
print(f"Unexpected keys: {len(unexpected_keys)}")

# Set up optimizer with lower learning rate
base_lr = 2e-5
warmup_steps = 5
optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr / 10, weight_decay=1e-4)

# Training loop - just a few steps
print("\nStarting training loop with warmup...")
model.train()

for step, (images, targets) in enumerate(train_loader):
    if step >= 15:  # Only run 15 steps
        break
    
    # Update learning rate with warmup
    if step < warmup_steps:
        lr = base_lr * (step + 1) / warmup_steps
    else:
        lr = base_lr
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Move to device
    images = [img.to(device) for img in images]
    targets_device = []
    for t in targets:
        t_device = {}
        for k, v in t.items():
            if isinstance(v, torch.Tensor):
                t_device[k] = v.to(device)
            else:
                t_device[k] = v
        targets_device.append(t_device)
    
    # Count annotations
    total_annotations = sum(len(t["boxes"]) for t in targets_device)
    
    # Make predictions in eval mode to count detections
    model.eval()
    with torch.no_grad():
        predictions = model(images)
        total_detections = sum(len(p.get('boxes', [])) for p in predictions)
    
    # Training step
    model.train()
    optimizer.zero_grad()
    losses = model(images, targets_device)
    total_loss = losses.get('total', sum(losses.values()))
    total_loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    print(f"\nStep {step} (lr={lr:.6f}):")
    print(f"  Loss: {total_loss.item():.4f}")
    print(f"  Annotations: {total_annotations}")
    print(f"  Detections: {total_detections}")
    print(f"  RPN proposals: {losses.get('num_proposals', 'N/A')}")
    print(f"  ROI detections: {losses.get('num_detections', 'N/A')}")
    
print("\nTraining complete!")