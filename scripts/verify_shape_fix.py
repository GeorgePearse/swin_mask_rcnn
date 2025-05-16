"""Test script to debug the shape mismatch issue."""

import torch
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
# Remove unused import
from swin_maskrcnn.data.transforms_simple import get_transform_simple
from swin_maskrcnn.utils.collate import collate_fn
from torch.utils.data import DataLoader
import numpy as np

# Create a dummy dataset for debugging
class DebugDataset:
    def __init__(self):
        pass  # No transforms for this simple test
        
    def __len__(self):
        return 2
    
    def __getitem__(self, idx):
        # Create synthetic data - use sizes that work with SWIN patch size
        image = np.random.rand(224, 224, 3).astype(np.float32)
        boxes = np.array([[50, 50, 100, 100], [120, 120, 170, 170]], dtype=np.float32)
        labels = np.array([1, 2], dtype=np.int64)
        masks = np.zeros((2, 224, 224), dtype=np.uint8)
        masks[0, 50:100, 50:100] = 1
        masks[1, 120:170, 120:170] = 1
        
        # Convert to tensor directly without transforms for this test
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.long),
            'masks': torch.as_tensor(masks, dtype=torch.uint8)
        }
        
        return image_tensor, target

# Create model and dataset
model = SwinMaskRCNN(
    num_classes=69,
    pretrained_backbone=None
)
model.cuda()

dataset = DebugDataset()
dataloader = DataLoader(
    dataset,
    batch_size=2,
    collate_fn=collate_fn,
    shuffle=False
)

# Test with debug prints
model.train()
for images, targets in dataloader:
    print(f"Batch info:")
    print(f"  Images: {len(images)} tensors")
    for i, img in enumerate(images):
        print(f"  Image {i}: {img.shape}")
    for i, t in enumerate(targets):
        print(f"  Target {i}: boxes={t['boxes'].shape}, labels={t['labels'].shape}")
    
    # Move images to CUDA
    images = [img.cuda() for img in images]
    targets = [{k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
    
    # Add debug print to ROI head
    with torch.enable_grad():
        try:
            loss_dict = model(images, targets)
            print(f"Losses: {loss_dict}")
        except Exception as e:
            print(f"Error: {e}")
            print("\nLet me add debug prints in the actual model to see what's happening...")
            
            # Add temporary debug prints
            import types
            
            def debug_loss(self, cls_scores, bbox_preds, mask_preds, labels, bbox_targets, mask_targets):
                print(f"\n=== DEBUG LOSS ===")
                print(f"cls_scores: {cls_scores.shape}")
                print(f"bbox_preds: {bbox_preds.shape}")
                print(f"labels: {[l.shape for l in labels]}")
                print(f"bbox_targets: {[b.shape for b in bbox_targets]}")
                
                # Count positives
                batch_pos_counts = [torch.sum(lbls > 0).item() for lbls in labels]
                print(f"Positive counts: {batch_pos_counts}")
                print(f"Total positives: {sum(batch_pos_counts)}")
                
                # Check prediction extraction logic
                pos_preds = []
                start_idx = 0
                for i, count in enumerate(batch_pos_counts):
                    print(f"\nBatch {i}: count={count}, start_idx={start_idx}, label_len={len(labels[i])}")
                    if count > 0:
                        # Get predictions for this batch
                        batch_preds = bbox_preds[start_idx:start_idx + len(labels[i])]
                        print(f"  batch_preds shape: {batch_preds.shape}")
                        pos_mask = labels[i] > 0
                        print(f"  pos_mask sum: {pos_mask.sum()}")
                        pos_batch_preds = batch_preds[pos_mask]
                        print(f"  pos_batch_preds shape: {pos_batch_preds.shape}")
                        
                        # Select predictions for corresponding classes
                        pos_labels_batch = labels[i][pos_mask]
                        print(f"  pos_labels: {pos_labels_batch}")
                        pos_batch_preds = pos_batch_preds.reshape(len(pos_labels_batch), -1, 4)
                        print(f"  reshaped: {pos_batch_preds.shape}")
                        pos_batch_preds = pos_batch_preds[torch.arange(len(pos_labels_batch)), pos_labels_batch - 1]
                        print(f"  selected: {pos_batch_preds.shape}")
                        pos_preds.append(pos_batch_preds)
                    start_idx += len(labels[i])
                
                print(f"\nFinal pos_preds: {[p.shape for p in pos_preds]}")
                if pos_preds:
                    pos_bbox_preds = torch.cat(pos_preds)
                    print(f"Concatenated predictions: {pos_bbox_preds.shape}")
                    print(f"Concatenated targets: {torch.cat(bbox_targets).shape}")
                
                # Call original loss function
                return self._original_loss(cls_scores, bbox_preds, mask_preds, labels, bbox_targets, mask_targets)
            
            # Monkey patch the loss function
            model.roi_head._original_loss = model.roi_head.loss
            model.roi_head.loss = types.MethodType(debug_loss, model.roi_head)
            
            # Try again with debug
            loss_dict = model(images, targets)
            print(f"Losses: {loss_dict}")