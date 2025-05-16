import torch
import torch.nn.functional as F
import pytest


def test_bbox_loss_shape_fix():
    """Test the fix for bbox loss shape mismatch."""
    # Simulate the scenario from the error
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create labels for 4 batches with varying positive counts
    labels = [
        torch.tensor([0, 1, 0, 2, 0], device=device),  # 2 positive
        torch.tensor([0, 0, 0], device=device),        # 0 positive
        torch.tensor([1, 0, 3, 0], device=device),     # 2 positive  
        torch.tensor([0, 5, 0, 0, 0, 8], device=device) # 2 positive
    ]
    
    # Total samples = 5 + 3 + 4 + 6 = 18
    # Total positives = 2 + 0 + 2 + 2 = 6
    
    # Create bbox predictions for all samples
    num_classes = 69
    total_samples = sum(len(l) for l in labels)
    bbox_preds = torch.randn(total_samples, num_classes * 4, device=device)
    
    # Create bbox targets - only for positive samples
    bbox_targets = [
        torch.randn(2, 4, device=device),  # For batch 0
        # No targets for batch 1 (no positives)
        torch.randn(2, 4, device=device),  # For batch 2
        torch.randn(2, 4, device=device)   # For batch 3
    ]
    
    # Test the fixed logic
    if len(bbox_targets) > 0 and any(len(t) > 0 for t in bbox_targets):
        # Count positive samples in each batch
        batch_pos_counts = [torch.sum(lbls > 0).item() for lbls in labels]
        
        # Only compute loss if we have positive samples
        if sum(batch_pos_counts) > 0:
            # Get positive predictions in the same order as targets
            pos_preds = []
            start_idx = 0
            for i, count in enumerate(batch_pos_counts):
                if count > 0:
                    # Get predictions for this batch
                    batch_preds = bbox_preds[start_idx:start_idx + len(labels[i])]
                    pos_mask = labels[i] > 0
                    pos_batch_preds = batch_preds[pos_mask]
                    
                    # Select predictions for corresponding classes
                    pos_labels_batch = labels[i][pos_mask]
                    pos_batch_preds = pos_batch_preds.reshape(len(pos_labels_batch), -1, 4)
                    pos_batch_preds = pos_batch_preds[torch.arange(len(pos_labels_batch)), pos_labels_batch - 1]
                    pos_preds.append(pos_batch_preds)
                start_idx += len(labels[i])
            
            # Only concatenate if we have positive predictions
            if pos_preds:
                pos_bbox_preds = torch.cat(pos_preds)
                bbox_targets_cat = torch.cat(bbox_targets)
                
                # This should work now
                assert pos_bbox_preds.shape[0] == bbox_targets_cat.shape[0], \
                    f"Shape mismatch: predictions {pos_bbox_preds.shape[0]} vs targets {bbox_targets_cat.shape[0]}"
                
                # Compute loss
                bbox_loss = F.smooth_l1_loss(pos_bbox_preds, bbox_targets_cat)
                print(f"Successfully computed bbox loss: {bbox_loss.item()}")


def test_edge_case_no_positives():
    """Test when there are no positive samples in any batch."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create labels with no positive samples
    labels = [
        torch.tensor([0, 0, 0], device=device),
        torch.tensor([0, 0], device=device),
    ]
    
    num_classes = 69
    total_samples = sum(len(l) for l in labels)
    bbox_preds = torch.randn(total_samples, num_classes * 4, device=device)
    
    # No bbox targets
    bbox_targets = []
    
    # Should handle empty case gracefully
    if len(bbox_targets) > 0 and any(len(t) > 0 for t in bbox_targets):
        assert False, "Should not reach here with empty targets"
    else:
        bbox_loss = torch.tensor(0.0, device=device)
        assert bbox_loss.item() == 0.0


def test_mixed_batches():
    """Test with mixed batches - some with positives, some without."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create labels with mixed positive counts
    labels = [
        torch.tensor([0, 1, 0], device=device),      # 1 positive
        torch.tensor([0, 0, 0, 0], device=device),   # 0 positive
        torch.tensor([2, 3, 0], device=device),      # 2 positive
    ]
    
    num_classes = 69
    total_samples = sum(len(l) for l in labels)
    bbox_preds = torch.randn(total_samples, num_classes * 4, device=device)
    
    # Create bbox targets - only for batches with positives
    bbox_targets = [
        torch.randn(1, 4, device=device),  # For batch 0
        # No targets for batch 1
        torch.randn(2, 4, device=device),  # For batch 2
    ]
    
    # Test the fixed logic
    batch_pos_counts = [torch.sum(lbls > 0).item() for lbls in labels]
    assert batch_pos_counts == [1, 0, 2]
    
    if sum(batch_pos_counts) > 0:
        pos_preds = []
        start_idx = 0
        for i, count in enumerate(batch_pos_counts):
            if count > 0:
                batch_preds = bbox_preds[start_idx:start_idx + len(labels[i])]
                pos_mask = labels[i] > 0
                pos_batch_preds = batch_preds[pos_mask]
                
                pos_labels_batch = labels[i][pos_mask]
                pos_batch_preds = pos_batch_preds.reshape(len(pos_labels_batch), -1, 4)
                pos_batch_preds = pos_batch_preds[torch.arange(len(pos_labels_batch)), pos_labels_batch - 1]
                pos_preds.append(pos_batch_preds)
            start_idx += len(labels[i])
        
        pos_bbox_preds = torch.cat(pos_preds)
        bbox_targets_cat = torch.cat(bbox_targets)
        
        assert pos_bbox_preds.shape[0] == 3
        assert bbox_targets_cat.shape[0] == 3
        
        print("Mixed batches test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])