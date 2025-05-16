import torch
import torch.nn as nn
import pytest
from swin_maskrcnn.models.roi_head import ROIHead


class TestROIHeadIntegration:
    """Integration test for ROI head with shape mismatch fix."""
    
    def test_forward_with_shape_mismatch_scenario(self):
        """Test ROI head forward pass with the scenario that causes shape mismatch."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create ROI head
        roi_head = ROIHead(
            num_classes=69,
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='ROIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]
            ),
            mask_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='ROIAlign', output_size=14, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]
            )
        ).to(device)
        
        # Mock features
        batch_size = 4
        features = [torch.randn(batch_size, 256, s, s).to(device) 
                   for s in [128, 64, 32, 16]]
        
        # Create proposals with varying numbers per image
        proposals = []
        for i in range(batch_size):
            n_props = 100 + i * 20  # Different number of proposals per image
            props = torch.rand(n_props, 4).to(device) * 400
            # Ensure valid box format [x1, y1, x2, y2]
            props[:, 2:] = props[:, :2] + torch.rand(n_props, 2).to(device) * 100
            proposals.append(props)
        
        # Create targets that will lead to different positive samples per image
        targets = []
        for i in range(batch_size):
            num_gt = 2 + (i % 3)  # Varying GT boxes: 2, 3, 4, 2
            target = {
                'boxes': torch.rand(num_gt, 4).to(device) * 400,
                'labels': torch.randint(1, 70, (num_gt,)).to(device),
                'masks': torch.randn(num_gt, 1, 512, 512).to(device),
                'image_id': torch.tensor([i])
            }
            # Ensure valid box format
            target['boxes'][:, 2:] = target['boxes'][:, :2] + torch.rand(num_gt, 2).to(device) * 100
            targets.append(target)
        
        # Run forward pass - this should not raise shape mismatch error
        try:
            losses = roi_head(features, proposals, targets)
            
            # Verify losses are returned
            assert 'cls_loss' in losses
            assert 'bbox_loss' in losses
            assert 'mask_loss' in losses
            
            # Verify losses are valid tensors
            assert torch.isfinite(losses['cls_loss'])
            assert torch.isfinite(losses['bbox_loss'])
            assert torch.isfinite(losses['mask_loss'])
            
            print("ROI head forward pass successful with shape mismatch fix!")
            
        except RuntimeError as e:
            if "must match the size of tensor" in str(e):
                pytest.fail(f"Shape mismatch still occurring: {e}")
            else:
                raise
    
    def test_compute_targets_consistency(self):
        """Test that compute_targets produces consistent outputs."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        roi_head = ROIHead(
            num_classes=69,
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='ROIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]
            ),
            mask_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='ROIAlign', output_size=14, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]
            )
        ).to(device)
        
        # Test with varying scenarios
        for batch_size in [1, 2, 4]:
            proposals = []
            targets = []
            
            for i in range(batch_size):
                # Create proposals
                n_props = 200
                props = torch.rand(n_props, 4).to(device) * 256
                props[:, 2:] = props[:, :2] + torch.rand(n_props, 2).to(device) * 50
                proposals.append(props)
                
                # Create targets with different GT counts
                num_gt = i % 5  # 0, 1, 2, 3, 4 GT boxes
                if num_gt > 0:
                    target = {
                        'boxes': torch.rand(num_gt, 4).to(device) * 256,
                        'labels': torch.randint(1, 70, (num_gt,)).to(device),
                        'masks': torch.randn(num_gt, 1, 256, 256).to(device),
                        'image_id': torch.tensor([i])
                    }
                    target['boxes'][:, 2:] = target['boxes'][:, :2] + torch.rand(num_gt, 2).to(device) * 50
                else:
                    # Empty target
                    target = {
                        'boxes': torch.zeros(0, 4).to(device),
                        'labels': torch.zeros(0, dtype=torch.long).to(device),
                        'masks': torch.zeros(0, 1, 256, 256).to(device),
                        'image_id': torch.tensor([i])
                    }
                targets.append(target)
            
            # Compute targets
            sampled_proposals, sampled_labels, sampled_bbox_targets, sampled_mask_targets = \
                roi_head.compute_targets(proposals, targets)
            
            # Verify consistency
            total_pos = sum((lbls > 0).sum().item() for lbls in sampled_labels)
            total_bbox_targets = sum(len(t) for t in sampled_bbox_targets)
            
            # Number of bbox targets should equal number of positive samples
            assert total_pos == total_bbox_targets, \
                f"Mismatch: {total_pos} positive samples vs {total_bbox_targets} bbox targets"
            
            print(f"Batch size {batch_size}: {total_pos} positive samples, consistency verified!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])