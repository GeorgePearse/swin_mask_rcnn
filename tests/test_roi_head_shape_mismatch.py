import torch
import torch.nn as nn
import pytest
from swin_maskrcnn.models.roi_head import ROIHead
from swin_maskrcnn.models.fpn import FPN
from swin_maskrcnn.models.swin import SwinTransformer


class TestROIHeadShapeMismatch:
    """Test for the shape mismatch error in ROI head loss calculation."""
    
    def test_bbox_shape_mismatch_reproduction(self):
        """Reproduce the shape mismatch error between pos_bbox_preds and bbox_targets."""
        # Setup model components
        backbone = SwinTransformer(
            img_size=224,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            frozen_stages=-1,
            use_checkpoint=False
        )
        
        fpn = FPN(
            in_channels=[96, 192, 384, 768],
            out_channels=256,
            num_outs=5
        )
        
        roi_head = ROIHead(
            num_classes=69,  # CMR dataset has 69 classes
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
        )
        
        # Create dummy inputs
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size = 4
        img_size = 512
        
        # Create features from backbone
        images = torch.randn(batch_size, 3, img_size, img_size).to(device)
        backbone = backbone.to(device)
        fpn = fpn.to(device)
        roi_head = roi_head.to(device)
        
        backbone_features = backbone(images)
        features = fpn(backbone_features)
        
        # Create proposals - intentionally create a mismatch scenario
        # This simulates what happens when ROI sampling produces different numbers
        # of positive samples than expected
        num_proposals = 1000
        proposals = []
        for i in range(batch_size):
            # Create varying numbers of proposals per image
            n_props = num_proposals // batch_size + (i * 50)  # Varying proposal counts
            prop = torch.rand(n_props, 4).to(device) * img_size
            proposals.append(prop)
        
        # Create targets with mismatched positive samples
        targets = []
        for i in range(batch_size):
            # Create ground truth boxes and labels
            num_gt = 3 + i  # Varying number of GT boxes
            target = {
                'boxes': torch.rand(num_gt, 4).to(device) * img_size,
                'labels': torch.randint(1, 70, (num_gt,)).to(device),
                'masks': torch.randn(num_gt, 1, img_size, img_size).to(device),
                'image_id': torch.tensor([i])
            }
            # Ensure boxes are in correct format [x1, y1, x2, y2]
            x1 = target['boxes'][:, 0]
            y1 = target['boxes'][:, 1]
            x2 = x1 + target['boxes'][:, 2]
            y2 = y1 + target['boxes'][:, 3]
            target['boxes'] = torch.stack([x1, y1, x2, y2], dim=-1)
            targets.append(target)
        
        # This should trigger the shape mismatch error
        with pytest.raises(RuntimeError, match="The size of tensor a"):
            roi_head(features, proposals, targets)
    
    def test_roi_sampling_consistency(self):
        """Test that ROI sampling produces consistent shapes between predictions and targets."""
        # Create a minimal test case
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
        
        # Create simple features
        batch_size = 2
        features = [torch.randn(batch_size, 256, 64, 64).to(device) for _ in range(4)]
        
        # Create proposals and targets
        proposals = []
        targets = []
        for i in range(batch_size):
            # Proposals
            n_props = 100
            prop = torch.rand(n_props, 4).to(device) * 256
            # Ensure valid box format
            prop[:, 2:] = prop[:, :2] + torch.rand(n_props, 2).to(device) * 50
            proposals.append(prop)
            
            # Targets
            num_gt = 5
            target = {
                'boxes': torch.rand(num_gt, 4).to(device) * 256,
                'labels': torch.randint(1, 70, (num_gt,)).to(device),
                'masks': torch.randn(num_gt, 1, 256, 256).to(device),
                'image_id': torch.tensor([i])
            }
            # Ensure valid box format
            target['boxes'][:, 2:] = target['boxes'][:, :2] + torch.rand(num_gt, 2).to(device) * 50
            targets.append(target)
        
        # Run forward pass and check shapes
        losses = roi_head(features, proposals, targets)
        
        # The losses should be computed without shape errors
        assert 'loss_cls' in losses
        assert 'loss_bbox' in losses
        assert 'loss_mask' in losses