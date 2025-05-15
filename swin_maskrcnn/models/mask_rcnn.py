"""
SWIN-based Mask R-CNN implementation.
"""
import torch
import torch.nn as nn
from torchvision.ops import batched_nms

from .swin import SwinTransformer
from .fpn import FPN
from .rpn import RPNHead
from .roi_head import StandardRoIHead


class SwinMaskRCNN(nn.Module):
    """SWIN-based Mask R-CNN model."""
    
    def __init__(
        self,
        num_classes=80,
        pretrained_backbone=None,
        freeze_backbone=False,
        rpn_pos_threshold=0.7,
        rpn_neg_threshold=0.3,
        rpn_batch_size=256,
        rpn_positive_fraction=0.5,
        box_pos_threshold=0.5,
        box_neg_threshold=0.5,
        box_batch_size=512,
        box_positive_fraction=0.25,
        fpn_out_channels=256,
        roi_pool_size=7,
        mask_roi_pool_size=14,
    ):
        super().__init__()
        
        # Backbone
        self.backbone = SwinTransformer(
            embed_dims=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            frozen_stages=-1 if not freeze_backbone else 4,
        )
        
        if pretrained_backbone:
            self.load_backbone_weights(pretrained_backbone)
        
        # FPN
        self.neck = FPN(
            in_channels_list=[96, 192, 384, 768],
            out_channels=fpn_out_channels,
            num_outs=5,
            start_level=0,
            end_level=-1,
            upsample_cfg={'mode': 'nearest'}
        )
        
        # RPN
        self.rpn_head = RPNHead(
            in_channels=fpn_out_channels,
            feat_channels=fpn_out_channels,
            pos_iou_thr=rpn_pos_threshold,
            neg_iou_thr=rpn_neg_threshold,
        )
        
        # ROI Head
        self.roi_head = StandardRoIHead(
            num_classes=num_classes,
            pos_iou_thr=box_pos_threshold,
            neg_iou_thr=box_neg_threshold,
            pos_ratio=box_positive_fraction,
            num_samples=box_batch_size
        )
        
        self.num_classes = num_classes
        
    def load_backbone_weights(self, checkpoint_path):
        """Load pretrained weights for backbone."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Filter out unnecessary keys
        backbone_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('backbone.'):
                backbone_state_dict[k.replace('backbone.', '')] = v
        
        self.backbone.load_state_dict(backbone_state_dict, strict=False)
    
    def forward(self, images, targets=None):
        """Forward pass through the network."""
        # Convert list of images to batch tensor if needed
        if isinstance(images, list):
            # Stack images to create batch tensor
            images = torch.stack(images)
        
        # Feature extraction
        features = self.backbone(images)
        features = self.neck(features)
        
        # RPN forward
        rpn_cls_scores, rpn_bbox_preds = self.rpn_head(features)
        
        if self.training and targets is not None:
            # During training, get proposals and calculate RPN loss
            proposals = self.rpn_head.get_proposals(
                rpn_cls_scores, rpn_bbox_preds, 
                [(img.shape[-2], img.shape[-1]) for img in images],
                {'nms_pre': 2000, 'nms_thr': 0.7, 'max_per_img': 1000}
            )
            # Debug what's being passed to RPN
            gt_bboxes = [t['boxes'] for t in targets]
            print(f"Passing {len(gt_bboxes)} gt_bboxes to RPN loss")
            for i, gt in enumerate(gt_bboxes):
                print(f"  gt_bbox {i}: shape={gt.shape}")
            
            rpn_losses_raw = self.rpn_head.loss(
                rpn_cls_scores, rpn_bbox_preds,
                gt_bboxes,
                [(img.shape[-2], img.shape[-1]) for img in images]
            )
            # Prefix with rpn_ for consistency in the final loss dict
            rpn_losses = {}
            for k, v in rpn_losses_raw.items():
                key = f"rpn_{k}" if not k.startswith('rpn_') else k
                rpn_losses[key] = v
        else:
            # During inference, just get proposals
            proposals = self.rpn_head.get_proposals(
                rpn_cls_scores, rpn_bbox_preds,
                [(img.shape[-2], img.shape[-1]) for img in images],
                {'nms_pre': 1000, 'nms_thr': 0.7, 'max_per_img': 1000}
            )
            rpn_losses = None
        
        # ROI head forward
        if self.training and targets is not None:
            roi_losses = self.roi_head(features, proposals, targets)
            losses = {}
            if rpn_losses:
                losses.update(rpn_losses)
            if roi_losses:
                for k, v in roi_losses.items():
                    losses[f'roi_{k}'] = v
            return losses
        else:
            detections = self.roi_head(features, proposals)
            return detections
    
    @torch.no_grad()
    def predict(self, images, score_threshold=0.5, nms_threshold=0.5):
        """Run inference and return filtered predictions."""
        self.eval()
        
        # Run forward pass
        raw_detections = self.forward(images)
        
        # Post-process detections
        processed_detections = []
        
        for det in raw_detections:
            boxes = det['boxes']
            labels = det['labels']
            scores = det['scores']
            masks = det.get('masks')
            
            # Apply score threshold
            keep = scores >= score_threshold
            boxes = boxes[keep]
            labels = labels[keep]
            scores = scores[keep]
            if masks is not None:
                masks = masks[keep]
            
            # Apply NMS per class
            final_keep = []
            for cls_id in labels.unique():
                cls_mask = labels == cls_id
                cls_keep = batched_nms(
                    boxes[cls_mask],
                    scores[cls_mask],
                    labels[cls_mask],
                    nms_threshold
                )
                cls_indices = torch.where(cls_mask)[0]
                final_keep.append(cls_indices[cls_keep])
            
            if len(final_keep) > 0:
                final_keep = torch.cat(final_keep)
                
                processed_det = {
                    'boxes': boxes[final_keep],
                    'labels': labels[final_keep],
                    'scores': scores[final_keep]
                }
                
                if masks is not None:
                    processed_det['masks'] = masks[final_keep]
                    
                processed_detections.append(processed_det)
            else:
                processed_detections.append({
                    'boxes': torch.empty((0, 4)),
                    'labels': torch.empty((0,), dtype=torch.long),
                    'scores': torch.empty((0,))
                })
        
        return processed_detections