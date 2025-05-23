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
from swin_maskrcnn.utils.logging import get_logger

logger = get_logger()


class SwinMaskRCNN(nn.Module):
    """SWIN-based Mask R-CNN model."""
    
    def __init__(
        self,
        num_classes=80,
        pretrained_backbone=None,
        freeze_backbone=False,
        frozen_backbone_stages=-1,  # New parameter for fine-grained control
        rpn_pos_threshold=0.7,
        rpn_neg_threshold=0.3,
        rpn_batch_size=256,
        rpn_positive_fraction=0.5,
        rpn_cls_pos_weight=1.0,
        rpn_loss_cls_weight=1.0,
        rpn_loss_bbox_weight=1.0,
        box_pos_threshold=0.5,
        box_neg_threshold=0.5,
        box_batch_size=512,
        box_positive_fraction=0.25,
        roi_cls_pos_weight=1.0,
        roi_loss_cls_weight=1.0,
        roi_loss_bbox_weight=1.0,
        roi_loss_mask_weight=1.0,
        fpn_out_channels=256,
        roi_pool_size=7,
        mask_roi_pool_size=14,
    ):
        super().__init__()
        
        # Backbone - use frozen_backbone_stages or fall back to freeze_backbone
        if frozen_backbone_stages >= 0:
            stages_to_freeze = frozen_backbone_stages
        elif freeze_backbone:
            stages_to_freeze = 4  # Freeze all stages
        else:
            stages_to_freeze = -1  # Don't freeze any stages
            
        self.backbone = SwinTransformer(
            embed_dims=96,
            depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            frozen_stages=stages_to_freeze,
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
            cls_pos_weight=rpn_cls_pos_weight,
            loss_cls_weight=rpn_loss_cls_weight,
            loss_bbox_weight=rpn_loss_bbox_weight,
        )
        
        # ROI Head
        self.roi_head = StandardRoIHead(
            num_classes=num_classes,
            pos_iou_thr=box_pos_threshold,
            neg_iou_thr=box_neg_threshold,
            pos_ratio=box_positive_fraction,
            num_samples=box_batch_size,
            cls_pos_weight=roi_cls_pos_weight,
            loss_cls_weight=roi_loss_cls_weight,
            loss_bbox_weight=roi_loss_bbox_weight,
            loss_mask_weight=roi_loss_mask_weight,
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
            # Get ground truth bboxes for RPN
            gt_bboxes = [t['boxes'] for t in targets]
            
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
            
            # Debug: log number of proposals
            if proposals is not None and len(proposals) > 0:
                logger.debug(f"[DEBUG] RPN proposals for batch: {[len(p) for p in proposals]}")
                total_proposals = sum(len(p) for p in proposals)
                logger.debug(f"RPN generated {total_proposals} proposals for {len(proposals)} images")
            else:
                logger.debug(f"[DEBUG] No RPN proposals generated")
        
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
            # Debug: log detections with more detail
            if detections is not None and len(detections) > 0:
                total_detections = sum(len(d.get('boxes', [])) for d in detections)
                logger.debug(f"[DEBUG] Total detections: {total_detections}, per image: {[len(d.get('boxes', [])) for d in detections]}")
                
                # Log score distribution
                all_scores = []
                for d in detections:
                    if 'scores' in d and len(d['scores']) > 0:
                        all_scores.extend(d['scores'].cpu().numpy().tolist())
                
                if all_scores:
                    import numpy as np
                    scores_np = np.array(all_scores)
                    logger.debug(f"[DEBUG] Score statistics - count: {len(scores_np)}, "
                          f"min: {scores_np.min():.4f}, max: {scores_np.max():.4f}, "
                          f"mean: {scores_np.mean():.4f}, std: {scores_np.std():.4f}")
                    
                    # Score histogram
                    score_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                    hist, _ = np.histogram(scores_np, bins=score_bins)
                    logger.debug(f"[DEBUG] Score histogram (0-0.1, 0.1-0.2, ...): {hist.tolist()}")
            else:
                logger.debug(f"[DEBUG] No detections from ROI head")
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