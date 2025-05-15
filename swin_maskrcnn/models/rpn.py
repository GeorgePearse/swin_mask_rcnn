"""
Region Proposal Network (RPN) implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms, box_convert, box_iou


class AnchorGenerator:
    """Generate anchors for RPN."""
    
    def __init__(
        self,
        strides=[4, 8, 16, 32, 64],
        ratios=[0.5, 1.0, 2.0],
        scales=[8],
    ):
        self.strides = strides
        self.ratios = torch.tensor(ratios, dtype=torch.float32)
        self.scales = torch.tensor(scales, dtype=torch.float32)
        
    def generate_anchors(self, feat_size, stride, device):
        """Generate anchors for a single feature map."""
        h, w = feat_size
        
        # Calculate base anchor widths and heights
        scales = self.scales.to(device)
        ratios = self.ratios.to(device)
        base_size = stride
        
        areas = (base_size * scales) ** 2
        anchor_widths = torch.sqrt(areas[:, None] / ratios[None, :])
        anchor_heights = torch.sqrt(areas[:, None] * ratios[None, :])
        anchor_widths = anchor_widths.reshape(-1)
        anchor_heights = anchor_heights.reshape(-1)
        
        # Generate grid center coordinates
        y_centers = torch.arange(h, device=device) * stride + stride // 2
        x_centers = torch.arange(w, device=device) * stride + stride // 2
        y_centers, x_centers = torch.meshgrid(y_centers, x_centers, indexing='ij')
        centers = torch.stack([x_centers, y_centers], dim=2).reshape(-1, 2)
        
        # Generate anchors
        num_anchors = len(anchor_widths)
        anchors = torch.zeros((len(centers) * num_anchors, 4), device=device)
        
        for idx, (w, h) in enumerate(zip(anchor_widths, anchor_heights)):
            start_idx = idx * len(centers)
            end_idx = (idx + 1) * len(centers)
            anchors[start_idx:end_idx, 0] = centers[:, 0] - w / 2  # x1
            anchors[start_idx:end_idx, 1] = centers[:, 1] - h / 2  # y1
            anchors[start_idx:end_idx, 2] = centers[:, 0] + w / 2  # x2
            anchors[start_idx:end_idx, 3] = centers[:, 1] + h / 2  # y2
            
        return anchors


class RPNHead(nn.Module):
    """RPN head for Faster R-CNN."""
    
    def __init__(
        self,
        in_channels,
        feat_channels=256,
        anchor_generator=None,
        bbox_coder=None,
        pos_iou_thr=0.7,
        neg_iou_thr=0.3,
        num_neg_ratio=1.0,
    ):
        super().__init__()
        
        if anchor_generator is None:
            anchor_generator = AnchorGenerator()
        self.anchor_generator = anchor_generator
        self.num_anchors = len(anchor_generator.ratios) * len(anchor_generator.scales)
        
        # RPN conv and prediction layers
        self.rpn_conv = nn.Conv2d(in_channels, feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(feat_channels, self.num_anchors * 1, 1)
        self.rpn_reg = nn.Conv2d(feat_channels, self.num_anchors * 4, 1)
        
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.num_neg_ratio = num_neg_ratio
        
        # Loss functions
        self.loss_cls = nn.BCEWithLogitsLoss(reduction='none')
        self.loss_bbox = nn.SmoothL1Loss(reduction='none')
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        nn.init.normal_(self.rpn_conv.weight, std=0.01)
        nn.init.normal_(self.rpn_cls.weight, std=0.01)
        nn.init.normal_(self.rpn_reg.weight, std=0.01)
        nn.init.constant_(self.rpn_conv.bias, 0)
        nn.init.constant_(self.rpn_cls.bias, 0)
        nn.init.constant_(self.rpn_reg.bias, 0)
        
    def forward(self, features):
        """Forward pass through RPN head."""
        cls_scores = []
        bbox_preds = []
        
        for feat in features:
            x = self.rpn_conv(feat)
            x = F.relu(x)
            
            cls_score = self.rpn_cls(x)
            bbox_pred = self.rpn_reg(x)
            
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
            
        return cls_scores, bbox_preds
    
    def get_anchors(self, featmap_sizes, device):
        """Get anchors for all feature maps."""
        anchors = []
        for i, size in enumerate(featmap_sizes):
            stride = self.anchor_generator.strides[i]
            anchor = self.anchor_generator.generate_anchors(size, stride, device)
            anchors.append(anchor)
        
        return anchors
    
    def loss(self, cls_scores, bbox_preds, gt_bboxes, img_sizes):
        """Calculate RPN loss."""
        device = cls_scores[0].device
        featmap_sizes = [score.shape[-2:] for score in cls_scores]
        anchors = self.get_anchors(featmap_sizes, device)
        
        # Debug prints for input shapes
        print(f"Number of gt_bboxes: {len(gt_bboxes)}")
        print(f"Number of levels in cls_scores: {len(cls_scores)}")
        print(f"Batch size from cls_scores: {cls_scores[0].shape[0]}")
        print(f"Type of anchors: {type(anchors)}, length: {len(anchors) if isinstance(anchors, list) else 'N/A'}")
        
        # Match anchors to ground truth
        cls_targets = []
        bbox_targets = []
        pos_indices = []  # Track positive indices for bbox matching
        
        for img_idx, gt_bbox in enumerate(gt_bboxes):
            img_cls_targets = []
            img_bbox_targets = []
            img_pos_indices = []
            
            # Debug print
            print(f"Processing image {img_idx}: gt_bbox shape={gt_bbox.shape if torch.is_tensor(gt_bbox) else type(gt_bbox)}")
            
            # Handle empty ground truth boxes
            if len(gt_bbox) == 0:
                # If no ground truth boxes, all anchors are negative
                for level_idx, level_anchors in enumerate(anchors):
                    labels = torch.zeros(len(level_anchors), dtype=torch.int64, device=device)
                    img_cls_targets.append(labels)
                    img_bbox_targets.append(torch.zeros((0, 4), device=device))
                    img_pos_indices.append(torch.zeros((0,), dtype=torch.long, device=device))
                
                cls_targets.append(img_cls_targets)
                bbox_targets.append(img_bbox_targets)
                pos_indices.append(img_pos_indices)
                continue
            
            for level_idx, level_anchors in enumerate(anchors):
                # Calculate IoU between anchors and ground truth
                print(f"  Level {level_idx}: anchors shape={level_anchors.shape}, gt_bbox shape={gt_bbox.shape}")
                ious = box_iou(level_anchors, gt_bbox)
                max_ious, matched_gt_idx = ious.max(dim=1)
                
                # Assign labels
                labels = torch.zeros_like(max_ious, dtype=torch.int64)
                labels[max_ious >= self.pos_iou_thr] = 1
                labels[max_ious < self.neg_iou_thr] = 0
                labels[(max_ious >= self.neg_iou_thr) & (max_ious < self.pos_iou_thr)] = -1
                
                # Calculate bbox targets for positive anchors
                pos_mask = labels == 1
                if pos_mask.any():
                    pos_anchors = level_anchors[pos_mask]
                    matched_gt = gt_bbox[matched_gt_idx[pos_mask]]
                    bbox_target = self.encode_bbox(pos_anchors, matched_gt)
                    # Store indices where positive anchors are located
                    pos_indices_level = torch.where(pos_mask)[0]
                else:
                    bbox_target = torch.zeros((0, 4), device=device)
                    pos_indices_level = torch.zeros((0,), dtype=torch.long, device=device)
                
                img_cls_targets.append(labels)
                img_bbox_targets.append(bbox_target)
                img_pos_indices.append(pos_indices_level)
                
            cls_targets.append(img_cls_targets)
            bbox_targets.append(img_bbox_targets)
            pos_indices.append(img_pos_indices)
        
        # Calculate losses
        cls_loss = self._cls_loss(cls_scores, cls_targets)
        bbox_loss = self._bbox_loss(bbox_preds, bbox_targets, pos_indices)
        
        return {'rpn_cls_loss': cls_loss, 'rpn_bbox_loss': bbox_loss}
    
    def _cls_loss(self, cls_scores, cls_targets):
        """Calculate classification loss."""
        loss = torch.tensor(0.0, device=cls_scores[0].device)
        
        # Process each batch item separately
        batch_size = cls_scores[0].size(0)
        for batch_idx in range(batch_size):
            for level_idx, level_scores in enumerate(cls_scores):
                # Get scores for this batch item and level
                scores = level_scores[batch_idx].permute(1, 2, 0).reshape(-1)
                
                # Get corresponding targets
                if batch_idx < len(cls_targets) and level_idx < len(cls_targets[batch_idx]):
                    targets = cls_targets[batch_idx][level_idx]
                    
                    # Only calculate loss for valid targets (exclude -1)
                    valid_mask = targets >= 0
                    if valid_mask.any():
                        loss = loss + self.loss_cls(scores[valid_mask], targets[valid_mask].float()).sum()
        return loss
    
    def _bbox_loss(self, bbox_preds, bbox_targets, pos_indices):
        """Calculate bounding box regression loss."""
        loss = torch.tensor(0.0, device=bbox_preds[0].device)
        
        # Process each batch item separately
        batch_size = bbox_preds[0].size(0)
        for batch_idx in range(batch_size):
            for level_idx, level_preds in enumerate(bbox_preds):
                # Get predictions for this batch item and level
                preds = level_preds[batch_idx].permute(1, 2, 0).reshape(-1, 4)
                
                # Get corresponding targets and positive indices
                if (batch_idx < len(bbox_targets) and 
                    level_idx < len(bbox_targets[batch_idx]) and
                    batch_idx < len(pos_indices) and 
                    level_idx < len(pos_indices[batch_idx])):
                    
                    targets = bbox_targets[batch_idx][level_idx]
                    indices = pos_indices[batch_idx][level_idx]
                    
                    if targets.numel() > 0 and indices.numel() > 0:
                        # Index predictions at positive anchor locations
                        pos_preds = preds[indices]
                        loss = loss + self.loss_bbox(pos_preds, targets).sum()
        return loss
    
    def encode_bbox(self, anchors, gt_bboxes):
        """Encode bounding box targets."""
        # Convert to center format
        anchor_centers = box_convert(anchors, in_fmt='xyxy', out_fmt='cxcywh')
        gt_centers = box_convert(gt_bboxes, in_fmt='xyxy', out_fmt='cxcywh')
        
        # Calculate deltas
        dx = (gt_centers[:, 0] - anchor_centers[:, 0]) / anchor_centers[:, 2]
        dy = (gt_centers[:, 1] - anchor_centers[:, 1]) / anchor_centers[:, 3]
        dw = torch.log(gt_centers[:, 2] / anchor_centers[:, 2])
        dh = torch.log(gt_centers[:, 3] / anchor_centers[:, 3])
        
        return torch.stack([dx, dy, dw, dh], dim=1)
    
    def decode_bbox(self, anchors, bbox_pred):
        """Decode predicted bounding boxes."""
        # Convert to center format
        anchor_centers = box_convert(anchors, in_fmt='xyxy', out_fmt='cxcywh')
        
        # Apply deltas
        pred_cx = bbox_pred[:, 0] * anchor_centers[:, 2] + anchor_centers[:, 0]
        pred_cy = bbox_pred[:, 1] * anchor_centers[:, 3] + anchor_centers[:, 1]
        pred_w = torch.exp(bbox_pred[:, 2]) * anchor_centers[:, 2]
        pred_h = torch.exp(bbox_pred[:, 3]) * anchor_centers[:, 3]
        
        pred_centers = torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=1)
        pred_bboxes = box_convert(pred_centers, in_fmt='cxcywh', out_fmt='xyxy')
        
        return pred_bboxes
    
    def get_proposals(self, cls_scores, bbox_preds, img_sizes, cfg):
        """Generate region proposals."""
        device = cls_scores[0].device
        featmap_sizes = [score.shape[-2:] for score in cls_scores]
        anchors = self.get_anchors(featmap_sizes, device)
        
        proposals = []
        
        for img_idx in range(len(img_sizes)):
            img_cls_scores = []
            img_bbox_preds = []
            img_anchors = []
            
            # Collect predictions for this image
            for level_idx in range(len(cls_scores)):
                cls_score = cls_scores[level_idx][img_idx]
                bbox_pred = bbox_preds[level_idx][img_idx]
                level_anchors = anchors[level_idx]
                
                # Flatten predictions
                cls_score = cls_score.reshape(-1)
                bbox_pred = bbox_pred.reshape(-1, 4)
                
                img_cls_scores.append(cls_score)
                img_bbox_preds.append(bbox_pred)
                img_anchors.append(level_anchors)
            
            # Concatenate all levels
            img_cls_scores = torch.cat(img_cls_scores)
            img_bbox_preds = torch.cat(img_bbox_preds)
            img_anchors = torch.cat(img_anchors)
            
            # Apply sigmoid to get probabilities
            img_cls_scores = torch.sigmoid(img_cls_scores)
            
            # Decode bboxes
            decoded_bboxes = self.decode_bbox(img_anchors, img_bbox_preds)
            
            # Clip to image size
            img_h, img_w = img_sizes[img_idx]
            decoded_bboxes[:, [0, 2]] = decoded_bboxes[:, [0, 2]].clamp(0, img_w)
            decoded_bboxes[:, [1, 3]] = decoded_bboxes[:, [1, 3]].clamp(0, img_h)
            
            # Apply NMS
            nms_pre = cfg.get('nms_pre', 2000)
            if len(img_cls_scores) > nms_pre:
                # Keep top scoring proposals
                topk_scores, topk_inds = img_cls_scores.topk(nms_pre)
                decoded_bboxes = decoded_bboxes[topk_inds]
                img_cls_scores = topk_scores
            
            keep = batched_nms(
                decoded_bboxes,
                img_cls_scores,
                torch.zeros_like(img_cls_scores, dtype=torch.long),
                cfg.get('nms_thr', 0.7)
            )
            
            # Keep only top proposals after NMS
            max_per_img = cfg.get('max_per_img', 1000)
            keep = keep[:max_per_img]
            
            proposals.append(decoded_bboxes[keep])
        
        return proposals