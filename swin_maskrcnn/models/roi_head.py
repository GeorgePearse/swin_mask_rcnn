"""
ROI Head implementation for Mask R-CNN.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign, box_iou
from swin_maskrcnn.utils.logging import get_logger

logger = get_logger()


class BBoxHead(nn.Module):
    """Bounding box regression and classification head."""
    
    def __init__(
        self,
        num_classes,
        in_channels=256,
        roi_feat_size=7,
        fc_channels=1024
    ):
        super().__init__()
        self.num_classes = num_classes
        self.roi_feat_size = roi_feat_size
        self.fc_channels = fc_channels
        
        # Shared fully connected layers
        self.shared_fc1 = nn.Linear(in_channels * roi_feat_size * roi_feat_size, fc_channels)
        self.shared_fc2 = nn.Linear(fc_channels, fc_channels)
        
        # Classification and regression branches
        self.fc_cls = nn.Linear(fc_channels, num_classes + 1)  # +1 for background
        self.fc_reg = nn.Linear(fc_channels, num_classes * 4)
        
        self.init_weights()
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.shared_fc1.weight)
        nn.init.xavier_uniform_(self.shared_fc2.weight)
        nn.init.xavier_uniform_(self.fc_cls.weight)
        nn.init.xavier_uniform_(self.fc_reg.weight)
        nn.init.constant_(self.shared_fc1.bias, 0)
        nn.init.constant_(self.shared_fc2.bias, 0)
        # Initialize classification biases to encourage more detections
        # Use a more balanced initialization to avoid extreme background confidence
        nn.init.constant_(self.fc_cls.bias, 0)  # Start neutral
        self.fc_cls.bias.data[0] = 0.0  # Background bias
        self.fc_cls.bias.data[1:] = -1.0  # Slight foreground bias
        nn.init.constant_(self.fc_reg.bias, 0)
        
    def forward(self, roi_feats):
        """Forward pass."""
        # Flatten
        x = roi_feats.flatten(1)
        
        # Shared layers with dropout for better generalization
        x = F.relu(self.shared_fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.shared_fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Classification and regression
        cls_score = self.fc_cls(x)
        bbox_pred = self.fc_reg(x)
        
        return cls_score, bbox_pred


class MaskHead(nn.Module):
    """Mask prediction head."""
    
    def __init__(
        self,
        num_classes,
        in_channels=256,
        conv_channels=256,
        mask_size=14
    ):
        super().__init__()
        self.num_classes = num_classes
        self.mask_size = mask_size
        
        # Mask convolution layers
        self.mask_conv1 = nn.Conv2d(in_channels, conv_channels, 3, padding=1)
        self.mask_conv2 = nn.Conv2d(conv_channels, conv_channels, 3, padding=1)
        self.mask_conv3 = nn.Conv2d(conv_channels, conv_channels, 3, padding=1)
        self.mask_conv4 = nn.Conv2d(conv_channels, conv_channels, 3, padding=1)
        
        # Deconvolution
        self.deconv = nn.ConvTranspose2d(conv_channels, conv_channels, 2, stride=2)
        
        # Final prediction
        self.conv_logits = nn.Conv2d(conv_channels, num_classes, 1)
        
        self.init_weights()
        
    def init_weights(self):
        for m in [self.mask_conv1, self.mask_conv2, self.mask_conv3, self.mask_conv4, self.conv_logits]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        
        nn.init.kaiming_normal_(self.deconv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.deconv.bias, 0)
        
    def forward(self, roi_feats):
        """Forward pass."""
        x = F.relu(self.mask_conv1(roi_feats))
        x = F.relu(self.mask_conv2(x))
        x = F.relu(self.mask_conv3(x))
        x = F.relu(self.mask_conv4(x))
        
        x = F.relu(self.deconv(x))
        mask_pred = self.conv_logits(x)
        
        return mask_pred


class StandardRoIHead(nn.Module):
    """Standard ROI head for Mask R-CNN."""
    
    def __init__(
        self,
        bbox_roi_extractor=None,
        bbox_head=None,
        mask_roi_extractor=None,
        mask_head=None,
        num_classes=80,
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        pos_ratio=0.25,
        num_samples=512,
        cls_pos_weight=1.0,
        loss_cls_weight=1.0,
        loss_bbox_weight=1.0,
        loss_mask_weight=1.0
    ):
        super().__init__()
        
        # ROI extractors
        if bbox_roi_extractor is None:
            # For 14x14 feature map from 224x224 input, scale is 14/224 = 1/16
            bbox_roi_extractor = RoIAlign(
                output_size=(7, 7), spatial_scale=1.0/16.0, sampling_ratio=2
            )
        self.bbox_roi_extractor = bbox_roi_extractor
        
        if mask_roi_extractor is None:
            mask_roi_extractor = RoIAlign(
                output_size=(14, 14), spatial_scale=1.0/16.0, sampling_ratio=2
            )
        self.mask_roi_extractor = mask_roi_extractor
        
        # Heads
        if bbox_head is None:
            bbox_head = BBoxHead(num_classes=num_classes)
        self.bbox_head = bbox_head
        
        if mask_head is None:
            mask_head = MaskHead(num_classes=num_classes)
        self.mask_head = mask_head
        
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.pos_ratio = pos_ratio
        self.num_samples = num_samples
        
        # Loss weights
        self.cls_pos_weight = cls_pos_weight
        self.loss_cls_weight = loss_cls_weight
        self.loss_bbox_weight = loss_bbox_weight
        self.loss_mask_weight = loss_mask_weight
        
    def forward(self, features, proposals, targets=None):
        """Forward pass through ROI head."""
        labels = None
        bbox_targets = None
        mask_targets = None
        
        if self.training and targets is not None:
            # Sample proposals
            proposals, labels, bbox_targets, mask_targets = self.sample_proposals(
                proposals, targets
            )
        
        # Extract ROI features
        bbox_feats = self.extract_roi_features(features, proposals, self.bbox_roi_extractor)
        
        # Bbox head forward
        cls_scores, bbox_preds = self.bbox_head(bbox_feats)
        
        # Get positive proposals for mask prediction
        if self.training and targets is not None:
            # Labels is a list
            pos_proposals = []
            pos_labels = []
            for props, lbls in zip(proposals, labels):
                pos_mask = lbls > 0
                if pos_mask.any():
                    pos_proposals.append(props[pos_mask])
                    pos_labels.append(lbls[pos_mask])
            
            # Concatenate for mask head
            if pos_proposals:
                pos_proposals = torch.cat(pos_proposals)
                pos_labels = torch.cat(pos_labels)
            else:
                pos_proposals = torch.empty((0, 4), device=features[0].device)
                pos_labels = torch.empty((0,), dtype=torch.long, device=features[0].device)
        else:
            # During inference, use predicted classes
            pred_labels = cls_scores.argmax(dim=1)
            pos_inds = pred_labels > 0  # Non-background
            
            # Create batch indices for proper ROI extraction
            if pos_inds.any():
                # proposals should be a flat tensor with all proposals
                all_proposals = torch.cat(proposals) if isinstance(proposals, list) else proposals
                pos_proposals = all_proposals[pos_inds]
                pos_labels = pred_labels[pos_inds]
            else:
                pos_proposals = torch.empty((0, 4), device=features[0].device)
                pos_labels = torch.empty((0,), dtype=torch.long, device=features[0].device)
        
        # Mask head forward
        mask_preds = None
        if len(pos_proposals) > 0:
            # Convert pos_proposals to list format if needed for extract_roi_features
            if not isinstance(pos_proposals, list):
                pos_proposals = [pos_proposals]
            mask_feats = self.extract_roi_features(features, pos_proposals, self.mask_roi_extractor)
            mask_preds = self.mask_head(mask_feats)
        
        if self.training:
            losses = self.loss(cls_scores, bbox_preds, mask_preds, labels, bbox_targets, mask_targets)
            return losses
        else:
            results = self.get_results(cls_scores, bbox_preds, mask_preds, proposals, pos_labels)
            return results
    
    def extract_roi_features(self, features, proposals, roi_extractor):
        """Extract ROI features from multiple feature levels."""
        # Use FPN level assignment based on proposal size
        if isinstance(features, (list, tuple)):
            if isinstance(proposals, list):
                # Multiple batches
                all_rois = []
                for batch_idx, batch_proposals in enumerate(proposals):
                    if len(batch_proposals) > 0:
                        # Calculate canonical sizes and assign FPN levels
                        w = batch_proposals[:, 2] - batch_proposals[:, 0]
                        h = batch_proposals[:, 3] - batch_proposals[:, 1]
                        canonical_size = torch.sqrt(w * h)
                        
                        # k0 = 4, canonical_size = 224, eps = 1e-6
                        k0 = 4
                        canonical_scale = 224.0
                        eps = 1e-6
                        
                        # Calculate the appropriate FPN level for each proposal
                        level = torch.floor(k0 + torch.log2(canonical_size / canonical_scale + eps))
                        level = level.clamp(min=0, max=len(features) - 1).long()
                        
                        # Extract features for each level
                        for lvl in range(len(features)):
                            level_mask = level == lvl
                            if level_mask.any():
                                level_proposals = batch_proposals[level_mask]
                                level_rois = torch.zeros((len(level_proposals), 5), device=level_proposals.device)
                                level_rois[:, 0] = batch_idx
                                level_rois[:, 1:] = level_proposals
                                all_rois.append((level_rois, lvl))
                
                # Extract features level by level
                roi_feats_list = []
                for rois, lvl in all_rois:
                    level_feats = roi_extractor(features[lvl], rois)
                    roi_feats_list.append(level_feats)
                
                if roi_feats_list:
                    roi_feats = torch.cat(roi_feats_list, dim=0)
                else:
                    roi_feats = torch.empty((0, features[0].size(1), roi_extractor.output_size[0], roi_extractor.output_size[1]), device=features[0].device)
                return roi_feats
            else:
                # Single tensor - unlikely in practice but handle anyway
                features = features[2]  # Default to mid-level
        
        # Create proper ROI format with batch indices
        if isinstance(proposals, list):
            # Convert list of proposals to single tensor with batch indices
            rois = []
            for batch_idx, batch_proposals in enumerate(proposals):
                if len(batch_proposals) > 0:
                    batch_rois = torch.zeros((len(batch_proposals), 5), device=batch_proposals.device)
                    batch_rois[:, 0] = batch_idx
                    batch_rois[:, 1:] = batch_proposals
                    rois.append(batch_rois)
            if len(rois) > 0:
                rois = torch.cat(rois, dim=0)
            else:
                # No proposals - return empty tensor
                rois = torch.empty((0, 5), device=features.device)
        else:
            rois = proposals
            
        roi_feats = roi_extractor(features, rois)
        return roi_feats
    
    def sample_proposals(self, proposals, targets):
        """Sample proposals for training."""
        sampled_proposals = []
        sampled_labels = []
        sampled_bbox_targets = []
        sampled_mask_targets = []
        
        for img_proposals, img_targets in zip(proposals, targets):
            gt_bboxes = img_targets['boxes']
            gt_labels = img_targets['labels']
            gt_masks = img_targets.get('masks')
            
            # Handle empty ground truth case
            if len(gt_bboxes) == 0:
                # All proposals become negative samples
                labels = torch.zeros(len(img_proposals), dtype=torch.long, device=img_proposals.device)
                
                # Sample negative proposals
                neg_inds = torch.arange(len(img_proposals), device=img_proposals.device)
                if len(neg_inds) > self.num_samples:
                    neg_inds = neg_inds[torch.randperm(len(neg_inds))[:self.num_samples]]
                
                sampled_proposals.append(img_proposals[neg_inds])
                sampled_labels.append(labels[neg_inds])
                sampled_bbox_targets.append(torch.zeros((len(neg_inds), 4), device=img_proposals.device))
                
                if gt_masks is not None:
                    sampled_mask_targets.append(torch.zeros((len(neg_inds), 14, 14), device=img_proposals.device))
                
                continue
            
            # Calculate IoU between proposals and ground truth
            ious = box_iou(img_proposals, gt_bboxes)
            max_ious, matched_gt_idxs = ious.max(dim=1)
            
            # Assign labels
            labels = torch.zeros(len(img_proposals), dtype=torch.long, device=img_proposals.device)
            labels[max_ious >= self.pos_iou_thr] = gt_labels[matched_gt_idxs[max_ious >= self.pos_iou_thr]]
            labels[max_ious < self.neg_iou_thr] = 0  # Background
            
            # Sample proposals
            pos_inds = torch.where(labels > 0)[0]
            neg_inds = torch.where(labels == 0)[0]
            
            num_pos = int(self.num_samples * self.pos_ratio)
            num_pos = min(num_pos, len(pos_inds))
            num_neg = self.num_samples - num_pos
            num_neg = min(num_neg, len(neg_inds))
            
            pos_inds = pos_inds[torch.randperm(len(pos_inds))[:num_pos]]
            neg_inds = neg_inds[torch.randperm(len(neg_inds))[:num_neg]]
            
            sampled_inds = torch.cat([pos_inds, neg_inds])
            sampled_proposals.append(img_proposals[sampled_inds])
            sampled_labels.append(labels[sampled_inds])
            
            # Calculate targets for positive samples
            if len(pos_inds) > 0:
                pos_matched_gt_idxs = matched_gt_idxs[pos_inds]
                pos_gt_bboxes = gt_bboxes[pos_matched_gt_idxs]
                pos_proposals = img_proposals[pos_inds]
                bbox_targets = self.encode_bbox_targets(pos_proposals, pos_gt_bboxes)
                sampled_bbox_targets.append(bbox_targets)
                
                if gt_masks is not None:
                    pos_gt_masks = gt_masks[pos_matched_gt_idxs]
                    mask_targets = self.encode_mask_targets(pos_proposals, pos_gt_masks)
                    sampled_mask_targets.append(mask_targets)
            else:
                sampled_bbox_targets.append(torch.zeros((0, 4), device=img_proposals.device))
                if gt_masks is not None:
                    sampled_mask_targets.append(torch.zeros((0, 28, 28), device=img_proposals.device))
        
        return sampled_proposals, sampled_labels, sampled_bbox_targets, sampled_mask_targets
    
    def encode_bbox_targets(self, proposals, gt_bboxes):
        """Encode bbox targets."""
        # Similar to RPN encoding
        px = (proposals[:, 0] + proposals[:, 2]) * 0.5
        py = (proposals[:, 1] + proposals[:, 3]) * 0.5
        pw = proposals[:, 2] - proposals[:, 0]
        ph = proposals[:, 3] - proposals[:, 1]
        
        gx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) * 0.5
        gy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) * 0.5
        gw = gt_bboxes[:, 2] - gt_bboxes[:, 0]
        gh = gt_bboxes[:, 3] - gt_bboxes[:, 1]
        
        dx = (gx - px) / pw
        dy = (gy - py) / ph
        dw = torch.log(gw / pw)
        dh = torch.log(gh / ph)
        
        return torch.stack([dx, dy, dw, dh], dim=1)
    
    def encode_mask_targets(self, proposals, gt_masks):
        """Encode mask targets by cropping and resizing."""
        mask_size = 28
        mask_targets = []
        
        for proposal, mask in zip(proposals, gt_masks):
            # Convert to integer coordinates
            x1, y1, x2, y2 = proposal.int().tolist()
            
            # Compute the proposal dimensions
            w = max(x2 - x1, 1)
            h = max(y2 - y1, 1)
            
            # Create padded mask to handle boundary cases
            padded_mask = F.pad(mask.float(), (1, 1, 1, 1), mode='constant', value=0)
            
            # Adjust coordinates for padding
            x1_padded = x1 + 1
            y1_padded = y1 + 1
            x2_padded = x2 + 1
            y2_padded = y2 + 1
            
            # Crop and resize using grid sample for sub-pixel accuracy
            mask_h, mask_w = padded_mask.shape
            
            # Create normalized grid coordinates
            xs = torch.linspace(-1, 1, mask_size, device=mask.device)
            ys = torch.linspace(-1, 1, mask_size, device=mask.device)
            x_grid, y_grid = torch.meshgrid(xs, ys, indexing='xy')
            
            # Convert to original coordinates
            x_grid = (x_grid + 1) * w / 2 + x1_padded
            y_grid = (y_grid + 1) * h / 2 + y1_padded
            
            # Normalize to [-1, 1] for grid_sample
            x_grid = 2.0 * x_grid / mask_w - 1.0
            y_grid = 2.0 * y_grid / mask_h - 1.0
            
            grid = torch.stack([x_grid, y_grid], dim=-1).unsqueeze(0)
            
            # Perform grid sampling
            resized_mask = F.grid_sample(
                padded_mask.unsqueeze(0).unsqueeze(0),
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
            ).squeeze()
            
            # Apply threshold to create binary mask
            resized_mask = (resized_mask > 0.5).float()
            
            mask_targets.append(resized_mask)
            
        return torch.stack(mask_targets)
    
    def loss(self, cls_scores, bbox_preds, mask_preds, labels, bbox_targets, mask_targets):
        """Calculate losses with proper normalization."""
        losses = {}
        
        # Concatenate labels from all images
        all_labels = torch.cat(labels)
        
        # Calculate avg_factor for classification
        # Count valid samples (not ignored)
        avg_factor = max(len(all_labels), 1.0)
        
        # Classification loss with avg_factor
        if self.cls_pos_weight > 1.0:
            # Create weight tensor for each class (shape: [num_classes + 1])
            # Apply positive weight to all foreground classes
            num_total_classes = cls_scores.size(1)  # num_classes + 1
            weight = torch.ones(num_total_classes, dtype=torch.float32, device=cls_scores.device)
            weight[1:] = self.cls_pos_weight  # Apply to all foreground classes
            cls_loss = F.cross_entropy(cls_scores, all_labels, weight=weight, reduction='mean')
        else:
            cls_loss = F.cross_entropy(cls_scores, all_labels, reduction='mean')
            
        losses['cls_loss'] = cls_loss * self.loss_cls_weight
        
        # Bbox regression loss with normalization by positive samples
        pos_mask = all_labels > 0
        num_pos = max(pos_mask.sum().item(), 1.0)
        
        if len(bbox_targets) > 0 and any(len(t) > 0 for t in bbox_targets):
            if pos_mask.any():
                # Get positive predictions
                pos_bbox_preds = bbox_preds[pos_mask]
                pos_labels = all_labels[pos_mask]
                
                # Select predictions for corresponding classes
                # bbox_preds has shape [N, num_classes * 4]
                pos_bbox_preds = pos_bbox_preds.reshape(pos_bbox_preds.size(0), -1, 4)
                pos_bbox_preds = pos_bbox_preds[torch.arange(pos_bbox_preds.size(0)), pos_labels - 1]
                
                # Concatenate targets
                bbox_targets_concat = torch.cat(bbox_targets)
                
                assert pos_bbox_preds.shape[0] == bbox_targets_concat.shape[0], \
                    f"Shape mismatch: predictions {pos_bbox_preds.shape[0]} vs targets {bbox_targets_concat.shape[0]}"
                
                bbox_loss = F.smooth_l1_loss(pos_bbox_preds, bbox_targets_concat, reduction='sum')
                bbox_loss = bbox_loss / num_pos
            else:
                bbox_loss = torch.tensor(0.0, device=cls_scores.device)
        else:
            bbox_loss = torch.tensor(0.0, device=cls_scores.device)
            
        losses['bbox_loss'] = bbox_loss * self.loss_bbox_weight
        
        # Mask loss with normalization
        if mask_preds is not None and len(mask_targets) > 0:
            # Mask predictions are already only for positive samples
            # They come in shape [N_pos, num_classes, 28, 28]
            # We need to select the predictions for the correct classes
            if pos_mask.any():
                pos_labels = all_labels[pos_mask]
                mask_targets_concat = torch.cat(mask_targets)
                
                # Select predictions for corresponding classes
                # mask_preds shape: [N_pos, num_classes, 28, 28]
                # We need to select the mask prediction for each sample's true class
                selected_mask_preds = mask_preds[torch.arange(len(pos_labels)), pos_labels - 1]
                
                assert selected_mask_preds.shape[0] == mask_targets_concat.shape[0], \
                    f"Shape mismatch: predictions {selected_mask_preds.shape[0]} vs targets {mask_targets_concat.shape[0]}"
                mask_loss = F.binary_cross_entropy_with_logits(
                    selected_mask_preds, mask_targets_concat, reduction='sum'
                )
                mask_loss = mask_loss / num_pos
            else:
                mask_loss = torch.tensor(0.0, device=cls_scores.device)
        else:
            mask_loss = torch.tensor(0.0, device=cls_scores.device)
            
        losses['mask_loss'] = mask_loss * self.loss_mask_weight
        
        return losses
    
    def get_results(self, cls_scores, bbox_preds, mask_preds, proposals, pos_labels):
        """Get detection results."""
        # Apply softmax to classification scores
        cls_probs = F.softmax(cls_scores, dim=1)
        
        # Get predicted classes and scores
        scores, pred_labels = cls_probs.max(dim=1)
        
        # Enhanced debugging
        logger.debug(f"[ROI] Score stats - min: {scores.min():.4f}, max: {scores.max():.4f}, mean: {scores.mean():.4f}")
        logger.debug(f"[ROI] Background probs - min: {cls_probs[:, 0].min():.4f}, max: {cls_probs[:, 0].max():.4f}, mean: {cls_probs[:, 0].mean():.4f}")
        logger.debug(f"[ROI] Predicted labels distribution: {torch.bincount(pred_labels).tolist()}")
        logger.debug(f"[ROI] Number of non-background predictions: {(pred_labels > 0).sum()}")
        
        # Track number of proposals per image for proper splitting
        proposal_splits = [len(p) for p in proposals]
        
        # Convert proposals to single tensor if it's a list
        if isinstance(proposals, list):
            proposals_cat = torch.cat(proposals)
        else:
            proposals_cat = proposals
            proposals = [proposals]  # Make it a list for consistency
        
        # Filter out background predictions on concatenated tensors
        keep = pred_labels > 0
        scores_filtered = scores[keep]
        pred_labels_filtered = pred_labels[keep]
        proposals_filtered = proposals_cat[keep]
        bbox_preds_filtered = bbox_preds[keep]
        
        # Decode bbox predictions
        if len(proposals_filtered) > 0:
            bbox_preds_filtered = bbox_preds_filtered.reshape(len(proposals_filtered), -1, 4)
            bbox_preds_filtered = bbox_preds_filtered[torch.arange(len(proposals_filtered)), pred_labels_filtered - 1]
            
            # Apply deltas to proposals
            px = (proposals_filtered[:, 0] + proposals_filtered[:, 2]) * 0.5
            py = (proposals_filtered[:, 1] + proposals_filtered[:, 3]) * 0.5
            pw = proposals_filtered[:, 2] - proposals_filtered[:, 0]
            ph = proposals_filtered[:, 3] - proposals_filtered[:, 1]
            
            dx, dy, dw, dh = bbox_preds_filtered.unbind(dim=1)
            
            gx = dx * pw + px
            gy = dy * ph + py
            gw = pw * torch.exp(dw)
            gh = ph * torch.exp(dh)
            
            x1 = gx - gw * 0.5
            y1 = gy - gh * 0.5
            x2 = gx + gw * 0.5
            y2 = gy + gh * 0.5
            
            refined_bboxes = torch.stack([x1, y1, x2, y2], dim=1)
        else:
            refined_bboxes = proposals_filtered
        
        # Process mask predictions
        if mask_preds is not None and len(pos_labels) > 0:
            # Get mask predictions for positive samples
            mask_preds_filtered = mask_preds[torch.arange(len(pos_labels)), pos_labels - 1]
            mask_probs = torch.sigmoid(mask_preds_filtered)
        else:
            mask_probs = None
        
        # Split results back into per-image dictionaries
        results = []
        
        # Calculate starting indices for each image's proposals
        cumsum = torch.cumsum(torch.tensor([0] + proposal_splits), dim=0)
        
        # For each image
        for i in range(len(proposal_splits)):
            # Find which filtered detections belong to this image
            # Map filtered indices back to original proposal indices
            
            # First, create mapping from original to filtered indices
            keep_indices = torch.where(keep)[0]
            
            # Find which keep indices fall within this image's proposal range
            image_mask = (keep_indices >= cumsum[i]) & (keep_indices < cumsum[i+1])
            
            image_indices = torch.where(image_mask)[0]
            
            # Extract detections for this image
            if len(image_indices) > 0:
                image_boxes = refined_bboxes[image_indices]
                image_labels = pred_labels_filtered[image_indices]
                image_scores = scores_filtered[image_indices]
                image_masks = mask_probs[image_indices] if mask_probs is not None else None
            else:
                # No detections for this image
                device = proposals[0].device
                image_boxes = torch.empty((0, 4), device=device)
                image_labels = torch.empty((0,), dtype=torch.int64, device=device)
                image_scores = torch.empty((0,), device=device)
                image_masks = None
            
            results.append({
                'boxes': image_boxes,
                'labels': image_labels,
                'scores': image_scores,
                'masks': image_masks
            })
        
        return results