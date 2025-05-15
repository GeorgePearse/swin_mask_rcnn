"""
Feature Pyramid Network (FPN) implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    """
    Feature Pyramid Network.
    
    Args:
        in_channels_list (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels.
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level.
        end_level (int): Index of the end input backbone level.
        upsample_cfg (dict): Config dict for interpolate layer.
    """
    
    def __init__(
        self,
        in_channels_list,
        out_channels,
        num_outs,
        start_level=0,
        end_level=-1,
        upsample_cfg=None,
        activation='relu',
        norm_cfg=None,
    ):
        super().__init__()
        assert isinstance(in_channels_list, list)
        self.in_channels = in_channels_list
        self.out_channels = out_channels
        self.num_ins = len(in_channels_list)
        self.num_outs = num_outs
        self.activation = activation
        self.upsample_cfg = upsample_cfg.copy() if upsample_cfg else None
        
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels_list)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = nn.Conv2d(
                in_channels_list[i],
                out_channels,
                kernel_size=1
            )
            fpn_conv = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        
        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1
                )
                self.fpn_convs.append(extra_fpn_conv)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        
        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            upsample_cfg = {}
            if self.upsample_cfg:
                upsample_cfg.update(self.upsample_cfg)
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, **upsample_cfg
            )
        
        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels using max pooling
        if self.num_outs > len(outs):
            for i in range(self.num_outs - used_backbone_levels):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        return tuple(outs)