import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import DFL
from ultralytics.nn.modules.conv import Conv
from ultralytics.utils.tal import dist2bbox, make_anchors
import torch.utils.checkpoint as cp

from ultralytics.nn.Addmodules.Recursive_AFPN import Upsample


__all__ = ("RAB_FPN_Add2","RAB_FPN_Add3",)


class Efficient_RecursiveFPN(nn.Module):
    """
    Lightweight version of RecursiveFPN to address Reviewer's computational overhead concern.
    Uses bottleneck design and eliminates heavy ASPP.
    """

    def __init__(self, out_indices=(0, 1, 2, 3), rfp_steps=2, stage_with_rfp=(False, True, True, True),
                 neck_out_channels=128):
        super().__init__()
        self.rfp_steps = rfp_steps
        self.out_indices = out_indices

        # 1. Bottleneck design: Squeeze channels to 1/4 to drastically reduce parameters
        hidden_channels = max(neck_out_channels // 4, 16)

        # Squeeze
        self.reduce_conv = nn.Conv2d(neck_out_channels, hidden_channels, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm2d(hidden_channels)
        self.act = nn.SiLU(inplace=True)

        # Lightweight spatial refinement (Depthwise Convolution instead of standard 3x3)
        self.spatial_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1,
                                      groups=hidden_channels, bias=False)
        self.norm2 = nn.BatchNorm2d(hidden_channels)

        # Expand
        self.expand_conv = nn.Conv2d(hidden_channels, neck_out_channels, kernel_size=1, bias=False)

        # Adaptive spatial weights (Kept to address Reviewer's adaptiveness comment)
        self.rfp_weight = nn.Conv2d(neck_out_channels, 1, kernel_size=1, bias=True)
        self.rfp_weight.weight.data.fill_(0)
        self.rfp_weight.bias.data.fill_(0)

    def rfp_forward(self, x):
        # Lightweight bottleneck forward
        x_reduced = self.act(self.norm1(self.reduce_conv(x)))
        x_spatial = self.act(self.norm2(self.spatial_conv(x_reduced)))
        out = self.expand_conv(x_spatial)
        return out

    def forward(self, x):
        for _ in range(self.rfp_steps - 1):
            x_idx = self.rfp_forward(x)

            # Adaptive scale weighting
            add_weight = torch.sigmoid(self.rfp_weight(x_idx))
            x = add_weight * x_idx + (1 - add_weight) * x
        return x




class RAB_FPN_Add2(nn.Module):
    """
    Revised Spatial-Aware BiFPN (2 levels) for Multispectral Fusion.
    Removes redundant scalar weights and uses DWConv to minimize parameters,
    while preserving strict spatial (pixel-level) alignment for IR-Visible features.
    """
    def __init__(self, c1, c2, level=0):
        super().__init__()
        c1_l, c1_h = c1[0], c1[1]
        self.level = level
        self.dim = c1_l, c1_h
        self.inter_dim = self.dim[self.level]
        compress_c = 8

        if level == 0:
            self.stride_level_1 = Upsample(c1_h, self.inter_dim)
        if level == 1:
            self.stride_level_0 = Conv(c1_l, self.inter_dim, 2, 2, 0)


        self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)


        self.weights_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)

        self.fusion_conv = nn.Sequential(

            nn.Conv2d(self.inter_dim, self.inter_dim, kernel_size=3, stride=1, padding=1, groups=self.inter_dim, bias=False),
            nn.BatchNorm2d(self.inter_dim),
            nn.SiLU(inplace=True),

            nn.Conv2d(self.inter_dim, self.inter_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_dim),
            nn.SiLU(inplace=True)
        )


        self.layer = Efficient_RecursiveFPN(neck_out_channels=self.inter_dim, stage_with_rfp=(True))

    def forward(self, x):
        x_level_0, x_level_1 = x[0], x[1]

        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
        elif self.level == 1:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = x_level_1


        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)


        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v), dim=1)
        levels_weight = F.softmax(self.weights_levels(levels_weight_v), dim=1)


        fused_out_reduced = (
            level_0_resized * levels_weight[:, 0:1, :, :] +
            level_1_resized * levels_weight[:, 1:2, :, :]
        )


        out = self.fusion_conv(fused_out_reduced)

        rout0 = self.layer(out)
        return rout0


class RAB_FPN_Add3(nn.Module):
    """
    Revised Spatial-Aware BiFPN (3 levels).
    """
    def __init__(self, c1, c2, level=0):
        super().__init__()
        c1_l, c1_m, c1_h = c1[0], c1[1], c1[2]
        self.level = level
        self.dim = c1_l, c1_m, c1_h
        self.inter_dim = self.dim[self.level]
        compress_c = 8

        if level == 0:
            self.stride_level_1 = Upsample(c1_m, self.inter_dim)
            self.stride_level_2 = Upsample(c1_h, self.inter_dim, scale_factor=4)
        if level == 1:
            self.stride_level_0 = Conv(c1_l, self.inter_dim, 2, 2, 0)
            self.stride_level_2 = Upsample(c1_h, self.inter_dim)
        if level == 2:
            self.stride_level_0 = Conv(c1_l, self.inter_dim, 4, 4, 0)
            self.stride_level_1 = Conv(c1_m, self.inter_dim, 2, 2, 0)

        self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weights_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.inter_dim, self.inter_dim, kernel_size=3, stride=1, padding=1, groups=self.inter_dim, bias=False),
            nn.BatchNorm2d(self.inter_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.inter_dim, self.inter_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_dim),
            nn.SiLU(inplace=True)
        )

        self.layer = Efficient_RecursiveFPN(neck_out_channels=self.inter_dim, stage_with_rfp=(True))

    def forward(self, x):
        x_level_0, x_level_1, x_level_2 = x[0], x[1], x[2]

        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 1:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)


        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), dim=1)
        w = F.softmax(self.weights_levels(levels_weight_v), dim=1)

        fused_out_reduced = (
            level_0_resized * w[:, 0:1, :, :] +
            level_1_resized * w[:, 1:2, :, :] +
            level_2_resized * w[:, 2:3, :, :]
        )

        out = self.fusion_conv(fused_out_reduced)
        rout0 = self.layer(out)
        return rout0
