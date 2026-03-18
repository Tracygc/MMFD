import torch
import torch.nn as nn
import torch.nn.functional as F


class DWConv(nn.Module):

    def __init__(self, c1, c2, k=3, s=1, act=True):
        super().__init__()

        self.dw_conv = nn.Conv2d(c1, c1, kernel_size=k, stride=s, padding=k // 2, groups=c1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)

        self.pw_conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        x = self.act(self.bn1(self.dw_conv(x)))
        x = self.act(self.bn2(self.pw_conv(x)))
        return x


class SpatialDifferentialModule_kernel7(nn.Module):

    def __init__(self, channels, kernel_size=7):
        super(SpatialDifferentialModule_kernel7, self).__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True)
        )

        padding = kernel_size // 2
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, mr, mt):
        cat_feat = torch.cat([mr, mt], dim=1)
        feat_fused = self.fusion_conv(cat_feat)

        # Channel Pooling
        max_out, _ = torch.max(feat_fused, dim=1, keepdim=True)
        avg_out = torch.mean(feat_fused, dim=1, keepdim=True)
        spatial_pool = torch.cat([max_out, avg_out], dim=1)

        spatial_weight = self.spatial_conv(spatial_pool)

        mr_enhanced = mr * spatial_weight + mr
        mt_enhanced = mt * spatial_weight + mt
        return mr_enhanced, mt_enhanced




class CoordinateCommonModule(nn.Module):
    """[CCA: 坐标共模注意力]"""

    def __init__(self, channels, reduction=16):
        super(CoordinateCommonModule, self).__init__()
        mid_channels = max(channels // reduction, 1)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mid_channels, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mid_channels, channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, mr, mt):
        mc = mr + mt
        identity = mc
        n, c, h, w = mc.size()

        x_h = self.pool_h(mc)
        x_w = self.pool_w(mc).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))

        out = identity * a_w * a_h
        return out


class DynamicAttentionGate(nn.Module):

    def __init__(self, channels):
        super(DynamicAttentionGate, self).__init__()
        self.fusion_weight = nn.Sequential(
            nn.Conv2d(channels * 2, 2, kernel_size=1, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, feat_a, feat_b):
        cat_feat = torch.cat([feat_a, feat_b], dim=1)
        weights = self.fusion_weight(cat_feat)
        w_a = weights[:, 0:1, :, :]
        w_b = weights[:, 1:2, :, :]
        return w_a * feat_a + w_b * feat_b

class DAF(nn.Module):

    def __init__(self, channels, reduction=16):
        super(DAF, self).__init__()
        self.channels = channels
        self.sda_module = SpatialDifferentialModule_kernel7(channels)
        self.cca_module = CoordinateCommonModule(channels, reduction)
        self.dag_module = DynamicAttentionGate(channels)

    def forward(self, visible_feat, infrared_feat):
        mr_sda, mt_sda = self.sda_module(visible_feat, infrared_feat)
        m_cca = self.cca_module(visible_feat, infrared_feat)

        feat_rgb_refined = mr_sda + m_cca
        feat_ir_refined = mt_sda + m_cca

        fused_feat = self.dag_module(feat_rgb_refined, feat_ir_refined)
        return fused_feat



class Enhanced_CMFM(nn.Module):

    def __init__(self, channels):
        super(Enhanced_CMFM, self).__init__()
        self.channels = channels
        self.msdaf = DAF(channels)

        self.enhance_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, visible_feat, infrared_feat):
        msdaf_feat = self.msdaf(visible_feat, infrared_feat)
        enhanced_feat = self.enhance_conv(msdaf_feat)
        return enhanced_feat


class CMFM(nn.Module):

    def __init__(self, c1, c2=None):
        super(CMFM, self).__init__()
        self.channels = c1
        self.enhanced_cmmf = Enhanced_CMFM(self.channels)

    def forward(self, x):
        backbone_feat, visible_feat, infrared_feat = x

        target_dtype = next(self.enhanced_cmmf.parameters()).dtype
        target_device = next(self.enhanced_cmmf.parameters()).device

        visible_feat = visible_feat.to(dtype=target_dtype, device=target_device)
        infrared_feat = infrared_feat.to(dtype=target_dtype, device=target_device)
        backbone_feat = backbone_feat.to(dtype=target_dtype, device=target_device)

        if backbone_feat.size(1) != self.channels:
            backbone_feat = self._adjust_channels(backbone_feat, self.channels)
        if visible_feat.size(1) != self.channels:
            visible_feat = self._adjust_channels(visible_feat, self.channels)
        if infrared_feat.size(1) != self.channels:
            infrared_feat = self._adjust_channels(infrared_feat, self.channels)

        fused_feat = self.enhanced_cmmf(visible_feat, infrared_feat)
        output = backbone_feat + fused_feat

        return output

    def _adjust_channels(self, x, target_channels):
        current_channels = x.size(1)
        if current_channels == target_channels:
            return x
        elif current_channels < target_channels:
            padding = torch.zeros(x.size(0), target_channels - current_channels,
                                  x.size(2), x.size(3), device=x.device)
            return torch.cat([x, padding], dim=1)
        else:
            return x[:, :target_channels, :, :]


