import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicAttentionGate(nn.Module):

    def __init__(self, channels, reduction=16):
        super(DynamicAttentionGate, self).__init__()
        self.channels = channels

        # 确保中间通道数至少为1
        self.reduction = max(reduction, 1)
        self.mid_channels = max(channels // self.reduction, 1)

        # 全局特征提取
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 门控网络
        self.gate_net = nn.Sequential(
            nn.Linear(channels * 2, self.mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.mid_channels, channels * 2),
            nn.Sigmoid()
        )

        # 动态权重调整
        self.weight_alpha = nn.Parameter(torch.tensor(0.5))
        self.weight_beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, visible_feat, infrared_feat):
        batch_size = visible_feat.size(0)

        # 全局特征提取
        visible_global = self.global_pool(visible_feat).view(batch_size, -1)
        infrared_global = self.global_pool(infrared_feat).view(batch_size, -1)

        # 拼接特征
        concat_feat = torch.cat([visible_global, infrared_global], dim=1)

        # 门控权重生成
        gate_weights = self.gate_net(concat_feat)

        # 动态调整形状 - 修复形状问题
        visible_weights = gate_weights[:, :self.channels].unsqueeze(-1).unsqueeze(-1)
        infrared_weights = gate_weights[:, self.channels:].unsqueeze(-1).unsqueeze(-1)

        # 应用动态权重
        visible_weights = visible_weights * self.weight_alpha
        infrared_weights = infrared_weights * self.weight_beta

        # 权重归一化
        total_weights = visible_weights + infrared_weights + 1e-8
        visible_weights = visible_weights / total_weights
        infrared_weights = infrared_weights / total_weights

        return visible_weights, infrared_weights
    

class MultiScaleDifferentialAttention_DAG(nn.Module):

    def __init__(self, channels, reduction=16):
        super(MultiScaleDifferentialAttention_DAG, self).__init__()
        self.channels = channels

        # 确保中间通道数至少为1
        self.mid_channels = max(channels // reduction, 1)

        # 差模注意力组件
        self.diff_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.diff_max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享卷积层 - 轻量化设计
        self.shared_conv = nn.Sequential(
            nn.Conv2d(channels, self.mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_channels, channels, 1, bias=False)
        )

        # 共模注意力组件
        self.common_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.common_fc_visible = nn.Sequential(
            nn.Linear(channels, self.mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.mid_channels, channels),
            nn.Sigmoid()
        )
        self.common_fc_infrared = nn.Sequential(
            nn.Linear(channels, self.mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.mid_channels, channels),
            nn.Sigmoid()
        )

        # 信息感知模块 - 用于获取权重Wv和Wi
        self.info_perception = InfoPerceptionModule(channels)

        # 动态注意力门控
        self.attention_gate = DynamicAttentionGate(channels, reduction)

        # 多尺度处理
        self.scale_reduce_conv = nn.Conv2d(channels, channels // 2, 1)

        # 可学习的对比度参数C
        self.contrast_param = nn.Parameter(torch.tensor(1.0))

    def forward(self, visible_feat, infrared_feat):
        """
        Args:
            visible_feat: 可见光特征 [B, C, H, W]
            infrared_feat: 红外特征 [B, C, H, W]
        Returns:
            fused_feat: 融合后的特征 [B, C, H, W]
        """
        # 1. 差模注意力
        m_da = self.differential_attention(visible_feat, infrared_feat)

        # 2. 共模注意力
        m_ca = self.common_attention(visible_feat, infrared_feat)

        # # 3. 多尺度融合
        # fused_feat = m_da + m_ca

        # 获取动态注意力权重
        wda, wca  = self.attention_gate(visible_feat, infrared_feat)
        
        # # # 通过信息感知模块获取权重
        # wda, wca = self.info_perception(m_da, m_ca)

        # # # 按比例相加得到差模注意力输出
        fused_feat = wda *  m_da  + wca * m_ca

        return fused_feat

    def differential_attention(self, mr, mt):
        """差模注意力"""
        # 计算差模特征 MD = MR - MT
        md = mr - mt

        # 全局平均池化和最大池化
        s1 = self.diff_avg_pool(md)  # [B, C, 1, 1]
        s2 = self.diff_max_pool(md)  # [B, C, 1, 1]

        # 共享卷积处理
        v1 = self.shared_conv(s1)  # [B, C, 1, 1]
        v2 = self.shared_conv(s2)  # [B, C, 1, 1]

        # 得到通道注意力图 VDA
        vda = torch.sigmoid(v1 + v2)  # [B, C, 1, 1]

        # 与输入特征相乘并通过跳跃连接
        enhanced_mr = mr * vda + mr  # 跳跃连接保留原始特征
        enhanced_mt = mt * vda + mt  # 跳跃连接保留原始特征

        # 通过信息感知模块获取权重
        wv, wi = self.info_perception(enhanced_mr, enhanced_mt)

        # 按比例相加得到差模注意力输出
        m_da = wv * enhanced_mr + wi * enhanced_mt

        return m_da

    def common_attention(self, mr, mt):
        """共模注意力"""
        # 计算共模特征 MC = MR + MT
        mc = mr + mt

        # 全局平均池化
        mc_pooled = self.common_avg_pool(mc).squeeze(-1).squeeze(-1)  # [B, C]

        # 分别得到可见光和红外的注意力图
        vv_ca = self.common_fc_visible(mc_pooled).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        vi_ca = self.common_fc_infrared(mc_pooled).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        # 应用注意力图并通过跳跃连接
        visible_enhanced = mr * vv_ca + mr  # 跳跃连接
        infrared_enhanced = mt * vi_ca + mt  # 跳跃连接

        # 红外分支利用对比度进行加权
        infrared_weighted = self.contrast_param * infrared_enhanced

        # 共模注意力输出
        m_ca = visible_enhanced + infrared_weighted

        return m_ca

class InfoPerceptionModule(nn.Module):
    """信息感知模块 - 用于获取权重Wv和Wi"""

    def __init__(self, channels, reduction=16):
        super(InfoPerceptionModule, self).__init__()
        self.channels = channels
        self.mid_channels = max(channels // reduction, 1)

        # 全局特征提取
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 权重生成网络
        self.weight_net = nn.Sequential(
            nn.Linear(channels * 2, self.mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.mid_channels, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, visible_feat, infrared_feat):
        batch_size = visible_feat.size(0)

        # 全局特征提取
        visible_global = self.global_pool(visible_feat).view(batch_size, -1)
        infrared_global = self.global_pool(infrared_feat).view(batch_size, -1)

        # 拼接特征
        concat_feat = torch.cat([visible_global, infrared_global], dim=1)

        # 生成权重 [B, 2]
        weights = self.weight_net(concat_feat)

        # 分离权重
        wv = weights[:, 0].view(batch_size, 1, 1, 1)  # [B, 1, 1, 1]
        wi = weights[:, 1].view(batch_size, 1, 1, 1)  # [B, 1, 1, 1]

        return wv, wi


class EnhancedCMMFWithMSDAF_DAG(nn.Module):
    """增强的CMMF模块 - 包含多尺度差分注意力融合"""

    def __init__(self, channels):
        super(EnhancedCMMFWithMSDAF_DAG, self).__init__()
        self.channels = channels

        # 单尺度差分注意力融合模块
        self.msdaf = MultiScaleDifferentialAttention_DAG(channels)

        # 特征增强
        self.enhance_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, visible_feat, infrared_feat):
        """
        Args:
            visible_feat: 可见光分支特征 [B, C, H, W]
            infrared_feat: 红外分支特征 [B, C, H, W]
        Returns:
            fused_feat: 融合后的特征 [B, C, H, W]
        """
        # 多尺度差分注意力融合
        msdaf_feat = self.msdaf(visible_feat, infrared_feat)

        # 特征增强
        enhanced_feat = self.enhance_conv(msdaf_feat)

        return enhanced_feat
    


class MMFD(nn.Module):
    "The code will be released soon."


# 适配YOLO的ZeroConv2d模块
class ZeroConv2d(nn.Module):
    """ZeroConv2d模块，权重初始化为0"""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super(ZeroConv2d, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)
        self.init_zero_weights()

    def init_zero_weights(self):
        """初始化权重为0"""
        nn.init.constant_(self.conv.weight, 0)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


# 辅助函数 - 自动填充
def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]

    return p
