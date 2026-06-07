"""
ERes2NetV2 Backbone for Speaker Recognition

适配说话人识别任务的 ERes2NetV2 架构。
输入：FBank 特征 (batch, n_mels, time_frames)，如 (batch, 80, 300)
输出：说话人 embedding (batch, embedding_dim)，如 (batch, 192)

核心创新：
- 分组特征提取：多尺度层级化特征处理
- 特征复用机制：后续分组复用前面分组输出
- 通道交互模块：轻量级注意力机制（SE-Block）
- 选择性残差融合：可学习权重的残差连接
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# 基础组件
# ──────────────────────────────────────────────

class ConvBNReLU(nn.Module):
    """2D 卷积 + BatchNorm + ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SELayer(nn.Module):
    """Squeeze-and-Excitation 通道注意力"""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        se = self.gap(x)
        se = self.relu(self.fc1(se))
        se = self.sigmoid(self.fc2(se))
        return x * se


# ──────────────────────────────────────────────
# ERes2NetV2 Basic Block (expansion=1, 用于 34 层)
# ──────────────────────────────────────────────

class ERes2NetV2BasicBlock(nn.Module):
    """
    ERes2NetV2 Basic Block (expansion=1)

    1x1 投影 → 分组特征提取 + 复用 → SE → 选择性残差融合
    """
    expansion = 1

    def __init__(self, in_channels, mid_channels, stride=1, scale=4,
                 downsample=None, se_reduction=8):
        super().__init__()
        self.scale = scale
        self.mid_channels = mid_channels
        group_channels = mid_channels // scale

        # 投影层：始终存在，处理通道数和空间降采样
        self.proj = ConvBNReLU(in_channels, mid_channels, kernel_size=1,
                               stride=stride, padding=0)

        # 分组 3x3 卷积（stride 已由 proj 处理，各组 stride=1）
        self.convs = nn.ModuleList()
        for i in range(scale):
            self.convs.append(ConvBNReLU(
                group_channels, group_channels,
                kernel_size=3, stride=1, padding=1))

        # SE-Block
        self.se = SELayer(mid_channels, reduction=se_reduction)

        # 选择性残差融合
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        # 投影到 mid_channels（含空间降采样）
        out = self.proj(x)

        # 分组特征提取 + 特征复用
        xs = torch.chunk(out, self.scale, dim=1)
        ys = []
        for i in range(self.scale):
            if i == 0:
                y_i = xs[i]
            elif i == 1:
                y_i = self.convs[i](xs[i])
            else:
                y_i = self.convs[i](xs[i] + ys[-1])
            ys.append(y_i)

        out = torch.cat(ys, dim=1)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.alpha * out + self.beta * identity
        return self.relu(out)


# ──────────────────────────────────────────────
# ERes2NetV2 Bottleneck Block (expansion=4, 用于 50/101/152)
# ──────────────────────────────────────────────

class ERes2NetV2Bottleneck(nn.Module):
    """
    ERes2NetV2 Bottleneck Block (expansion=4)

    1x1 降维 → 分组特征提取 + 复用 → SE → 1x1 升维 → 选择性残差融合
    """
    expansion = 4

    def __init__(self, in_channels, mid_channels, stride=1, scale=4,
                 downsample=None, se_reduction=8):
        super().__init__()
        self.scale = scale
        self.mid_channels = mid_channels
        group_channels = mid_channels // scale

        # 1x1 降维（含空间降采样）
        self.conv1 = ConvBNReLU(in_channels, mid_channels, kernel_size=1,
                                stride=stride, padding=0)

        # 分组 3x3 卷积（stride 已由 conv1 处理，各组 stride=1）
        self.convs = nn.ModuleList()
        for i in range(scale):
            self.convs.append(ConvBNReLU(
                group_channels, group_channels,
                kernel_size=3, stride=1, padding=1))

        # SE-Block
        self.se = SELayer(mid_channels, reduction=se_reduction)

        # 1x1 升维
        self.conv3 = nn.Conv2d(mid_channels, mid_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels * self.expansion)

        # 选择性残差融合
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)

        # 分组特征提取 + 特征复用
        xs = torch.chunk(out, self.scale, dim=1)
        ys = []
        for i in range(self.scale):
            if i == 0:
                y_i = xs[i]
            elif i == 1:
                y_i = self.convs[i](xs[i])
            else:
                y_i = self.convs[i](xs[i] + ys[-1])
            ys.append(y_i)

        out = torch.cat(ys, dim=1)
        out = self.se(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.alpha * out + self.beta * identity
        return self.relu(out)


# ──────────────────────────────────────────────
# Attentive Statistics Pooling
# ──────────────────────────────────────────────

class AttentiveStatsPool(nn.Module):
    """注意力加权统计池化（说话人识别标准池化方式）"""

    def __init__(self, in_dim, attention_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_dim, attention_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(attention_dim, in_dim, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x):
        """x: (batch, channels, time) → (batch, channels * 2)"""
        weights = self.attention(x)
        mean = torch.sum(x * weights, dim=2)
        diff = x * weights - mean.unsqueeze(2)
        std = torch.sqrt(torch.sum(diff ** 2, dim=2) / x.size(2) + 1e-9)
        return torch.cat([mean, std], dim=1)


# ──────────────────────────────────────────────
# ERes2NetV2 Backbone
# ──────────────────────────────────────────────

class ERes2NetV2(nn.Module):
    """
    ERes2NetV2 说话人识别模型

    输入：(batch, 1, n_mels, time_frames)
    输出：(batch, embedding_dim)
    """

    block_configs = {
        "34":  [3, 4, 6, 3],
        "50":  [3, 4, 6, 3],
        "101": [3, 4, 23, 3],
        "152": [3, 8, 36, 3],
    }

    channel_configs = {
        "34":  [64, 64, 128, 256, 512],
        "50":  [64, 256, 512, 1024, 2048],
        "101": [64, 256, 512, 1024, 2048],
        "152": [64, 256, 512, 1024, 2048],
    }

    def __init__(self, variant="34", n_mels=80, embedding_dim=192,
                 num_speakers=None, scale=4, se_reduction=8, pool_attention_dim=128):
        super().__init__()
        assert variant in self.block_configs, f"Unsupported variant: {variant}"
        self.variant = variant
        self.embedding_dim = embedding_dim
        self.num_speakers = num_speakers

        channels = self.channel_configs[variant]
        blocks = self.block_configs[variant]
        is_bottleneck = variant in ("50", "101", "152")
        block_class = ERes2NetV2Bottleneck if is_bottleneck else ERes2NetV2BasicBlock

        # ── Stem ──
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # ── Stages ──
        self.layers = nn.ModuleList()
        in_ch = channels[0]

        for stage_idx, (num_blocks, out_ch) in enumerate(zip(blocks, channels[1:])):
            downsample = None
            ds_stride = 2 if stage_idx > 0 else 1
            if stage_idx > 0 or (is_bottleneck and in_ch != out_ch):
                ds_out = out_ch
                downsample = nn.Sequential(
                    nn.Conv2d(in_ch, ds_out, kernel_size=1, stride=ds_stride, bias=False),
                    nn.BatchNorm2d(ds_out),
                )

            stage_blocks = []
            for i in range(num_blocks):
                stride = 2 if i == 0 and stage_idx > 0 else 1

                if is_bottleneck:
                    block_in = in_ch if i == 0 else out_ch
                    block_mid = out_ch // block_class.expansion
                else:
                    block_in = in_ch if i == 0 else out_ch
                    block_mid = out_ch

                stage_blocks.append(block_class(
                    in_channels=block_in, mid_channels=block_mid,
                    stride=stride, scale=scale,
                    downsample=downsample if i == 0 else None,
                    se_reduction=se_reduction,
                ))
                in_ch = out_ch

            self.layers.append(nn.Sequential(*stage_blocks))

        # ── Attentive Statistics Pooling ──
        final_channels = channels[-1]
        self.pool = AttentiveStatsPool(final_channels, attention_dim=pool_attention_dim)

        # ── BN + Embedding ──
        self.bn = nn.BatchNorm1d(final_channels * 2)
        self.dropout = nn.Dropout(p=0.3)
        self.embedding_fc = nn.Linear(final_channels * 2, embedding_dim)

        # ── Classifier (optional) ──
        self.classifier = nn.Linear(embedding_dim, num_speakers) if num_speakers else None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.stem(x)
        for layer in self.layers:
            x = layer(x)

        # (batch, channels, freq, time) → (batch, channels, time)
        x = x.mean(dim=2)
        x = self.pool(x)
        x = self.bn(x)
        x = self.dropout(x)
        embedding = self.embedding_fc(x)

        logits = self.classifier(embedding) if self.classifier is not None else None
        return embedding, logits

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────
# 工厂函数
# ──────────────────────────────────────────────

def create_eres2netv2(variant="34", n_mels=80, embedding_dim=192, num_speakers=None, **kwargs):
    return ERes2NetV2(variant=variant, n_mels=n_mels, embedding_dim=embedding_dim,
                      num_speakers=num_speakers, **kwargs)

def eres2netv2_34(**kwargs):
    return create_eres2netv2("34", **kwargs)

def eres2netv2_50(**kwargs):
    return create_eres2netv2("50", **kwargs)

def eres2netv2_101(**kwargs):
    return create_eres2netv2("101", **kwargs)

def eres2netv2_152(**kwargs):
    return create_eres2netv2("152", **kwargs)
