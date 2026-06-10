"""
说话人识别损失函数

包含：
- AAMSoftmaxLoss（Additive Angular Margin Softmax）
- ArcFaceLoss
- 标准 CrossEntropyLoss（baseline）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AAMSoftmaxLoss(nn.Module):
    """
    AAM-Softmax（Additive Angular Margin Softmax）损失

    说话人识别最常用的损失函数，通过在角度空间添加 margin 来增大类间距离。

    L = -log( exp(s * (cos(θ_yi) - m)) / (exp(s * (cos(θ_yi) - m)) + Σ_j≠yi exp(s * cos(θ_j))) )

    参数:
        embedding_dim: 说话人 embedding 维度
        num_speakers: 说话人数
        margin: 角度 margin（通常 0.2）
        scale: 缩放因子 s（通常 30）
    """

    def __init__(
        self,
        embedding_dim: int,
        num_speakers: int,
        margin: float = 0.2,
        scale: float = 30.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_speakers = num_speakers
        self.margin = margin
        self.scale = scale

        # 权重矩阵 W: (embedding_dim, num_speakers)
        self.weight = nn.Parameter(torch.randn(embedding_dim, num_speakers))
        nn.init.xavier_uniform_(self.weight)

        # 余弦 margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.threshold = math.cos(math.pi - margin)

    def forward(self, embedding: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        参数:
            embedding: (batch, embedding_dim) — 归一化后的说话人 embedding
            labels: (batch,) — 说话人标签

        返回:
            loss: 标量
        """
        # 归一化 embedding 和权重
        x_norm = F.normalize(embedding, dim=1)
        w_norm = F.normalize(self.weight, dim=0)

        # 计算余弦相似度 cos(θ)
        cos_theta = torch.matmul(x_norm, w_norm)  # (batch, num_speakers)
        cos_theta = cos_theta.clamp(-1, 1)

        # 计算 sin(θ)
        sin_theta = torch.sqrt(1.0 - cos_theta.pow(2))
        sin_theta = sin_theta.clamp(0, 1)

        # 计算 cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

        # 处理 θ > π - m 的情况（确保 dtype 一致以兼容 AMP）
        mm_tensor = torch.tensor(self.mm, dtype=cos_theta.dtype, device=cos_theta.device)
        mask = cos_theta > self.threshold
        cos_theta_m = cos_theta_m.clone()
        cos_theta_m[mask] = cos_theta[mask] - mm_tensor

        # One-hot 标签
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

        # 应用 margin：只在目标类上减去 margin
        output = cos_theta * (1.0 - one_hot) + cos_theta_m * one_hot

        # 乘以 scale 并计算 softmax cross entropy
        output *= self.scale
        loss = F.cross_entropy(output, labels)

        return loss

    def extra_repr(self):
        return f"embedding_dim={self.embedding_dim}, num_speakers={self.num_speakers}, " \
               f"margin={self.margin}, scale={self.scale}"


class ArcFaceLoss(nn.Module):
    """
    ArcFace 损失

    与 AAM-Softmax 类似，但 margin 的添加方式略有不同。

    参数:
        embedding_dim: 说话人 embedding 维度
        num_speakers: 说话人数
        margin: 角度 margin（通常 0.5）
        scale: 缩放因子 s（通常 64）
    """

    def __init__(
        self,
        embedding_dim: int,
        num_speakers: int,
        margin: float = 0.5,
        scale: float = 64.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_speakers = num_speakers
        self.margin = margin
        self.scale = scale

        self.weight = nn.Parameter(torch.randn(embedding_dim, num_speakers))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.threshold = math.cos(math.pi - margin)

    def forward(self, embedding: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(embedding, dim=1)
        w_norm = F.normalize(self.weight, dim=0)

        cos_theta = torch.matmul(x_norm, w_norm).clamp(-1, 1)
        sin_theta = torch.sqrt(1.0 - cos_theta.pow(2)).clamp(0, 1)

        # ArcFace: cos(θ + m)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        mm_tensor = torch.tensor(self.mm, dtype=cos_theta.dtype, device=cos_theta.device)
        mask = cos_theta > self.threshold
        cos_theta_m = cos_theta_m.clone()
        cos_theta_m[mask] = cos_theta[mask] - mm_tensor

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

        output = cos_theta * (1.0 - one_hot) + cos_theta_m * one_hot
        output *= self.scale

        return F.cross_entropy(output, labels)

    def extra_repr(self):
        return f"embedding_dim={self.embedding_dim}, num_speakers={self.num_speakers}, " \
               f"margin={self.margin}, scale={self.scale}"
