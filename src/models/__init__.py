"""
说话人识别模型模块
"""

from .eres2netv2 import (
    ERes2NetV2,
    ERes2NetV2BasicBlock,
    ERes2NetV2Bottleneck,
    SELayer,
    AttentiveStatsPool,
    create_eres2netv2,
    eres2netv2_34,
    eres2netv2_50,
    eres2netv2_101,
    eres2netv2_152,
)
from .losses import AAMSoftmaxLoss, ArcFaceLoss

__all__ = [
    "ERes2NetV2",
    "ERes2NetV2BasicBlock",
    "ERes2NetV2Bottleneck",
    "SELayer",
    "AttentiveStatsPool",
    "create_eres2netv2",
    "eres2netv2_34",
    "eres2netv2_50",
    "eres2netv2_101",
    "eres2netv2_152",
    "AAMSoftmaxLoss",
    "ArcFaceLoss",
]
