"""
说话人识别数据集模块
"""

from .speaker_dataset import SpeakerDataset, SpeakerDataLoader, NWayKShotSampler, create_datasets

__all__ = [
    "SpeakerDataset",
    "SpeakerDataLoader",
    "NWayKShotSampler",
    "create_datasets",
]
