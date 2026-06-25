"""
说话人识别数据集和 DataLoader

用法:
    # 测试数据集加载
    python src/datasets/test_dataset.py

    # 在训练脚本中使用
    from src.datasets import SpeakerDataset, SpeakerDataLoader
    dataset = SpeakerDataset(data_dir="data/voxceleb/features", split="train")
    dataloader = SpeakerDataLoader(dataset, batch_size=64, num_workers=4)
"""

import os
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler


class SpeakerDataset(Dataset):
    """
    说话人识别数据集

    从预提取的 FBank 特征文件（.npy）加载数据。
    支持单目录或多目录加载（合并多个数据源）。

    目录结构：
        data_dir/
            speaker_001/
                utt_001.npy
                utt_002.npy
                ...
            speaker_002/
                ...

    多目录模式：
        data_dirs = ["data/voxceleb/features", "data/voxceleb2/features"]
        每个目录下的说话人会被合并，说话人 ID 全局唯一。

    返回:
        fbank: (n_frames, n_mels) 或 (n_mels, n_frames) 根据配置
        speaker_id: 说话人整数 ID
    """

    def __init__(
        self,
        data_dir: str = None,
        split: str = "train",
        n_mels: int = 80,
        fixed_length: Optional[int] = None,
        use_energy: bool = False,
        include_energy: bool = False,
        transform=None,
        data_dirs: List[str] = None,
    ):
        """
        参数:
            data_dir: 单个 FBank 特征目录（向后兼容）
            data_dirs: 多个 FBank 特征目录列表（优先于 data_dir）
            split: 'train' / 'val' / 'test'
            n_mels: FBank 维度
            fixed_length: 固定帧数（None 表示使用实际长度）
            use_energy: 是否使用能量特征
            include_energy: FBank 是否包含能量维度
            transform: 额外的数据增强变换
        """
        # 支持 data_dirs (多目录) 或 data_dir (单目录)
        if data_dirs is not None:
            self.data_dirs = [Path(d) for d in data_dirs]
        else:
            self.data_dirs = [Path(data_dir)] if data_dir else []

        if not self.data_dirs:
            raise ValueError("必须提供 data_dir 或 data_dirs 参数")

        self.split = split
        self.n_mels = n_mels
        self.fixed_length = fixed_length
        self.include_energy = include_energy
        self.transform = transform

        # 说话人目录（从所有数据源收集）
        self.speaker_dirs = []
        for d in self.data_dirs:
            if d.exists():
                self.speaker_dirs.extend(sorted([x for x in d.iterdir() if x.is_dir()]))

        # 构建文件列表和标签
        self.file_list = []  # List[Path]
        self.speaker_to_id = {}  # str -> int
        self.id_to_speaker = {}  # int -> str
        self.labels = []  # List[int]
        self.source_dir = {}  # speaker_name -> Path (记录数据来源)

        self._build_file_list()

    def _build_file_list(self):
        """构建文件列表和说话人标签映射（支持多数据源）"""
        file_idx = 0
        speaker_id = 0
        duplicate_speakers = 0

        for speaker_dir in self.speaker_dirs:
            speaker_name = speaker_dir.name

            # 如果说话人在多个数据源中重复，跳过（保留第一次出现的）
            if speaker_name in self.speaker_to_id:
                duplicate_speakers += 1
                continue

            self.speaker_to_id[speaker_name] = speaker_id
            self.id_to_speaker[speaker_id] = speaker_name
            self.source_dir[speaker_name] = speaker_dir.parent

            # 查找所有 .npy 文件（递归，因为结构是 idXXXX/video_id/0000N.npy）
            npy_files = sorted(speaker_dir.rglob("*.npy"))
            for npy_path in npy_files:
                self.file_list.append(npy_path)
                self.labels.append(speaker_id)
                file_idx += 1

            speaker_id += 1

        print(f"[DATA] {self.split} 数据集:")
        print(f"   数据源: {len(self.data_dirs)} 个目录")
        for d in self.data_dirs:
            n_spk = sum(1 for x in d.iterdir() if x.is_dir()) if d.exists() else 0
            print(f"     - {d} ({n_spk} 说话人)")
        if duplicate_speakers > 0:
            print(f"   重复说话人（已跳过）: {duplicate_speakers}")
        print(f"   说话人数: {len(self.speaker_to_id)}")
        print(f"   样本数: {len(self.file_list)}")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        返回:
            fbank: (n_mels, n_frames) tensor
            speaker_id: int
        """
        # 加载 FBank 特征
        fbank = np.load(str(self.file_list[idx]))  # (n_frames, n_mels)

        # 处理固定长度
        if self.fixed_length is not None:
            fbank = self._pad_or_truncate(fbank, self.fixed_length)

        # 转置为 (n_mels, n_frames)（模型输入格式）
        fbank = fbank.T  # (n_mels, n_frames)

        # 数据增强
        if self.transform is not None:
            fbank = self.transform(fbank)

        # 转换为 tensor
        fbank_tensor = torch.tensor(fbank, dtype=torch.float32)
        speaker_id = self.labels[idx]

        return fbank_tensor, speaker_id

    def _pad_or_truncate(self, fbank: np.ndarray, target_frames: int) -> np.ndarray:
        """填充或截断到固定帧数"""
        n_frames = fbank.shape[0]
        if n_frames >= target_frames:
            return fbank[:target_frames]
        else:
            # 重复填充
            repeats = (target_frames // n_frames) + 1
            fbank = np.tile(fbank, (repeats, 1))
            return fbank[:target_frames]

    def get_speaker_id(self, speaker_name: str) -> int:
        return self.speaker_to_id.get(speaker_name, -1)

    def get_speaker_name(self, speaker_id: int) -> str:
        return self.id_to_speaker.get(speaker_id, "Unknown")

    @property
    def num_speakers(self) -> int:
        return len(self.speaker_to_id)


class NWayKShotSampler(Sampler):
    """
    N-way K-shot 采样器
    每个 batch 包含 N 个说话人，每个说话人 K 条语音
    """

    def __init__(
        self,
        dataset: SpeakerDataset,
        n_way: int = 18,
        k_shot: int = 4,
    ):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.batch_size = n_way * k_shot

        # 按说话人分组
        self.speaker_to_indices = defaultdict(list)
        for idx, label in enumerate(dataset.labels):
            self.speaker_to_indices[label].append(idx)

        self.speakers = list(self.speaker_to_indices.keys())

    def __iter__(self):
        for _ in range(len(self)):
            # 随机选择 N 个说话人
            selected_speakers = random.sample(self.speakers, self.n_way)

            batch_indices = []
            for speaker in selected_speakers:
                indices = self.speaker_to_indices[speaker]
                # 随机选择 K 条语音
                selected = random.sample(indices, min(self.k_shot, len(indices)))
                batch_indices.extend(selected)

            yield batch_indices

    def __len__(self):
        return len(self.dataset) // self.batch_size


class SpeakerDataLoader:
    """说话人识别 DataLoader 工厂"""

    def __init__(
        self,
        dataset: SpeakerDataset,
        batch_size: int = 64,
        num_workers: int = 4,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = True,
        n_way: int = None,
        k_shot: int = None,
    ):
        """
        参数:
            dataset: SpeakerDataset 实例
            batch_size: batch 大小
            num_workers: 数据加载线程数
            shuffle: 是否打乱
            pin_memory: 是否使用 pin memory
            drop_last: 是否丢弃最后一个不完整的 batch
            n_way: N-way K-shot 的 N（如果为 None，使用普通采样）
            k_shot: N-way K-shot 的 K
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        if n_way is not None and k_shot is not None:
            # N-way K-shot 采样
            self.sampler = NWayKShotSampler(dataset, n_way, k_shot)
            self.loader = DataLoader(
                dataset,
                batch_sampler=self.sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=self._collate_fn,
            )
        else:
            # 普通 DataLoader
            self.loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
                collate_fn=self._collate_fn,
            )

    def _collate_fn(self, batch: List[Tuple[torch.Tensor, int]]):
        """自定义 collate 函数"""
        fbanks = []
        labels = []

        for fbank, label in batch:
            fbanks.append(fbank)
            labels.append(label)

        # 堆叠
        fbanks = torch.stack(fbanks, dim=0)  # (batch, n_mels, max_frames)
        labels = torch.tensor(labels, dtype=torch.long)

        return fbanks, labels

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)


def create_datasets(
    data_dir: str,
    n_mels: int = 80,
    fixed_length: int = 300,  # ~3 秒
    batch_size: int = 64,
    num_workers: int = 4,
    n_way: int = None,
    k_shot: int = None,
):
    """
    创建训练集和验证集的 DataLoader

    参数:
        data_dir: 数据根目录
        n_mels: FBank 维度
        fixed_length: 固定帧数
        batch_size: batch 大小
        num_workers: 数据加载线程数
        n_way: N-way K-shot 的 N
        k_shot: N-way K-shot 的 K

    返回:
        train_loader, val_loader, num_speakers
    """
    data_path = Path(data_dir)

    # 训练集
    train_dir = data_path / "dev" / "features"
    if train_dir.exists():
        train_dataset = SpeakerDataset(
            data_dir=str(train_dir),
            split="train",
            n_mels=n_mels,
            fixed_length=fixed_length,
        )
        train_loader = SpeakerDataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            n_way=n_way,
            k_shot=k_shot,
        )
    else:
        print(f"[WARN]  训练数据目录不存在: {train_dir}")
        train_loader = None
        train_dataset = None

    num_speakers = train_dataset.num_speakers if train_dataset else 0

    return train_loader, num_speakers


if __name__ == "__main__":
    print("=" * 60)
    print(" SpeakerDataset 测试")
    print("=" * 60)

    # 检查数据目录
    data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "voxceleb"
    print(f"[DIR] 数据目录: {data_dir}")

    if not data_dir.exists():
        print("[FAIL] 数据目录不存在，请先下载 VoxCeleb 数据集")
        print("   运行: python scripts/download_voxceleb.py")
        sys.exit(1)

    # 查找 FBank 特征目录
    feature_dir = data_dir / "dev" / "features"
    if feature_dir.exists():
        print(f"[DIR] 找到 FBank 特征目录: {feature_dir}")

        # 创建数据集
        dataset = SpeakerDataset(
            data_dir=str(feature_dir),
            split="train",
            n_mels=80,
            fixed_length=300,
        )

        # 创建 DataLoader
        dataloader = SpeakerDataLoader(
            dataset,
            batch_size=32,
            num_workers=0,  # 测试时用 0
            shuffle=True,
        )

        # 测试加载
        print(f"\n[DATA] 数据集信息:")
        print(f"   说话人数: {dataset.num_speakers}")
        print(f"   样本数: {len(dataset)}")
        print(f"   Batch 数: {len(dataloader)}")

        # 加载一个 batch
        for fbank, label in dataloader:
            print(f"\n[OK] 加载成功!")
            print(f"   FBank 形状: {fbank.shape}")
            print(f"   Label 形状: {label.shape}")
            print(f"   Label 范围: {label.min().item()} - {label.max().item()}")
            break
    else:
        print(f"[WARN]  FBank 特征目录不存在: {feature_dir}")
        print("   请先运行特征提取: python scripts/extract_features.py --batch --data-dir data/voxceleb/dev")
