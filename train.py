"""
ERes2NetV2 说话人识别训练入口

用法:
    python train.py                          # 使用默认配置训练
    python train.py --resume checkpoints/best_model.pth
    python train.py --variant 50             # 使用 ERes2NetV2-50
"""

import argparse
import sys
import random
import json
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.eres2netv2 import create_eres2netv2
from src.models.losses import AAMSoftmaxLoss, ArcFaceLoss
from src.datasets.speaker_dataset import SpeakerDataset, SpeakerDataLoader
from src.training.trainer import Trainer, WarmupCosineScheduler, load_checkpoint


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_split(data_dir, val_ratio=0.1, seed=42, splits_dir=None):
    """按说话人不重叠划分 train/val"""
    from pathlib import Path
    data_path = Path(data_dir)
    speaker_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    if not speaker_dirs:
        raise ValueError(f"没有找到说话人目录: {data_dir}")

    if splits_dir:
        splits_path = Path(splits_dir)
        splits_path.mkdir(parents=True, exist_ok=True)
        train_file = splits_path / "train_speakers.json"
        val_file = splits_path / "val_speakers.json"
        if train_file.exists() and val_file.exists():
            with open(train_file) as f:
                train_speakers = json.load(f)
            with open(val_file) as f:
                val_speakers = json.load(f)
            print(f"[SPLIT] 加载已有划分: train={len(train_speakers)}, val={len(val_speakers)}")
            return train_speakers, val_speakers

    rng = random.Random(seed)
    shuffled = speaker_dirs.copy()
    rng.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio))
    val_dirs, train_dirs = shuffled[:val_count], shuffled[val_count:]
    train_speakers = [str(d) for d in train_dirs]
    val_speakers = [str(d) for d in val_dirs]

    if splits_dir:
        with open(splits_path / "train_speakers.json", "w") as f:
            json.dump(train_speakers, f, indent=2)
        with open(splits_path / "val_speakers.json", "w") as f:
            json.dump(val_speakers, f, indent=2)
        print(f"[SPLIT] 划分已保存: train={len(train_speakers)}, val={len(val_speakers)}")
    return train_speakers, val_speakers


class SplitSpeakerDataset(SpeakerDataset):
    def __init__(self, speaker_dirs, transform=None, **kwargs):
        self.transform = transform
        self.speaker_dirs = speaker_dirs
        self.split = kwargs.pop("split", "train")
        self.n_mels = kwargs.pop("n_mels", 80)
        self.fixed_length = kwargs.pop("fixed_length", None)
        self.include_energy = kwargs.pop("include_energy", False)
        self.file_list, self.speaker_to_id, self.id_to_speaker, self.labels = [], {}, {}, []
        self._build_file_list()

    def _build_file_list(self):
        from pathlib import Path
        speaker_id = 0
        for speaker_dir_str in self.speaker_dirs:
            speaker_dir = Path(speaker_dir_str)
            if not speaker_dir.is_dir():
                continue
            speaker_name = speaker_dir.name
            self.speaker_to_id[speaker_name] = speaker_id
            self.id_to_speaker[speaker_id] = speaker_name
            for npy_path in sorted(speaker_dir.rglob("*.npy")):
                self.file_list.append(npy_path)
                self.labels.append(speaker_id)
            speaker_id += 1
        print(f"[DATA] {self.split} 数据集: 说话人数={len(self.speaker_to_id)}, 样本数={len(self.file_list)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print(" ERes2NetV2 说话人识别训练")
    print("=" * 60)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 数据
    features_dir = args.data_dir or config.get("data", {}).get("features_dir", "data/voxceleb/features")
    features_path = PROJECT_ROOT / features_dir
    if not features_path.exists():
        raise FileNotFoundError(f"特征数据目录不存在: {features_path}")

    sample_feat = np.load(list(features_path.rglob("*.npy"))[0])
    actual_n_mels = sample_feat.shape[1]
    print(f"\n特征维度: {sample_feat.shape} (n_mels={actual_n_mels})")

    val_ratio = config.get("data", {}).get("val_split_ratio", 0.1)
    val_seed = config.get("data", {}).get("val_random_seed", 42)
    splits_dir = str(PROJECT_ROOT / "data" / "voxceleb" / "splits")

    print(f"\n数据划分 (val_ratio={val_ratio}):")
    train_speaker_dirs, val_speaker_dirs = create_split(
        str(features_path), val_ratio=val_ratio, seed=val_seed, splits_dir=splits_dir)

    fixed_length = config.get("data", {}).get("fixed_length", 200)
    num_workers = config.get("data", {}).get("num_workers", 0)

    # 数据增强（仅训练集）
    augment_enabled = config.get("data", {}).get("augment", False)
    train_transform = None
    if augment_enabled:
        from src.datasets.augmentation import create_augmentation
        augment_config = config.get("data", {}).get("augment_config", {})
        train_transform = create_augmentation(augment_config)
        print(f"数据增强: 已启用 (SpecAugment + RandomGain)")
    else:
        print(f"数据增强: 未启用")

    train_dataset = SplitSpeakerDataset(
        speaker_dirs=train_speaker_dirs, split="train", n_mels=actual_n_mels,
        fixed_length=fixed_length, transform=train_transform)
    val_dataset = SplitSpeakerDataset(
        speaker_dirs=val_speaker_dirs, split="val", n_mels=actual_n_mels,
        fixed_length=fixed_length, transform=None)

    batch_size = args.batch_size or config.get("training", {}).get("batch_size", 128)
    pin_memory = config.get("data", {}).get("pin_memory", True)

    train_loader = SpeakerDataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers,
        shuffle=True, pin_memory=pin_memory, drop_last=True)
    val_loader = SpeakerDataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, pin_memory=pin_memory, drop_last=False)

    num_speakers = train_dataset.num_speakers
    print(f"\n说话人数 (训练集): {num_speakers}")

    # 模型
    model_variant = args.variant or config.get("model", {}).get("variant", "34")
    embedding_dim = config.get("model", {}).get("embedding_dim", 192)
    model = create_eres2netv2(
        variant=model_variant, n_mels=actual_n_mels, embedding_dim=embedding_dim,
        num_speakers=num_speakers, scale=config.get("model", {}).get("scale", 4),
        se_reduction=config.get("model", {}).get("se_reduction", 8),
        pool_attention_dim=config.get("model", {}).get("pool_attention_dim", 128),
        dropout=config.get("model", {}).get("dropout", 0.3))
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型: ERes2NetV2-{model_variant} ({total_params / 1e6:.2f}M 参数)")

    # 损失函数
    loss_type = config.get("loss", {}).get("type", "ce")
    margin = config.get("loss", {}).get("margin", 0.2)
    loss_scale = config.get("loss", {}).get("scale", 30.0)
    label_smoothing = config.get("loss", {}).get("label_smoothing",
                         config.get("training", {}).get("label_smoothing", 0.0))

    if loss_type == "ce":
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        print(f"损失函数: CrossEntropyLoss (label_smoothing={label_smoothing})")
    elif loss_type == "aam_softmax":
        criterion = AAMSoftmaxLoss(embedding_dim=embedding_dim, num_speakers=num_speakers,
                                   margin=margin, scale=loss_scale)
        print(f"损失函数: AAM-Softmax (margin={margin}, scale={loss_scale})")
    elif loss_type == "arcface":
        criterion = ArcFaceLoss(embedding_dim=embedding_dim, num_speakers=num_speakers,
                                margin=margin, scale=loss_scale)
        print(f"损失函数: ArcFace (margin={margin}, scale={loss_scale})")
    else:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        print(f"损失函数: CrossEntropyLoss (默认)")
    criterion = criterion.to(device)

    # 优化器
    opt_type = config.get("training", {}).get("optimizer", "SGD")
    base_lr = args.lr or config.get("training", {}).get("lr", 0.1)
    momentum = config.get("training", {}).get("momentum", 0.9)
    weight_decay = config.get("training", {}).get("weight_decay", 1e-4)

    if opt_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)
    elif opt_type == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)
    print(f"优化器: {opt_type} (lr={base_lr})")

    # 学习率调度
    epochs = args.epochs or config.get("training", {}).get("epochs", 80)
    warmup_epochs = config.get("training", {}).get("warmup_epochs", 5)
    scheduler = WarmupCosineScheduler(optimizer, base_lr=base_lr, warmup_epochs=warmup_epochs, total_epochs=epochs)
    print(f"LR 调度: Warmup({warmup_epochs}) + Cosine({epochs})")

    # 恢复训练
    start_epoch = 0
    resume_path = args.resume or config.get("checkpoint", {}).get("resume")
    if resume_path and Path(resume_path).exists():
        checkpoint = load_checkpoint(resume_path, model, optimizer, criterion)
        start_epoch = checkpoint["epoch"] + 1
        print(f"从 epoch {start_epoch} 继续训练")

    # 训练配置
    trainer_config = {
        "epochs": epochs,
        "grad_clip": config.get("training", {}).get("grad_clip", 1.0),
        "amp": config.get("training", {}).get("amp", True),
        "gradient_accumulation": config.get("training", {}).get("gradient_accumulation", 1),
        "early_stopping_patience": config.get("training", {}).get("early_stopping_patience", 15),
        "save_dir": config.get("checkpoint", {}).get("save_dir", "checkpoints"),
        "save_best": config.get("checkpoint", {}).get("save_best", True),
        "save_interval": config.get("checkpoint", {}).get("save_interval", 5),
        "tensorboard_dir": config.get("logging", {}).get("tensorboard_dir", "runs"),
        "log_interval": config.get("logging", {}).get("log_interval", 50),
        "gpu_monitor_interval": config.get("logging", {}).get("gpu_monitor_interval", 30),
        "loss_type": loss_type,
    }

    trainer = Trainer(
        model=model, train_loader=train_loader.loader, val_loader=val_loader.loader,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        device=device, config=trainer_config)

    trainer.train(start_epoch=start_epoch)


if __name__ == "__main__":
    main()
