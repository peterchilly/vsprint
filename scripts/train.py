"""
ERes2NetV2 说话人识别训练脚本 v3

修复项:
1. 优化器支持 AdamW (从 config 读取, 不再硬编码 SGD)
2. 数据增强接入 (SpecAugment + RandomGain)
3. 训练准确率从 AAM-Softmax logits 计算 (不再用未训练的 classifier head)
4. 验证使用 embedding-based EER (不再用 CE on logits)
5. fixed_length 统一
6. AMP 兼容性修复

用法:
    # v3 训练 (使用 v2 配置)
    python scripts/train.py --config configs/train_v2.yaml

    # 从 checkpoint 恢复
    python scripts/train.py --config configs/train_v2.yaml --resume checkpoints_v2/best_model.pth

    # 快速测试 1 epoch
    python scripts/train.py --config configs/train_v2.yaml --dry-run
"""

import argparse
import os
import sys
import time
import json
import random
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import yaml

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.eres2netv2 import create_eres2netv2
from src.models.losses import AAMSoftmaxLoss, ArcFaceLoss
from src.datasets.speaker_dataset import SpeakerDataset
from src.datasets.augmentation import create_augmentation


def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class RemappedSubset(Subset):
    """Subset that remaps labels to 0..N-1"""
    def __init__(self, dataset, indices, label_map):
        super().__init__(dataset, indices)
        self.label_map = label_map

    def __getitem__(self, idx):
        fbank, label = self.dataset[self.indices[idx]]
        return fbank, self.label_map[label]

    def __getitems__(self, indices):
        return [self.__getitem__(idx) for idx in indices]


def split_dataset_by_speakers(dataset, val_ratio=0.1, seed=42):
    """按说话人划分训练集和验证集，同时重映射标签为 0~N-1"""
    rng = random.Random(seed)
    speakers = sorted(list(dataset.speaker_to_id.keys()))
    num_val_speakers = max(1, int(len(speakers) * val_ratio))
    val_speakers = set(rng.sample(speakers, num_val_speakers))
    train_speakers = sorted(set(speakers) - val_speakers)

    train_label_map = {dataset.speaker_to_id[spk]: i for i, spk in enumerate(train_speakers)}
    val_label_map = {dataset.speaker_to_id[spk]: i for i, spk in enumerate(sorted(val_speakers))}

    train_indices = []
    val_indices = []
    for idx, label in enumerate(dataset.labels):
        speaker_name = dataset.id_to_speaker[label]
        if speaker_name in val_speakers:
            val_indices.append(idx)
        elif label in train_label_map:
            train_indices.append(idx)

    train_subset = RemappedSubset(dataset, train_indices, train_label_map)
    val_subset = RemappedSubset(dataset, val_indices, val_label_map)

    print(f"[SPLIT] 数据集划分:")
    print(f"   总说话人: {len(speakers)}")
    print(f"   训练集: {len(train_speakers)} 说话人, {len(train_indices)} 样本")
    print(f"   验证集: {len(val_speakers)} 说话人, {len(val_indices)} 样本")

    return train_subset, val_subset, len(train_speakers)


# ──────────────────────────────────────────────
# AAM-Softmax 辅助：从 loss 模块获取分类 logits
# ──────────────────────────────────────────────

def get_aam_logits(criterion, embeddings):
    """从 AAMSoftmaxLoss 的权重矩阵计算分类 logits（用于训练准确率）"""
    if isinstance(criterion, (AAMSoftmaxLoss, ArcFaceLoss)):
        x_norm = F.normalize(embeddings.float(), dim=1)
        w_norm = F.normalize(criterion.weight, dim=0)
        cos_theta = torch.matmul(x_norm, w_norm).clamp(-1, 1)
        return cos_theta * criterion.scale
    return None


# ──────────────────────────────────────────────
# Embedding 验证 (EER)
# ──────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model, loader, device):
    """提取所有样本的 L2-normalized embedding"""
    model.eval()
    all_embeddings, all_labels = [], []
    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        if features.dim() == 3:
            features = features.unsqueeze(1)
        embedding, _ = model(features)
        embedding = F.normalize(embedding, dim=1)
        all_embeddings.append(embedding.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_embeddings, axis=0), np.concatenate(all_labels, axis=0)


def compute_eer(pos_scores, neg_scores):
    """计算 EER (Equal Error Rate)"""
    thresholds = np.linspace(min(pos_scores.min(), neg_scores.min()),
                             max(pos_scores.max(), neg_scores.max()), 1000)
    best_eer, best_threshold = 1.0, 0.0
    for threshold in thresholds:
        far = np.mean(neg_scores >= threshold)
        frr = np.mean(pos_scores < threshold)
        eer = (far + frr) / 2.0
        if eer < best_eer:
            best_eer, best_threshold = eer, threshold
    return best_eer, best_threshold


def compute_min_dcf(pos_scores, neg_scores, c_miss=1.0, c_fa=1.0, p_target=0.5):
    """计算 minDCF"""
    thresholds = np.linspace(min(pos_scores.min(), neg_scores.min()),
                             max(pos_scores.max(), neg_scores.max()), 1000)
    min_dcf, best_threshold = float('inf'), 0.0
    for threshold in thresholds:
        p_miss = np.mean(pos_scores < threshold)
        p_fa = np.mean(neg_scores >= threshold)
        dcf = c_miss * p_miss * p_target + c_fa * p_fa * (1 - p_target)
        if dcf < min_dcf:
            min_dcf, best_threshold = dcf, threshold
    baseline = min(c_miss * p_target, c_fa * (1 - p_target))
    return min_dcf / baseline if baseline > 0 else min_dcf, best_threshold


@torch.no_grad()
def validate_embedding(model, val_loader, device):
    """基于 embedding 的验证 (EER/minDCF)"""
    embeddings, labels = extract_embeddings(model, val_loader, device)

    unique_labels = np.unique(labels)
    speaker_indices = {lbl: np.where(labels == lbl)[0] for lbl in unique_labels}

    rng = np.random.RandomState(42)

    # 正样本对（同一说话人）
    pos_scores = []
    for lbl in unique_labels:
        indices = speaker_indices[lbl]
        if len(indices) < 2:
            continue
        n_pairs = min(len(indices) * (len(indices) - 1) // 2, 500)
        for _ in range(n_pairs):
            i, j = rng.choice(indices, size=2, replace=False)
            score = float(np.sum(embeddings[i] * embeddings[j]))
            pos_scores.append(score)

    # 负样本对（不同说话人）
    neg_scores = []
    target_neg = len(pos_scores)
    for _ in range(target_neg * 3):
        lbl_a, lbl_b = rng.choice(unique_labels, size=2, replace=False)
        i = rng.choice(speaker_indices[lbl_a])
        j = rng.choice(speaker_indices[lbl_b])
        score = float(np.sum(embeddings[i] * embeddings[j]))
        neg_scores.append(score)
        if len(neg_scores) >= target_neg:
            break

    pos_scores = np.array(pos_scores)
    neg_scores = np.array(neg_scores)

    eer, eer_threshold = compute_eer(pos_scores, neg_scores)
    min_dcf, dcf_threshold = compute_min_dcf(pos_scores, neg_scores)

    return {
        "eer": eer,
        "min_dcf": min_dcf,
        "eer_threshold": eer_threshold,
        "pos_mean": float(np.mean(pos_scores)),
        "neg_mean": float(np.mean(neg_scores)),
        "pos_std": float(np.std(pos_scores)),
        "neg_std": float(np.std(neg_scores)),
        "n_pos_pairs": len(pos_scores),
        "n_neg_pairs": len(neg_scores),
    }


# ──────────────────────────────────────────────
# 训练一个 epoch
# ──────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch, config, device):
    """训练一个 epoch

    修复: 训练准确率从 AAM-Softmax logits 计算, 不用 model.classifier
    修复: AMP 下 AAM-Softmax 的 sqrt 操作导致 inf 梯度 — 整个前向+损失在 FP32 计算
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    grad_norm_sum = 0
    skipped_steps = 0

    log_interval = config.get("logging", {}).get("log_interval", 50)
    amp_enabled = config["training"]["amp"] and device.type == "cuda"

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch_idx, (fbanks, labels) in enumerate(pbar):
        fbanks = fbanks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if amp_enabled:
            # 模型前向用 AMP (卷积/BN 在 FP16 加速)
            with autocast("cuda"):
                embeddings, logits = model(fbanks)
            # 损失在 FP32 计算 (AAM-Softmax 的 sqrt/trig 操作在 FP16 下会产生 inf 梯度)
            embeddings = embeddings.float()
            loss = criterion(embeddings, labels)
        else:
            embeddings, logits = model(fbanks)
            loss = criterion(embeddings, labels)

        scaler.scale(loss).backward()

        # 梯度裁剪 (先 unscale 再 clip)
        max_norm = config["training"]["grad_clip"]
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        grad_norm_val = grad_norm.item()
        if not math.isfinite(grad_norm_val):
            skipped_steps += 1
            grad_norm_val = 0.0
            optimizer.zero_grad()
            scaler.update()
            continue
        grad_norm_sum += grad_norm_val

        scaler.step(optimizer)
        scaler.update()

        # 统计：用 AAM-Softmax 的 logits 计算准确率（而非 model.classifier）
        total_loss += loss.item()
        aam_logits = get_aam_logits(criterion, embeddings)
        if aam_logits is not None:
            _, predicted = aam_logits.max(1)
        else:
            _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        # 进度条更新
        avg_loss = total_loss / max(total, 1)
        acc = 100.0 * correct / max(total, 1)
        avg_grad = grad_norm_sum / max(batch_idx + 1 - skipped_steps, 1)
        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{acc:.1f}%", "grad": f"{avg_grad:.2f}", "skip": skipped_steps})

        # 定期日志
        if (batch_idx + 1) % log_interval == 0:
            print(f"  [{batch_idx + 1}/{len(loader)}] "
                  f"Loss: {avg_loss:.4f} | Acc: {acc:.1f}% | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                  f"Grad Norm: {avg_grad:.2f} | Skip: {skipped_steps}")

    return total_loss / len(loader), 100.0 * correct / total


# ──────────────────────────────────────────────
# Checkpoint
# ──────────────────────────────────────────────

def save_checkpoint(state, is_best, save_dir, filename="checkpoint.pth"):
    """保存 checkpoint"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    filepath = save_path / filename
    torch.save(state, filepath)

    if is_best:
        best_path = save_path / "best_model.pth"
        torch.save(state, best_path)
        print(f"  [BEST] 最优模型已保存: {best_path}")


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ERes2NetV2 说话人识别训练 v3")
    parser.add_argument("--config", type=str, default="configs/train_v2.yaml",
                        help="配置文件路径")
    parser.add_argument("--resume", type=str, default=None,
                        help="从 checkpoint 恢复训练")
    parser.add_argument("--dry-run", action="store_true",
                        help="快速测试模式 (1 epoch)")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    print("=" * 60)
    print(" ERes2NetV2 说话人识别训练 v3")
    print("=" * 60)
    print(f"[DIR] 项目目录: {PROJECT_ROOT}")
    print(f"[CFG] 配置文件: {args.config}")

    # 设置随机种子
    set_seed(42)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEV] 设备: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── 数据增强 ──
    augment_transform = None
    if config["data"].get("augment", False):
        augment_transform = create_augmentation(config["data"]["augment_config"])
        aug_types = []
        if config["data"]["augment_config"].get("spec_augment"):
            aug_types.append("SpecAugment")
        if config["data"]["augment_config"].get("random_gain"):
            aug_types.append("RandomGain")
        if config["data"]["augment_config"].get("speed_perturb"):
            aug_types.append("SpeedPerturb")
        print(f"[AUG] 数据增强: 已启用 ({' + '.join(aug_types)})")
    else:
        print(f"[AUG] 数据增强: 未启用")

    # ── 数据集 ──
    # 支持多数据源 (v4): features_dirs (列表) 或 features_dir (单目录, 向后兼容)
    if "features_dirs" in config["data"]:
        features_dirs = [PROJECT_ROOT / d for d in config["data"]["features_dirs"]]
        missing = [d for d in features_dirs if not d.exists()]
        if missing:
            for d in missing:
                print(f"[WARN]  特征目录不存在: {d}")
            # 过滤掉不存在的目录
            features_dirs = [d for d in features_dirs if d.exists()]
        if not features_dirs:
            print("[FAIL] 没有可用的特征目录")
            sys.exit(1)

        print(f"\n[DATA] 加载多数据源:")
        for d in features_dirs:
            n_spk = sum(1 for x in d.iterdir() if x.is_dir()) if d.exists() else 0
            print(f"   - {d} ({n_spk} 说话人)")

        full_dataset = SpeakerDataset(
            data_dirs=[str(d) for d in features_dirs],
            split="train",
            n_mels=config["model"]["n_mels"],
            fixed_length=config["data"]["fixed_length"],
            transform=augment_transform,
        )
    else:
        features_dir = PROJECT_ROOT / config["data"]["features_dir"]
        if not features_dir.exists():
            print(f"[FAIL] 特征目录不存在: {features_dir}")
            print("   请先运行特征提取: python scripts/extract_features.py --batch")
            sys.exit(1)

        print(f"\n[DATA] 加载数据集: {features_dir}")
        full_dataset = SpeakerDataset(
            data_dir=str(features_dir),
            split="train",
            n_mels=config["model"]["n_mels"],
            fixed_length=config["data"]["fixed_length"],
            transform=augment_transform,
        )

    if len(full_dataset) == 0:
        print("[FAIL] 数据集为空，请先提取 FBank 特征")
        sys.exit(1)

    # 按说话人划分训练/验证集
    train_subset, val_subset, num_train_speakers = split_dataset_by_speakers(
        full_dataset,
        val_ratio=config["data"]["val_split_ratio"],
        seed=config["data"]["val_random_seed"],
    )

    # DataLoaders
    batch_size = config["training"]["batch_size"]
    num_workers = config["data"]["num_workers"]

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config["data"]["pin_memory"],
        drop_last=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config["data"]["pin_memory"],
    )

    # ── 模型 ──
    model_config = config["model"]
    model = create_eres2netv2(
        variant=model_config["variant"],
        n_mels=model_config["n_mels"],
        embedding_dim=model_config["embedding_dim"],
        num_speakers=num_train_speakers,
        scale=model_config["scale"],
        se_reduction=model_config["se_reduction"],
        pool_attention_dim=model_config["pool_attention_dim"],
        dropout=model_config.get("dropout", 0.3),  # 修复: 从配置读取 dropout
    )
    model = model.to(device)

    total_params = model.num_parameters()
    print(f"\n[MODEL] ERes2NetV2-{model_config['variant']}")
    print(f"   参数量: {total_params / 1e6:.2f}M")
    print(f"   Embedding 维度: {model_config['embedding_dim']}")
    print(f"   Dropout: {model_config.get('dropout', 0.3)}")
    print(f"   训练说话人数: {num_train_speakers}")

    # ── 损失函数 ──
    loss_config = config["loss"]
    if loss_config["type"] == "aam_softmax":
        criterion = AAMSoftmaxLoss(
            embedding_dim=model_config["embedding_dim"],
            num_speakers=num_train_speakers,
            margin=loss_config["margin"],
            scale=loss_config["scale"],
        )
    elif loss_config["type"] == "arcface":
        criterion = ArcFaceLoss(
            embedding_dim=model_config["embedding_dim"],
            num_speakers=num_train_speakers,
            margin=loss_config.get("margin", 0.5),
            scale=loss_config.get("scale", 64.0),
        )
    else:
        criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    print(f"[LOSS] 损失函数: {loss_config['type']} (margin={loss_config['margin']}, scale={loss_config['scale']})")

    # ── 优化器 (修复: 从配置读取, 支持 AdamW) ──
    train_config = config["training"]
    optimizer_name = train_config.get("optimizer", "SGD").lower()
    lr = train_config["lr"]
    weight_decay = train_config["weight_decay"]

    if optimizer_name == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        print(f"[OPT] AdamW (lr={lr}, wd={weight_decay})")
    elif optimizer_name == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        print(f"[OPT] Adam (lr={lr}, wd={weight_decay})")
    else:
        momentum = train_config.get("momentum", 0.9)
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        print(f"[OPT] SGD (lr={lr}, momentum={momentum}, wd={weight_decay})")

    # ── 学习率调度器 ──
    warmup_epochs = train_config["warmup_epochs"]
    total_epochs = train_config["epochs"] if not args.dry_run else 1

    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                                     total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs),
                                              eta_min=1e-7)
        scheduler = SequentialLR(optimizer,
                                  schedulers=[warmup_scheduler, cosine_scheduler],
                                  milestones=[warmup_epochs])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-7)

    # ── 混合精度 ──
    amp_enabled = train_config["amp"] and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=amp_enabled)
    print(f"[AMP] 混合精度: {'ON' if amp_enabled else 'OFF'}")

    # ── 恢复训练 ──
    start_epoch = 1
    best_val_eer = 1.0  # EER 越低越好
    patience_counter = 0

    if args.resume or train_config.get("checkpoint", {}).get("resume"):
        resume_path = args.resume or train_config["checkpoint"]["resume"]
        if Path(resume_path).exists():
            print(f"\n[LOAD] 从 checkpoint 恢复: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            if "criterion_state_dict" in checkpoint:
                try:
                    criterion.load_state_dict(checkpoint["criterion_state_dict"])
                except Exception as e:
                    print(f"   [WARN] Criterion state skipped: {e}")
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch", 1) + 1
            best_val_eer = checkpoint.get("best_val_eer", checkpoint.get("best_val_score", 1.0))
            print(f"   从 epoch {start_epoch} 继续 | best EER: {best_val_eer:.4f}")
        else:
            print(f"[WARN] Checkpoint 不存在: {resume_path}, 从头开始训练")

    # ── TensorBoard ──
    try:
        from torch.utils.tensorboard import SummaryWriter
        run_name = f"v3-{model_config['variant']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        writer = SummaryWriter(log_dir=PROJECT_ROOT / config["logging"]["tensorboard_dir"] / run_name)
        writer.add_text("config", json.dumps(config, indent=2, ensure_ascii=False))
        print(f"\n[TB] TensorBoard: tensorboard --logdir {PROJECT_ROOT / config['logging']['tensorboard_dir']}")
    except ImportError:
        writer = None
        print("\n[WARN] TensorBoard 未安装，跳过日志记录")

    # ── 训练循环 ──
    log_file = Path(config["checkpoint"]["save_dir"]) / "training.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    def log_msg(msg):
        print(msg)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log_msg(f"\n{'=' * 60}")
    log_msg(f"[START] Training v3 | Epochs: {total_epochs} | Batch size: {batch_size}")
    log_msg(f"   Train samples: {len(train_subset)} | Val samples: {len(val_subset)}")
    log_msg(f"   Optimizer: {optimizer_name} | LR: {lr} | WD: {weight_decay}")
    log_msg(f"   Loss: {loss_config['type']} (margin={loss_config['margin']}, scale={loss_config['scale']})")
    log_msg(f"   AMP: {'ON' if amp_enabled else 'OFF'} | Augment: {'ON' if augment_transform else 'OFF'}")
    log_msg(f"   Early stop patience: {train_config.get('early_stopping_patience', 15)}")
    log_msg(f"{'=' * 60}")

    training_start = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start = time.time()

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, epoch, config, device
        )

        # 验证 (EER-based)
        val_metrics = validate_embedding(model, val_loader, device)

        epoch_time = time.time() - epoch_start

        # 打印汇总
        log_msg(f"\n{'─' * 60}")
        log_msg(f"Epoch {epoch}/{total_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
                f"EER: {val_metrics['eer']:.4f} | "
                f"Pos: {val_metrics['pos_mean']:.4f} | Neg: {val_metrics['neg_mean']:.4f} | "
                f"Time: {epoch_time:.1f}s")

        # TensorBoard
        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Acc/train", train_acc, epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("EER/val", val_metrics["eer"], epoch)
            writer.add_scalar("minDCF/val", val_metrics["min_dcf"], epoch)
            writer.add_scalar("Sim/pos_mean", val_metrics["pos_mean"], epoch)
            writer.add_scalar("Sim/neg_mean", val_metrics["neg_mean"], epoch)

        # 更新学习率
        scheduler.step()

        # 保存 checkpoint (EER 越低越好)
        current_eer = val_metrics["eer"]
        is_best = current_eer < best_val_eer
        if is_best:
            best_val_eer = current_eer
            patience_counter = 0
        else:
            patience_counter += 1

        save_checkpoint({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_eer": best_val_eer,
            "config": config,
            "num_speakers": num_train_speakers,
        }, is_best, config["checkpoint"]["save_dir"])

        # 定期保存
        save_interval = config["checkpoint"]["save_interval"]
        if epoch % save_interval == 0:
            save_checkpoint({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "criterion_state_dict": criterion.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_val_eer": best_val_eer,
                "config": config,
                "num_speakers": num_train_speakers,
            }, False, config["checkpoint"]["save_dir"], f"checkpoint_epoch_{epoch}.pth")

        # 早停
        patience = train_config.get("early_stopping_patience", 15)
        if patience_counter >= patience:
            log_msg(f"\n[EARLY STOP] 触发! Best EER: {best_val_eer:.4f} (patience: {patience})")
            break

        if args.dry_run:
            log_msg("\n[DRY RUN] 完成")
            break

    training_time = time.time() - training_start
    log_msg(f"\n{'=' * 60}")
    log_msg(f"[DONE] 训练完成! Best EER: {best_val_eer:.4f}")
    log_msg(f"   总耗时: {training_time / 3600:.1f} 小时")
    log_msg(f"   模型保存位置: {config['checkpoint']['save_dir']}")
    log_msg(f"{'=' * 60}")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
