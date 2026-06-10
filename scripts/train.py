"""
ERes2NetV2 说话人识别训练脚本

用法:
    # 从头开始训练
    python scripts/train.py

    # 指定配置文件
    python scripts/train.py --config configs/train_config.yaml

    # 从 checkpoint 恢复
    python scripts/train.py --resume checkpoints/best_model.pth

    # 快速测试1 epoch
    python scripts/train.py --dry-run
"""

import argparse
import os
import sys
import time
import json
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm
import yaml

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.eres2netv2 import create_eres2netv2
from src.models.losses import AAMSoftmaxLoss, ArcFaceLoss
from src.datasets.speaker_dataset import SpeakerDataset


def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def split_dataset_by_speakers(dataset, val_ratio=0.1, seed=42):
    """
    按说话人划分训练集和验证集
    同时重映射训练集标签为 0~N-1
    """
    rng = random.Random(seed)
    speakers = sorted(list(dataset.speaker_to_id.keys()))
    num_val_speakers = max(1, int(len(speakers) * val_ratio))
    val_speakers = set(rng.sample(speakers, num_val_speakers))
    train_speakers = sorted(set(speakers) - val_speakers)

    # 标签重映射: original_id -> 0..N-1
    label_map = {dataset.speaker_to_id[spk]: i for i, spk in enumerate(train_speakers)}

    train_indices = []
    val_indices = []
    for idx, label in enumerate(dataset.labels):
        speaker_name = dataset.id_to_speaker[label]
        if speaker_name in val_speakers:
            val_indices.append(idx)
        elif label in label_map:
            train_indices.append(idx)

    train_subset = RemappedSubset(dataset, train_indices, label_map)
    val_subset = Subset(dataset, val_indices)

    print(f"[TB] 数据集划分:")
    print(f"   总说话人: {len(speakers)}")
    print(f"   训练集: {len(train_speakers)} 说话人, {len(train_indices)} 样本")
    print(f"   验证集: {len(val_speakers)} 说话人, {len(val_indices)} 样本")
    print(f"   验证集说话人: {sorted(list(val_speakers))[:5]}...")

    return train_subset, val_subset, len(train_speakers)


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


def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch, config, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    grad_norm_sum = 0

    log_interval = config.get("logging", {}).get("log_interval", 50)

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch_idx, (fbanks, labels) in enumerate(pbar):
        fbanks = fbanks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass with autocast
        with autocast(enabled=config["training"]["amp"]):
            embeddings, logits = model(fbanks)

        # Loss in FP32 (AAM-Softmax indexing incompatible with mixed precision)
        with torch.cuda.amp.autocast(enabled=False):
            loss = criterion(embeddings.float(), labels)

        scaler.scale(loss).backward()

        # 梯度裁剪
        max_norm = config["training"]["grad_clip"]
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        grad_norm_sum += grad_norm.item()

        scaler.step(optimizer)
        scaler.update()

        # 统计
        total_loss += loss.item()
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        # 进度条更新
        avg_loss = total_loss / (batch_idx + 1)
        acc = 100.0 * correct / total
        avg_grad = grad_norm_sum / (batch_idx + 1)
        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{acc:.1f}%", "grad": f"{avg_grad:.2f}"})

        # 定期日志
        if (batch_idx + 1) % log_interval == 0:
            print(f"  [{batch_idx + 1}/{len(loader)}] "
                  f"Loss: {avg_loss:.4f} | Acc: {acc:.1f}% | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                  f"Grad Norm: {avg_grad:.2f}")

    return total_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Validating", leave=False)
    for fbanks, labels in pbar:
        fbanks = fbanks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        embeddings, logits = model(fbanks)
        loss = criterion(embeddings, labels)

        total_loss += loss.item()
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        avg_loss = total_loss / (pbar.n + 1)
        acc = 100.0 * correct / total
        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{acc:.1f}%"})

    return total_loss / len(loader), 100.0 * correct / total


def save_checkpoint(state, is_best, save_dir, filename="checkpoint.pth"):
    """保存 checkpoint"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 保存当前 checkpoint
    filepath = save_path / filename
    torch.save(state, filepath)

    # 保存最优模型
    if is_best:
        best_path = save_path / "best_model.pth"
        torch.save(state, best_path)
        print(f"  [BEST] 最优模型已保存: {best_path}")


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="ERes2NetV2 说话人识别训练")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml",
                        help="配置文件路径")
    parser.add_argument("--resume", type=str, default=None,
                        help="从 checkpoint 恢复训练")
    parser.add_argument("--dry-run", action="store_true",
                        help="快速测试模式1 epoch")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    print("=" * 60)
    print(" ERes2NetV2 说话人识别训练")
    print("=" * 60)
    print(f"[DIR] 项目目录: {PROJECT_ROOT}")
    print(f" 配置文件: {args.config}")

    # 设置随机种子
    set_seed(42)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TOOL] 设备: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    #  数据集 
    features_dir = PROJECT_ROOT / config["data"]["features_dir"]
    if not features_dir.exists():
        print(f"[FAIL] 特征目录不存在: {features_dir}")
        print("   请先运行特征提取:")
        print("   python scripts/extract_features.py --batch --data-dir data/voxceleb/dev")
        sys.exit(1)

    print(f"\n 加载数据集: {features_dir}")
    full_dataset = SpeakerDataset(
        data_dir=str(features_dir),
        split="train",
        n_mels=config["model"]["n_mels"],
        fixed_length=config["data"]["fixed_length"],
    )

    if len(full_dataset) == 0:
        print("[FAIL] 数据集为空请先提取 FBank 特征")
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

    #  模型 
    model_config = config["model"]
    model = create_eres2netv2(
        variant=model_config["variant"],
        n_mels=model_config["n_mels"],
        embedding_dim=model_config["embedding_dim"],
        num_speakers=num_train_speakers,
        scale=model_config["scale"],
        se_reduction=model_config["se_reduction"],
        pool_attention_dim=model_config["pool_attention_dim"],
    )
    model = model.to(device)

    total_params = model.num_parameters()
    print(f"\n[MODEL] 模型: ERes2NetV2-{model_config['variant']}")
    print(f"   参数量: {total_params / 1e6:.2f}M")
    print(f"   Embedding 维度: {model_config['embedding_dim']}")
    print(f"   训练说话人数: {num_train_speakers}")

    #  损失函数 
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
    print(f"   损失函数: {loss_config['type']}")

    #  优化器 
    train_config = config["training"]
    optimizer = optim.SGD(
        model.parameters(),
        lr=train_config["lr"],
        momentum=train_config["momentum"],
        weight_decay=train_config["weight_decay"],
    )

    #  学习率调度器 
    warmup_epochs = train_config["warmup_epochs"]
    total_epochs = train_config["epochs"] if not args.dry_run else 1

    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                                     total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs,
                                              eta_min=1e-6)
        scheduler = SequentialLR(optimizer,
                                  schedulers=[warmup_scheduler, cosine_scheduler],
                                  milestones=[warmup_epochs])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)

    #  混合精度 
    scaler = GradScaler(enabled=train_config["amp"])

    #  恢复训练 
    start_epoch = 1
    best_val_acc = 0.0
    patience_counter = 0

    if args.resume or train_config.get("checkpoint", {}).get("resume"):
        resume_path = args.resume or train_config["checkpoint"]["resume"]
        if Path(resume_path).exists():
            print(f"\n[CKPT] 从 checkpoint 恢复: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            criterion.load_state_dict(checkpoint.get("criterion_state_dict", {}))
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint.get("scheduler_state_dict", {}))
            start_epoch = checkpoint.get("epoch", 1) + 1
            best_val_acc = checkpoint.get("best_val_acc", 0.0)
            scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))
            print(f"   从 epoch {start_epoch} 继续最优验证准确率: {best_val_acc:.1f}%")
        else:
            print(f"[WARN]  Checkpoint 不存在: {resume_path}从头开始训练")

    #  TensorBoard 
    try:
        from torch.utils.tensorboard import SummaryWriter
        run_name = f"ERes2NetV2-{model_config['variant']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        writer = SummaryWriter(log_dir=PROJECT_ROOT / config["logging"]["tensorboard_dir"] / run_name)
        writer.add_text("config", json.dumps(config, indent=2, ensure_ascii=False))
        print(f"\n[TB] TensorBoard: tensorboard --logdir {PROJECT_ROOT / config['logging']['tensorboard_dir']}")
    except ImportError:
        writer = None
        print("\n[WARN]  TensorBoard 未安装跳过日志记录")

    #  训练循环 
    print(f"\n{'=' * 60}")
    print(f"[GO] 开始训练")
    print(f"   Epochs: {total_epochs}")
    print(f"   Batch Size: {batch_size}")
    print(f"   学习率: {train_config['lr']} (Warmup: {warmup_epochs} epochs)")
    print(f"   混合精度: {'[OK]' if train_config['amp'] else '[FAIL]'}")
    print(f"{'=' * 60}\n")

    training_start = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start = time.time()

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, epoch, config, device
        )

        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start

        # 打印汇总
        print(f"\n{'' * 50}")
        print(f"Epoch {epoch}/{total_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}% | "
              f"Time: {epoch_time:.1f}s")
        print(f"{'' * 50}")

        # TensorBoard 日志
        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        # 更新学习率
        scheduler.step()

        # 保存 checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
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
            "best_val_acc": best_val_acc,
            "config": config,
            "num_speakers": num_train_speakers,
        }, is_best, config["checkpoint"]["save_dir"])

        # 早停检查
        patience = train_config.get("early_stopping_patience", 20)
        if patience_counter >= patience:
            print(f"\n[STOP]  早停触发验证准确率 {patience} 个 epoch 未提升")
            print(f"   最优验证准确率: {best_val_acc:.1f}%")
            break

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
                "best_val_acc": best_val_acc,
                "config": config,
                "num_speakers": num_train_speakers,
            }, False, config["checkpoint"]["save_dir"], f"checkpoint_epoch_{epoch}.pth")

        if args.dry_run:
            print("\n Dry run 完成")
            break

    training_time = time.time() - training_start
    print(f"\n{'=' * 60}")
    print(f"[DONE] 训练完成")
    print(f"   总耗时: {training_time / 3600:.1f} 小时")
    print(f"   最优验证准确率: {best_val_acc:.1f}%")
    print(f"   模型保存位置: {config['checkpoint']['save_dir']}")
    print(f"{'=' * 60}")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
