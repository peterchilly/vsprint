"""
训练核心模块

包含:
- Trainer: 主训练循环 (AMP + 梯度裁剪 + 梯度累积)
- validate: 训练集验证（Top-1/Top-5 准确率，仅训练集参考）
- validate_embedding: 验证集基于 embedding 的评估（EER/minDCF）
- WarmupCosineScheduler: Warmup + Cosine Annealing 学习率调度
- save_checkpoint / load_checkpoint: checkpoint 管理
"""

import os
import time
import math
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.amp import autocast, GradScaler


# ──────────────────────────────────────────────
# 学习率调度器: Warmup + Cosine Annealing
# ──────────────────────────────────────────────

class WarmupCosineScheduler:
    """Warmup + Cosine Annealing 学习率调度"""

    def __init__(self, optimizer, base_lr: float, warmup_epochs: int, total_epochs: int):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


# ──────────────────────────────────────────────
# Checkpoint 管理
# ──────────────────────────────────────────────

def save_checkpoint(save_dir, model, optimizer, criterion, scheduler_base_lr,
                    epoch, best_val_score, is_best=False, save_interval=5):
    """保存 checkpoint"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "criterion_state_dict": criterion.state_dict(),
        "scheduler_base_lr": scheduler_base_lr,
        "best_val_score": best_val_score,
    }

    torch.save(checkpoint, save_path / "last_model.pth")

    if is_best:
        torch.save(checkpoint, save_path / "best_model.pth")
        print(f"  [BEST] New best model! val_score={best_val_score:.4f}")

    if (epoch + 1) % save_interval == 0:
        torch.save(checkpoint, save_path / f"checkpoint_epoch_{epoch+1}.pth")
        print(f"  [CKPT] Saved checkpoint: epoch_{epoch+1}")


def load_checkpoint(checkpoint_path, model, optimizer=None, criterion=None):
    """加载 checkpoint"""
    print(f"  [LOAD] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  [OK] Model weights loaded (epoch {checkpoint['epoch']})")
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"  [OK] Optimizer state loaded")
    if criterion is not None and "criterion_state_dict" in checkpoint:
        try:
            criterion.load_state_dict(checkpoint["criterion_state_dict"])
            print(f"  [OK] Criterion weights loaded")
        except Exception as e:
            print(f"  [WARN] Skipping Criterion weights: {e}")
    return checkpoint


# ──────────────────────────────────────────────
# Embedding 提取
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


# ──────────────────────────────────────────────
# EER 计算
# ──────────────────────────────────────────────

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
    """
    基于 embedding 的验证（说话人识别标准评估）

    返回:
        eer, min_dcf, pos_mean, neg_mean 等指标
    """
    import random
    from itertools import combinations

    embeddings, labels = extract_embeddings(model, val_loader, device)

    # 按说话人分组
    unique_labels = np.unique(labels)
    speaker_indices = {lbl: np.where(labels == lbl)[0] for lbl in unique_labels}

    rng = np.random.RandomState(42)

    # 正样本对（同一说话人）
    pos_scores = []
    for lbl in unique_labels:
        indices = speaker_indices[lbl]
        if len(indices) < 2:
            continue
        for _ in range(min(len(indices) * (len(indices) - 1) // 2, 500)):
            i, j = rng.choice(indices, size=2, replace=False)
            score = np.sum(embeddings[i] * embeddings[j])
            pos_scores.append(score)

    # 负样本对（不同说话人），数量与正样本对平衡
    neg_scores = []
    target_neg = len(pos_scores)
    for _ in range(target_neg * 3):
        lbl_a, lbl_b = rng.choice(unique_labels, size=2, replace=False)
        i = rng.choice(speaker_indices[lbl_a])
        j = rng.choice(speaker_indices[lbl_b])
        score = np.sum(embeddings[i] * embeddings[j])
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
        "dcf_threshold": dcf_threshold,
        "pos_mean": float(np.mean(pos_scores)),
        "neg_mean": float(np.mean(neg_scores)),
        "pos_std": float(np.std(pos_scores)),
        "neg_std": float(np.std(neg_scores)),
        "n_pos_pairs": len(pos_scores),
        "n_neg_pairs": len(neg_scores),
    }


# ──────────────────────────────────────────────
# 训练集分类验证（仅参考）
# ──────────────────────────────────────────────

@torch.no_grad()
def validate_classification(model, val_loader, criterion, device, loss_type="ce"):
    """训练集分类准确率验证（参考用）"""
    model.eval()
    total_loss, correct_top1, correct_top5, total_samples = 0.0, 0, 0, 0

    for features, labels in val_loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if features.dim() == 3:
            features = features.unsqueeze(1)
        embedding, logits = model(features)
        if logits is None:
            raise ValueError("Model returned no logits.")

        if loss_type == "ce":
            loss = criterion(logits, labels)
        else:
            loss = criterion(embedding, labels)
        total_loss += loss.item() * features.size(0)

        _, predictions = logits.topk(5, dim=1, largest=True, sorted=True)
        correct = predictions.eq(labels.unsqueeze(1).expand_as(predictions))
        correct_top1 += correct[:, 0].sum().item()
        correct_top5 += correct.any(dim=1).sum().item()
        total_samples += features.size(0)

    return {
        "val_loss": total_loss / total_samples,
        "val_top1": correct_top1 / total_samples,
        "val_top5": correct_top5 / total_samples,
    }


# ──────────────────────────────────────────────
# 训练器
# ──────────────────────────────────────────────

class Trainer:
    """主训练器 (AMP + 梯度裁剪 + gradient accumulation + EER 验证)"""

    def __init__(self, model, train_loader, val_loader, criterion, optimizer,
                 scheduler, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.epochs = config.get("epochs", 80)
        self.grad_clip = config.get("grad_clip", 1.0)
        self.amp_enabled = config.get("amp", True) and device.type == "cuda"
        self.grad_accumulation = config.get("gradient_accumulation", 1)
        self.log_interval = config.get("log_interval", 50)
        self.loss_type = config.get("loss_type", "ce")

        self.early_stopping_patience = config.get("early_stopping_patience", 15)
        self.best_val_eer = 1.0  # EER 越低越好
        self.patience_counter = 0

        self.save_dir = config.get("save_dir", "checkpoints")
        self.save_best = config.get("save_best", True)
        self.save_interval = config.get("save_interval", 5)

        self.tensorboard_dir = config.get("tensorboard_dir", "runs")
        self.writer = None
        self._init_tensorboard()

        self.scaler = GradScaler("cuda") if self.amp_enabled else None
        self.gpu_monitor_interval = config.get("gpu_monitor_interval", 30)

        self.log_file = Path(self.save_dir) / "training.log"
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

    def _init_tensorboard(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.tensorboard_dir)
            print(f"  [TB] TensorBoard: tensorboard --logdir {self.tensorboard_dir}")
        except ImportError:
            print("  [WARN] TensorBoard not installed, skipping logging")
            self.writer = None

    def _log(self, message):
        print(message)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def _monitor_gpu(self, step):
        if self.device.type != "cuda":
            return
        if step % self.gpu_monitor_interval != 0:
            return
        try:
            mem_alloc = torch.cuda.memory_allocated(self.device) / 1024**2
            mem_reserve = torch.cuda.memory_reserved(self.device) / 1024**2
            print(f"  [GPU] MEM: {mem_alloc:.0f}MB / {mem_reserve:.0f}MB")
        except Exception:
            pass

    def _train_one_epoch(self, epoch):
        self.model.train()
        total_loss, correct, total_samples = 0.0, 0, 0
        accumulation_steps = self.grad_accumulation

        for batch_idx, (features, labels) in enumerate(self.train_loader):
            features = features.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            if features.dim() == 3:
                features = features.unsqueeze(1)

            if self.amp_enabled and self.scaler is not None:
                with autocast("cuda"):
                    embedding, logits = self.model(features)
                    if logits is None:
                        raise ValueError("Model returned no logits.")
                    if self.loss_type == "ce":
                        loss = self.criterion(logits, labels)
                    else:
                        loss = self.criterion(embedding, labels)
                    loss = loss / accumulation_steps
                self.scaler.scale(loss).backward()
                if (batch_idx + 1) % accumulation_steps == 0:
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                embedding, logits = self.model(features)
                if self.loss_type == "ce":
                    loss = self.criterion(logits, labels)
                else:
                    loss = self.criterion(embedding, labels)
                loss = loss / accumulation_steps
                loss.backward()
                if (batch_idx + 1) % accumulation_steps == 0:
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total_loss += loss.item() * features.size(0) * accumulation_steps
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += features.size(0)

            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = total_loss / total_samples
                acc = correct / total_samples
                lr = self.scheduler.get_lr()
                self._log(
                    f"  [TRAIN] Epoch {epoch+1}/{self.epochs} | "
                    f"Batch {batch_idx+1}/{len(self.train_loader)} | "
                    f"Loss: {avg_loss:.4f} | Acc: {acc:.4f} | LR: {lr:.6f}"
                )
                self._monitor_gpu(batch_idx)

        return {"train_loss": total_loss / total_samples, "train_acc": correct / total_samples}

    def train(self, start_epoch=0):
        self._log("=" * 60)
        self._log(f"[START] Training | Epochs: {self.epochs} | Batch size: {self.train_loader.batch_size}")
        self._log(f"   Train samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            self._log(f"   Val samples: {len(self.val_loader.dataset)}")
        self._log(f"   Loss: {self.loss_type}")
        self._log(f"   AMP: {'ON' if self.amp_enabled else 'OFF'}")
        self._log(f"   Early stop patience: {self.early_stopping_patience}")
        self._log("=" * 60)

        for epoch in range(start_epoch, self.epochs):
            lr = self.scheduler.step(epoch)
            self._log(f"\n{'─' * 60}")
            self._log(f"Epoch {epoch+1}/{self.epochs} (LR: {lr:.6f})")

            train_metrics = self._train_one_epoch(epoch)

            val_metrics = None
            if self.val_loader is not None:
                self._log(f"  Validating...")
                val_metrics = validate_embedding(self.model, self.val_loader, self.device)

                self._log(
                    f"  Train Loss: {train_metrics['train_loss']:.4f} | "
                    f"Train Acc: {train_metrics['train_acc']:.4f}"
                )
                self._log(
                    f"  EER: {val_metrics['eer']:.4f} | "
                    f"minDCF: {val_metrics['min_dcf']:.4f} | "
                    f"Pos mean: {val_metrics['pos_mean']:.4f} | "
                    f"Neg mean: {val_metrics['neg_mean']:.4f}"
                )

            if self.writer is not None:
                self.writer.add_scalar("Loss/train", train_metrics["train_loss"], epoch)
                self.writer.add_scalar("Acc/train", train_metrics["train_acc"], epoch)
                self.writer.add_scalar("LR", lr, epoch)
                if val_metrics:
                    self.writer.add_scalar("EER/val", val_metrics["eer"], epoch)
                    self.writer.add_scalar("minDCF/val", val_metrics["min_dcf"], epoch)
                    self.writer.add_scalar("Sim/pos_mean", val_metrics["pos_mean"], epoch)
                    self.writer.add_scalar("Sim/neg_mean", val_metrics["neg_mean"], epoch)

            # EER 越低越好
            current_eer = val_metrics["eer"] if val_metrics else 1.0
            is_best = current_eer < self.best_val_eer

            if is_best:
                self.best_val_eer = current_eer
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            save_checkpoint(
                save_dir=self.save_dir, model=self.model, optimizer=self.optimizer,
                criterion=self.criterion, scheduler_base_lr=self.scheduler.base_lr,
                epoch=epoch, best_val_score=self.best_val_eer, is_best=is_best,
                save_interval=self.save_interval,
            )

            if self.patience_counter >= self.early_stopping_patience:
                self._log(f"\n[EARLY STOP] Triggered! Best EER: {self.best_val_eer:.4f} "
                          f"(patience: {self.early_stopping_patience})")
                break

        self._log(f"\n{'=' * 60}")
        self._log(f"[DONE] Training complete! Best EER: {self.best_val_eer:.4f}")
        self._log(f"{'=' * 60}")

        if self.writer is not None:
            self.writer.close()
