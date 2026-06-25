"""
ERes2NetV2 说话人识别评估脚本

用法:
    python scripts/evaluate.py --config configs/train_config.yaml --checkpoint checkpoints/best_model.pth

功能:
    1. 加载最优模型 checkpoint
    2. 验证集评估：EER、minDCF、混淆矩阵、ROC/DET 曲线数据
    3. 效率指标：参数量、FLOPs、推理延迟、GPU 显存
    4. 生成评估报告 reports/evaluation_report.md
    5. 错误分析：收集误分类样本，分析模式
"""

import argparse
import sys
import time
import json
import random
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.eres2netv2 import create_eres2netv2
from src.datasets.speaker_dataset import SpeakerDataset
from src.training.trainer import compute_eer, compute_min_dcf, extract_embeddings


# ──────────────────────────────────────────────
# 数据集划分（复用训练脚本的逻辑）
# ──────────────────────────────────────────────

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
    """按说话人划分训练集和验证集（与 train.py 保持一致）"""
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

    return train_subset, val_subset, len(train_speakers)


# ──────────────────────────────────────────────
# 评估指标计算
# ──────────────────────────────────────────────

def compute_roc_curve(pos_scores: np.ndarray, neg_scores: np.ndarray, n_points: int = 200):
    """计算 ROC 曲线数据（FPR, TPR）"""
    thresholds = np.linspace(
        min(pos_scores.min(), neg_scores.min()),
        max(pos_scores.max(), neg_scores.max()),
        n_points,
    )
    fpr_list, tpr_list = [], []
    for threshold in thresholds:
        fpr = np.mean(neg_scores >= threshold)  # 假正率
        tpr = np.mean(pos_scores >= threshold)  # 真正率
        fpr_list.append(float(fpr))
        tpr_list.append(float(tpr))
    return fpr_list, tpr_list, thresholds.tolist()


def compute_det_curve(pos_scores: np.ndarray, neg_scores: np.ndarray, n_points: int = 200):
    """计算 DET 曲线数据（FNR, FPR）"""
    thresholds = np.linspace(
        min(pos_scores.min(), neg_scores.min()),
        max(pos_scores.max(), neg_scores.max()),
        n_points,
    )
    fnr_list, fpr_list = [], []
    for threshold in thresholds:
        fpr = np.mean(neg_scores >= threshold)  # 假正率
        fnr = np.mean(pos_scores < threshold)   # 假负率
        fpr_list.append(float(fpr))
        fnr_list.append(float(fnr))
    return fnr_list, fpr_list, thresholds.tolist()


def compute_confusion_matrix(
    embeddings: np.ndarray,
    labels: np.ndarray,
    top_n: int = 10,
    n_trials: int = 5000,
):
    """
    计算混淆矩阵（top-N 最易混淆的说话人对）

    通过 embedding 余弦相似度找到最容易混淆的说话人对。
    """
    unique_labels = np.unique(labels)
    speaker_indices = {lbl: np.where(labels == lbl)[0] for lbl in unique_labels}

    # 每个说话人计算平均 embedding（说话人中心）
    speaker_centers = {}
    for lbl in unique_labels:
        indices = speaker_indices[lbl]
        if len(indices) > 0:
            center = np.mean(embeddings[indices], axis=0)
            center = center / (np.linalg.norm(center) + 1e-8)
            speaker_centers[lbl] = center

    # 计算所有说话人对之间的相似度
    confusion_pairs = []
    for i, lbl_a in enumerate(unique_labels):
        for j, lbl_b in enumerate(unique_labels):
            if lbl_a >= lbl_b:
                continue
            if lbl_a not in speaker_centers or lbl_b not in speaker_centers:
                continue
            sim = float(np.sum(speaker_centers[lbl_a] * speaker_centers[lbl_b]))
            confusion_pairs.append((int(lbl_a), int(lbl_b), sim))

    # 按相似度降序排列（最易混淆的在前）
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    top_confused = confusion_pairs[:top_n]

    return top_confused


def collect_misclassified(
    embeddings: np.ndarray,
    labels: np.ndarray,
    eer_threshold: float,
    max_samples: int = 100,
) -> List[Dict]:
    """
    收集误分类样本（基于 EER 阈值的错误分析）

    统计每个说话人的误分类次数和最常被误判为的说话人。
    """
    unique_labels = np.unique(labels)
    speaker_indices = {lbl: np.where(labels == lbl)[0] for lbl in unique_labels}

    misclassified = []
    rng = np.random.RandomState(42)

    # 随机选择正样本对和负样本对进行分析
    for _ in range(min(n_trials := 3000, max_samples * 10)):
        lbl_a, lbl_b = rng.choice(unique_labels, size=2, replace=False)
        i = rng.choice(speaker_indices[lbl_a])
        j = rng.choice(speaker_indices[lbl_b])
        score = float(np.sum(embeddings[i] * embeddings[j]))
        if score >= eer_threshold:
            misclassified.append({
                "true_speaker_a": int(lbl_a),
                "true_speaker_b": int(lbl_b),
                "similarity": score,
                "error_type": "false_accept",
            })

    # 按相似度降序排列
    misclassified.sort(key=lambda x: x["similarity"], reverse=True)
    return misclassified[:max_samples]


# ──────────────────────────────────────────────
# 效率指标
# ──────────────────────────────────────────────

def count_parameters(model: nn.Module) -> Dict:
    """统计模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_M": total_params / 1e6,
        "trainable_params_M": trainable_params / 1e6,
    }


def compute_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> Optional[Dict]:
    """计算 FLOPs（优先使用 thop，备选 fvcore）"""
    dummy_input = torch.randn(*input_shape).to(next(model.parameters()).device)
    model.eval()

    # 尝试 thop
    try:
        from thop import profile
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        return {
            "flops": int(flops),
            "flops_G": float(flops) / 1e9,
            "params_from_thop": int(params),
            "method": "thop",
        }
    except Exception as e:
        pass

    # 尝试 fvcore
    try:
        from fvcore.nn import FlopCountAnalysis
        flop_counter = FlopCountAnalysis(model, dummy_input)
        flop_counter.unsupported_ops_warnings(False)
        flop_counter.uncalled_modules_warnings(False)
        flops = flop_counter.total()
        return {
            "flops": int(flops),
            "flops_G": float(flops) / 1e9,
            "method": "fvcore",
        }
    except Exception as e:
        pass

    # 手动估算（粗略：卷积层 FLOPs ≈ 2 * KH * KW * Cin * Cout * Hout * Wout）
    return {
        "flops": None,
        "flops_G": None,
        "method": "manual_estimate_not_available",
        "note": "请安装 thop 或 fvcore 以获取准确 FLOPs",
    }


def measure_inference_latency(model: nn.Module, input_shape: Tuple[int, ...],
                              device: torch.device, n_warmup: int = 10, n_runs: int = 100) -> Dict:
    """测量推理延迟（ms/batch）"""
    model.eval()
    dummy_input = torch.randn(*input_shape).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # 计时
    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

    latencies = np.array(latencies)
    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "n_runs": n_runs,
    }


def measure_gpu_memory(model: nn.Module, input_shape: Tuple[int, ...],
                      device: torch.device) -> Dict:
    """测量 GPU 显存占用"""
    if device.type != "cuda":
        return {"available": False, "note": "GPU 不可用"}

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    model.eval()
    dummy_input = torch.randn(*input_shape).to(device)

    # 记录推理前显存
    mem_before = torch.cuda.memory_allocated(device) / 1024**2  # MB

    with torch.no_grad():
        _ = model(dummy_input)

    torch.cuda.synchronize()
    mem_peak = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
    mem_after = torch.cuda.memory_allocated(device) / 1024**2

    return {
        "available": True,
        "mem_before_mb": float(mem_before),
        "mem_peak_mb": float(mem_peak),
        "mem_after_mb": float(mem_after),
        "mem_inference_mb": float(mem_peak - mem_before),
    }


# ──────────────────────────────────────────────
# 报告生成
# ──────────────────────────────────────────────

def generate_report(
    config: dict,
    checkpoint_info: dict,
    metrics: dict,
    efficiency: dict,
    confusion_pairs: list,
    misclassified_samples: list,
    report_path: Path,
):
    """生成评估报告 Markdown 文件"""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# ERes2NetV2 说话人识别评估报告\n")
    lines.append(f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"**模型**: ERes2NetV2-{config['model']['variant']}\n")
    lines.append(f"**Checkpoint**: {checkpoint_info.get('epoch', 'unknown')} epoch\n")
    lines.append("---\n")

    # 1. 评估概览
    lines.append("## 1. 评估概览\n")
    lines.append("| 指标 | 值 |")
    lines.append("|------|------|")
    lines.append(f"| EER (Equal Error Rate) | {metrics['eer']:.4f} ({metrics['eer']*100:.2f}%) |")
    lines.append(f"| EER 阈值 | {metrics['eer_threshold']:.4f} |")
    lines.append(f"| minDCF | {metrics['min_dcf']:.4f} |")
    lines.append(f"| minDCF 阈值 | {metrics['dcf_threshold']:.4f} |")
    lines.append(f"| 正样本对平均相似度 | {metrics['pos_mean']:.4f} ± {metrics['pos_std']:.4f} |")
    lines.append(f"| 负样本对平均相似度 | {metrics['neg_mean']:.4f} ± {metrics['neg_std']:.4f} |")
    lines.append(f"| 正样本对数 | {metrics['n_pos_pairs']} |")
    lines.append(f"| 负样本对数 | {metrics['n_neg_pairs']} |")
    lines.append("")

    # 2. 效率指标
    lines.append("## 2. 效率指标\n")
    lines.append("### 2.1 模型参数\n")
    params = efficiency["parameters"]
    lines.append("| 指标 | 值 |")
    lines.append("|------|------|")
    lines.append(f"| 总参数量 | {params['total_params']:,} ({params['total_params_M']:.2f}M) |")
    lines.append(f"| 可训练参数量 | {params['trainable_params']:,} ({params['trainable_params_M']:.2f}M) |")
    lines.append("")

    lines.append("### 2.2 FLOPs\n")
    flops = efficiency["flops"]
    if flops["flops"] is not None:
        lines.append(f"- **FLOPs**: {flops['flops']:,} ({flops['flops_G']:.4f} GFLOPs)")
    else:
        lines.append(f"- **FLOPs**: 未能计算 ({flops.get('note', '')})")
    lines.append(f"- **计算方法**: {flops['method']}")
    lines.append("")

    lines.append("### 2.3 推理延迟\n")
    for batch_size, latency in efficiency["latency"].items():
        lines.append(f"**Batch size = {batch_size}**:\n")
        lines.append("| 统计量 | 值 (ms) |")
        lines.append("|--------|---------|")
        lines.append(f"| 平均 | {latency['mean_ms']:.3f} |")
        lines.append(f"| 标准差 | {latency['std_ms']:.3f} |")
        lines.append(f"| P50 | {latency['p50_ms']:.3f} |")
        lines.append(f"| P95 | {latency['p95_ms']:.3f} |")
        lines.append(f"| P99 | {latency['p99_ms']:.3f} |")
        lines.append(f"| 最小 | {latency['min_ms']:.3f} |")
        lines.append(f"| 最大 | {latency['max_ms']:.3f} |")
        lines.append(f"| 运行次数 | {latency['n_runs']} |")
        lines.append("")

    lines.append("### 2.4 GPU 显存\n")
    gpu_mem = efficiency["gpu_memory"]
    if gpu_mem.get("available", False):
        lines.append("| 指标 | 值 (MB) |")
        lines.append("|------|----------|")
        lines.append(f"| 推理前 | {gpu_mem['mem_before_mb']:.2f} |")
        lines.append(f"| 推理峰值 | {gpu_mem['mem_peak_mb']:.2f} |")
        lines.append(f"| 推理后 | {gpu_mem['mem_after_mb']:.2f} |")
        lines.append(f"| 推理增量 | {gpu_mem['mem_inference_mb']:.2f} |")
    else:
        lines.append("GPU 不可用")
    lines.append("")

    # 3. 混淆分析
    lines.append("## 3. 混淆分析\n")
    lines.append("### 3.1 Top-10 最易混淆的说话人对\n")
    lines.append("| 排名 | 说话人 A | 说话人 B | 余弦相似度 |")
    lines.append("|------|----------|----------|------------|")
    for rank, (spk_a, spk_b, sim) in enumerate(confusion_pairs, 1):
        lines.append(f"| {rank} | {spk_a} | {spk_b} | {sim:.4f} |")
    lines.append("")

    # 4. 错误分析
    lines.append("## 4. 错误分析\n")
    lines.append(f"共收集 {len(misclassified_samples)} 个误分类样本（假接受）。\n")
    if misclassified_samples:
        lines.append("### 4.1 错误样本统计\n")
        # 统计每个说话人的误分类次数
        speaker_errors = Counter()
        for sample in misclassified_samples:
            speaker_errors[sample["true_speaker_a"]] += 1

        lines.append("| 说话人 | 误分类次数 |")
        lines.append("|--------|------------|")
        for spk, count in speaker_errors.most_common(10):
            lines.append(f"| {spk} | {count} |")
        lines.append("")

        lines.append("### 4.2 高相似度错误样本（Top-10）\n")
        lines.append("| 说话人 A | 说话人 B | 相似度 | 错误类型 |")
        lines.append("|----------|----------|--------|----------|")
        for sample in misclassified_samples[:10]:
            lines.append(f"| {sample['true_speaker_a']} | {sample['true_speaker_b']} | "
                        f"{sample['similarity']:.4f} | {sample['error_type']} |")
        lines.append("")

    # 5. ROC/DET 曲线
    lines.append("## 5. ROC/DET 曲线数据\n")
    lines.append("ROC 和 DET 曲线数据已保存到 JSON 文件，可用于绘图。\n")
    lines.append("- `reports/roc_curve.json`")
    lines.append("- `reports/det_curve.json`")
    lines.append("")

    # 6. 配置信息
    lines.append("## 6. 评估配置\n")
    lines.append(f"- **配置文件**: configs/train_config.yaml")
    lines.append(f"- **模型变体**: ERes2NetV2-{config['model']['variant']}")
    lines.append(f"- **Embedding 维度**: {config['model']['embedding_dim']}")
    lines.append(f"- **FBank 维度**: {config['model']['n_mels']}")
    lines.append(f"- **固定帧数**: {config['data']['fixed_length']}")
    lines.append(f"- **验证集比例**: {config['data']['val_split_ratio']}")
    lines.append("")

    lines.append("---")
    lines.append("*本报告由 `scripts/evaluate.py` 自动生成*")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[REPORT] 评估报告已保存: {report_path}")


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ERes2NetV2 说话人识别评估")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml",
                        help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth",
                        help="模型 checkpoint 路径")
    parser.add_argument("--output-dir", type=str, default="reports",
                        help="输出目录")
    args = parser.parse_args()

    # 加载配置
    config_path = PROJECT_ROOT / args.config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print(" ERes2NetV2 说话人识别评估")
    print("=" * 60)
    print(f"[DIR] 项目目录: {PROJECT_ROOT}")
    print(f"[CFG] 配置文件: {args.config}")
    print(f"[CKPT] Checkpoint: {args.checkpoint}")

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TOOL] 设备: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # 加载数据集
    features_dir = PROJECT_ROOT / config["data"]["features_dir"]
    if not features_dir.exists():
        print(f"[FAIL] 特征目录不存在: {features_dir}")
        sys.exit(1)

    print(f"\n[DATA] 加载数据集: {features_dir}")
    full_dataset = SpeakerDataset(
        data_dir=str(features_dir),
        split="train",
        n_mels=config["model"]["n_mels"],
        fixed_length=config["data"]["fixed_length"],
    )

    if len(full_dataset) == 0:
        print("[FAIL] 数据集为空")
        sys.exit(1)

    # 按说话人划分（与训练脚本一致）
    _, val_subset, _ = split_dataset_by_speakers(
        full_dataset,
        val_ratio=config["data"]["val_split_ratio"],
        seed=config["data"]["val_random_seed"],
    )

    batch_size = config["training"]["batch_size"]
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
    )

    print(f"   验证集样本数: {len(val_subset)}")
    print(f"   验证集说话人数: {len(set(val_subset.label_map.values()))}")

    # 加载模型
    model_config = config["model"]
    # 从 checkpoint 获取说话人数
    checkpoint_path = PROJECT_ROOT / args.checkpoint
    if not checkpoint_path.exists():
        print(f"[FAIL] Checkpoint 不存在: {checkpoint_path}")
        sys.exit(1)

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    num_speakers = checkpoint.get("num_speakers", None)
    if num_speakers is None:
        # 从模型状态字典推断
        classifier_weight = None
        for key in checkpoint["model_state_dict"]:
            if "classifier.weight" in key:
                classifier_weight = checkpoint["model_state_dict"][key]
                break
        if classifier_weight is not None:
            num_speakers = classifier_weight.shape[0]
        else:
            print("[FAIL] 无法从 checkpoint 推断说话人数")
            sys.exit(1)

    print(f"\n[MODEL] 创建模型: ERes2NetV2-{model_config['variant']}")
    print(f"   说话人数: {num_speakers}")

    model = create_eres2netv2(
        variant=model_config["variant"],
        n_mels=model_config["n_mels"],
        embedding_dim=model_config["embedding_dim"],
        num_speakers=num_speakers,
        scale=model_config["scale"],
        se_reduction=model_config["se_reduction"],
        pool_attention_dim=model_config["pool_attention_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"   参数量: {model.num_parameters() / 1e6:.2f}M")
    print(f"   Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")

    # ── 1. 验证集评估 ──
    print(f"\n{'─' * 60}")
    print("[1/5] 验证集评估 (EER / minDCF)...")
    embeddings, labels = extract_embeddings(model, val_loader, device)

    # 按说话人分组，生成正负样本对
    unique_labels = np.unique(labels)
    speaker_indices = {lbl: np.where(labels == lbl)[0] for lbl in unique_labels}
    rng = np.random.RandomState(42)

    pos_scores = []
    for lbl in unique_labels:
        indices = speaker_indices[lbl]
        if len(indices) < 2:
            continue
        for _ in range(min(len(indices) * (len(indices) - 1) // 2, 500)):
            i, j = rng.choice(indices, size=2, replace=False)
            score = float(np.sum(embeddings[i] * embeddings[j]))
            pos_scores.append(score)

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

    metrics = {
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "min_dcf": float(min_dcf),
        "dcf_threshold": float(dcf_threshold),
        "pos_mean": float(np.mean(pos_scores)),
        "neg_mean": float(np.mean(neg_scores)),
        "pos_std": float(np.std(pos_scores)),
        "neg_std": float(np.std(neg_scores)),
        "n_pos_pairs": len(pos_scores),
        "n_neg_pairs": len(neg_scores),
    }

    print(f"   EER: {eer:.4f} ({eer*100:.2f}%) | 阈值: {eer_threshold:.4f}")
    print(f"   minDCF: {min_dcf:.4f} | 阈值: {dcf_threshold:.4f}")
    print(f"   正样本对: {len(pos_scores)} | 负样本对: {len(neg_scores)}")
    print(f"   正样本相似度: {metrics['pos_mean']:.4f} ± {metrics['pos_std']:.4f}")
    print(f"   负样本相似度: {metrics['neg_mean']:.4f} ± {metrics['neg_std']:.4f}")

    # ── 2. ROC / DET 曲线 ──
    print(f"\n[2/5] 计算 ROC / DET 曲线...")
    fpr_list, tpr_list, roc_thresholds = compute_roc_curve(pos_scores, neg_scores)
    fnr_list, det_fpr_list, det_thresholds = compute_det_curve(pos_scores, neg_scores)

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    roc_data = {"fpr": fpr_list, "tpr": tpr_list, "thresholds": roc_thresholds}
    det_data = {"fnr": fnr_list, "fpr": det_fpr_list, "thresholds": det_thresholds}

    with open(output_dir / "roc_curve.json", "w") as f:
        json.dump(roc_data, f, indent=2)
    with open(output_dir / "det_curve.json", "w") as f:
        json.dump(det_data, f, indent=2)

    print(f"   ROC 曲线: {output_dir / 'roc_curve.json'}")
    print(f"   DET 曲线: {output_dir / 'det_curve.json'}")

    # ── 3. 混淆分析 ──
    print(f"\n[3/5] 混淆分析...")
    confusion_pairs = compute_confusion_matrix(embeddings, labels, top_n=10)
    print(f"   Top-10 最易混淆说话人对已收集")

    # ── 4. 错误分析 ──
    print(f"\n[4/5] 错误分析...")
    misclassified_samples = collect_misclassified(
        embeddings, labels, eer_threshold, max_samples=100
    )
    print(f"   收集到 {len(misclassified_samples)} 个误分类样本")

    # ── 5. 效率指标 ──
    print(f"\n[5/5] 效率指标...")
    efficiency = {}

    # 参数量
    efficiency["parameters"] = count_parameters(model)
    print(f"   总参数量: {efficiency['parameters']['total_params_M']:.2f}M")

    # FLOPs
    n_mels = config["model"]["n_mels"]
    fixed_length = config["data"]["fixed_length"]
    input_shape = (1, 1, n_mels, fixed_length)
    efficiency["flops"] = compute_flops(model, input_shape)
    if efficiency["flops"]["flops"] is not None:
        print(f"   FLOPs: {efficiency['flops']['flops_G']:.4f} GFLOPs ({efficiency['flops']['method']})")
    else:
        print(f"   FLOPs: 未能计算")

    # 推理延迟
    efficiency["latency"] = {}
    for bs in [1, 32]:
        shape = (bs, 1, n_mels, fixed_length)
        latency = measure_inference_latency(model, shape, device, n_warmup=10, n_runs=50)
        efficiency["latency"][f"batch_{bs}"] = latency
        print(f"   延迟 (batch={bs}): {latency['mean_ms']:.3f} ± {latency['std_ms']:.3f} ms")

    # GPU 显存
    efficiency["gpu_memory"] = measure_gpu_memory(model, input_shape, device)
    if efficiency["gpu_memory"].get("available", False):
        print(f"   GPU 显存峰值: {efficiency['gpu_memory']['mem_peak_mb']:.2f} MB")
    else:
        print(f"   GPU 显存: 不可用")

    # ── 生成报告 ──
    print(f"\n{'─' * 60}")
    print("[REPORT] 生成评估报告...")

    checkpoint_info = {
        "epoch": checkpoint.get("epoch", "unknown"),
        "best_val_acc": checkpoint.get("best_val_acc", "unknown"),
    }

    generate_report(
        config=config,
        checkpoint_info=checkpoint_info,
        metrics=metrics,
        efficiency=efficiency,
        confusion_pairs=confusion_pairs,
        misclassified_samples=misclassified_samples,
        report_path=output_dir / "evaluation_report.md",
    )

    # 保存完整评估数据 JSON
    eval_data = {
        "metrics": metrics,
        "efficiency": efficiency,
        "confusion_pairs": confusion_pairs,
        "misclassified_samples": misclassified_samples[:20],
    }
    with open(output_dir / "evaluation_data.json", "w") as f:
        json.dump(eval_data, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print("[DONE] 评估完成!")
    print(f"   报告: {output_dir / 'evaluation_report.md'}")
    print(f"   数据: {output_dir / 'evaluation_data.json'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
