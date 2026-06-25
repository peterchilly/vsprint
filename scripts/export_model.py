"""
ERes2NetV2 模型导出脚本

功能:
    1. 导出 ONNX 格式
    2. 导出 TorchScript 格式
    3. 验证导出模型与 PyTorch 输出一致性
    4. 报告模型大小对比

用法:
    python scripts/export_model.py --config configs/train_config.yaml --checkpoint checkpoints/best_model.pth
    python scripts/export_model.py --config configs/train_config.yaml --checkpoint checkpoints/best_model.pth --format onnx
    python scripts/export_model.py --config configs/train_config.yaml --checkpoint checkpoints/best_model.pth --format torchscript
"""

import argparse
import sys
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.eres2netv2 import create_eres2netv2


def load_model(config: dict, checkpoint_path: Path, device: torch.device) -> nn.Module:
    """加载训练好的模型"""
    print(f"[CKPT] 加载 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    model_config = config["model"]
    num_speakers = checkpoint.get("num_speakers", None)
    if num_speakers is None:
        # 从模型状态字典推断
        for key in checkpoint["model_state_dict"]:
            if "classifier.weight" in key:
                num_speakers = checkpoint["model_state_dict"][key].shape[0]
                break

    print(f"   说话人数: {num_speakers}")
    print(f"   Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")

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

    print(f"   模型参数量: {model.num_parameters() / 1e6:.2f}M")
    return model


def export_onnx(model: nn.Module, dummy_input: torch.Tensor, output_path: Path) -> Dict:
    """导出 ONNX 格式"""
    print(f"\n[ONNX] 导出 ONNX 格式...")

    # 创建只返回 embedding 的包装模型（ONNX 不支持多输出 tuple）
    class EmbeddingWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            embedding, _ = self.model(x)
            return embedding

    wrapper = EmbeddingWrapper(model).eval()
    wrapper = wrapper.to(dummy_input.device)

    onnx_path = output_path / "eres2netv2.onnx"

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["fbank"],
        output_names=["embedding"],
        dynamic_axes={
            "fbank": {0: "batch", 3: "time"},
            "embedding": {0: "batch"},
        },
    )

    file_size = onnx_path.stat().st_size / 1024**2  # MB
    print(f"   [OK] ONNX 导出成功: {onnx_path}")
    print(f"   文件大小: {file_size:.2f} MB")

    # 验证 ONNX 模型
    try:
        import onnx
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print(f"   [OK] ONNX 模型验证通过")
    except Exception as e:
        print(f"   [WARN] ONNX 验证失败: {e}")

    # 用 ONNX Runtime 验证输出一致性
    try:
        import onnxruntime as ort
        ort_session = ort.InferenceSession(str(onnx_path))
        ort_input = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
        ort_output = ort_session.run(None, ort_input)[0]

        with torch.no_grad():
            torch_output = wrapper(dummy_input).cpu().numpy()

        max_diff = np.max(np.abs(torch_output - ort_output))
        mean_diff = np.mean(np.abs(torch_output - ort_output))
        print(f"   [OK] ONNX Runtime 验证: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        return {
            "path": str(onnx_path),
            "size_mb": file_size,
            "max_diff": float(max_diff),
            "mean_diff": float(mean_diff),
            "verified": True,
        }
    except ImportError:
        print(f"   [WARN] onnxruntime 未安装，跳过输出验证")
        return {
            "path": str(onnx_path),
            "size_mb": file_size,
            "verified": False,
            "note": "onnxruntime 未安装",
        }
    except Exception as e:
        print(f"   [WARN] ONNX 输出验证失败: {e}")
        return {
            "path": str(onnx_path),
            "size_mb": file_size,
            "verified": False,
            "error": str(e),
        }


def export_torchscript(model: nn.Module, dummy_input: torch.Tensor, output_path: Path) -> Dict:
    """导出 TorchScript 格式"""
    print(f"\n[TS] 导出 TorchScript 格式...")

    class EmbeddingWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            embedding, _ = self.model(x)
            return embedding

    wrapper = EmbeddingWrapper(model).eval()
    wrapper = wrapper.to(dummy_input.device)

    ts_path = output_path / "eres2netv2.pt"

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy_input)

    traced.save(str(ts_path))

    file_size = ts_path.stat().st_size / 1024**2  # MB
    print(f"   [OK] TorchScript 导出成功: {ts_path}")
    print(f"   文件大小: {file_size:.2f} MB")

    # 验证 TorchScript 输出
    with torch.no_grad():
        torch_output = wrapper(dummy_input)
        ts_output = traced(dummy_input)

    max_diff = (torch_output - ts_output).abs().max().item()
    mean_diff = (torch_output - ts_output).abs().mean().item()
    print(f"   [OK] TorchScript 验证: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    return {
        "path": str(ts_path),
        "size_mb": file_size,
        "max_diff": float(max_diff),
        "mean_diff": float(mean_diff),
        "verified": True,
    }


def main():
    parser = argparse.ArgumentParser(description="ERes2NetV2 模型导出")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml",
                        help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth",
                        help="模型 checkpoint 路径")
    parser.add_argument("--output-dir", type=str, default="exports",
                        help="输出目录")
    parser.add_argument("--format", type=str, default="all", choices=["all", "onnx", "torchscript"],
                        help="导出格式")
    args = parser.parse_args()

    # 加载配置
    config_path = PROJECT_ROOT / args.config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print(" ERes2NetV2 模型导出")
    print("=" * 60)
    print(f"[DIR] 项目目录: {PROJECT_ROOT}")

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TOOL] 设备: {device}")

    # 加载模型
    model = load_model(config, PROJECT_ROOT / args.checkpoint, device)

    # 创建 dummy 输入
    n_mels = config["model"]["n_mels"]
    fixed_length = config["data"]["fixed_length"]
    dummy_input = torch.randn(1, 1, n_mels, fixed_length).to(device)
    print(f"\n[DATA] Dummy 输入: {dummy_input.shape}")

    # 输出目录
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # PyTorch 模型大小
    pytorch_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    print(f"\n[SIZE] PyTorch 模型大小（参数）: {pytorch_size:.2f} MB")

    results = {"pytorch_size_mb": pytorch_size}

    # 导出
    if args.format in ("all", "onnx"):
        results["onnx"] = export_onnx(model, dummy_input, output_dir)

    if args.format in ("all", "torchscript"):
        results["torchscript"] = export_torchscript(model, dummy_input, output_dir)

    # 汇总
    print(f"\n{'=' * 60}")
    print("[DONE] 模型导出完成!")
    print(f"{'─' * 60}")
    print(f"PyTorch 参数大小: {pytorch_size:.2f} MB")
    if "onnx" in results:
        r = results["onnx"]
        print(f"ONNX 文件大小: {r['size_mb']:.2f} MB | 验证: {'✓' if r['verified'] else '✗'}")
    if "torchscript" in results:
        r = results["torchscript"]
        print(f"TorchScript 文件大小: {r['size_mb']:.2f} MB | 验证: {'✓' if r['verified'] else '✗'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
