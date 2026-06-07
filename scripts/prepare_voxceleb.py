"""
VoxCeleb 数据准备完整 Pipeline

用法:
    # 完整流程（下载 + 预处理 + 特征提取 + 划分）
    python scripts/prepare_voxceleb.py --all

    # 仅下载
    python scripts/prepare_voxceleb.py --download

    # 仅提取 FBank 特征
    python scripts/prepare_voxceleb.py --extract

    # 仅划分数据集
    python scripts/prepare_voxceleb.py --split

    # 仅运行数据探索分析
    python scripts/prepare_voxceleb.py --eda

说明:
    这个脚本整合了数据准备的所有步骤。
    由于 VoxCeleb1 dev 集约 12GB，下载需要较长时间。
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def step_download():
    """步骤 1: 下载 VoxCeleb 数据集"""
    print("\n" + "=" * 60)
    print("📥 步骤 1: 下载 VoxCeleb 数据集")
    print("=" * 60)

    # 调用下载脚本
    download_script = PROJECT_ROOT / "scripts" / "download_voxceleb.py"
    if not download_script.exists():
        print("❌ 下载脚本不存在")
        return False

    import subprocess
    result = subprocess.run(
        [sys.executable, str(download_script), "--dev", "--test", "--meta"],
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode == 0:
        print("\n✅ 下载完成！")
        return True
    else:
        print("\n⚠️  下载未完成，可以手动下载:")
        print("   https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html")
        return False


def step_eda(data_dir=None):
    """步骤 2: 数据探索分析"""
    print("\n" + "=" * 60)
    print("📊 步骤 2: 数据探索分析（EDA）")
    print("=" * 60)

    if data_dir is None:
        data_dir = PROJECT_ROOT / "data" / "voxceleb" / "dev"

    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        print("   请先运行下载步骤")
        return False

    # 查找音频文件
    wav_files = list(data_dir.rglob("*.wav"))
    print(f"📁 数据目录: {data_dir}")
    print(f"📄 找到 {len(wav_files)} 个音频文件")

    if not wav_files:
        print("⚠️  没有找到 .wav 文件")
        return False

    # 分析说话人分布
    speaker_stats = defaultdict(int)
    for f in wav_files:
        speaker_id = f.parts[-2]  # 倒数第二层是说话人目录
        speaker_stats[speaker_id] += 1

    speakers = list(speaker_stats.keys())
    counts = list(speaker_stats.values())

    print(f"\n👤 说话人统计:")
    print(f"   说话人数: {len(speakers)}")
    print(f"   最少语音: {min(counts)} 条")
    print(f"   最多语音: {max(counts)} 条")
    print(f"   平均语音: {np.mean(counts):.1f} 条")
    print(f"   中位数: {np.median(counts):.0f} 条")

    # 分析文件大小
    file_sizes = [f.stat().st_size for f in wav_files]
    print(f"\n📦 文件大小统计:")
    print(f"   最小: {min(file_sizes) / 1024:.1f} KB")
    print(f"   最大: {max(file_sizes) / 1024 / 1024:.1f} MB")
    print(f"   平均: {np.mean(file_sizes) / 1024:.1f} KB")
    print(f"   总大小: {sum(file_sizes) / 1024 / 1024 / 1024:.2f} GB")

    # 保存 EDA 报告
    report = {
        "data_dir": str(data_dir),
        "num_speakers": len(speakers),
        "num_files": len(wav_files),
        "min_files_per_speaker": min(counts),
        "max_files_per_speaker": max(counts),
        "mean_files_per_speaker": float(np.mean(counts)),
        "median_files_per_speaker": float(np.median(counts)),
        "min_file_size_kb": min(file_sizes) / 1024,
        "max_file_size_mb": max(file_sizes) / 1024 / 1024,
        "mean_file_size_kb": np.mean(file_sizes) / 1024,
        "total_size_gb": sum(file_sizes) / 1024 / 1024 / 1024,
    }

    output_dir = PROJECT_ROOT / "data" / "voxceleb"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "eda_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n📝 EDA 报告已保存: {report_path}")
    return True


def step_extract(data_dir=None, output_dir=None):
    """步骤 3: 提取 FBank 特征"""
    print("\n" + "=" * 60)
    print("🔬 步骤 3: 提取 FBank 特征")
    print("=" * 60)

    if data_dir is None:
        data_dir = PROJECT_ROOT / "data" / "voxceleb" / "dev"
    if output_dir is None:
        output_dir = data_dir / "features"

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return False

    # 尝试导入 librosa
    try:
        import librosa
        import soundfile as sf
    except ImportError:
        print("❌ 需要安装 librosa 和 soundfile:")
        print("   pip install librosa soundfile")
        return False

    # 导入特征提取器
    import subprocess
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "extract_features.py"),
         "--batch", "--data-dir", str(data_dir), "--output-dir", str(output_dir)],
        cwd=str(PROJECT_ROOT),
    )
    return result.returncode == 0


def step_split(data_dir=None, val_ratio=0.1, seed=42):
    """步骤 4: 数据集划分"""
    print("\n" + "=" * 60)
    print("🔀 步骤 4: 数据集划分")
    print("=" * 60)

    if data_dir is None:
        data_dir = PROJECT_ROOT / "data" / "voxceleb" / "dev" / "features"

    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"❌ 特征目录不存在: {data_dir}")
        print("   请先运行特征提取步骤")
        return False

    random.seed(seed)

    # 按说话人分组
    speaker_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    speaker_files = {}

    for speaker_dir in speaker_dirs:
        npy_files = sorted(speaker_dir.glob("*.npy"))
        if npy_files:
            speaker_files[speaker_dir.name] = [f.relative_to(data_dir) for f in npy_files]

    speakers = list(speaker_files.keys())
    print(f"👤 找到 {len(speakers)} 个说话人")
    print(f"📄 总文件数: {sum(len(f) for f in speaker_files.values())}")

    # 划分说话人：90% 训练，10% 验证
    random.shuffle(speakers)
    n_val = max(1, int(len(speakers) * val_ratio))

    train_speakers = speakers[n_val:]
    val_speakers = speakers[:n_val]

    print(f"   训练集说话人: {len(train_speakers)}")
    print(f"   验证集说话人: {len(val_speakers)}")

    # 构建划分索引
    splits_dir = PROJECT_ROOT / "data" / "voxceleb" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    def save_split(split_name, speaker_list):
        split_files = []
        for speaker in speaker_list:
            split_files.extend([str(f) for f in speaker_files[speaker]])

        split_data = {
            "speakers": speaker_list,
            "num_speakers": len(speaker_list),
            "num_files": len(split_files),
            "files": split_files,
        }

        output_path = splits_dir / f"{split_name}.json"
        with open(output_path, "w") as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)

        print(f"   {split_name}: {len(speaker_list)} 说话人, {len(split_files)} 文件")
        return split_data

    train_data = save_split("train", train_speakers)
    val_data = save_split("val", val_speakers)

    # 保存划分摘要
    summary = {
        "seed": seed,
        "val_ratio": val_ratio,
        "train": {
            "speakers": train_data["num_speakers"],
            "files": train_data["num_files"],
        },
        "val": {
            "speakers": val_data["num_speakers"],
            "files": val_data["num_files"],
        },
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    summary_path = splits_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n📝 划分索引已保存: {splits_dir}")
    print(f"   - train.json")
    print(f"   - val.json")
    print(f"   - summary.json")

    return True


def check_dependencies():
    """检查必要的依赖"""
    print("🔍 检查依赖...")

    required = ["librosa", "soundfile", "numpy", "torch"]
    missing = []

    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"❌ 缺少依赖: {', '.join(missing)}")
        print(f"   安装: pip install {' '.join(missing)}")
        return False

    print("✅ 所有依赖已安装")
    return True


def main():
    parser = argparse.ArgumentParser(description="VoxCeleb 数据准备完整 Pipeline")
    parser.add_argument("--all", action="store_true", help="运行所有步骤")
    parser.add_argument("--download", action="store_true", help="仅下载")
    parser.add_argument("--eda", action="store_true", help="仅数据探索分析")
    parser.add_argument("--extract", action="store_true", help="仅提取 FBank 特征")
    parser.add_argument("--split", action="store_true", help="仅划分数据集")
    parser.add_argument("--data-dir", type=str, help="数据目录")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")
    args = parser.parse_args()

    # 默认运行所有步骤
    if not any([args.all, args.download, args.eda, args.extract, args.split]):
        args.all = True

    print("=" * 60)
    print("VoxCeleb 数据准备 Pipeline")
    print("=" * 60)
    print(f"📁 项目目录: {PROJECT_ROOT}")
    print()

    # 检查依赖
    if not check_dependencies():
        sys.exit(1)

    # 步骤 1: 下载
    if args.all or args.download:
        success = step_download()
        if not success and args.download:
            print("\n⚠️  下载未完成，可以手动下载数据集")

    # 步骤 2: EDA
    if args.all or args.eda:
        data_dir = Path(args.data_dir) if args.data_dir else None
        step_eda(data_dir)

    # 步骤 3: 特征提取
    if args.all or args.extract:
        data_dir = Path(args.data_dir) if args.data_dir else None
        step_extract(data_dir)

    # 步骤 4: 数据集划分
    if args.all or args.split:
        step_split(val_ratio=args.val_ratio)

    print("\n" + "=" * 60)
    print("✅ 数据准备完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
