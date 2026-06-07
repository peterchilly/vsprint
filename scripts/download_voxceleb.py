"""
VoxCeleb 数据集下载脚本

用法:
    python scripts/download_voxceleb.py [--dev] [--test] [--meta] [--all]

说明:
    默认下载 VoxCeleb1 Dev 集（训练用）。
    下载路径：data/voxceleb/
    由于 VoxCeleb 文件较大（Dev 集约 12GB），建议使用稳定网络连接。
"""

import argparse
import os
import sys
import subprocess
import shutil
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "voxceleb"

# 下载链接（来自牛津大学 VGG 官方源）
URLS = {
    "dev": "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1/vox1_dev_wav_part{part}.zip",
    "dev_parts": 4,  # 分 4 个部分
    "test": "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1/vox1_test_wav.zip",
    "meta": {
        "vox1_meta.csv": "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/vox1_meta.csv",
        "veri_test2.txt": "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1/veri_test2.txt",
    },
}


def download_file(url: str, output_path: str):
    """使用 curl 下载文件（Windows 兼容）"""
    print(f"  📥 下载: {url}")
    print(f"  📁 保存: {output_path}")

    cmd = [
        "curl", "-L", "-#",
        "-o", output_path,
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ❌ 下载失败: {result.stderr}")
        return False
    print(f"  ✅ 下载完成")
    return True


def extract_zip(zip_path: str, extract_to: str):
    """解压 ZIP 文件"""
    print(f"  📦 解压: {zip_path}")
    print(f"  📁 目标: {extract_to}")

    # 使用 PowerShell 解压（Windows 原生）
    cmd = [
        "powershell", "-Command",
        f"Expand-Archive -Path '{zip_path}' -DestinationPath '{extract_to}' -Force"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ❌ 解压失败: {result.stderr}")
        return False
    print(f"  ✅ 解压完成")
    return True


def download_dev(data_dir: Path):
    """下载 VoxCeleb1 Dev 集（分 4 部分）"""
    dev_dir = data_dir / "dev"
    dev_dir.mkdir(parents=True, exist_ok=True)

    zip_dir = dev_dir / "zips"
    zip_dir.mkdir(parents=True, exist_ok=True)

    for part in range(1, URLS["dev_parts"] + 1):
        url = URLS["dev"].format(part=part)
        zip_name = f"vox1_dev_wav_part{part}.zip"
        zip_path = zip_dir / zip_name

        if zip_path.exists() and zip_path.stat().st_size > 100_000_000:  # > 100MB
            print(f"  ⏭️  跳过（已存在）: {zip_name}")
            continue

        if not download_file(url, str(zip_path)):
            print(f"  ⚠️  下载失败，跳过 part {part}")
            continue

        if not extract_zip(str(zip_path), str(dev_dir)):
            print(f"  ⚠️  解压失败")


def download_test(data_dir: Path):
    """下载 VoxCeleb1 Test 集"""
    test_dir = data_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    zip_path = test_dir / "vox1_test_wav.zip"
    if zip_path.exists() and zip_path.stat().st_size > 100_000_000:
        print(f"  ⏭️  跳过（已存在）: vox1_test_wav.zip")
        return

    if download_file(URLS["test"], str(zip_path)):
        extract_zip(str(zip_path), str(test_dir))


def download_meta(data_dir: Path):
    """下载元数据文件"""
    meta_dir = data_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in URLS["meta"].items():
        output_path = meta_dir / filename
        if output_path.exists():
            print(f"  ⏭️  跳过（已存在）: {filename}")
            continue
        download_file(url, str(output_path))


def main():
    parser = argparse.ArgumentParser(description="下载 VoxCeleb 数据集")
    parser.add_argument("--dev", action="store_true", help="下载 Dev 集（训练用）")
    parser.add_argument("--test", action="store_true", help="下载 Test 集（测试用）")
    parser.add_argument("--meta", action="store_true", help="下载元数据")
    parser.add_argument("--all", action="store_true", help="下载所有内容")
    args = parser.parse_args()

    # 默认下载 dev
    if not any([args.dev, args.test, args.meta, args.all]):
        args.dev = True

    print("=" * 60)
    print("VoxCeleb 数据集下载")
    print("=" * 60)
    print(f"📁 下载目录: {DATA_DIR}")
    print()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.dev or args.all:
        print("\n📥 下载 VoxCeleb1 Dev 集（~12GB，分 4 部分）")
        print("   这可能需要较长时间，取决于网络速度。")
        download_dev(DATA_DIR)

    if args.test or args.all:
        print("\n📥 下载 VoxCeleb1 Test 集（~1.7GB）")
        download_test(DATA_DIR)

    if args.meta or args.all:
        print("\n📥 下载元数据文件")
        download_meta(DATA_DIR)

    print("\n" + "=" * 60)
    print("✅ 下载完成！")
    print("=" * 60)

    # 显示目录结构
    print("\n📁 数据目录结构:")
    for p in sorted(DATA_DIR.rglob("*")):
        if p.is_file():
            size = p.stat().st_size
            size_str = f"{size / 1024 / 1024:.1f}MB" if size > 1024 * 1024 else f"{size / 1024:.1f}KB"
            print(f"   {p.relative_to(DATA_DIR)} ({size_str})")


if __name__ == "__main__":
    main()
