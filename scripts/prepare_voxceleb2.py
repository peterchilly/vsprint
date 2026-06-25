"""
VoxCeleb2 数据准备脚本

流程:
1. 解压 aac1-5.zip 到 F:/voxceleb2/dev/aac/
2. 用 ffmpeg 将 .m4a 转为 .wav (16kHz mono)
3. 批量提取 FBank 特征到 data/voxceleb2/features/
4. 支持断点续传（已处理的文件自动跳过）

用法:
    # 完整流程
    python scripts/prepare_voxceleb2.py --step all

    # 仅转换 AAC → WAV
    python scripts/prepare_voxceleb2.py --step convert

    # 仅提取 FBank 特征
    python scripts/prepare_voxceleb2.py --step features

    # 使用多进程加速转换
    python scripts/prepare_voxceleb2.py --step convert --workers 4
"""

import os
import sys
import subprocess
import zipfile
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
VC2_ZIP_DIR = Path(r"F:\BaiduNetdiskDownload\vox数据集\voxceleb2")
VC2_DEV_DIR = Path(r"F:\voxceleb2\dev")
VC2_AAC_DIR = VC2_DEV_DIR / "aac"
VC2_WAV_DIR = VC2_DEV_DIR / "wav"
VC2_FEATURES_DIR = PROJECT_ROOT / "data" / "voxceleb2" / "features"

# ffmpeg 路径
def _find_ffmpeg():
    import shutil as _shutil
    ffmpeg_in_path = _shutil.which("ffmpeg")
    if ffmpeg_in_path:
        return ffmpeg_in_path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    for p in [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    ]:
        if Path(p).exists():
            return p
    return None

FFMPEG_PATH = _find_ffmpeg()

# 断点续传状态文件
CONVERT_PROGRESS_FILE = VC2_DEV_DIR / ".convert_progress.json"
FEATURES_PROGRESS_FILE = VC2_FEATURES_DIR / ".features_progress.json"


def load_progress(progress_file):
    """加载进度文件"""
    if progress_file.exists():
        with open(progress_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": [], "failed": [], "last_update": ""}


def save_progress(progress_file, progress):
    """保存进度文件"""
    progress["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def extract_zip(zip_path, target_dir):
    """解压 zip 文件"""
    zip_path = Path(zip_path)
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    print(f"  Extracting {zip_path.name} → {target}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(target)
    print(f"  Done: {zip_path.name}")
    return True


def convert_aac_to_wav(aac_path, wav_path, sample_rate=16000):
    """用 ffmpeg 将 AAC/M4A 转为 WAV"""
    if FFMPEG_PATH is None:
        raise RuntimeError("ffmpeg not found")
    cmd = [
        FFMPEG_PATH, "-y", "-i", str(aac_path),
        "-ar", str(sample_rate), "-ac", "1", "-f", "wav",
        str(wav_path)
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    return result.returncode == 0


def process_speaker_convert(speaker_dir, wav_output_dir, sample_rate=16000):
    """处理一个说话人的所有 AAC → WAV 转换

    返回:
        dict: {speaker, converted, skipped, failed, failed_files}
    """
    speaker_dir = Path(speaker_dir)
    speaker_name = speaker_dir.name
    output_speaker_dir = wav_output_dir / speaker_name

    converted = 0
    skipped = 0
    failed = 0
    failed_files = []

    for m4a_file in speaker_dir.rglob("*.m4a"):
        relative = m4a_file.relative_to(speaker_dir)
        wav_file = output_speaker_dir / relative.with_suffix(".wav")

        if wav_file.exists():
            skipped += 1
            continue

        wav_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            if convert_aac_to_wav(m4a_file, wav_file, sample_rate):
                converted += 1
            else:
                failed += 1
                failed_files.append(str(m4a_file))
        except Exception as e:
            failed += 1
            failed_files.append(f"{m4a_file} | {e}")

    return {
        "speaker": speaker_name,
        "converted": converted,
        "skipped": skipped,
        "failed": failed,
        "failed_files": failed_files,
    }


def step_extract():
    """Step 1: 解压 VoxCeleb2 zip 文件"""
    print("=" * 60)
    print("Step 1: Extract VoxCeleb2 zip files")
    print("=" * 60)
    VC2_DEV_DIR.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(VC2_ZIP_DIR.glob("aac*.zip"))
    print(f"Found {len(zip_files)} zip files")

    for zf in zip_files:
        marker = VC2_DEV_DIR / f".{zf.stem}_done"
        if marker.exists():
            print(f"  Skip {zf.name} (already extracted)")
            continue
        extract_zip(zf, VC2_DEV_DIR / "aac")
        marker.touch()

    # Check extracted structure
    if VC2_AAC_DIR.exists():
        speakers = [d for d in VC2_AAC_DIR.iterdir() if d.is_dir()]
        print(f"Extracted speakers: {len(speakers)}")


def step_convert(workers=1):
    """Step 2: 将 AAC 转换为 WAV (16kHz mono)，支持多进程和断点续传"""
    print("\n" + "=" * 60)
    print(f"Step 2: Convert AAC to WAV (16kHz mono) | workers={workers}")
    print("=" * 60)
    VC2_WAV_DIR.mkdir(parents=True, exist_ok=True)

    if not VC2_AAC_DIR.exists():
        print("ERROR: AAC directory not found. Run --step extract first.")
        sys.exit(1)

    speakers = sorted([d for d in VC2_AAC_DIR.iterdir() if d.is_dir()])
    print(f"Speakers to convert: {len(speakers)}")

    # 加载进度
    progress = load_progress(CONVERT_PROGRESS_FILE)
    completed_speakers = set(progress.get("completed", []))
    print(f"Already completed: {len(completed_speakers)} speakers (resuming)")

    # 过滤未完成的说话人
    pending = [s for s in speakers if s.name not in completed_speakers]
    print(f"Pending: {len(pending)} speakers")

    if not pending:
        print("All speakers already converted!")
        return

    total_converted = 0
    total_skipped = 0
    total_failed = 0
    all_failed_files = []

    if workers <= 1:
        # 单进程
        for idx, spk_dir in enumerate(tqdm(pending, desc="Converting speakers")):
            result = process_speaker_convert(spk_dir, VC2_WAV_DIR)
            total_converted += result["converted"]
            total_skipped += result["skipped"]
            total_failed += result["failed"]
            all_failed_files.extend(result.get("failed_files", []))

            # 更新进度
            completed_speakers.add(spk_dir.name)
            progress["completed"] = list(completed_speakers)
            if result["failed"] > 0:
                progress["failed"].extend(result["failed_files"])
            save_progress(CONVERT_PROGRESS_FILE, progress)

            if (idx + 1) % 100 == 0:
                print(f"  Progress: {idx + 1}/{len(pending)} speakers | "
                      f"Converted: {total_converted} | Skipped: {total_skipped} | Failed: {total_failed}")
    else:
        # 多进程
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_speaker_convert, spk_dir, VC2_WAV_DIR): spk_dir
                for spk_dir in pending
            }
            for idx, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Converting speakers")):
                spk_dir = futures[future]
                result = future.result()
                total_converted += result["converted"]
                total_skipped += result["skipped"]
                total_failed += result["failed"]
                all_failed_files.extend(result.get("failed_files", []))

                completed_speakers.add(spk_dir.name)
                progress["completed"] = list(completed_speakers)
                save_progress(CONVERT_PROGRESS_FILE, progress)

                if (idx + 1) % 100 == 0:
                    print(f"  Progress: {idx + 1}/{len(pending)} speakers | "
                          f"Converted: {total_converted} | Skipped: {total_skipped} | Failed: {total_failed}")

    print(f"\nConversion complete:")
    print(f"  Converted: {total_converted}")
    print(f"  Skipped (already exists): {total_skipped}")
    print(f"  Failed: {total_failed}")
    if all_failed_files:
        fail_log = VC2_DEV_DIR / "convert_failed.log"
        with open(fail_log, "w", encoding="utf-8") as f:
            for item in all_failed_files:
                f.write(item + "\n")
        print(f"  Failed files logged: {fail_log}")


def step_features(apply_cmvn=False):
    """Step 3: 批量提取 FBank 特征"""
    print("\n" + "=" * 60)
    print("Step 3: Extract FBank features")
    print("=" * 60)

    if not VC2_WAV_DIR.exists():
        print("ERROR: WAV directory not found. Run --step convert first.")
        sys.exit(1)

    # 使用 extract_features.py 的 batch_extract
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from extract_features import batch_extract

    batch_extract(
        data_dir=str(VC2_WAV_DIR),
        output_dir=str(VC2_FEATURES_DIR),
        n_mels=80,
        sample_rate=16000,
        apply_cmvn=apply_cmvn,
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="VoxCeleb2 数据准备")
    parser.add_argument("--step", choices=["extract", "convert", "features", "all"], default="all")
    parser.add_argument("--workers", type=int, default=1, help="并行 worker 数（转换步骤）")
    parser.add_argument("--cmvn", action="store_true", help="特征提取时应用 CMVN 归一化")
    args = parser.parse_args()

    if args.step in ("extract", "all"):
        step_extract()

    if args.step in ("convert", "all"):
        step_convert(workers=args.workers)

    if args.step in ("features", "all"):
        step_features(apply_cmvn=args.cmvn)

    print("\n" + "=" * 60)
    print("VoxCeleb2 data preparation complete!")
    print(f"  Features: {VC2_FEATURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
