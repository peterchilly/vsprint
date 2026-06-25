"""
音频预处理和 FBank 特征提取工具

用法:
    # 从音频文件提取 FBank 特征
    python scripts/extract_features.py --audio path/to/audio.wav

    # 批量提取整个数据集的 FBank 特征
    python scripts/extract_features.py --batch --data-dir data/voxceleb/dev --output-dir data/voxceleb/features

    # 测试特征提取 pipeline
    python scripts/extract_features.py --test
"""

import argparse
import os
import sys
import logging
import datetime
import subprocess
import tempfile
from pathlib import Path
import numpy as np

import librosa
import soundfile as sf
from tqdm import tqdm

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 日志目录
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ffmpeg 路径：优先 PATH，其次 imageio_ffmpeg
def _find_ffmpeg():
    """查找 ffmpeg 可执行文件路径"""
    # 1. 系统 PATH
    import shutil as _shutil
    ffmpeg_in_path = _shutil.which("ffmpeg")
    if ffmpeg_in_path:
        return ffmpeg_in_path
    # 2. imageio_ffmpeg 包附带的 ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    # 3. 常见安装路径
    for p in [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    ]:
        if Path(p).exists():
            return p
    return None

FFMPEG_PATH = _find_ffmpeg()


class AudioPreprocessor:
    """音频预处理器"""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        frame_length: int = 400,    # 25ms @ 16kHz
        frame_shift: int = 160,     # 10ms @ 16kHz
        n_fft: int = 512,
        f_min: float = 0.0,
        f_max: float = 8000.0,
        preemphasis: float = 0.97,
        apply_cmvn: bool = False,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.n_fft = n_fft
        self.f_min = f_min
        self.f_max = f_max
        self.preemphasis = preemphasis
        self.apply_cmvn = apply_cmvn

    def load_audio(self, audio_path: str, duration: float = None):
        """加载音频文件，重采样到 16kHz 单声道

        支持 .wav, .m4a (AAC), .ogg, .flac, .mp3 等格式。
        对于非 WAV 格式，使用 ffmpeg 转为临时 WAV 再加载。
        """
        audio_path = str(audio_path)
        ext = Path(audio_path).suffix.lower()

        # WAV / FLAC 可以直接用 librosa
        if ext in (".wav", ".flac"):
            audio, sr = librosa.load(
                audio_path,
                sr=self.sample_rate,
                mono=True,
                duration=duration,
            )
            return audio

        # 非 WAV 格式（m4a, mp3, ogg 等）使用 ffmpeg 转码
        if FFMPEG_PATH is None:
            raise RuntimeError(
                "ffmpeg not found. Install ffmpeg or imageio-ffmpeg to read non-WAV formats."
            )

        # 使用 ffmpeg 解码为 raw PCM，通过 soundfile 读取
        cmd = [
            FFMPEG_PATH, "-y", "-i", audio_path,
            "-f", "s16le", "-acodec", "pcm_s16le",
            "-ar", str(self.sample_rate), "-ac", "1",
            "-",  # 输出到 stdout
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)

        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed to decode {audio_path}: {result.stderr.decode('utf-8', errors='replace')[:500]}"
            )

        # 从 raw PCM 字节流解析为 numpy 数组
        audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0

        if duration is not None:
            audio = audio[:int(self.sample_rate * duration)]

        return audio

    def apply_preemphasis(self, audio: np.ndarray):
        """预加重"""
        return np.append(audio[0], audio[1:] - self.preemphasis * audio[:-1])

    def extract_fbank(self, audio: np.ndarray, use_energy: bool = True) -> np.ndarray:
        """
        提取 FBank（Filter Bank）特征

        参数:
            audio: 音频信号
            use_energy: 是否包含对数能量

        返回:
            FBank 特征 (n_frames, n_mels) 或 (n_frames, n_mels+1)
        """
        # 预加重
        audio = self.apply_preemphasis(audio)

        # 计算 FBank
        fbank = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.frame_shift,
            win_length=self.frame_length,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
            power=2.0,
        )

        # 对数压缩
        log_fbank = np.log(np.clip(fbank, a_min=1e-10, a_max=None))  # (n_mels, n_frames)

        # 转置为 (n_frames, n_mels)
        log_fbank = log_fbank.T

        if use_energy:
            # 添加对数能量
            energy = np.log(np.clip(np.sum(audio ** 2, axis=-1), a_min=1e-10, a_max=None))
            energy = np.full((log_fbank.shape[0], 1), energy)
            log_fbank = np.concatenate([log_fbank, energy], axis=-1)

        # CMVN 归一化（Cepstral Mean Variance Normalization）
        if self.apply_cmvn:
            log_fbank = self.apply_cmvn_norm(log_fbank)

        return log_fbank

    @staticmethod
    def apply_cmvn_norm(fbank: np.ndarray) -> np.ndarray:
        """应用 CMVN (Cepstral Mean Variance Normalization)

        对每一句话的特征做均值方差归一化，消除信道效应。

        参数:
            fbank: (n_frames, n_mels) 特征矩阵

        返回:
            归一化后的特征 (n_frames, n_mels)
        """
        mean = fbank.mean(axis=0, keepdims=True)
        std = fbank.std(axis=0, keepdims=True) + 1e-10
        return (fbank - mean) / std

    def pad_or_truncate(self, fbank: np.ndarray, target_frames: int) -> np.ndarray:
        """填充或截断到固定帧数"""
        n_frames = fbank.shape[0]
        if n_frames >= target_frames:
            return fbank[:target_frames]
        else:
            # 循环填充
            repeats = (target_frames // n_frames) + 1
            fbank = np.tile(fbank, (repeats, 1))
            return fbank[:target_frames]

    def process_file(self, audio_path: str, fixed_length: int = None) -> np.ndarray:
        """处理单个音频文件，返回 FBank 特征"""
        audio = self.load_audio(audio_path)
        fbank = self.extract_fbank(audio)

        if fixed_length is not None:
            fbank = self.pad_or_truncate(fbank, fixed_length)

        return fbank


def test_pipeline():
    """测试特征提取 pipeline"""
    print("[TEST] FBank feature extraction pipeline...")

    preprocessor = AudioPreprocessor(
        sample_rate=16000,
        n_mels=80,
    )

    # 生成测试音频（纯合成）
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    # 合成一个包含多个频率的测试信号
    test_audio = (
        0.5 * np.sin(2 * np.pi * 200 * t) +
        0.3 * np.sin(2 * np.pi * 400 * t) +
        0.2 * np.sin(2 * np.pi * 800 * t)
    )

    # 测试 FBank 提取
    fbank = preprocessor.extract_fbank(test_audio)
    print(f"  [OK] FBank shape: {fbank.shape}")
    print(f"     frames: {fbank.shape[0]}")
    print(f"     dims: {fbank.shape[1]}")

    # 测试固定长度填充
    fbank_fixed = preprocessor.pad_or_truncate(fbank, 300)
    print(f"  [OK] Fixed length FBank shape: {fbank_fixed.shape}")

    # 测试截断
    fbank_short = preprocessor.pad_or_truncate(fbank, 100)
    print(f"  [OK] Truncated FBank shape: {fbank_short.shape}")

    # 测试 preemphasis
    emphasized = preprocessor.apply_preemphasis(test_audio)
    print(f"  [OK] Pre-emphasized signal length: {len(emphasized)}")

    print("\n[OK] Pipeline test passed!")


def batch_extract(data_dir: str, output_dir: str, n_mels: int = 80, sample_rate: int = 16000, apply_cmvn: bool = False):
    """批量提取 FBank 特征（支持断点续传 + 日志记录）

    支持 .wav, .m4a (AAC), .ogg, .flac, .mp3 等音频格式。
    librosa 底层通过 ffmpeg 读取非 WAV 格式。

    参数:
        data_dir: 输入音频目录
        output_dir: 输出特征目录
        n_mels: FBank 维度
        sample_rate: 采样率
        apply_cmvn: 是否应用 CMVN 归一化
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置日志
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"extract_features_{timestamp}.log"

    # 创建独立的 logger，不干扰第三方库
    logger = logging.getLogger("extract_features")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # 清除之前的 handlers

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    # 压制第三方库的噪音
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("librosa").setLevel(logging.WARNING)
    logger.info("=" * 60)
    logger.info("FBank 特征提取开始")
    logger.info(f"日志文件: {log_file}")
    logger.info(f"数据目录: {data_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"参数: n_mels={n_mels}, sample_rate={sample_rate}, CMVN={apply_cmvn}")
    logger.info("=" * 60)

    preprocessor = AudioPreprocessor(
        sample_rate=sample_rate,
        n_mels=n_mels,
        apply_cmvn=apply_cmvn,
    )

    # 查找所有音频文件（支持多种格式）
    audio_files = []
    for ext in ("*.wav", "*.m4a", "*.ogg", "*.flac", "*.mp3"):
        audio_files.extend(data_dir.rglob(ext))

    if not audio_files:
        logger.error(f"未找到音频文件: {data_dir}")
        sys.exit(1)

    logger.info(f"找到 {len(audio_files)} 个音频文件")

    success = 0
    failed = 0
    skipped = 0  # 已提取的文件跳过
    error_list = []  # 记录错误详情

    for idx, audio_path in enumerate(tqdm(audio_files, desc="Extracting features"), 1):
        try:
            # 构建输出路径（保持说话人目录结构）
            relative_path = audio_path.relative_to(data_dir)
            output_path = output_dir / relative_path.with_suffix(".npy")

            # 跳过已提取的文件（断点续传）
            if output_path.exists():
                skipped += 1
                continue

            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 提取特征
            fbank = preprocessor.process_file(str(audio_path))

            # 保存
            np.save(str(output_path), fbank.astype(np.float32))
            success += 1

        except Exception as e:
            error_info = f"[{idx}/{len(audio_files)}] {audio_path.relative_to(data_dir)} | {type(e).__name__}: {e}"
            logger.error(error_info)
            error_list.append(error_info)
            failed += 1

        # 每 5000 个文件输出一次进度汇总
        total_done = success + failed + skipped
        if total_done > 0 and total_done % 5000 == 0:
            logger.info(
                f"进度: {total_done}/{len(audio_files)}, "
                f"成功={success}, 跳过={skipped}, 失败={failed}"
            )

    # 最终汇总
    logger.info("=" * 60)
    logger.info("提取完成")
    logger.info(f"总数: {len(audio_files)}")
    logger.info(f"成功: {success}")
    logger.info(f"跳过: {skipped}")
    logger.info(f"失败: {failed}")
    if error_list:
        logger.info(f"\n--- 失败文件列表 ({len(error_list)} 个) ---")
        for err in error_list:
            logger.info(f"  {err}")
    logger.info("=" * 60)

    return failed > 0  # 有失败则返回 True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="音频预处理和 FBank 特征提取")
    parser.add_argument("--audio", type=str, help="单个音频文件路径")
    parser.add_argument("--batch", action="store_true", help="批量处理模式")
    parser.add_argument("--data-dir", type=str, help="数据集目录")
    parser.add_argument("--output-dir", type=str, default="data/voxceleb/features", help="输出目录")
    parser.add_argument("--n-mels", type=int, default=80, help="FBank 维度")
    parser.add_argument("--sample-rate", type=int, default=16000, help="采样率")
    parser.add_argument("--cmvn", action="store_true", help="应用 CMVN 归一化")
    parser.add_argument("--test", action="store_true", help="运行测试")
    args = parser.parse_args()

    if args.test:
        test_pipeline()
    elif args.batch:
        if not args.data_dir:
            print("❌ --batch 模式需要指定 --data-dir")
            sys.exit(1)
        has_error = batch_extract(args.data_dir, args.output_dir, args.n_mels, args.sample_rate, apply_cmvn=args.cmvn)
        if has_error:
            sys.exit(2)  # 有失败文件，返回非零状态码
    elif args.audio:
        preprocessor = AudioPreprocessor(n_mels=args.n_mels, sample_rate=args.sample_rate, apply_cmvn=args.cmvn)
        fbank = preprocessor.process_file(args.audio)
        print(f"[OK] FBank shape: {fbank.shape}")
        print(f"   frames: {fbank.shape[0]}")
        print(f"   dims: {fbank.shape[1]}")
    else:
        print("Use --help for usage")
