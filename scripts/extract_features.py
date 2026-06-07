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
from pathlib import Path
import numpy as np

import librosa
import librosa.display
import soundfile as sf
from tqdm import tqdm

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


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
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.n_fft = n_fft
        self.f_min = f_min
        self.f_max = f_max
        self.preemphasis = preemphasis

    def load_audio(self, audio_path: str, duration: float = None):
        """加载音频文件，重采样到 16kHz 单声道"""
        audio, sr = librosa.load(
            audio_path,
            sr=self.sample_rate,
            mono=True,
            duration=duration,
        )
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

        return log_fbank

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
    print("🧪 测试 FBank 特征提取 pipeline...")

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
    print(f"  ✅ FBank 特征形状: {fbank.shape}")
    print(f"     帧数: {fbank.shape[0]}")
    print(f"     特征维度: {fbank.shape[1]}")

    # 测试固定长度填充
    fbank_fixed = preprocessor.pad_or_truncate(fbank, 300)
    print(f"  ✅ 固定长度 FBank 形状: {fbank_fixed.shape}")

    # 测试截断
    fbank_short = preprocessor.pad_or_truncate(fbank, 100)
    print(f"  ✅ 截断 FBank 形状: {fbank_short.shape}")

    # 测试 preemphasis
    emphasized = preprocessor.apply_preemphasis(test_audio)
    print(f"  ✅ 预加重后信号长度: {len(emphasized)}")

    print("\n✅ Pipeline 测试通过！")


def batch_extract(data_dir: str, output_dir: str, n_mels: int = 80, sample_rate: int = 16000):
    """批量提取 FBank 特征"""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = AudioPreprocessor(
        sample_rate=sample_rate,
        n_mels=n_mels,
    )

    # 查找所有音频文件
    audio_files = list(data_dir.rglob("*.wav"))
    if not audio_files:
        audio_files = list(data_dir.rglob("*.ogg"))
    if not audio_files:
        audio_files = list(data_dir.rglob("*.m4a"))

    if not audio_files:
        print(f"❌ 在 {data_dir} 中未找到音频文件")
        return

    print(f"📁 找到 {len(audio_files)} 个音频文件")
    print(f"📦 输出目录: {output_dir}")

    success = 0
    failed = 0

    for audio_path in tqdm(audio_files, desc="提取特征"):
        try:
            # 构建输出路径（保持说话人目录结构）
            relative_path = audio_path.relative_to(data_dir)
            output_path = output_dir / relative_path.with_suffix(".npy")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 提取特征
            fbank = preprocessor.process_file(str(audio_path))

            # 保存
            np.save(str(output_path), fbank.astype(np.float32))
            success += 1

        except Exception as e:
            print(f"\n  ❌ 处理失败: {audio_path}")
            print(f"     错误: {e}")
            failed += 1

    print(f"\n✅ 完成！成功: {success}, 失败: {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="音频预处理和 FBank 特征提取")
    parser.add_argument("--audio", type=str, help="单个音频文件路径")
    parser.add_argument("--batch", action="store_true", help="批量处理模式")
    parser.add_argument("--data-dir", type=str, help="数据集目录")
    parser.add_argument("--output-dir", type=str, default="data/voxceleb/features", help="输出目录")
    parser.add_argument("--n-mels", type=int, default=80, help="FBank 维度")
    parser.add_argument("--sample-rate", type=int, default=16000, help="采样率")
    parser.add_argument("--test", action="store_true", help="运行测试")
    args = parser.parse_args()

    if args.test:
        test_pipeline()
    elif args.batch:
        if not args.data_dir:
            print("❌ --batch 模式需要指定 --data-dir")
            sys.exit(1)
        batch_extract(args.data_dir, args.output_dir, args.n_mels, args.sample_rate)
    elif args.audio:
        preprocessor = AudioPreprocessor(n_mels=args.n_mels, sample_rate=args.sample_rate)
        fbank = preprocessor.process_file(args.audio)
        print(f"✅ FBank 特征形状: {fbank.shape}")
        print(f"   帧数: {fbank.shape[0]}")
        print(f"   维度: {fbank.shape[1]}")
    else:
        print("使用 --help 查看用法")
