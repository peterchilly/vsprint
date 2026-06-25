"""
ERes2NetV2 说话人识别推理模块

包含:
- SpeakerEncoder: 加载模型，从音频文件提取说话人 embedding
- SpeakerVerifier: 比较两个音频文件，返回相似度分数和判定结果
- 支持批量推理

用法:
    from src.deploy.inference import SpeakerEncoder, SpeakerVerifier

    # 提取 embedding
    encoder = SpeakerEncoder(config_path="configs/train_config.yaml",
                             checkpoint_path="checkpoints/best_model.pth")
    embedding = encoder.extract_from_file("audio.wav")

    # 说话人验证
    verifier = SpeakerVerifier(config_path="configs/train_config.yaml",
                               checkpoint_path="checkpoints/best_model.pth",
                               threshold=0.5)
    score, is_same = verifier.verify("enroll.wav", "test.wav")
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import librosa
import yaml

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.eres2netv2 import create_eres2netv2


class SpeakerEncoder:
    """
    说话人编码器：加载模型，提取说话人 embedding

    参数:
        config_path: 配置文件路径
        checkpoint_path: 模型 checkpoint 路径
        device: 推理设备（None 自动选择）
    """

    def __init__(
        self,
        config_path: str = "configs/train_config.yaml",
        checkpoint_path: str = "checkpoints/best_model.pth",
        device: Optional[torch.device] = None,
    ):
        self.project_root = PROJECT_ROOT

        # 加载配置
        config_full = self.project_root / config_path
        with open(config_full, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # 设备
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型
        self.model = self._load_model(checkpoint_path)

        # 音频预处理参数
        audio_cfg = self._load_audio_config()
        self.sample_rate = audio_cfg.get("sample_rate", 16000)
        self.n_mels = audio_cfg.get("n_mels", 80)
        self.frame_length = audio_cfg.get("frame_length", 400)
        self.frame_shift = audio_cfg.get("frame_shift", 160)
        self.n_fft = audio_cfg.get("n_fft", 512)
        self.f_min = audio_cfg.get("f_min", 0.0)
        self.f_max = audio_cfg.get("f_max", 8000.0)
        self.preemphasis = audio_cfg.get("preemphasis", 0.97)

        self.fixed_length = self.config["data"].get("fixed_length", 200)

    def _load_audio_config(self) -> dict:
        """加载音频配置"""
        data_config_path = self.project_root / "configs" / "data_config.yaml"
        if data_config_path.exists():
            with open(data_config_path, "r", encoding="utf-8") as f:
                data_cfg = yaml.safe_load(f)
            return data_cfg.get("audio", {})
        return {}

    def _load_model(self, checkpoint_path: str) -> nn.Module:
        """加载训练好的模型"""
        ckpt_full = self.project_root / checkpoint_path
        print(f"[CKPT] 加载 checkpoint: {ckpt_full}")
        checkpoint = torch.load(str(ckpt_full), map_location="cpu", weights_only=False)

        model_config = self.config["model"]
        num_speakers = checkpoint.get("num_speakers", None)
        if num_speakers is None:
            for key in checkpoint["model_state_dict"]:
                if "classifier.weight" in key:
                    num_speakers = checkpoint["model_state_dict"][key].shape[0]
                    break

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
        model = model.to(self.device)
        model.eval()

        print(f"   [OK] 模型加载成功 (epoch {checkpoint.get('epoch', '?')}, "
              f"params={model.num_parameters() / 1e6:.2f}M)")
        return model

    def load_audio(self, audio_path: str, target_sr: Optional[int] = None) -> np.ndarray:
        """加载音频文件，重采样到目标采样率"""
        target_sr = target_sr or self.sample_rate
        waveform, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        return waveform  # (n_samples,) numpy array

    def extract_fbank(self, waveform: np.ndarray) -> torch.Tensor:
        """
        从波形提取 FBank 特征

        参数:
            waveform: (n_samples,) 音频波形

        返回:
            fbank: (1, n_mels, fixed_length) FBank 特征
        """
        # 预加重
        if self.preemphasis > 0:
            waveform = np.append(waveform[0], waveform[1:] - self.preemphasis * waveform[:-1])

        # FBank 提取 (librosa)
        fbank = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.frame_shift,
            win_length=self.frame_length,
            fmin=self.f_min,
            fmax=self.f_max,
            n_mels=self.n_mels,
            power=2.0,
        )  # (n_mels, time)

        # 对数压缩
        fbank = np.log(np.clip(fbank, a_min=1e-10, a_max=None))

        # CMVN 归一化
        mean = fbank.mean(axis=1, keepdims=True)
        std = fbank.std(axis=1, keepdims=True)
        fbank = (fbank - mean) / (std + 1e-8)

        # 转为 tensor
        fbank = torch.tensor(fbank, dtype=torch.float32, device=self.device)

        # 填充/截断到固定长度
        time_frames = fbank.shape[1]
        if time_frames >= self.fixed_length:
            fbank = fbank[:, :self.fixed_length]
        else:
            # 重复填充
            repeats = (self.fixed_length // time_frames) + 1
            fbank = fbank.repeat(1, repeats)[:, :self.fixed_length]

        return fbank.unsqueeze(0)  # (1, n_mels, fixed_length)

    @torch.no_grad()
    def extract_from_file(self, audio_path: str) -> np.ndarray:
        """
        从音频文件提取说话人 embedding

        参数:
            audio_path: 音频文件路径

        返回:
            embedding: (embedding_dim,) L2 归一化的 embedding
        """
        waveform = self.load_audio(audio_path)
        fbank = self.extract_fbank(waveform)  # (1, n_mels, fixed_length)

        # 模型输入: (batch, 1, n_mels, time)
        fbank = fbank.unsqueeze(0)  # (1, 1, n_mels, time) — extract_fbank already added channel dim

        embedding, _ = self.model(fbank)
        embedding = torch.nn.functional.normalize(embedding, dim=1)

        return embedding.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def extract_from_files(self, audio_paths: List[str]) -> np.ndarray:
        """
        批量提取多个音频文件的说话人 embedding

        参数:
            audio_paths: 音频文件路径列表

        返回:
            embeddings: (n_files, embedding_dim) L2 归一化的 embedding
        """
        fbanks = []
        for audio_path in audio_paths:
            waveform = self.load_audio(audio_path)
            fbank = self.extract_fbank(waveform)  # (1, n_mels, fixed_length)
            fbanks.append(fbank.squeeze(0))  # (n_mels, fixed_length)

        # 堆叠
        fbank_batch = torch.stack(fbanks, dim=0)  # (n_files, n_mels, fixed_length)
        fbank_batch = fbank_batch.unsqueeze(1).to(self.device)  # (n_files, 1, n_mels, fixed_length)

        embeddings, _ = self.model(fbank_batch)
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)

        return embeddings.cpu().numpy()

    @torch.no_grad()
    def extract_from_waveform(self, waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        从原始波形提取说话人 embedding

        参数:
            waveform: (n_samples,) numpy 数组
            sr: 采样率

        返回:
            embedding: (embedding_dim,) L2 归一化的 embedding
        """
        if sr != self.sample_rate:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)

        fbank = self.extract_fbank(waveform)
        fbank = fbank.unsqueeze(0)  # (1, 1, n_mels, fixed_length)

        embedding, _ = self.model(fbank)
        embedding = torch.nn.functional.normalize(embedding, dim=1)

        return embedding.squeeze(0).cpu().numpy()


class SpeakerVerifier:
    """
    说话人验证器：比较两个音频文件，返回相似度分数和判定结果

    参数:
        config_path: 配置文件路径
        checkpoint_path: 模型 checkpoint 路径
        threshold: 判定阈值（余弦相似度）
        device: 推理设备
    """

    def __init__(
        self,
        config_path: str = "configs/train_config.yaml",
        checkpoint_path: str = "checkpoints/best_model.pth",
        threshold: float = 0.5,
        device: Optional[torch.device] = None,
    ):
        self.encoder = SpeakerEncoder(config_path, checkpoint_path, device)
        self.threshold = threshold

    def verify(self, enroll_path: str, test_path: str) -> Tuple[float, bool]:
        """
        验证两个音频是否来自同一说话人

        参数:
            enroll_path: 注册音频路径
            test_path: 测试音频路径

        返回:
            score: 余弦相似度分数
            is_same: 是否判定为同一说话人
        """
        enroll_emb = self.encoder.extract_from_file(enroll_path)
        test_emb = self.encoder.extract_from_file(test_path)

        # 余弦相似度（embedding 已 L2 归一化）
        score = float(np.dot(enroll_emb, test_emb))
        is_same = score >= self.threshold

        return score, is_same

    def verify_batch(
        self,
        pairs: List[Tuple[str, str]],
    ) -> List[Tuple[float, bool]]:
        """
        批量验证多个音频对

        参数:
            pairs: (注册音频, 测试音频) 路径列表

        返回:
            results: (相似度分数, 是否同一说话人) 列表
        """
        # 收集所有音频路径（去重）
        all_paths = []
        path_to_idx = {}
        for enroll, test in pairs:
            if enroll not in path_to_idx:
                path_to_idx[enroll] = len(all_paths)
                all_paths.append(enroll)
            if test not in path_to_idx:
                path_to_idx[test] = len(all_paths)
                all_paths.append(test)

        # 批量提取 embedding
        embeddings = self.encoder.extract_from_files(all_paths)

        # 计算每对的相似度
        results = []
        for enroll, test in pairs:
            enroll_emb = embeddings[path_to_idx[enroll]]
            test_emb = embeddings[path_to_idx[test]]
            score = float(np.dot(enroll_emb, test_emb))
            is_same = score >= self.threshold
            results.append((score, is_same))

        return results

    def set_threshold(self, threshold: float):
        """更新判定阈值"""
        self.threshold = threshold


class SpeakerIdentifier:
    """
    说话人识别器：从已注册的说话人中识别

    参数:
        config_path: 配置文件路径
        checkpoint_path: 模型 checkpoint 路径
        device: 推理设备
    """

    def __init__(
        self,
        config_path: str = "configs/train_config.yaml",
        checkpoint_path: str = "checkpoints/best_model.pth",
        device: Optional[torch.device] = None,
    ):
        self.encoder = SpeakerEncoder(config_path, checkpoint_path, device)
        self.enrolled_speakers: Dict[str, np.ndarray] = {}  # name -> embedding

    def enroll(self, name: str, audio_path: str) -> np.ndarray:
        """注册说话人

        参数:
            name: 说话人名称
            audio_path: 音频文件路径

        返回:
            embedding: 说话人 embedding
        """
        embedding = self.encoder.extract_from_file(audio_path)
        self.enrolled_speakers[name] = embedding
        print(f"[ENROLL] 已注册说话人: {name} (共 {len(self.enrolled_speakers)} 人)")
        return embedding

    def enroll_batch(self, speakers: Dict[str, str]):
        """批量注册说话人

        参数:
            speakers: {name: audio_path} 字典
        """
        names = list(speakers.keys())
        paths = list(speakers.values())

        embeddings = self.encoder.extract_from_files(paths)
        for name, emb in zip(names, embeddings):
            self.enrolled_speakers[name] = emb

        print(f"[ENROLL] 批量注册 {len(names)} 个说话人 (共 {len(self.enrolled_speakers)} 人)")

    def identify(self, audio_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        识别音频中的说话人

        参数:
            audio_path: 音频文件路径
            top_k: 返回最相似的 top_k 个说话人

        返回:
            results: [(speaker_name, score), ...] 按相似度降序
        """
        if not self.enrolled_speakers:
            raise RuntimeError("未注册任何说话人，请先调用 enroll()")

        test_emb = self.encoder.extract_from_file(audio_path)

        scores = []
        for name, enroll_emb in self.enrolled_speakers.items():
            score = float(np.dot(test_emb, enroll_emb))
            scores.append((name, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def list_speakers(self) -> List[str]:
        """列出所有已注册的说话人"""
        return list(self.enrolled_speakers.keys())

    def remove_speaker(self, name: str) -> bool:
        """移除已注册的说话人"""
        if name in self.enrolled_speakers:
            del self.enrolled_speakers[name]
            return True
        return False

    def clear(self):
        """清空所有已注册的说话人"""
        self.enrolled_speakers.clear()


if __name__ == "__main__":
    # 简单测试
    import argparse as _argparse

    _parser = _argparse.ArgumentParser(description="说话人识别推理测试")
    _parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    _parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    _args = _parser.parse_args()

    encoder = SpeakerEncoder(_args.config, _args.checkpoint)
    print(f"\n[OK] SpeakerEncoder 初始化成功")
    print(f"   Embedding 维度: {encoder.config['model']['embedding_dim']}")
    print(f"   设备: {encoder.device}")
