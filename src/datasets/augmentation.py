"""
说话人识别数据增强模块

作用于 FBank 特征（非原始音频），可集成到 SpeakerDataset。

包含：
- SpecAugment: 时间遮蔽 + 频率遮蔽
- RandomGain: 随机增益偏移
- SpeedPerturb: 时间轴 resampling 模拟速度扰动
- Compose: 组合多个增强
"""

import numpy as np
from scipy.interpolate import interp1d


class SpecAugment:
    """SpecAugment: 时间遮蔽 + 频率遮蔽

    直接作用于 FBank 特征矩阵，将部分区域置零。

    参数:
        time_mask: 时间遮蔽最大宽度（帧数）
        freq_mask: 频率遮蔽最大宽度（mel 数）
        num_masks: 遮蔽数量（每种各应用几次）
    """

    def __init__(self, time_mask=10, freq_mask=20, num_masks=2):
        self.time_mask = time_mask
        self.freq_mask = freq_mask
        self.num_masks = num_masks

    def __call__(self, fbank):
        """
        参数:
            fbank: (n_mels, n_frames) numpy 数组

        返回:
            fbank: (n_mels, n_frames) 增强后的特征
        """
        n_mels, n_frames = fbank.shape
        fbank = fbank.copy()

        # 时间遮蔽
        for _ in range(self.num_masks):
            if n_frames > self.time_mask:
                t = np.random.randint(0, self.time_mask)
                t0 = np.random.randint(0, n_frames - t)
                fbank[:, t0:t0 + t] = 0.0

        # 频率遮蔽
        for _ in range(self.num_masks):
            if n_mels > self.freq_mask:
                f = np.random.randint(0, self.freq_mask)
                f0 = np.random.randint(0, n_mels - f)
                fbank[f0:f0 + f, :] = 0.0

        return fbank


class RandomGain:
    """随机增益：对 FBank 特征添加偏移

    模拟 ±gain_range dB 的音量变化。在 log-FBank 域中，
    增益变化对应于特征的常数偏移。

    参数:
        gain_range: 增益范围（dB），实际偏移 = gain_range / 10 * (2*random-1)
    """

    def __init__(self, gain_range=5.0):
        self.gain_range = gain_range

    def __call__(self, fbank):
        """
        参数:
            fbank: (n_mels, n_frames) numpy 数组

        返回:
            fbank: (n_mels, n_frames) 增强后的特征
        """
        # 在 log 域中，±dB 的增益对应于乘以一个系数
        # log_power + gain_offset, 其中 gain_offset = gain_db / 10
        gain_db = np.random.uniform(-self.gain_range, self.gain_range)
        gain_offset = gain_db / 10.0
        return fbank + gain_offset


class SpeedPerturb:
    """速度扰动：在 FBank 特征层面通过时间轴 resampling 模拟速度变化

    速度因子 > 1 加速（时间缩短），< 1 减速（时间拉长）。
    在 (n_mels, n_frames) 特征上沿时间轴做线性插值。

    参数:
        speed_factors: 速度因子列表，如 [0.9, 1.0, 1.1]
        prob: 应用概率（1.0 时总是应用随机扰动，0.5 时半概率）
    """

    def __init__(self, speed_factors=None, prob=0.5):
        self.speed_factors = speed_factors or [0.9, 1.0, 1.1]
        self.prob = prob

    def __call__(self, fbank):
        """
        参数:
            fbank: (n_mels, n_frames) numpy 数组

        返回:
            fbank: (n_mels, n_frames) 速度扰动后的特征（帧数不变）
        """
        if np.random.random() > self.prob:
            return fbank

        # 随机选择速度因子（排除 1.0 的无操作情况时仍然可以选到）
        speed = np.random.choice(self.speed_factors)
        if speed == 1.0:
            return fbank

        n_mels, n_frames = fbank.shape
        # 速度因子 > 1 → 时间缩短 → 帧数减少
        # 速度因子 < 1 → 时间拉长 → 帧数增多
        new_n_frames = max(1, int(n_frames / speed))

        # 原始时间轴坐标
        orig_indices = np.linspace(0, n_frames - 1, n_frames)
        # 新时间轴坐标
        new_indices = np.linspace(0, n_frames - 1, new_n_frames)

        # 对每个 mel 频带做线性插值
        perturbed = np.zeros((n_mels, new_n_frames), dtype=fbank.dtype)
        for m in range(n_mels):
            interp_func = interp1d(
                orig_indices, fbank[m, :],
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate',
            )
            perturbed[m, :] = interp_func(new_indices)

        # 截断或循环填充到原始帧数
        if new_n_frames >= n_frames:
            return perturbed[:, :n_frames]
        else:
            # 不够则重复填充
            repeats = (n_frames // new_n_frames) + 1
            perturbed = np.tile(perturbed, (1, repeats))
            return perturbed[:, :n_frames]


class Compose:
    """组合多个增强变换

    参数:
        transforms: 增强变换列表
    """

    def __init__(self, transforms=None):
        self.transforms = transforms or []

    def __call__(self, fbank):
        for t in self.transforms:
            fbank = t(fbank)
        return fbank


def create_augmentation(config):
    """根据配置创建增强 Compose

    参数:
        config: dict, 增强配置（来自 YAML）

    返回:
        Compose 实例
    """
    transforms = []

    if config.get("spec_augment"):
        sa = config["spec_augment"]
        transforms.append(SpecAugment(
            time_mask=sa.get("time_mask", 10),
            freq_mask=sa.get("freq_mask", 20),
            num_masks=sa.get("num_masks", 2),
        ))

    if config.get("random_gain"):
        rg = config["random_gain"]
        transforms.append(RandomGain(
            gain_range=rg.get("gain_range", 5.0),
        ))

    if config.get("speed_perturb"):
        sp = config["speed_perturb"]
        transforms.append(SpeedPerturb(
            speed_factors=sp.get("speed_factors", [0.9, 1.0, 1.1]),
            prob=sp.get("prob", 0.5),
        ))

    return Compose(transforms)
