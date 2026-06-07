# 数据集说明

## 目录结构

```
data/
├── README.md                          # 本文件
├── voxceleb/                          # VoxCeleb 音频数据集
│   ├── dev/                           # VoxCeleb1 Dev（训练集）
│   │   ├── wav/
│   │   │   └── idXXXXX/               # 每个说话人一个子目录
│   │   │       └── *.wav              # 语音片段
│   │   └── vox1_dev_wav.zip           # 原始压缩包（下载后删除）
│   ├── test/                          # VoxCeleb1 Test（测试集）
│   │   ├── wav/
│   │   └── vox1_test_wav.zip
│   ├── meta/                          # 元数据
│   │   ├── vox1_meta.csv              # 说话人信息
│   │   └── veri_test2.txt             # 测试 trials 文件
│   └── splits/                        # 数据集划分索引
│       ├── train.json
│       ├── val.json
│       └── test.json
├── musan/                             # MUSAN 噪声库（可选，用于数据增强）
│   ├── speech/
│   ├── music/
│   └── noise/
└── rir/                               # RIR（可选，用于数据增强）
    └── RIRS_NOISES/
```

## 数据集信息

### VoxCeleb1

| 子集 | 说话人数 | 语音条数 | 用途 |
|------|----------|----------|------|
| Dev | 1,211 | ~148,000 | 训练 + 验证 |
| Test (VoxCeleb-O) | 40 | ~6,000 | 测试 |

- **采样率**：16kHz（需从原始 48kHz 降采样）
- **声道**：单声道
- **格式**：.wav（从 .m4a 或 .ogg 提取）
- **来源**：YouTube 视频片段

### 获取方式

1. **手动下载**（推荐）：
   - VoxCeleb1: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html
   - 下载 `vox1_dev_wav.zip`（~12GB）和 `vox1_test_wav.zip`（~1.7GB）
   - 下载 `vox1_meta.csv` 和 `veri_test2.txt`

2. **Hugging Face Datasets**：
   ```python
   from datasets import load_dataset
   ds = load_dataset("essexial/voxceleb1")
   ```

3. **脚本自动下载**：
   运行 `python scripts/download_voxceleb.py`

### MUSAN 噪声库（可选）

- **用途**：数据增强（加性噪声）
- **下载**：https://www.openslr.org/17/

### RIR（可选）

- **用途**：数据增强（房间脉冲响应）
- **下载**：https://www.openslr.org/28/
