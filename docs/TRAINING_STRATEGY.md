# ERes2NetV2 训练策略总结

> **项目路径：** `C:\Users\Administrator\vsprint`
> **生成时间：** 2026-06-29
> **基于训练版本：** v1 ~ v6 共 6 次训练记录

---

## 目录

1. [历史训练概览](#1-历史训练概览)
2. [核心问题诊断](#2-核心问题诊断)
3. [根因分析](#3-根因分析)
4. [训练策略建议](#4-训练策略建议)
5. [推荐配置方案](#5-推荐配置方案)

---

## 1. 历史训练概览

| 版本 | 数据集 | 优化器 | LR | Margin/Scale | Dropout | 数据增强 | Best EER | 早停Epoch | 最终EER(退化) |
|------|--------|--------|-----|-------------|---------|----------|----------|-----------|---------------|
| v1 | VoxCeleb1 (1210人) | SGD | 0.1 | 0.2/30 | 0.4 | 无 | **0.1553** | 71 | 0.2494 |
| v2 | VoxCeleb1 | AdamW | 0.001 | 0.2/32 | 0.3 | SpecAugment+Gain | 0.1944 | 24 | 0.2169 |
| v3 | VoxCeleb1 | AdamW | 0.0005 | 0.3/40 | 0.1 | SpecAugment+Gain | 0.2050 | 26 | 0.2388 |
| v4 | VoxCeleb1+2 (7204人) | AdamW | 0.001 | 0.2/32 | 0.2 | SpecAugment+Gain+Speed | 未完成 | — | — |
| v5 | VoxCeleb1 | AdamW | 0.001 | 0.2/32 | 0.2 | SpecAugment+Gain+Speed | 未完成 | — | — |
| v6 | VoxCeleb1 (从v3恢复) | AdamW | 0.0001 | 0.3/40 | 0.3 | SpecAugment(增强) | **0.1478** | ~29 | 0.1663 |

### 关键观察

- **SOTA 参考线：** VoxCeleb1 上 SOTA 系统的 EER 通常在 3-5%，本项目最佳 14.78%，差距明显
- **所有完成的训练都出现过拟合：** Train Loss 持续下降，但 EER 在达到最佳后不可逆地恶化
- **v6 是目前最好结果（EER=0.1478）**，策略是从 v3 best checkpoint 恢复 + 极低 LR 微调

---

## 2. 核心问题诊断

### 问题一：过拟合 — 所有训练的首要矛盾

**现象：** 每次训练都呈现相同模式：

```
训练前期：Train Loss ↓  EER ↓  → 模型在学习泛化特征
训练后期：Train Loss ↓  EER ↑  → 模型在记忆训练集说话人
```

**具体数据：**

| 版本 | 最佳EER出现Epoch | 最佳EER | 末轮EER | 退化幅度 | Train Acc末轮 |
|------|-----------------|---------|---------|---------|--------------|
| v1 | ~46 | 0.1553 | 0.2494 | +60% | 0.1% (分类无意义) |
| v2 | 9 | 0.1944 | 0.2169 | +12% | 73.3% |
| v3 | 10 | 0.2050 | 0.2388 | +16% | 84.2% |
| v6 | ~5 | 0.1478 | 0.1663 | +12% | 0.1% |

**Pos mean 持续下降的警示：**
- v1 训练中，同一说话人的余弦相似度均值从 0.3128 降至 0.1621
- 这意味着模型把训练集说话人的 embedding 推向了分类权重方向，但牺牲了 embedding 空间的聚类结构
- embedding 不再"聚拢"，而是"分散"在单位球面上

### 问题二：数据量不足

- VoxCeleb1 Dev 仅 1,210 说话人、约 147K 条音频
- 说话人识别需要大量不同说话人来学习泛化性 embedding
- SOTA 系统普遍使用 VoxCeleb1+2（~7000说话人、100万+条）
- 1,210 说话人 → 分类头仅 1089 维，模型容易"记住"所有人

### 问题三：工程稳定性问题

| 问题 | 影响 | 涉及版本 |
|------|------|---------|
| AMP 混合精度溢出 | AAM-Softmax 在 FP16 下数值溢出，梯度变 inf，权重被破坏 | v1 |
| 验证集 label 越界 | 验证集保留了原始 speaker ID (1089-1209)，但分类头只有 1089 类 | v1 |
| scatter_ CUDA 崩溃 | PyTorch 2.11 + RTX 5070 Ti 上 one-hot scatter 触发 kernel bug | v1 |
| 多数据源不兼容 | train.py 只读 `features_dir`，v4 配置写了 `features_dirs` | v4 |
| num_workers 内存溢出 | 100万+文件路径 pickle 到 8 个子进程导致 OOM | v4 |
| CUDA OOM | batch_size=128 超出 16GB 显存 | v4 |
| 损坏的 .npy 文件 | 磁盘满导致特征文件写不完整，np.load 报错 | v4/v5 |
| SpeedPerturb 崩溃 | 实时速度扰动实现有 bug，多次导致训练中断 | v4/v5 |

### 问题四：超参数调整缺乏方向性

- v2→v3：增大 margin (0.2→0.3)、增大 scale (32→40)、降低 dropout (0.3→0.1) → EER 反而变差 (0.1944→0.2050)
- v3 降低了 dropout 反而加剧过拟合
- margin=0.3 + scale=40 的组合过于激进，模型训练压力过大
- v6 回到 margin=0.3 + scale=40 但配合极低 LR (0.0001) 和高 dropout (0.3)，才取得最好结果

### 问题五：数据增强不充分

- v1 无任何增强 → 过拟合最严重
- v2/v3 加入 SpecAugment + RandomGain → 有改善但不够
- v4/v5 尝试 SpeedPerturb 但工程实现不稳定，未能完成训练
- 缺少加噪（MUSAN/RIR）、CMVN 归一化等关键增强

---

## 3. 根因分析

### 3.1 为什么过拟合如此严重

```
数据少 (1210人) + 模型容量大 (4.96M参数) + 说话人分类头 (1089类)
    → 模型有足够能力记住每个训练说话人的特征
    → 但泛化到验证集的 121 个新说话人时表现差
```

**关键矛盾：** 说话人识别的推理场景是"开集"——遇到的是训练时从未见过的说话人。但训练损失（AAM-Softmax）是"闭集"分类损失，天然倾向于记忆而非泛化。

### 3.2 为什么 EER 在后期不降反升

AAM-Softmax 的角度 margin 在训练后期会过度压缩训练集说话人的 embedding 分布：

1. **前期：** 模型学习通用的说话人特征（音高、共鸣峰、语速模式等），EER 改善
2. **后期：** 模型开始利用训练集说话人的个体特异性（环境噪声、录音通道等）来做分类
3. **结果：** embedding 空间被"扭曲"以适应训练集分类，但失去了对新说话人的泛化能力

**Pos mean 下降的证据：** 同一说话人的 embedding 相似度下降，说明 embedding 空间的几何结构被破坏。

### 3.3 为什么 v6 效果最好

v6 的成功策略：
- **从 v3 最佳 checkpoint 恢复**（跳过初始训练的不稳定阶段）
- **极低 LR (0.0001)**（微调而非从头训练，避免过度扰动）
- **高 dropout (0.3)**（强力正则化）
- **更激进的 SpecAugment**（4 masks，更宽遮蔽）
- **去掉 SpeedPerturb**（避免工程崩溃）
- **缩短 patience (12)**（一旦开始过拟合就停止）

本质上是一个"精心控制的微调"策略，但仍然无法避免后期过拟合。

---

## 4. 训练策略建议

### 策略一：扩充训练数据 — 最高优先级

**目标：** 从 1210 说话人扩充到 7000+ 说话人

**方案：**
- 修复 VoxCeleb2 特征提取（F盘 332 个说话人因磁盘满未完成）
- 将 VoxCeleb2 数据迁移到更大容量的磁盘
- 使用 `create_split_multi` 函数合并 VoxCeleb1+2
- 预期：说话人多样性提升 6 倍，过拟合大幅缓解

**预期收益：** EER 从 ~15% 降至 ~8-10%

### 策略二：数据增强体系化

**推荐的增强组合（按优先级排序）：**

| 增强方式 | 实现方式 | 预期收益 | 风险 |
|---------|---------|---------|------|
| SpecAugment | 时间/频率遮蔽，4 masks | 中 | 低（已验证可行） |
| CMVN | 对 FBank 做 per-utteration 均值方差归一化 | 中 | 低 |
| 加噪 (MUSAN) | 随机混入噪声（Babble/Noise/Music） | 高 | 中（需下载 MUSAN） |
| 加混响 (RIR) | 随机卷积房间脉冲响应 | 高 | 中（需下载 RIR） |
| Speed Perturb | 0.9x/1.0x/1.1x 变速 | 高 | 高（之前有崩溃问题，需修复实现） |
| RandomGain | 随机增益 ±5dB | 低 | 低 |

**关键：** 增强应在训练时**实时**进行（在 DataLoader 的 collate_fn 中），而非预计算。这样每个 epoch 看到的增强样本不同，等效于扩大数据集。

### 策略三：修复 SpeedPerturb 实现

之前的 SpeedPerturb 崩溃根因推测：
- 在 FBank 特征层面做速度扰动不准确（应该在原始音频层面做）
- 或重采样实现存在数值问题

**正确做法：**
- Speed perturbation 应在**音频→FBank 提取之前**进行
- 即：原始音频 → 变速重采样 → FBank 提取 → 模型输入
- 当前 pipeline 是两阶段（预提取 Fbank → 训练），无法在训练时做速度扰动
- **解决方案：** 预提取 3 个速度版本 (0.9x/1.0x/1.1x) 的 FBank 特征，训练时随机选取

### 策略四：调整损失函数策略

**当前问题：** AAM-Softmax 的 margin 压力在后期导致 embedding 空间扭曲

**建议方案：**

**方案 A — 降级 margin（保守）：**
- margin=0.2, scale=32（v2 配置，已验证）
- 配合更强的数据增强和更多数据
- 风险最低

**方案 B — 半监督对比学习（激进）：**
- 在 AAM-Softmax 之外加入 SubCenter 损失或 ProtoNet 损失
- 对 embedding 空间施加额外的聚类约束
- 让 embedding 不仅可分，还要"聚拢"

**方案 C — 两阶段训练：**
- 阶段1（Epoch 1-30）：AAM-Softmax (margin=0.1, scale=20) — 宽松训练，学泛化特征
- 阶段2（Epoch 31-60）：AAM-Softmax (margin=0.2, scale=32) — 收紧 margin，精调
- 通过损失函数退火避免后期过拟合

### 策略五：正则化组合拳

| 正则化手段 | 当前值 | 建议值 | 说明 |
|-----------|--------|--------|------|
| Dropout | 0.1-0.4 | **0.3** | v6 验证 0.3 效果好 |
| Weight Decay | 0.01-0.05 | **0.01** | AdamW 下 0.01 已足够 |
| Label Smoothing | 0.1-0.15 | **0.1** | 保持 |
| Early Stop Patience | 12-25 | **15** | 不宜过大，及时止损 |
| Gradient Clip | 1.0 | **1.0** | 保持 |

### 策略六：学习率策略优化

**当前问题：** v1 用 SGD lr=0.1 过于激进；v2-v6 用 AdamW lr=0.001-0.0001

**建议：**
- 优化器：**AdamW**（在小数据集上比 SGD 更稳定）
- 初始 LR：**0.001**
- 调度：**Warmup(5) + Cosine Annealing**
- 关键改进：**加入 LR Reduce on Plateau** — 当 EER 连续 5 个 epoch 不改善时，LR × 0.5
- 这样在 EER 开始恶化时自动降低学习率，而非等到 patience 用完

### 策略七：验证策略改进

**当前问题：** 验证集仅 121 说话人，EER 估计方差大

**建议：**
- 加入 VoxCeleb1 Test set (40 说话人) 作为额外验证
- 每次 EER 评估使用多组 trial pairs，减少方差
- 保存 embedding 用于后续 AS-Norm（自适应评分归一化）

### 策略八：工程稳定性保障

| 问题 | 修复方案 |
|------|---------|
| AMP 溢出 | 保持 `amp: false`，或对 AAM-Softmax 强制 FP32 |
| scatter_ 崩溃 | 已用 advanced indexing 替代，保持 |
| 损坏 .npy | DataLoader 加 try-except 跳过损坏文件 |
| num_workers OOM | 保持 `num_workers=0` 或 `4`（VoxCeleb1），大数据库可用 `memmap` |
| 多数据源 | 已修复 `create_split_multi`，确保可用 |
| 长时间训练 | 用 `Start-Process` 或 `background:true` 启动，避免 session 阻塞 |

---

## 5. 推荐配置方案

### 5.1 短期方案（立即可执行，基于现有数据）

```yaml
# train_v7.yaml — VoxCeleb1 only，最大化现有数据价值
training:
  epochs: 40              # 不要训练太久，40 足够
  batch_size: 64
  lr: 0.0005
  optimizer: AdamW
  weight_decay: 0.01
  lr_schedule: cosine
  warmup_epochs: 5
  grad_clip: 1.0
  amp: false              # 安全第一
  early_stopping_patience: 15
  label_smoothing: 0.1

model:
  variant: "34"
  n_mels: 80
  embedding_dim: 192
  scale: 4
  se_reduction: 8
  pool_attention_dim: 128
  dropout: 0.3            # v6 验证的最佳值

loss:
  type: aam_softmax
  margin: 0.2             # 回到保守值
  scale: 32.0
  label_smoothing: 0.1

data:
  features_dirs:
    - data/voxceleb/features
  fixed_length: 200       # 2秒，够用且省显存
  num_workers: 4
  augment: true
  augment_config:
    spec_augment:
      time_mask: 20
      freq_mask: 30
      num_masks: 4        # v6 验证的激进遮蔽
    random_gain:
      gain_range: 5.0
    # 不用 SpeedPerturb（预提取 Fbank 无法做）

checkpoint:
  save_dir: checkpoints_v7
  save_best: true
  save_interval: 3
  resume: checkpoints_v6/best_model.pth  # 从最佳 checkpoint 恢复
```

**预期 EER：** 12-14%（在 v6 基础上略有改善）

### 5.2 中期方案（修复 VoxCeleb2 后执行）

```yaml
# train_v8.yaml — VoxCeleb1+2，大数据量
training:
  epochs: 60
  batch_size: 64           # RTX 5070 Ti 16GB 安全值
  lr: 0.001
  optimizer: AdamW
  weight_decay: 0.01
  lr_schedule: cosine
  warmup_epochs: 5
  grad_clip: 1.0
  amp: false               # 大数据量下更需稳定
  early_stopping_patience: 15
  label_smoothing: 0.1

model:
  variant: "34"
  n_mels: 80
  embedding_dim: 192
  scale: 4
  se_reduction: 8
  pool_attention_dim: 128
  dropout: 0.2             # 数据量大，可以降低 dropout

loss:
  type: aam_softmax
  margin: 0.2
  scale: 32.0
  label_smoothing: 0.1

data:
  features_dirs:
    - data/voxceleb/features          # VoxCeleb1
    - F:/voxceleb2/dev/features       # VoxCeleb2 (修复后)
  fixed_length: 200
  num_workers: 0           # 避免 OOM，大数据库用 memmap
  memmap_dir: "E:/voxceleb_consolidated"
  augment: true
  augment_config:
    spec_augment:
      time_mask: 20
      freq_mask: 30
      num_masks: 4
    random_gain:
      gain_range: 5.0

checkpoint:
  save_dir: checkpoints_v8
  save_best: true
  save_interval: 5
  resume: null             # 从头训练
```

**预期 EER：** 6-9%（数据量提升是最大的变量）

### 5.3 长期方案（追 SOTA）

需要追加的技术栈：

1. **预提取 3 速度版本的 FBank**（0.9x/1.0x/1.1x），等效 3 倍数据
2. **加入 MUSAN 加噪 + RIR 加混响**（需下载 ~10GB 数据集）
3. **CMVN 归一化**（per-utterance 均值方差归一化）
4. **AS-Norm 评分归一化**（后处理，提升 EER 1-2%）
5. **大模型**（ERes2NetV2-50 或 101，配合更大 batch）
6. **多模型融合**（不同配置训练 3-4 个模型，embedding 拼接）

**预期 EER：** 4-6%

---

## 附：训练决策树

```
当前最佳：v6, EER=0.1478

下一步行动优先级：
│
├─ 1. 修复 VoxCeleb2 特征提取（磁盘空间问题）
│   └─ 预期：EER 降至 6-9%
│
├─ 2. 用完整 VoxCeleb1+2 重新训练 (v8 方案)
│   └─ 预期：EER 降至 6-9%
│
├─ 3. 加入 MUSAN 加噪 + RIR 加混响
│   └─ 预期：EER 再降 1-2%
│
├─ 4. 预提取多速度 FBank 特征
│   └─ 预期：EER 再降 1-2%
│
└─ 5. AS-Norm 后处理 + 多模型融合
    └─ 预期：EER 降至 4-6%
```

---

> **文档路径：** `C:\Users\Administrator\vsprint\docs\TRAINING_STRATEGY.md`
> **基于数据：** v1-v6 训练日志、配置文件、checkpoint 记录
