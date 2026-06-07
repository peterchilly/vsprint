# ERes2NetV2 Backbone 实现（说话人识别版）

> **所属阶段：** 阶段三：模型构建
> **阶段目标：** 基于 ERes2NetV2 架构实现完整的声纹识别模型，包括 Backbone、统计池化层和分类头。
> **阶段交付物：**
- ERes2NetV2 模型代码完整实现
- 多种变体配置
- 预训练权重加载功能
- 模型验证通过

---

## 目标
实现 ERes2NetV2 的核心 Backbone 架构，适配说话人识别任务。

输入：FBank 特征 (batch, n_mels, time_frames)，如 (batch, 80, 300)
输出：说话人 embedding (batch, embedding_dim)，如 (batch, 192)

## 任务清单

### 1. 基础组件
- [ ] `ConvBNReLU`（2D 卷积 + BN + ReLU，用于 FBank 特征处理）
- [ ] `SELayer`（Squeeze-and-Excitation 通道注意力）
- [ ] `Batch Normalization`

### 2. ERes2NetV2 Block（说话人识别版）
- [ ] **分组特征提取**：将输入通道按 scale 分组，逐组 2D 卷积
- [ ] **特征重用机制**：后续分组复用前面分组输出 `y_i = conv(x_i + y_{i-1})`
- [ ] **通道交互模块**：SE-Block 实现通道注意力重校准
- [ ] **选择性残差融合**：可学习权重 `output = α·F(x) + β·x`

### 3. 网络结构
- [ ] **Stem**：初始 2D 卷积层，将 FBank (80×T) 映射到隐藏通道
- [ ] **Stage 1-4**：逐步降采样，通道数递增，每个 Stage 堆叠 ERes2NetV2 Blocks
- [ ] **Attentive Statistics Pooling**：注意力加权均值和标准差池化
- [ ] **Embedding 层**：FC 层输出说话人 embedding (192 维)
- [ ] **分类头**：Linear 层输出说话人 logits (num_speakers 类)

## 验收标准
- [ ] Backbone 可独立运作，前向传播无报错
- [ ] 输入 (batch=2, n_mels=80, time=300) → 输出 embedding (batch=2, dim=192)
- [ ] 参数量符合预期（ERes2NetV2-50 约 ~10M 参数）
