# DataLoader 构建

> **所属阶段：** 阶段二：数据准备
> **阶段目标：** 构建高质量、规范化的训练/验证/测试音频数据集，用于说话人识别任务。
> **阶段交付物：**
- 划分好的训练/验证/测试音频数据集
- 数据增强策略确定
- DataLoader 实现完成
- 数据质量检查报告

---

## 目标
实现自定义音频 Dataset 类和 DataLoader，用于说话人识别训练。

## 任务清单
- [ ] 继承 `torch.utils.data.Dataset`，实现 `__len__` 和 `__getitem__`
  - `__getitem__` 返回：FBank 特征 tensor + 说话人 label（整数 ID）
- [ ] 实现说话人标签编码（string → int 映射）
- [ ] 配置 DataLoader：
  - `batch_size`：根据 GPU 显存调整（如 64/128）
  - `num_workers`：根据 CPU 核心数设置（建议 4–8）
  - `pin_memory=True` 加速 GPU 数据传输
  - `shuffle=True`（仅训练集）
  - `drop_last=True`（训练集，确保 batch 大小一致）
- [ ] 实现采样策略：
  - 每个 batch 包含 N 个说话人，每个说话人 K 条语音（N-way K-shot）
  - 或使用普通随机采样 + CrossEntropyLoss
- [ ] 测试数据加载速度和内存占用
- [ ] 验证 FBank 特征维度正确、label 范围正确

## 验收标准
- [ ] SpeakerDataset 类实现完成（位于 `src/datasets/`）
- [ ] DataLoader 配置合理，能正确加载训练和验证数据
- [ ] 数据加载性能满足训练需求（不成为训练瓶颈）
- [ ] 通过单元测试：验证特征维度、label 范围、batch 组成
