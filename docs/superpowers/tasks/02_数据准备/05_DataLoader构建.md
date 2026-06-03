# DataLoader 构建

> **所属阶段：** 阶段二：数据准备
> **阶段目标：** 构建高质量、规范化的训练/验证/测试数据集，确保模型训练效果。
> **阶段交付物：**
- 划分好的训练/验证/测试数据集
- 数据增强策略确定
- DataLoader 实现完成
- 数据质量检查报告

---

## 目标
实现自定义 Dataset 类和 DataLoader。

## 任务清单
- [ ] 继承 `torch.utils.data.Dataset`，实现 `__len__` 和 `__getitem__`
- [ ] 配置 batch_size、num_workers、pin_memory、shuffle
- [ ] 测试加载速度和内存占用

## 验收标准
- [ ] Dataset 类实现完成
- [ ] DataLoader 配置合理
- [ ] 数据加载性能满足训练需求
