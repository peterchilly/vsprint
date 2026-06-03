# ERes2NetV2 Backbone 实现

> **所属阶段：** 阶段三：模型构建
> **阶段目标：** 基于 ERes2NetV2 架构实现完整的模型代码，包括 Backbone、分类头及必要的辅助模块。
> **阶段交付物：**
- ERes2NetV2 模型代码完整实现
- 多种变体配置
- 预训练权重加载功能
- 模型验证通过

---

## 目标
实现 ERes2NetV2 的核心 Backbone 架构。

## 任务清单

### 1. 基础组件
- [ ] `ConvBNReLU`（卷积+BN+ReLU）
- [ ] 1x1 卷积层
- [ ] Batch Normalization

### 2. ERes2NetV2 Block
- [ ] 分组特征提取（通道分组处理）
- [ ] 特征重用机制（y_i = conv(x_i + y_{i-1})）
- [ ] 通道交互模块（GAP -> FC -> 通道权重）
- [ ] 选择性残差融合（alpha * F(x) + beta * x）

### 3. 网络结构
- [ ] Stem 层、Stage 1-4、全局平均池化 + 分类头

## 验收标准
- [ ] Backbone 可独立运行
- [ ] 前向传播无报错
- [ ] 参数量符合预期
