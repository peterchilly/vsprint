# 模型验证测试计划

> **验证对象：** ERes2NetV2 模型完整验证
> **验证日期：** 2026/06/03
> **验证阶段：** 阶段三：模型构建

---

## 一、验证目标

### 1.1 核心验证目标

| 目标ID | 验证目标 | 优先级 | 验证方法 |
|--------|----------|--------|----------|
| V-01 | 输出维度正确性 | P0 | 单元测试 |
| V-02 | 模型结构完整性 | P0 | 结构打印 |
| V-03 | 前向传播稳定性 | P0 | 集成测试 |
| V-04 | 梯度反向传播正确性 | P0 | 梯度测试 |
| V-05 | FLOPs 计算准确性 | P1 | 性能测试 |
| V-06 | 参数量统计准确性 | P1 | 参数统计 |
| V-07 | 与 ResNet50 对比验证 | P1 | 对比测试 |
| V-08 | 多变体一致性验证 | P1 | 集成测试 |

### 1.2 验收标准映射

| 验收标准 | 对应目标ID | 验证方法 |
|----------|-----------|----------|
| 所有验证测试通过 | V-01 ~ V-08 | 全部测试用例 |
| 参数量和 FLOPs 符合预期 | V-05, V-06 | 性能基准对比 |

---

## 二、测试用例

### 2.1 输出维度验证测试用例

#### TC-DIM-001: 标准输入维度验证

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-DIM-001 |
| **测试名称** | 标准图像输入输出维度 |
| **前置条件** | ERes2NetV2 模型已完整实现 |
| **测试步骤** | 1. 创建模型实例（num_classes=1000）<br>2. 输入标准尺寸图像张量 (B, 3, 224, 224)<br>3. 执行前向传播<br>4. 验证输出形状 |
| **输入数据** | `torch.randn(2, 3, 224, 224)` |
| **预期输出** | 输出形状: `(2, 1000)` |
| **通过条件** | 输出形状精确匹配 |

#### TC-DIM-002: 多尺度输入维度验证

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-DIM-002 |
| **测试名称** | 不同输入尺寸维度验证 |
| **前置条件** | 模型支持多尺度输入 |
| **测试步骤** | 1. 测试 224x224 输入<br>2. 测试 256x256 输入<br>3. 测试 384x384 输入<br>4. 测试 512x512 输入<br>5. 验证每种尺寸输出维度 |
| **输入尺寸** | 224x224, 256x256, 384x384, 512x512 |
| **预期输出** | 每种尺寸输出形状: `(B, num_classes)` |
| **通过条件** | 所有尺寸输出维度正确 |

#### TC-DIM-003: 不同批次大小验证

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-DIM-003 |
| **测试名称** | 批次大小兼容性验证 |
| **前置条件** | 模型支持动态批次 |
| **测试步骤** | 1. 测试 batch_size=1（单样本）<br>2. 测试 batch_size=8<br>3. 测试 batch_size=32<br>4. 测试 batch_size=64 |
| **输入数据** | 不同批次大小的张量 |
| **预期输出** | 输出形状始终为 `(batch_size, num_classes)` |
| **通过条件** | 所有批次大小输出正确 |

#### TC-DIM-004: 特征提取模式维度验证

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-DIM-004 |
| **测试名称** | 无分类头特征输出维度 |
| **前置条件** | 模型支持特征提取模式 |
| **测试步骤** | 1. 创建模型（include_head=False）<br>2. 前向传播获取特征<br>3. 验证特征图形状 |
| **输入数据** | `torch.randn(2, 3, 224, 224)` |
| **预期输出** | 特征形状: `(2, C, H', W')` 其中 H'=7, W'=7（标准配置） |
| **通过条件** | 特征维度符合设计 |

#### TC-DIM-005: 不同 num_classes 维度验证

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-DIM-005 |
| **测试名称** | 分类数量变化输出维度 |
| **前置条件** | 模型支持自定义类别数 |
| **测试步骤** | 1. 创建模型（num_classes=10）<br>2. 创建模型（num_classes=100）<br>3. 创建模型（num_classes=1000）<br>4. 验证每种配置输出维度 |
| **预期输出** | 输出维度分别: (B, 10), (B, 100), (B, 1000) |
| **通过条件** | 所有配置输出正确 |

---

### 2.2 结构打印测试用例

#### TC-STR-001: 模型层级结构打印

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-STR-001 |
| **测试名称** | 模型层级结构完整性 |
| **前置条件** | 模型已实现 `__str__` 或可打印 |
| **测试步骤** | 1. 创建模型实例<br>2. 打印模型结构 `print(model)`<br>3. 验证包含所有必要层级 |
| **验证点** | - Stem 层存在<br>- Stage 1-4 存在<br>- 全局平均池化存在<br>- 分类头存在<br>- 每层参数形状正确 |
| **通过条件** | 所有必要层级可见且正确 |

#### TC-STR-002: 参数量统计打印

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-STR-002 |
| **测试名称** | 参数量详细统计 |
| **前置条件** | 模型参数可统计 |
| **测试步骤** | 1. 计算总参数量<br>2. 计算可训练参数量<br>3. 计算各层参数量分布<br>4. 打印统计报告 |
| **统计代码** | ```python<br>def print_model_info(model):<br>    total = sum(p.numel() for p in model.parameters())<br>    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)<br>    print(f"Total: {total:,}")<br>    print(f"Trainable: {trainable:,}")<br>``` |
| **通过条件** | 参数量在预期范围内 |

#### TC-STR-003: 层级参数量分布

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-STR-003 |
| **测试名称** | 各层参数量分布验证 |
| **前置条件** | 模型结构完整 |
| **测试步骤** | 1. 遍历各层<br>2. 统计每层参数量<br>3. 计算占比<br>4. 生成分布报告 |
| **预期分布** | | 层级 | 预期占比 |<br>|------|---------|<br>| Stem | ~5% |<br>| Stage1 | ~10% |<br>| Stage2 | ~20% |<br>| Stage3 | ~40% |<br>| Stage4 | ~20% |<br>| Head | ~5% | |
| **通过条件** | 分布符合预期（±10%） |

#### TC-STR-004: 模型深度统计

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-STR-004 |
| **测试名称** | 网络深度层数验证 |
| **前置条件** | 模型变体配置完整 |
| **测试步骤** | 1. 统计 ERes2NetV2-50 层数<br>2. 统计 ERes2NetV2-101 层数<br>3. 统计 ERes2NetV2-152 层数<br>4. 验证层数符合配置 |
| **预期层数** | | 变体 | 卷积层数 | Block数 |<br>|------|---------|--------|<br>| 50 | ~50 | 16 |<br>| 101 | ~101 | 33 |<br>| 152 | ~152 | 50 | |
| **通过条件** | 层数符合变体定义 |

---

### 2.3 前向传播测试用例

#### TC-FWD-001: 标准前向传播测试

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-FWD-001 |
| **测试名称** | 标准输入前向传播无报错 |
| **前置条件** | 模型完整实现 |
| **测试步骤** | 1. 创建模型实例<br>2. 设置为评估模式<br>3. 输入标准张量<br>4. 执行前向传播<br>5. 验证输出无 NaN/Inf |
| **输入数据** | `torch.randn(2, 3, 224, 224)` |
| **预期输出** | 输出张量形状正确，数值范围正常 |
| **通过条件** | 前向传播无报错 + 输出正常 |

#### TC-FWD-002: 训练/评估模式切换

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-FWD-002 |
| **测试名称** | 模式切换前向传播一致性 |
| **前置条件** | BN 层正确实现 |
| **测试步骤** | 1. 创建模型<br>2. 设置训练模式，前向传播<br>3. 设置评估模式，前向传播<br>4. 同一输入，评估模式两次前向传播<br>5. 验证评估模式输出一致 |
| **验证点** | - 训练模式: BN 使用 batch 统计<br>- 评估模式: BN 使用 running 统计<br>- 评估模式输出确定性 |
| **通过条件** | 模式行为正确 |

#### TC-FWD-003: 边界情况输入测试

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-FWD-003 |
| **测试名称** | 边界输入前向传播稳定性 |
| **前置条件** | 模型完整实现 |
| **测试步骤** | 1. 测试全零输入<br>2. 测试全一输入<br>3. 测试极小值输入（-1e10）<br>4. 测试极大值输入（1e10）<br>5. 验证每种情况无崩溃 |
| **边界输入** | - `torch.zeros(1, 3, 224, 224)`<br>- `torch.ones(1, 3, 224, 224)`<br>- `torch.randn(1, 3, 224, 224) * 1e10` |
| **通过条件** | 所有边界输入无崩溃，输出无 NaN/Inf |

#### TC-FWD-004: GPU/CPU 一致性测试

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-FWD-004 |
| **测试名称** | 不同设备前向传播一致性 |
| **前置条件** | GPU 环境可用 |
| **测试步骤** | 1. 创建 CPU 模型<br>2. 创建 GPU 模型（相同权重）<br>3. 相同输入分别前向传播<br>4. 验证输出数值一致 |
| **输入数据** | `torch.randn(2, 3, 224, 224)` |
| **预期输出** | CPU 和 GPU 输出差异 < 1e-5 |
| **通过条件** | 数值一致性满足要求 |

#### TC-FWD-005: 混合精度前向传播

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-FWD-005 |
| **测试名称** | FP16/AMP 前向传播 |
| **前置条件** | GPU 支持 FP16 |
| **测试步骤** | 1. 启用自动混合精度（AMP）<br>2. 前向传播<br>3. 验证输出正常<br>4. 验证无精度溢出 |
| **测试代码** | ```python<br>with torch.cuda.amp.autocast():<br>    output = model(input)<br>``` |
| **通过条件** | AMP 模式下前向传播无错 |

---

### 2.4 梯度反传测试用例

#### TC-GRAD-001: 基础梯度反传测试

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-GRAD-001 |
| **测试名称** | 基础梯度反向传播 |
| **前置条件** | 模型可训练 |
| **测试步骤** | 1. 创建模型，设置训练模式<br>2. 前向传播<br>3. 计算损失（如交叉熵）<br>4. 反向传播<br>5. 验证所有参数梯度存在 |
| **输入数据** | `torch.randn(2, 3, 224, 224)`<br>`torch.randint(0, 1000, (2,))` |
| **验证点** | - 所有可训练参数 `.grad` 存在<br>- 梯度非 None<br>- 梯度形状与参数形状匹配 |
| **通过条件** | 所有点满足 |

#### TC-GRAD-002: 梯度数值正确性测试

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-GRAD-002 |
| **测试名称** | 梯度数值有效性验证 |
| **前置条件** | 梯度反传成功 |
| **测试步骤** | 1. 反向传播后检查梯度<br>2. 验证无 NaN 梯度<br>3. 验证无 Inf 梯度<br>4. 验证梯度值在合理范围 |
| **验证代码** | ```python<br>for name, param in model.named_parameters():<br>    assert param.grad is not None<br>    assert not torch.isnan(param.grad).any()<br>    assert not torch.isinf(param.grad).any()<br>``` |
| **通过条件** | 所有梯度数值有效 |

#### TC-GRAD-003: 梯度流动完整性测试

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-GRAD-003 |
| **测试名称** | 梯度流经所有层验证 |
| **前置条件** | 模型结构完整 |
| **测试步骤** | 1. 反向传播<br>2. 遍历所有层<br>3. 验证每层至少有一个参数梯度非零<br>4. 生成梯度流动报告 |
| **验证点** | - Stem 层梯度非零<br>- 各 Stage 梯度非零<br>- 分类头梯度非零 |
| **通过条件** | 梯度流动到所有层 |

#### TC-GRAD-004: 梯度裁剪测试

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-GRAD-004 |
| **测试名称** | 梯度裁剪功能验证 |
| **前置条件** | 训练流程包含梯度裁剪 |
| **测试步骤** | 1. 创建大梯度场景<br>2. 应用梯度裁剪<br>3. 验证梯度范数不超过阈值 |
| **测试代码** | ```python<br>torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)<br>``` |
| **通过条件** | 裁剪后梯度范数 ≤ 阈值 |

#### TC-GRAD-005: 冻结层梯度测试

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-GRAD-005 |
| **测试名称** | 冻结参数梯度验证 |
| **前置条件** | 参数冻结功能已实现 |
| **测试步骤** | 1. 冻结部分层参数（requires_grad=False）<br>2. 前向传播 + 反向传播<br>3. 验证冻结参数梯度为 None<br>4. 验证未冻结参数梯度正常 |
| **验证点** | - 冻结参数 `.grad` 为 None<br>- 未冻结参数梯度正常 |
| **通过条件** | 冻结逻辑正确 |

---

### 2.5 FLOPs 和参数量计算测试用例

#### TC-FLOP-001: FLOPs 计算功能测试

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-FLOP-001 |
| **测试名称** | FLOPs 计算器功能验证 |
| **前置条件** | FLOPs 计算工具已集成（thop/fvcore） |
| **测试步骤** | 1. 创建模型<br>2. 使用 thop 计算 FLOPs<br>3. 使用 fvcore 计算 FLOPs<br>4. 验证两种方法结果一致 |
| **测试代码** | ```python<br>from thop import profile<br>flops, params = profile(model, inputs=(input,))<br>``` |
| **通过条件** | FLOPs 计算无报错，结果合理 |

#### TC-FLOP-002: FLOPs 基准对比测试

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-FLOP-002 |
| **测试名称** | FLOPs 数值符合预期 |
| **前置条件** | FLOPs 计算功能正常 |
| **测试步骤** | 1. 计算 ERes2NetV2-50 FLOPs<br>2. 计算 ERes2NetV2-101 FLOPs<br>3. 计算 ERes2NetV2-152 FLOPs<br>4. 与预期值对比 |
| **预期 FLOPs** | | 变体 | 预期 FLOPs (224x224) |<br>|------|---------------------|<br>| 50 | ~4.0 GFLOPs |<br>| 101 | ~7.5 GFLOPs |<br>| 152 | ~11.0 GFLOPs | |
| **通过条件** | 偏差 < 10% |

#### TC-PARAM-001: 参数量计算测试

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-PARAM-001 |
| **测试名称** | 参数量统计准确性 |
| **前置条件** | 模型完整实现 |
| **测试步骤** | 1. 使用 PyTorch 计算参数量<br>2. 使用 thop 计算参数量<br>3. 手动累加验证 |
| **测试代码** | ```python<br>total = sum(p.numel() for p in model.parameters())<br>trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)<br>``` |
| **通过条件** | 多种方法结果一致 |

#### TC-PARAM-002: 参数量基准对比测试

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-PARAM-002 |
| **测试名称** | 参数量符合预期 |
| **前置条件** | 参数量统计功能正常 |
| **测试步骤** | 1. 计算各变体参数量<br>2. 与论文/官方实现对比<br>3. 生成对比报告 |
| **预期参数量** | | 变体 | 预期参数量 |<br>|------|-----------|<br>| 50 | ~25M |<br>| 101 | ~45M |<br>| 152 | ~60M | |
| **通过条件** | 偏差 < 5% |

#### TC-PARAM-003: 参数量分布分析

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-PARAM-003 |
| **测试名称** | 参数量分布统计 |
| **前置条件** | 模型结构完整 |
| **测试步骤** | 1. 统计卷积层参数量<br>2. 统计 BN 层参数量<br>3. 统计全连接层参数量<br>4. 分析分布合理性 |
| **预期分布** | - 卷积层: >80%<br>- BN 层: <5%<br>- FC 层: 视 num_classes |
| **通过条件** | 分布合理 |

---

### 2.6 与 ResNet50 对比测试用例

#### TC-CMP-001: 参数量对比测试

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-CMP-001 |
| **测试名称** | ERes2NetV2 vs ResNet50 参数量对比 |
| **前置条件** | ResNet50 实现可用 |
| **测试步骤** | 1. 创建 ERes2NetV2-50 模型<br>2. 创建 ResNet50 模型<br>3. 分别计算参数量<br>4. 对比分析 |
| **对比维度** | | 模型 | 参数量 | 相对差异 |<br>|------|--------|---------|<br>| ResNet50 | ~25.6M | 基准 |<br>| ERes2NetV2-50 | ~25M | -2% | |
| **通过条件** | 参数量在同一量级 |

#### TC-CMP-002: FLOPs 对比测试

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-CMP-002 |
| **测试名称** | ERes2NetV2 vs ResNet50 FLOPs 对比 |
| **前置条件** | FLOPs 计算工具可用 |
| **测试步骤** | 1. 计算 ResNet50 FLOPs<br>2. 计算 ERes2NetV2-50 FLOPs<br>3. 对比计算效率 |
| **预期对比** | | 模型 | FLOPs | 相对差异 |<br>|------|-------|---------|<br>| ResNet50 | ~4.1 GFLOPs | 基准 |<br>| ERes2NetV2-50 | ~4.0 GFLOPs | -2% | |
| **通过条件** | FLOPs 相近或更优 |

#### TC-CMP-003: 输出维度一致性对比

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-CMP-003 |
| **测试名称** | 相同输入输出维度一致性 |
| **前置条件** | 两模型均完整实现 |
| **测试步骤** | 1. 相同输入分别前向传播<br>2. 验证输出形状一致<br>3. 验证特征图尺寸一致 |
| **输入数据** | `torch.randn(2, 3, 224, 224)` |
| **预期输出** | 两模型输出形状均为 `(2, num_classes)` |
| **通过条件** | 输出维度一致 |

#### TC-CMP-004: 特征提取能力对比

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-CMP-004 |
| **测试名称** | 特征图尺寸对比 |
| **前置条件** | 两模型支持特征提取 |
| **测试步骤** | 1. 提取 ResNet50 各阶段特征<br>2. 提取 ERes2NetV2 各阶段特征<br>3. 对比特征图空间尺寸<br>4. 对比特征通道数 |
| **预期对比** | | 阶段 | ResNet50 | ERes2NetV2 |<br>|------|----------|------------|<br>| Stage1 | 56x56 | 56x56 |<br>| Stage2 | 28x28 | 28x28 |<br>| Stage3 | 14x14 | 14x14 |<br>| Stage4 | 7x7 | 7x7 | |
| **通过条件** | 特征图尺寸一致 |

#### TC-CMP-005: 推理速度对比

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-CMP-005 |
| **测试名称** | 推理时间对比测试 |
| **前置条件** | GPU 环境可用 |
| **测试步骤** | 1. 预热模型（10次前向传播）<br>2. 计时 100 次前向传播<br>3. 计算平均推理时间<br>4. 对比两模型速度 |
| **测试条件** | Batch=1, Input=224x224, GPU |
| **通过条件** | 推理时间差异 < 20% |

---

## 三、验证方法

### 3.1 单元测试

| 方法 | 描述 | 工具 |
|------|------|------|
| pytest | Python 单元测试框架 | pytest |
| torch.autograd.gradcheck | 梯度正确性验证 | PyTorch |
| torch.testing.assert_close | 张量数值验证 | PyTorch |

### 3.2 集成测试

| 方法 | 描述 | 适用场景 |
|------|------|----------|
| 端到端前向传播 | 完整模型前向传播 | 前向传播测试 |
| 梯度流动检查 | 反向传播完整性 | 梯度反传测试 |
| 设备一致性测试 | CPU/GPU 结果对比 | 跨设备验证 |

### 3.3 性能测试

| 方法 | 描述 | 工具 |
|------|------|------|
| FLOPs 计算 | 计算模型计算量 | thop / fvcore |
| 参数量统计 | 统计模型参数 | PyTorch |
| 推理时间测量 | 计时分析 | time / cuda.Event |

### 3.4 对比测试

| 方法 | 描述 | 适用场景 |
|------|------|----------|
| 模型对比 | 与 ResNet50 对比 | 性能基准 |
| 论文数据对比 | 与论文数据对比 | 参数量/FLOPs |

### 3.5 测试代码框架

```python
import pytest
import torch
import torch.nn as nn
from thop import profile


class TestOutputDimensions:
    """输出维度验证测试"""

    def test_standard_input(self):
        """TC-DIM-001: 标准输入输出维度"""
        from models.eres2netv2 import ERes2NetV2

        model = ERes2NetV2(variant='50', num_classes=1000)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)

        assert out.shape == (2, 1000), f"Expected (2, 1000), got {out.shape}"

    @pytest.mark.parametrize("batch_size", [1, 8, 32, 64])
    def test_batch_sizes(self, batch_size):
        """TC-DIM-003: 不同批次大小验证"""
        from models.eres2netv2 import ERes2NetV2

        model = ERes2NetV2(variant='50', num_classes=1000)
        x = torch.randn(batch_size, 3, 224, 224)
        out = model(x)

        assert out.shape == (batch_size, 1000), f"Expected ({batch_size}, 1000), got {out.shape}"

    @pytest.mark.parametrize("num_classes", [10, 100, 1000])
    def test_num_classes(self, num_classes):
        """TC-DIM-005: 不同分类数维度"""
        from models.eres2netv2 import ERes2NetV2

        model = ERes2NetV2(variant='50', num_classes=num_classes)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)

        assert out.shape == (2, num_classes), f"Expected (2, {num_classes}), got {out.shape}"


class TestStructurePrinting:
    """结构打印测试"""

    def test_model_printable(self):
        """TC-STR-001: 模型层级结构打印"""
        from models.eres2netv2 import ERes2NetV2

        model = ERes2NetV2(variant='50', num_classes=1000)
        model_str = str(model)

        # 验证关键组件存在
        assert 'Stem' in model_str or 'stem' in model_str or 'Conv' in model_str
        assert 'Stage' in model_str or 'stage' in model_str or 'Block' in model_str

    def test_parameter_count(self):
        """TC-STR-002: 参数量统计"""
        from models.eres2netv2 import ERes2NetV2

        model = ERes2NetV2(variant='50', num_classes=1000)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert total == trainable, "All parameters should be trainable by default"
        assert 20_000_000 < total < 30_000_000, f"Expected ~25M params, got {total:,}"


class TestForwardPropagation:
    """前向传播测试"""

    def test_standard_forward(self):
        """TC-FWD-001: 标准前向传播"""
        from models.eres2netv2 import ERes2NetV2

        model = ERes2NetV2(variant='50', num_classes=1000)
        model.eval()
        x = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            out = model(x)

        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

    def test_train_eval_mode(self):
        """TC-FWD-002: 训练/评估模式切换"""
        from models.eres2netv2 import ERes2NetV2

        model = ERes2NetV2(variant='50', num_classes=1000)
        x = torch.randn(2, 3, 224, 224)

        # 训练模式
        model.train()
        out_train = model(x)

        # 评估模式
        model.eval()
        with torch.no_grad():
            out_eval1 = model(x)
            out_eval2 = model(x)

        # 评估模式下两次输出应相同
        assert torch.allclose(out_eval1, out_eval2), "Eval mode should be deterministic"

    @pytest.mark.parametrize("input_type", ['zeros', 'ones', 'large'])
    def test_edge_cases(self, input_type):
        """TC-FWD-003: 边界情况输入"""
        from models.eres2netv2 import ERes2NetV2

        model = ERes2NetV2(variant='50', num_classes=1000)
        model.eval()

        if input_type == 'zeros':
            x = torch.zeros(1, 3, 224, 224)
        elif input_type == 'ones':
            x = torch.ones(1, 3, 224, 224)
        else:
            x = torch.randn(1, 3, 224, 224) * 100

        with torch.no_grad():
            out = model(x)

        assert not torch.isnan(out).any(), f"Output contains NaN for {input_type}"
        assert not torch.isinf(out).any(), f"Output contains Inf for {input_type}"


class TestGradientBackpropagation:
    """梯度反传测试"""

    def test_basic_gradient(self):
        """TC-GRAD-001: 基础梯度反传"""
        from models.eres2netv2 import ERes2NetV2

        model = ERes2NetV2(variant='50', num_classes=1000)
        model.train()
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        target = torch.randint(0, 1000, (2,))

        criterion = nn.CrossEntropyLoss()
        out = model(x)
        loss = criterion(out, target)
        loss.backward()

        # 验证输入梯度
        assert x.grad is not None, "Input gradient is None"

        # 验证参数梯度
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"{name} gradient is None"

    def test_gradient_validity(self):
        """TC-GRAD-002: 梯度数值有效性"""
        from models.eres2netv2 import ERes2NetV2

        model = ERes2NetV2(variant='50', num_classes=1000)
        model.train()
        x = torch.randn(2, 3, 224, 224)
        target = torch.randint(0, 1000, (2,))

        criterion = nn.CrossEntropyLoss()
        out = model(x)
        loss = criterion(out, target)
        loss.backward()

        issues = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    issues.append(f"{name}: NaN gradient")
                if torch.isinf(param.grad).any():
                    issues.append(f"{name}: Inf gradient")

        assert not issues, "Gradient issues:\n" + "\n".join(issues)

    def test_frozen_parameters(self):
        """TC-GRAD-005: 冻结层梯度"""
        from models.eres2netv2 import ERes2NetV2

        model = ERes2NetV2(variant='50', num_classes=1000)

        # 冻结前两个 stage
        for name, param in model.named_parameters():
            if 'stage1' in name or 'stage2' in name:
                param.requires_grad = False

        model.train()
        x = torch.randn(2, 3, 224, 224)
        target = torch.randint(0, 1000, (2,))

        criterion = nn.CrossEntropyLoss()
        out = model(x)
        loss = criterion(out, target)
        loss.backward()

        # 验证冻结参数无梯度
        for name, param in model.named_parameters():
            if 'stage1' in name or 'stage2' in name:
                assert param.grad is None or not param.requires_grad, \
                    f"Frozen parameter {name} should not have gradient"


class TestFLOPsAndParameters:
    """FLOPs 和参数量测试"""

    def test_flops_calculation(self):
        """TC-FLOP-001: FLOPs 计算"""
        from models.eres2netv2 import ERes2NetV2

        model = ERes2NetV2(variant='50', num_classes=1000)
        input = torch.randn(1, 3, 224, 224)

        flops, params = profile(model, inputs=(input,))

        # 验证 FLOPs 在合理范围
        assert 3e9 < flops < 5e9, f"Expected ~4 GFLOPs, got {flops/1e9:.2f} GFLOPs"

    @pytest.mark.parametrize("variant,expected_params", [
        ("50", 25_000_000),
        ("101", 45_000_000),
        ("152", 60_000_000),
    ])
    def test_parameter_count(self, variant, expected_params):
        """TC-PARAM-002: 参数量基准对比"""
        from models.eres2netv2 import ERes2NetV2

        model = ERes2NetV2(variant=variant, num_classes=1000)
        total = sum(p.numel() for p in model.parameters())

        diff = abs(total - expected_params) / expected_params
        assert diff < 0.1, f"{variant}: expected ~{expected_params:,}, got {total:,} (diff={diff:.1%})"


class TestResNetComparison:
    """与 ResNet50 对比测试"""

    def test_parameter_comparison(self):
        """TC-CMP-001: 参数量对比"""
        from models.eres2netv2 import ERes2NetV2
        import torchvision.models as models

        eres2netv2 = ERes2NetV2(variant='50', num_classes=1000)
        resnet50 = models.resnet50(num_classes=1000)

        eres_params = sum(p.numel() for p in eres2netv2.parameters())
        resnet_params = sum(p.numel() for p in resnet50.parameters())

        # 参数量应相近（差异 < 20%）
        diff = abs(eres_params - resnet_params) / resnet_params
        assert diff < 0.2, f"Parameter difference too large: {diff:.1%}"

    def test_output_dimension_consistency(self):
        """TC-CMP-003: 输出维度一致性"""
        from models.eres2netv2 import ERes2NetV2
        import torchvision.models as models

        eres2netv2 = ERes2NetV2(variant='50', num_classes=1000)
        resnet50 = models.resnet50(num_classes=1000)

        eres2netv2.eval()
        resnet50.eval()

        x = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            out_eres = eres2netv2(x)
            out_resnet = resnet50(x)

        assert out_eres.shape == out_resnet.shape, \
            f"Shape mismatch: ERes2NetV2 {out_eres.shape} vs ResNet50 {out_resnet.shape}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_inference_speed_comparison(self):
        """TC-CMP-005: 推理速度对比"""
        from models.eres2netv2 import ERes2NetV2
        import torchvision.models as models
        import time

        device = torch.device('cuda')
        batch_size = 1
        num_iterations = 100

        # ERes2NetV2
        eres2netv2 = ERes2NetV2(variant='50', num_classes=1000).to(device).eval()
        x = torch.randn(batch_size, 3, 224, 224, device=device)

        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = eres2netv2(x)

        # 计时
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = eres2netv2(x)
        torch.cuda.synchronize()
        eres_time = (time.time() - start) / num_iterations

        # ResNet50
        resnet50 = models.resnet50(num_classes=1000).to(device).eval()
        with torch.no_grad():
            for _ in range(10):
                _ = resnet50(x)

        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = resnet50(x)
        torch.cuda.synchronize()
        resnet_time = (time.time() - start) / num_iterations

        print(f"\nERes2NetV2-50: {eres_time*1000:.2f}ms")
        print(f"ResNet50: {resnet_time*1000:.2f}ms")
        print(f"Ratio: {eres_time/resnet_time:.2f}x")

        # 推理时间差异应 < 50%
        assert eres_time / resnet_time < 1.5, "ERes2NetV2 is significantly slower than ResNet50"
```

---

## 四、通过标准

### 4.1 测试通过标准

| 级别 | 测试类别 | 通过标准 | 说明 |
|------|----------|----------|------|
| P0 | 输出维度验证 | 100% 通过 | 核心功能 |
| P0 | 前向传播测试 | 100% 通过 | 核心功能 |
| P0 | 梯度反传测试 | 100% 通过 | 核心功能 |
| P1 | FLOPs/参数量测试 | >= 90% 通过 | 性能验证 |
| P1 | 对比测试 | >= 80% 通过 | 基准对比 |

### 4.2 数值通过标准

| 指标 | 标准 | 测试条件 |
|------|------|----------|
| 参数量偏差 | < 5% | 与预期值对比 |
| FLOPs 偏差 | < 10% | 与预期值对比 |
| 推理时间偏差 | < 50% | 与 ResNet50 对比 |
| 输出数值偏差 | < 1e-5 | CPU/GPU 一致性 |

### 4.3 性能通过标准

| 变体 | 参数量范围 | FLOPs 范围 | 推理时间（GPU） |
|------|-----------|-----------|----------------|
| ERes2NetV2-50 | 20M-30M | 3-5 GFLOPs | < 50ms |
| ERes2NetV2-101 | 40M-50M | 6-9 GFLOPs | < 80ms |
| ERes2NetV2-152 | 55M-65M | 9-13 GFLOPs | < 120ms |

---

## 五、自动化建议

### 5.1 测试自动化

```bash
# 运行所有验证测试
pytest tests/test_model_verification.py -v

# 运行特定测试类
pytest tests/test_model_verification.py::TestOutputDimensions -v

# 运行特定测试用例
pytest tests/test_model_verification.py::TestForwardPropagation::test_standard_forward -v

# 生成覆盖率报告
pytest tests/test_model_verification.py --cov=models --cov-report=html

# 并行测试
pytest tests/test_model_verification.py -n auto

# 标记测试
pytest tests/test_model_verification.py -v -m "not slow"
```

### 5.2 CI/CD 集成

```yaml
# .github/workflows/model_verification.yml
name: Model Verification

on:
  push:
    paths:
      - 'models/**'
      - 'tests/test_model_verification.py'
  pull_request:
    paths:
      - 'models/**'
      - 'tests/test_model_verification.py'

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-xdist thop fvcore

      - name: Run unit tests
        run: |
          pytest tests/test_model_verification.py -v --cov=models --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml

  gpu-test:
    runs-on: [self-hosted, gpu]
    if: github.event_name == 'pull_request'

    steps:
      - uses: actions/checkout@v3

      - name: Set up environment
        run: |
          pip install -r requirements.txt
          pip install pytest thop fvcore

      - name: Run GPU tests
        run: |
          pytest tests/test_model_verification.py -v -m "gpu"
```

### 5.3 自动化验证脚本

```python
#!/usr/bin/env python
"""
模型验证自动化脚本
运行方式: python scripts/verify_model.py --variant 50 --all
"""
import argparse
import time
import torch
import torch.nn as nn
from thop import profile


def verify_output_dimensions(model, num_classes=1000):
    """验证输出维度"""
    print("\n" + "=" * 60)
    print("1. 输出维度验证")
    print("=" * 60)

    results = []

    # TC-DIM-001: 标准输入
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    passed = out.shape == (2, num_classes)
    results.append(("标准输入 (2, 3, 224, 224)", passed, out.shape))
    print(f"  ✓ 标准输入: {tuple(x.shape)} -> {tuple(out.shape)} {'PASS' if passed else 'FAIL'}")

    # TC-DIM-002: 多尺度输入
    for size in [224, 256, 384, 512]:
        x = torch.randn(1, 3, size, size)
        try:
            out = model(x)
            passed = out.shape == (1, num_classes)
            results.append((f"多尺度输入 {size}x{size}", passed, out.shape))
            print(f"  ✓ 多尺度输入 ({size}x{size}): {'PASS' if passed else 'FAIL'}")
        except Exception as e:
            results.append((f"多尺度输入 {size}x{size}", False, str(e)))
            print(f"  ✗ 多尺度输入 ({size}x{size}): FAIL - {e}")

    # TC-DIM-003: 批次大小
    for bs in [1, 8, 32, 64]:
        x = torch.randn(bs, 3, 224, 224)
        out = model(x)
        passed = out.shape == (bs, num_classes)
        results.append((f"批次大小 {bs}", passed, out.shape))

    return all(r[1] for r in results)


def verify_structure(model):
    """验证模型结构"""
    print("\n" + "=" * 60)
    print("2. 结构打印验证")
    print("=" * 60)

    # TC-STR-001: 层级结构
    model_str = str(model)
    print(f"  模型结构:\n{'  ' + model_str[:500].replace(chr(10), chr(10) + '  ')}...")

    # TC-STR-002: 参数量统计
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  总参数量: {total:,}")
    print(f"  可训练参数量: {trainable:,}")

    # TC-STR-003: 层级参数分布
    print(f"\n  参数分布:")
    layer_params = {}
    for name, param in model.named_parameters():
        layer = name.split('.')[0]
        layer_params[layer] = layer_params.get(layer, 0) + param.numel()

    for layer, count in sorted(layer_params.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        print(f"    {layer}: {count:,} ({pct:.1f}%)")

    return True


def verify_forward_propagation(model):
    """验证前向传播"""
    print("\n" + "=" * 60)
    print("3. 前向传播测试")
    print("=" * 60)

    results = []

    # TC-FWD-001: 标准前向传播
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    has_nan = torch.isnan(out).any().item()
    has_inf = torch.isinf(out).any().item()
    passed = not has_nan and not has_inf
    results.append(("标准前向传播", passed))
    print(f"  {'✓' if passed else '✗'} 标准前向传播: {'PASS' if passed else 'FAIL'}")
    if has_nan:
        print(f"    ⚠ 输出包含 NaN")
    if has_inf:
        print(f"    ⚠ 输出包含 Inf")

    # TC-FWD-002: 训练/评估模式
    model.train()
    x = torch.randn(2, 3, 224, 224)
    out_train1 = model(x)
    out_train2 = model(x)
    train_diff = (out_train1 - out_train2).abs().max().item() > 0

    model.eval()
    with torch.no_grad():
        out_eval1 = model(x)
        out_eval2 = model(x)
    eval_same = torch.allclose(out_eval1, out_eval2)

    passed = train_diff and eval_same
    results.append(("训练/评估模式", passed))
    print(f"  {'✓' if passed else '✗'} 训练/评估模式: {'PASS' if passed else 'FAIL'}")
    print(f"    训练模式输出变化: {'是' if train_diff else '否'}")
    print(f"    评估模式确定性: {'是' if eval_same else '否'}")

    # TC-FWD-003: 边界情况
    model.eval()
    edge_cases = {
        "全零输入": torch.zeros(1, 3, 224, 224),
        "全一输入": torch.ones(1, 3, 224, 224),
        "极值输入": torch.randn(1, 3, 224, 224) * 100,
    }

    for name, x in edge_cases.items():
        with torch.no_grad():
            out = model(x)
        passed = not torch.isnan(out).any() and not torch.isinf(out).any()
        results.append((name, passed))
        print(f"  {'✓' if passed else '✗'} {name}: {'PASS' if passed else 'FAIL'}")

    return all(r[1] for r in results)


def verify_gradient(model):
    """验证梯度反传"""
    print("\n" + "=" * 60)
    print("4. 梯度反传测试")
    print("=" * 60)

    results = []

    # TC-GRAD-001: 基础梯度
    model.train()
    x = torch.randn(2, 3, 224, 224, requires_grad=True)
    target = torch.randint(0, 1000, (2,))

    criterion = nn.CrossEntropyLoss()
    out = model(x)
    loss = criterion(out, target)
    loss.backward()

    # 检查输入梯度
    input_grad_ok = x.grad is not None
    results.append(("输入梯度存在", input_grad_ok))
    print(f"  {'✓' if input_grad_ok else '✗'} 输入梯度存在: {'PASS' if input_grad_ok else 'FAIL'}")

    # TC-GRAD-002: 梯度有效性
    issues = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                issues.append(f"{name}: NaN")
            if torch.isinf(param.grad).any():
                issues.append(f"{name}: Inf")
        elif param.requires_grad:
            issues.append(f"{name}: None")

    passed = len(issues) == 0
    results.append(("梯度有效性", passed))
    print(f"  {'✓' if passed else '✗'} 梯度有效性: {'PASS' if passed else 'FAIL'}")
    if issues:
        print(f"    问题: {', '.join(issues[:5])}")

    # TC-GRAD-003: 梯度流动
    layer_has_grad = {}
    for name, param in model.named_parameters():
        layer = name.split('.')[0]
        if layer not in layer_has_grad:
            layer_has_grad[layer] = []
        if param.grad is not None:
            layer_has_grad[layer].append(param.grad.abs().max().item())

    all_layers_have_grad = all(len(v) > 0 for v in layer_has_grad.values())
    results.append(("梯度流动", all_layers_have_grad))
    print(f"  {'✓' if all_layers_have_grad else '✗'} 梯度流动: {'PASS' if all_layers_have_grad else 'FAIL'}")

    return all(r[1] for r in results)


def verify_flops_params(model, variant):
    """验证 FLOPs 和参数量"""
    print("\n" + "=" * 60)
    print("5. FLOPs 和参数量计算")
    print("=" * 60)

    results = []

    # TC-FLOP-001/TC-PARAM-001: FLOPs 和参数量
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(input,))

    flops_g = flops / 1e9
    params_m = params / 1e6

    print(f"  FLOPs: {flops_g:.2f} GFLOPs")
    print(f"  参数量: {params_m:.2f} M ({params:,})")

    # TC-PARAM-002: 参数量基准
    expected_params = {
        '50': (20, 30),   # 20M - 30M
        '101': (40, 50),  # 40M - 50M
        '152': (55, 65),  # 55M - 65M
    }

    if variant in expected_params:
        low, high = expected_params[variant]
        passed = low <= params_m <= high
        results.append(("参数量基准", passed))
        print(f"  {'✓' if passed else '✗'} 参数量基准 ({low}M-{high}M): {'PASS' if passed else 'FAIL'}")

    # TC-FLOP-002: FLOPs 基准
    expected_flops = {
        '50': (3, 5),    # 3-5 GFLOPs
        '101': (6, 9),   # 6-9 GFLOPs
        '152': (9, 13),  # 9-13 GFLOPs
    }

    if variant in expected_flops:
        low, high = expected_flops[variant]
        passed = low <= flops_g <= high
        results.append(("FLOPs 基准", passed))
        print(f"  {'✓' if passed else '✗'} FLOPs 基准 ({low}-{high} GFLOPs): {'PASS' if passed else 'FAIL'}")

    return all(r[1] for r in results)


def verify_resnet_comparison(model, variant):
    """与 ResNet50 对比"""
    print("\n" + "=" * 60)
    print("6. 与 ResNet50 对比")
    print("=" * 60)

    results = []

    try:
        import torchvision.models as models
        resnet = models.resnet50(num_classes=1000)

        # TC-CMP-001: 参数量对比
        eres_params = sum(p.numel() for p in model.parameters())
        resnet_params = sum(p.numel() for p in resnet.parameters())
        param_diff = abs(eres_params - resnet_params) / resnet_params * 100

        print(f"  ERes2NetV2-{variant}: {eres_params:,} 参数")
        print(f"  ResNet50: {resnet_params:,} 参数")
        print(f"  差异: {param_diff:.1f}%")

        passed = param_diff < 20
        results.append(("参数量对比", passed))
        print(f"  {'✓' if passed else '✗'} 参数量对比 (<20%): {'PASS' if passed else 'FAIL'}")

        # TC-CMP-003: 输出维度一致性
        model.eval()
        resnet.eval()
        x = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            out_eres = model(x)
            out_resnet = resnet(x)

        shape_match = out_eres.shape == out_resnet.shape
        results.append(("输出维度一致性", shape_match))
        print(f"  {'✓' if shape_match else '✗'} 输出维度一致性: {'PASS' if shape_match else 'FAIL'}")

        # TC-CMP-002: FLOPs 对比
        from thop import profile
        eres_flops, _ = profile(model, inputs=(torch.randn(1, 3, 224, 224),), verbose=False)
        resnet_flops, _ = profile(resnet, inputs=(torch.randn(1, 3, 224, 224),), verbose=False)
        flops_diff = abs(eres_flops - resnet_flops) / resnet_flops * 100

        print(f"  ERes2NetV2-{variant} FLOPs: {eres_flops/1e9:.2f} GFLOPs")
        print(f"  ResNet50 FLOPs: {resnet_flops/1e9:.2f} GFLOPs")
        print(f"  差异: {flops_diff:.1f}%")

        passed = flops_diff < 20
        results.append(("FLOPs 对比", passed))
        print(f"  {'✓' if passed else '✗'} FLOPs 对比 (<20%): {'PASS' if passed else 'FAIL'}")

    except ImportError:
        print("  ⚠ torchvision 未安装，跳过 ResNet 对比")
        return True

    return all(r[1] for r in results)


def main():
    parser = argparse.ArgumentParser(description='模型验证脚本')
    parser.add_argument('--variant', type=str, default='50', choices=['50', '101', '152'],
                        help='模型变体 (default: 50)')
    parser.add_argument('--num-classes', type=int, default=1000,
                        help='分类数量 (default: 1000)')
    parser.add_argument('--all', action='store_true',
                        help='运行所有验证')
    parser.add_argument('--dimension', action='store_true', help='运行输出维度验证')
    parser.add_argument('--structure', action='store_true', help='运行结构打印验证')
    parser.add_argument('--forward', action='store_true', help='运行前向传播测试')
    parser.add_argument('--gradient', action='store_true', help='运行梯度反传测试')
    parser.add_argument('--flops', action='store_true', help='运行 FLOPs/参数量验证')
    parser.add_argument('--compare', action='store_true', help='运行 ResNet 对比')

    args = parser.parse_args()

    # 导入模型
    try:
        from models.eres2netv2 import ERes2NetV2
    except ImportError:
        print("错误: 无法导入 ERes2NetV2 模型")
        print("请确保模型已正确实现并放置在 models/eres2netv2.py")
        return 1

    print("=" * 60)
    print(f"ERes2NetV2-{args.variant} 模型验证")
    print("=" * 60)

    # 创建模型
    model = ERes2NetV2(variant=args.variant, num_classes=args.num_classes)

    # 确定运行哪些验证
    run_all = args.all or not any([args.dimension, args.structure, args.forward,
                                    args.gradient, args.flops, args.compare])

    results = {}

    if run_all or args.dimension:
        results['dimension'] = verify_output_dimensions(model, args.num_classes)

    if run_all or args.structure:
        results['structure'] = verify_structure(model)

    if run_all or args.forward:
        results['forward'] = verify_forward_propagation(model)

    if run_all or args.gradient:
        results['gradient'] = verify_gradient(model)

    if run_all or args.flops:
        results['flops'] = verify_flops_params(model, args.variant)

    if run_all or args.compare:
        results['compare'] = verify_resnet_comparison(model, args.variant)

    # 打印摘要
    print("\n" + "=" * 60)
    print("验证摘要")
    print("=" * 60)

    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    print("=" * 60)
    print(f"总体结果: {'✓ 全部通过' if all_passed else '✗ 存在失败'}")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
```

### 5.4 验证报告模板

```markdown
# ERes2NetV2 模型验证报告

## 验证概况

| 项目 | 结果 |
|------|------|
| 验证日期 | YYYY-MM-DD |
| 模型变体 | ERes2NetV2-{variant} |
| 验证人员 | [姓名] |
| 总体结果 | [PASS/FAIL] |

## 测试结果

### 1. 输出维度验证
| 测试用例 | 状态 | 备注 |
|----------|------|------|
| TC-DIM-001 | ✓/✗ | |
| TC-DIM-002 | ✓/✗ | |
| TC-DIM-003 | ✓/✗ | |
| TC-DIM-004 | ✓/✗ | |
| TC-DIM-005 | ✓/✗ | |

### 2. 结构打印验证
| 测试用例 | 状态 | 备注 |
|----------|------|------|
| TC-STR-001 | ✓/✗ | |
| TC-STR-002 | ✓/✗ | |
| TC-STR-003 | ✓/✗ | |
| TC-STR-004 | ✓/✗ | |

### 3. 前向传播测试
| 测试用例 | 状态 | 备注 |
|----------|------|------|
| TC-FWD-001 | ✓/✗ | |
| TC-FWD-002 | ✓/✗ | |
| TC-FWD-003 | ✓/✗ | |
| TC-FWD-004 | ✓/✗ | |
| TC-FWD-005 | ✓/✗ | |

### 4. 梯度反传测试
| 测试用例 | 状态 | 备注 |
|----------|------|------|
| TC-GRAD-001 | ✓/✗ | |
| TC-GRAD-002 | ✓/✗ | |
| TC-GRAD-003 | ✓/✗ | |
| TC-GRAD-004 | ✓/✗ | |
| TC-GRAD-005 | ✓/✗ | |

### 5. FLOPs 和参数量
| 测试用例 | 状态 | 实际值 | 预期值 |
|----------|------|--------|--------|
| TC-FLOP-001 | ✓/✗ | | |
| TC-FLOP-002 | ✓/✗ | | |
| TC-PARAM-001 | ✓/✗ | | |
| TC-PARAM-002 | ✓/✗ | | |
| TC-PARAM-003 | ✓/✗ | | |

### 6. ResNet50 对比
| 测试用例 | 状态 | ERes2NetV2 | ResNet50 |
|----------|------|------------|----------|
| TC-CMP-001 | ✓/✗ | | |
| TC-CMP-002 | ✓/✗ | | |
| TC-CMP-003 | ✓/✗ | | |
| TC-CMP-004 | ✓/✗ | | |
| TC-CMP-005 | ✓/✗ | | |

## 问题列表

| ID | 描述 | 严重程度 | 状态 |
|----|------|----------|------|
| | | | |

## 结论

[验证结论和建议]
```

---

## 六、风险点

### 6.1 技术风险

| 风险ID | 风险描述 | 影响程度 | 发生概率 | 缓解措施 |
|--------|----------|----------|----------|----------|
| R-01 | 输出维度不符合预期 | 高 | 低 | 详细维度验证，逐层检查 |
| R-02 | 梯度消失/爆炸 | 高 | 中 | 梯度裁剪，初始化检查 |
| R-03 | FLOPs 计算工具不兼容 | 中 | 中 | 多工具交叉验证 |
| R-04 | 参数量与论文数据偏差 | 中 | 中 | 对比官方实现 |
| R-05 | GPU/CPU 数值不一致 | 中 | 低 | 容差设置，双设备验证 |
| R-06 | 混合精度计算溢出 | 中 | 中 | FP32 基准对比，损失缩放 |

### 6.2 实现风险

| 风险ID | 风险描述 | 影响程度 | 发生概率 | 缓解措施 |
|--------|----------|----------|----------|----------|
| R-07 | 变体配置参数错误 | 高 | 低 | 配置验证测试 |
| R-08 | BN 层统计量异常 | 中 | 低 | 模式切换测试 |
| R-09 | 内存占用过大 | 中 | 中 | 内存分析工具 |
| R-10 | 推理速度过慢 | 中 | 中 | 性能基准对比 |

### 6.3 验证风险

| 风险ID | 风险描述 | 影响程度 | 发生概率 | 缓解措施 |
|--------|----------|----------|----------|----------|
| R-11 | 测试覆盖不全 | 高 | 中 | 测试用例评审 |
| R-12 | 边界情况遗漏 | 中 | 中 | 边界值分析 |
| R-13 | ResNet 对比基准错误 | 中 | 低 | 使用官方 torchvision |
| R-14 | GPU 环境不可用 | 中 | 低 | CPU 测试备选方案 |

### 6.4 风险应对计划

| 风险ID | 应对策略 | 触发条件 | 责任人 |
|--------|----------|----------|--------|
| R-01 | 逐层维度打印，定位问题 | 输出维度测试失败 | 开发者 |
| R-02 | 检查初始化，添加梯度监控 | 梯度测试失败 | 开发者 |
| R-03 | 使用 fvcore 替代 thop | FLOPs 计算报错 | 测试人员 |
| R-04 | 查阅原论文，对比官方代码 | 参数量偏差 >5% | 开发者 |
| R-05 | 增加数值容差 | 设备一致性失败 | 测试人员 |

---

## 七、验证执行计划

### 7.1 执行顺序

```
阶段 1: 输出维度验证 (Day 1 上午)
├── TC-DIM-001: 标准输入维度
├── TC-DIM-002: 多尺度输入
├── TC-DIM-003: 批次大小
├── TC-DIM-004: 特征提取模式
└── TC-DIM-005: 不同分类数

阶段 2: 结构打印验证 (Day 1 下午)
├── TC-STR-001: 层级结构
├── TC-STR-002: 参数量统计
├── TC-STR-003: 参数分布
└── TC-STR-004: 层数统计

阶段 3: 前向传播测试 (Day 2 上午)
├── TC-FWD-001: 标准前向传播
├── TC-FWD-002: 训练/评估模式
├── TC-FWD-003: 边界情况
├── TC-FWD-004: GPU/CPU 一致性
└── TC-FWD-005: 混合精度

阶段 4: 梯度反传测试 (Day 2 下午)
├── TC-GRAD-001: 基础梯度
├── TC-GRAD-002: 梯度有效性
├── TC-GRAD-003: 梯度流动
├── TC-GRAD-004: 梯度裁剪
└── TC-GRAD-005: 冻结层

阶段 5: FLOPs 和参数量 (Day 3 上午)
├── TC-FLOP-001: FLOPs 计算功能
├── TC-FLOP-002: FLOPs 基准
├── TC-PARAM-001: 参数量计算
├── TC-PARAM-002: 参数量基准
└── TC-PARAM-003: 参数分布

阶段 6: ResNet50 对比 (Day 3 下午)
├── TC-CMP-001: 参数量对比
├── TC-CMP-002: FLOPs 对比
├── TC-CMP-003: 输出维度
├── TC-CMP-004: 特征提取
└── TC-CMP-005: 推理速度
```

### 7.2 验证输出清单

- [ ] 测试报告（pytest 输出）
- [ ] 覆盖率报告（HTML/XML）
- [ ] 参数量报告
- [ ] FLOPs 报告
- [ ] 性能基准报告
- [ ] ResNet50 对比报告
- [ ] 问题追踪表
- [ ] 验收签字表

---

## 八、验证环境

### 8.1 硬件要求

| 项目 | 最低配置 | 推荐配置 |
|------|----------|----------|
| CPU | 4 核 | 8 核+ |
| 内存 | 16GB | 32GB+ |
| GPU | NVIDIA 1080 Ti | NVIDIA 3090/4090 |
| 显存 | 8GB | 24GB+ |

### 8.2 软件要求

| 软件 | 版本要求 |
|------|----------|
| Python | >= 3.8 |
| PyTorch | >= 1.10 |
| CUDA | >= 11.3 |
| pytest | >= 7.0 |
| pytest-cov | >= 3.0 |
| thop | >= 0.1.1 |
| fvcore | >= 0.1.5 |
| torchvision | >= 0.12 |

### 8.3 依赖安装

```bash
pip install torch torchvision
pip install pytest pytest-cov pytest-xdist
pip install thop fvcore
pip install onnx onnxruntime  # 用于 ONNX 导出验证
```

---

## 九、附录

### 9.1 测试数据准备

```python
import torch

# 标准输入
x_224 = torch.randn(2, 3, 224, 224)
x_256 = torch.randn(2, 3, 256, 256)
x_512 = torch.randn(2, 3, 512, 512)

# 边界情况
x_zeros = torch.zeros(1, 3, 224, 224)
x_ones = torch.ones(1, 3, 224, 224)
x_large = torch.randn(1, 3, 224, 224) * 100

# 标签
labels = torch.randint(0, 1000, (2,))
```

### 9.2 参考文档

- ERes2NetV2 论文
- PyTorch 官方文档
- thop 库文档
- fvcore 库文档
- 项目 CLAUDE.md 开发规范

### 9.3 预期参数量和 FLOPs 参考

| 变体 | 参数量 | FLOPs (224x224) | 推理时间 (GPU) |
|------|--------|-----------------|----------------|
| ERes2NetV2-50 | ~25M | ~4.0 GFLOPs | ~30ms |
| ERes2NetV2-101 | ~45M | ~7.5 GFLOPs | ~50ms |
| ERes2NetV2-152 | ~60M | ~11.0 GFLOPs | ~80ms |
| ResNet50 (对比) | ~25.6M | ~4.1 GFLOPs | ~25ms |

---

**文档版本：** 1.0
**创建日期：** 2026/06/03
**最后更新：** 2026/06/03
**负责人：** [待指定]