# ERes2NetV2 Backbone 实现验证计划

> **验证对象：** ERes2NetV2 Backbone 核心架构实现
> **验证日期：** 2026/06/02
> **验证阶段：** 阶段三：模型构建

---

## 一、验证目标

### 1.1 核心验证目标

| 目标ID | 验证目标 | 优先级 | 验证阶段 |
|--------|----------|--------|----------|
| V-01 | 基础组件功能正确性 | P0 | 单元测试 |
| V-02 | ERes2NetV2 Block 架构完整性 | P0 | 单元测试 |
| V-03 | 网络前向传播无报错 | P0 | 集成测试 |
| V-04 | 参数量符合预期 | P0 | 验收测试 |
| V-05 | 梯度反向传播正常 | P1 | 集成测试 |
| V-06 | 多变体配置兼容性 | P1 | 集成测试 |
| V-07 | 预训练权重加载 | P1 | 验收测试 |

### 1.2 验收标准映射

| 验收标准 | 对应目标ID | 验证方法 |
|----------|-----------|----------|
| Backbone 可独立运行 | V-01, V-02, V-03 | 集成测试 |
| 前向传播无报错 | V-03 | 集成测试 |
| 参数量符合预期 | V-04 | 验收测试 |

---

## 二、测试用例

### 2.1 基础组件测试用例

#### TC-CONV-001: ConvBNReLU 模块测试

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-CONV-001 |
| **测试名称** | ConvBNReLU 基础功能测试 |
| **前置条件** | ConvBNReLU 类已实现 |
| **测试步骤** | 1. 创建 ConvBNReLU 实例（in_channels=64, out_channels=128, kernel_size=3）<br>2. 输入随机张量 (B, 64, H, W)<br>3. 执行前向传播<br>4. 验证输出形状<br>5. 验证输出数值范围（ReLU 后非负） |
| **输入数据** | `torch.randn(2, 64, 32, 32)` |
| **预期输出** | 输出形状: `(2, 128, H', W')`，所有值 >= 0 |
| **通过条件** | 形状正确 + 数值非负 + 无报错 |

#### TC-CONV-002: ConvBNReLU 参数初始化测试

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-CONV-002 |
| **测试名称** | ConvBNReLU 参数初始化验证 |
| **前置条件** | ConvBNReLU 类已实现 |
| **测试步骤** | 1. 创建多个 ConvBNReLU 实例<br>2. 检查卷积层权重初始化<br>3. 检查 BN 层参数初始化 |
| **验证点** | - 卷积权重非全零<br>- BN gamma 非零<br>- BN beta 初始化为 0<br>- BN running_mean/var 存在 |
| **通过条件** | 所有点满足 |

#### TC-CONV-003: 1x1 卷积层测试

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-CONV-003 |
| **测试名称** | 1x1 卷积通道变换测试 |
| **前置条件** | 1x1 卷积组件已实现 |
| **测试步骤** | 1. 创建 1x1 卷积（in_channels=256, out_channels=512）<br>2. 输入 (B, 256, H, W)<br>3. 验证输出通道数<br>4. 验证空间维度不变 |
| **输入数据** | `torch.randn(1, 256, 14, 14)` |
| **预期输出** | 输出形状: `(1, 512, 14, 14)` |
| **通过条件** | 形状匹配 + 空间维度不变 |

#### TC-BN-001: Batch Normalization 测试

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-BN-001 |
| **测试名称** | BN 层训练/推理模式测试 |
| **前置条件** | BN 层已实现 |
| **测试步骤** | 1. 创建 BN 层<br>2. 设置训练模式，输入相同数据两次，验证输出不同（running stats 更新）<br>3. 设置推理模式，输入相同数据两次，验证输出相同 |
| **输入数据** | `torch.randn(4, 64, 8, 8)` |
| **验证点** | 训练模式: running_mean/var 更新<br>推理模式: 输出确定性 |
| **通过条件** | 两种模式行为正确 |

---

### 2.2 ERes2NetV2 Block 测试用例

#### TC-BLOCK-001: 分组特征提取测试

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-BLOCK-001 |
| **测试名称** | 通道分组处理正确性 |
| **前置条件** | ERes2NetV2 Block 已实现 |
| **测试步骤** | 1. 创建 Block（channels=256, groups=4）<br>2. 输入张量 (B, 256, H, W)<br>3. 验证每个分组独立处理<br>4. 验证分组后特征拼接正确 |
| **输入数据** | `torch.randn(2, 256, 16, 16)` |
| **验证点** | - 分组数 = groups<br>- 每组通道数 = channels/groups<br>- 输出通道数 = 输入通道数 |
| **通过条件** | 分组处理正确，输出形状匹配 |

#### TC-BLOCK-002: 特征重用机制测试

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-BLOCK-002 |
| **测试名称** | 特征重用 y_i = conv(x_i + y_{i-1}) 验证 |
| **前置条件** | ERes2NetV2 Block 已实现，特征重用机制已实现 |
| **测试步骤** | 1. 创建 Block（groups=4）<br>2. Hook 记录每个分组的中间输出<br>3. 前向传播<br>4. 验证 y_i 依赖于 y_{i-1} |
| **验证逻辑** | ```python<br># 验证: y_i = conv(x_i + y_{i-1})<br># 理论上如果 x_i = 0，则 y_i 仅依赖于 y_{i-1}<br>``` |
| **通过条件** | 级联依赖关系正确建立 |

#### TC-BLOCK-003: 通道交互模块测试

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-BLOCK-003 |
| **测试名称** | GAP -> FC -> 通道权重 流程验证 |
| **前置条件** | 通道交互模块已实现 |
| **测试步骤** | 1. 创建通道交互模块（channels=256, reduction=4）<br>2. 输入特征图<br>3. 验证 GAP 输出形状 (B, C, 1, 1)<br>4. 验证 FC 输出形状<br>5. 验证权重归一化（Sigmoid 后 [0, 1]） |
| **输入数据** | `torch.randn(2, 256, 14, 14)` |
| **预期输出** | 通道权重形状: `(2, 256, 1, 1)`，值域 [0, 1] |
| **通过条件** | 形状正确 + 值域正确 + 可微分 |

#### TC-BLOCK-004: 选择性残差融合测试

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-BLOCK-004 |
| **测试名称** | alpha * F(x) + beta * x 融合验证 |
| **前置条件** | 选择性残差融合模块已实现 |
| **测试步骤** | 1. 创建融合模块<br>2. 输入特征 x 和变换特征 F(x)<br>3. 验证输出 = alpha * F(x) + beta * x<br>4. 验证 alpha, beta 可学习 |
| **输入数据** | x: `torch.randn(2, 256, 14, 14)`<br>F_x: `torch.randn(2, 256, 14, 14)` |
| **验证点** | - alpha, beta 为可学习参数<br>- alpha + beta ≈ 1（初始化时）<br>- 融合公式正确 |
| **通过条件** | 公式正确 + 参数可学习 |

#### TC-BLOCK-005: Block 端到端测试

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-BLOCK-005 |
| **测试名称** | ERes2NetV2 Block 完整流程 |
| **前置条件** | Block 所有组件已实现 |
| **测试步骤** | 1. 创建完整 Block<br>2. 前向传播<br>3. 验证输出形状<br>4. 验证梯度流动 |
| **输入数据** | `torch.randn(2, 256, 14, 14)` |
| **预期输出** | 输出形状: `(2, 256, 14, 14)` |
| **通过条件** | 前向无错 + 梯度可传播 |

---

### 2.3 网络结构测试用例

#### TC-NET-001: Stem 层测试

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-NET-001 |
| **测试名称** | Stem 层特征提取验证 |
| **前置条件** | Stem 层已实现 |
| **测试步骤** | 1. 创建 Stem 层<br>2. 输入图像 (B, 3, H, W)<br>3. 验证输出形状<br>4. 验证下采样倍数 |
| **输入数据** | `torch.randn(2, 3, 224, 224)` |
| **预期输出** | 输出下采样 4x 或 8x（根据配置） |
| **通过条件** | 下采样倍数正确 + 通道数正确 |

#### TC-NET-002: Stage 结构测试

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-NET-002 |
| **测试名称** | Stage 1-4 级联测试 |
| **前置条件** | 所有 Stage 已实现 |
| **测试步骤** | 1. 依次通过 Stage 1-4<br>2. 验证每阶段下采样<br>3. 验证通道数变化 |
| **预期输出** | | Stage | 输入通道 | 输出通道 | 空间尺寸 |<br>|-------|---------|---------|---------|<br>| Stem  | 3       | C1      | H/4     |<br>| Stage1| C1      | C2      | H/8     |<br>| Stage2| C2      | C3      | H/16    |<br>| Stage3| C3      | C4      | H/32    |<br>| Stage4| C4      | C5      | H/64    | |
| **通过条件** | 每阶段输出符合预期 |

#### TC-NET-003: 全局平均池化测试

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-NET-003 |
| **测试名称** | GAP 层功能验证 |
| **前置条件** | GAP 层已实现 |
| **测试步骤** | 1. 输入特征图 (B, C, H, W)<br>2. 执行 GAP<br>3. 验证输出形状 (B, C) |
| **输入数据** | `torch.randn(2, 2048, 7, 7)` |
| **预期输出** | 输出形状: `(2, 2048)` |
| **通过条件** | 形状正确 + 数值正确（平均值） |

#### TC-NET-004: 分类头测试

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-NET-004 |
| **测试名称** | 分类头输出验证 |
| **前置条件** | 分类头已实现 |
| **测试步骤** | 1. 输入 GAP 后特征 (B, C)<br>2. 通过分类头<br>3. 验证输出形状 (B, num_classes)<br>4. 验证 Softmax 后概率和为 1 |
| **输入数据** | `torch.randn(2, 2048)` |
| **预期输出** | 输出形状: `(2, num_classes)`，概率和 = 1 |
| **通过条件** | 形状正确 + 概率和为 1 |

#### TC-NET-005: Backbone 完整前向传播测试

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-NET-005 |
| **测试名称** | Backbone 端到端前向传播 |
| **前置条件** | Backbone 完整实现 |
| **测试步骤** | 1. 创建 Backbone（默认配置）<br>2. 输入标准图像<br>3. 前向传播<br>4. 验证输出形状<br>5. 验证无内存泄漏 |
| **输入数据** | `torch.randn(2, 3, 224, 224)` |
| **预期输出** | 输出形状: `(2, num_classes)` 或 `(2, C, 1, 1)`（无分类头时） |
| **通过条件** | 前向无错 + 形状正确 + 内存无泄漏 |

#### TC-NET-006: 梯度反向传播测试

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-NET-006 |
| **测试名称** | Backbone 梯度流动验证 |
| **前置条件** | Backbone 完整实现 |
| **测试步骤** | 1. 创建 Backbone<br>2. 前向传播得到输出<br>3. 计算损失（如 MSE）<br>4. 反向传播<br>5. 验证所有参数梯度存在<br>6. 验证梯度非 NaN/Inf |
| **输入数据** | `torch.randn(2, 3, 224, 224)` |
| **验证点** | - 所有参数 `.grad` 存在<br>- 所有梯度非 NaN<br>- 所有梯度非 Inf<br>- 梯度值在合理范围 |
| **通过条件** | 所有点满足 |

---

### 2.4 多变体配置测试用例

#### TC-VAR-001: 变体配置加载测试

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-VAR-001 |
| **测试名称** | 多种变体配置兼容性 |
| **前置条件** | 变体配置文件已定义 |
| **测试步骤** | 1. 加载每种变体配置<br>2. 验证配置参数完整<br>3. 创建对应模型 |
| **变体列表** | - ERes2NetV2_small<br>- ERes2NetV2_base<br>- ERes2NetV2_large |
| **通过条件** | 每种变体均可实例化 |

#### TC-VAR-002: 变体参数量对比测试

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-VAR-002 |
| **测试名称** | 变体参数量符合预期 |
| **前置条件** | 所有变体已实现 |
| **测试步骤** | 1. 计算每种变体参数量<br>2. 与预期参数量对比 |
| **预期参数量** | | 变体 | 预期参数量 |<br>|------|-----------|<br>| small  | ~10M      |<br>| base   | ~25M      |<br>| large  | ~50M      | |
| **通过条件** | 偏差 < 5% |

---

### 2.5 预训练权重测试用例

#### TC-PRETRAIN-001: 权重加载测试

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-PRETRAIN-001 |
| **测试名称** | 预训练权重加载功能 |
| **前置条件** | 预训练权重文件可用 |
| **测试步骤** | 1. 初始化模型<br>2. 加载预训练权重<br>3. 验证参数加载成功<br>4. 验证参数值变化 |
| **验证点** | - 加载无报错<br>- 参数名匹配<br>- 参数形状匹配<br>- 参数值已更新 |
| **通过条件** | 所有点满足 |

#### TC-PRETRAIN-002: 部分权重加载测试

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-PRETRAIN-002 |
| **测试名称** | 部分权重加载（迁移学习） |
| **前置条件** | 预训练权重文件可用 |
| **测试步骤** | 1. 创建不同 num_classes 的模型<br>2. 加载预训练权重（strict=False）<br>3. 验证 Backbone 权重加载<br>4. 验证分类头随机初始化 |
| **通过条件** | Backbone 权重正确加载 + 分类头独立 |

---

### 2.6 参数量验证测试用例

#### TC-PARAM-001: 参数量统计测试

| 字段 | 内容 |
|------|------|
| **测试ID** | TC-PARAM-001 |
| **测试名称** | Backbone 参数量统计 |
| **前置条件** | Backbone 完整实现 |
| **测试步骤** | 1. 计算总参数量<br>2. 计算可训练参数量<br>3. 计算各层参数量分布 |
| **验证代码** | ```python<br>def count_parameters(model):<br>    total = sum(p.numel() for p in model.parameters())<br>    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)<br>    return total, trainable<br>``` |
| **通过条件** | 参数量在预期范围内 |

---

## 三、验证方法

### 3.1 单元测试

| 方法 | 描述 | 适用测试用例 |
|------|------|-------------|
| pytest | Python 单元测试框架 | TC-CONV-*, TC-BN-*, TC-BLOCK-* |
| torch.autograd.gradcheck | 梯度正确性验证 | TC-BLOCK-005, TC-NET-006 |
| torch.testing.assert_close | 张量数值验证 | 所有数值验证用例 |

### 3.2 集成测试

| 方法 | 描述 | 适用测试用例 |
|------|------|-------------|
| 端到端前向传播 | 完整模型前向传播 | TC-NET-005 |
| 梯度流动检查 | 反向传播完整性 | TC-NET-006 |
| 内存泄漏检测 | 长时间运行监控 | TC-NET-005 |

### 3.3 验收测试

| 方法 | 描述 | 适用测试用例 |
|------|------|-------------|
| 参数量验证 | 与论文/官方实现对比 | TC-PARAM-001, TC-VAR-002 |
| 预训练权重加载 | 官方权重加载验证 | TC-PRETRAIN-* |

### 3.4 测试代码框架

```python
import pytest
import torch
import torch.nn as nn


class TestConvBNReLU:
    """基础组件测试"""

    def test_conv_bn_relu_forward(self):
        """TC-CONV-001: ConvBNReLU 前向传播"""
        from models.backbones import ConvBNReLU

        layer = ConvBNReLU(64, 128, kernel_size=3, stride=1, padding=1)
        x = torch.randn(2, 64, 32, 32)
        out = layer(x)

        assert out.shape == (2, 128, 32, 32), f"Expected (2, 128, 32, 32), got {out.shape}"
        assert (out >= 0).all(), "ReLU output should be non-negative"


class TestERes2NetV2Block:
    """ERes2NetV2 Block 测试"""

    def test_block_forward(self):
        """TC-BLOCK-005: Block 端到端"""
        from models.backbones import ERes2NetV2Block

        block = ERes2NetV2Block(channels=256, groups=4)
        x = torch.randn(2, 256, 14, 14)
        out = block(x)

        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_block_gradient(self):
        """TC-NET-006: 梯度流动"""
        from models.backbones import ERes2NetV2Block

        block = ERes2NetV2Block(channels=256, groups=4)
        x = torch.randn(2, 256, 14, 14, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "Input gradient is None"
        assert not torch.isnan(x.grad).any(), "Gradient contains NaN"
        assert not torch.isinf(x.grad).any(), "Gradient contains Inf"


class TestERes2NetV2Backbone:
    """Backbone 完整测试"""

    def test_backbone_forward(self):
        """TC-NET-005: 完整前向传播"""
        from models.backbones import ERes2NetV2

        model = ERes2NetV2(variant='base', num_classes=1000)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)

        assert out.shape == (2, 1000), f"Expected (2, 1000), got {out.shape}"

    def test_backbone_gradient(self):
        """TC-NET-006: 梯度反向传播"""
        from models.backbones import ERes2NetV2

        model = ERes2NetV2(variant='base', num_classes=1000)
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()

        # 检查所有参数梯度
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} gradient is None"
            assert not torch.isnan(param.grad).any(), f"{name} gradient contains NaN"
            assert not torch.isinf(param.grad).any(), f"{name} gradient contains Inf"

    @pytest.mark.parametrize("variant,expected_params", [
        ("small", 10_000_000),
        ("base", 25_000_000),
        ("large", 50_000_000),
    ])
    def test_variant_parameters(self, variant, expected_params):
        """TC-VAR-002: 变体参数量验证"""
        from models.backbones import ERes2NetV2

        model = ERes2NetV2(variant=variant)
        total_params = sum(p.numel() for p in model.parameters())

        # 允许 5% 偏差
        assert abs(total_params - expected_params) / expected_params < 0.05, \
            f"{variant}: expected ~{expected_params}, got {total_params}"
```

---

## 四、通过标准

### 4.1 测试通过标准

| 级别 | 通过标准 | 说明 |
|------|----------|------|
| P0 测试 | 100% 通过 | 核心功能，必须全部通过 |
| P1 测试 | >= 90% 通过 | 重要功能，允许少量失败但需记录 |
| P2 测试 | >= 80% 通过 | 辅助功能 |

### 4.2 参数量通过标准

| 变体 | 预期参数量 | 允许偏差 | 通过条件 |
|------|-----------|----------|----------|
| small | ~10M | ±5% | 9.5M - 10.5M |
| base | ~25M | ±5% | 23.75M - 26.25M |
| large | ~50M | ±5% | 47.5M - 52.5M |

### 4.3 性能通过标准

| 指标 | 标准 | 测试条件 |
|------|------|----------|
| 前向传播时间 | < 100ms | Batch=1, 224x224, GPU |
| 内存占用 | < 2GB | Batch=32, 224x224, GPU |
| 梯度计算时间 | < 150ms | Batch=1, 224x224, GPU |

---

## 五、自动化建议

### 5.1 测试自动化

```bash
# 运行所有测试
pytest tests/test_backbone.py -v

# 运行特定测试
pytest tests/test_backbone.py::TestERes2NetV2Backbone -v

# 生成覆盖率报告
pytest tests/test_backbone.py --cov=models.backbones --cov-report=html

# 并行测试
pytest tests/test_backbone.py -n auto
```

### 5.2 CI/CD 集成

```yaml
# .github/workflows/test_backbone.yml
name: Backbone Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-xdist
      - name: Run tests
        run: |
          pytest tests/test_backbone.py -v --cov=models.backbones --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### 5.3 自动化验证脚本

```python
# scripts/verify_backbone.py
#!/usr/bin/env python
"""自动化验证脚本"""
import argparse
import torch
from models.backbones import ERes2NetV2


def verify_forward(variant='base'):
    """验证前向传播"""
    model = ERes2NetV2(variant=variant)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    print(f"✓ Forward pass: input {tuple(x.shape)} -> output {tuple(out.shape)}")
    return True


def verify_gradient(variant='base'):
    """验证梯度传播"""
    model = ERes2NetV2(variant=variant)
    model.train()
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    issues = []
    for name, param in model.named_parameters():
        if param.grad is None:
            issues.append(f"{name}: no gradient")
        elif torch.isnan(param.grad).any():
            issues.append(f"{name}: NaN gradient")
        elif torch.isinf(param.grad).any():
            issues.append(f"{name}: Inf gradient")

    if issues:
        print(f"✗ Gradient check failed:\n  " + "\n  ".join(issues))
        return False
    print(f"✓ Gradient check passed for {sum(p.numel() for p in model.parameters()):,} parameters")
    return True


def verify_parameters(variant='base', expected=None):
    """验证参数量"""
    model = ERes2NetV2(variant=variant)
    total = sum(p.numel() for p in model.parameters())

    expected_map = {'small': 10_000_000, 'base': 25_000_000, 'large': 50_000_000}
    expected = expected or expected_map.get(variant)

    if expected:
        diff = abs(total - expected) / expected
        if diff < 0.05:
            print(f"✓ Parameter count: {total:,} (expected ~{expected:,}, diff={diff:.1%})")
            return True
        else:
            print(f"✗ Parameter count: {total:,} (expected ~{expected:,}, diff={diff:.1%})")
            return False
    else:
        print(f"○ Parameter count: {total:,} (no expected value)")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', default='base', choices=['small', 'base', 'large'])
    parser.add_argument('--skip-gradient', action='store_true')
    args = parser.parse_args()

    results = []
    results.append(('Forward', verify_forward(args.variant)))
    if not args.skip_gradient:
        results.append(('Gradient', verify_gradient(args.variant)))
    results.append(('Parameters', verify_parameters(args.variant)))

    print("\n" + "=" * 40)
    print("Verification Summary")
    print("=" * 40)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)
    print("=" * 40)
    print(f"Overall: {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}")
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
```

---

## 六、风险点

### 6.1 技术风险

| 风险ID | 风险描述 | 影响程度 | 发生概率 | 缓解措施 |
|--------|----------|----------|----------|----------|
| R-01 | 分组卷积实现与预期不符 | 高 | 中 | 逐层验证分组数量和通道分配 |
| R-02 | 特征重用机制梯度消失 | 高 | 中 | 梯度检查，添加跳跃连接 |
| R-03 | 通道交互模块计算量大 | 中 | 低 | 性能测试，考虑优化方案 |
| R-04 | 预训练权重格式不匹配 | 高 | 中 | 编写权重转换脚本 |
| R-05 | 多变体配置参数遗漏 | 中 | 低 | 配置验证测试 |

### 6.2 实现风险

| 风险ID | 风险描述 | 影响程度 | 发生概率 | 缓解措施 |
|--------|----------|----------|----------|----------|
| R-06 | BN 层统计量初始化不当 | 中 | 低 | 检查初始化策略 |
| R-07 | 残差融合 alpha/beta 初始化 | 中 | 低 | 验证初始值范围 |
| R-08 | 内存占用过大 | 中 | 中 | 内存分析，梯度检查点 |
| R-09 | 特定输入尺寸报错 | 中 | 中 | 多尺寸输入测试 |

### 6.3 验证风险

| 风险ID | 风险描述 | 影响程度 | 发生概率 | 缓解措施 |
|--------|----------|----------|----------|----------|
| R-10 | 参数量基准数据缺失 | 高 | 中 | 参考官方实现/论文 |
| R-11 | 预训练权重不可用 | 高 | 低 | 使用 mock 数据测试加载逻辑 |
| R-12 | GPU 环境差异 | 中 | 中 | CPU/GPU 双环境测试 |

---

## 七、验证执行计划

### 7.1 执行顺序

```
阶段 1: 基础组件验证 (Day 1)
├── TC-CONV-001: ConvBNReLU 基础功能
├── TC-CONV-002: ConvBNReLU 参数初始化
├── TC-CONV-003: 1x1 卷积层
└── TC-BN-001: Batch Normalization

阶段 2: Block 组件验证 (Day 2-3)
├── TC-BLOCK-001: 分组特征提取
├── TC-BLOCK-002: 特征重用机制
├── TC-BLOCK-003: 通道交互模块
├── TC-BLOCK-004: 选择性残差融合
└── TC-BLOCK-005: Block 端到端

阶段 3: 网络结构验证 (Day 4-5)
├── TC-NET-001: Stem 层
├── TC-NET-002: Stage 结构
├── TC-NET-003: 全局平均池化
├── TC-NET-004: 分类头
├── TC-NET-005: Backbone 完整前向传播
└── TC-NET-006: 梯度反向传播

阶段 4: 变体与预训练验证 (Day 6)
├── TC-VAR-001: 变体配置加载
├── TC-VAR-002: 变体参数量对比
├── TC-PRETRAIN-001: 权重加载
└── TC-PRETRAIN-002: 部分权重加载

阶段 5: 验收测试 (Day 7)
├── TC-PARAM-001: 参数量统计
├── 性能基准测试
└── 回归测试
```

### 7.2 验证输出清单

- [ ] 测试报告（pytest 输出）
- [ ] 覆盖率报告（HTML/XML）
- [ ] 参数量报告
- [ ] 性能基准报告
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

---

## 九、附录

### 9.1 测试数据准备

```python
# 生成测试数据
import torch

# 标准输入
x_224 = torch.randn(2, 3, 224, 224)
x_512 = torch.randn(2, 3, 512, 512)

# 边界情况
x_single = torch.randn(1, 3, 224, 224)  # 单样本
x_large_batch = torch.randn(64, 3, 224, 224)  # 大批次
x_small = torch.randn(2, 3, 32, 32)  # 小尺寸
```

### 9.2 参考文档

- ERes2NetV2 论文
- PyTorch 官方文档
- 项目 CLAUDE.md 开发规范
- 验收标准文档

---

**文档版本：** 1.0
**创建日期：** 2026/06/02
**最后更新：** 2026/06/02
**负责人：** [待指定]