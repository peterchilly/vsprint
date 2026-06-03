# 模型变体配置验证测试计划

> **验证阶段：** 阶段三：模型构建
> **验证目标：** 确保 ERes2NetV2 模型变体配置正确实现，所有变体可正常创建且参数量符合预期
> **创建日期：** 2026/06/02

---

## 一、验证目标

### 1.1 主要验证目标

| 序号 | 验证目标 | 优先级 | 说明 |
|------|---------|--------|------|
| 1 | 模型变体可创建性 | P0 | 验证所有定义的变体都能成功实例化 |
| 2 | 参数量正确性 | P0 | 验证各变体参数量符合设计预期 |
| 3 | 配置文件正确性 | P1 | 验证配置文件参数传递正确 |
| 4 | 统一接口可用性 | P1 | 验证统一创建接口功能完整 |
| 5 | 前向传播正确性 | P0 | 验证各变体能完成前向计算 |
| 6 | 输出维度正确性 | P0 | 验证输出特征维度符合预期 |

### 1.2 验证范围

```
├── ERes2NetV2-50（轻量级）
│   ├── 模型创建测试
│   ├── 参数量验证
│   └── 前向传播测试
├── ERes2NetV2-101（标准版）
│   ├── 模型创建测试
│   ├── 参数量验证
│   └── 前向传播测试
├── ERes2NetV2-152（深度版）
│   ├── 模型创建测试
│   ├── 参数量验证
│   └── 前向传播测试
├── 配置文件管理
│   ├── 配置加载测试
│   └── 参数覆盖测试
└── 统一创建接口
    ├── 接口调用测试
    └── 异常处理测试
```

---

## 二、测试用例

### 2.1 模型创建测试用例

#### TC-001: ERes2NetV2-50 模型创建

| 项目 | 内容 |
|------|------|
| **测试ID** | TC-001 |
| **测试名称** | ERes2NetV2-50 模型创建 |
| **前置条件** | 1. 模型代码已实现<br>2. 依赖环境已安装 |
| **测试步骤** | 1. 导入模型模块<br>2. 调用创建接口，指定 variant='50'<br>3. 验证模型实例化成功 |
| **测试输入** | `model = create_model('ERes2NetV2-50', num_classes=1000)` |
| **预期输出** | 模型对象成功创建，无异常抛出 |
| **实际结果** | _待填写_ |
| **测试状态** | _待执行_ |

#### TC-002: ERes2NetV2-101 模型创建

| 项目 | 内容 |
|------|------|
| **测试ID** | TC-002 |
| **测试名称** | ERes2NetV2-101 模型创建 |
| **前置条件** | 1. 模型代码已实现<br>2. 依赖环境已安装 |
| **测试步骤** | 1. 导入模型模块<br>2. 调用创建接口，指定 variant='101'<br>3. 验证模型实例化成功 |
| **测试输入** | `model = create_model('ERes2NetV2-101', num_classes=1000)` |
| **预期输出** | 模型对象成功创建，无异常抛出 |
| **实际结果** | _待填写_ |
| **测试状态** | _待执行_ |

#### TC-003: ERes2NetV2-152 模型创建

| 项目 | 内容 |
|------|------|
| **测试ID** | TC-003 |
| **测试名称** | ERes2NetV2-152 模型创建 |
| **前置条件** | 1. 模型代码已实现<br>2. 依赖环境已安装 |
| **测试步骤** | 1. 导入模型模块<br>2. 调用创建接口，指定 variant='152'<br>3. 验证模型实例化成功 |
| **测试输入** | `model = create_model('ERes2NetV2-152', num_classes=1000)` |
| **预期输出** | 模型对象成功创建，无异常抛出 |
| **实际结果** | _待填写_ |
| **测试状态** | _待执行_ |

### 2.2 参数量验证测试用例

#### TC-004: ERes2NetV2-50 参数量验证

| 项目 | 内容 |
|------|------|
| **测试ID** | TC-004 |
| **测试名称** | ERes2NetV2-50 参数量验证 |
| **前置条件** | TC-001 通过 |
| **测试步骤** | 1. 创建 ERes2NetV2-50 模型<br>2. 计算总参数量<br>3. 计算可训练参数量<br>4. 与预期值比较 |
| **测试输入** | `count_parameters(model)` |
| **预期输出** | 总参数量 ≈ 25M ± 5%<br>可训练参数量 = 总参数量 |
| **验证公式** | `abs(actual - expected) / expected < 0.05` |
| **实际结果** | _待填写_ |
| **测试状态** | _待执行_ |

#### TC-005: ERes2NetV2-101 参数量验证

| 项目 | 内容 |
|------|------|
| **测试ID** | TC-005 |
| **测试名称** | ERes2NetV2-101 参数量验证 |
| **前置条件** | TC-002 通过 |
| **测试步骤** | 1. 创建 ERes2NetV2-101 模型<br>2. 计算总参数量<br>3. 计算可训练参数量<br>4. 与预期值比较 |
| **测试输入** | `count_parameters(model)` |
| **预期输出** | 总参数量 ≈ 45M ± 5%<br>可训练参数量 = 总参数量 |
| **验证公式** | `abs(actual - expected) / expected < 0.05` |
| **实际结果** | _待填写_ |
| **测试状态** | _待执行_ |

#### TC-006: ERes2NetV2-152 参数量验证

| 项目 | 内容 |
|------|------|
| **测试ID** | TC-006 |
| **测试名称** | ERes2NetV2-152 参数量验证 |
| **前置条件** | TC-003 通过 |
| **测试步骤** | 1. 创建 ERes2NetV2-152 模型<br>2. 计算总参数量<br>3. 计算可训练参数量<br>4. 与预期值比较 |
| **测试输入** | `count_parameters(model)` |
| **预期输出** | 总参数量 ≈ 60M ± 5%<br>可训练参数量 = 总参数量 |
| **验证公式** | `abs(actual - expected) / expected < 0.05` |
| **实际结果** | _待填写_ |
| **测试状态** | _待执行_ |

### 2.3 前向传播测试用例

#### TC-007: 标准输入前向传播测试

| 项目 | 内容 |
|------|------|
| **测试ID** | TC-007 |
| **测试名称** | 标准输入前向传播测试 |
| **前置条件** | TC-001 ~ TC-003 通过 |
| **测试步骤** | 1. 创建各变体模型<br>2. 构造标准输入张量 (B, 3, 224, 224)<br>3. 执行前向传播<br>4. 验证输出形状 |
| **测试输入** | `input_tensor = torch.randn(1, 3, 224, 224)` |
| **预期输出** | 输出形状: (1, num_classes)<br>无异常抛出 |
| **实际结果** | _待填写_ |
| **测试状态** | _待执行_ |

#### TC-008: 批量输入前向传播测试

| 项目 | 内容 |
|------|------|
| **测试ID** | TC-008 |
| **测试名称** | 批量输入前向传播测试 |
| **前置条件** | TC-007 通过 |
| **测试步骤** | 1. 创建各变体模型<br>2. 构造批量输入张量 (B, 3, 224, 224), B ∈ {2, 4, 8, 16}<br>3. 执行前向传播<br>4. 验证输出形状 |
| **测试输入** | `for batch_size in [2, 4, 8, 16]: ...` |
| **预期输出** | 输出形状: (B, num_classes)<br>所有批次大小测试通过 |
| **实际结果** | _待填写_ |
| **测试状态** | _待执行_ |

#### TC-009: 不同分辨率输入测试

| 项目 | 内容 |
|------|------|
| **测试ID** | TC-009 |
| **测试名称** | 不同分辨率输入测试 |
| **前置条件** | TC-007 通过 |
| **测试步骤** | 1. 创建各变体模型<br>2. 构造不同分辨率输入<br>3. 执行前向传播<br>4. 验证输出形状 |
| **测试输入** | 分辨率: 224×224, 256×256, 384×384, 512×512 |
| **预期输出** | 各分辨率下输出形状正确<br>无异常抛出 |
| **实际结果** | _待填写_ |
| **测试状态** | _待执行_ |

### 2.4 配置文件测试用例

#### TC-010: 配置文件加载测试

| 项目 | 内容 |
|------|------|
| **测试ID** | TC-010 |
| **测试名称** | 配置文件加载测试 |
| **前置条件** | 配置文件已创建 |
| **测试步骤** | 1. 加载配置文件<br>2. 验证配置项完整性<br>3. 验证配置值有效性 |
| **测试输入** | 配置文件路径 |
| **预期输出** | 所有必需配置项存在<br>配置值在有效范围内 |
| **必需配置项** | - layers_per_block<br>- channels<br>- expansion<br>- num_classes |
| **实际结果** | _待填写_ |
| **测试状态** | _待执行_ |

#### TC-011: 配置参数覆盖测试

| 项目 | 内容 |
|------|------|
| **测试ID** | TC-011 |
| **测试名称** | 配置参数覆盖测试 |
| **前置条件** | TC-010 通过 |
| **测试步骤** | 1. 加载默认配置<br>2. 应用自定义参数<br>3. 验证参数正确覆盖 |
| **测试输入** | `config = load_config('ERes2NetV2-50', num_classes=100, dropout=0.5)` |
| **预期输出** | 自定义参数生效<br>默认参数保持不变 |
| **实际结果** | _待填写_ |
| **测试状态** | _待执行_ |

### 2.5 统一接口测试用例

#### TC-012: 统一创建接口测试

| 项目 | 内容 |
|------|------|
| **测试ID** | TC-012 |
| **测试名称** | 统一创建接口测试 |
| **前置条件** | 接口已实现 |
| **测试步骤** | 1. 调用统一创建接口<br>2. 传入不同变体名称<br>3. 验证返回正确的模型实例 |
| **测试输入** | 所有支持的变体名称 |
| **预期输出** | 返回对应变体的模型实例<br>类型检查通过 |
| **实际结果** | _待填写_ |
| **测试状态** | _待执行_ |

#### TC-013: 无效变体名称异常测试

| 项目 | 内容 |
|------|------|
| **测试ID** | TC-013 |
| **测试名称** | 无效变体名称异常测试 |
| **前置条件** | 接口已实现 |
| **测试步骤** | 1. 调用创建接口<br>2. 传入无效的变体名称<br>3. 验证异常抛出 |
| **测试输入** | `create_model('Invalid-Variant')` |
| **预期输出** | 抛出 ValueError 或 KeyError<br>异常信息包含有效变体列表 |
| **实际结果** | _待填写_ |
| **测试状态** | _待执行_ |

#### TC-014: 接口参数验证测试

| 项目 | 内容 |
|------|------|
| **测试ID** | TC-014 |
| **测试名称** | 接口参数验证测试 |
| **前置条件** | 接口已实现 |
| **测试步骤** | 1. 传入无效参数值<br>2. 验证参数验证逻辑<br>3. 验证错误提示清晰 |
| **测试输入** | 无效参数: num_classes=-1, dropout=2.0 |
| **预期输出** | 抛出适当的异常<br>错误信息清晰明确 |
| **实际结果** | _待填写_ |
| **测试状态** | _待执行_ |

### 2.6 特性提取测试用例

#### TC-015: 特征提取接口测试

| 项目 | 内容 |
|------|------|
| **测试ID** | TC-015 |
| **测试名称** | 特征提取接口测试 |
| **前置条件** | 模型创建成功 |
| **测试步骤** | 1. 创建模型并设为特征提取模式<br>2. 执行前向传播<br>3. 验证返回的特征形状 |
| **测试输入** | `model = create_model('ERes2NetV2-50', features_only=True)` |
| **预期输出** | 返回多尺度特征列表<br>特征维度逐层递减 |
| **实际结果** | _待填写_ |
| **测试状态** | _待执行_ |

---

## 三、验证方法

### 3.1 单元测试

```python
# tests/test_model_variants.py

import pytest
import torch
from model import create_model, get_model_config

class TestModelCreation:
    """模型创建测试类"""

    @pytest.mark.parametrize("variant,expected_params", [
        ("ERes2NetV2-50", 25_000_000),
        ("ERes2NetV2-101", 45_000_000),
        ("ERes2NetV2-152", 60_000_000),
    ])
    def test_model_creation(self, variant, expected_params):
        """测试模型创建和参数量"""
        model = create_model(variant, num_classes=1000)
        assert model is not None

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert total_params == trainable_params
        assert abs(total_params - expected_params) / expected_params < 0.05

class TestForwardPass:
    """前向传播测试类"""

    @pytest.fixture
    def models(self):
        """创建所有变体模型"""
        variants = ["ERes2NetV2-50", "ERes2NetV2-101", "ERes2NetV2-152"]
        return {v: create_model(v, num_classes=1000) for v in variants}

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
    def test_batch_forward(self, models, batch_size):
        """测试批量前向传播"""
        input_tensor = torch.randn(batch_size, 3, 224, 224)

        for variant, model in models.items():
            model.eval()
            with torch.no_grad():
                output = model(input_tensor)

            assert output.shape == (batch_size, 1000), \
                f"{variant} output shape mismatch"

    @pytest.mark.parametrize("resolution", [224, 256, 384, 512])
    def test_resolution_forward(self, models, resolution):
        """测试不同分辨率输入"""
        input_tensor = torch.randn(1, 3, resolution, resolution)

        for variant, model in models.items():
            model.eval()
            with torch.no_grad():
                output = model(input_tensor)

            assert output.shape[0] == 1
            assert output.shape[1] == 1000

class TestUnifiedInterface:
    """统一接口测试类"""

    def test_valid_variants(self):
        """测试有效变体名称"""
        valid_variants = ["ERes2NetV2-50", "ERes2NetV2-101", "ERes2NetV2-152"]

        for variant in valid_variants:
            model = create_model(variant)
            assert model is not None

    def test_invalid_variant(self):
        """测试无效变体名称"""
        with pytest.raises((ValueError, KeyError)):
            create_model("ERes2NetV2-999")

    def test_invalid_parameters(self):
        """测试无效参数"""
        with pytest.raises((ValueError, TypeError)):
            create_model("ERes2NetV2-50", num_classes=-1)

        with pytest.raises((ValueError, TypeError)):
            create_model("ERes2NetV2-50", dropout=2.0)
```

### 3.2 集成测试

```python
# tests/test_model_integration.py

import pytest
import torch
import torch.nn as nn

class TestModelIntegration:
    """模型集成测试"""

    def test_training_step(self):
        """测试训练步骤"""
        model = create_model("ERes2NetV2-50", num_classes=1000)
        model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # 模拟训练步骤
        input_tensor = torch.randn(4, 3, 224, 224)
        target = torch.randint(0, 1000, (4,))

        optimizer.zero_grad()
        output = model(input_tensor)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        assert loss.item() > 0

    def test_inference_mode(self):
        """测试推理模式"""
        model = create_model("ERes2NetV2-101", num_classes=1000)
        model.eval()

        input_tensor = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            output1 = model(input_tensor)
            output2 = model(input_tensor)

        # 推理模式下相同输入应产生相同输出
        assert torch.allclose(output1, output2)
```

### 3.3 验证脚本

```bash
#!/bin/bash
# scripts/verify_model_variants.sh

echo "=== ERes2NetV2 模型变体验证脚本 ==="

# 运行单元测试
echo "1. 运行单元测试..."
pytest tests/test_model_variants.py -v --tb=short

# 运行集成测试
echo "2. 运行集成测试..."
pytest tests/test_model_integration.py -v --tb=short

# 生成参数量报告
echo "3. 生成参数量报告..."
python scripts/report_model_params.py

# 运行内存分析
echo "4. 运行内存分析..."
python scripts/analyze_memory.py

echo "=== 验证完成 ==="
```

---

## 四、通过标准

### 4.1 必须满足的标准

| 标准ID | 标准描述 | 验证方法 | 通过条件 |
|--------|---------|---------|---------|
| CR-001 | 所有变体创建成功 | TC-001~003 | 无异常抛出 |
| CR-002 | 参数量在预期范围内 | TC-004~006 | 误差 < 5% |
| CR-003 | 前向传播正常 | TC-007~009 | 输出形状正确 |
| CR-004 | 配置文件正确加载 | TC-010~011 | 配置项完整有效 |
| CR-005 | 统一接口功能完整 | TC-012~014 | 所有测试通过 |

### 4.2 应当满足的标准

| 标准ID | 标准描述 | 验证方法 | 通过条件 |
|--------|---------|---------|---------|
| SR-001 | 测试覆盖率 ≥ 90% | pytest-cov | coverage ≥ 90% |
| SR-002 | 无 lint 警告 | flake8/pylint | 警告数 = 0 |
| SR-003 | 类型检查通过 | mypy | 无类型错误 |
| SR-004 | 文档完整 | 手动检查 | 所有接口有文档 |

### 4.3 可选优化标准

| 标准ID | 标准描述 | 验证方法 | 目标 |
|--------|---------|---------|------|
| OP-001 | 推理速度基准 | benchmark | 建立性能基线 |
| OP-002 | 内存占用分析 | memory_profiler | 优化内存使用 |
| OP-003 | GPU 利用率 | nvidia-smi | ≥ 80% 利用率 |

---

## 五、自动化建议

### 5.1 CI/CD 集成

```yaml
# .github/workflows/test_model_variants.yml

name: Model Variant Tests

on:
  push:
    paths:
      - 'model/**'
      - 'tests/**'
  pull_request:
    paths:
      - 'model/**'
      - 'tests/**'

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
          pip install pytest pytest-cov

      - name: Run unit tests
        run: |
          pytest tests/test_model_variants.py -v --cov=model --cov-report=xml

      - name: Run integration tests
        run: |
          pytest tests/test_model_integration.py -v

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

      - name: Generate parameter report
        run: python scripts/report_model_params.py

      - name: Archive reports
        uses: actions/upload-artifact@v3
        with:
          name: model-reports
          path: reports/
```

### 5.2 参数量报告脚本

```python
# scripts/report_model_params.py

import torch
from tabulate import tabulate
from model import create_model, MODEL_CONFIGS

def count_parameters(model):
    """计算模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def generate_report():
    """生成参数量报告"""
    results = []

    for variant in ["ERes2NetV2-50", "ERes2NetV2-101", "ERes2NetV2-152"]:
        model = create_model(variant, num_classes=1000)
        total, trainable = count_parameters(model)

        config = MODEL_CONFIGS[variant]

        results.append({
            "变体": variant,
            "总参数量": f"{total:,}",
            "可训练参数量": f"{trainable:,}",
            "层数": sum(config["layers"]),
            "模型大小(MB)": f"{total * 4 / 1024 / 1024:.2f}"
        })

    print(tabulate(results, headers="keys", tablefmt="grid"))

    # 保存报告
    with open("reports/model_params.md", "w") as f:
        f.write(tabulate(results, headers="keys", tablefmt="pipe"))

if __name__ == "__main__":
    generate_report()
```

### 5.3 自动化验证检查清单

```markdown
## 自动化验证检查清单

### 每次提交触发
- [ ] 单元测试通过
- [ ] 代码覆盖率检查
- [ ] Lint 检查通过

### 每日构建触发
- [ ] 完整测试套件
- [ ] 参数量报告生成
- [ ] 内存泄漏检测
- [ ] 性能基准测试

### 发布前触发
- [ ] 所有变体完整验证
- [ ] 文档生成
- [ ] 兼容性测试
```

---

## 六、风险点与缓解措施

### 6.1 技术风险

| 风险ID | 风险描述 | 可能性 | 影响 | 缓解措施 |
|--------|---------|--------|------|---------|
| R-001 | 参数量与预期偏差大 | 中 | 高 | 建立参数量计算公式，与参考实现对比 |
| R-002 | GPU 内存不足 | 中 | 中 | 提供内存占用估算，支持梯度检查点 |
| R-003 | 前向传播数值不稳定 | 低 | 高 | 添加数值范围检查，使用 LayerNorm |
| R-004 | 配置参数冲突 | 中 | 中 | 实现配置验证器，提供清晰错误信息 |
| R-005 | 预训练权重不兼容 | 高 | 高 | 实现权重转换脚本，添加兼容性检查 |

### 6.2 实现风险

| 风险ID | 风险描述 | 可能性 | 影响 | 缓解措施 |
|--------|---------|--------|------|---------|
| R-006 | 接口设计不合理 | 中 | 中 | 早期原型评审，参考成熟框架设计 |
| R-007 | 测试覆盖不足 | 中 | 中 | 使用测试覆盖率工具，设定最低阈值 |
| R-008 | 文档不完整 | 高 | 低 | 文档与代码同步更新，CI 检查文档 |
| R-009 | 代码风格不一致 | 低 | 低 | 配置自动格式化工具，Code Review |

### 6.3 外部依赖风险

| 风险ID | 风险描述 | 可能性 | 影响 | 缓解措施 |
|--------|---------|--------|------|---------|
| R-010 | PyTorch 版本不兼容 | 低 | 高 | 明确支持的 PyTorch 版本范围 |
| R-011 | CUDA 版本问题 | 中 | 中 | 测试多个 CUDA 版本兼容性 |
| R-012 | 第三方库更新 | 低 | 低 | 锁定依赖版本，定期更新 |

### 6.4 风险应对计划

```
高优先级风险应对：
├── R-005 预训练权重不兼容
│   ├── 实现权重转换工具
│   ├── 添加权重验证脚本
│   └── 提供详细迁移文档
├── R-001 参数量偏差
│   ├── 建立计算公式
│   ├── 与官方实现对比
│   └── 设置合理容差范围
└── R-003 数值不稳定
    ├── 添加数值检查断言
    ├── 使用混合精度训练
    └── 实现梯度裁剪

中优先级风险监控：
├── R-002 GPU 内存
│   └── 监控内存使用峰值
├── R-004 配置冲突
│   └── 实现配置验证器
└── R-006 接口设计
    └── 早期原型验证
```

---

## 七、验证执行记录

### 7.1 执行计划

| 阶段 | 执行时间 | 执行内容 | 负责人 | 状态 |
|------|---------|---------|--------|------|
| 阶段1 | - | 单元测试执行 | - | 待执行 |
| 阶段2 | - | 集成测试执行 | - | 待执行 |
| 阶段3 | - | 性能基准测试 | - | 待执行 |
| 阶段4 | - | 文档审查 | - | 待执行 |

### 7.2 测试结果汇总

| 测试类别 | 总数 | 通过 | 失败 | 跳过 | 通过率 |
|---------|------|------|------|------|--------|
| 模型创建测试 | 3 | - | - | - | - |
| 参数量验证 | 3 | - | - | - | - |
| 前向传播测试 | 3 | - | - | - | - |
| 配置文件测试 | 2 | - | - | - | - |
| 统一接口测试 | 3 | - | - | - | - |
| **总计** | **14** | **-** | **-** | **-** | **-** |

### 7.3 验证结论

| 项目 | 状态 |
|------|------|
| 所有必须标准满足 | ⬜ 待验证 |
| 所有测试用例通过 | ⬜ 待验证 |
| 无高风险问题遗留 | ⬜ 待验证 |
| 文档完整更新 | ⬜ 待验证 |
| **最终验证状态** | **⬜ 待验证** |

---

## 八、附录

### 8.1 参考文档

- ERes2NetV2 论文: [待补充]
- PyTorch 模型定义最佳实践: https://pytorch.org/docs/stable/notes/modules.html
- timm 库参考实现: https://github.com/huggingface/pytorch-image-models

### 8.2 相关文件路径

```
model/
├── eres2netv2.py          # 模型定义
├── config.py              # 配置管理
└── __init__.py            # 统一接口

tests/
├── test_model_variants.py # 单元测试
└── test_model_integration.py # 集成测试

scripts/
├── verify_model_variants.sh # 验证脚本
├── report_model_params.py   # 参数报告
└── analyze_memory.py        # 内存分析
```

### 8.3 修改历史

| 版本 | 日期 | 修改内容 | 作者 |
|------|------|---------|------|
| v1.0 | 2026/06/02 | 初始版本 | - |

---

> **备注：** 本验证计划应在实现开始前完成评审，测试用例应与代码实现同步开发。所有"待填写"字段需在验证执行时填写实际结果。