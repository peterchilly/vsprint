# 辅助模块验证测试计划

> **文档版本:** v1.0
> **创建日期:** 2026/06/03
> **关联任务:** 阶段三：模型构建 - 辅助模块

---

## 一、验证目标

### 1.1 总体目标
验证 ERes2NetV2 训练流程中所有辅助模块功能的正确性、可用性和性能表现，确保：
- 混合精度训练功能完整且稳定
- FLOPs 计算工具输出准确
- 模型可视化工具正常工作
- ONNX 导出流程完整可用

### 1.2 具体验证目标

| 模块 | 验证目标 | 优先级 |
|------|----------|--------|
| AMP | 验证混合精度训练可正常启用，模型收敛正常，显存占用优化有效 | P0 |
| FLOPs | 验证 FLOPs 和参数量计算准确，支持多输入尺寸 | P0 |
| 可视化 | 验证网络结构图可正确生成，节点信息完整 | P1 |
| ONNX | 验证 ONNX 导出成功，导出模型可推理，结果与 PyTorch 一致 | P0 |

---

## 二、测试用例

### 2.1 混合精度训练（AMP）

#### TC-AMP-001: AMP 启用验证
| 项目 | 内容 |
|------|------|
| **测试名称** | AMP 基本功能启用 |
| **前置条件** | 安装 PyTorch ≥ 1.6，CUDA ≥ 10.0，GPU 支持 FP16 |
| **测试步骤** | 1. 创建 ERes2NetV2 模型实例<br>2. 配置 GradScaler 和 autocast<br>3. 执行单次前向传播<br>4. 执行单次反向传播<br>5. 检查梯度类型和优化器状态 |
| **输入数据** | 随机生成 batch_size=2, input_shape=(3, 224, 224) |
| **预期结果** | - autocast 正确启用无报错<br>- 前向传播输出类型为 float16<br>- 梯度类型正确转换<br>- 无数值溢出警告 |
| **验证命令** | `pytest tests/test_amp.py::test_amp_enable -v` |

#### TC-AMP-002: AMP 训练收敛性验证
| 项目 | 内容 |
|------|------|
| **测试名称** | AMP 训练收敛对比 |
| **前置条件** | 准备小型数据集（如 CIFAR-10 子集，1000 样本） |
| **测试步骤** | 1. 使用 FP32 训练 10 个 epoch，记录 loss 曲线<br>2. 使用 AMP 训练相同配置 10 个 epoch<br>3. 对比两者的 loss 曲线和最终精度 |
| **输入数据** | CIFAR-10 子集，batch_size=32 |
| **预期结果** | - AMP 训练 loss 曲线与 FP32 趋势一致<br>- 最终精度差异 < 1%<br>- 无 NaN/Inf 出现 |
| **验证命令** | `pytest tests/test_amp.py::test_amp_convergence -v --slow` |

#### TC-AMP-003: AMP 显存优化验证
| 项目 | 内容 |
|------|------|
| **测试名称** | AMP 显存占用对比 |
| **前置条件** | CUDA 环境，nvidia-smi 可用 |
| **测试步骤** | 1. 记录 FP32 训练峰值显存<br>2. 记录 AMP 训练峰值显存<br>3. 计算显存节省比例 |
| **输入数据** | batch_size=64, input_shape=(3, 224, 224) |
| **预期结果** | - AMP 显存占用 < FP32 的 60%<br>- 无 OOM 错误 |
| **验证命令** | `pytest tests/test_amp.py::test_amp_memory -v` |

#### TC-AMP-004: AMP 动态 loss scale 验证
| 项目 | 内容 |
|------|------|
| **测试名称** | GradScaler 动态调整 |
| **前置条件** | AMP 环境配置完成 |
| **测试步骤** | 1. 启用 GradScaler<br>2. 监控 scale 值变化<br>3. 触发梯度溢出场景<br>4. 验证 scale 自动调整 |
| **输入数据** | 包含极端值的模拟数据 |
| **预期结果** | - 初始 scale = 65536<br>- 检测到溢出时 scale 自动降低<br>- 训练可继续进行 |
| **验证命令** | `pytest tests/test_amp.py::test_grad_scaler -v` |

---

### 2.2 FLOPs 计算

#### TC-FLOP-001: 基础 FLOPs 计算
| 项目 | 内容 |
|------|------|
| **测试名称** | FLOPs 计算准确性 |
| **前置条件** | 安装 thop 或 fvcore |
| **测试步骤** | 1. 加载各 ERes2NetV2 变体<br>2. 计算标准输入尺寸 (224x224) 的 FLOPs<br>3. 与官方数值对比 |
| **输入数据** | input_shape=(1, 3, 224, 224) |
| **预期结果** | - ERes2NetV2_m: FLOPs ≈ 4.5G<br>- ERes2NetV2_l: FLOPs ≈ 6.8G<br>- 误差 < 5% |
| **验证命令** | `pytest tests/test_flops.py::test_flops_calculation -v` |

#### TC-FLOP-002: 多输入尺寸验证
| 项目 | 内容 |
|------|------|
| **测试名称** | 不同输入尺寸 FLOPs |
| **前置条件** | thop/fvcore 正常工作 |
| **测试步骤** | 1. 测试输入尺寸: 112x112, 224x224, 448x448<br>2. 记录各尺寸 FLOPs<br>3. 验证 FLOPs 与尺寸平方成正比 |
| **输入数据** | 多种尺寸输入 |
| **预期结果** | - FLOPs(448) / FLOPs(224) ≈ 4<br>- FLOPs(224) / FLOPs(112) ≈ 4 |
| **验证命令** | `pytest tests/test_flops.py::test_flops_multi_size -v` |

#### TC-FLOP-003: 参数量统计
| 项目 | 内容 |
|------|------|
| **测试名称** | 模型参数量统计 |
| **前置条件** | 模型加载成功 |
| **测试步骤** | 1. 计算各变体参数量<br>2. 与官方数值对比<br>3. 验证可训练参数统计 |
| **输入数据** | 各 ERes2NetV2 变体 |
| **预期结果** | - ERes2NetV2_s: ≈ 8.5M<br>- ERes2NetV2_m: ≈ 14M<br>- ERes2NetV2_l: ≈ 22M<br>- 误差 < 2% |
| **验证命令** | `pytest tests/test_flops.py::test_params_count -v` |

#### TC-FLOP-004: 详细层统计
| 项目 | 内容 |
|------|------|
| **测试名称** | 各层 FLOPs 明细 |
| **前置条件** | thop 支持 verbose 输出 |
| **测试步骤** | 1. 启用详细模式<br>2. 记录各层 FLOPs<br>3. 验证主要计算层占比 |
| **输入数据** | 标准输入 |
| **预期结果** | - 输出包含每层 FLOPs<br>- 卷积层占比 > 90%<br>- 可识别计算密集层 |
| **验证命令** | `pytest tests/test_flops.py::test_layer_details -v` |

---

### 2.3 模型可视化

#### TC-VIZ-001: TorchView 结构图生成
| 项目 | 内容 |
|------|------|
| **测试名称** | TorchView 结构图 |
| **前置条件** | 安装 torchview, graphviz |
| **测试步骤** | 1. 创建 ERes2NetV2 模型<br>2. 使用 torchview 生成结构图<br>3. 保存为 PNG/SVG |
| **输入数据** | input_shape=(1, 3, 224, 224) |
| **预期结果** | - 成功生成图片文件<br>- 包含所有模块层级<br>- 输入输出形状正确标注<br>- 文件大小 > 0 |
| **验证命令** | `pytest tests/test_visualization.py::test_torchview -v` |

#### TC-VIZ-002: Netron 交互式可视化
| 项目 | 内容 |
|------|------|
| **测试名称** | Netron 在线可视化 |
| **前置条件** | 安装 netron, 完成 ONNX 导出 |
| **测试步骤** | 1. 导出 ONNX 模型<br>2. 使用 netron 打开<br>3. 验证节点信息 |
| **输入数据** | 已导出的 ONNX 文件 |
| **预期结果** | - Netron 正确加载模型<br>- 显示完整网络结构<br>- 可查看各层参数 |
| **验证命令** | `pytest tests/test_visualization.py::test_netron -v` |

#### TC-VIZ-003: 模型摘要输出
| 项目 | 内容 |
|------|------|
| **测试名称** | Model Summary |
| **前置条件** | 安装 torchsummary 或 torchinfo |
| **测试步骤** | 1. 打印模型摘要<br>2. 检查各层输出形状<br>3. 验证参数统计 |
| **输入数据** | input_shape=(1, 3, 224, 224) |
| **预期结果** | - 输出包含各层名称、形状、参数量<br>- 总参数量与计算一致<br>- 层级结构清晰 |
| **验证命令** | `pytest tests/test_visualization.py::test_model_summary -v` |

#### TC-VIZ-004: 异常输入处理
| 项目 | 内容 |
|------|------|
| **测试名称** | 可视化异常处理 |
| **前置条件** | 可视化工具正常 |
| **测试步骤** | 1. 测试极小输入 (1x1)<br>2. 测试超大输入 (1024x1024)<br>3. 测试动态 batch size |
| **输入数据** | 边界情况输入 |
| **预期结果** | - 极小输入正常处理或报错<br>- 超大输入合理处理（可截断/报错）<br>- 动态 batch 正确显示 |
| **验证命令** | `pytest tests/test_visualization.py::test_viz_edge_cases -v` |

---

### 2.4 ONNX 导出

#### TC-ONNX-001: 基础 ONNX 导出
| 项目 | 内容 |
|------|------|
| **测试名称** | ONNX 模型导出 |
| **前置条件** | 安装 onnx, onnxruntime, opset_version ≥ 11 |
| **测试步骤** | 1. 加载 ERes2NetV2 模型<br>2. 设置 eval 模式<br>3. 导出为 ONNX 格式<br>4. 验证 ONNX 模型完整性 |
| **输入数据** | dummy_input = torch.randn(1, 3, 224, 224) |
| **预期结果** | - 导出无错误<br>- .onnx 文件成功创建<br>- onnx.checker.check_model 通过 |
| **验证命令** | `pytest tests/test_onnx.py::test_onnx_export -v` |

#### TC-ONNX-002: ONNX 推理一致性
| 项目 | 内容 |
|------|------|
| **测试名称** | PyTorch vs ONNX 推理对比 |
| **前置条件** | ONNX 导出成功, onnxruntime 可用 |
| **测试步骤** | 1. PyTorch 模型推理<br>2. ONNX 模型推理相同输入<br>3. 对比输出结果 |
| **输入数据** | 随机输入 + 边界值输入 |
| **预期结果** | - 输出形状一致<br>- 数值差异 < 1e-4 (L2 范数)<br>- 最大绝对误差 < 1e-5 |
| **验证命令** | `pytest tests/test_onnx.py::test_onnx_inference -v` |

#### TC-ONNX-003: 动态输入尺寸
| 项目 | 内容 |
|------|------|
| **测试名称** | ONNX 动态维度 |
| **前置条件** | ONNX 导出基础功能正常 |
| **测试步骤** | 1. 导出时设置动态 batch 和尺寸<br>2. 使用不同尺寸推理<br>3. 验证结果正确 |
| **输入数据** | batch: [1, 4, 8], size: [112, 224, 448] |
| **预期结果** | - 动态维度正确导出<br>- 不同 batch/size 均可推理<br>- 结果与 PyTorch 一致 |
| **验证命令** | `pytest tests/test_onnx.py::test_onnx_dynamic -v` |

#### TC-ONNX-004: ONNX 优化验证
| 项目 | 内容 |
|------|------|
| **测试名称** | ONNX 优化器应用 |
| **前置条件** | 安装 onnx-simplifier 或 onnxoptimizer |
| **测试步骤** | 1. 导出原始 ONNX<br>2. 应用优化<br>3. 对比模型大小和推理速度 |
| **输入数据** | 标准模型 |
| **预期结果** | - 优化后模型更小<br>- 推理速度更快或持平<br>- 精度不变 |
| **验证命令** | `pytest tests/test_onnx.py::test_onnx_optimization -v` |

#### TC-ONNX-005: 多变体导出
| 项目 | 内容 |
|------|------|
| **测试名称** | 所有变体 ONNX 导出 |
| **前置条件** | 所有 ERes2NetV2 变体可用 |
| **测试步骤** | 1. 依次导出各变体<br>2. 验证各变体 ONNX<br>3. 对比推理结果 |
| **输入数据** | 各变体配置 |
| **预期结果** | - 所有变体成功导出<br>- 推理一致性通过<br>- 无 ops 不兼容错误 |
| **验证命令** | `pytest tests/test_onnx.py::test_all_variants -v` |

---

## 三、验证方法

### 3.1 自动化测试框架

```
tests/
├── test_amp.py           # AMP 相关测试
├── test_flops.py         # FLOPs 计算测试
├── test_visualization.py # 可视化测试
├── test_onnx.py          # ONNX 导出测试
├── conftest.py           # pytest fixtures
└── fixtures/
    └── sample_data.py    # 测试数据 fixtures
```

### 3.2 测试执行策略

| 测试类型 | 执行时机 | 执行方式 |
|----------|----------|----------|
| 单元测试 | 每次提交 | `pytest tests/ -v --tb=short` |
| 集成测试 | PR 合并 | `pytest tests/ -v --slow` |
| 性能测试 | 发版前 | 手动执行 + CI 标记 |
| 兼容性测试 | 周期性 | 不同 PyTorch/CUDA 版本 |

### 3.3 测试覆盖率要求

| 模块 | 行覆盖率 | 分支覆盖率 |
|------|----------|------------|
| AMP 工具 | ≥ 90% | ≥ 85% |
| FLOPs 工具 | ≥ 95% | ≥ 90% |
| 可视化工具 | ≥ 80% | ≥ 75% |
| ONNX 工具 | ≥ 90% | ≥ 85% |

```bash
# 覆盖率执行命令
pytest tests/ --cov=src/utils --cov-report=html --cov-report=term
```

---

## 四、通过标准

### 4.1 必须通过的测试

| 测试用例 | 通过标准 | 阻塞级别 |
|----------|----------|----------|
| TC-AMP-001 | AMP 正常启用，无报错 | 🔴 阻塞 |
| TC-AMP-002 | 精度损失 < 1% | 🔴 阻塞 |
| TC-FLOP-001 | 误差 < 5% | 🔴 阻塞 |
| TC-FLOP-003 | 参数量误差 < 2% | 🔴 阻塞 |
| TC-VIZ-001 | 图片文件生成成功 | 🟡 非阻塞 |
| TC-ONNX-001 | ONNX 文件生成成功 | 🔴 阻塞 |
| TC-ONNX-002 | 数值误差 < 1e-4 | 🔴 阻塞 |

### 4.2 性能基准

| 指标 | 基准值 | 通过条件 |
|------|--------|----------|
| AMP 显存节省 | ≥ 40% | 实测值 ≥ 基准值 |
| AMP 训练速度提升 | ≥ 30% | 实测值 ≥ 基准值 |
| FLOPs 计算耗时 | < 1s | 标准输入尺寸 |
| ONNX 导出耗时 | < 30s | 标准模型 |
| ONNX 推理延迟 | 与 PyTorch 差异 < 10% | 相同硬件 |

### 4.3 质量门禁

```yaml
# CI 质量门禁配置
quality_gates:
  - name: test_pass_rate
    threshold: 100%
    critical_tests:
      - TC-AMP-001
      - TC-AMP-002
      - TC-FLOP-001
      - TC-ONNX-001
      - TC-ONNX-002

  - name: code_coverage
    threshold: 85%

  - name: performance_regression
    baseline: main_branch
    max_degradation: 5%
```

---

## 五、自动化建议

### 5.1 CI/CD 流水线集成

```yaml
# .github/workflows/verify_auxiliary.yml
name: Auxiliary Modules Verification

on:
  pull_request:
    paths:
      - 'src/utils/amp/**'
      - 'src/utils/flops/**'
      - 'src/utils/visualization/**'
      - 'src/utils/onnx/**'
      - 'tests/test_amp.py'
      - 'tests/test_flops.py'
      - 'tests/test_visualization.py'
      - 'tests/test_onnx.py'

jobs:
  verify:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python: ['3.9', '3.10', '3.11']
        pytorch: ['1.12', '2.0', '2.1']

    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}

      - name: Install Dependencies
        run: |
          pip install torch==${{ matrix.pytorch }} --index-url https://download.pytorch.org/whl/cpu
          pip install -r requirements.txt
          pip install pytest pytest-cov thop fvcore onnx onnxruntime

      - name: Run Tests
        run: |
          pytest tests/ -v \
            --cov=src/utils \
            --cov-report=xml \
            --junitxml=test-results.xml

      - name: Upload Coverage
        uses: codecov/codecov-action@v4
        with:
          files: ./coverage.xml

      - name: Publish Test Results
        uses: EnricoMi/publish-unit-test-result-action@v2
        if: always()
        with:
          files: test-results.xml
```

### 5.2 Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "Running auxiliary module tests..."

# 快速测试（跳过慢速测试）
pytest tests/test_amp.py tests/test_flops.py tests/test_onnx.py \
  -v -x --tb=short -m "not slow"

if [ $? -ne 0 ]; then
    echo "❌ Pre-commit tests failed. Please fix before committing."
    exit 1
fi

echo "✅ Pre-commit tests passed."
```

### 5.3 测试报告生成

```python
# scripts/generate_test_report.py
import pytest
from datetime import datetime

def generate_report():
    """生成测试报告"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "modules": {
            "amp": {"tests": 4, "passed": None},
            "flops": {"tests": 4, "passed": None},
            "visualization": {"tests": 4, "passed": None},
            "onnx": {"tests": 5, "passed": None}
        },
        "summary": {
            "total": 17,
            "passed": None,
            "failed": None,
            "skipped": None
        }
    }

    # 运行 pytest 并收集结果
    result = pytest.main([
        "tests/",
        "-v",
        "--json-report",
        "--json-report-file=test_report.json"
    ])

    return report
```

### 5.4 自动化验证脚本

```bash
#!/bin/bash
# scripts/verify_auxiliary.sh

set -e

echo "=========================================="
echo "    辅助模块自动化验证脚本"
echo "=========================================="

# 1. AMP 验证
echo -e "\n[1/4] 验证混合精度训练..."
python -c "
import torch
from src.utils.amp import setup_amp

model = torch.nn.Linear(10, 10)
scaler, enabled = setup_amp()
assert enabled or not torch.cuda.is_available(), 'AMP should be enabled with CUDA'
print('✅ AMP 验证通过')
"

# 2. FLOPs 验证
echo -e "\n[2/4] 验证 FLOPs 计算..."
python -c "
from src.models import ERes2NetV2_s
from src.utils.flops import calculate_flops

model = ERes2NetV2_s()
flops, params = calculate_flops(model, (1, 3, 224, 224))
assert flops > 0, 'FLOPs should be positive'
assert params > 0, 'Params should be positive'
print(f'✅ FLOPs: {flops/1e9:.2f}G, Params: {params/1e6:.2f}M')
"

# 3. 可视化验证
echo -e "\n[3/4] 验证模型可视化..."
python -c "
import os
from src.models import ERes2NetV2_s
from src.utils.visualization import visualize_model

model = ERes2NetV2_s()
output_path = 'test_model_graph.png'
visualize_model(model, (1, 3, 224, 224), output_path)
assert os.path.exists(output_path), 'Graph file should exist'
os.remove(output_path)
print('✅ 可视化验证通过')
"

# 4. ONNX 验证
echo -e "\n[4/4] 验证 ONNX 导出..."
python -c "
import os
import torch
import onnx
from src.models import ERes2NetV2_s
from src.utils.onnx_export import export_onnx

model = ERes2NetV2_s()
model.eval()
output_path = 'test_model.onnx'

dummy_input = torch.randn(1, 3, 224, 224)
export_onnx(model, dummy_input, output_path)
assert os.path.exists(output_path), 'ONNX file should exist'

# 验证 ONNX 模型
onnx_model = onnx.load(output_path)
onnx.checker.check_model(onnx_model)
os.remove(output_path)
print('✅ ONNX 导出验证通过')
"

echo -e "\n=========================================="
echo "    所有辅助模块验证通过！"
echo "=========================================="
```

---

## 六、风险点

### 6.1 技术风险

| 风险项 | 影响程度 | 发生概率 | 缓解措施 |
|--------|----------|----------|----------|
| **AMP 数值不稳定** | 高 | 中 | 动态 loss scaling；梯度裁剪；NaN 检测 |
| **FLOPs 计算不兼容** | 中 | 低 | 支持多后端（thop/fvcore）；自定义计算逻辑 |
| **ONNX 算子不支持** | 高 | 中 | opset 版本兼容性检查；自定义算子注册 |
| **可视化工具依赖冲突** | 低 | 中 | 隔离依赖；Docker 环境；可选安装 |
| **GPU 显存不足** | 中 | 中 | 梯度检查点；小 batch 测试；CPU 回退 |

### 6.2 环境风险

| 风险项 | 影响程度 | 发生概率 | 缓解措施 |
|--------|----------|----------|----------|
| **PyTorch 版本不兼容** | 高 | 中 | 支持 PyTorch 1.12+；版本检测提示 |
| **CUDA 版本不匹配** | 高 | 低 | 文档明确版本要求；自动检测 |
| **Python 版本差异** | 中 | 低 | 支持 Python 3.8-3.11；CI 矩阵测试 |
| **依赖包冲突** | 中 | 中 | 锁定依赖版本；虚拟环境；requirements.txt |

### 6.3 兼容性风险

| 风险项 | 影响程度 | 发生概率 | 缓解措施 |
|--------|----------|----------|----------|
| **不同 GPU 架构** | 中 | 高 | 多 GPU 测试；架构感知代码 |
| **ONNX Runtime 版本** | 中 | 中 | 指定最低版本；兼容性测试 |
| **操作系统差异** | 低 | 低 | 跨平台 CI；路径处理规范化 |

### 6.4 风险监控

```python
# src/utils/risk_monitor.py
import warnings
from functools import wraps

def check_amp_compatibility(func):
    """AMP 兼容性检查装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        import torch
        if not torch.cuda.is_available():
            warnings.warn("CUDA not available, AMP will be disabled")
        elif not hasattr(torch.cuda, 'amp'):
            warnings.warn("PyTorch version does not support AMP")
        return func(*args, **kwargs)
    return wrapper

def check_onnx_compatibility():
    """检查 ONNX 兼容性"""
    try:
        import onnx
        import onnxruntime
        return True
    except ImportError as e:
        raise ImportError(
            f"ONNX dependencies not installed: {e}\n"
            "Install with: pip install onnx onnxruntime"
        )
```

---

## 七、测试数据准备

### 7.1 标准测试数据

```python
# tests/fixtures/sample_data.py
import pytest
import torch

@pytest.fixture
def sample_input_224():
    """标准 224x224 输入"""
    return torch.randn(1, 3, 224, 224)

@pytest.fixture
def sample_batch():
    """批量测试输入"""
    return torch.randn(8, 3, 224, 224)

@pytest.fixture
def sample_inputs_multi_size():
    """多尺寸测试输入"""
    return {
        'small': torch.randn(1, 3, 112, 112),
        'standard': torch.randn(1, 3, 224, 224),
        'large': torch.randn(1, 3, 448, 448)
    }

@pytest.fixture
def edge_case_inputs():
    """边界情况输入"""
    return {
        'zeros': torch.zeros(1, 3, 224, 224),
        'ones': torch.ones(1, 3, 224, 224),
        'extreme': torch.full((1, 3, 224, 224), 1e6),
        'negative': torch.randn(1, 3, 224, 224) - 10
    }
```

### 7.2 预期结果基线

```python
# tests/fixtures/baseline_results.py

# ERes2NetV2 各变体 FLOPs 基线 (input: 224x224)
FLOPS_BASELINE = {
    'ERes2NetV2_s': {'flops': 4.5e9, 'params': 8.5e6},
    'ERes2NetV2_m': {'flops': 6.8e9, 'params': 14e6},
    'ERes2NetV2_l': {'flops': 10.2e9, 'params': 22e6}
}

# ONNX 推理误差阈值
ONNX_TOLERANCE = {
    'l2_norm': 1e-4,
    'max_abs': 1e-5
}

# AMP 训练精度损失阈值
AMP_ACCURACY_TOLERANCE = 0.01  # 1%
```

---

## 八、验收检查清单

### 8.1 功能验收

- [ ] **AMP 模块**
  - [ ] AMP 可正常启用和禁用
  - [ ] 训练过程无 NaN/Inf
  - [ ] 精度损失 < 1%
  - [ ] 显存节省 ≥ 40%
  - [ ] 支持动态 loss scaling

- [ ] **FLOPs 模块**
  - [ ] FLOPs 计算准确（误差 < 5%）
  - [ ] 参数量统计准确（误差 < 2%）
  - [ ] 支持多输入尺寸
  - [ ] 支持各 ERes2NetV2 变体
  - [ ] 提供层级别统计

- [ ] **可视化模块**
  - [ ] TorchView 结构图生成成功
  - [ ] Model Summary 输出正确
  - [ ] 支持多种输出格式
  - [ ] 异常输入正确处理

- [ ] **ONNX 模块**
  - [ ] ONNX 导出成功
  - [ ] ONNX 模型验证通过
  - [ ] 推理结果与 PyTorch 一致
  - [ ] 支持动态输入尺寸
  - [ ] 所有变体均可导出

### 8.2 质量验收

- [ ] 代码覆盖率 ≥ 85%
- [ ] 所有测试通过
- [ ] 无 P0 级 Bug
- [ ] 文档完整
- [ ] CI 流水线通过

### 8.3 文档验收

- [ ] 使用文档完整
- [ ] API 文档清晰
- [ ] 示例代码可运行
- [ ] 错误信息有意义

---

## 九、附录

### A. 测试命令速查

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定模块测试
pytest tests/test_amp.py -v
pytest tests/test_flops.py -v
pytest tests/test_visualization.py -v
pytest tests/test_onnx.py -v

# 运行并生成覆盖率
pytest tests/ --cov=src/utils --cov-report=html

# 运行慢速测试
pytest tests/ -v --slow

# 并行运行
pytest tests/ -v -n auto
```

### B. 依赖版本要求

```txt
# requirements-test.txt
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-xdist>=3.0.0

# AMP
torch>=1.12.0

# FLOPs
thop>=0.1.1
fvcore>=0.1.5

# 可视化
torchview>=0.2.0
graphviz>=0.20.0
netron>=6.0.0

# ONNX
onnx>=1.12.0
onnxruntime>=1.12.0
onnx-simplifier>=0.4.0
```

### C. 参考资源

- [PyTorch AMP 文档](https://pytorch.org/docs/stable/amp.html)
- [thop GitHub](https://github.com/Lyken17/pytorch-OpCounter)
- [fvcore FLOPs](https://detectron2.readthedocs.io/en/latest/modules/fvcore.html)
- [TorchView 文档](https://github.com/karpathy/pytorchviz)
- [ONNX 官方文档](https://onnx.ai/get-started.html)