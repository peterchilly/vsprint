# 标准评估指标验证测试计划

> **文档类型：** 验证测试计划
> **关联任务：** 阶段五 - 标准评估指标
> **创建日期：** 2026-06-03
> **验证阶段：** 模型评估与测试

---

## 1. 验证目标

### 1.1 主要目标
验证模型评估系统的正确性、完整性和可靠性，确保：
1. **准确性**：所有评估指标计算结果正确，与理论公式一致
2. **完整性**：覆盖所有规定的评估指标和可视化输出
3. **可复现性**：评估结果可重复，不受随机因素影响
4. **可解释性**：评估报告清晰易懂，支持模型优化决策

### 1.2 验证范围
| 指标类别 | 具体指标 | 优先级 |
|---------|---------|--------|
| 整体准确率 | Top-1 准确率、Top-5 准确率 | P0 |
| 分类性能 | 每类别精确率、召回率、F1分数 | P0 |
| 可视化 | 混淆矩阵、PR曲线、ROC曲线 | P1 |
| 报告输出 | 评估报告、错误分析 | P1 |

---

## 2. 测试用例

### 2.1 Top-1/Top-5 准确率测试

| 用例ID | 测试场景 | 输入数据 | 预期结果 | 验证方法 |
|--------|---------|---------|---------|---------|
| TC-ACC-001 | 标准测试集评估 | 完整测试集（含ground truth） | 准确率在合理范围内（参考基线模型） | 与手工计算对比 |
| TC-ACC-002 | 小规模数据验证 | 100样本已知标签数据集 | 准确率 = 正确预测数/总样本数 | 手工验证计算正确性 |
| TC-ACC-003 | 边界条件-全正确 | 模型完美预测场景 | Top-1=100%, Top-5=100% | 边界值测试 |
| TC-ACC-004 | 边界条件-全错误 | 模型完全错误场景 | Top-1=0%, Top-5取决于类别数 | 边界值测试 |
| TC-ACC-005 | Top-5 vs Top-1关系 | 多类别分类数据 | Top-5 ≥ Top-1 准确率 | 逻辑验证 |
| TC-ACC-006 | 类别不平衡数据 | 不平衡测试集 | 准确率计算不受类别分布影响 | 加权验证 |
| TC-ACC-007 | 空测试集处理 | 空数据集输入 | 返回错误或NaN，不崩溃 | 异常处理测试 |

### 2.2 每类别精确率/召回率/F1测试

| 用例ID | 测试场景 | 输入数据 | 预期结果 | 验证方法 |
|--------|---------|---------|---------|---------|
| TC-PRF-001 | 标准多类别评估 | 完整测试集 | 每类别输出P/R/F1三个值 | 输出格式验证 |
| TC-PRF-002 | 精确率计算验证 | 已知混淆矩阵 | Precision = TP/(TP+FP) | 公式验证 |
| TC-PRF-003 | 召回率计算验证 | 已知混淆矩阵 | Recall = TP/(TP+FN) | 公式验证 |
| TC-PRF-004 | F1计算验证 | 已知P和R值 | F1 = 2*P*R/(P+R) | 公式验证 |
| TC-PRF-005 | 零样本类别处理 | 类别在测试集中无样本 | 返回0或NaN，记录警告 | 异常处理 |
| TC-PRF-006 | 完美分类场景 | 每类别完美预测 | 所有类别P=R=F1=1.0 | 边界值测试 |
| TC-PRF-007 | 宏平均vs微平均 | 不平衡数据集 | 宏平均和微平均结果不同 | 平均方式验证 |
| TC-PRF-008 | 加权F1计算 | 多类别不平衡数据 | Weighted F1正确计算 | 公式验证 |

### 2.3 混淆矩阵测试

| 用例ID | 测试场景 | 输入数据 | 预期结果 | 验证方法 |
|--------|---------|---------|---------|---------|
| TC-CM-001 | 标准混淆矩阵生成 | 完整测试集预测结果 | NxN矩阵，N=类别数 | 维度验证 |
| TC-CM-002 | 对角线元素验证 | 已知预测结果 | 对角线=正确预测数 | 手工验证 |
| TC-CM-003 | 行列和验证 | 完整测试集 | 每行和=该类真实样本数，每列和=预测为该类样本数 | 数学验证 |
| TC-CM-004 | 归一化混淆矩阵 | 混淆矩阵 + 归一化参数 | 每行和为1（按真实类别归一化） | 归一化验证 |
| TC-CM-005 | 可视化输出 | 混淆矩阵数据 | 生成清晰图像，标签完整 | 人工检查 |
| TC-CM-006 | 大类别数处理 | 100+类别 | 矩阵可读性保持，支持缩放/裁剪 | 可用性测试 |
| TC-CM-007 | 类别名称显示 | 带类别名称的数据集 | 类别名称正确显示 | 标签验证 |

### 2.4 PR/ROC曲线测试（二分类）

| 用例ID | 测试场景 | 输入数据 | 预期结果 | 验证方法 |
|--------|---------|---------|---------|---------|
| TC-ROC-001 | 标准ROC曲线 | 二分类预测概率 | 曲线从(0,0)到(1,1) | 形状验证 |
| TC-ROC-002 | AUC计算验证 | 已知预测分布 | AUC值在[0,1]区间 | 范围验证 |
| TC-ROC-003 | 完美分类器AUC | 完美预测概率 | AUC=1.0 | 边界值测试 |
| TC-ROC-004 | 随机猜测AUC | 随机预测概率 | AUC≈0.5 | 基线测试 |
| TC-ROC-005 | PR曲线生成 | 二分类预测概率 | 曲线形状符合Precision-Recall关系 | 形状验证 |
| TC-ROC-006 | 多类别处理 | 多类别数据集 | 报错或提供One-vs-Rest选项 | 异常处理 |
| TC-ROC-007 | 阈值变化响应 | 不同决策阈值 | 曲线正确反映阈值变化 | 阈值扫描 |

### 2.5 端到端集成测试

| 用例ID | 测试场景 | 测试步骤 | 预期结果 |
|--------|---------|---------|---------|
| TC-E2E-001 | 完整评估流程 | 1. 加载模型 2. 加载测试集 3. 执行评估 4. 生成报告 | 所有指标正确输出，无报错 |
| TC-E2E-002 | 报告完整性 | 执行完整评估 | 报告包含所有必填字段 |
| TC-E2E-003 | 结果持久化 | 执行评估并保存 | 结果可加载，与原始结果一致 |
| TC-E2E-004 | 多次运行一致性 | 同一模型/数据运行3次 | 结果完全一致（确定性模式） |

---

## 3. 验证方法

### 3.1 单元测试验证

```python
# 测试框架示例
import pytest
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class TestAccuracyMetrics:
    """准确率指标单元测试"""

    def test_top1_accuracy_perfect(self):
        """TC-ACC-003: 完美预测场景"""
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4])
        assert calculate_top1_accuracy(y_true, y_pred) == 1.0

    def test_top1_accuracy_zero(self):
        """TC-ACC-004: 全错误场景"""
        y_true = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 1])
        assert calculate_top1_accuracy(y_true, y_pred) == 0.0

    def test_top5_geq_top1(self):
        """TC-ACC-005: Top-5 >= Top-1"""
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred_probs = np.random.rand(5, 10)  # 5样本，10类别
        top1 = calculate_top1_accuracy(y_true, y_pred_probs)
        top5 = calculate_top5_accuracy(y_true, y_pred_probs)
        assert top5 >= top1
```

### 3.2 对比验证

| 验证项 | 实现方法 | 验证标准 |
|--------|---------|---------|
| 准确率计算 | 与sklearn accuracy_score对比 | 差异 < 1e-10 |
| P/R/F1计算 | 与sklearn precision_recall_fscore_support对比 | 差异 < 1e-10 |
| 混淆矩阵 | 与sklearn confusion_matrix对比 | 完全一致 |
| AUC计算 | 与sklearn roc_auc_score对比 | 差异 < 1e-6 |

### 3.3 可视化验证

```python
# 可视化检查清单
visualization_checklist = {
    "confusion_matrix": [
        "标题正确显示",
        "坐标轴标签清晰",
        "颜色映射合理",
        "数值可读",
        "类别名称正确"
    ],
    "roc_curve": [
        "曲线从(0,0)到(1,1)",
        "AUC值显示在图例中",
        "对角参考线存在",
        "坐标轴标签正确",
        "图像分辨率足够"
    ],
    "pr_curve": [
        "曲线形状合理",
        "基线显示",
        "坐标轴标签正确",
        "图例完整"
    ]
}
```

### 3.4 性能验证

| 指标 | 测试方法 | 通过标准 |
|-----|---------|---------|
| 评估时间 | 计时测试集评估 | < 10分钟/10000样本（取决于硬件） |
| 内存使用 | 监控峰值内存 | < 16GB（可调整） |
| GPU利用 | 监控GPU使用率 | > 80%（批量推理时） |

---

## 4. 通过标准

### 4.1 功能通过标准

| 指标类型 | 通过标准 | 验证方法 |
|---------|---------|---------|
| Top-1/Top-5准确率 | 计算正确，差异<1e-10 vs sklearn | 自动化测试 |
| 每类别P/R/F1 | 计算正确，支持宏平均/微平均/加权平均 | 自动化测试 |
| 混淆矩阵 | 矩阵维度正确，元素和=样本总数，可视化清晰 | 自动化+人工检查 |
| PR/ROC曲线 | 曲线形状正确，AUC值准确，图像可读 | 自动化+人工检查 |

### 4.2 质量通过标准

| 质量维度 | 标准 | 验证方法 |
|---------|-----|---------|
| 代码覆盖率 | > 90% | pytest-cov |
| 文档完整性 | 所有公开函数有docstring | 人工检查 |
| 类型提示 | 所有函数有类型注解 | mypy检查 |
| 代码风格 | 符合PEP8/项目规范 | ruff/black检查 |

### 4.3 验收清单

- [ ] 所有自动化测试通过（覆盖率>90%）
- [ ] 与sklearn对比验证通过
- [ ] 混淆矩阵可视化正确且美观
- [ ] PR/ROC曲线可视化正确且美观
- [ ] 评估报告格式符合模板要求
- [ ] 性能在可接受范围内
- [ ] 文档完整

---

## 5. 自动化建议

### 5.1 测试自动化框架

```yaml
# .github/workflows/evaluation_tests.yml
name: Evaluation Metrics Tests

on:
  push:
    paths: ['src/evaluation/**', 'tests/evaluation/**']
  pull_request:
    paths: ['src/evaluation/**', 'tests/evaluation/**']

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements-test.txt
      - name: Run unit tests
        run: pytest tests/evaluation/ -v --cov=src/evaluation --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v4

  integration-tests:
    runs-on: [self-hosted, gpu]
    needs: unit-tests
    steps:
      - name: Run integration tests
        run: pytest tests/evaluation/integration/ -v
```

### 5.2 持续验证脚本

```python
# scripts/verify_evaluation.py
"""
评估模块持续验证脚本
运行所有验证并生成报告
"""

import subprocess
import json
from datetime import datetime
from pathlib import Path

def run_verification():
    results = {
        "timestamp": datetime.now().isoformat(),
        "tests": {}
    }

    # 1. 运行单元测试
    result = subprocess.run(
        ["pytest", "tests/evaluation/", "-v", "--tb=short"],
        capture_output=True, text=True
    )
    results["tests"]["unit"] = {
        "passed": result.returncode == 0,
        "output": result.stdout
    }

    # 2. 运行对比验证
    result = subprocess.run(
        ["python", "tests/evaluation/verify_against_sklearn.py"],
        capture_output=True, text=True
    )
    results["tests"]["sklearn_comparison"] = {
        "passed": result.returncode == 0,
        "output": result.stdout
    }

    # 3. 生成可视化并检查
    result = subprocess.run(
        ["python", "tests/evaluation/check_visualizations.py"],
        capture_output=True, text=True
    )
    results["tests"]["visualization"] = {
        "passed": result.returncode == 0,
        "output": result.stdout
    }

    # 保存结果
    output_path = Path("reports/verification_report.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return all(t["passed"] for t in results["tests"].values())

if __name__ == "__main__":
    success = run_verification()
    exit(0 if success else 1)
```

### 5.3 回归测试套件

```python
# tests/evaluation/test_regression.py
"""
回归测试：确保评估指标计算结果与历史基线一致
"""

import pytest
import json
from pathlib import Path

# 基线结果（从已知模型/数据集获得）
BASELINE_FILE = Path("tests/evaluation/baselines/metrics_baseline.json")

class TestRegression:
    @pytest.fixture
    def baseline(self):
        with open(BASELINE_FILE) as f:
            return json.load(f)

    def test_accuracy_regression(self, baseline):
        """准确率与基线一致"""
        current = run_evaluation()
        assert abs(current["accuracy"] - baseline["accuracy"]) < 1e-6

    def test_per_class_metrics_regression(self, baseline):
        """每类别指标与基线一致"""
        current = run_evaluation()
        for class_id, metrics in baseline["per_class"].items():
            for metric in ["precision", "recall", "f1"]:
                assert abs(
                    current["per_class"][class_id][metric] - metrics[metric]
                ) < 1e-6
```

### 5.4 可视化自动化检查

```python
# tests/evaluation/check_visualizations.py
"""
自动化检查可视化输出
"""

import matplotlib.pyplot as plt
from pathlib import Path

def check_confusion_matrix_image(image_path):
    """检查混淆矩阵图像"""
    img = plt.imread(image_path)

    checks = {
        "file_exists": Path(image_path).exists(),
        "valid_image": img is not None and img.size > 0,
        "reasonable_size": img.shape[0] > 100 and img.shape[1] > 100,
    }

    return all(checks.values()), checks

def check_roc_curve_image(image_path):
    """检查ROC曲线图像"""
    img = plt.imread(image_path)

    checks = {
        "file_exists": Path(image_path).exists(),
        "valid_image": img is not None and img.size > 0,
    }

    return all(checks.values()), checks

def main():
    results = {}

    cm_path = "outputs/visualizations/confusion_matrix.png"
    passed, details = check_confusion_matrix_image(cm_path)
    results["confusion_matrix"] = {"passed": passed, "details": details}

    roc_path = "outputs/visualizations/roc_curve.png"
    passed, details = check_roc_curve_image(roc_path)
    results["roc_curve"] = {"passed": passed, "details": details}

    print(json.dumps(results, indent=2))
    return all(r["passed"] for r in results.values())

if __name__ == "__main__":
    import json
    success = main()
    exit(0 if success else 1)
```

---

## 6. 风险点

### 6.1 技术风险

| 风险ID | 风险描述 | 可能性 | 影响 | 缓解措施 |
|--------|---------|--------|-----|---------|
| R-TECH-001 | GPU内存不足导致评估失败 | 中 | 高 | 实现批量推理，支持CPU回退 |
| R-TECH-002 | 浮点精度差异导致指标不一致 | 低 | 中 | 使用一致的数值类型，设置容差 |
| R-TECH-003 | 类别不平衡导致指标误导 | 高 | 高 | 补充加权指标，提供类别分布报告 |
| R-TECH-004 | 大规模数据集评估时间过长 | 中 | 中 | 支持采样评估，提供进度条 |
| R-TECH-005 | 可视化在无GUI环境失败 | 中 | 低 | 支持Agg后端，仅保存文件 |

### 6.2 数据风险

| 风险ID | 风险描述 | 可能性 | 影响 | 缓解措施 |
|--------|---------|--------|-----|---------|
| R-DATA-001 | 测试集标签错误 | 低 | 高 | 交叉验证测试集完整性 |
| R-DATA-002 | 数据加载器配置错误 | 中 | 高 | 添加数据验证步骤 |
| R-DATA-003 | 预处理与训练不一致 | 中 | 高 | 使用相同预处理管道 |
| R-DATA-004 | 测试集泄露到训练集 | 低 | 极高 | 严格数据集划分检查 |

### 6.3 流程风险

| 风险ID | 风险描述 | 可能性 | 影响 | 缓解措施 |
|--------|---------|--------|-----|---------|
| R-PROC-001 | 评估脚本版本混乱 | 中 | 中 | 使用版本控制，记录commit hash |
| R-PROC-002 | 随机种子未固定 | 高 | 中 | 固定所有随机种子 |
| R-PROC-003 | 评估结果未及时保存 | 中 | 高 | 实现自动保存和检查点 |
| R-PROC-004 | 报告解读错误 | 中 | 中 | 提供指标解释文档 |

### 6.4 风险优先级矩阵

```
                    影响程度
              低         中         高
         ┌─────────┬─────────┬─────────┐
    高   │R-PROC-002│R-TECH-004│R-TECH-003│
可       │         │         │R-DATA-003│
能       ├─────────┼─────────┼─────────┤
性       中   │         │R-TECH-001│R-TECH-002│
         │   │         │R-PROC-001│R-DATA-002│
         │   │         │R-PROC-004│         │
         ├─────────┼─────────┼─────────┤
    低   │R-TECH-005│         │R-DATA-001│
         │         │         │R-DATA-004│
         └─────────┴─────────┴─────────┘
```

---

## 7. 验证执行计划

### 7.1 验证阶段

| 阶段 | 活动 | 负责人 | 时间 | 交付物 |
|-----|-----|-------|-----|-------|
| 阶段1 | 单元测试开发 | 开发人员 | Day 1-2 | 测试代码 |
| 阶段2 | 集成测试开发 | 开发人员 | Day 2-3 | 集成测试套件 |
| 阶段3 | 自动化框架搭建 | DevOps | Day 3-4 | CI/CD配置 |
| 阶段4 | 执行验证 | QA | Day 4-5 | 测试报告 |
| 阶段5 | 问题修复 | 开发人员 | Day 5-6 | 修复记录 |
| 阶段6 | 回归验证 | QA | Day 6-7 | 最终报告 |

### 7.2 验证入口条件
- [ ] 评估模块代码完成
- [ ] 测试数据准备完毕
- [ ] 测试环境配置完成

### 7.3 验证出口条件
- [ ] 所有测试用例通过
- [ ] 代码覆盖率达标
- [ ] 无严重缺陷遗留
- [ ] 验证报告完成

---

## 8. 附录

### 8.1 测试数据要求

| 数据类型 | 规模 | 用途 | 来源 |
|---------|-----|-----|-----|
| 小规模验证集 | 100-500样本 | 快速验证 | 测试集子集 |
| 标准测试集 | 完整测试集 | 正式评估 | 项目数据集 |
| 边界测试集 | 特制数据 | 边界条件 | 人工构造 |
| 对比基准集 | 已知结果数据 | sklearn对比 | 公开数据集 |

### 8.2 参考文档

- [sklearn.metrics文档](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)
- [PyTorch评估指南](https://pytorch.org/tutorials/beginner/introyt/trainingyt.html#the-training-loop)
- 项目评估模块设计文档

### 8.3 术语表

| 术语 | 定义 |
|-----|-----|
| Top-1准确率 | 预测概率最高的类别与真实标签一致的样本比例 |
| Top-5准确率 | 真实标签在预测概率最高的前5个类别中的样本比例 |
| 精确率(Precision) | TP / (TP + FP)，预测为正的样本中真正为正的比例 |
| 召回率(Recall) | TP / (TP + FN)，真正为正的样本被正确预测的比例 |
| F1分数 | 精确率和召回率的调和平均 |
| 混淆矩阵 | 展示预测类别与真实类别对应关系的矩阵 |
| AUC | ROC曲线下面积，衡量分类器整体性能 |

---

**文档版本：** 1.0
**最后更新：** 2026-06-03
**审核状态：** 待审核