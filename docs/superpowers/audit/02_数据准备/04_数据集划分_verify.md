# 数据集划分 — 验证测试设计

> **来源任务：** docs/superpowers/tasks/02_数据准备/04_数据集划分.md
> **生成日期：** 2026-06-02

---

## 1. 验证目标概述

本文档针对「数据集划分」任务设计完整的测试验证方案，确保以下验收标准得到严格验证：

| 验收标准 | 验证方法 | 验证类型 |
|---------|---------|---------|
| 训练/验证/测试集划分完成 | 单元测试 + 集成测试 | 自动化 |
| 各子集类别分布一致 | 统计检验 + 可视化对比 | 半自动 |
| 划分索引文件已保存 | 文件存在性检查 + 完整性验证 | 自动化 |

---

## 2. 测试用例设计

### 2.1 划分比例测试用例

#### TC-SPLIT-001: 基础比例划分验证 (8:1:1)

| 属性 | 值 |
|-----|-----|
| **测试ID** | TC-SPLIT-001 |
| **测试项** | 训练/验证/测试集 8:1:1 比例划分 |
| **前置条件** | 存在完整数据集（样本量 ≥ 100） |
| **输入数据** | 1000 个样本的数据集 |
| **测试步骤** | 1. 执行 8:1:1 划分<br>2. 统计各子集样本数<br>3. 计算实际比例<br>4. 验证比例偏差 |
| **预期结果** | 训练集约 800，验证集约 100，测试集约 100 |
| **通过标准** | 各子集比例偏差 < 1%（考虑取整误差），总数守恒 |

#### TC-SPLIT-002: 备选比例划分验证 (7:2:1)

| 属性 | 值 |
|-----|-----|
| **测试ID** | TC-SPLIT-002 |
| **测试项** | 训练/验证/测试集 7:2:1 比例划分 |
| **前置条件** | 存在完整数据集 |
| **输入数据** | 1000 个样本的数据集 |
| **测试步骤** | 1. 执行 7:2:1 划分<br>2. 统计各子集样本数<br>3. 验证训练:验证:测试 ≈ 700:200:100 |
| **预期结果** | 训练集约 700，验证集约 200，测试集约 100 |
| **通过标准** | 各子集比例偏差 < 1%，总数守恒 |

#### TC-SPLIT-003: 小数据集划分边界验证

| 属性 | 值 |
|-----|-----|
| **测试ID** | TC-SPLIT-003 |
| **测试项** | 小样本数据集划分 |
| **前置条件** | 样本量不足以精确按比例划分 |
| **输入数据** | 15 个样本的数据集 |
| **测试步骤** | 1. 执行 8:1:1 划分<br>2. 检查各子集至少有 1 个样本<br>3. 验证总数守恒<br>4. 检查是否有警告信息 |
| **预期结果** | 各子集均有样本，总数守恒，有适当警告 |
| **通过标准** | 所有子集非空，总数正确，边界情况有明确提示 |

#### TC-SPLIT-004: 大数据集划分性能验证

| 属性 | 值 |
|-----|-----|
| **测试ID** | TC-SPLIT-004 |
| **测试项** | 大规模数据集划分性能 |
| **前置条件** | 数据量 ≥ 100,000 样本 |
| **输入数据** | 100,000 个样本的数据集 |
| **测试步骤** | 1. 记录开始时间<br>2. 执行划分<br>3. 记录结束时间<br>4. 验证耗时在可接受范围 |
| **预期结果** | 划分耗时 < 10 秒 |
| **通过标准** | 划分完成且耗时合理，内存占用正常 |

---

### 2.2 分层抽样测试用例

#### TC-STRAT-001: 类别分布一致性验证

| 属性 | 值 |
|-----|-----|
| **测试ID** | TC-STRAT-001 |
| **测试项** | 分层抽样后类别分布一致性 |
| **前置条件** | 数据集有明确的类别标签 |
| **输入数据** | 含 3 个类别（比例 50:30:20）的 1000 样本数据集 |
| **测试步骤** | 1. 执行分层抽样划分<br>2. 统计各子集的类别分布<br>3. 计算各子集类别比例<br>4. 比较子集间类别分布差异 |
| **预期结果** | 各子集类别比例与原数据集一致（50:30:20） |
| **通过标准** | 各子集类别比例偏差 < 2%，卡方检验 p > 0.05 |

#### TC-STRAT-002: 稀有类别保留验证

| 属性 | 值 |
|-----|-----|
| **测试ID** | TC-STRAT-002 |
| **测试项** | 稀有类别在各子集中的保留 |
| **前置条件** | 存在样本量极少的类别 |
| **输入数据** | 含稀有类别（占比 < 1%）的数据集 |
| **测试步骤** | 1. 识别稀有类别样本<br>2. 执行分层抽样<br>3. 检查各子集是否包含稀有类别<br>4. 统计稀有类别分布 |
| **预期结果** | 稀有类别在各子集中均有代表 |
| **通过标准** | 所有稀有类别在训练集中至少出现一次，验证/测试集中尽可能有代表 |

#### TC-STRAT-003: 多标签分层抽样验证

| 属性 | 值 |
|-----|-----|
| **测试ID** | TC-STRAT-003 |
| **测试项** | 多标签场景的分层抽样 |
| **前置条件** | 样本有多个标签（如多标签分类） |
| **输入数据** | 多标签数据集（每个样本有 1-3 个标签） |
| **测试步骤** | 1. 执行多标签分层抽样（或近似方法）<br>2. 统计各标签在各子集中的分布<br>3. 验证标签分布一致性 |
| **预期结果** | 各标签在各子集中的分布与原数据集一致 |
| **通过标准** | 各标签分布偏差 < 3%，无标签在任一子集中缺失 |

#### TC-STRAT-004: 高度不平衡数据集验证

| 属性 | 值 |
|-----|-----|
| **测试ID** | TC-STRAT-004 |
| **测试项** | 极度不平衡数据集的分层抽样 |
| **前置条件** | 类别比例极度不平衡（如 95:3:1:1） |
| **输入数据** | 1000 样本，类别比例 [950, 30, 10, 10] |
| **测试步骤** | 1. 执行分层抽样<br>2. 检查小类别是否被正确分配<br>3. 验证训练/验证/测试集中小类别的存在性 |
| **预期结果** | 即使最小类别也在训练集中有代表 |
| **通过标准** | 训练集中所有类别均有样本，样本数 < 10 的类别有特殊处理策略 |

---

### 2.3 随机种子与可复现性测试用例

#### TC-SEED-001: 随机种子固定性验证

| 属性 | 值 |
|-----|-----|
| **测试ID** | TC-SEED-001 |
| **测试项** | 固定随机种子产生相同划分 |
| **前置条件** | 划分函数支持随机种子参数 |
| **输入数据** | 100 样本数据集 |
| **测试步骤** | 1. 使用 seed=42 执行划分<br>2. 记录划分结果（训练/验证/测试索引）<br>3. 使用相同 seed=42 再次划分<br>4. 比较两次划分结果 |
| **预期结果** | 两次划分的索引完全相同 |
| **通过标准** | 索引集合完全一致（逐元素比较） |

#### TC-SEED-002: 不同随机种子产生不同划分

| 属性 | 值 |
|-----|-----|
| **测试ID** | TC-SEED-002 |
| **测试项** | 不同随机种子产生不同划分 |
| **前置条件** | 划分函数支持随机种子参数 |
| **输入数据** | 100 样本数据集 |
| **测试步骤** | 1. 使用 seed=42 执行划分<br>2. 使用 seed=123 执行划分<br>3. 比较两次划分结果 |
| **预期结果** | 两次划分的索引不同 |
| **通过标准** | 至少 50% 的样本分配到不同的子集 |

#### TC-SEED-003: 跨平台可复现性验证

| 属性 | 值 |
|-----|-----|
| **测试ID** | TC-SEED-003 |
| **测试项** | 不同操作系统下相同种子产生相同结果 |
| **前置条件** | 代码需跨平台部署 |
| **输入数据** | 100 样本数据集 |
| **测试步骤** | 1. 在 Windows 上使用 seed=42 划分并保存索引<br>2. 在 Linux 上使用 seed=42 划分<br>3. 比较结果 |
| **预期结果** | 两平台产生相同划分 |
| **通过标准** | 索引完全一致（需在 CI 中测试或人工验证） |

#### TC-SEED-004: 多次运行一致性验证

| 属性 | 值 |
|-----|-----|
| **测试ID** | TC-SEED-004 |
| **测试项** | 多次运行产生一致结果 |
| **前置条件** | 无 |
| **输入数据** | 100 样本数据集 |
| **测试步骤** | 1. 使用相同种子运行 10 次划分<br>2. 比较所有运行结果 |
| **预期结果** | 10 次结果完全相同 |
| **通过标准** | 所有运行结果完全一致 |

---

### 2.4 划分索引保存与加载测试用例

#### TC-INDEX-001: 索引文件保存验证

| 属性 | 值 |
|-----|-----|
| **测试ID** | TC-INDEX-001 |
| **测试项** | 划分索引正确保存到文件 |
| **前置条件** | 划分完成 |
| **输入数据** | 划分后的索引数据 |
| **测试步骤** | 1. 执行划分<br>2. 保存索引到文件<br>3. 验证文件存在<br>4. 检查文件格式正确 |
| **预期结果** | 生成包含训练/验证/测试索引的文件 |
| **通过标准** | 文件存在，格式为 JSON/NPY/Pickle，可正常解析 |

#### TC-INDEX-002: 索引文件加载验证

| 属性 | 值 |
|-----|-----|
| **测试ID** | TC-INDEX-002 |
| **测试项** | 从文件加载划分索引 |
| **前置条件** | 存在保存的索引文件 |
| **输入数据** | 索引文件路径 |
| **测试步骤** | 1. 加载索引文件<br>2. 验证加载的索引与原始划分一致<br>3. 使用加载的索引重建数据子集 |
| **预期结果** | 加载的索引与原始划分完全一致 |
| **通过标准** | 索引值完全匹配，数据子集重建正确 |

#### TC-INDEX-003: 索引完整性验证

| 属性 | 值 |
|-----|-----|
| **测试ID** | TC-INDEX-003 |
| **测试项** | 索引覆盖所有样本且无重复 |
| **前置条件** | 索引文件已保存 |
| **输入数据** | 加载的索引数据 |
| **测试步骤** | 1. 加载索引文件<br>2. 合并所有子集索引<br>3. 验证无重复<br>4. 验证覆盖全集<br>5. 验证无越界索引 |
| **预期结果** | 索引并集为全集，无交集，无越界 |
| **通过标准** | `len(train_idx) + len(val_idx) + len(test_idx) == total_samples`，索引无重复，无负值或超界值 |

#### TC-INDEX-004: 索引文件格式兼容性

| 属性 | 值 |
|-----|-----|
| **测试ID** | TC-INDEX-004 |
| **测试项** | 不同格式的索引文件读写 |
| **前置条件** | 支持多种格式保存 |
| **输入数据** | 划分索引数据 |
| **测试步骤** | 1. 分别保存为 JSON、NPY、Pickle 格式<br>2. 分别加载各格式文件<br>3. 比较加载结果 |
| **预期结果** | 各格式加载结果一致 |
| **通过标准** | 所有格式加载的索引完全相同 |

---

### 2.5 统计记录与报告测试用例

#### TC-STAT-001: 样本数统计验证

| 属性 | 值 |
|-----|-----|
| **测试ID** | TC-STAT-001 |
| **测试项** | 各子集样本数正确统计 |
| **前置条件** | 划分完成 |
| **输入数据** | 划分结果 |
| **测试步骤** | 1. 执行划分<br>2. 获取样本数统计<br>3. 手动计数验证 |
| **预期结果** | 统计的样本数与实际一致 |
| **通过标准** | 报告样本数与实际样本数 100% 一致 |

#### TC-STAT-002: 类别分布报告验证

| 属性 | 值 |
|-----|-----|
| **测试ID** | TC-STAT-002 |
| **测试项** | 类别分布统计报告正确 |
| **前置条件** | 划分完成，有类别标签 |
| **输入数据** | 划分结果及类别标签 |
| **测试步骤** | 1. 生成类别分布报告<br>2. 手动统计各类别数量<br>3. 比较报告与手动统计 |
| **预期结果** | 报告的类别分布与实际一致 |
| **通过标准** | 各类别计数完全准确，百分比计算正确 |

#### TC-STAT-003: 分布可视化生成

| 属性 | 值 |
|-----|-----|
| **测试ID** | TC-STAT-003 |
| **测试项** | 生成类别分布可视化图表 |
| **前置条件** | 实现可视化功能 |
| **输入数据** | 划分结果 |
| **测试步骤** | 1. 调用可视化函数<br>2. 检查输出文件存在<br>3. 人工审核图表内容 |
| **预期结果** | 生成清晰的分布对比图 |
| **通过标准** | 图表包含各子集类别分布，便于对比，格式正确（PNG/PDF） |

---

### 2.6 集成与端到端测试用例

#### TC-E2E-001: 完整划分流程验证

| 属性 | 值 |
|-----|-----|
| **测试ID** | TC-E2E-001 |
| **测试项** | 从原始数据到划分完成的完整流程 |
| **前置条件** | 原始数据集就绪 |
| **输入数据** | 完整数据集（含元数据） |
| **测试步骤** | 1. 加载数据集<br>2. 执行分层抽样划分<br>3. 保存索引<br>4. 生成统计报告<br>5. 验证所有输出 |
| **预期结果** | 完整流程无错误，输出完整 |
| **通过标准** | 所有步骤成功完成，输出文件完整有效 |

#### TC-E2E-002: DataLoader 集成验证

| 属性 | 值 |
|-----|-----|
| **测试ID** | TC-E2E-002 |
| **测试项** | 划分结果与 DataLoader 集成 |
| **前置条件** | DataLoader 实现已完成 |
| **输入数据** | 划分索引文件 |
| **测试步骤** | 1. 加载划分索引<br>2. 创建训练 DataLoader<br>3. 验证迭代正常<br>4. 验证样本数正确 |
| **预期结果** | DataLoader 正确加载各子集 |
| **通过标准** | DataLoader 迭代无错误，epoch 样本数正确 |

---

## 3. 验证方法与步骤

### 3.1 单元测试执行步骤

```bash
# 1. 进入项目根目录
cd /path/to/project

# 2. 运行数据划分模块单元测试
python -m pytest tests/data/test_split.py -v

# 3. 运行单个测试用例
python -m pytest tests/data/test_split.py::TestSplitRatio::test_8_1_1_split -v

# 4. 生成覆盖率报告
python -m pytest tests/data/test_split.py --cov=src/data/split --cov-report=html
```

### 3.2 统计检验执行步骤

```bash
# 1. 运行分层抽样一致性检验
python tests/statistics/test_stratified_split.py

# 2. 执行卡方检验（类别分布一致性）
python -c "
from tests.statistics import chi_square_test_distribution
result = chi_square_test_distribution(
    train_labels, val_labels, test_labels
)
print(f'Chi-square p-value: {result.p_value}')
"

# 3. 生成分布可视化
python scripts/visualize_split_distribution.py \
    --index_path outputs/split_indices.json \
    --data_path data/raw/ \
    --output docs/split_distribution/
```

### 3.3 索引完整性验证步骤

```bash
# 1. 验证索引文件格式
python scripts/validate_split_indices.py \
    --index_path outputs/split_indices.json \
    --total_samples 10000

# 2. 验证索引完整性（无重复、全覆盖）
python -c "
import json
with open('outputs/split_indices.json') as f:
    indices = json.load(f)
train = set(indices['train'])
val = set(indices['val'])
test = set(indices['test'])
assert len(train & val) == 0, 'Train-Val overlap!'
assert len(train & test) == 0, 'Train-Test overlap!'
assert len(val & test) == 0, 'Val-Test overlap!'
print('Index validation passed!')
"
```

### 3.4 可复现性验证步骤

```bash
# 1. 运行可复现性测试
python -m pytest tests/data/test_split.py::TestReproducibility -v

# 2. 手动验证跨运行一致性
python scripts/split_dataset.py --seed 42 --output run1/
python scripts/split_dataset.py --seed 42 --output run2/
python scripts/compare_splits.py --split1 run1/ --split2 run2/

# 3. 验证不同种子产生不同结果
python scripts/split_dataset.py --seed 42 --output seed42/
python scripts/split_dataset.py --seed 123 --output seed123/
python scripts/compare_splits.py --split1 seed42/ --split2 seed123/
```

---

## 4. 通过标准

### 4.1 量化通过标准

| 测试类别 | 通过条件 |
|---------|---------|
| 划分比例 | 各子集比例偏差 < 1%，总数守恒 |
| 类别分布一致性 | 卡方检验 p > 0.05，或 KS 统计量 < 0.05 |
| 索引完整性 | 无重复、无遗漏、无越界 |
| 可复现性 | 相同种子结果 100% 一致 |
| 性能要求 | 100K 样本划分耗时 < 10 秒 |

### 4.2 必须通过的测试用例

以下测试用例为 **阻塞级**，必须通过：

1. **TC-SPLIT-001**: 8:1:1 比例划分 — 核心功能
2. **TC-STRAT-001**: 类别分布一致性 — 分层抽样关键
3. **TC-SEED-001**: 随机种子固定性 — 可复现性关键
4. **TC-INDEX-001**: 索引文件保存 — 交付物要求
5. **TC-INDEX-003**: 索引完整性 — 数据正确性关键
6. **TC-STAT-001**: 样本数统计 — 报告准确性

### 4.3 建议通过的测试用例

以下测试用例为 **建议级**，建议通过但非阻塞：

1. **TC-SPLIT-002**: 7:2:1 比例 — 备选方案
2. **TC-STRAT-003**: 多标签分层 — 扩展功能
3. **TC-SEED-003**: 跨平台可复现 — 跨平台部署场景需要
4. **TC-STAT-003**: 分布可视化 — 辅助功能

---

## 5. 自动化验证建议

### 5.1 完全自动化的测试

| 测试类型 | 自动化方式 | CI 集成 |
|---------|-----------|--------|
| 比例划分测试 | pytest + 参数化 | ✅ 每次 PR |
| 索引完整性测试 | pytest + 断言 | ✅ 每次 PR |
| 可复现性测试 | pytest + 固定种子 | ✅ 每次 PR |
| 分层抽样统计检验 | pytest + scipy | ✅ 每次 PR |

### 5.2 半自动化的测试

| 测试类型 | 自动化部分 | 手动部分 |
|---------|-----------|---------|
| 分布可视化验证 | 自动生成图表 | 人工审核图表质量 |
| 小数据集边界处理 | 自动运行测试 | 人工确认警告信息合理 |
| 跨平台可复现性 | 自动在单平台测试 | 人工验证跨平台 |

### 5.3 自动化脚本示例

```python
# tests/data/test_split.py

import pytest
import numpy as np
from sklearn.model_selection import train_test_split
from src.data.split import stratified_split, save_split_indices, load_split_indices


class TestSplitRatio:
    """划分比例测试"""

    @pytest.mark.parametrize("total_samples,train_ratio,val_ratio,test_ratio,expected_train,expected_val,expected_test", [
        (1000, 0.8, 0.1, 0.1, 800, 100, 100),
        (1000, 0.7, 0.2, 0.1, 700, 200, 100),
        (100, 0.8, 0.1, 0.1, 80, 10, 10),
    ])
    def test_split_ratio(self, total_samples, train_ratio, val_ratio, test_ratio,
                         expected_train, expected_val, expected_test):
        """TC-SPLIT-001/002: 验证划分比例"""
        X = np.arange(total_samples)
        y = np.random.randint(0, 3, total_samples)

        train_idx, val_idx, test_idx = stratified_split(
            X, y,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_state=42
        )

        # 验证总数守恒
        assert len(train_idx) + len(val_idx) + len(test_idx) == total_samples

        # 验证比例（允许 1% 偏差）
        actual_train_ratio = len(train_idx) / total_samples
        assert abs(actual_train_ratio - train_ratio) < 0.01

    def test_small_dataset_split(self):
        """TC-SPLIT-003: 小数据集划分"""
        X = np.arange(15)
        y = np.array([0]*5 + [1]*5 + [2]*5)

        # 应该有警告但不报错
        with pytest.warns(UserWarning):
            train_idx, val_idx, test_idx = stratified_split(
                X, y,
                train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                random_state=42
            )

        # 验证所有子集非空
        assert len(train_idx) > 0
        assert len(val_idx) > 0
        assert len(test_idx) > 0


class TestStratifiedSampling:
    """分层抽样测试"""

    def test_class_distribution_consistency(self):
        """TC-STRAT-001: 类别分布一致性"""
        # 创建不平衡数据集
        n_samples = 1000
        y = np.concatenate([
            np.zeros(500),   # 50%
            np.ones(300),    # 30%
            np.full(200, 2)  # 20%
        ])
        X = np.arange(n_samples)
        np.random.shuffle(y)

        train_idx, val_idx, test_idx = stratified_split(
            X, y,
            train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
            random_state=42
        )

        # 计算各子集类别分布
        for idx in [train_idx, val_idx, test_idx]:
            subset_y = y[idx]
            unique, counts = np.unique(subset_y, return_counts=True)
            ratios = counts / len(subset_y)

            # 验证比例接近原始分布（允许 2% 偏差）
            expected_ratios = [0.5, 0.3, 0.2]
            for i, (actual, expected) in enumerate(zip(sorted(ratios), sorted(expected_ratios))):
                assert abs(actual - expected) < 0.02, f"Class {i} ratio {actual} deviates from {expected}"

    def test_rare_class_preservation(self):
        """TC-STRAT-002: 稀有类别保留"""
        # 创建含稀有类别的数据集
        y = np.concatenate([
            np.zeros(900),
            np.ones(50),
            np.full(49, 2),
            np.full(1, 3)  # 仅 1 个样本
        ])
        X = np.arange(len(y))

        train_idx, val_idx, test_idx = stratified_split(
            X, y,
            train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
            random_state=42
        )

        # 验证稀有类别（类别 3）在训练集中存在
        train_classes = set(y[train_idx])
        assert 3 in train_classes, "Rare class missing from training set"


class TestReproducibility:
    """可复现性测试"""

    def test_fixed_seed_reproducibility(self):
        """TC-SEED-001: 固定种子可复现"""
        X = np.arange(100)
        y = np.random.randint(0, 3, 100)

        # 两次使用相同种子划分
        result1 = stratified_split(X, y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42)
        result2 = stratified_split(X, y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42)

        # 验证结果完全相同
        for idx1, idx2 in zip(result1, result2):
            np.testing.assert_array_equal(idx1, idx2)

    def test_different_seeds_differ(self):
        """TC-SEED-002: 不同种子产生不同结果"""
        X = np.arange(100)
        y = np.random.randint(0, 3, 100)

        result1 = stratified_split(X, y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42)
        result2 = stratified_split(X, y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=123)

        # 验证结果不同
        total_diff = 0
        for idx1, idx2 in zip(result1, result2):
            total_diff += np.sum(idx1 != idx2)

        # 至少 50% 样本分配不同
        assert total_diff > 50, "Different seeds produced too similar results"


class TestIndexIntegrity:
    """索引完整性测试"""

    def test_no_overlap(self):
        """TC-INDEX-003: 索引无重叠"""
        X = np.arange(1000)
        y = np.random.randint(0, 3, 1000)

        train_idx, val_idx, test_idx = stratified_split(
            X, y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42
        )

        train_set = set(train_idx)
        val_set = set(val_idx)
        test_set = set(test_idx)

        assert len(train_set & val_set) == 0, "Train-Val overlap detected"
        assert len(train_set & test_set) == 0, "Train-Test overlap detected"
        assert len(val_set & test_set) == 0, "Val-Test overlap detected"

    def test_complete_coverage(self):
        """TC-INDEX-003: 索引全覆盖"""
        n_samples = 1000
        X = np.arange(n_samples)
        y = np.random.randint(0, 3, n_samples)

        train_idx, val_idx, test_idx = stratified_split(
            X, y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42
        )

        all_indices = set(train_idx) | set(val_idx) | set(test_idx)
        expected_indices = set(range(n_samples))

        assert all_indices == expected_indices, "Not all samples covered"

    def test_no_out_of_bounds(self):
        """TC-INDEX-003: 无越界索引"""
        n_samples = 1000
        X = np.arange(n_samples)
        y = np.random.randint(0, 3, n_samples)

        train_idx, val_idx, test_idx = stratified_split(
            X, y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42
        )

        for idx in [train_idx, val_idx, test_idx]:
            assert idx.min() >= 0, "Negative index found"
            assert idx.max() < n_samples, "Out-of-bounds index found"


class TestIndexPersistence:
    """索引持久化测试"""

    def test_save_and_load(self, tmp_path):
        """TC-INDEX-001/002: 索引保存与加载"""
        n_samples = 100
        X = np.arange(n_samples)
        y = np.random.randint(0, 3, n_samples)

        # 执行划分
        train_idx, val_idx, test_idx = stratified_split(
            X, y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42
        )

        # 保存索引
        index_file = tmp_path / "split_indices.json"
        save_split_indices(train_idx, val_idx, test_idx, index_file)

        # 加载索引
        loaded_train, loaded_val, loaded_test = load_split_indices(index_file)

        # 验证一致性
        np.testing.assert_array_equal(train_idx, loaded_train)
        np.testing.assert_array_equal(val_idx, loaded_val)
        np.testing.assert_array_equal(test_idx, loaded_test)
```

### 5.4 CI 集成配置

```yaml
# .github/workflows/test-data-split.yml
name: Data Split Tests

on: [push, pull_request]

jobs:
  test-split:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -e .[dev]
      - name: Run split unit tests
        run: pytest tests/data/test_split.py -v --cov=src/data/split --cov-report=xml
      - name: Run stratified sampling tests
        run: pytest tests/data/test_split.py::TestStratifiedSampling -v
      - name: Run reproducibility tests
        run: pytest tests/data/test_split.py::TestReproducibility -v
      - name: Upload coverage
        uses: codecov/codecov-action@v4
```

---

## 6. 风险点与注意事项

### 6.1 小样本类别问题

**风险描述**: 当某类别样本数极少时，分层抽样可能导致验证/测试集中该类别缺失。

**缓解措施**:
1. 实现最小样本数检查，当样本数 < 10 时发出警告
2. 对稀有类别优先保证训练集有代表
3. 考虑使用 `StratifiedKFold` 的变体处理稀有类别

```python
# 检查类别样本数
class_counts = np.bincount(y)
min_samples = class_counts.min()
if min_samples < 10:
    warnings.warn(f"Class {class_counts.argmin()} has only {min_samples} samples. "
                  "Stratification may fail for validation/test sets.")
```

### 6.2 多标签场景的复杂性

**风险描述**: 多标签分类场景下，传统分层抽样难以保证各标签分布一致。

**缓解措施**:
1. 使用 `iterative_stratification` 算法（如 `skmultilearn` 库）
2. 近似方法：按主要标签分层
3. 文档中明确说明多标签场景的限制

### 6.3 随机种子的全局影响

**风险描述**: 设置随机种子可能影响后续其他随机操作（如数据增强）。

**缓解措施**:
1. 仅在划分时设置种子，划分后立即恢复
2. 使用独立的 `RandomState` 对象而非全局种子

```python
# 推荐：使用独立的 RandomState
rng = np.random.RandomState(seed)
indices = rng.permutation(n_samples)

# 不推荐：修改全局种子
np.random.seed(seed)  # 影响后续所有随机操作
```

### 6.4 划分后数据泄露

**风险描述**: 如果同一场景/人物的多张图片被分到不同子集，可能导致信息泄露。

**缓解措施**:
1. 实现 Group-aware 分层抽样（如 `GroupKFold`）
2. 按场景/人物 ID 分组后再划分
3. 添加数据泄露检测脚本

```python
from sklearn.model_selection import GroupShuffleSplit

# 按 group_id 分组划分
split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(split.split(X, y, groups=group_ids))
```

### 6.5 类别标签编码不一致

**风险描述**: 划分时使用的标签编码与训练时可能不一致。

**缓解措施**:
1. 划分索引文件中同时保存标签映射
2. 使用 `LabelEncoder` 并持久化
3. 加载时验证标签编码一致性

### 6.6 大规模数据集的内存问题

**风险描述**: 超大规模数据集（> 1M 样本）划分时可能内存不足。

**缓解措施**:
1. 使用基于索引的划分，不实际移动数据
2. 分块处理大规模数据
3. 限制内存占用的单元测试

### 6.7 时间序列数据的时序泄露

**风险描述**: 时间序列数据按随机划分会导致未来信息泄露。

**缓解措施**:
1. 时间序列数据使用时序划分（按时间切分）
2. 检测时间列并使用适当的划分策略
3. 文档中说明时序数据的特殊处理

```python
# 时间序列划分
train_end = int(len(data) * 0.8)
train_data = data[:train_end]
test_data = data[train_end:]
```

### 6.8 分布偏移检测

**风险描述**: 划分后各子集间存在统计显著的分布偏移。

**缓解措施**:
1. 自动计算各子集的统计量（均值、方差）
2. 运行 KS 检验检测分布偏移
3. 生成分布对比可视化供人工审核

```python
from scipy.stats import ks_2samp

# 检测分布偏移
stat, p_value = ks_2samp(train_feature, val_feature)
if p_value < 0.05:
    warnings.warn(f"Distribution shift detected between train and val (p={p_value:.4f})")
```

---

## 附录 A: 统计检验方法

### 卡方检验（类别分布一致性）

```python
from scipy.stats import chi2_contingency

def test_distribution_consistency(train_y, val_y, test_y):
    """
    使用卡方检验验证各子集类别分布一致性

    H0: 各子集类别分布相同
    H1: 各子集类别分布不同
    """
    # 构建列联表
    train_counts = np.bincount(train_y)
    val_counts = np.bincount(val_y)
    test_counts = np.bincount(test_y)

    contingency = np.vstack([train_counts, val_counts, test_counts])

    chi2, p_value, dof, expected = chi2_contingency(contingency)

    return {
        'chi2': chi2,
        'p_value': p_value,
        'passed': p_value > 0.05  # p > 0.05 表示分布一致
    }
```

### KS 检验（连续特征分布一致性）

```python
from scipy.stats import ks_2samp

def test_feature_distribution(train_feature, val_feature):
    """
    使用 KS 检验验证连续特征分布一致性
    """
    stat, p_value = ks_2samp(train_feature, val_feature)
    return {
        'ks_statistic': stat,
        'p_value': p_value,
        'passed': p_value > 0.05
    }
```

---

## 附录 B: 划分索引文件格式建议

### JSON 格式（推荐）

```json
{
    "version": "1.0",
    "created_at": "2026-06-02T10:30:00Z",
    "random_seed": 42,
    "split_ratio": {
        "train": 0.8,
        "val": 0.1,
        "test": 0.1
    },
    "total_samples": 10000,
    "indices": {
        "train": [0, 1, 2, ...],
        "val": [100, 101, ...],
        "test": [200, 201, ...]
    },
    "statistics": {
        "train": {"count": 8000, "class_distribution": {"0": 4000, "1": 2400, "2": 1600}},
        "val": {"count": 1000, "class_distribution": {"0": 500, "1": 300, "2": 200}},
        "test": {"count": 1000, "class_distribution": {"0": 500, "1": 300, "2": 200}}
    },
    "label_mapping": {"0": "cat", "1": "dog", "2": "bird"}
}
```

### NPY 格式（大规模数据）

```python
# 保存
np.savez('split_indices.npz',
         train=train_idx,
         val=val_idx,
         test=test_idx,
         metadata={
             'seed': 42,
             'ratio': (0.8, 0.1, 0.1)
         })

# 加载
data = np.load('split_indices.npz')
train_idx = data['train']
val_idx = data['val']
test_idx = data['test']
```

---

## 附录 C: 验收检查清单

```markdown
## 数据集划分验收检查清单

### 划分比例
- [ ] TC-SPLIT-001: 8:1:1 比例划分验证通过
- [ ] TC-SPLIT-002: 7:2:1 比例划分验证通过（如适用）
- [ ] TC-SPLIT-003: 小数据集边界处理验证通过
- [ ] TC-SPLIT-004: 大数据集性能验证通过

### 分层抽样
- [ ] TC-STRAT-001: 类别分布一致性验证通过
- [ ] TC-STRAT-002: 稀有类别保留验证通过
- [ ] TC-STRAT-003: 多标签分层抽样验证通过（如适用）
- [ ] TC-STRAT-004: 高度不平衡数据集验证通过

### 可复现性
- [ ] TC-SEED-001: 随机种子固定性验证通过
- [ ] TC-SEED-002: 不同种子差异性验证通过
- [ ] TC-SEED-003: 跨平台可复现性验证通过（如需要）
- [ ] TC-SEED-004: 多次运行一致性验证通过

### 索引管理
- [ ] TC-INDEX-001: 索引文件保存验证通过
- [ ] TC-INDEX-002: 索引文件加载验证通过
- [ ] TC-INDEX-003: 索引完整性验证通过
- [ ] TC-INDEX-004: 索引格式兼容性验证通过

### 统计报告
- [ ] TC-STAT-001: 样本数统计验证通过
- [ ] TC-STAT-002: 类别分布报告验证通过
- [ ] TC-STAT-003: 分布可视化生成完成

### 集成测试
- [ ] TC-E2E-001: 完整划分流程验证通过
- [ ] TC-E2E-002: DataLoader 集成验证通过

### 覆盖率
- [ ] 单元测试覆盖率 ≥ 90%

### 文档
- [ ] 划分索引文件保存完成
- [ ] 样本数和类别分布记录完成
- [ ] 划分脚本使用说明完成
```