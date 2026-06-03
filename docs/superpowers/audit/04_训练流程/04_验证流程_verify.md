# 验证流程测试计划

> **文档版本：** v1.0
> **创建日期：** 2026-06-03
> **关联任务：** 阶段四：训练流程 - 验证流程

---

## 一、验证目标

### 1.1 主要目标
验证训练流程中的验证模块能够正确、稳定地执行模型评估，并可靠地保存最优模型。

### 1.2 具体验证目标
| 编号 | 验证目标 | 优先级 |
|------|----------|--------|
| VO-001 | 验证流程在每个 epoch 结束后正确触发 | P0 |
| VO-002 | 验证时模型正确切换到 eval 模式 | P0 |
| VO-003 | 验证过程禁用梯度计算 (no_grad) | P0 |
| VO-004 | Top-1 准确率计算正确 | P0 |
| VO-005 | Top-5 准确率计算正确 | P0 |
| VO-006 | best_model.pth 正确保存 | P0 |
| VO-007 | last_model.pth 正确保存 | P1 |
| VO-008 | 定期 checkpoint 正确生成 | P1 |
| VO-009 | checkpoint 包含完整的优化器状态 | P0 |
| VO-010 | checkpoint 可正确恢复训练 | P0 |

---

## 二、测试用例

### 2.1 验证触发测试

| 用例ID | 测试项 | 前置条件 | 测试步骤 | 预期结果 |
|--------|--------|----------|----------|----------|
| TC-001 | Epoch 后验证触发 | 训练循环已启动 | 1. 启动训练 2. 观察 epoch 结束时的日志 | 验证流程在每个 epoch 结束后自动执行 |
| TC-002 | 验证频率配置 | 配置文件可修改 | 1. 设置 `eval_interval=2` 2. 运行 5 个 epoch | 仅在 epoch 2, 4 后执行验证 |
| TC-003 | 验证跳过配置 | 配置文件可修改 | 1. 设置 `eval_interval=0` 2. 运行训练 | 不执行任何验证 |

### 2.2 模型状态测试

| 用例ID | 测试项 | 前置条件 | 测试步骤 | 预期结果 |
|--------|--------|----------|----------|----------|
| TC-004 | eval 模式切换 | 模型定义完成 | 1. 在验证前后检查 `model.training` 属性 | 验证期间为 `False`，验证后恢复为 `True` |
| TC-005 | BatchNorm 行为 | 模型包含 BN 层 | 1. 训练若干 epoch 2. 检查验证时 BN 统计量 | 验证时使用运行统计量，不更新 |
| TC-006 | Dropout 行为 | 模型包含 Dropout | 1. 多次运行相同验证数据 2. 比较输出 | 输出一致（Dropout 被禁用） |

### 2.3 梯度计算测试

| 用例ID | 测试项 | 前置条件 | 测试步骤 | 预期结果 |
|--------|--------|----------|----------|----------|
| TC-007 | no_grad 上下文 | 验证代码已实现 | 1. 在验证循环中检查 `torch.is_grad_enabled()` | 返回 `False` |
| TC-008 | 梯度内存占用 | 训练流程可运行 | 1. 监控 GPU 内存 2. 对比训练和验证时的内存 | 验证时内存占用显著低于训练 |
| TC-009 | 梯度累积检查 | 模型参数可访问 | 1. 验证后检查 `param.grad` | 参数梯度未被修改 |

### 2.4 准确率计算测试

| 用例ID | 测试项 | 前置条件 | 测试步骤 | 预期结果 |
|--------|--------|----------|----------|----------|
| TC-010 | Top-1 计算正确性 | 小批量测试数据 | 1. 构造已知标签数据 2. 计算准确率 | 结果与手动计算一致 |
| TC-011 | Top-5 计算正确性 | 类别数 ≥ 5 | 1. 构造预测分布 2. 计算准确率 | 结果与手动计算一致 |
| TC-012 | 准确率边界值 | 空数据集 | 1. 使用空验证集 2. 执行验证 | 返回 0 或抛出明确异常 |
| TC-013 | 准确率范围检查 | 正常数据集 | 1. 运行完整验证 | 准确率在 [0, 1] 范围内 |
| TC-014 | Top-5 边界 (类别<5) | 类别数 < 5 | 1. 使用 3 类别数据集 2. 计算 Top-5 | 返回 Top-3 或与 Top-1 相同 |

### 2.5 模型保存测试

| 用例ID | 测试项 | 前置条件 | 测试步骤 | 预期结果 |
|--------|--------|----------|----------|----------|
| TC-015 | best_model 保存条件 | 验证指标可比较 | 1. 模拟准确率提升 2. 检查保存行为 | 仅当准确率超过历史最佳时保存 |
| TC-016 | best_model 覆盖保护 | 已存在 best_model | 1. 设置较低准确率 2. 执行验证 | 不覆盖已有 best_model |
| TC-017 | last_model 每次更新 | 训练进行中 | 1. 运行 3 个 epoch 2. 检查文件时间戳 | 每个 epoch 后都更新 last_model |
| TC-018 | 定期 checkpoint 命名 | 配置 `save_interval=2` | 1. 运行 5 个 epoch 2. 检查 checkpoint 文件 | 生成 `checkpoint_epoch_2.pth`, `checkpoint_epoch_4.pth` |
| TC-019 | checkpoint 完整性 | checkpoint 已保存 | 1. 加载 checkpoint 2. 检查键值 | 包含 `model_state_dict`, `optimizer_state_dict`, `epoch`, `best_acc` |
| TC-020 | 优化器状态包含 | checkpoint 已保存 | 1. 加载 checkpoint 2. 检查优化器状态 | 优化器状态字典非空且结构正确 |

### 2.6 训练恢复测试

| 用例ID | 测试项 | 前置条件 | 测试步骤 | 预期结果 |
|--------|--------|----------|----------|----------|
| TC-021 | 从 checkpoint 恢复 | checkpoint 文件存在 | 1. 加载 checkpoint 2. 继续训练 | 从正确 epoch 继续，优化器状态正确 |
| TC-022 | 恢复后准确率连续性 | 训练中断场景 | 1. 保存 checkpoint 2. 恢复 3. 验证 | 准确率从中断点平滑继续 |
| TC-023 | best_acc 恢复 | 恢复训练 | 1. 从 checkpoint 恢复 2. 检查 best_acc | 正确恢复历史最佳准确率 |

---

## 三、验证方法

### 3.1 单元测试方法

```python
# 测试文件结构建议
tests/
├── test_validation.py      # 验证流程测试
├── test_metrics.py        # 准确率计算测试
├── test_checkpoint.py     # checkpoint 测试
└── fixtures/
    └── mock_data.py       # 测试数据 fixture
```

**关键测试代码示例：**

```python
# test_validation.py
import pytest
import torch
import torch.nn as nn

class TestValidationFlow:
    """验证流程测试类"""
    
    def test_eval_mode_enabled(self, model, val_loader):
        """TC-004: 验证时模型处于 eval 模式"""
        model.train()
        with torch.no_grad():
            # 模拟验证调用
            model.eval()
            assert model.training is False
    
    def test_no_grad_context(self, model, val_loader):
        """TC-007: 验证时禁用梯度计算"""
        with torch.no_grad():
            assert torch.is_grad_enabled() is False
    
    def test_top1_accuracy(self):
        """TC-010: Top-1 准确率计算正确性"""
        predictions = torch.tensor([0.9, 0.05, 0.05])
        labels = torch.tensor([0])
        # 验证计算逻辑
        pass

# test_checkpoint.py
class TestCheckpoint:
    """Checkpoint 测试类"""
    
    def test_checkpoint_keys(self, checkpoint_path):
        """TC-019: checkpoint 包含必要键"""
        ckpt = torch.load(checkpoint_path)
        required_keys = ['model_state_dict', 'optimizer_state_dict', 
                         'epoch', 'best_acc', 'scaler_state_dict']
        for key in required_keys:
            assert key in ckpt, f"Missing key: {key}"
    
    def test_optimizer_state_not_empty(self, checkpoint_path):
        """TC-020: 优化器状态非空"""
        ckpt = torch.load(checkpoint_path)
        assert len(ckpt['optimizer_state_dict']['state']) > 0
```

### 3.2 集成测试方法

```bash
# 运行集成测试脚本
python scripts/test_validation_integration.py \
    --config configs/test_config.yaml \
    --epochs 3 \
    --eval-interval 1
```

### 3.3 手动验证清单

| 验证项 | 验证命令 | 预期输出 |
|--------|----------|----------|
| 验证日志输出 | `grep "Validation" logs/train.log` | 显示每个 epoch 的验证结果 |
| 文件生成检查 | `ls -la checkpoints/` | 包含 best_model.pth, last_model.pth |
| TensorBoard 可视化 | `tensorboard --logdir runs/` | 显示准确率曲线 |

### 3.4 性能验证方法

```python
# 内存监控脚本
import torch

def monitor_memory_during_validation():
    """验证时内存占用监控"""
    torch.cuda.reset_peak_memory_stats()
    
    # 训练阶段
    train_one_epoch()
    train_memory = torch.cuda.max_memory_allocated() / 1e9
    
    # 验证阶段
    torch.cuda.reset_peak_memory_stats()
    validate()
    val_memory = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"训练内存: {train_memory:.2f} GB")
    print(f"验证内存: {val_memory:.2f} GB")
    assert val_memory < train_memory * 0.7  # TC-008
```

---

## 四、通过标准

### 4.1 必须通过项 (P0)

| 指标 | 通过标准 | 验证方式 |
|------|----------|----------|
| 验证触发率 | 100% epoch 正确触发 | TC-001 自动测试 |
| eval 模式正确性 | 验证期间 model.training=False | TC-004 自动测试 |
| no_grad 正确性 | 梯度计算完全禁用 | TC-007 自动测试 |
| Top-1 准确率误差 | < 0.01% vs 手动计算 | TC-010 自动测试 |
| Top-5 准确率误差 | < 0.01% vs 手动计算 | TC-011 自动测试 |
| best_model 保存 | 正确触发且文件完整 | TC-015, TC-019 自动测试 |
| 优化器状态完整性 | 状态字典非空且结构正确 | TC-020 自动测试 |
| 训练恢复一致性 | 恢复后准确率变化 < 0.1% | TC-022 自动测试 |

### 4.2 应该通过项 (P1)

| 指标 | 通过标准 | 验证方式 |
|------|----------|----------|
| last_model 更新 | 每个 epoch 后更新 | TC-017 自动测试 |
| 定期 checkpoint | 按 save_interval 正确生成 | TC-018 自动测试 |
| 内存效率 | 验证内存 < 训练内存 * 0.7 | TC-008 性能测试 |

### 4.3 测试覆盖率要求

```
验证模块代码覆盖率: ≥ 90%
关键路径覆盖率:     100%
边界条件覆盖率:    100%
```

---

## 五、自动化建议

### 5.1 CI 集成配置

```yaml
# .github/workflows/validation_test.yml
name: Validation Flow Tests

on:
  push:
    paths:
      - 'src/training/validation.py'
      - 'src/training/checkpoint.py'
      - 'tests/test_validation*.py'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run validation tests
        run: |
          pytest tests/test_validation.py -v \
            --cov=src/training/validation \
            --cov-report=xml
      
      - name: Run checkpoint tests
        run: |
          pytest tests/test_checkpoint.py -v \
            --cov=src/training/checkpoint \
            --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v4
```

### 5.2 Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "Running validation flow tests..."
pytest tests/test_validation.py tests/test_checkpoint.py -v --tb=short

if [ $? -ne 0 ]; then
    echo "Validation tests failed. Commit aborted."
    exit 1
fi
```

### 5.3 自动化测试脚本

```python
# scripts/run_validation_tests.py
#!/usr/bin/env python
"""自动化验证测试脚本"""

import subprocess
import sys

def run_tests():
    """运行所有验证相关测试"""
    test_files = [
        'tests/test_validation.py',
        'tests/test_checkpoint.py',
        'tests/test_metrics.py',
    ]
    
    results = []
    for test_file in test_files:
        result = subprocess.run(
            ['pytest', test_file, '-v', '--tb=short'],
            capture_output=True,
            text=True
        )
        results.append((test_file, result.returncode, result.stdout))
    
    # 汇总结果
    print("\n" + "="*50)
    print("测试结果汇总")
    print("="*50)
    
    failed = False
    for test_file, code, output in results:
        status = "✓ PASS" if code == 0 else "✗ FAIL"
        print(f"{test_file}: {status}")
        if code != 0:
            failed = True
            print(output)
    
    return 0 if not failed else 1

if __name__ == '__main__':
    sys.exit(run_tests())
```

### 5.4 持续监控建议

```python
# 监控指标配置
MONITORING_CONFIG = {
    'validation_time_limit': 300,    # 单次验证时间上限 (秒)
    'memory_limit': 4 * 1024,         # 内存上限 (MB)
    'checkpoint_size_limit': 500,     # checkpoint 大小上限 (MB)
    'accuracy_variance_limit': 0.05,  # 准确率波动上限
}
```

---

## 六、风险点识别

### 6.1 高风险项

| 风险ID | 风险描述 | 影响程度 | 缓解措施 |
|--------|----------|----------|----------|
| R-001 | 验证时未正确切换 eval 模式导致 BatchNorm/Dropout 行为异常 | 高 | 单元测试强制检查；训练日志记录模式状态 |
| R-002 | checkpoint 损坏导致无法恢复训练 | 高 | 保存后立即验证加载；保留最近 N 个 checkpoint |
| R-003 | 准确率计算错误导致最优模型判断失误 | 高 | 与 PyTorch 官方实现对比测试；多数据集验证 |
| R-004 | 并发写入 checkpoint 导致文件损坏 | 高 | 使用原子写入；文件锁保护 |

### 6.2 中风险项

| 风险ID | 风险描述 | 影响程度 | 缓解措施 |
|--------|----------|----------|----------|
| R-005 | 验证时间过长影响训练效率 | 中 | 设置验证超时；支持验证数据采样 |
| R-006 | 磁盘空间不足导致 checkpoint 保存失败 | 中 | 保存前检查磁盘空间；自动清理旧 checkpoint |
| R-007 | 优化器状态过大导致 checkpoint 加载缓慢 | 中 | 压缩保存；异步加载 |
| R-008 | GPU 内存碎片导致验证 OOM | 中 | 验证前清理缓存；使用 torch.cuda.empty_cache() |

### 6.3 低风险项

| 风险ID | 风险描述 | 影响程度 | 缓解措施 |
|--------|----------|----------|----------|
| R-009 | 日志输出过多影响性能 | 低 | 控制日志频率；异步写入 |
| R-010 | Top-5 在类别数少时无意义 | 低 | 自动降级为 Top-N |

### 6.4 风险矩阵

```
                    影响程度
                 高        中        低
        ┌─────────┬─────────┬─────────┐
   高   │ R-001   │ R-005   │ R-009   │
发生    │ R-002   │ R-006   │ R-010   │
概率    │ R-003   │ R-007   │         │
        │ R-004   │ R-008   │         │
        ├─────────┼─────────┼─────────┤
   中   │         │         │         │
        │         │         │         │
        └─────────┴─────────┴─────────┘
```

---

## 七、验证执行计划

### 7.1 执行顺序

```
阶段 1: 单元测试 (预计 2 小时)
├── TC-001 ~ TC-009: 验证流程基础功能
├── TC-010 ~ TC-014: 准确率计算
└── TC-015 ~ TC-020: Checkpoint 功能

阶段 2: 集成测试 (预计 1 小时)
├── TC-021: Checkpoint 恢复
├── TC-022: 准确率连续性
└── TC-023: best_acc 恢复

阶段 3: 性能测试 (预计 30 分钟)
├── TC-008: 内存占用验证
└── 验证时间性能测试

阶段 4: 边界测试 (预计 30 分钟)
├── TC-003: 验证跳过
├── TC-012: 空数据集
└── TC-014: 类别数 < 5
```

### 7.2 验收检查清单

```markdown
## 验证流程验收检查清单

- [ ] 所有 P0 测试用例通过
- [ ] 所有 P1 测试用例通过
- [ ] 代码覆盖率 ≥ 90%
- [ ] 文档更新完成
- [ ] CI 流水线配置完成
- [ ] 性能指标达标
- [ ] 风险缓解措施已实施
```

---

## 八、附录

### 8.1 测试数据准备

```python
# scripts/prepare_test_data.py
"""准备验证测试所需数据"""

import torch
from torch.utils.data import Dataset, DataLoader

class MockDataset(Dataset):
    """模拟数据集用于测试"""
    def __init__(self, size=100, num_classes=10):
        self.size = size
        self.num_classes = num_classes
        self.data = torch.randn(size, 3, 224, 224)
        self.labels = torch.randint(0, num_classes, (size,))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def get_test_loaders():
    """获取测试数据加载器"""
    train_dataset = MockDataset(size=100)
    val_dataset = MockDataset(size=20)
    
    train_loader = DataLoader(train_dataset, batch_size=10)
    val_loader = DataLoader(val_dataset, batch_size=10)
    
    return train_loader, val_loader
```

### 8.2 相关文件清单

| 文件路径 | 说明 |
|----------|------|
| `src/training/validation.py` | 验证流程实现 |
| `src/training/checkpoint.py` | Checkpoint 保存/加载 |
| `src/training/metrics.py` | 准确率计算 |
| `tests/test_validation.py` | 验证流程测试 |
| `tests/test_checkpoint.py` | Checkpoint 测试 |
| `tests/test_metrics.py` | 准确率计算测试 |
| `configs/training_config.yaml` | 训练配置文件 |

---

## 九、变更记录

| 版本 | 日期 | 修改人 | 修改内容 |
|------|------|--------|----------|
| v1.0 | 2026-06-03 | Claude | 初始版本 |