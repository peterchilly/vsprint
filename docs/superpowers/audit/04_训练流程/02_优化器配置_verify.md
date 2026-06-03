# 优化器配置验证测试计划

> **验证文档版本：** v1.0
> **创建日期：** 2026-06-03
> **关联任务：** 阶段四：训练流程 - 优化器配置

---

## 一、验证目标

### 1.1 主要目标
1. 验证优化器（SGD/AdamW）配置正确且参数合理
2. 验证学习率调度器（Warmup + Cosine Annealing/StepLR）按预期工作
3. 验证学习率缩放策略符合 batch_size 变化
4. 确保优化器与模型参数正确绑定

### 1.2 验证范围
- 优化器类型选择与参数配置
- 学习率调度器初始化与状态管理
- Warmup 阶段行为验证
- 学习率衰减曲线验证
- batch_size 缩放规则验证
- 梯度更新正确性验证

---

## 二、测试用例

### 2.1 优化器初始化测试

| 用例ID | 测试项 | 输入 | 预期输出 | 优先级 |
|--------|--------|------|----------|--------|
| TC-OPT-001 | SGD优化器初始化 | `optimizer='sgd', momentum=0.9, weight_decay=1e-4` | 优化器实例正确创建，参数绑定模型 | P0 |
| TC-OPT-002 | AdamW优化器初始化 | `optimizer='adamw', betas=(0.9, 0.999), weight_decay=0.01` | 优化器实例正确创建，参数绑定模型 | P0 |
| TC-OPT-003 | 无效优化器类型处理 | `optimizer='invalid_type'` | 抛出 ValueError，提示有效选项 | P1 |
| TC-OPT-004 | 空模型参数列表 | `model.parameters()` 为空 | 抛出警告或错误 | P1 |
| TC-OPT-005 | 不同参数组配置 | `param_groups` 含不同 lr/weight_decay | 各参数组独立配置生效 | P1 |

### 2.2 学习率调度器测试

| 用例ID | 测试项 | 输入 | 预期输出 | 优先级 |
|--------|--------|------|----------|--------|
| TC-SCHED-001 | Cosine Annealing 初始化 | `T_max=100, eta_min=1e-6` | 调度器正确创建，初始 lr 等于 base_lr | P0 |
| TC-SCHED-002 | StepLR 初始化 | `step_size=30, gamma=0.1` | 调度器正确创建，衰减因子正确 | P0 |
| TC-SCHED-003 | Warmup 调度器初始化 | `warmup_epochs=5, warmup_start_lr=1e-7` | 前5个epoch线性增长 | P0 |
| TC-SCHED-004 | 组合调度器（Warmup+Cosine） | warmup=5, T_max=100 | warmup结束后接cosine衰减 | P0 |
| TC-SCHED-005 | 调度器状态持久化 | `state_dict()` 保存/加载 | 状态正确恢复，lr 位置一致 | P1 |

### 2.3 学习率缩放测试

| 用例ID | 测试项 | 输入 | 预期输出 | 优先级 |
|--------|--------|------|----------|--------|
| TC-SCALE-001 | batch_size 线性缩放 | base_batch=256, base_lr=0.1; target_batch=512 | target_lr = 0.2 | P0 |
| TC-SCALE-002 | batch_size 缩小场景 | base_batch=256, base_lr=0.1; target_batch=128 | target_lr = 0.05 | P0 |
| TC-SCALE-003 | batch_size 非整数倍缩放 | base_batch=256, base_lr=0.1; target_batch=384 | target_lr ≈ 0.15 | P1 |
| TC-SCALE-004 | 学习率上限检查 | batch_size 过大导致 lr > 1.0 | 触发警告或截断 | P1 |
| TC-SCALE-005 | 学习率下限检查 | batch_size 过小导致 lr < 1e-7 | 触发警告或截断 | P1 |

### 2.4 训练流程集成测试

| 用例ID | 测试项 | 输入 | 预期输出 | 优先级 |
|--------|--------|------|----------|--------|
| TC-INT-001 | 完整epoch训练循环 | 训练1个epoch | 优化器step()正常执行，无报错 | P0 |
| TC-INT-002 | 梯度裁剪集成 | `max_grad_norm=5.0` | 梯度范数不超过5.0 | P1 |
| TC-INT-003 | 混合精度训练兼容 | AMP enabled | 梯度缩放正确，loss scale 合理 | P1 |
| TC-INT-004 | 多GPU分布式训练 | DDP enabled | 梯度同步正确，lr 一致 | P1 |
| TC-INT-005 | checkpoint恢复后lr一致性 | 保存/加载模型 | scheduler状态正确恢复 | P0 |

### 2.5 边界条件与异常测试

| 用例ID | 测试项 | 输入 | 预期输出 | 优先级 |
|--------|--------|------|----------|--------|
| TC-EDGE-001 | epoch=0时调度器行为 | 训练开始前调用scheduler.step() | 正确处理或抛出警告 | P2 |
| TC-EDGE-002 | 负学习率输入 | `lr=-0.1` | 抛出 ValueError | P1 |
| TC-EDGE-003 | 超大学习率输入 | `lr=100.0` | 触发警告或限制 | P1 |
| TC-EDGE-004 | NaN/Inf梯度处理 | 梯度出现NaN/Inf | 检测并跳过该batch或报错 | P1 |
| TC-EDGE-005 | 空batch处理 | batch_size=0 | 跳过更新或报错 | P2 |

---

## 三、验证方法

### 3.1 单元测试
```python
# 示例测试框架
import pytest
import torch
import torch.nn as nn

class TestOptimizerConfig:
    """优化器配置单元测试"""

    def test_sgd_initialization(self):
        """TC-OPT-001: SGD优化器初始化"""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=1e-4
        )
        assert optimizer.defaults['lr'] == 0.1
        assert optimizer.defaults['momentum'] == 0.9
        assert optimizer.defaults['weight_decay'] == 1e-4

    def test_lr_scaling_linear(self):
        """TC-SCALE-001: batch_size线性缩放"""
        base_batch = 256
        base_lr = 0.1
        target_batch = 512
        target_lr = base_lr * (target_batch / base_batch)
        assert target_lr == 0.2
```

### 3.2 集成测试
- **测试环境：** PyTorch 2.0+, 单GPU/多GPU环境
- **测试数据：** 小规模合成数据集，快速验证
- **验证工具：** pytest, tensorboard 日志分析

### 3.3 可视化验证
```python
# 学习率曲线可视化验证脚本
import matplotlib.pyplot as plt

def visualize_lr_schedule(scheduler, optimizer, num_epochs):
    """绘制学习率变化曲线"""
    lrs = []
    for epoch in range(num_epochs):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

    plt.figure(figsize=(10, 6))
    plt.plot(lrs)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.savefig('lr_schedule.png')

    # 验证关键点
    # 1. Warmup阶段：前5个epoch线性增长
    # 2. Cosine阶段：平滑衰减至eta_min
    # 3. 最终lr接近eta_min
```

### 3.4 数值验证
- **梯度更新验证：** 对比手动计算与优化器更新的参数差异
- **学习率验证：** 打印每个step的学习率，与预期值对比
- **收敛性验证：** 监控loss曲线，确保训练正常收敛

---

## 四、验收标准（Pass Criteria）

### 4.1 必须满足（P0级别全部通过）
- [ ] SGD优化器正确初始化并绑定模型参数
- [ ] AdamW优化器正确初始化并绑定模型参数
- [ ] Warmup阶段学习率线性增长，从 `warmup_start_lr` 到 `base_lr`
- [ ] Cosine Annealing/StepLR 按预期衰减
- [ ] 学习率缩放公式正确：`lr = base_lr × (batch_size / base_batch_size)`
- [ ] 训练循环无报错执行一个完整epoch
- [ ] checkpoint恢复后调度器状态一致

### 4.2 应该满足（P1级别80%以上通过）
- [ ] 无效参数触发适当错误处理
- [ ] 梯度裁剪正常工作
- [ ] 混合精度训练兼容
- [ ] NaN/Inf梯度检测和处理
- [ ] 异常学习率值触发警告

### 4.3 可选满足（P2级别）
- [ ] 边界条件完整测试覆盖
- [ ] 性能基准测试

### 4.4 量化指标
| 指标 | 目标值 | 测量方法 |
|------|--------|----------|
| 单元测试覆盖率 | ≥ 90% | pytest-cov |
| 集成测试通过率 | 100% (P0) | pytest report |
| 学习率误差 | < 1e-6 | 数值对比 |
| Warmup阶段lr增长率 | 线性（R² > 0.999） | 线性回归分析 |
| 训练loss收敛 | 在预期epoch内下降 | loss曲线监控 |

---

## 五、自动化建议

### 5.1 CI/CD 集成
```yaml
# .github/workflows/test_optimizer.yml
name: Optimizer Config Tests

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
          pip install torch pytest pytest-cov
      - name: Run optimizer unit tests
        run: |
          pytest tests/test_optimizer.py -v --cov=src/optimizer --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### 5.2 自动化测试脚本
```bash
# scripts/test_optimizer.sh
#!/bin/bash

echo "=== 优化器配置验证测试 ==="

# 1. 运行单元测试
pytest tests/test_optimizer.py -v

# 2. 生成学习率曲线图
python scripts/visualize_lr_schedule.py --output lr_schedule.png

# 3. 验证训练循环
python scripts/verify_training_loop.py --epochs 5 --batch_size 32

# 4. 生成测试报告
pytest tests/test_optimizer.py --html=report.html --self-contained-html

echo "=== 测试完成 ==="
```

### 5.3 持续监控
- **TensorBoard集成：** 自动记录学习率、梯度范数、loss
- **预警机制：** 学习率异常（NaN、过大、过小）触发告警
- **回归测试：** 每次代码提交自动运行P0级测试

---

## 六、风险点与缓解措施

### 6.1 技术风险

| 风险ID | 风险描述 | 可能影响 | 概率 | 缓解措施 |
|--------|----------|----------|------|----------|
| R-001 | SGD与AdamW效果差异大 | 训练不收敛 | 中 | 提供默认超参数配置，建议SGD |
| R-002 | 学习率缩放公式不适用某些模型 | 收敛变慢 | 中 | 提供缩放开关，允许手动调整 |
| R-003 | Warmup epoch数不合适 | 早期训练不稳定 | 低 | 提供配置项，建议范围3-10 |
| R-004 | 分布式训练lr同步问题 | 各GPU学习率不一致 | 低 | 强制使用单一scheduler |
| R-005 | 混合精度训练梯度缩放问题 | loss scale异常 | 中 | 监控loss scale，设置阈值 |

### 6.2 实现风险

| 风险ID | 风险描述 | 可能影响 | 概率 | 缓解措施 |
|--------|----------|----------|------|----------|
| R-006 | PyTorch版本兼容性 | API变化导致失败 | 低 | 明确PyTorch版本要求≥2.0 |
| R-007 | 调度器状态保存/加载不一致 | 训练恢复后lr错误 | 中 | 完整测试state_dict持久化 |
| R-008 | 参数分组配置错误 | 部分参数未更新 | 低 | 添加参数组验证日志 |

### 6.3 测试风险

| 风险ID | 风险描述 | 可能影响 | 概率 | 缓解措施 |
|--------|----------|----------|------|----------|
| R-009 | 测试覆盖不全 | 隐藏bug未发现 | 中 | Code Review + 覆盖率检查 |
| R-010 | 模拟环境与实际不符 | 生产环境出问题 | 低 | 在真实数据集上验证 |

---

## 七、验证执行清单

### 7.1 执行前准备
- [ ] 确认PyTorch版本 ≥ 2.0
- [ ] 准备测试数据和模型
- [ ] 配置测试环境（GPU/CPU）
- [ ] 安装测试依赖（pytest, pytest-cov, tensorboard）

### 7.2 执行顺序
1. **单元测试** → 验证各组件独立功能
2. **集成测试** → 验证组件协同工作
3. **可视化验证** → 确认学习率曲线正确
4. **端到端测试** → 小规模训练验证收敛

### 7.3 验收确认
- [ ] 所有P0测试用例通过
- [ ] P1测试通过率 ≥ 80%
- [ ] 学习率曲线符合预期
- [ ] 训练loss正常收敛
- [ ] 代码审查完成
- [ ] 文档更新完成

---

## 八、附录

### 8.1 参考配置示例

```yaml
# configs/optimizer/sgd_cosine.yaml
optimizer:
  type: SGD
  lr: 0.1
  momentum: 0.9
  weight_decay: 1e-4
  nesterov: true

scheduler:
  type: CosineAnnealingLR
  T_max: 100
  eta_min: 1e-6

warmup:
  epochs: 5
  start_lr: 1e-7
  mode: linear

lr_scaling:
  enabled: true
  base_batch_size: 256
  base_lr: 0.1
```

### 8.2 预期学习率曲线特征

```
Epoch    Learning Rate
  0      1e-7          ← Warmup start
  1      0.02          ← Warmup (linear increase)
  2      0.04
  3      0.06
  4      0.08
  5      0.1           ← Base LR (warmup end)
  6      0.0998        ← Cosine Annealing start
  10     0.095
  20     0.08
  50     0.03
 100     1e-6          ← eta_min
```

### 8.3 相关文档
- [PyTorch优化器文档](https://pytorch.org/docs/stable/optim.html)
- [学习率调度器文档](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
- [大型Batch训练最佳实践](https://arxiv.org/abs/1706.02677)

---

**文档维护者：** 训练流程组
**最后更新：** 2026-06-03