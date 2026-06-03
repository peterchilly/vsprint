# 损失函数设计验证测试计划

> **文档版本：** v1.0
> **创建日期：** 2026/06/03
> **关联任务：** 阶段四：训练流程 - 损失函数设计

---

## 一、验证目标

### 1.1 核心验证目标
1. **功能正确性**：验证各类损失函数的数学计算正确无误
2. **配置灵活性**：验证可通过配置文件无缝切换不同损失函数
3. **边界处理**：验证损失函数对极端输入的处理能力
4. **性能可接受性**：验证损失函数的计算效率满足训练需求
5. **集成兼容性**：验证与训练循环的正确集成

### 1.2 验证范围
| 损失函数类型 | 验证优先级 | 验证深度 |
|-------------|-----------|----------|
| CrossEntropyLoss | P0（必须） | 完整验证 |
| 加权交叉熵 | P0（必须） | 完整验证 |
| Focal Loss | P1（重要） | 完整验证 |
| Label Smoothing | P0（必须） | 完整验证 |
| 知识蒸馏损失 | P2（可选） | 基础验证 |

---

## 二、测试用例

### 2.1 CrossEntropyLoss 基础验证

| 用例ID | 测试场景 | 输入数据 | 预期输出 | 验证方法 |
|--------|---------|---------|---------|---------|
| CE-001 | 标准多分类 | logits: [batch=4, classes=10], labels: [4] | 损失值 ∈ (0, +∞)，梯度形状正确 | 数值对比 + 梯度检查 |
| CE-002 | 二分类特例 | logits: [batch=8, classes=2], labels: [8] | 与 BCEWithLogitsLoss 结果一致 | 等价性对比 |
| CE-003 | 单样本边界 | logits: [1, 5], labels: [1] | 损失值有效，无维度错误 | 边界测试 |
| CE-004 | 极端置信度 | logits: [100, -100, 0], label: 0 | 损失值趋近于 0 | 数值精度测试 |
| CE-005 | 梯度数值稳定性 | logits 含极小值 (1e-10) | 无 NaN/Inf | 稳定性检查 |

### 2.2 类别不平衡处理验证

#### 2.2.1 加权交叉熵

| 用例ID | 测试场景 | 输入数据 | 预期输出 | 验证方法 |
|--------|---------|---------|---------|---------|
| WCE-001 | 类别权重正确应用 | 类别0权重=2.0, 类别1权重=1.0 | 少数类样本损失权重更高 | 权重效果验证 |
| WCE-002 | 自动类别权重计算 | 不平衡数据集 (1:10 比例) | 自动计算逆频率权重 | 公式验证 |
| WCE-003 | 权重归一化 | 权重 [100, 1, 1] | 损失值在合理范围 | 数值范围检查 |
| WCE-004 | 单类别样本 | 所有样本属于同一类别 | 损失值有效，无除零错误 | 边界测试 |
| WCE-005 | 权重与梯度的关系 | 高权重类别的梯度范数 | 梯度范数与权重成正比 | 梯度分析 |

#### 2.2.2 Focal Loss

| 用例ID | 测试场景 | 输入数据 | 预期输出 | 验证方法 |
|--------|---------|---------|---------|---------|
| FL-001 | 标准参数配置 | gamma=2.0, alpha=0.25 | 输出符合 Focal Loss 公式 | 公式验证 |
| FL-002 | gamma 参数效果 | gamma=0 vs gamma=2 | gamma=2 时简单样本权重更低 | 参数敏感性测试 |
| FL-003 | alpha 参数平衡 | alpha=0.5 vs alpha=0.75 | 正负样本平衡变化 | 平衡性测试 |
| FL-004 | 退化验证 | gamma=0 | 等价于标准交叉熵 | 等价性对比 |
| FL-005 | 极端 gamma 值 | gamma=5.0 | 数值稳定性保持 | 稳定性测试 |
| FL-006 | 梯度正确性 | 反向传播检查 | 梯度公式正确 | 数值梯度对比 |

### 2.3 Label Smoothing 验证

| 用例ID | 测试场景 | 输入数据 | 预期输出 | 验证方法 |
|--------|---------|---------|---------|---------|
| LS-001 | 默认平滑系数 | smoothing=0.1 | 标签分布正确调整 | 分布验证 |
| LS-002 | 平滑效果验证 | 原始标签 [0, 0, 1, 0] | 平滑后非零值数量增加 | 分布检查 |
| LS-003 | 极端平滑值 | smoothing=0.5 | 损失值变化方向正确 | 效果测试 |
| LS-004 | 零平滑退化 | smoothing=0.0 | 等价于无平滑交叉熵 | 等价性对比 |
| LS-005 | 数值边界 | smoothing=0.999 | 无负值/NaN | 边界测试 |
| LS-006 | 类别数适应性 | 10类 vs 100类 | 平滑分布自动适应 | 自适应性测试 |

### 2.4 知识蒸馏损失验证（可选）

| 用例ID | 测试场景 | 输入数据 | 预期输出 | 验证方法 |
|--------|---------|---------|---------|---------|
| KD-001 | 软标签蒸馏 | teacher_logits, student_logits | KL 散度正确计算 | 公式验证 |
| KD-002 | 温度参数效果 | T=1 vs T=4 | 高温软化程度更高 | 温度效果测试 |
| KD-003 | 混合损失 | 硬标签 + 软标签 | 加权组合正确 | 组合验证 |
| KD-004 | 梯度流 | 反向传播到学生模型 | 梯度正确传递 | 梯度检查 |

### 2.5 配置切换验证

| 用例ID | 测试场景 | 配置变更 | 预期行为 | 验证方法 |
|--------|---------|---------|---------|---------|
| CFG-001 | 损失函数切换 | CE → Focal Loss | 无需代码修改 | 配置热切换 |
| CFG-002 | 参数动态调整 | gamma: 2.0 → 3.0 | 参数正确更新 | 参数读取验证 |
| CFG-003 | 组合损失配置 | CE + Label Smoothing | 组合正确应用 | 组合测试 |
| CFG-004 | 无效配置处理 | 未知损失函数名 | 抛出明确错误 | 异常处理 |
| CFG-005 | 默认值回退 | 缺失可选参数 | 使用默认值 | 默认值测试 |

### 2.6 集成验证

| 用例ID | 测试场景 | 测试环境 | 预期行为 | 验证方法 |
|--------|---------|---------|---------|---------|
| INT-001 | 训练循环集成 | 完整训练流程 | 损失正确反向传播 | 端到端测试 |
| INT-002 | 混合精度兼容 | AMP 环境 | 损失值与 FP32 一致 | 精度对比 |
| INT-003 | 多 GPU 同步 | DDP 环境 | 损失值正确聚合 | 分布式测试 |
| INT-004 | 梯度累积 | 累积步数=4 | 损失计算正确缩放 | 累积验证 |

---

## 三、验证方法

### 3.1 单元测试方法

```python
# 测试框架：pytest
# 断言库：assert + numpy/torch 对比

# 示例：数值对比测试
def test_cross_entropy_correctness():
    """验证 CrossEntropyLoss 与 PyTorch 官方实现一致"""
    custom_loss = CustomCrossEntropyLoss()
    reference_loss = nn.CrossEntropyLoss()

    logits = torch.randn(32, 10)
    labels = torch.randint(0, 10, (32,))

    custom_value = custom_loss(logits, labels)
    reference_value = reference_loss(logits, labels)

    assert torch.allclose(custom_value, reference_value, atol=1e-5)
```

### 3.2 数值梯度检查

```python
def numerical_gradient_check(loss_fn, logits, labels, eps=1e-5):
    """数值梯度验证"""
    logits = logits.requires_grad_(True)
    loss = loss_fn(logits, labels)
    loss.backward()
    analytical_grad = logits.grad.clone()

    # 数值梯度
    numerical_grad = torch.zeros_like(logits)
    for i in range(logits.numel()):
        logits_flat = logits.view(-1)
        logits_flat[i] += eps
        loss_plus = loss_fn(logits, labels)
        logits_flat[i] -= 2 * eps
        loss_minus = loss_fn(logits, labels)
        logits_flat[i] += eps
        numerical_grad.view(-1)[i] = (loss_plus - loss_minus) / (2 * eps)

    # 梯度误差应在 1e-4 以内
    assert torch.allclose(analytical_grad, numerical_grad, atol=1e-4)
```

### 3.3 边界值测试方法

```python
def test_boundary_conditions():
    """边界值测试"""
    test_cases = [
        ("极小 logits", torch.full((4, 10), -1e10)),
        ("极大 logits", torch.full((4, 10), 1e10)),
        ("全零 logits", torch.zeros((4, 10))),
        ("单样本", torch.randn((1, 5))),
        ("单类别", torch.randn((4, 1))),  # 二分类特例
    ]

    for name, logits in test_cases:
        labels = torch.randint(0, logits.size(1), (logits.size(0),))
        try:
            loss = loss_fn(logits, labels)
            assert not torch.isnan(loss), f"{name}: 产生 NaN"
            assert not torch.isinf(loss), f"{name}: 产生 Inf"
            assert loss >= 0, f"{name}: 损失为负"
        except Exception as e:
            pytest.fail(f"{name}: 异常 - {e}")
```

### 3.4 配置验证方法

```python
def test_config_switching():
    """配置切换验证"""
    config = load_config("configs/loss/loss_config.yaml")

    # 测试所有支持的损失函数
    supported_losses = ["cross_entropy", "weighted_ce", "focal", "label_smoothing_ce"]

    for loss_name in supported_losses:
        config.loss.type = loss_name
        loss_fn = create_loss_from_config(config)

        logits = torch.randn(8, 10)
        labels = torch.randint(0, 10, (8,))

        loss = loss_fn(logits, labels)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # 标量损失
```

---

## 四、通过标准

### 4.1 必须通过的测试（阻塞发布）

| 标准ID | 标准描述 | 通过条件 |
|--------|---------|---------|
| PASS-001 | 所有 P0 优先级测试用例通过 | 100% 通过率 |
| PASS-002 | 数值精度正确性 | 与参考实现误差 < 1e-5 |
| PASS-003 | 无 NaN/Inf 产生 | 边界测试全部通过 |
| PASS-004 | 梯度计算正确 | 数值梯度误差 < 1e-4 |
| PASS-005 | 配置切换正常 | 所有支持类型可切换 |
| PASS-006 | 单元测试覆盖率 | 核心代码覆盖率 ≥ 90% |

### 4.2 性能标准

| 指标 | 要求 | 测量方法 |
|------|------|---------|
| 单次前向计算延迟 | < 10ms (batch=128, classes=1000) | 时间统计 |
| 反向传播延迟 | < 20ms (同上) | 时间统计 |
| 内存占用增量 | < 100MB | 内存分析 |
| GPU 利用率 | > 80%（大规模批次） | nvidia-smi 监控 |

### 4.3 质量标准

| 指标 | 要求 |
|------|------|
| 代码风格 | 通过 pylint/flake8 检查 |
| 类型注解 | 所有公开函数有类型注解 |
| 文档字符串 | 所有类和函数有 docstring |
| 异常处理 | 明确的错误消息和异常类型 |

---

## 五、自动化建议

### 5.1 自动化测试框架

```
tests/
├── test_losses/
│   ├── __init__.py
│   ├── test_cross_entropy.py      # CE 相关测试
│   ├── test_weighted_ce.py         # 加权 CE 测试
│   ├── test_focal_loss.py         # Focal Loss 测试
│   ├── test_label_smoothing.py    # Label Smoothing 测试
│   ├── test_knowledge_distillation.py  # KD 测试
│   ├── test_config.py             # 配置切换测试
│   ├── test_integration.py        # 集成测试
│   └── conftest.py                 # 共享 fixtures
```

### 5.2 CI/CD 集成

```yaml
# .github/workflows/test_losses.yml
name: Loss Function Tests

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
        run: pip install -r requirements-test.txt
      - name: Run unit tests
        run: pytest tests/test_losses/ -v --cov=losses --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### 5.3 自动化测试脚本

```bash
#!/bin/bash
# scripts/run_loss_tests.sh

echo "Running loss function verification..."

# 1. 单元测试
pytest tests/test_losses/ -v --tb=short

# 2. 覆盖率检查
pytest tests/test_losses/ --cov=losses --cov-fail-under=90

# 3. 性能基准测试
python scripts/benchmark_losses.py --output logs/benchmark.json

# 4. 数值稳定性测试
python scripts/test_numerical_stability.py --iterations 1000

echo "All tests completed."
```

### 5.4 测试报告生成

```python
# scripts/generate_test_report.py
def generate_report():
    """生成测试报告"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "tests_passed": 0,
        "tests_failed": 0,
        "coverage": 0,
        "performance": {},
        "details": []
    }

    # 运行测试并收集结果
    results = pytest.main(["-v", "--json-report", "tests/test_losses/"])

    # 生成 Markdown 报告
    with open("docs/test_report.md", "w") as f:
        f.write(render_markdown_report(report))
```

---

## 六、风险点与缓解措施

### 6.1 数值稳定性风险

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| Log(0) 数值错误 | NaN 损失 | 添加 epsilon clamp (1e-8) |
| 梯度爆炸 | 训练不稳定 | 梯度裁剪、损失值监控 |
| FP16 溢出 | 精度损失 | 使用 FP32 累加、损失缩放 |
| 极端 logits | Softmax 溢出 | Logits 减去最大值 |

### 6.2 配置风险

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 参数类型错误 | 运行时异常 | 配置验证 + 类型检查 |
| 缺少必要参数 | 初始化失败 | 默认值 + 参数完整性检查 |
| 参数范围越界 | 行为异常 | 参数边界校验 |

### 6.3 集成风险

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 损失函数与优化器不兼容 | 训练失败 | 优化器适配测试 |
| 多 GPU 损失同步错误 | 梯度计算错误 | DDP 集成测试 |
| 混合精度损失缩放不当 | 梯度消失/爆炸 | AMP 兼容性测试 |

### 6.4 性能风险

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| Focal Loss 计算开销大 | 训练速度慢 | 向量化实现 + CUDA 优化 |
| 内存峰值过高 | OOM | 梯度检查点 + 批次调整 |
| 小批次效率低 | GPU 利用率低 | 批次合并优化 |

---

## 七、验证执行清单

### 7.1 前置条件检查

- [ ] PyTorch 环境正确安装 (版本 ≥ 1.10)
- [ ] GPU 可用（如需 GPU 测试）
- [ ] 测试数据集准备完成
- [ ] 配置文件模板准备完成

### 7.2 执行顺序

1. **阶段一：单元测试** (预计 2 小时)
   - [ ] 运行 CrossEntropyLoss 测试
   - [ ] 运行加权 CE 测试
   - [ ] 运行 Focal Loss 测试
   - [ ] 运行 Label Smoothing 测试
   - [ ] 运行知识蒸馏测试（如实现）

2. **阶段二：配置测试** (预计 30 分钟)
   - [ ] 运行配置切换测试
   - [ ] 运行参数验证测试
   - [ ] 运行异常处理测试

3. **阶段三：集成测试** (预计 1 小时)
   - [ ] 训练循环集成
   - [ ] 混合精度兼容性
   - [ ] 多 GPU 同步（如适用）

4. **阶段四：性能测试** (预计 30 分钟)
   - [ ] 基准性能测试
   - [ ] 内存占用测试
   - [ ] GPU 利用率测试

### 7.3 验收确认

| 项目 | 状态 | 负责人 | 日期 |
|------|------|--------|------|
| 所有测试通过 | ☐ | | |
| 覆盖率达标 | ☐ | | |
| 性能指标达标 | ☐ | | |
| 文档完整 | ☐ | | |
| 代码审查完成 | ☐ | | |

---

## 八、附录

### A. 参考公式

**CrossEntropyLoss:**
```
CE(p, q) = -Σ p_i * log(q_i)
```

**Focal Loss:**
```
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```

**Label Smoothing:**
```
y'_i = (1 - ε) * y_i + ε / K
```

**Knowledge Distillation:**
```
L_KD = T^2 * KL(softmax(z_s/T) || softmax(z_t/T))
```

### B. 测试数据集

- **标准测试集**: 随机生成，batch_size=32, classes=10
- **不平衡测试集**: 类别比例 1:2:5:10:20
- **边界测试集**: 极值输入、零向量、单样本

---

**文档结束**