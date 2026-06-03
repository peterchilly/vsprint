# 预训练权重支持验证测试计划

> **验证阶段：** 阶段三：模型构建
> **验证目标：** 确保预训练权重加载、冻结策略和 checkpoint 功能正确可靠
> **文档版本：** v1.0
> **创建日期：** 2026/06/02

---

## 1. 验证目标

### 1.1 核心验证目标

| 编号 | 验证目标 | 优先级 | 验证类型 |
|------|----------|--------|----------|
| VO-01 | 预训练权重可正常加载（全量/部分） | P0 | 功能验证 |
| VO-02 | 权重冻结策略正确生效 | P0 | 功能验证 |
| VO-03 | Checkpoint 保存和恢复正常 | P0 | 功能验证 |
| VO-04 | 加载失败时的错误处理和提示 | P1 | 异常验证 |
| VO-05 | 权重格式兼容性 | P1 | 兼容性验证 |

### 1.2 验证范围

```
验证范围边界：
├── 输入验证
│   ├── 权重文件格式验证
│   ├── 权重文件路径验证
│   └── 模型配置匹配验证
├── 功能验证
│   ├── 全量权重加载
│   ├── 部分权重加载（缺失键/多余键）
│   ├── 权重冻结/解冻
│   └── Checkpoint 保存/加载
├── 边界验证
│   ├── 空权重文件处理
│   ├── 损坏权重文件处理
│   └── 版本不匹配处理
└── 集成验证
    ├── 训练流程中的权重加载
    └── 推理流程中的权重加载
```

---

## 2. 测试用例

### 2.1 预训练权重加载测试

#### TC-01: 全量权重加载测试

| 属性 | 值 |
|------|-----|
| **测试ID** | TC-01 |
| **测试名称** | 全量权重加载测试 |
| **优先级** | P0 |
| **前置条件** | 1. 已获取 ERes2NetV2 ImageNet 预训练权重文件<br>2. 模型代码实现完成 |

**测试步骤：**

```python
# Step 1: 创建模型实例
model = ERes2NetV2(config)

# Step 2: 加载预训练权重
checkpoint_path = "path/to/pretrained_weights.pth"
load_result = model.load_pretrained(checkpoint_path)

# Step 3: 验证加载结果
assert load_result.success == True
assert len(load_result.missing_keys) == 0
assert len(load_result.unexpected_keys) == 0

# Step 4: 验证权重值
pretrained_state = torch.load(checkpoint_path)
model_state = model.state_dict()
for key in pretrained_state.keys():
    assert torch.allclose(model_state[key], pretrained_state[key])
```

**期望结果：**
- 加载成功返回 `success=True`
- 无缺失键（`missing_keys=[]`）
- 无多余键（`unexpected_keys=[]`）
- 权重值完全一致

---

#### TC-02: 部分权重加载测试（缺失键）

| 属性 | 值 |
|------|-----|
| **测试ID** | TC-02 |
| **测试名称** | 部分权重加载测试（缺失键） |
| **优先级** | P0 |
| **前置条件** | 1. 准备一个只包含 backbone 权重的文件<br>2. 模型包含分类头等额外层 |

**测试步骤：**

```python
# Step 1: 创建完整模型（含分类头）
model = ERes2NetV2(num_classes=1000)

# Step 2: 加载部分权重（只有 backbone）
partial_checkpoint = create_partial_checkpoint("backbone_only.pth")
load_result = model.load_pretrained(partial_checkpoint, strict=False)

# Step 3: 验证加载结果
assert load_result.success == True
assert len(load_result.missing_keys) > 0  # 分类头权重缺失
assert "classifier" in str(load_result.missing_keys)

# Step 4: 验证 backbone 权重已加载
for key in partial_checkpoint.keys():
    assert torch.allclose(model.state_dict()[key], partial_checkpoint[key])
```

**期望结果：**
- `strict=False` 时加载成功
- 缺失键正确报告
- 已有键的权重正确加载

---

#### TC-03: 部分权重加载测试（多余键）

| 属性 | 值 |
|------|-----|
| **测试ID** | TC-03 |
| **测试名称** | 部分权重加载测试（多余键） |
| **优先级** | P0 |
| **前置条件** | 准备包含额外层权重的 checkpoint |

**测试步骤：**

```python
# Step 1: 创建模型（较少分类数）
model = ERes2NetV2(num_classes=10)

# Step 2: 加载包含1000类的预训练权重
pretrained_1000_class = load_pretrained_1000_class_checkpoint()
load_result = model.load_pretrained(pretrained_1000_class, strict=False)

# Step 3: 验证加载结果
assert load_result.success == True
assert len(load_result.unexpected_keys) > 0  # 分类头维度不匹配
```

**期望结果：**
- 多余键正确报告
- 可加载的键正确加载
- 不影响模型正常运行

---

### 2.2 权重冻结策略测试

#### TC-04: 全局冻结策略测试

| 属性 | 值 |
|------|-----|
| **测试ID** | TC-04 |
| **测试名称** | 全局冻结策略测试 |
| **优先级** | P0 |
| **前置条件** | 模型已加载预训练权重 |

**测试步骤：**

```python
# Step 1: 创建并加载模型
model = ERes2NetV2(config)
model.load_pretrained(pretrained_path)

# Step 2: 冻结全部参数
model.freeze_all()

# Step 3: 验证所有参数已冻结
for param in model.parameters():
    assert param.requires_grad == False

# Step 4: 验证前向传播正常
output = model(dummy_input)
assert output is not None

# Step 5: 验证反向传播
loss = output.sum()
loss.backward()
for param in model.parameters():
    assert param.grad is None or torch.all(param.grad == 0)
```

**期望结果：**
- 所有 `requires_grad=False`
- 前向传播正常
- 反向传播不更新参数

---

#### TC-05: 分层冻结策略测试

| 属性 | 值 |
|------|-----|
| **测试ID** | TC-05 |
| **测试名称** | 分层冻结策略测试 |
| **优先级** | P0 |
| **前置条件** | 模型结构已知 |

**测试步骤：**

```python
# Step 1: 创建模型
model = ERes2NetV2(config)

# Step 2: 冻结指定层
model.freeze_layers(["stem", "stage1", "stage2"])

# Step 3: 验证冻结状态
frozen_params = []
trainable_params = []
for name, param in model.named_parameters():
    if any(layer in name for layer in ["stem", "stage1", "stage2"]):
        assert param.requires_grad == False
        frozen_params.append(name)
    else:
        assert param.requires_grad == True
        trainable_params.append(name)

# Step 4: 验证梯度计算
loss = model(dummy_input).sum()
loss.backward()
for name, param in model.named_parameters():
    if name in frozen_params:
        assert param.grad is None or torch.all(param.grad == 0)
    else:
        assert param.grad is not None and not torch.all(param.grad == 0)
```

**期望结果：**
- 指定层参数冻结
- 其他层参数可训练
- 梯度计算正确

---

#### TC-06: BN层冻结策略测试

| 属性 | 值 |
|------|-----|
| **测试ID** | TC-06 |
| **测试名称** | BatchNorm层冻结策略测试 |
| **优先级** | P0 |
| **前置条件** | 模型包含BN层 |

**测试步骤：**

```python
# Step 1: 创建模型并设为eval模式
model = ERes2NetV2(config)
model.eval()
model.freeze_bn()

# Step 2: 验证BN层状态
for module in model.modules():
    if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
        assert module.training == False
        assert module.track_running_stats == False

# Step 3: 验证BN层参数冻结
for name, param in model.named_parameters():
    if "bn" in name.lower() or "batch" in name.lower():
        assert param.requires_grad == False
```

**期望结果：**
- BN层处于eval模式
- BN层参数不可训练
- running stats 不更新

---

#### TC-07: 解冻策略测试

| 属性 | 值 |
|------|-----|
| **测试ID** | TC-07 |
| **测试名称** | 解冻策略测试 |
| **优先级** | P0 |
| **前置条件** | 模型已冻结部分层 |

**测试步骤：**

```python
# Step 1: 创建并冻结模型
model = ERes2NetV2(config)
model.freeze_all()

# Step 2: 解冻特定层
model.unfreeze_layers(["stage3", "stage4", "classifier"])

# Step 3: 验证解冻状态
for name, param in model.named_parameters():
    if any(layer in name for layer in ["stage3", "stage4", "classifier"]):
        assert param.requires_grad == True
    else:
        assert param.requires_grad == False
```

**期望结果：**
- 指定层参数解冻
- 其他层保持冻结状态

---

### 2.3 Checkpoint 保存和加载测试

#### TC-08: 完整Checkpoint保存测试

| 属性 | 值 |
|------|-----|
| **测试ID** | TC-08 |
| **测试名称** | 完整Checkpoint保存测试 |
| **优先级** | P0 |
| **前置条件** | 模型已训练若干步 |

**测试步骤：**

```python
# Step 1: 创建模型并进行训练
model = ERes2NetV2(config)
optimizer = torch.optim.Adam(model.parameters())
train_one_epoch(model, optimizer)

# Step 2: 保存checkpoint
checkpoint_path = "test_checkpoint.pth"
save_checkpoint(
    path=checkpoint_path,
    model=model,
    optimizer=optimizer,
    epoch=1,
    loss=0.5,
    config=config
)

# Step 3: 验证checkpoint内容
checkpoint = torch.load(checkpoint_path)
assert "model_state_dict" in checkpoint
assert "optimizer_state_dict" in checkpoint
assert "epoch" in checkpoint
assert "loss" in checkpoint
assert "config" in checkpoint
assert checkpoint["epoch"] == 1
assert checkpoint["loss"] == 0.5

# Step 4: 验证权重一致性
for key, value in model.state_dict().items():
    assert torch.allclose(checkpoint["model_state_dict"][key], value)
```

**期望结果：**
- Checkpoint包含所有必要字段
- 权重值完全一致
- 元数据正确保存

---

#### TC-09: Checkpoint加载恢复测试

| 属性 | 值 |
|------|-----|
| **测试ID** | TC-09 |
| **测试名称** | Checkpoint加载恢复测试 |
| **优先级** | P0 |
| **前置条件** | 存在有效checkpoint文件 |

**测试步骤：**

```python
# Step 1: 训练并保存checkpoint
model = ERes2NetV2(config)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_one_epoch(model, optimizer)
initial_loss = evaluate(model)
save_checkpoint("checkpoint.pth", model, optimizer, epoch=1, loss=initial_loss)

# Step 2: 创建新模型并加载checkpoint
new_model = ERes2NetV2(config)
new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
checkpoint = load_checkpoint("checkpoint.pth", new_model, new_optimizer)

# Step 3: 验证恢复状态
assert checkpoint["epoch"] == 1

# Step 4: 验证模型权重一致
for (name1, param1), (name2, param2) in zip(
    model.named_parameters(), new_model.named_parameters()
):
    assert name1 == name2
    assert torch.allclose(param1, param2)

# Step 5: 验证优化器状态一致
# 验证可以继续训练
train_one_epoch(new_model, new_optimizer)
continued_loss = evaluate(new_model)
assert continued_loss is not None
```

**期望结果：**
- 模型权重完全恢复
- 优化器状态完全恢复
- 可以从checkpoint继续训练

---

#### TC-10: Checkpoint跨设备加载测试

| 属性 | 值 |
|------|-----|
| **测试ID** | TC-10 |
| **测试名称** | Checkpoint跨设备加载测试 |
| **优先级** | P1 |
| **前置条件** | 支持CPU和GPU环境 |

**测试步骤：**

```python
# Step 1: 在GPU上保存checkpoint
if torch.cuda.is_available():
    model = ERes2NetV2(config).cuda()
    save_checkpoint("gpu_checkpoint.pth", model)

    # Step 2: 加载到CPU
    cpu_model = ERes2NetV2(config)
    load_checkpoint("gpu_checkpoint.pth", cpu_model, map_location="cpu")

    # Step 3: 验证加载成功
    for name, param in cpu_model.named_parameters():
        assert param.device == torch.device("cpu")

    # Step 4: 从CPU加载到GPU
    gpu_model2 = ERes2NetV2(config).cuda()
    load_checkpoint("gpu_checkpoint.pth", gpu_model2, map_location="cuda:0")
```

**期望结果：**
- 支持GPU→CPU加载
- 支持CPU→GPU加载
- 权重值正确转换

---

### 2.4 异常处理测试

#### TC-11: 无效权重文件处理

| 属性 | 值 |
|------|-----|
| **测试ID** | TC-11 |
| **测试名称** | 无效权重文件处理测试 |
| **优先级** | P1 |
| **前置条件** | 无 |

**测试步骤：**

```python
# Step 1: 测试不存在的文件
model = ERes2NetV2(config)
try:
    model.load_pretrained("nonexistent.pth")
    assert False, "Should raise FileNotFoundError"
except FileNotFoundError:
    pass

# Step 2: 测试损坏的文件
create_corrupted_file("corrupted.pth")
try:
    model.load_pretrained("corrupted.pth")
    assert False, "Should raise exception for corrupted file"
except (RuntimeError, pickle.UnpicklingError):
    pass

# Step 3: 测试空文件
create_empty_file("empty.pth")
try:
    model.load_pretrained("empty.pth")
    assert False, "Should raise exception for empty file"
except Exception:
    pass
```

**期望结果：**
- 不存在的文件抛出 `FileNotFoundError`
- 损坏文件抛出适当异常
- 错误信息清晰明确

---

#### TC-12: 权重不匹配处理

| 属性 | 值 |
|------|-----|
| **测试ID** | TC-12 |
| **测试名称** | 权重不匹配处理测试 |
| **优先级** | P1 |
| **前置条件** | 准备不匹配的权重文件 |

**测试步骤：**

```python
# Step 1: 测试严格模式下的不匹配
model = ERes2NetV2(config)
mismatched_weights = create_mismatched_weights()

try:
    model.load_pretrained(mismatched_weights, strict=True)
    assert False, "Should raise error in strict mode"
except RuntimeError as e:
    assert "missing_keys" in str(e) or "unexpected_keys" in str(e)

# Step 2: 测试非严格模式下的不匹配
result = model.load_pretrained(mismatched_weights, strict=False)
assert result.success == True
assert len(result.missing_keys) > 0 or len(result.unexpected_keys) > 0
```

**期望结果：**
- 严格模式下抛出异常
- 非严格模式下正常加载并报告不匹配
- 错误信息包含详细的键名列表

---

### 2.5 集成测试

#### TC-13: 训练流程集成测试

| 属性 | 值 |
|------|-----|
| **测试ID** | TC-13 |
| **测试名称** | 训练流程集成测试 |
| **优先级** | P0 |
| **前置条件** | 训练流程代码完成 |

**测试步骤：**

```python
# Step 1: 创建模型并加载预训练权重
model = ERes2NetV2(config)
model.load_pretrained("pretrained.pth")
model.freeze_layers(["stem", "stage1", "stage2"])

# Step 2: 执行训练
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
initial_params = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}

train_one_epoch(model, optimizer)

# Step 3: 验证冻结层未更新
for name, param in model.named_parameters():
    if not param.requires_grad:
        assert torch.allclose(param, initial_params[name]), f"Frozen layer {name} was updated!"

# Step 4: 保存checkpoint
save_checkpoint("trained.pth", model, optimizer, epoch=1, loss=0.5)

# Step 5: 加载checkpoint继续训练
new_model = ERes2NetV2(config)
load_checkpoint("trained.pth", new_model)
train_one_epoch(new_model, optimizer)
```

**期望结果：**
- 冻结层权重保持不变
- 可训练层权重更新
- Checkpoint保存和加载正常
- 可从checkpoint继续训练

---

#### TC-14: 推理流程集成测试

| 属性 | 值 |
|------|-----|
| **测试ID** | TC-14 |
| **测试名称** | 推理流程集成测试 |
| **优先级** | P0 |
| **前置条件** | 推理流程代码完成 |

**测试步骤：**

```python
# Step 1: 加载模型用于推理
model = ERes2NetV2(config)
model.load_pretrained("pretrained.pth")
model.eval()

# Step 2: 冻结所有参数
model.freeze_all()

# Step 3: 执行推理
with torch.no_grad():
    output = model(dummy_input)

# Step 4: 验证输出
assert output is not None
assert output.shape == (batch_size, num_classes)

# Step 5: 验证多次推理结果一致
with torch.no_grad():
    output2 = model(dummy_input)
assert torch.allclose(output, output2)
```

**期望结果：**
- 推理正常完成
- 输出形状正确
- 多次推理结果一致

---

## 3. 验证方法

### 3.1 验证方法矩阵

| 测试类型 | 验证方法 | 工具 | 自动化程度 |
|----------|----------|------|------------|
| 单元测试 | pytest | pytest, torch | 100% |
| 集成测试 | pytest + 训练脚本 | pytest, torch | 100% |
| 回归测试 | pytest | pytest | 100% |
| 性能测试 | 手动 + 脚本 | time, memory_profiler | 50% |
| 边界测试 | pytest | pytest | 100% |

### 3.2 具体验证方法

#### 3.2.1 权重加载验证

```python
def verify_weight_loading(model, checkpoint_path, strict=True):
    """验证权重加载的正确性"""
    # 加载前的模型状态
    before_keys = set(model.state_dict().keys())

    # 执行加载
    result = model.load_pretrained(checkpoint_path, strict=strict)

    # 加载后的模型状态
    after_keys = set(model.state_dict().keys())

    # 验证结果
    verification = {
        "success": result.success,
        "keys_match": before_keys == after_keys,
        "missing_keys": result.missing_keys,
        "unexpected_keys": result.unexpected_keys,
        "weight_values_match": verify_weight_values(model, checkpoint_path)
    }

    return verification

def verify_weight_values(model, checkpoint_path):
    """验证权重值是否正确加载"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state = model.state_dict()

    for key in checkpoint.get("model_state_dict", checkpoint).keys():
        if key in model_state:
            if not torch.allclose(model_state[key], checkpoint[key]):
                return False
    return True
```

#### 3.2.2 冻结策略验证

```python
def verify_freeze_strategy(model, freeze_config):
    """验证冻结策略是否正确应用"""
    verification_results = []

    for name, param in model.named_parameters():
        expected_frozen = should_be_frozen(name, freeze_config)
        actual_frozen = not param.requires_grad

        verification_results.append({
            "layer": name,
            "expected_frozen": expected_frozen,
            "actual_frozen": actual_frozen,
            "match": expected_frozen == actual_frozen
        })

    return all(r["match"] for r in verification_results)

def verify_gradient_flow(model, input_tensor):
    """验证梯度流是否正确"""
    model.zero_grad()
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    gradient_status = []
    for name, param in model.named_parameters():
        gradient_status.append({
            "layer": name,
            "requires_grad": param.requires_grad,
            "has_gradient": param.grad is not None,
            "gradient_is_zero": torch.all(param.grad == 0) if param.grad is not None else None,
            "correct": (param.requires_grad and param.grad is not None and not torch.all(param.grad == 0)) or
                      (not param.requires_grad and (param.grad is None or torch.all(param.grad == 0)))
        })

    return all(g["correct"] for g in gradient_status)
```

#### 3.2.3 Checkpoint验证

```python
def verify_checkpoint_integrity(checkpoint_path, model, optimizer=None):
    """验证checkpoint完整性"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    required_keys = ["model_state_dict", "epoch"]
    optional_keys = ["optimizer_state_dict", "loss", "config"]

    # 验证必需字段
    missing_required = [k for k in required_keys if k not in checkpoint]
    if missing_required:
        return {"valid": False, "reason": f"Missing required keys: {missing_required}"}

    # 验证模型权重
    model_state = model.state_dict()
    for key in model_state.keys():
        if key not in checkpoint["model_state_dict"]:
            return {"valid": False, "reason": f"Model key missing in checkpoint: {key}"}

    return {"valid": True, "keys": list(checkpoint.keys())}

def verify_checkpoint_recovery(checkpoint_path, model_class, config):
    """验证checkpoint恢复能力"""
    # 创建新模型
    new_model = model_class(config)

    # 加载checkpoint
    checkpoint = load_checkpoint(checkpoint_path, new_model)

    # 验证可以执行前向传播
    dummy_input = torch.randn(1, 3, 224, 224)
    try:
        with torch.no_grad():
            output = new_model(dummy_input)
        return {"valid": True, "output_shape": output.shape}
    except Exception as e:
        return {"valid": False, "reason": str(e)}
```

---

## 4. 通过标准

### 4.1 功能通过标准

| 测试ID | 通过条件 | 严格程度 |
|--------|----------|----------|
| TC-01 | 全量权重加载成功，无缺失/多余键 | 必须 |
| TC-02 | 部分加载正确报告缺失键 | 必须 |
| TC-03 | 部分加载正确报告多余键 | 必须 |
| TC-04 | 所有参数正确冻结 | 必须 |
| TC-05 | 指定层正确冻结/解冻 | 必须 |
| TC-06 | BN层正确冻结 | 必须 |
| TC-07 | 解冻策略正确执行 | 必须 |
| TC-08 | Checkpoint包含所有必需字段 | 必须 |
| TC-09 | Checkpoint完全恢复训练状态 | 必须 |
| TC-10 | 跨设备加载正确映射 | 推荐 |
| TC-11 | 异常情况正确处理 | 必须 |
| TC-12 | 权重不匹配正确处理 | 必须 |
| TC-13 | 训练流程正常 | 必须 |
| TC-14 | 推理流程正常 | 必须 |

### 4.2 质量通过标准

```
代码覆盖率要求：
├── 权重加载模块: ≥ 90%
├── 冻结策略模块: ≥ 95%
├── Checkpoint模块: ≥ 90%
└── 整体覆盖率: ≥ 90%

性能要求：
├── 权重加载时间: < 5s (标准模型)
├── Checkpoint保存: < 3s
└── Checkpoint加载: < 5s

稳定性要求：
├── 测试通过率: 100%
├── 无致命错误
└── 无内存泄漏
```

### 4.3 文档通过标准

- [ ] 所有公开API有docstring
- [ ] 错误信息清晰明确
- [ ] 使用示例完整
- [ ] 参数说明完整

---

## 5. 自动化建议

### 5.1 自动化测试框架

```python
# test_pretrained_weights.py

import pytest
import torch
import torch.nn as nn
from pathlib import Path

class TestPretrainedWeights:
    """预训练权重加载测试套件"""

    @pytest.fixture
    def model(self):
        """创建测试模型"""
        return ERes2NetV2(config=test_config)

    @pytest.fixture
    def pretrained_path(self, tmp_path):
        """创建临时预训练权重文件"""
        path = tmp_path / "pretrained.pth"
        torch.save(create_mock_pretrained_weights(), path)
        return str(path)

    def test_full_weight_loading(self, model, pretrained_path):
        """TC-01: 全量权重加载测试"""
        result = model.load_pretrained(pretrained_path)
        assert result.success
        assert len(result.missing_keys) == 0
        assert len(result.unexpected_keys) == 0

    def test_partial_weight_loading(self, model, pretrained_path):
        """TC-02: 部分权重加载测试"""
        partial_weights = create_partial_weights(pretrained_path)
        result = model.load_pretrained(partial_weights, strict=False)
        assert result.success
        assert len(result.missing_keys) > 0

    # ... 更多测试用例

class TestFreezeStrategy:
    """冻结策略测试套件"""

    @pytest.fixture
    def frozen_model(self, model, pretrained_path):
        """创建已加载预训练权重的模型"""
        model.load_pretrained(pretrained_path)
        return model

    def test_freeze_all(self, frozen_model):
        """TC-04: 全局冻结测试"""
        frozen_model.freeze_all()
        for param in frozen_model.parameters():
            assert not param.requires_grad

    def test_freeze_layers(self, frozen_model):
        """TC-05: 分层冻结测试"""
        layers_to_freeze = ["stem", "stage1"]
        frozen_model.freeze_layers(layers_to_freeze)

        for name, param in frozen_model.named_parameters():
            if any(layer in name for layer in layers_to_freeze):
                assert not param.requires_grad

    # ... 更多测试用例

class TestCheckpoint:
    """Checkpoint测试套件"""

    def test_save_checkpoint(self, model, tmp_path):
        """TC-08: Checkpoint保存测试"""
        path = tmp_path / "checkpoint.pth"
        save_checkpoint(path, model, epoch=1, loss=0.5)

        assert path.exists()
        checkpoint = torch.load(path)
        assert "model_state_dict" in checkpoint
        assert checkpoint["epoch"] == 1

    def test_load_checkpoint(self, model, tmp_path):
        """TC-09: Checkpoint加载测试"""
        # 保存
        path = tmp_path / "checkpoint.pth"
        save_checkpoint(path, model, epoch=1, loss=0.5)

        # 加载到新模型
        new_model = ERes2NetV2(config=test_config)
        checkpoint = load_checkpoint(path, new_model)

        assert checkpoint["epoch"] == 1
        # 验证权重一致
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), new_model.named_parameters()
        ):
            assert torch.allclose(p1, p2)

    # ... 更多测试用例
```

### 5.2 CI/CD 配置

```yaml
# .github/workflows/test_pretrained_weights.yml

name: Pretrained Weights Tests

on:
  push:
    paths:
      - 'models/**'
      - 'utils/checkpoint.py'
      - 'tests/test_pretrained_weights.py'
  pull_request:
    paths:
      - 'models/**'
      - 'utils/checkpoint.py'
      - 'tests/test_pretrained_weights.py'

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10]
        torch-version: [1.12, 2.0]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install torch==${{ matrix.torch-version }}
        pip install pytest pytest-cov

    - name: Download pretrained weights
      run: |
        python scripts/download_pretrained.py --variant all

    - name: Run tests
      run: |
        pytest tests/test_pretrained_weights.py \
          --cov=models \
          --cov=utils/checkpoint \
          --cov-report=xml \
          --cov-report=html \
          -v

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        files: ./coverage.xml
```

### 5.3 测试脚本

```bash
#!/bin/bash
# scripts/run_weight_tests.sh

echo "Running pretrained weight support tests..."

# 下载预训练权重（如果不存在）
if [ ! -f "weights/eres2netv2_pretrained.pth" ]; then
    echo "Downloading pretrained weights..."
    python scripts/download_pretrained.py
fi

# 运行测试
pytest tests/test_pretrained_weights.py \
    -v \
    --cov=models \
    --cov-report=term-missing \
    --cov-report=html:coverage_report \
    --junitxml=test_results.xml

# 检查覆盖率
coverage report --fail-under=90

echo "Tests completed!"
```

---

## 6. 风险点

### 6.1 技术风险

| 风险ID | 风险描述 | 影响程度 | 可能性 | 缓解措施 |
|--------|----------|----------|--------|----------|
| R-01 | 公开预训练权重不可用或版本不兼容 | 高 | 中 | 1. 准备多个权重源<br>2. 实现自动下载和缓存机制<br>3. 提供本地权重转换工具 |
| R-02 | 权重文件格式与模型定义不匹配 | 高 | 中 | 1. 实现严格的键名映射检查<br>2. 提供权重转换脚本<br>3. 详细日志记录 |
| R-03 | GPU/CPU权重加载兼容性问题 | 中 | 低 | 1. 使用 map_location 参数<br>2. 实现设备自动检测<br>3. 跨设备测试 |
| R-04 | 大模型权重加载内存溢出 | 高 | 低 | 1. 实现分块加载<br>2. 内存监控<br>3. 提供低内存模式 |
| R-05 | Checkpoint损坏或版本不兼容 | 中 | 中 | 1. 实现校验和验证<br>2. 版本兼容性检查<br>3. 提供修复工具 |

### 6.2 流程风险

| 风险ID | 风险描述 | 影响程度 | 可能性 | 缓解措施 |
|--------|----------|----------|--------|----------|
| R-06 | 测试用例覆盖不足 | 中 | 中 | 1. 代码审查<br>2. 覆盖率监控<br>3. 边界条件分析 |
| R-07 | 测试数据准备困难 | 低 | 中 | 1. 模拟数据生成脚本<br>2. 共享测试资源<br>3. CI缓存机制 |
| R-08 | 测试环境差异 | 低 | 低 | 1. Docker容器化测试<br>2. 多环境测试矩阵<br>3. 环境检查脚本 |

### 6.3 风险应对计划

```
风险应对流程：
┌─────────────────────────────────────────────────────────┐
│                    风险监控                              │
├─────────────────────────────────────────────────────────┤
│  1. 每日测试执行监控                                    │
│  2. 覆盖率趋势分析                                      │
│  3. 内存使用监控                                        │
│  4. 权重下载成功率监控                                  │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    风险触发                              │
├─────────────────────────────────────────────────────────┤
│  触发条件：                                              │
│  - 测试失败率 > 5%                                      │
│  - 覆盖率 < 90%                                         │
│  - 内存使用 > 阈值                                      │
│  - 权重加载失败                                         │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    应急响应                              │
├─────────────────────────────────────────────────────────┤
│  1. 记录详细错误信息                                    │
│  2. 通知相关负责人                                      │
│  3. 启动应急预案                                        │
│  4. 修复并验证                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 7. 验证执行计划

### 7.1 验证阶段

| 阶段 | 内容 | 时间预估 | 负责人 |
|------|------|----------|--------|
| 阶段一 | 单元测试开发 | 2天 | 开发团队 |
| 阶段二 | 集成测试开发 | 1天 | 开发团队 |
| 阶段三 | 测试执行 | 1天 | QA团队 |
| 阶段四 | 问题修复 | 1天 | 开发团队 |
| 阶段五 | 回归测试 | 0.5天 | QA团队 |

### 7.2 验收清单

```
预训练权重支持验收清单：

□ 功能验收
  □ TC-01: 全量权重加载测试通过
  □ TC-02: 部分权重加载（缺失键）测试通过
  □ TC-03: 部分权重加载（多余键）测试通过
  □ TC-04: 全局冻结策略测试通过
  □ TC-05: 分层冻结策略测试通过
  □ TC-06: BN层冻结策略测试通过
  □ TC-07: 解冻策略测试通过
  □ TC-08: Checkpoint保存测试通过
  □ TC-09: Checkpoint加载测试通过
  □ TC-10: Checkpoint跨设备测试通过
  □ TC-11: 无效权重文件处理测试通过
  □ TC-12: 权重不匹配处理测试通过
  □ TC-13: 训练流程集成测试通过
  □ TC-14: 推理流程集成测试通过

□ 质量验收
  □ 代码覆盖率 ≥ 90%
  □ 所有测试用例通过
  □ 无内存泄漏
  □ 性能指标达标

□ 文档验收
  □ API文档完整
  □ 使用示例完整
  □ 错误处理文档完整

□ 风险验收
  □ 已识别风险有缓解措施
  □ 应急预案已制定
```

---

## 8. 附录

### 8.1 测试数据准备

```python
# scripts/prepare_test_data.py

import torch
from models.eres2netv2 import ERes2NetV2

def create_mock_pretrained_weights():
    """创建模拟预训练权重用于测试"""
    model = ERes2NetV2(config="test")
    # 初始化为特定值以便验证
    for param in model.parameters():
        torch.nn.init.constant_(param, 0.5)
    return model.state_dict()

def create_partial_weights(full_weights, exclude_keys=None):
    """创建部分权重用于测试"""
    if exclude_keys is None:
        exclude_keys = ["classifier"]
    return {k: v for k, v in full_weights.items()
            if not any(ex in k for ex in exclude_keys)}

def create_mismatched_weights():
    """创建不匹配的权重用于测试"""
    return {
        "wrong_key_name": torch.randn(64, 3, 7, 7),
        "another_wrong_key": torch.randn(64)
    }
```

### 8.2 参考资源

1. [PyTorch模型保存与加载](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
2. [ERes2NetV2论文](https://arxiv.org/abs/xxx)
3. [预训练权重下载地址](https://github.com/xxx)

### 8.3 变更历史

| 版本 | 日期 | 变更内容 | 作者 |
|------|------|----------|------|
| v1.0 | 2026/06/02 | 初始版本 | Claude |

---

**文档结束**