# 训练策略 - 验证测试计划

> **文档类型：** 验证测试计划
> **关联任务：** 阶段四：训练流程 - 训练策略
> **创建日期：** 2026/06/03
> **验证范围：** 两阶段训练、渐进式训练、早停策略、训练流程实现

---

## 1. 验证目标

### 1.1 主要验证目标

| 目标ID | 验证目标 | 优先级 | 验证类型 |
|--------|----------|--------|----------|
| VO-001 | 验证两阶段训练流程正确实现 | P0 | 功能验证 |
| VO-002 | 验证浅层冻结机制生效 | P0 | 功能验证 |
| VO-003 | 验证解冻时机与参数释放正确 | P0 | 功能验证 |
| VO-004 | 验证渐进式分辨率训练正确执行 | P1 | 功能验证 |
| VO-005 | 验证分辨率切换时数据/模型适配正确 | P1 | 功能验证 |
| VO-006 | 验证早停策略触发条件正确 | P0 | 功能验证 |
| VO-007 | 验证早停后模型恢复最优状态 | P0 | 功能验证 |
| VO-008 | 验证训练策略组合使用稳定性 | P1 | 性能验证 |

### 1.2 验证范围边界

**包含：**
- Warm-up阶段参数冻结验证
- Fine-tuning阶段参数解冻验证
- 渐进式分辨率切换验证
- 早停触发与恢复验证
- 策略组合集成验证

**不包含：**
- 基础训练循环验证（已在前序任务验证）
- 模型架构正确性（属于模型构建阶段）
- 数据加载正确性（属于数据准备阶段）
- 学习率调度策略（属于优化器配置）

---

## 2. 测试用例

### 2.1 两阶段训练测试

#### TC-001: 参数冻结机制验证

| 属性 | 值 |
|------|-----|
| 测试ID | TC-001 |
| 测试名称 | Warm-up阶段参数冻结验证 |
| 优先级 | P0 |
| 前置条件 | 模型已正确初始化，层级结构已知 |

**测试步骤：**
1. 识别模型各层（按深度编号：layer_0, layer_1, ..., layer_N）
2. 定义冻结策略：冻结前K层（通常K=N/2或前3个stage）
3. 执行冻结操作：设置`requires_grad=False`
4. 验证冻结层参数不参与梯度计算
5. 验证未冻结层参数正常计算梯度
6. 执行一次训练迭代，验证冻结层参数值不变
7. 验证未冻结层参数值更新

**预期结果：**
- 冻结层`requires_grad=False`
- 冻结层梯度为`None`
- 冻结层参数值训练前后相等
- 未冻结层参数值训练前后不等

**验证方法：**
```python
# 伪代码示例
def freeze_shallow_layers(model, freeze_depth=3):
    """冻结浅层参数"""
    frozen_layers = []
    for i, layer in enumerate(model.layers[:freeze_depth]):
        for param in layer.parameters():
            param.requires_grad = False
        frozen_layers.append(layer.__class__.__name__)
    return frozen_layers

# 验证冻结生效
frozen_layers = freeze_shallow_layers(model, freeze_depth=3)

# 训练前保存冻结层参数
frozen_params_before = {
    name: p.clone() for name, p in model.named_parameters()
    if not p.requires_grad
}

# 执行训练迭代
loss = train_step(model, batch)
optimizer.step()

# 验证冻结层参数不变
frozen_params_after = {
    name: p.clone() for name, p in model.named_parameters()
    if not p.requires_grad
}

for name in frozen_params_before:
    assert torch.equal(frozen_params_before[name], frozen_params_after[name]), \
        f"Frozen layer {name} changed unexpectedly"

# 验证未冻结层参数变化
unfrozen_params_before = {
    name: p.clone() for name, p in model.named_parameters()
    if p.requires_grad
}
# ... 类似验证，应不相等
```

---

#### TC-002: 参数解冻时机验证

| 属性 | 值 |
|------|-----|
| 测试ID | TC-002 |
| 测试名称 | Fine-tuning阶段参数解冻验证 |
| 优先级 | P0 |
| 前置条件 | Warm-up阶段已完成，达到解冻条件 |

**测试步骤：**
1. 定义解冻触发条件：
   - 基于epoch数（如epoch >= 5）
   - 基于loss阈值（如loss < 2.0）
   - 基于验证指标（如acc > 70%）
2. 在Warm-up阶段监控触发条件
3. 当条件满足时，执行解冻操作
4. 验证所有层`requires_grad=True`
5. 验证解冻后所有层参与梯度计算
6. 鐰证解冻后学习率是否调整（通常降低）

**预期结果：**
- 解冻后所有参数`requires_grad=True`
- 解冻后所有层梯度非None
- 解冻后学习率按策略调整
- 解冻日志正确记录

**验证方法：**
```python
# 解冻条件检查
def should_unfreeze(current_epoch, current_loss, current_acc, config):
    """判断是否应触发解冻"""
    conditions = config['unfreeze_conditions']
    checks = []

    if 'epoch' in conditions:
        checks.append(current_epoch >= conditions['epoch'])

    if 'loss_threshold' in conditions:
        checks.append(current_loss < conditions['loss_threshold'])

    if 'accuracy_threshold' in conditions:
        checks.append(current_acc > conditions['accuracy_threshold'])

    return any(checks)  # 或 all(checks) 根据策略

# 执行解冻
def unfreeze_all_layers(model):
    """解冻所有参数"""
    for param in model.parameters():
        param.requires_grad = True

# 验证解冻后训练
unfreeze_all_layers(model)
trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_count = sum(p.numel() for p in model.parameters())
assert trainable_count == total_count, "Not all parameters unfrozen"
```

---

#### TC-003: 两阶段学习率切换验证

| 属性 | 值 |
|------|-----|
| 测试ID | TC-003 |
| 测试名称 | 两阶段学习率调整验证 |
| 优先级 | P0 |
| 前置条件 | 配置了不同阶段的学习率策略 |

**测试步骤：**
1. 配置Warm-up阶段学习率（通常较大，如lr=0.01）
2. 配置Fine-tuning阶段学习率（通常较小，如lr=0.001）
3. 在Warm-up阶段记录学习率
4. 触发解冻时，调整学习率
5. 在Fine-tuning阶段记录学习率
6. 验证学习率变化符合预期

**预期结果：**
- Warm-up阶段学习率 = 配置的warmup_lr
- Fine-tuning阶段学习率 = 配置的finetune_lr
- 学习率切换时机与解冻时机一致

**量化指标：**
| 指标 | 通过标准 |
|------|----------|
| Warm-up lr | = config.warmup_lr ± 1% |
| Fine-tune lr | = config.finetune_lr ± 1% |
| 切换时机误差 | ≤ 1 epoch |

---

#### TC-004: 两阶段训练效果对比验证

| 属性 | 值 |
|------|-----|
| 测试ID | TC-004 |
| 测试名称 | 两阶段训练有效性验证 |
| 优先级 | P1 |
| 前置条件 | 准备对照组实验 |

**测试步骤：**
1. 设计对比实验：
   - 组A：直接全参数训练（从头到尾）
   - 组B：两阶段训练（Warm-up + Fine-tuning）
2. 使用相同数据、相同超参数（除阶段策略）
3. 训练相同epoch数
4. 对比最终模型性能
5. 对比训练稳定性（loss曲线平滑度）
6. 对比收敛速度

**预期结果：**
- 两阶段训练最终性能 ≥ 直接训练
- 两阶段训练loss曲线更平滑
- 两阶段训练收敛更稳定

**量化指标：**
| 指标 | 通过标准 |
|------|----------|
| 性能提升 | ≥ 0% vs 直接训练 |
| Loss波动减少 | ≥ 20% vs 直接训练 |
| 收敛epoch减少 | ≤ 直接训练epoch数 |

---

### 2.2 渐进式分辨率训练测试

#### TC-005: 分辨率切换机制验证

| 属性 | 值 |
|------|-----|
| 测试ID | TC-005 |
| 测试名称 | 渐进式分辨率切换验证 |
| 优先级 | P1 |
| 前置条件 | 配置多阶段分辨率计划 |

**测试步骤：**
1. 定义分辨率渐进计划：
   - Stage 1: 64x64 (epoch 0-5)
   - Stage 2: 128x128 (epoch 6-10)
   - Stage 3: 224x224 (epoch 11+)
2. 在每个阶段边界检查当前分辨率
3. 验证分辨率切换时机正确
4. 验证数据增强适配新分辨率
5. 验证模型输入适配新分辨率
6. 执行训练迭代验证无错误

**预期结果：**
- 分辨率按计划切换
- 数据加载器输出正确尺寸
- 模型接收正确尺寸输入
- 切换时无OOM或错误

**验证方法：**
```python
# 渐进式分辨率配置
RESOLUTION_SCHEDULE = {
    0: 64,    # epoch 0-5: 64x64
    6: 128,   # epoch 6-10: 128x128
    11: 224,  # epoch 11+: 224x224
}

def get_current_resolution(epoch):
    """获取当前epoch应使用的分辨率"""
    for start_epoch in sorted(RESOLUTION_SCHEDULE.keys(), reverse=True):
        if epoch >= start_epoch:
            return RESOLUTION_SCHEDULE[start_epoch]
    return RESOLUTION_SCHEDULE[0]

# 验证切换
for test_epoch in [0, 5, 6, 10, 11, 20]:
    expected_res = get_current_resolution(test_epoch)
    actual_res = dataloader.get_resolution(test_epoch)
    assert actual_res == expected_res, f"Epoch {test_epoch}: expected {expected_res}, got {actual_res}"
```

---

#### TC-006: 分辨率切换时数据适配验证

| 属性 | 值 |
|------|-----|
| 测试ID | TC-006 |
| 测试名称 | 分辨率切换数据适配验证 |
| 优先级 | P1 |
| 前置条件 | 数据增强pipeline支持动态分辨率 |

**测试步骤：**
1. 配置数据增强pipeline，支持resolution参数
2. 在分辨率切换点（如epoch 5→6）执行以下验证：
3. 验证图像resize操作正确执行
4. 验证数据增强参数适配新分辨率
5. 验证batch shape变化正确
6. 验证切换后首个batch无异常数据

**预期结果：**
- 图像正确resize到新分辨率
- 数据增强参数适配（如crop_size调整）
- Batch shape从(B, C, H_old, W_old)变为(B, C, H_new, W_new)
- 无异常值（NaN/极端值）

**验证方法：**
```python
# 验证分辨率切换时的数据处理
def verify_resolution_switch(dataloader, old_res, new_res):
    """验证分辨率切换"""
    # 获取切换前最后一个batch
    old_batch = dataloader.get_batch(epoch=5, batch_idx=-1)
    assert old_batch.shape[-2:] == (old_res, old_res)

    # 获取切换后第一个batch
    new_batch = dataloader.get_batch(epoch=6, batch_idx=0)
    assert new_batch.shape[-2:] == (new_res, new_res)

    # 验证数据范围正常
    assert new_batch.min() >= 0 and new_batch.max() <= 1  # 假设归一化到[0,1]
    assert not torch.isnan(new_batch).any()
    assert not torch.isinf(new_batch).any()
```

---

#### TC-007: 分辨率切换时学习率调整验证

| 属性 | 值 |
|------|-----|
| 测试ID | TC-007 |
| 测试名称 | 渐进式训练学习率调整验证 |
| 优先级 | P2 |
| 前置条件 | 配置了分辨率-学习率关联策略 |

**测试步骤：**
1. 定义分辨率-学习率关系：
   - 小分辨率：较大学习率（快速学习粗粒度特征）
   - 大分辨率：较小学习率（精细调整）
2. 在分辨率切换时验证学习率变化
3. 验证学习率调整比例符合预期
4. 记录调整后的训练稳定性

**预期结果：**
- 分辨率增大时学习率按比例降低
- 学习率调整系数符合配置
- 调整后训练继续稳定进行

**验证方法：**
```python
# 分辨率-学习率关系
LR_SCALE_FACTOR = {
    64: 1.0,    # 基准
    128: 0.7,   # 128x128时学习率降低
    224: 0.5,   # 224x224时学习率进一步降低
}

def adjust_lr_for_resolution(optimizer, resolution, base_lr):
    """根据分辨率调整学习率"""
    scale = LR_SCALE_FACTOR.get(resolution, 1.0)
    new_lr = base_lr * scale
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr

# 验证
base_lr = 0.01
for res in [64, 128, 224]:
    expected_lr = base_lr * LR_SCALE_FACTOR[res]
    actual_lr = adjust_lr_for_resolution(optimizer, res, base_lr)
    assert abs(actual_lr - expected_lr) < 1e-6
```

---

#### TC-008: 渐进式训练效果验证

| 属性 | 值 |
|------|-----|
| 测试ID | TC-008 |
| 测试名称 | 渐进式训练有效性验证 |
| 优先级 | P2 |
| 前置条件 | 准备对比实验 |

**测试步骤：**
1. 设计对比实验：
   - 组A：直接使用目标分辨率训练
   - 组B：渐进式分辨率训练
2. 训练相同总epoch数
3. 对比最终性能
4. 对比训练时间
5. 对比训练稳定性

**预期结果：**
- 渐进式训练最终性能 ≥ 直接训练
- 渐进式训练总时间可能更长（因多次resize）
- 渐进式训练稳定性更好

---

### 2.3 早停策略测试

#### TC-009: 早停触发条件验证

| 属性 | 值 |
|------|-----|
| 测试ID | TC-009 |
| 测试名称 | 早停触发条件验证 |
| 优先级 | P0 |
| 前置条件 | 配置早停参数：patience、min_delta、monitor指标 |

**测试步骤：**
1. 配置早停参数：
   - `patience=10`（容忍10个epoch无改善）
   - `min_delta=0.001`（最小改善阈值）
   - `monitor='val_loss'`（监控验证损失）
2. 构造测试场景：
   - 场景A：连续10个epoch验证loss无改善
   - 场景B：验证loss在第8个epoch改善
   - 场景C：验证loss持续改善
3. 验证各场景早停触发行为

**预期结果：**
| 场景 | 预期行为 |
|------|----------|
| A | 第11个epoch触发早停 |
| B | 计数器重置，不触发 |
| C | 不触发早停 |

**验证方法：**
```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, monitor='val_loss', mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def step(self, current_score):
        """检查是否应早停"""
        if self.best_score is None:
            self.best_score = current_score
            return False

        if self.mode == 'min':
            improved = current_score < self.best_score - self.min_delta
        else:
            improved = current_score > self.best_score + self.min_delta

        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False

# 验证场景A：连续无改善
es = EarlyStopping(patience=3, min_delta=0.001)
val_losses = [1.0, 1.0, 1.0, 1.0, 1.0]  # 无改善
for i, loss in enumerate(val_losses):
    should_stop = es.step(loss)
    if i < 3:
        assert not should_stop, f"Early stop triggered too early at epoch {i}"
    else:
        assert should_stop, f"Early stop should trigger at epoch {i}"
```

---

#### TC-010: 早停最优模型恢复验证

| 属性 | 值 |
|------|-----|
| 测试ID | TC-010 |
| 测试名称 | 早停后最优模型恢复验证 |
| 优先级 | P0 |
| 前置条件 | 早停已触发，最优checkpoint已保存 |

**测试步骤：**
1. 配置早停保存最优模型
2. 训练过程中记录最优epoch及对应模型参数
3. 触发早停后：
   - 验证最优epoch标识正确
   - 验证模型参数恢复为最优状态
   - 验证恢复后性能指标与最优epoch一致
4. 比较恢复模型与最后epoch模型的性能差异

**预期结果：**
- 恢复的模型参数与最优epoch一致
- 恢复后验证指标与最优epoch指标相等
- 恢复模型性能 ≥ 最后epoch模型

**验证方法：**
```python
class EarlyStoppingWithCheckpoint(EarlyStopping):
    def __init__(self, save_path, **kwargs):
        super().__init__(**kwargs)
        self.save_path = save_path
        self.best_checkpoint = None

    def step(self, current_score, model):
        """检查并保存最优模型"""
        if self.best_score is None or self._is_improved(current_score):
            self.best_score = current_score
            self.best_checkpoint = {
                'model_state_dict': model.state_dict(),
                'score': current_score,
                'epoch': current_epoch
            }
            torch.save(self.best_checkpoint, self.save_path)

        return super().step(current_score)

    def restore_best(self, model):
        """恢复最优模型"""
        checkpoint = torch.load(self.save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint['score'], checkpoint['epoch']

# 验证
es = EarlyStoppingWithCheckpoint('best_model.pth', patience=3)
# 训练过程...
# 早停触发后
best_score, best_epoch = es.restore_best(model)
# 验证恢复效果
current_score = evaluate(model, val_dataset)
assert abs(current_score - best_score) < 0.001, "Restored model performance mismatch"
```

---

#### TC-011: 早停与学习率调度协调验证

| 属性 | 值 |
|------|-----|
| 测试ID | TC-011 |
| 测试名称 | 早停与学习率调度协调验证 |
| 优先级 | P1 |
| 前置条件 | 同时配置了早停和学习率调度器 |

**测试步骤：**
1. 配置ReduceLROnPlateau学习率调度器
2. 配置早停策略
3. 验证两者协调工作：
   - 学习率降低后，早停计数器是否重置？
   - 学习率降至最小值后，早停应生效
4. 模拟验证loss平台期场景

**预期结果：**
- 学习率降低触发时，早停计数器可重置（配置项）
- 学习率降至最小值后，早停正常触发
- 两者不相互冲突

---

#### TC-012: 早停边界条件测试

| 属性 | 值 |
|------|-----|
| 测试ID | TC-012 |
| 测试名称 | 早停边界条件验证 |
| 优先级 | P1 |
| 前置条件 | 配置各种边界值 |

**测试步骤：**
1. 测试边界配置：
   - `patience=0`：任何无改善立即触发
   - `patience=100`：几乎不触发
   - `min_delta=0`：任何改善都有效
   - `min_delta=1.0`：几乎不可能改善
2. 验证各边界下的行为

**预期结果：**
| 配置 | 预期行为 |
|------|----------|
| patience=0 | 第一个无改善epoch触发 |
| patience=100 | 训练正常结束 |
| min_delta=0 | 任何改善重置计数器 |
| min_delta=1.0 | 很难达到改善阈值 |

---

### 2.4 策略组合集成测试

#### TC-013: 两阶段+早停组合验证

| 属性 | 值 |
|------|-----|
| 测试ID | TC-013 |
| 测试名称 | 两阶段训练与早停组合验证 |
| 优先级 | P0 |
| 前置条件 | 两阶段训练与早停策略均已实现 |

**测试步骤：**
1. 配置两阶段训练+早停策略
2. 场景A：Warm-up阶段触发早停
   - 验证早停是否生效
   - 验证是否保存Warm-up阶段最优模型
3. 场景B：Fine-tuning阶段触发早停
   - 验证早停是否生效
   - 验证解冻后早停计数器行为
4. 验证两阶段切换时早停计数器处理

**预期结果：**
| 场景 | 预期行为 |
|------|----------|
| Warm-up早停 | 正常触发，保存Warm-up最优模型 |
| Fine-tuning早停 | 正常触发，保存Fine-tuning最优模型 |
| 阶段切换 | 早停计数器可选重置 |

**验证方法：**
```python
# 两阶段+早停组合训练
def train_with_two_stage_and_early_stop(config):
    # Warm-up阶段
    freeze_shallow_layers(model, config.freeze_depth)
    es = EarlyStopping(patience=config.patience)

    for epoch in range(config.warmup_epochs):
        train_loss = train_epoch(model, train_loader)
        val_loss = validate(model, val_loader)

        if es.step(val_loss, model):
            print(f"Early stop in Warm-up at epoch {epoch}")
            es.restore_best(model)
            return 'warmup_early_stop'

        if should_unfreeze(epoch, train_loss):
            break

    # Fine-tuning阶段
    unfreeze_all_layers(model)
    adjust_learning_rate(optimizer, config.finetune_lr)
    es.counter = 0  # 可选：重置早停计数器

    for epoch in range(config.finetune_epochs):
        # ... 类似训练流程
        if es.step(val_loss, model):
            print(f"Early stop in Fine-tuning at epoch {epoch}")
            es.restore_best(model)
            return 'finetune_early_stop'

    return 'completed'
```

---

#### TC-014: 渐进式+两阶段组合验证

| 属性 | 值 |
|------|-----|
| 测试ID | TC-014 |
| 测试名称 | 渐进式训练与两阶段组合验证 |
| 优先级 | P1 |
| 前置条件 | 渐进式训练与两阶段策略均已实现 |

**测试步骤：**
1. 配置渐进式分辨率+两阶段训练
2. 验证分辨率切换与阶段切换的协调：
   - 分辨率切换是否影响冻结状态？
   - 阶段切换是否影响分辨率计划？
3. 执行完整训练流程
4. 验证各策略按预期执行

**预期结果：**
- 分辨率切换不改变冻结状态
- 阶段切换不改变分辨率计划
- 策略组合无冲突

---

#### TC-015: 全策略组合端到端验证

| 属性 | 值 |
|------|-----|
| 测试ID | TC-015 |
| 测试名称 | 全策略组合端到端验证 |
| 优先级 | P0 |
| 前置条件 | 所有策略均已实现 |

**测试步骤：**
1. 配置完整策略组合：
   - 两阶段训练：Warm-up 5 epochs冻结，Fine-tuning解冻
   - 渐进式分辨率：64→128→224
   - 早停：patience=10
2. 执行完整训练流程
3. 记录各策略触发时机
4. 验证训练日志完整性
5. 验证最终模型性能

**验证清单：**
- [ ] Warm-up阶段正确执行
- [ ] 参数冻结/解冻正确
- [ ] 分辨率正确切换
- [ ] 学习率正确调整
- [ ] 早停正确触发（如适用）
- [ ] 最优模型正确保存/恢复
- [ ] 训练日志完整

---

## 3. 验证方法

### 3.1 单元测试方法

```python
# tests/test_training_strategy.py

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

class TestTwoStageTraining:
    """两阶段训练单元测试"""

    @pytest.fixture
    def model(self):
        """测试用分层模型"""
        class LayeredModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_0 = nn.Linear(10, 20)
                self.layer_1 = nn.Linear(20, 30)
                self.layer_2 = nn.Linear(30, 40)
                self.layer_3 = nn.Linear(40, 1)

            def forward(self, x):
                x = self.layer_0(x)
                x = self.layer_1(x)
                x = self.layer_2(x)
                x = self.layer_3(x)
                return x

        return LayeredModel()

    def test_freeze_layers(self, model):
        """TC-001: 参数冻结验证"""
        # 冻结前2层
        freeze_shallow_layers(model, freeze_depth=2)

        # 验证冻结层
        frozen_count = 0
        for name, param in model.named_parameters():
            if 'layer_0' in name or 'layer_1' in name:
                assert not param.requires_grad, f"{name} should be frozen"
                frozen_count += 1
            else:
                assert param.requires_grad, f"{name} should be trainable"

        assert frozen_count > 0, "No layers were frozen"

    def test_freeze_prevents_update(self, model):
        """TC-001: 冻结层不更新"""
        freeze_shallow_layers(model, freeze_depth=2)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randn(4, 10)
        y = torch.randn(4, 1)

        # 保存冻结层初始参数
        frozen_params_before = {
            n: p.clone() for n, p in model.named_parameters()
            if not p.requires_grad
        }

        # 训练迭代
        output = model(x)
        loss = nn.MSELoss()(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 验证冻结层不变
        for name in frozen_params_before:
            assert torch.equal(
                frozen_params_before[name],
                model.state_dict()[name]
            ), f"Frozen layer {name} changed"

    def test_unfreeze_all(self, model):
        """TC-002: 参数解冻验证"""
        # 先冻结
        freeze_shallow_layers(model, freeze_depth=2)

        # 解冻
        unfreeze_all_layers(model)

        # 验证所有层可训练
        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        total = sum(1 for p in model.parameters())
        assert trainable == total, "Not all layers unfrozen"

    def test_unfreeze_condition(self, model):
        """TC-002: 解冻触发条件"""
        config = {'unfreeze_conditions': {'epoch': 5}}

        # 测试各epoch
        assert not should_unfreeze(0, 1.0, 50, config)
        assert not should_unfreeze(4, 1.0, 50, config)
        assert should_unfreeze(5, 1.0, 50, config)
        assert should_unfreeze(10, 1.0, 50, config)


class TestProgressiveResolution:
    """渐进式分辨率训练单元测试"""

    def test_resolution_schedule(self):
        """TC-005: 分辨率计划验证"""
        RESOLUTION_SCHEDULE = {0: 64, 6: 128, 11: 224}

        test_cases = [
            (0, 64), (5, 64), (6, 128), (10, 128),
            (11, 224), (20, 224)
        ]

        for epoch, expected_res in test_cases:
            actual_res = get_current_resolution(epoch, RESOLUTION_SCHEDULE)
            assert actual_res == expected_res, f"Epoch {epoch}: expected {expected_res}, got {actual_res}"

    def test_resolution_switch(self):
        """TC-006: 分辨率切换验证"""
        dataloader = MockDataLoader()

        # 切换前
        dataloader.set_resolution(64)
        batch = dataloader.get_batch()
        assert batch.shape[-2:] == (64, 64)

        # 切换后
        dataloader.set_resolution(128)
        batch = dataloader.get_batch()
        assert batch.shape[-2:] == (128, 128)


class TestEarlyStopping:
    """早停策略单元测试"""

    def test_early_stop_trigger(self):
        """TC-009: 早停触发验证"""
        es = EarlyStopping(patience=3, min_delta=0.001)

        # 模拟loss序列：改善后连续无改善
        losses = [1.0, 0.9, 0.9, 0.9, 0.9]

        results = []
        for loss in losses:
            results.append(es.step(loss))

        # 验证触发时机
        assert not results[0]  # 第一个epoch不触发
        assert not results[1]  # 改善，不触发
        assert not results[2]  # patience计数1
        assert not results[3]  # patience计数2
        assert results[4]      # patience计数3，触发

    def test_early_stop_with_improvement(self):
        """TC-009: 改善重置计数器"""
        es = EarlyStopping(patience=3, min_delta=0.001)

        losses = [1.0, 1.0, 0.8, 0.8, 0.8, 0.8]
        results = [es.step(l) for l in losses]

        # 改善后计数器重置
        assert not any(results), "Should not trigger with improvements"

    def test_best_model_restore(self):
        """TC-010: 最优模型恢复"""
        model = nn.Linear(10, 1)
        es = EarlyStoppingWithCheckpoint(
            save_path='test_checkpoint.pth',
            patience=3
        )

        # 模拟训练
        best_epoch = 2
        best_loss = 0.5
        for epoch, loss in enumerate([1.0, 0.8, 0.5, 0.6, 0.6]):
            es.step(loss, model)

        # 验证最优epoch记录
        assert es.best_score == best_loss
        assert es.best_epoch == best_epoch
```

### 3.2 集成测试方法

```python
# tests/integration/test_training_strategy_integration.py

import pytest
import torch
from training import TwoStageTrainer, ProgressiveTrainer, EarlyStoppingTrainer

class TestTrainingStrategyIntegration:
    """训练策略集成测试"""

    @pytest.fixture
    def config(self):
        """测试配置"""
        return {
            'model': 'eres2netv2',
            'warmup_epochs': 5,
            'finetune_epochs': 20,
            'freeze_depth': 3,
            'warmup_lr': 0.01,
            'finetune_lr': 0.001,
            'resolution_schedule': {0: 64, 6: 128, 11: 224},
            'patience': 10,
            'min_delta': 0.001,
        }

    def test_two_stage_full_cycle(self, config):
        """TC-013: 两阶段完整周期"""
        trainer = TwoStageTrainer(config)
        history = trainer.train()

        # 验证阶段记录
        assert 'warmup_history' in history
        assert 'finetune_history' in history

        # 验证Warm-up完成
        assert len(history['warmup_history']['loss']) == config['warmup_epochs']

        # 验证解冻时机
        assert history['unfreeze_epoch'] == config['warmup_epochs']

    def test_early_stop_integration(self, config):
        """TC-013: 早停集成"""
        # 设置较短patience强制触发
        config['patience'] = 3
        config['warmup_epochs'] = 10

        trainer = EarlyStoppingTrainer(config)
        history = trainer.train()

        # 验证早停触发
        assert history['early_stopped'] == True
        assert history['stopped_epoch'] < config['warmup_epochs'] + config['finetune_epochs']

        # 验证最优模型恢复
        assert history['restored_epoch'] == history['best_epoch']

    def test_full_strategy_combination(self, config):
        """TC-015: 全策略组合"""
        trainer = CombinedTrainer(config)
        history = trainer.train()

        # 验证各策略执行
        assert history['warmup_completed'] or history['early_stopped']

        # 验证分辨率切换记录
        if 'resolution_changes' in history:
            for change in history['resolution_changes']:
                assert change['epoch'] in config['resolution_schedule'].keys()

        # 验证最终模型可用
        model = trainer.get_best_model()
        assert model is not None
        test_output = model(torch.randn(1, 3, 224, 224))
        assert not torch.isnan(test_output).any()
```

### 3.3 手动验证清单

| 检查项 | 验证方法 | 通过标准 |
|--------|----------|----------|
| 两阶段训练日志 | 检查训练日志 | 显示"Warm-up phase"和"Fine-tuning phase" |
| 冻结层参数统计 | 打印requires_grad统计 | 冻结层requires_grad=False |
| 解冻时机日志 | 检查日志 | 显示"Unfreezing all layers" |
| 分辨率切换日志 | 检查日志 | 显示"Resolution changed to XXX" |
| 早停触发日志 | 检查日志 | 显示"Early stopping triggered" |
| 最优模型恢复 | 检查checkpoint文件 | 存在best_model.pth |
| TensorBoard曲线 | 打开TensorBoard | 显示阶段分隔线 |

---

## 4. 通过标准

### 4.1 必须通过的测试用例

| 测试ID | 测试名称 | 必须通过原因 |
|--------|----------|--------------|
| TC-001 | Warm-up阶段参数冻结验证 | 两阶段训练核心功能 |
| TC-002 | Fine-tuning阶段参数解冻验证 | 两阶段训练核心功能 |
| TC-003 | 两阶段学习率切换验证 | 训练策略有效性保证 |
| TC-009 | 早停触发条件验证 | 早停核心功能 |
| TC-010 | 早停后最优模型恢复验证 | 早停有效性保证 |
| TC-013 | 两阶段+早停组合验证 | 实际使用场景 |
| TC-015 | 全策略组合端到端验证 | 整体功能验证 |

### 4.2 量化通过标准

| 指标 | 通过标准 | 测量方法 |
|------|----------|----------|
| 测试用例通过率 | ≥ 90% | pytest执行结果 |
| 冻结层参数更新率 | = 0% | 参数前后对比 |
| 解冻层参数更新率 | > 0% | 参数前后对比 |
| 早停触发准确率 | = 100% | 手动场景验证 |
| 最优模型恢复成功率 | = 100% | 指标对比验证 |
| 分辨率切换成功率 | = 100% | 数据shape验证 |
| 训练稳定性 | 无NaN/Inf异常 | 日志分析 |

### 4.3 验收签字

| 角色 | 姓名 | 签字 | 日期 |
|------|------|------|------|
| 开发者 | | | |
| 测试工程师 | | | |
| 技术负责人 | | | |

---

## 5. 自动化建议

### 5.1 单元测试自动化

```yaml
# .github/workflows/test-training-strategy.yml
name: Training Strategy Tests

on:
  push:
    paths:
      - 'src/training/strategy/**'
      - 'tests/test_training_strategy.py'
  pull_request:
    branches: [main]

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
          pip install pytest pytest-cov pytest-mock
      - name: Run unit tests
        run: pytest tests/test_training_strategy.py -v --cov=src/training/strategy
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### 5.2 集成测试自动化

```python
# scripts/run_strategy_verification.py

import argparse
import json
import subprocess
import sys
from pathlib import Path

def verify_two_stage_training(config_path):
    """验证两阶段训练"""
    result = subprocess.run(
        ['python', 'train.py', '--config', config_path, '--strategy', 'two_stage'],
        capture_output=True, text=True
    )

    # 解析日志验证阶段切换
    logs = result.stdout

    checks = {
        'warmup_started': 'Warm-up phase started' in logs,
        'freeze_applied': 'Frozen layers:' in logs,
        'unfreeze_triggered': 'Unfreezing all layers' in logs,
        'finetune_started': 'Fine-tuning phase started' in logs,
        'lr_adjusted': 'Learning rate adjusted' in logs,
    }

    return checks

def verify_early_stopping(config_path):
    """验证早停策略"""
    # 使用会触发早停的配置
    result = subprocess.run(
        ['python', 'train.py', '--config', config_path, '--patience', '3'],
        capture_output=True, text=True
    )

    logs = result.stdout

    checks = {
        'early_stop_triggered': 'Early stopping triggered' in logs,
        'best_epoch_recorded': 'Best epoch:' in logs,
        'model_restored': 'Restoring best model' in logs,
        'final_performance': 'Final validation loss' in logs,
    }

    return checks

def verify_progressive_resolution(config_path):
    """验证渐进式分辨率"""
    result = subprocess.run(
        ['python', 'train.py', '--config', config_path, '--strategy', 'progressive'],
        capture_output=True, text=True
    )

    logs = result.stdout

    # 检查分辨率切换记录
    resolution_changes = []
    for line in logs.split('\n'):
        if 'Resolution changed to' in line:
            # 提取分辨率值
            res = int(line.split('to')[-1].strip())
            resolution_changes.append(res)

    checks = {
        'resolution_switched': len(resolution_changes) > 0,
        'correct_sequence': resolution_changes in [[64, 128], [64, 128, 224], [128, 224]],
        'no_errors': 'Error' not in logs and 'OOM' not in logs,
    }

    return checks

def generate_report(results):
    """生成验证报告"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'overall_passed': all(all(r.values()) for r in results.values()),
        'results': results
    }

    report_path = Path('verification_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Report saved to {report_path}")
    return report['overall_passed']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    results = {
        'two_stage': verify_two_stage_training(args.config),
        'early_stopping': verify_early_stopping(args.config),
        'progressive': verify_progressive_resolution(args.config),
    }

    success = generate_report(results)
    sys.exit(0 if success else 1)
```

### 5.3 训练配置文件验证

```python
# scripts/validate_strategy_config.py

import yaml
from pathlib import Path

STRATEGY_CONFIG_SCHEMA = {
    'two_stage': {
        'required': ['warmup_epochs', 'finetune_epochs', 'freeze_depth'],
        'optional': ['warmup_lr', 'finetune_lr', 'unfreeze_conditions'],
    },
    'progressive': {
        'required': ['resolution_schedule'],
        'optional': ['lr_scale_factor', 'switch_epochs'],
    },
    'early_stopping': {
        'required': ['patience'],
        'optional': ['min_delta', 'monitor', 'mode', 'restore_best'],
    }
}

def validate_config(config_path):
    """验证策略配置文件"""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    errors = []

    for strategy, schema in STRATEGY_CONFIG_SCHEMA.items():
        if strategy in config:
            strategy_config = config[strategy]

            # 检查必需字段
            for field in schema['required']:
                if field not in strategy_config:
                    errors.append(f"{strategy}: missing required field '{field}'")

            # 检查字段类型和范围
            if strategy == 'two_stage':
                if strategy_config.get('warmup_epochs', 0) < 1:
                    errors.append("two_stage: warmup_epochs must be >= 1")
                if strategy_config.get('freeze_depth', 0) < 0:
                    errors.append("two_stage: freeze_depth must be >= 0")

            elif strategy == 'progressive':
                schedule = strategy_config.get('resolution_schedule', {})
                for epoch, res in schedule.items():
                    if res not in [32, 64, 128, 224, 256]:
                        errors.append(f"progressive: unsupported resolution {res}")

            elif strategy == 'early_stopping':
                if strategy_config.get('patience', 0) < 0:
                    errors.append("early_stopping: patience must be >= 0")
                if strategy_config.get('min_delta', 0) < 0:
                    errors.append("early_stopping: min_delta must be >= 0")

    return errors
```

### 5.4 监控告警

```python
# 建议的监控告警规则

TRAINING_STRATEGY_ALERTS = {
    'freeze_failure': {
        'condition': 'frozen_layer_param_changed',
        'action': 'stop_training',
        'notification': 'slack',
        'message': 'Frozen layer parameters changed unexpectedly'
    },
    'unfreeze_missed': {
        'condition': 'warmup_epochs_passed_but_not_unfrozen',
        'action': 'log_warning',
        'notification': 'dashboard',
        'message': 'Unfreeze condition not triggered after warmup'
    },
    'resolution_switch_failure': {
        'condition': 'resolution_mismatch_at_epoch_boundary',
        'action': 'stop_training',
        'notification': 'email',
        'message': 'Resolution switch failed'
    },
    'early_stop_no_restore': {
        'condition': 'early_stopped_without_best_model_restore',
        'action': 'log_error',
        'notification': 'slack',
        'message': 'Early stop triggered but best model not restored'
    }
}
```

---

## 6. 风险点

### 6.1 高风险项

| 风险ID | 风险描述 | 影响 | 缓解措施 |
|--------|----------|------|----------|
| R-001 | 冻结层参数意外更新 | Warm-up无效 | TC-001严格验证，日志监控 |
| R-002 | 解冻时机判断错误 | Fine-tuning延迟或过早 | 多条件触发机制，手动可干预 |
| R-003 | 早停最优模型恢复失败 | 使用次优模型 | TC-010验证，checkpoint完整性检查 |
| R-004 | 分辨率切换OOM | 训练中断 | 渐进式batch size调整，预检查显存 |

### 6.2 中风险项

| 风险ID | 风险描述 | 影响 | 缓解措施 |
|--------|----------|------|----------|
| R-005 | 两阶段学习率切换时机不一致 | 训练不稳定 | 统一触发机制，日志记录 |
| R-006 | 早停与学习率调度冲突 | 早停误触发 | 配置协调，ReduceLROnPlateau联动 |
| R-007 | 分辨率切换数据异常 | 训练数据错误 | TC-006验证，数据完整性检查 |
| R-008 | 策略组合执行顺序错误 | 训练流程混乱 | 明确执行顺序，状态机设计 |

### 6.3 低风险项

| 风险ID | 风险描述 | 影响 | 缓解措施 |
|--------|----------|------|----------|
| R-009 | 日志信息不完整 | 问题排查困难 | 标准化日志格式 |
| R-010 | 配置文件格式错误 | 策略未生效 | 配置验证脚本 |
| R-011 | checkpoint文件损坏 | 模型无法恢复 | 多副本保存，校验检查 |

### 6.4 风险应对矩阵

```
影响
  高 │ R-001, R-002 │ R-003, R-004 │
     │              │              │
  中 │ R-005, R-006 │ R-007, R-008 │
     │              │              │
  低 │              │ R-009, R-010 │ R-011
     └──────────────┴──────────────┴──────────
         高              中            低
                    概率
```

---

## 7. 测试数据准备

### 7.1 策略验证数据集

```python
# 测试数据配置

STRATEGY_TEST_DATASETS = {
    'quick_verify': {
        'samples': 50,
        'epochs': 5,
        'purpose': '快速验证冻结/解冻机制',
        'expected_time': '<5分钟'
    },
    'early_stop_test': {
        'samples': 200,
        'epochs': 30,
        'patience': 5,
        'purpose': '早停触发验证',
        'expected_time': '10-15分钟'
    },
    'resolution_test': {
        'samples': 100,
        'epochs': 15,
        'purpose': '渐进式分辨率验证',
        'expected_time': '15-20分钟'
    },
    'full_integration': {
        'samples': 500,
        'epochs': 30,
        'purpose': '全策略组合验证',
        'expected_time': '30-45分钟'
    }
}
```

### 7.2 边界测试用例

| 场景 | 输入配置 | 预期行为 |
|------|----------|----------|
| 冻结深度为0 | freeze_depth=0 | 所有层从头开始训练 |
| 冻结深度为全部 | freeze_depth=N | Warm-up阶段无参数更新 |
| Warm-up 1 epoch | warmup_epochs=1 | 快速进入Fine-tuning |
| patience=1 | patience=1 | 任何无改善立即触发早停 |
| 分辨率跳跃 | 32→224 | 正常切换或显存不足提示 |
| 分辨率计划冲突 | 多个同epoch不同分辨率 | 配置验证报错 |

---

## 8. 验证执行计划

### 8.1 验证阶段

| 阶段 | 内容 | 预计时间 | 负责人 |
|------|------|----------|--------|
| 阶段1 | 单元测试（冻结/解冻） | 30分钟 | 开发者 |
| 阶段2 | 单元测试（早停） | 30分钟 | 开发者 |
| 阶段3 | 单元测试（渐进式） | 30分钟 | 开发者 |
| 阶段4 | 集成测试（策略组合） | 1小时 | 测试工程师 |
| 阶段5 | 效果对比验证 | 2小时 | 测试工程师 |
| 阶段6 | 回归测试 | 30分钟 | 自动化 |

### 8.2 验证执行顺序

```
┌─────────────────────────────────────────────────────────────┐
│                     验证执行流程                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                 │
│  │ TC-001  │───▶│ TC-002  │───▶│ TC-003  │                 │
│  │冻结验证 │    │解冻验证 │    │LR切换  │                 │
│  └─────────┘    └─────────┘    └─────────┘                 │
│       │                                               │      │
│       │                                               │      │
│       ▼                                               ▼      │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │ TC-004  │    │ TC-005  │───▶│ TC-006  │───▶│ TC-007  │  │
│  │效果对比 │    │分辨率切换│    │数据适配│    │LR适配  │  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘  │
│       │                                               │      │
│       │                                               │      │
│       ▼                                               ▼      │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │ TC-008  │    │ TC-009  │───▶│ TC-010  │───▶│ TC-011  │  │
│  │渐进效果 │    │早停触发 │    │模型恢复│    │LR协调  │  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘  │
│       │                                               │      │
│       │                                               │      │
│       └───────────────────────┬───────────────────────┘      │
│                               ▼                              │
│                        ┌─────────┐                          │
│                        │ TC-012  │                          │
│                        │边界条件 │                          │
│                        └─────────┘                          │
│                               │                              │
│                               ▼                              │
│                   ┌───────────────────┐                     │
│                   │ TC-013/14/15      │                     │
│                   │ 策略组合集成测试   │                     │
│                   └───────────────────┘                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.3 验证优先级矩阵

```
优先级
  P0 │ TC-001  TC-002 │ TC-009  TC-010 │ TC-013  TC-015
     │ 冻结/解冻核心   │ 早停核心功能   │ 组合集成
     │                │                │
  P1 │ TC-003  TC-004 │ TC-005  TC-006 │ TC-014
     │ LR切换/效果    │ 分辨率核心     │ 组合验证
     │                │                │
  P2 │                │ TC-007  TC-008 │ TC-011  TC-012
     │                │ LR适配/效果    │ 协调/边界
     └────────────────┴────────────────┴──────────
           功能验证         验证扩展        边界验证
```

---

## 9. 附录

### 9.1 相关文档

- [PyTorch参数冻结指南](https://pytorch.org/docs/stable/notes/autograd.html)
- [早停策略最佳实践](https://machinelearningmastery.com/early-stopping-to-avoid-overfitting/)
- [渐进式训练论文参考](https://arxiv.org/abs/1804.07635)
- [两阶段训练策略](./训练策略说明.md)

### 9.2 测试工具版本

| 工具 | 版本 | 用途 |
|------|------|------|
| pytest | ≥7.0 | 单元测试框架 |
| pytest-mock | ≥3.0 | Mock测试支持 |
| torch | ≥1.12 | 深度学习框架 |
| tensorboard | ≥2.10 | 可视化工具 |
| yaml | ≥5.0 | 配置文件解析 |

### 9.3 验收标准对照

| 原始验收标准 | 对应测试用例 | 验证方法 |
|--------------|--------------|----------|
| 两阶段训练流程实现 | TC-001, TC-002, TC-003 | 单元测试 + 集成测试 |
| 早停策略生效 | TC-009, TC-010, TC-011 | 单元测试 + 场景验证 |

### 9.4 修订历史

| 版本 | 日期 | 作者 | 变更说明 |
|------|------|------|----------|
| 1.0 | 2026/06/03 | Claude | 初始版本 |

---

> **文档状态：** 草稿
> **审核状态：** 待审核
> **下一步：** 提交技术负责人审核