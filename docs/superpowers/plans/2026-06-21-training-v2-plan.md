# ERes2NetV2 训练流程重规划 v2

> **目标**：诊断并修复当前训练问题，使模型 EER < 0.10、同一人余弦相似度 > 0.50
> **背景**：当前 best EER=0.1553（epoch 45），训练过程 EER 持续退化，模型存在坍塌问题

---

## 一、问题诊断

### 1.1 训练日志分析

| 指标 | 第一轮 epoch 0 | 第一轮 epoch 25 | 第二轮 epoch 45 | 第二轮 epoch 71 |
|------|---------------|----------------|----------------|----------------|
| EER | 0.1347 | 0.2321 | 0.1553 | 0.2494 |
| Pos mean | 0.3573 | 0.1897 | 0.3128 | 0.1621 |
| Neg mean | 0.0210 | 0.0087 | 0.0281 | 0.0022 |
| Train acc | — | — | ~0.001 | ~0.001 |
| Train loss | — | — | 0.002 | 0.0015 |

**结论**：EER 随训练持续升高，模型在退化而非收敛。

### 1.2 根因分析

| # | 问题 | 严重度 | 证据 |
|---|------|--------|------|
| R1 | 配置不一致：fixed_length 200 vs 300 | 高 | train_config.yaml vs data_config.yaml |
| R2 | dropout 配置 0.4 但模型代码未使用 | 高 | model config 有 dropout，ERes2NetV2.__init__ 无 dropout 参数 |
| R3 | label_smoothing 训练 0.15 vs 损失 0.1 | 中 | train_config.yaml 内部不一致 |
| R4 | 无数据增强：配置了但 SpeakerDataset 未实现 | 高 | speaker_dataset.py 无 transform 逻辑 |
| R5 | AAM-Softmax margin=0.2, scale=30 可能过大 | 中 | 1211 说话人，margin 偏大导致训练困难 |
| R6 | AMP 关闭 + CPU 训练 | 高 | amp: false, device: cpu |
| R7 | 训练准确率 0.1%≈随机，loss 却很低 | 高 | 模型坍塌或标签错位 |
| R8 | 无 VoxCeleb1 标准测试集验证 | 中 | 用 dev 内部划分，未用 test/ 40 说话人 |

### 1.3 假设排序

1. **R7 模型坍塌**：AAM-Softmax + 无增强 + 学习率过高 → 模型找到捷径，loss 降但 embedding 无判别力
2. **R4 无数据增强**：FBank 特征过拟合训练集分布，验证集分布稍偏即崩溃
3. **R1/R2/R3 配置不一致**：特征维度不匹配 / dropout 缺失 / 正则化参数错位
4. **R6 CPU 训练**：batch_size=128 在 CPU 上极慢，可能训练不充分 + 无 AMP

---

## 二、修复方案

### Phase 1：诊断与根因确认（产出验证报告）

| Task | 做什么 | 产出物 |
|------|--------|--------|
| T1.1 | 训练日志审计，提取 EER/loss/acc 曲线数据 | `reports/diagnostic/training_log_analysis.json` |
| T1.2 | 数据管线完整性验证（特征 shape、标签对齐、CMVN） | `reports/diagnostic/data_pipeline_report.json` |
| T1.3 | 损失函数正确性验证（AAM-Softmax 数值检查） | `reports/diagnostic/loss_verification.md` |

### Phase 2：数据管线修复（产出修复后的代码 + 测试）

| Task | 做什么 | 产出物 |
|------|--------|--------|
| T2.1 | 修复配置一致性：fixed_length、dropout、label_smoothing | `configs/train_config_v2.yaml` |
| T2.2 | 实现 on-the-fly 数据增强（speed/noise/specAugment） | `src/datasets/augmentation.py` + 测试输出 |
| T2.3 | 特征提取管线验证（重新提取一批样本检查） | `reports/diagnostic/feature_sample_check.json` |

### Phase 3：模型与训练修复（产出修复后的模型 + 配置）

| Task | 做什么 | 产出物 |
|------|--------|--------|
| T3.1 | 修复模型 dropout + 分类头 | `src/models/eres2netv2.py`（修改后）+ 模型测试输出 |
| T3.2 | 调整 AAM-Softmax 参数（margin/scale 网格搜索） | `reports/diagnostic/loss_param_search.json` |
| T3.3 | 编写 v2 训练配置（含 AMP、GPU、增强） | `configs/train_v2.yaml` |

### Phase 4：训练执行与监控（产出训练日志 + checkpoint）

| Task | 做什么 | 产出物 |
|------|--------|--------|
| T4.1 | 冒烟测试：3 epoch 快速验证不坍塌 | `reports/smoke_test_3epoch.json` |
| T4.2 | 完整训练：50 epoch + AMP + GPU + 增强 | `checkpoints_v2/` + `logs_v2/training.log` |
| T4.3 | 训练过程监控：每 5 epoch 记录 EER/loss/similarity | `logs_v2/metrics_per_epoch.json` |

### Phase 5：评估与对比（产出评估报告）

| Task | 做什么 | 产出物 |
|------|--------|--------|
| T5.1 | VoxCeleb1 标准测试集评估（40 说话人 trials） | `reports_v2/evaluation_report.md` |
| T5.2 | 对比报告：v1 vs v2 各项指标 | `reports_v2/comparison_v1_v2.md` |

---

## 三、验收标准

| 指标 | 当前(v1) | 目标(v2) | 优秀 |
|------|---------|---------|------|
| EER | 0.1553 | < 0.10 | < 0.05 |
| 同一人余弦相似度 | 0.16 | > 0.50 | > 0.70 |
| 不同人余弦相似度 | 0.002 | < 0.10 | < 0.05 |
| 训练准确率 | 0.1% | > 30% | > 60% |
| EER 不随训练退化 | 否 | 是 | 是 |

---

## 四、技术决策

| 决策点 | v1（当前） | v2（修复） | 理由 |
|--------|-----------|-----------|------|
| fixed_length | 200 | 200（统一） | 2秒语音足够说话人识别 |
| dropout | 配置有，代码无 | 代码实现 0.3 | 正则化防坍塌 |
| AAM margin | 0.2 | 0.2→0.3 渐进 | 先易后难 |
| AAM scale | 30 | 32 | 标准值 |
| 数据增强 | 无 | speed+noise+specAugment | 提升泛化 |
| AMP | off | on | 4x 加速 |
| 学习率 | 0.1 | 0.001 (AdamW) | SGD+CPU 可能不充分 |
| 优化器 | SGD | AdamW | 收敛更稳定 |
| 训练设备 | CPU | GPU | 必须用 GPU |

---

*创建时间：2026-06-21*
*基于：训练日志分析 + 代码审计 + 测试结果*
