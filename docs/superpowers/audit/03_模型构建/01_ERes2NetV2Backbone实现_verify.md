# T3.1 ERes2NetV2 Backbone 实现 — 验收报告

## 验收结果：✅ 通过

### 验收项
| # | 检查项 | 标准 | 实际结果 | 状态 |
|---|--------|------|----------|------|
| 1 | Backbone 可独立运作 | 前向传播无报错 | 4 个变体全部通过 | ✅ |
| 2 | 输入输出维度 | (batch,1,80,300)→(batch,192) | embedding 维度正确 | ✅ |
| 3 | 参数量 | 符合预期 | 34:4.75M, 50:16.65M, 101:28.40M, 152:38.08M | ✅ |

### 详细测试结果

**前向传播测试**
- ERes2NetV2-34: ✅ input(2,1,80,300) → embedding(2,192), 4.75M 参数
- ERes2NetV2-50: ✅ input(2,1,80,300) → embedding(2,192), 16.65M 参数
- ERes2NetV2-101: ✅ input(2,1,80,300) → embedding(2,192), 28.40M 参数
- ERes2NetV2-152: ✅ input(2,1,80,300) → embedding(2,192), 38.08M 参数

**带分类头测试**
- ✅ ERes2NetV2-34 + classifier: input(2,1,80,300) → embedding(2,192) + logits(2,100)

**梯度反传测试**
- ✅ loss=7.5297, 无 NaN/Inf 梯度

**AAM-Softmax 损失测试**
- ✅ AAM-Softmax: batch=32, loss=7.7520
- ✅ CrossEntropy 对比: loss=5.2634

**GPU 推理测试**
- ✅ GPU 推理正常, loss=6.9708
- ✅ GPU 显存占用: 51.9 MB

### 交付物清单
- `src/models/eres2netv2.py` — ERes2NetV2 Backbone (BasicBlock + Bottleneck + AttentiveStatsPool)
- `src/models/losses.py` — AAMSoftmaxLoss + ArcFaceLoss
- `src/models/__init__.py` — 模块导出
- `scripts/test_model.py` — 模型验证脚本
- `scripts/extract_features.py` — FBank 特征提取脚本
