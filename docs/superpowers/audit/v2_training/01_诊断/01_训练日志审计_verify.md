# 训练日志审计 — 验证检查清单

> **对应任务**：tasks/v2_training/01_诊断/01_训练日志审计.md
> **验证日期**：待填写
> **验证人**：待填写

## 验证项目

### 1. 产出物存在性检查
| 检查项 | 预期 | 实际 | 通过 |
|--------|------|------|------|
| reports/diagnostic/training_log_analysis.json | 存在且非空 | | ☐ |

### 2. 内容完整性检查
| 检查项 | 预期内容 | 验证方法 | 通过 |
|--------|----------|----------|------|
| round_1 epochs | 包含 ~25 个 epoch 数据 | `python -c "import json; d=json.load(open('reports/diagnostic/training_log_analysis.json')); assert len(d['round_1']['epochs'])>20"` | ☐ |
| round_2 epochs | 包含 ~46 个 epoch 数据 | 同上检查 round_2 | ☐ |
| 每个 epoch 指标 | eer, loss, acc, pos_mean, neg_mean | 检查第一个 epoch 的 keys | ☐ |
| 趋势分析结论 | model_collapse 或其他明确结论 | `assert 'conclusion' in d` | ☐ |

### 3. 技术正确性检查
| 检查项 | 标准 | 验证方法 | 通过 |
|--------|------|----------|------|
| EER 数据与 training.log 一致 | 误差 < 0.001 | 抽查 3 个 epoch 与原始日志对比 | ☐ |
| 退化趋势被正确识别 | round_1 首epoch EER < 末epoch EER | 检查 JSON 中的趋势 | ☐ |

## 验证命令
```powershell
cd C:\Users\Administrator\vsprint
python -c "import json; d=json.load(open('reports/diagnostic/training_log_analysis.json')); print(d['conclusion']); print(f'R1: {len(d[\"round_1\"][\"epochs\"])} epochs, R2: {len(d[\"round_2\"][\"epochs\"])} epochs')"
```

## 验证结论
- [ ] 通过
- [ ] 不通过（问题描述：___）
