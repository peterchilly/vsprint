# 标准测试集评估 — 验证检查清单

> **对应任务**：tasks/v2_training/05_评估对比/01_标准测试集评估.md

## 验证项目

### 1. 产出物存在性检查
| 检查项 | 预期 | 实际 | 通过 |
|--------|------|------|------|
| reports_v2/evaluation_report.md | 存在且非空 | | ☐ |
| reports_v2/evaluation_data.json | 存在且非空 | | ☐ |
| reports_v2/roc_curve.json | 存在 | | ☐ |
| reports_v2/det_curve.json | 存在 | | ☐ |

### 2. 内容完整性检查
| 检查项 | 预期内容 | 验证方法 | 通过 |
|--------|----------|----------|------|
| EER 指标 | 数值 | 检查 JSON metrics.eer | ☐ |
| minDCF 指标 | 数值 | 检查 JSON | ☐ |
| pos_mean + std | 数值 | 检查 JSON | ☐ |
| neg_mean + std | 数值 | 检查 JSON | ☐ |
| 混淆分析 | Top-10 说话人对 | 检查 JSON/报告 | ☐ |
| 效率指标 | 参数量、FLOPs、延迟 | 检查 JSON/报告 | ☐ |
| ROC 曲线 | fpr/tpr 数据 | 检查 roc_curve.json | ☐ |

### 3. 技术正确性检查
| 检查项 | 标准 | 验证方法 | 通过 |
|--------|------|----------|------|
| 测试集为 40 说话人 | 评估使用 test/ 目录 | 检查报告 | ☐ |
| EER < 0.10 | 目标值 | 检查 metrics.eer | ☐ |
| pos_mean > 0.50 | 目标值 | 检查 metrics.pos_mean | ☐ |
| neg_mean < 0.05 | 目标值 | 检查 metrics.neg_mean | ☐ |

## 验证命令
```powershell
cd C:\Users\Administrator\vsprint
python -c "
import json
d = json.load(open('reports_v2/evaluation_data.json'))
m = d['metrics']
print(f'EER: {m[\"eer\"]:.4f}')
print(f'pos_mean: {m[\"pos_mean\"]:.4f}')
print(f'neg_mean: {m[\"neg_mean\"]:.4f}')
assert m['eer'] < 0.10, f'EER {m[\"eer\"]} not meeting target'
assert m['pos_mean'] > 0.50, f'pos_mean {m[\"pos_mean\"]} too low'
assert m['neg_mean'] < 0.05, f'neg_mean {m[\"neg_mean\"]} too high'
print('PASS')
"
```

## 验证结论
- [ ] 通过
- [ ] 不通过（问题描述：___）
