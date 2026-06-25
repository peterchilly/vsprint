# 训练过程监控 — 验证检查清单

> **对应任务**：tasks/v2_training/04_训练执行/03_训练过程监控.md

## 验证项目

### 1. 产出物存在性检查
| 检查项 | 预期 | 实际 | 通过 |
|--------|------|------|------|
| logs_v2/metrics_per_epoch.json | 存在且非空 | | ☐ |

### 2. 内容完整性检查
| 检查项 | 预期内容 | 验证方法 | 通过 |
|--------|----------|----------|------|
| epochs 数组 | 包含所有训练 epoch | 检查 JSON | ☐ |
| 每个 epoch 含 5+ 指标 | eer, loss, acc, pos_mean, neg_mean, lr | 检查 JSON | ☐ |
| trend_analysis | 含 eer_trend, pos_mean_trend, healthy | 检查 JSON | ☐ |
| best_eer 和 best_epoch | 数值 | 检查 JSON | ☐ |

### 3. 技术正确性检查
| 检查项 | 标准 | 验证方法 | 通过 |
|--------|------|----------|------|
| EER 趋势下降 | eer_trend="decreasing" | 检查 JSON | ☐ |
| pos_mean 趋势上升 | pos_mean_trend="increasing" | 检查 JSON | ☐ |
| 训练健康 | healthy=true | 检查 JSON | ☐ |
| best_eer < 0.10 | 目标值 | 检查 JSON | ☐ |
| 数据与 training.log 一致 | 抽查 3 个 epoch | 对比 | ☐ |

## 验证命令
```powershell
cd C:\Users\Administrator\vsprint
python -c "
import json
d = json.load(open('logs_v2/metrics_per_epoch.json'))
print(f'epochs: {len(d[\"epochs\"])}')
print(f'best_eer: {d[\"best_eer\"]}')
print(f'healthy: {d[\"trend_analysis\"][\"healthy\"]}')
assert d['trend_analysis']['healthy'] == True
assert d['best_eer'] < 0.10
print('PASS')
"
```

## 验证结论
- [ ] 通过
- [ ] 不通过（问题描述：___）
