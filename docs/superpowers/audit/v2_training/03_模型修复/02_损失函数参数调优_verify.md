# 损失函数参数调优 — 验证检查清单

> **对应任务**：tasks/v2_training/03_模型修复/02_损失函数参数调优.md

## 验证项目

### 1. 产出物存在性检查
| 检查项 | 预期 | 实际 | 通过 |
|--------|------|------|------|
| scripts/loss_param_search.py | 存在 | | ☐ |
| reports/diagnostic/loss_param_search.json | 存在且非空 | | ☐ |

### 2. 内容完整性检查
| 检查项 | 预期内容 | 验证方法 | 通过 |
|--------|----------|----------|------|
| 实验数量 | 9 组（3 margin × 3 scale） | 检查 JSON len | ☐ |
| 每组含 epoch 数据 | 3 epoch 的 EER/loss/acc | 检查 JSON | ☐ |
| best_params | 包含 margin 和 scale | 检查 JSON | ☐ |
| recommendation | 文字说明 | 检查 JSON | ☐ |

### 3. 技术正确性检查
| 检查项 | 标准 | 验证方法 | 通过 |
|--------|------|----------|------|
| 最优 3-epoch EER | < 0.15 | 检查 JSON best_params | ☐ |
| 参数组合覆盖完整 | margins=[0.1,0.2,0.3], scales=[16,24,32] | 检查 experiments | ☐ |

## 验证命令
```powershell
cd C:\Users\Administrator\vsprint
python -c "
import json
d = json.load(open('reports/diagnostic/loss_param_search.json'))
print(f'experiments: {len(d[\"experiments\"])}')
print(f'best: {d[\"best_params\"]}')
assert len(d['experiments']) == 9
assert 'margin' in d['best_params']
assert 'scale' in d['best_params']
print('PASS')
"
```

## 验证结论
- [ ] 通过
- [ ] 不通过（问题描述：___）
