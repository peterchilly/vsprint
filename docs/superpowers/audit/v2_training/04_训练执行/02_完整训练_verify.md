# 完整训练 — 验证检查清单

> **对应任务**：tasks/v2_training/04_训练执行/02_完整训练.md

## 验证项目

### 1. 产出物存在性检查
| 检查项 | 预期 | 实际 | 通过 |
|--------|------|------|------|
| checkpoints_v2/best_model.pth | 存在 | | ☐ |
| checkpoints_v2/last_model.pth | 存在 | | ☐ |
| logs_v2/training.log | 存在且非空 | | ☐ |

### 2. 内容完整性检查
| 检查项 | 预期内容 | 验证方法 | 通过 |
|--------|----------|----------|------|
| training.log 含所有 epoch | epoch 0 到 best_epoch | 读取日志 | ☐ |
| best_model 含 epoch 信息 | epoch 和 best_val_score | torch.load 检查 | ☐ |
| 定期 checkpoint 存在 | 至少 2 个 epoch_N.pth | 列目录 | ☐ |

### 3. 技术正确性检查
| 检查项 | 标准 | 验证方法 | 通过 |
|--------|------|----------|------|
| best EER < 0.10 | best_val_score < 0.10 | `torch.load` 检查 | ☐ |
| EER 不退化 | 末 10 epoch EER ≤ 前 10 epoch | 读取 training.log | ☐ |
| pos_mean > 0.30 | best epoch 的 pos_mean | 读取日志 | ☐ |
| train_acc > 30% | 末 epoch train_acc | 读取日志 | ☐ |

## 验证命令
```powershell
cd C:\Users\Administrator\vsprint
python -c "
import torch
c = torch.load('checkpoints_v2/best_model.pth', map_location='cpu', weights_only=False)
print(f'best epoch: {c[\"epoch\"]}')
print(f'best EER: {c[\"best_val_score\"]:.4f}')
assert c['best_val_score'] < 0.10, f'EER {c[\"best_val_score\"]} not meeting target'
print('PASS')
"
```

## 验证结论
- [ ] 通过
- [ ] 不通过（问题描述：___）
