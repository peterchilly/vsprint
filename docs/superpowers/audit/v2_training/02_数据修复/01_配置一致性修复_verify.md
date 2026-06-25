# 配置一致性修复 — 验证检查清单

> **对应任务**：tasks/v2_training/02_数据修复/01_配置一致性修复.md

## 验证项目

### 1. 产出物存在性检查
| 检查项 | 预期 | 实际 | 通过 |
|--------|------|------|------|
| configs/train_config_v2.yaml | 存在 | | ☐ |
| configs/data_config_v2.yaml | 存在 | | ☐ |

### 2. 内容完整性检查
| 检查项 | 预期值 | 验证方法 | 通过 |
|--------|--------|----------|------|
| training.amp | true | YAML 读取 | ☐ |
| training.optimizer | AdamW | YAML 读取 | ☐ |
| training.lr | 0.001 | YAML 读取 | ☐ |
| training.weight_decay | 0.05 | YAML 读取 | ☐ |
| training.label_smoothing | 0.1 | YAML 读取 | ☐ |
| data.fixed_length | 200 | YAML 读取 | ☐ |
| data.num_workers | 4 | YAML 读取 | ☐ |
| model.dropout | 0.3 | YAML 读取 | ☐ |

### 3. 一致性检查
| 检查项 | 标准 | 验证方法 | 通过 |
|--------|------|----------|------|
| fixed_length 两处统一 | train=200, data=200 | 对比两个 YAML | ☐ |
| label_smoothing 统一 | training=loss=0.1 | 对比 | ☐ |

## 验证命令
```powershell
cd C:\Users\Administrator\vsprint
python -c "
import yaml
c = yaml.safe_load(open('configs/train_config_v2.yaml'))
assert c['training']['amp'] == True
assert c['training']['optimizer'] == 'AdamW'
assert c['training']['lr'] == 0.001
assert c['training']['label_smoothing'] == 0.1
assert c['data']['fixed_length'] == 200
assert c['data']['num_workers'] == 4
assert c['model']['dropout'] == 0.3
print('Config v2 OK')
"
```

## 验证结论
- [ ] 通过
- [ ] 不通过（问题描述：___）
