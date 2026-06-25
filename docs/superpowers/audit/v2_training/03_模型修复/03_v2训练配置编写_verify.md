# v2 训练配置编写 — 验证检查清单

> **对应任务**：tasks/v2_training/03_模型修复/03_v2训练配置编写.md

## 验证项目

### 1. 产出物存在性检查
| 检查项 | 预期 | 实际 | 通过 |
|--------|------|------|------|
| configs/train_v2.yaml | 存在 | | ☐ |

### 2. 内容完整性检查
| 检查项 | 预期值 | 验证方法 | 通过 |
|--------|--------|----------|------|
| training.optimizer | AdamW | YAML 读取 | ☐ |
| training.lr | 0.001 | YAML 读取 | ☐ |
| training.amp | true | YAML 读取 | ☐ |
| training.epochs | 50 | YAML 读取 | ☐ |
| model.dropout | 0.3 | YAML 读取 | ☐ |
| data.augment | true | YAML 读取 | ☐ |
| data.fixed_length | 200 | YAML 读取 | ☐ |
| checkpoint.save_dir | checkpoints_v2 | YAML 读取 | ☐ |
| loss.margin | T3.2 最优值 | YAML 读取 | ☐ |
| loss.scale | T3.2 最优值 | YAML 读取 | ☐ |

### 3. 一致性检查
| 检查项 | 标准 | 验证方法 | 通过 |
|--------|------|----------|------|
| 不覆盖旧 checkpoint | save_dir ≠ checkpoints | 检查 | ☐ |
| 增强配置存在 | augment_config 非空 | 检查 | ☐ |

## 验证命令
```powershell
cd C:\Users\Administrator\vsprint
python -c "
import yaml
c = yaml.safe_load(open('configs/train_v2.yaml'))
assert c['training']['optimizer'] == 'AdamW'
assert c['training']['lr'] == 0.001
assert c['training']['amp'] == True
assert c['model']['dropout'] == 0.3
assert c['data']['augment'] == True
assert c['checkpoint']['save_dir'] == 'checkpoints_v2'
assert c['loss']['margin'] is not None
assert c['loss']['scale'] is not None
print('train_v2.yaml OK')
"
```

## 验证结论
- [ ] 通过
- [ ] 不通过（问题描述：___）
