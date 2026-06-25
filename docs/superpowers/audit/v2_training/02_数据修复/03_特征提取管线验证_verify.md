# 特征提取管线验证 — 验证检查清单

> **对应任务**：tasks/v2_training/02_数据修复/03_特征提取管线验证.md

## 验证项目

### 1. 产出物存在性检查
| 检查项 | 预期 | 实际 | 通过 |
|--------|------|------|------|
| reports/diagnostic/feature_sample_check.json | 存在且非空 | | ☐ |

### 2. 内容完整性检查
| 检查项 | 预期内容 | 验证方法 | 通过 |
|--------|----------|----------|------|
| n_samples | ≥ 50 | 检查 JSON | ☐ |
| n_speakers | ≥ 10 | 检查 JSON | ☐ |
| feature_shape.n_mels | 80 | 检查 JSON | ☐ |
| end_to_end_check | "pass" | 检查 JSON | ☐ |

### 3. 技术正确性检查
| 检查项 | 标准 | 验证方法 | 通过 |
|--------|------|----------|------|
| 无 NaN | has_nan=false | 检查 JSON | ☐ |
| 无 Inf | has_inf=false | 检查 JSON | ☐ |
| 数值范围合理 | min > -30, max < 10 | 检查 value_range | ☐ |

## 验证命令
```powershell
cd C:\Users\Administrator\vsprint
python -c "
import json
d = json.load(open('reports/diagnostic/feature_sample_check.json'))
print(f'samples: {d[\"n_samples\"]}, speakers: {d[\"n_speakers\"]}')
print(f'range: [{d[\"value_range\"][\"min\"]}, {d[\"value_range\"][\"max\"]}]')
print(f'end_to_end: {d[\"end_to_end_check\"]}')
assert d['has_nan'] == False
assert d['has_inf'] == False
assert d['end_to_end_check'] == 'pass'
print('PASS')
"
```

## 验证结论
- [ ] 通过
- [ ] 不通过（问题描述：___）
