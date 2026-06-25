# 数据管线完整性验证 — 验证检查清单

> **对应任务**：tasks/v2_training/01_诊断/02_数据管线验证.md

## 验证项目

### 1. 产出物存在性检查
| 检查项 | 预期 | 实际 | 通过 |
|--------|------|------|------|
| reports/diagnostic/data_pipeline_report.json | 存在且非空 | | ☐ |

### 2. 内容完整性检查
| 检查项 | 预期内容 | 验证方法 | 通过 |
|--------|----------|----------|------|
| feature_check | n_samples≥100, n_mels=80, dtype=float32 | 检查 JSON key | ☐ |
| cmvn_status | "applied" 或 "not_applied" | `assert d['cmvn_status'] in ['applied','not_applied']` | ☐ |
| label_alignment | "correct" 或 "misaligned" | `assert d['label_alignment'] in ['correct','misaligned']` | ☐ |
| speaker_distribution | 包含 n_speakers=1211 | 检查 JSON key | ☐ |

### 3. 技术正确性检查
| 检查项 | 标准 | 验证方法 | 通过 |
|--------|------|----------|------|
| 无 NaN/Inf | has_nan=false, has_inf=false | 检查 JSON | ☐ |
| mel 维度一致 | n_mels=80 | 检查 JSON | ☐ |

## 验证命令
```powershell
cd C:\Users\Administrator\vsprint
python -c "import json; d=json.load(open('reports/diagnostic/data_pipeline_report.json')); print(d['cmvn_status'], d['label_alignment'], d['speaker_distribution']['n_speakers'])"
```

## 验证结论
- [ ] 通过
- [ ] 不通过（问题描述：___）
