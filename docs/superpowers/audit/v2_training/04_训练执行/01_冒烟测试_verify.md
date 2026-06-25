# 冒烟测试 — 验证检查清单

> **对应任务**：tasks/v2_training/04_训练执行/01_冒烟测试.md

## 验证项目

### 1. 产出物存在性检查
| 检查项 | 预期 | 实际 | 通过 |
|--------|------|------|------|
| reports/smoke_test_3epoch.json | 存在且非空 | | ☐ |

### 2. 内容完整性检查
| 检查项 | 预期内容 | 验证方法 | 通过 |
|--------|----------|----------|------|
| epochs 数据 | 3 个 epoch 的完整指标 | 检查 JSON | ☐ |
| collapse_check | 4 项检查结果 | 检查 JSON | ☐ |
| verdict | "pass" 或 "fail" | 检查 JSON | ☐ |

### 3. 技术正确性检查
| 检查项 | 标准 | 验证方法 | 通过 |
|--------|------|----------|------|
| EER 不退化 | epoch 3 EER < 0.20 | 检查 JSON | ☐ |
| 训练准确率 > 随机 | epoch 3 acc > 0.05 | 检查 JSON | ☐ |
| loss 正常下降 | 不坍塌到 < 0.01 | 检查 JSON | ☐ |
| pos_mean 合理 | > 0.10 | 检查 JSON | ☐ |
| verdict=pass | 模型未坍塌 | 检查 JSON | ☐ |

## 验证命令
```powershell
cd C:\Users\Administrator\vsprint
python -c "
import json
d = json.load(open('reports/smoke_test_3epoch.json'))
assert len(d['epochs']) == 3
assert d['verdict'] == 'pass'
assert d['collapse_check']['eer_ok'] == True
assert d['collapse_check']['acc_ok'] == True
print(f'Smoke test: {d[\"verdict\"]}')
print('PASS')
"
```

## 验证结论
- [ ] 通过
- [ ] 不通过（问题描述：___）
