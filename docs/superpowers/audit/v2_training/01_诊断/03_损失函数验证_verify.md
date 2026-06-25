# 损失函数正确性验证 — 验证检查清单

> **对应任务**：tasks/v2_training/01_诊断/03_损失函数验证.md

## 验证项目

### 1. 产出物存在性检查
| 检查项 | 预期 | 实际 | 通过 |
|--------|------|------|------|
| reports/diagnostic/loss_verification.md | 存在且非空 | | ☐ |
| tests/test_losses.py | 存在 | | ☐ |

### 2. 内容完整性检查
| 检查项 | 预期内容 | 验证方法 | 通过 |
|--------|----------|----------|------|
| 场景 A/B/C 测试 | loss(A) < loss(B) < loss(C) | 检查报告内容 | ☐ |
| 梯度检查结果 | 误差 < 1e-4 | 检查报告 | ☐ |
| 权重矩阵检查 | 无异常（非全零/爆炸） | 检查报告 | ☐ |
| 边界条件测试 | 无 NaN/Inf | 检查报告 | ☐ |
| 明确结论 | 损失函数正确/有bug | 检查报告末尾 | ☐ |

### 3. 技术正确性检查
| 检查项 | 标准 | 验证方法 | 通过 |
|--------|------|----------|------|
| 测试脚本可运行 | pytest 退出码=0 | `python -m pytest tests/test_losses.py -v` | ☐ |
| 场景 A loss 接近 0 | loss < 0.5 | 检查报告数值 | ☐ |
| 场景 C loss 最高 | loss > 3.0 | 检查报告数值 | ☐ |

## 验证命令
```powershell
cd C:\Users\Administrator\vsprint
python -m pytest tests/test_losses.py -v
# 预期：所有测试 PASSED
```

## 验证结论
- [ ] 通过
- [ ] 不通过（问题描述：___）
