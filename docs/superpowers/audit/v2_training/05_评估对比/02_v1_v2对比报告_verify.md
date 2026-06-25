# v1 vs v2 对比报告 — 验证检查清单

> **对应任务**：tasks/v2_training/05_评估对比/02_v1_v2对比报告.md

## 验证项目

### 1. 产出物存在性检查
| 检查项 | 预期 | 实际 | 通过 |
|--------|------|------|------|
| reports_v2/comparison_v1_v2.md | 存在且非空 | | ☐ |
| reports_v2/v1_eval/ | v1 评估结果目录存在 | | ☐ |

### 2. 内容完整性检查
| 检查项 | 预期内容 | 验证方法 | 通过 |
|--------|----------|----------|------|
| EER 对比 | v1 和 v2 的 EER 数值 | 检查报告 | ☐ |
| minDCF 对比 | v1 和 v2 的 minDCF | 检查报告 | ☐ |
| pos_mean 对比 | v1 和 v2 的 pos_mean | 检查报告 | ☐ |
| neg_mean 对比 | v1 和 v2 的 neg_mean | 检查报告 | ☐ |
| 训练曲线对比 | EER 退化 vs 收敛 | 检查报告 | ☐ |
| 实际音频对比 | 同一人/不同人相似度 | 检查报告 | ☐ |
| 结论 | v2 优于 v1 + 后续建议 | 检查报告 | ☐ |

### 3. 技术正确性检查
| 检查项 | 标准 | 验证方法 | 通过 |
|--------|------|----------|------|
| v2 EER < v1 EER | 0.1553 → < 0.10 | 检查报告数值 | ☐ |
| v2 pos_mean > v1 pos_mean | 0.16 → > 0.50 | 检查报告数值 | ☐ |
| v2 train_acc > v1 train_acc | 0.1% → > 30% | 检查报告数值 | ☐ |
| 相同测试集 | v1 和 v2 都在 test/ 评估 | 检查报告 | ☐ |

## 验证命令
```powershell
cd C:\Users\Administrator\vsprint
python -c "
with open('reports_v2/comparison_v1_v2.md', 'r', encoding='utf-8') as f:
    content = f.read()
    assert 'EER' in content
    assert 'pos_mean' in content
    assert 'v1' in content and 'v2' in content
    assert '改善' in content or 'improvement' in content.lower()
print('Comparison report OK')
"
```

## 验证结论
- [ ] 通过
- [ ] 不通过（问题描述：___）
