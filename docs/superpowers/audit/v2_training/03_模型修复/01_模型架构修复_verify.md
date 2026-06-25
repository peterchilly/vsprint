# 模型架构修复 — 验证检查清单

> **对应任务**：tasks/v2_training/03_模型修复/01_模型架构修复.md

## 验证项目

### 1. 产出物存在性检查
| 检查项 | 预期 | 实际 | 通过 |
|--------|------|------|------|
| src/models/eres2netv2.py（修改后） | dropout 参数被实际使用 | | ☐ |
| train.py（修改后） | 传入 dropout 参数 | | ☐ |

### 2. 内容完整性检查
| 检查项 | 预期内容 | 验证方法 | 通过 |
|--------|----------|----------|------|
| __init__ 接受 dropout 参数 | `dropout=0.3` 在签名中 | 检查代码 | ☐ |
| create_eres2netv2 传递 dropout | 工厂函数含 dropout | 检查代码 | ☐ |
| train.py 传入 dropout | 从 config 读取 | 检查代码 | ☐ |

### 3. 技术正确性检查
| 检查项 | 标准 | 验证方法 | 通过 |
|--------|------|----------|------|
| dropout.p 与配置一致 | p=0.3 | `python -c "from src.models.eres2netv2 import create_eres2netv2; m=create_eres2netv2('34',dropout=0.3); print(m.dropout.p)"` | ☐ |
| 前向传播输出正确 | emb: [2, 192] | 构造输入测试 | ☐ |
| classifier 与 embedding 分离 | classifier 可为 None | 检查代码 | ☐ |

## 验证命令
```powershell
cd C:\Users\Administrator\vsprint
python -c "
from src.models.eres2netv2 import create_eres2netv2
import torch
m = create_eres2netv2('34', dropout=0.3)
assert m.dropout.p == 0.3, f'dropout p mismatch: {m.dropout.p}'
x = torch.randn(2, 1, 80, 200)
emb, logits = m(x)
assert emb.shape == (2, 192), f'emb shape: {emb.shape}'
print(f'dropout={m.dropout.p}, emb={emb.shape}, logits={logits.shape}')
print('PASS')
"
```

## 验证结论
- [ ] 通过
- [ ] 不通过（问题描述：___）
