# 实验追踪工具 — 验证测试设计

> **来源任务：** docs\superpowers\tasks\01_环境搭建\05_实验追踪工具.md
> **生成日期：** 2026/06/02

---

## 1. 验证目标概述

### 验收标准映射

| 验收标准 | 验证目标 | 验证方式 |
|---------|---------|---------|
| 实验工具安装完成 | 确认选定工具正确安装并可导入/启动 | 自动化检查 + 手动验证 |
| 首次实验数据正常记录 | 确认指标上报、checkpoint保存、元数据记录功能正常 | 集成测试 |

### 验证优先级

1. **P0 - 阻塞性验证**：工具安装成功、基础连接正常
2. **P1 - 核心功能验证**：指标上报、checkpoint保存
3. **P2 - 辅助功能验证**：命名规范、UI可访问性

---

## 2. 测试用例设计

### 2.1 工具选择与安装

#### TC-001: 工具安装验证
```yaml
用例编号: TC-001
用例名称: 实验追踪工具安装验证
优先级: P0
前置条件: Python虚拟环境已激活
测试步骤:
  1. 执行安装命令（以WandB为例）: pip install wandb
  2. 验证导入: python -c "import wandb; print(wandb.__version__)"
  3. 检查版本兼容性: pip show wandb
预期结果:
  - 安装命令执行成功，无报错
  - import语句执行成功，输出版本号
  - 版本号 >= 0.15.0（推荐稳定版本）
```

#### TC-002: 用户认证验证
```yaml
用例编号: TC-002
用例名称: WandB/MLflow用户认证验证
优先级: P0
前置条件: TC-001通过
测试步骤:
  WandB:
    1. 执行: wandb login
    2. 输入API Key（或设置环境变量WANDB_API_KEY）
    3. 执行: wandb whoami
  MLflow:
    1. 检查MLflow追踪服务器连接: mlflow.set_tracking_uri("<server_url>")
    2. 创建测试实验验证连接
预期结果:
  - WandB: 显示已登录用户信息
  - MLflow: 成功连接追踪服务器，无连接错误
```

#### TC-003: TensorBoard本地启动验证
```yaml
用例编号: TC-003
用例名称: TensorBoard本地启动验证（若选择TensorBoard）
优先级: P0
前置条件: TC-001通过，tensorboard已安装
测试步骤:
  1. 创建临时日志目录: mkdir -p ./test_logs
  2. 启动TensorBoard: tensorboard --logdir ./test_logs --port 6006
  3. 访问 http://localhost:6006
  4. 检查进程: tasklist | findstr tensorboard
预期结果:
  - TensorBoard进程成功启动
  - 浏览器可访问TensorBoard UI
  - 无启动错误日志
```

### 2.2 实验元数据配置

#### TC-004: 指标记录配置验证
```yaml
用例编号: TC-004
用例名称: 核心指标记录配置验证
优先级: P0
前置条件: TC-002通过
测试步骤:
  1. 创建测试脚本 test_metrics.py
  2. 模拟记录以下指标:
     - loss: 随机递减值
     - accuracy: 随机递增值
     - learning_rate: 固定值或衰减
     - gpu_utilization: 随机值（模拟）
  3. 检查实验面板是否显示所有指标
预期结果:
  - 所有4项指标成功记录
  - 指标在UI中正确显示
  - 指标值与代码中记录的值一致
```

#### TC-005: Checkpoint保存路径验证
```yaml
用例编号: TC-005
用例名称: Checkpoint保存路径配置验证
优先级: P1
前置条件: TC-002通过
测试步骤:
  1. 配置checkpoint保存路径: ./checkpoints/
  2. 运行简单训练脚本，保存模型checkpoint
  3. 验证:
     - checkpoint文件存在于指定路径
     - 文件命名符合规范
     - 文件可正常加载
预期结果:
  - checkpoint目录存在且包含预期文件
  - 文件格式正确（.pt/.pth/.ckpt）
  - torch.load() 或等效方法可成功加载
```

#### TC-006: 实验命名规范验证
```yaml
用例编号: TC-006
用例名称: 实验命名规范验证
优先级: P2
前置条件: TC-002通过
测试步骤:
  1. 定义命名规范格式，如:
     {project}_{model}_{dataset}_{YYYYMMDD}_{timestamp}
  2. 创建符合规范的实验名称
  3. 验证命名在追踪系统中正确显示
  4. 测试特殊字符处理
预期结果:
  - 实验名称按规范生成
  - 特殊字符被正确转义或替换
  - 名称在UI中可读且唯一
```

### 2.3 集成测试

#### TC-007: 端到端训练追踪验证
```yaml
用例编号: TC-007
用例名称: 简单训练追踪端到端验证
优先级: P0
前置条件: TC-001~TC-004通过
测试步骤:
  1. 编写最小训练脚本:
     - 使用虚拟数据集（如torch.randn生成）
     - 简单模型（如单层Linear）
     - 训练3-5个epoch
     - 每个step记录loss和accuracy
  2. 执行训练脚本
  3. 验证:
     - 实验在追踪平台创建成功
     - loss曲线呈下降趋势
     - checkpoint按配置保存
     - GPU利用率记录（如适用）
预期结果:
  - 实验记录完整
  - 曲线数据点数 = epoch数 × batch数
  - 无上报错误或警告
  - 可在UI中查看完整训练历史
```

#### TC-008: 多实验对比验证
```yaml
用例编号: TC-008
用例名称: 多实验对比功能验证
优先级: P1
前置条件: TC-007通过，至少有2个已记录的实验
测试步骤:
  1. 使用不同超参数运行第二次实验
  2. 在追踪平台中选择两个实验进行对比
  3. 验证对比功能:
     - 曲线叠加显示
     - 超参数差异高亮
     - 指标对比表格
预期结果:
  - 多实验对比UI正常工作
  - 指标曲线可区分
  - 超参数差异正确显示
```

---

## 3. 验证方法与步骤

### 3.1 自动化验证脚本

```python
# verify_tracking_tool.py
"""
实验追踪工具验证脚本
运行方式: python verify_tracking_tool.py --tool wandb|mlflow|tensorboard
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

class TrackingToolVerifier:
    def __init__(self, tool: str):
        self.tool = tool
        self.results = []

    def check(self, name: str, condition: bool, detail: str = ""):
        """记录检查结果"""
        status = "✓ PASS" if condition else "✗ FAIL"
        self.results.append((name, status, detail))
        print(f"{status}: {name}")
        if detail:
            print(f"  └─ {detail}")
        return condition

    def verify_installation(self) -> bool:
        """TC-001: 验证安装"""
        try:
            if self.tool == "wandb":
                import wandb
                version = wandb.__version__
            elif self.tool == "mlflow":
                import mlflow
                version = mlflow.__version__
            elif self.tool == "tensorboard":
                import tensorboard
                version = tensorboard.__version__
            else:
                return self.check("工具安装", False, f"未知工具: {self.tool}")

            return self.check("工具安装", True, f"版本: {version}")
        except ImportError as e:
            return self.check("工具安装", False, str(e))

    def verify_authentication(self) -> bool:
        """TC-002: 验证认证"""
        if self.tool == "wandb":
            try:
                import wandb
                api = wandb.Api()
                user = api.default_entity
                return self.check("用户认证", True, f"已登录用户: {user}")
            except Exception as e:
                return self.check("用户认证", False, str(e))
        elif self.tool == "mlflow":
            # MLflow本地模式无需认证
            return self.check("用户认证", True, "MLflow本地模式")
        return self.check("用户认证", True)

    def verify_metric_logging(self) -> bool:
        """TC-004/TC-007: 验证指标记录"""
        test_script = f"""
import {self.tool}
import random

# 初始化实验
{'wandb.init(project="verify_test", name="tc_007")' if self.tool == 'wandb' else 'mlflow.start_run(run_name="tc_007")'}

# 记录指标
for epoch in range(3):
    loss = 1.0 / (epoch + 1)
    accuracy = 0.5 + epoch * 0.15
    lr = 0.001 * (0.9 ** epoch)
    gpu_util = random.uniform(60, 95)

    {'wandb.log({' if self.tool == 'wandb' else 'mlflow.log_metrics({'}
        "loss": loss,
        "accuracy": accuracy,
        "learning_rate": lr,
        "gpu_utilization": gpu_util
    }{', step=epoch)' if self.tool == 'wandb' else ')'}

{'wandb.finish()' if self.tool == 'wandb' else 'mlflow.end_run()'}
print("METRICS_LOGGED_SUCCESSFULLY")
"""
        try:
            result = subprocess.run(
                [sys.executable, "-c", test_script],
                capture_output=True,
                text=True,
                timeout=60
            )
            success = "METRICS_LOGGED_SUCCESSFULLY" in result.stdout
            return self.check("指标记录", success, result.stderr or "无错误")
        except Exception as e:
            return self.check("指标记录", False, str(e))

    def verify_checkpoint_save(self) -> bool:
        """TC-005: 验证checkpoint保存"""
        checkpoint_dir = Path("./test_checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)

        test_script = """
import torch
import torch.nn as nn
from pathlib import Path

# 简单模型
model = nn.Linear(10, 1)
checkpoint_path = Path("./test_checkpoints/test_model.pt")

# 保存
torch.save({
    'model_state_dict': model.state_dict(),
    'epoch': 1,
    'loss': 0.5
}, checkpoint_path)

# 加载验证
checkpoint = torch.load(checkpoint_path, weights_only=True)
assert 'model_state_dict' in checkpoint
print("CHECKPOINT_VERIFIED")
"""
        try:
            result = subprocess.run(
                [sys.executable, "-c", test_script],
                capture_output=True,
                text=True,
                timeout=30
            )
            success = "CHECKPOINT_VERIFIED" in result.stdout
            return self.check("Checkpoint保存", success,
                            "文件已保存并可加载" if success else result.stderr)
        except Exception as e:
            return self.check("Checkpoint保存", False, str(e))

    def run_all(self):
        """运行所有验证"""
        print(f"\n{'='*50}")
        print(f"实验追踪工具验证 - {self.tool.upper()}")
        print(f"{'='*50}\n")

        all_passed = True
        all_passed &= self.verify_installation()
        all_passed &= self.verify_authentication()
        all_passed &= self.verify_metric_logging()
        all_passed &= self.verify_checkpoint_save()

        print(f"\n{'='*50}")
        print(f"总结果: {'✓ 全部通过' if all_passed else '✗ 存在失败项'}")
        print(f"{'='*50}")

        return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", choices=["wandb", "mlflow", "tensorboard"],
                       required=True, help="要验证的追踪工具")
    args = parser.parse_args()

    verifier = TrackingToolVerifier(args.tool)
    success = verifier.run_all()
    sys.exit(0 if success else 1)
```

### 3.2 手动验证清单

```
手动验证清单（需人工确认）
====================================

□ UI可访问性验证
  ├─ 打开追踪平台Web界面
  ├─ 验证登录状态
  └─ 确认项目/实验列表显示

□ 实验数据可视化验证
  ├─ 打开实验详情页
  ├─ 确认loss曲线显示
  ├─ 确认accuracy曲线显示
  └─ 确认learning_rate图表显示

□ 命名规范验证
  ├─ 检查实验名称格式
  ├─ 确认命名在列表中可识别
  └─ 验证特殊字符处理

□ Checkpoint文件验证
  ├─ 检查checkpoint目录结构
  ├─ 确认文件命名规范
  └─ 手动加载checkpoint测试
```

---

## 4. 通过标准

### 4.1 量化标准

| 检查项 | 通过条件 | 失败条件 |
|-------|---------|---------|
| 工具安装 | import成功，版本符合要求 | import失败或版本过低 |
| 用户认证 | 登录状态有效，API可访问 | 认证失败或无权限 |
| 指标上报 | 所有指标在UI可见 | 任一指标缺失或错误 |
| Checkpoint | 文件存在且可加载 | 文件不存在或损坏 |
| 端到端测试 | 完整训练流程无错误 | 中途中断或数据缺失 |

### 4.2 验收判定

```yaml
验收通过条件:
  - P0测试用例全部通过（TC-001, TC-002, TC-004, TC-007）
  - P1测试用例至少1项通过
  - 无阻塞性错误

需要修复后重验:
  - 任一P0测试用例失败
  - 核心功能（指标上报）不工作

可接受但有改进项:
  - P2测试用例失败
  - UI显示异常但不影响数据记录
```

---

## 5. 自动化验证建议

### 5.1 可自动化项目

| 测试项 | 自动化方式 | 频率建议 |
|-------|-----------|---------|
| 工具安装检查 | Python import + 版本比对 | 每次环境重建 |
| 认证状态检查 | API调用检查 | 每次实验前 |
| 指标上报验证 | 脚本模拟训练 | CI/CD流水线 |
| Checkpoint保存 | 文件存在性+加载测试 | 每次模型保存 |
| 日志完整性 | 数据点计数校验 | 实验结束后 |

### 5.2 需手动验证项目

| 测试项 | 原因 | 建议 |
|-------|------|------|
| UI可视化正确性 | 需要人工判断图表准确性 | 首次部署时手动验证 |
| 命名规范可读性 | 主观评估 | 首次实验时确认 |
| 多实验对比功能 | UI交互测试 | 定期手动抽检 |
| GPU利用率准确性 | 需对比nvidia-smi | 首次配置时验证 |

### 5.3 CI/CD集成建议

```yaml
# .github/workflows/verify-tracking.yml
name: Verify Experiment Tracking

on:
  workflow_dispatch:
  schedule:
    - cron: '0 0 * * 0'  # 每周验证

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install wandb torch
      - name: Verify tracking setup
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
        run: |
          python verify_tracking_tool.py --tool wandb
```

---

## 6. 风险点与注意事项

### 6.1 环境风险

| 风险 | 影响 | 缓解措施 |
|-----|------|---------|
| 网络不稳定 | WandB云服务连接失败 | 配置离线模式 `wandb.init(mode="offline")` |
| API Key泄露 | 账户安全风险 | 使用环境变量，不硬编码 |
| 存储空间不足 | checkpoint保存失败 | 定期清理旧实验，设置retention策略 |
| 版本不兼容 | 功能异常或报错 | 锁定依赖版本，使用requirements.txt |

### 6.2 工具特定风险

#### WandB
```yaml
风险点:
  - 云服务依赖: 需要稳定网络连接
  - 免费版限制: 团队成员数、存储空间限制
  - 数据隐私: 实验数据存储在云端

缓解措施:
  - 配置离线模式备份
  - 定期导出关键数据
  - 敏感项目考虑私有部署
```

#### MLflow
```yaml
风险点:
  - 需自行部署服务器: 增加运维负担
  - UI功能有限: 对比分析功能较弱

缓解措施:
  - 使用Docker部署标准化
  - 配置数据库后端(PostgreSQL)
```

#### TensorBoard
```yaml
风险点:
  - 无云端同步: 数据仅在本地
  - 协作困难: 多人难以共享实验结果

缓解措施:
  - 使用TensorBoard.dev共享
  - 配置共享存储(NFS/S3)
```

### 6.3 数据风险

| 风险 | 影响 | 缓解措施 |
|-----|------|---------|
| 实验数据丢失 | 无法复现结果 | 定期备份，多副本存储 |
| 指标记录遗漏 | 实验分析不完整 | 使用装饰器自动记录 |
| 命名冲突 | 实验覆盖 | 使用时间戳+UUID保证唯一 |
| Checkpoint损坏 | 无法恢复模型 | 保存多个版本，校验文件完整性 |

### 6.4 验证注意事项

1. **首次验证**应在网络稳定的环境下进行
2. 验证前确认有足够的存储空间（至少10GB用于checkpoint）
3. GPU利用率记录需要实际GPU运行才能验证
4. 多实验对比验证需要先创建至少2个实验
5. 验证脚本应在虚拟环境中执行，避免环境污染

---

## 附录：验证执行记录模板

```markdown
## 验证执行记录

**执行日期：** YYYY-MM-DD
**执行人员：**
**环境信息：**
- Python版本：
- 操作系统：
- GPU型号（如适用）：

### 执行结果

| 用例编号 | 用例名称 | 执行结果 | 备注 |
|---------|---------|---------|------|
| TC-001 | 工具安装验证 | □通过 □失败 | |
| TC-002 | 用户认证验证 | □通过 □失败 | |
| TC-003 | TensorBoard启动 | □通过 □失败 □跳过 | |
| TC-004 | 指标记录配置 | □通过 □失败 | |
| TC-005 | Checkpoint保存 | □通过 □失败 | |
| TC-006 | 命名规范验证 | □通过 □失败 | |
| TC-007 | 端到端验证 | □通过 □失败 | |
| TC-008 | 多实验对比 | □通过 □失败 | |

### 问题记录

| 问题描述 | 严重程度 | 处理方式 |
|---------|---------|---------|
| | | |

### 最终结论

□ 验收通过
□ 需修复后重验
□ 验收失败，需重新规划

**签字确认：** ________________
```