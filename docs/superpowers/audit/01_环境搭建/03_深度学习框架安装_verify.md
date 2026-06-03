# 深度学习框架安装 — 验证测试设计

> **来源任务：** docs/superpowers/tasks/01_环境搭建/03_深度学习框架安装.md
> **生成日期：** 2026-06-02

---

## 1. 验证目标概述

本验证方案针对深度学习框架安装任务，确保以下目标达成：

| 验收标准 | 验证方式 | 量化指标 |
|---------|---------|---------|
| `torch.cuda.is_available()` 返回 True | Python 脚本执行 | 返回值为 True |
| 所有辅助库可正常导入 | Python 导入测试 | 无 ImportError 异常 |
| GPU 张量运算正常 | 计算验证 | 运算结果正确，无报错 |

---

## 2. 测试用例设计

### 2.1 PyTorch 安装验证

#### TC-PYTORCH-001: PyTorch 版本兼容性检查
| 项目 | 内容 |
|------|------|
| **测试目的** | 确认 PyTorch 版本与 CUDA 版本匹配 |
| **前置条件** | CUDA 已安装，环境变量配置完成 |
| **测试步骤** | 1. 执行 `python -c "import torch; print(torch.__version__)"`<br>2. 执行 `nvcc --version` 获取 CUDA 版本<br>3. 对比 PyTorch 编译时的 CUDA 版本 |
| **预期结果** | PyTorch 版本号显示的 CUDA 版本与系统 CUDA 版本兼容（大版本号一致） |
| **通过标准** | 版本号匹配或兼容 |

#### TC-PYTORCH-002: CUDA 可用性验证
| 项目 | 内容 |
|------|------|
| **测试目的** | 验证 PyTorch 能正确识别 GPU |
| **前置条件** | PyTorch 已安装，GPU 驱动正常 |
| **测试步骤** | 执行 Python 代码：<br>```python<br>import torch<br>print(f"CUDA available: {torch.cuda.is_available()}")<br>``` |
| **预期结果** | 输出 `CUDA available: True` |
| **通过标准** | 返回值为 True |

#### TC-PYTORCH-003: GPU 设备信息获取
| 项目 | 内容 |
|------|------|
| **测试目的** | 记录 GPU 名称和数量 |
| **前置条件** | TC-PYTORCH-002 通过 |
| **测试步骤** | 执行 Python 代码：<br>```python<br>import torch<br>print(f"GPU count: {torch.cuda.device_count()}")<br>for i in range(torch.cuda.device_count()):<br>    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")<br>``` |
| **预期结果** | 正确输出 GPU 数量和名称 |
| **通过标准** | GPU 数量 ≥ 1，名称非空 |

#### TC-PYTORCH-004: CUDA 版本一致性检查
| 项目 | 内容 |
|------|------|
| **测试目的** | 确认 PyTorch 编译的 CUDA 版本 |
| **前置条件** | PyTorch 已安装 |
| **测试步骤** | 执行：<br>```python<br>import torch<br>print(f"PyTorch CUDA version: {torch.version.cuda}")<br>print(f"cuDNN version: {torch.backends.cudnn.version()}")<br>``` |
| **预期结果** | 版本号正常输出 |
| **通过标准** | 版本号非空且格式正确 |

---

### 2.2 辅助库安装验证

#### TC-DEPS-001: torchvision 安装验证
| 项目 | 内容 |
|------|------|
| **测试目的** | 验证 torchvision 正确安装并与 PyTorch 兼容 |
| **前置条件** | PyTorch 已安装 |
| **测试步骤** | 执行：<br>```python<br>import torchvision<br>print(f"torchvision version: {torchvision.__version__}")<br>``` |
| **预期结果** | 无 ImportError，版本号正确输出 |
| **通过标准** | 导入成功，版本号非空 |

#### TC-DEPS-002: timm 安装验证
| 项目 | 内容 |
|------|------|
| **测试目的** | 验证 timm（PyTorch Image Models）库正确安装 |
| **前置条件** | 无 |
| **测试步骤** | 执行：<br>```python<br>import timm<br>print(f"timm version: {timm.__version__}")<br>print(f"Available models: {len(timm.list_models())} models")<br>``` |
| **预期结果** | 导入成功，模型列表非空 |
| **通过标准** | 导入成功，模型数量 > 0 |

#### TC-DEPS-003: numpy 安装验证
| 项目 | 内容 |
|------|------|
| **测试目的** | 验证 numpy 正确安装 |
| **前置条件** | 无 |
| **测试步骤** | 执行：<br>```python<br>import numpy as np<br>print(f"numpy version: {np.__version__}")<br># 基本运算测试<br>a = np.array([1, 2, 3])<br>assert np.sum(a) == 6, "numpy calculation failed"<br>print("numpy calculation: OK")<br>``` |
| **预期结果** | 导入成功，基本运算正确 |
| **通过标准** | 导入成功，断言通过 |

#### TC-DEPS-004: opencv-python 安装验证
| 项目 | 内容 |
|------|------|
| **测试目的** | 验证 OpenCV 正确安装 |
| **前置条件** | 无 |
| **测试步骤** | 执行：<br>```python<br>import cv2<br>print(f"opencv version: {cv2.__version__}")<br>``` |
| **预期结果** | 导入成功，版本号正确输出 |
| **通过标准** | 导入成功，版本号非空 |

#### TC-DEPS-005: pillow 安装验证
| 项目 | 内容 |
|------|------|
| **测试目的** | 验证 Pillow 正确安装 |
| **前置条件** | 无 |
| **测试步骤** | 执行：<br>```python<br>from PIL import Image<br>import PIL<br>print(f"Pillow version: {PIL.__version__}")<br>``` |
| **预期结果** | 导入成功，版本号正确输出 |
| **通过标准** | 导入成功，版本号非空 |

#### TC-DEPS-006: 实验追踪工具安装验证
| 项目 | 内容 |
|------|------|
| **测试目的** | 验证 tensorboard 或 wandb 安装 |
| **前置条件** | 至少安装其一 |
| **测试步骤** | 执行：<br>```python<br>try:<br>    import tensorboard<br>    print(f"tensorboard: {tensorboard.__version__}")<br>except ImportError:<br>    print("tensorboard: not installed")<br><br>try:<br>    import wandb<br>    print(f"wandb: {wandb.__version__}")<br>except ImportError:<br>    print("wandb: not installed")<br>``` |
| **预期结果** | 至少一个实验追踪工具可用 |
| **通过标准** | tensorboard 或 wandb 至少一个导入成功 |

---

### 2.3 环境综合验证

#### TC-ENV-001: GPU 张量创建测试
| 项目 | 内容 |
|------|------|
| **测试目的** | 验证 GPU 张量能正确创建 |
| **前置条件** | TC-PYTORCH-002 通过 |
| **测试步骤** | 执行：<br>```python<br>import torch<br>x = torch.tensor([1.0, 2.0, 3.0]).cuda()<br>print(f"Device: {x.device}")<br>print(f"Values: {x}")<br>``` |
| **预期结果** | 张量创建成功，device 为 cuda:0 |
| **通过标准** | 无报错，device 显示为 cuda 设备 |

#### TC-ENV-002: GPU 张量运算测试
| 项目 | 内容 |
|------|------|
| **测试目的** | 验证 GPU 上基本矩阵运算正确 |
| **前置条件** | TC-ENV-001 通过 |
| **测试步骤** | 执行：<br>```python<br>import torch<br>a = torch.randn(1000, 1000, device='cuda')<br>b = torch.randn(1000, 1000, device='cuda')<br>c = torch.matmul(a, b)<br>assert c.shape == (1000, 1000), "Shape mismatch"<br>print(f"Matrix multiplication result shape: {c.shape}")<br>``` |
| **预期结果** | 矩阵乘法正确执行，形状正确 |
| **通过标准** | 无报错，结果形状正确 |

#### TC-ENV-003: GPU 内存分配测试
| 项目 | 内容 |
|------|------|
| **测试目的** | 验证 GPU 内存能正确分配和释放 |
| **前置条件** | TC-PYTORCH-002 通过 |
| **测试步骤** | 执行：<br>```python<br>import torch<br>torch.cuda.empty_cache()<br>before = torch.cuda.memory_allocated()<br>x = torch.randn(100, 100, device='cuda')<br>after = torch.cuda.memory_allocated()<br>print(f"Memory before: {before}, after: {after}")<br>del x<br>torch.cuda.empty_cache()<br>final = torch.cuda.memory_allocated()<br>print(f"Memory after cleanup: {final}")<br>``` |
| **预期结果** | 内存分配后增加，释放后减少 |
| **通过标准** | after > before，final ≤ after |

#### TC-ENV-004: 模型前向传播测试
| 项目 | 内容 |
|------|------|
| **测试目的** | 验证完整模型能在 GPU 上运行 |
| **前置条件** | torch 和 torchvision 已安装 |
| **测试步骤** | 执行：<br>```python<br>import torch<br>import torchvision.models as models<br><br>model = models.resnet18(weights=None)<br>model = model.cuda()<br>model.eval()<br><br>x = torch.randn(1, 3, 224, 224, device='cuda')<br>with torch.no_grad():<br>    output = model(x)<br><br>print(f"Input shape: {x.shape}")<br>print(f"Output shape: {output.shape}")<br>``` |
| **预期结果** | 模型加载成功，前向传播正确执行 |
| **通过标准** | 无报错，output.shape == (1, 1000) |

#### TC-ENV-005: PyTorch 与 numpy 互操作测试
| 项目 | 内容 |
|------|------|
| **测试目的** | 验证 PyTorch 张量与 numpy 数组转换正常 |
| **前置条件** | torch 和 numpy 已安装 |
| **测试步骤** | 执行：<br>```python<br>import torch<br>import numpy as np<br><br># numpy -> torch<br>a = np.array([1, 2, 3])<br>t = torch.from_numpy(a)<br>assert t.shape == (3,), "Shape mismatch"<br><br># torch -> numpy<br>t2 = torch.tensor([4, 5, 6])<br>a2 = t2.numpy()<br>assert a2.shape == (3,), "Shape mismatch"<br><br>print("PyTorch-numpy interop: OK")<br>``` |
| **预期结果** | 双向转换成功 |
| **通过标准** | 无报错，形状正确 |

---

## 3. 验证方法与步骤

### 3.1 验证执行流程

```
┌─────────────────────────────────────────────────────────────┐
│                     验证执行流程                              │
├─────────────────────────────────────────────────────────────┤
│  1. 环境检查                                                 │
│     └─ 确认 Python 虚拟环境已激活                            │
│     └─ 确认 CUDA 驱动正常加载                                │
│                                                             │
│  2. PyTorch 验证                                            │
│     └─ TC-PYTORCH-001 ~ TC-PYTORCH-004 顺序执行             │
│     └─ 任一失败则停止后续验证                                │
│                                                             │
│  3. 辅助库验证                                               │
│     └─ TC-DEPS-001 ~ TC-DEPS-006 可并行执行                 │
│     └─ 记录每个库的验证结果                                  │
│                                                             │
│  4. 综合验证                                                 │
│     └─ TC-ENV-001 ~ TC-ENV-005 顺序执行                     │
│     └─ 记录 GPU 运算性能数据                                │
│                                                             │
│  5. 结果汇总                                                 │
│     └─ 生成验证报告                                         │
│     └─ 记录问题和建议                                        │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 单一测试执行方法

每个测试用例可通过以下方式执行：

**方式一：交互式 Python**
```bash
python
>>> import torch
>>> # 执行测试代码
```

**方式二：单行命令**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**方式三：脚本文件**
```bash
python test_pytorch_install.py
```

### 3.3 批量验证脚本

创建 `verify_dl_framework.py` 进行批量验证：

```python
#!/usr/bin/env python
"""深度学习框架安装验证脚本"""
import sys
from dataclasses import dataclass
from typing import Callable, List

@dataclass
class TestResult:
    name: str
    passed: bool
    message: str

def run_test(name: str, test_func: Callable) -> TestResult:
    """执行单个测试"""
    try:
        test_func()
        return TestResult(name, True, "OK")
    except Exception as e:
        return TestResult(name, False, str(e))

# ===== 测试函数定义 =====

def test_torch_import():
    import torch
    assert torch.__version__, "torch version is empty"

def test_cuda_available():
    import torch
    assert torch.cuda.is_available(), "CUDA not available"

def test_gpu_count():
    import torch
    count = torch.cuda.device_count()
    assert count > 0, f"No GPU found, count={count}"

def test_torchvision_import():
    import torchvision
    assert torchvision.__version__, "torchvision version is empty"

def test_timm_import():
    import timm
    assert timm.__version__, "timm version is empty"

def test_numpy_import():
    import numpy as np
    a = np.array([1, 2, 3])
    assert np.sum(a) == 6

def test_cv2_import():
    import cv2
    assert cv2.__version__, "cv2 version is empty"

def test_pillow_import():
    from PIL import Image
    import PIL
    assert PIL.__version__, "PIL version is empty"

def test_tensorboard_or_wandb():
    has_tb = False
    has_wandb = False
    try:
        import tensorboard
        has_tb = True
    except ImportError:
        pass
    try:
        import wandb
        has_wandb = True
    except ImportError:
        pass
    assert has_tb or has_wandb, "Neither tensorboard nor wandb installed"

def test_gpu_tensor_ops():
    import torch
    a = torch.randn(100, 100, device='cuda')
    b = torch.randn(100, 100, device='cuda')
    c = torch.matmul(a, b)
    assert c.shape == (100, 100)

# ===== 主程序 =====

def main():
    tests = [
        ("TC-PYTORCH-001", test_torch_import),
        ("TC-PYTORCH-002", test_cuda_available),
        ("TC-PYTORCH-003", test_gpu_count),
        ("TC-DEPS-001", test_torchvision_import),
        ("TC-DEPS-002", test_timm_import),
        ("TC-DEPS-003", test_numpy_import),
        ("TC-DEPS-004", test_cv2_import),
        ("TC-DEPS-005", test_pillow_import),
        ("TC-DEPS-006", test_tensorboard_or_wandb),
        ("TC-ENV-002", test_gpu_tensor_ops),
    ]
    
    results: List[TestResult] = []
    for name, func in tests:
        result = run_test(name, func)
        results.append(result)
        status = "✓" if result.passed else "✗"
        print(f"{status} {name}: {result.message}")
    
    # 汇总
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"\n{'='*50}")
    print(f"结果: {passed}/{total} 通过")
    
    if passed == total:
        print("所有验证通过！")
        return 0
    else:
        print("部分验证失败，请检查上述错误。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

---

## 4. 通过标准

### 4.1 总体判定规则

| 级别 | 条件 | 说明 |
|------|------|------|
| **完全通过** | 所有 TC-PYTORCH-* 和 TC-ENV-* 通过 | 核心功能正常 |
| **部分通过** | 所有 TC-PYTORCH-* 通过，部分 TC-DEPS-* 失败 | 核心功能正常，部分辅助库缺失 |
| **未通过** | 任一 TC-PYTORCH-* 失败 | PyTorch 核心功能不可用 |

### 4.2 各测试项判定标准

| 测试ID | 通过条件 | 失败影响 |
|--------|----------|----------|
| TC-PYTORCH-001 | 版本号输出，无 ImportError | 阻塞所有后续测试 |
| TC-PYTORCH-002 | `is_available()` 返回 True | 阻塞所有 GPU 相关测试 |
| TC-PYTORCH-003 | device_count >= 1 | 可继续，但影响多 GPU 功能 |
| TC-PYTORCH-004 | 版本号输出 | 可继续，但需记录警告 |
| TC-DEPS-001 | 导入成功 | 影响图像处理功能 |
| TC-DEPS-002 | 导入成功 | 影响预训练模型使用 |
| TC-DEPS-003 | 导入成功 | 影响数值计算 |
| TC-DEPS-004 | 导入成功 | 影响图像 I/O |
| TC-DEPS-005 | 导入成功 | 影响图像处理 |
| TC-DEPS-006 | 至少一个导入成功 | 影响实验追踪 |
| TC-ENV-001 | 张量创建成功 | 阻塞 GPU 运算测试 |
| TC-ENV-002 | 运算结果正确 | GPU 不可用 |
| TC-ENV-003 | 内存正常分配释放 | 可能存在内存泄漏 |
| TC-ENV-004 | 模型运行成功 | 影响模型训练 |
| TC-ENV-005 | 转换正常 | 影响数据处理 |

---

## 5. 自动化验证建议

### 5.1 可完全自动化的测试

| 测试ID | 自动化方式 | 备注 |
|--------|-----------|------|
| TC-PYTORCH-001~004 | Python 脚本 + 断言 | 完全自动化 |
| TC-DEPS-001~006 | Python 脚本 + try/except | 完全自动化 |
| TC-ENV-001~005 | Python 脚本 + 断言 | 完全自动化 |

### 5.2 需要人工检查的项目

| 检查项 | 原因 | 建议 |
|--------|------|------|
| CUDA 版本与硬件兼容性 | 硬件依赖 | 首次安装人工确认 |
| GPU 驱动版本 | 系统依赖 | 首次安装人工确认 |
| 性能基准 | 需对比历史数据 | 人工评估 |

### 5.3 自动化验证脚本部署

```bash
# 将验证脚本放入项目
cp verify_dl_framework.py scripts/

# 添加到 CI/CD 流水线（示例）
# .github/workflows/verify-env.yml
"""
name: Verify DL Environment
on: [push, pull_request]
jobs:
  verify:
    runs-on: self-hosted  # 需要 GPU runner
    steps:
      - uses: actions/checkout@v3
      - name: Verify Deep Learning Framework
        run: python scripts/verify_dl_framework.py
"""
```

### 5.4 快速验证命令

创建一键验证别名：

```bash
# 添加到 ~/.bashrc 或项目 Makefile
alias verify-dl='python scripts/verify_dl_framework.py'
```

---

## 6. 风险点与注意事项

### 6.1 安装风险

| 风险点 | 可能性 | 影响 | 缓解措施 |
|--------|--------|------|----------|
| **CUDA 版本不匹配** | 中 | 高 | 安装前执行 `nvidia-smi` 和 `nvcc --version` 确认版本 |
| **PyTorch 与 CUDA 版本冲突** | 中 | 高 | 使用 PyTorch 官网提供的安装命令，指定 CUDA 版本 |
| **cuDNN 未安装** | 低 | 中 | 验证 `torch.backends.cudnn.version()` |
| **驱动版本过旧** | 中 | 高 | 更新 NVIDIA 驱动到最新稳定版 |

### 6.2 环境风险

| 风险点 | 可能性 | 影响 | 缓解措施 |
|--------|--------|------|----------|
| **虚拟环境未激活** | 高 | 低 | 验证脚本开头检查 `VIRTUAL_ENV` 环境变量 |
| **多个 Python 版本冲突** | 中 | 中 | 使用 `which python` 确认路径 |
| **PATH 环境变量错误** | 中 | 中 | 确保 CUDA 路径在 PATH 中 |
| **GPU 内存不足** | 低 | 中 | 测试时使用小批量数据 |

### 6.3 兼容性风险

| 风险点 | 可能性 | 影响 | 缓解措施 |
|--------|--------|------|----------|
| **timm 与 torch 版本不兼容** | 低 | 中 | 查看 timm 文档确认兼容版本 |
| **torchvision 与 torch 版本不匹配** | 中 | 高 | 使用 `pip install torchvision` 会自动匹配版本 |
| **numpy 版本冲突** | 低 | 低 | 使用 `pip check` 验证依赖冲突 |

### 6.4 验证注意事项

1. **首次安装后重启**
   - 安装 CUDA 相关库后，建议重启终端或重新登录
   - 确保环境变量生效

2. **多 GPU 环境**
   - 如有多个 GPU，需测试 `CUDA_VISIBLE_DEVICES` 环境变量
   - 验证多 GPU 并行（如使用 DataParallel）

3. **权限问题**
   - 避免使用 `sudo pip install`，使用虚拟环境
   - 确保 GPU 设备权限正确（Linux 下 `/dev/nvidia*`）

4. **网络问题**
   - 使用国内镜像源加速下载
   - PyTorch 官方源可能较慢，可配置清华/阿里镜像

### 6.5 验证失败排查流程

```
验证失败
    │
    ├─ ImportError: No module named 'torch'
    │       └─ 检查虚拟环境是否激活
    │       └─ 检查 pip list 是否包含 torch
    │
    ├─ torch.cuda.is_available() == False
    │       └─ 检查 nvidia-smi 是否正常
    │       └─ 检查 CUDA 是否安装
    │       └─ 检查 PyTorch 是否为 CUDA 版本
    │
    ├─ CUDA out of memory
    │       └─ 减少 batch size
    │       └─ 检查是否有其他进程占用 GPU
    │
    └─ cuDNN 相关错误
            └─ 安装对应 CUDA 版本的 cuDNN
            └─ 检查环境变量 CUDA_HOME, LD_LIBRARY_PATH
```

---

## 附录 A：验证脚本输出示例

```
========================================
深度学习框架安装验证
========================================

[PyTorch 验证]
✓ TC-PYTORCH-001: torch version 2.1.0+cu118
✓ TC-PYTORCH-002: CUDA available: True
✓ TC-PYTORCH-003: GPU count: 1, GPU 0: NVIDIA GeForce RTX 3090
✓ TC-PYTORCH-004: CUDA version: 11.8, cuDNN version: 8902

[辅助库验证]
✓ TC-DEPS-001: torchvision 0.16.0
✓ TC-DEPS-002: timm 0.9.12, models: 1185
✓ TC-DEPS-003: numpy 1.24.3
✓ TC-DEPS-004: opencv 4.8.1
✓ TC-DEPS-005: Pillow 10.1.0
✓ TC-DEPS-006: tensorboard 2.15.0

[环境验证]
✓ TC-ENV-001: GPU tensor creation OK
✓ TC-ENV-002: GPU matrix ops OK (1000x1000 @ 1000x1000)
✓ TC-ENV-003: Memory alloc OK
✓ TC-ENV-004: Model forward OK (ResNet18)
✓ TC-ENV-005: PyTorch-numpy interop OK

========================================
结果: 15/15 通过
所有验证通过！
========================================
```

---

## 附录 B：环境信息收集脚本

```python
#!/usr/bin/env python
"""收集环境诊断信息"""
import subprocess
import sys

def run_cmd(cmd: str) -> str:
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip() or result.stderr.strip()
    except Exception as e:
        return str(e)

print("=" * 60)
print("环境诊断信息")
print("=" * 60)

print(f"\n[Python]")
print(f"  版本: {sys.version}")
print(f"  路径: {sys.executable}")

print(f"\n[CUDA]")
print(f"  nvidia-smi: {run_cmd('nvidia-smi --query-gpu=name,driver_version --format=csv,noheader')}")
print(f"  nvcc: {run_cmd('nvcc --version')}")

print(f"\n[已安装包]")
for pkg in ['torch', 'torchvision', 'timm', 'numpy', 'opencv-python', 'pillow', 'tensorboard', 'wandb']:
    result = run_cmd(f"pip show {pkg}")
    if result and "not found" not in result.lower():
        version = [l for l in result.split('\n') if l.startswith('Version:')]
        print(f"  {pkg}: {version[0].split(':')[1].strip() if version else 'unknown'}")
    else:
        print(f"  {pkg}: 未安装")

try:
    import torch
    print(f"\n[PyTorch 详情]")
    print(f"  版本: {torch.__version__}")
    print(f"  CUDA 可用: {torch.cuda.is_available()}")
    print(f"  CUDA 版本: {torch.version.cuda}")
    print(f"  cuDNN 版本: {torch.backends.cudnn.version()}")
    if torch.cuda.is_available():
        print(f"  GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
except ImportError:
    print(f"\n[PyTorch 详情] torch 未安装")
```