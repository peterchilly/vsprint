# 硬件评估报告 — 验证测试设计

> **来源任务：** docs/superpowers/tasks/01_环境搭建/hardware_report.md
> **生成日期：** 2026-06-02

---

## 1. 验证目标概述

本验证方案旨在确认硬件评估报告中的各项数据准确无误，并验证后续行动项的完成状态。验证分为两个层面：

| 验证层面 | 目标 | 验证时机 |
|----------|------|----------|
| **数据准确性验证** | 确认报告中的硬件参数与实际系统一致 | 报告生成后立即验证 |
| **行动项完成验证** | 确认后续安装配置任务正确执行 | 每项任务完成后验证 |

### 验收标准映射

| 报告验收项 | 验证方法 | 验证类型 |
|------------|----------|----------|
| GPU 型号/显存正确 | 命令行验证 | 自动化 |
| CUDA/驱动版本正确 | 命令行验证 | 自动化 |
| CPU/内存参数正确 | 系统命令验证 | 自动化 |
| 磁盘空间数据正确 | 系统命令验证 | 自动化 |
| Python 版本确认 | 命令行验证 | 自动化 |
| PyTorch 安装成功 | 导入测试 + CUDA 可用性 | 自动化 |
| 数据存储路径配置正确 | 路径存在性检查 | 自动化 |
| CUDA Toolkit 可用 | nvcc 命令验证 | 自动化 |

---

## 2. 测试用例设计

### 2.1 GPU 资源验证

| 用例 ID | 用例名称 | 验证内容 | 优先级 |
|---------|----------|----------|--------|
| TC-GPU-001 | GPU 型号验证 | 确认 GPU 型号为 RTX 5070 Ti | P0 |
| TC-GPU-002 | 显存大小验证 | 确认显存 ≥ 16GB | P0 |
| TC-GPU-003 | 驱动版本验证 | 确认驱动版本 591.86 或更高 | P1 |
| TC-GPU-004 | CUDA 驱动版本验证 | 确认驱动支持 CUDA 13.1 | P1 |
| TC-GPU-005 | 计算能力验证 | 确认 SM 12.0 (Blackwell) | P1 |
| TC-GPU-006 | GPU 功耗确认 | 确认功耗 300W，电源充足 | P2 |

### 2.2 训练设备验证

| 用例 ID | 用例名称 | 验证内容 | 优先级 |
|---------|----------|----------|--------|
| TC-SYS-001 | 操作系统验证 | 确认 Windows 11 x64 | P1 |
| TC-SYS-002 | 架构验证 | 确认 x64 架构 | P2 |

### 2.3 系统资源验证

| 用例 ID | 用例名称 | 验证内容 | 优先级 |
|---------|----------|----------|--------|
| TC-CPU-001 | CPU 型号验证 | 确认 Intel Core i5-12600KF | P1 |
| TC-CPU-002 | CPU 核心数验证 | 确认 10 核 16 线程 | P0 |
| TC-CPU-003 | 内存大小验证 | 确认 32GB | P0 |
| TC-DISK-001 | C 盘空间验证 | 确认可用空间 ≥ 50GB | P1 |
| TC-DISK-002 | D 盘空间验证 | 确认可用空间 ≥ 50GB 或已清理 | P1 |
| TC-DISK-003 | E 盘空间验证 | 确认可用空间 ≥ 200GB | P0 |
| TC-DISK-004 | F 盘空间验证 | 确认可用空间 ≥ 100GB | P2 |

### 2.4 软件环境验证

| 用例 ID | 用例名称 | 验证内容 | 优先级 |
|---------|----------|----------|--------|
| TC-SW-001 | Python 版本验证 | 确认 Python 3.10 或 3.12 可用 | P0 |
| TC-SW-002 | PyTorch 安装验证 | 确认 PyTorch 已安装 | P0 |
| TC-SW-003 | PyTorch CUDA 验证 | 确认 `torch.cuda.is_available()` 返回 True | P0 |
| TC-SW-004 | CUDA Toolkit 验证 | 确认 nvcc 可用（可选） | P2 |
| TC-SW-005 | Git 安装验证 | 确认 Git 已安装 | P1 |

### 2.5 行动项完成验证

| 用例 ID | 用例名称 | 验证内容 | 优先级 |
|---------|----------|----------|--------|
| TC-ACT-001 | Python 环境验证 | 确认 Python 版本兼容 | P0 |
| TC-ACT-002 | PyTorch CUDA 版本验证 | 确认安装 CUDA 12.1 版本 PyTorch | P0 |
| TC-ACT-003 | 数据存储路径验证 | 确认 E: 盘存储路径已配置 | P0 |
| TC-ACT-004 | D 盘清理验证 | 确认 D: 盘空间充足 | P1 |
| TC-ACT-005 | CUDA Toolkit 验证 | 确认 nvcc 命令可用 | P2 |

---

## 3. 验证方法与步骤

### 3.1 GPU 验证命令

#### TC-GPU-001 ~ TC-GPU-006

```powershell
# TC-GPU-001: GPU 型号验证
# 预期输出包含: NVIDIA GeForce RTX 5070 Ti
nvidia-smi --query-gpu=name --format=csv,noheader

# TC-GPU-002: 显存大小验证
# 预期输出: 16384 (MB) 或更高
nvidia-smi --query-gpu=memory.total --format=csv,noheader

# TC-GPU-003: 驱动版本验证
# 预期输出: 591.86 或更高
nvidia-smi --query-gpu=driver_version --format=csv,noheader

# TC-GPU-004: CUDA 驱动版本验证
# 预期输出: 13.1 或更高
nvidia-smi --query-gpu=cuda_version --format=csv,noheader

# TC-GPU-005: 计算能力验证
# 需要 Python + PyTorch
python -c "import torch; print(torch.cuda.get_device_capability())"
# 预期输出: (12, 0)

# TC-GPU-006: GPU 功耗确认
nvidia-smi --query-gpu=power.limit --format=csv,noheader
# 预期输出: 300.00 W
# 手动确认: 电源额定功率 ≥ 600W
```

### 3.2 系统资源验证命令

#### TC-SYS-001 ~ TC-SYS-002, TC-CPU-001 ~ TC-CPU-003

```powershell
# TC-SYS-001, TC-SYS-002: 操作系统和架构验证
systeminfo | findstr /B /C:"OS Name" /C:"OS Version" /C:"System Type"
# 预期输出包含: Windows 11, x64-based PC

# TC-CPU-001, TC-CPU-002: CPU 型号和核心数验证
wmic cpu get Name, NumberOfCores, NumberOfLogicalProcessors /format:list
# 预期输出: Name=Intel(R) Core(TM) i5-12600KF
#          NumberOfCores=10 (或 6P+4E 结构)
#          NumberOfLogicalProcessors=16

# TC-CPU-003: 内存大小验证
wmic OS get TotalVisibleMemorySize /value
# 预期输出: TotalVisibleMemorySize=33554432 (KB) ≈ 32GB
# 或使用
systeminfo | findstr /C:"Total Physical Memory"
```

#### TC-DISK-001 ~ TC-DISK-004

```powershell
# 磁盘空间验证 (所有盘符)
wmic logicaldisk get DeviceID,Size,FreeSpace /format:table

# 或者逐个验证
# TC-DISK-001: C 盘
fsutil volume diskfree C:
# 预期: 可用空间 ≥ 50GB (53687091200 bytes)

# TC-DISK-002: D 盘
fsutil volume diskfree D:
# 预期: 可用空间 ≥ 50GB 或已清理

# TC-DISK-003: E 盘
fsutil volume diskfree E:
# 预期: 可用空间 ≥ 200GB (214748364800 bytes)

# TC-DISK-004: F 盘
fsutil volume diskfree F:
# 预期: 可用空间 ≥ 100GB (107374182400 bytes)
```

### 3.3 软件环境验证命令

#### TC-SW-001 ~ TC-SW-005

```powershell
# TC-SW-001: Python 版本验证
python --version
# 预期输出: Python 3.10.x 或 Python 3.12.x

# TC-SW-002: PyTorch 安装验证
python -c "import torch; print(torch.__version__)"
# 预期输出: 2.x.x (具体版本号)

# TC-SW-003: PyTorch CUDA 可用性验证
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
# 预期输出: CUDA available: True
#          CUDA version: 12.1

# TC-SW-004: CUDA Toolkit 验证 (可选)
nvcc --version
# 预期输出: nvcc: NVIDIA Cuda compiler driver
#          version 12.x

# TC-SW-005: Git 安装验证
git --version
# 预期输出: git version 2.x.x
```

### 3.4 行动项完成验证命令

#### TC-ACT-001 ~ TC-ACT-005

```powershell
# TC-ACT-001: Python 环境兼容性验证
python -c "import sys; print(f'Python {sys.version_info.major}.{sys.version_info.minor}'); assert sys.version_info.major == 3 and sys.version_info.minor in [10, 11, 12], 'Python version incompatible'"
# 预期输出: Python 3.10 或 Python 3.11 或 Python 3.12
#          无 AssertionError

# TC-ACT-002: PyTorch CUDA 版本验证
python -c "import torch; ver = torch.version.cuda; print(f'PyTorch CUDA version: {ver}'); assert ver and ver.startswith('12'), 'CUDA version mismatch'"
# 预期输出: PyTorch CUDA version: 12.x
#          无 AssertionError

# TC-ACT-003: 数据存储路径验证
# 检查 E 盘是否有项目数据目录
if exist "E:\vsprint_data" (echo Data path configured) else (echo Data path NOT configured)
# 或检查配置文件中的数据路径设置

# TC-ACT-004: D 盘清理验证
wmic logicaldisk where "DeviceID='D:'" get FreeSpace /value
# 预期: FreeSpace ≥ 53687091200 (50GB)

# TC-ACT-005: CUDA Toolkit 验证
nvcc --version
# 预期输出包含: version 12.x
```

---

## 4. 通过标准

### 4.1 量化通过标准

| 用例 ID | 通过条件 | 失败判定 |
|---------|----------|----------|
| TC-GPU-001 | 输出包含 "RTX 5070 Ti" 或同等/更高型号 | GPU 型号不匹配 |
| TC-GPU-002 | 显存 ≥ 16384 MB | 显存 < 16GB |
| TC-GPU-003 | 驱动版本 ≥ 591.86 | 驱动版本过低 |
| TC-GPU-004 | CUDA 版本 ≥ 12.0 | CUDA 版本 < 12.0 |
| TC-GPU-005 | 计算能力 = (12, 0) | 计算能力不匹配 |
| TC-GPU-006 | 功耗限制 = 300W 且电源 ≥ 600W | 电源功率不足 |
| TC-CPU-002 | 核心数 = 10, 线程数 = 16 | 核心数/线程数不符 |
| TC-CPU-003 | 内存 ≥ 32GB | 内存 < 32GB |
| TC-DISK-003 | E: 盘可用空间 ≥ 200GB | E: 盘空间不足 |
| TC-SW-001 | Python 3.10 或 3.12 可用 | Python 未安装或版本不兼容 |
| TC-SW-002 | torch 模块可导入 | PyTorch 未安装 |
| TC-SW-003 | `torch.cuda.is_available()` = True | CUDA 不可用 |
| TC-ACT-002 | PyTorch CUDA 版本以 "12" 开头 | CUDA 版本不匹配 |

### 4.2 综合判定标准

| 等级 | 条件 | 后续处理 |
|------|------|----------|
| **✅ 全部通过** | P0 用例全部通过，无失败项 | 环境就绪，可开始训练 |
| **⚠️ 条件通过** | P0 用例全部通过，P1/P2 有失败 | 可开始训练，但需关注风险项 |
| **❌ 验证失败** | 任一 P0 用例失败 | 必须修复后再进行训练 |

---

## 5. 自动化验证建议

### 5.1 可自动化的验证项

| 验证项 | 自动化方式 | 脚本类型 |
|--------|------------|----------|
| GPU 信息采集 | nvidia-smi + PowerShell 解析 | PowerShell |
| CPU/内存信息 | WMI 查询 | PowerShell |
| 磁盘空间 | wmic/fsutil | PowerShell |
| Python/PyTorch 验证 | Python 脚本 | Python |
| CUDA 可用性 | torch.cuda API | Python |

### 5.2 自动化验证脚本示例

**verify_hardware.ps1:**

```powershell
# 硬件验证自动化脚本
# 用法: .\verify_hardware.ps1

Write-Host "=== 硬件验证开始 ===" -ForegroundColor Cyan

# TC-GPU-001 ~ TC-GPU-004
Write-Host "`n[GPU 验证]" -ForegroundColor Yellow
$gpu = nvidia-smi --query-gpu=name,memory.total,driver_version,cuda_version --format=csv,noheader
Write-Host "GPU 信息: $gpu"

# TC-CPU-001 ~ TC-CPU-003
Write-Host "`n[CPU/内存验证]" -ForegroundColor Yellow
$cpu = wmic cpu get Name, NumberOfCores, NumberOfLogicalProcessors /format:list | Where-Object { $_ -ne "" }
Write-Host "CPU 信息:"
$cpu
$mem = wmic OS get TotalVisibleMemorySize /value | Where-Object { $_ -ne "" }
Write-Host "内存: $mem"

# TC-DISK-001 ~ TC-DISK-004
Write-Host "`n[磁盘验证]" -ForegroundColor Yellow
$disks = wmic logicaldisk get DeviceID,Size,FreeSpace /format:table
Write-Host "磁盘信息:"
$disks

# TC-SW-001 ~ TC-SW-003
Write-Host "`n[软件环境验证]" -ForegroundColor Yellow
Write-Host "Python 版本:"
python --version

Write-Host "`nPyTorch 验证:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

Write-Host "`n=== 硬件验证完成 ===" -ForegroundColor Cyan
```

**verify_torch_cuda.py:**

```python
"""
PyTorch CUDA 完整验证脚本
运行: python verify_torch_cuda.py
"""

import sys

def verify():
    results = []

    # TC-SW-002: PyTorch 安装验证
    try:
        import torch
        results.append(("PyTorch 安装", True, torch.__version__))
    except ImportError:
        results.append(("PyTorch 安装", False, "未安装"))
        print_results(results)
        return

    # TC-SW-003: CUDA 可用性
    cuda_available = torch.cuda.is_available()
    results.append(("CUDA 可用", cuda_available, str(cuda_available)))

    if cuda_available:
        # TC-GPU-005: 计算能力
        capability = torch.cuda.get_device_capability()
        results.append(("计算能力", capability == (12, 0), str(capability)))

        # 显存信息
        mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        results.append(("显存总量", mem_total >= 16, f"{mem_total:.2f} GB"))

        # CUDA 版本
        results.append(("CUDA 版本", torch.version.cuda.startswith("12"), torch.version.cuda))

    print_results(results)

    # 综合判定
    passed = all(r[1] for r in results)
    print(f"\n{'✅ 全部通过' if passed else '❌ 存在失败项'}")
    sys.exit(0 if passed else 1)

def print_results(results):
    print(f"\n{'验证项':<20} {'状态':<8} {'实际值'}")
    print("-" * 60)
    for name, passed, value in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name:<20} {status:<8} {value}")

if __name__ == "__main__":
    verify()
```

### 5.3 需要手动验证的项目

| 验证项 | 手动原因 | 验证方式 |
|--------|----------|----------|
| TC-GPU-006 功耗确认 | 需确认电源硬件 | 查看电源铭牌或系统信息 |
| 数据路径配置正确性 | 需确认项目配置 | 检查项目配置文件或环境变量 |
| D 盘清理效果 | 需判断清理内容 | 确认已删除非必要文件 |

---

## 6. 风险点与注意事项

### 6.1 硬件风险

| 风险项 | 影响 | 缓解措施 |
|--------|------|----------|
| **RTX 5070 Ti 新架构** | Blackwell 架构较新，部分软件可能不兼容 | 使用最新版 PyTorch/CUDA；关注 NVIDIA 发布的兼容性说明 |
| **D 盘空间不足** | 6GB 可能在训练过程中耗尽 | 优先使用 E: 盘存储数据和模型 |
| **CPU 10 核瓶颈** | 数据加载可能成为瓶颈 | DataLoader `num_workers` 设为 6-8 |
| **功耗 300W** | 高负载可能触发电源保护 | 确认电源 ≥ 750W；监控训练时功耗 |

### 6.2 软件风险

| 风险项 | 影响 | 缓解措施 |
|--------|------|----------|
| **Python 3.12 兼容性** | 部分深度学习库不支持 3.12 | 建议安装 Python 3.10 或 3.11 |
| **CUDA 13.1 驱动** | PyTorch 可能尚未支持 CUDA 13.x | 安装 PyTorch CUDA 12.1 版本（向下兼容） |
| **PyTorch + Blackwell** | 新架构可能需要特殊编译 | 使用 PyTorch nightly 或 NVIDIA 优化版本 |

### 6.3 验证执行注意事项

1. **验证时机**
   - 硬件数据验证：系统启动后、无高负载任务时执行
   - 软件验证：每次环境变更后重新验证

2. **环境变量**
   - 确保 Python、CUDA、PyTorch 路径已加入 PATH
   - 验证前检查 `CUDA_HOME` 环境变量

3. **权限要求**
   - 部分 WMI 命令需要管理员权限
   - 磁盘空间验证需要读取权限

4. **验证顺序**
   ```
   硬件验证 → 软件安装 → 软件验证 → 集成测试
   ```

5. **回滚计划**
   - 验证失败时记录详细错误信息
   - 保留原有环境快照（如使用虚拟环境）
   - 失败后先检查驱动/版本兼容性

---

## 附录：验证检查清单

```
□ GPU 验证
  □ TC-GPU-001 GPU 型号正确
  □ TC-GPU-002 显存 ≥ 16GB
  □ TC-GPU-003 驱动版本正确
  □ TC-GPU-004 CUDA 版本正确
  □ TC-GPU-005 计算能力正确
  □ TC-GPU-006 电源功率充足

□ 系统验证
  □ TC-CPU-001 CPU 型号正确
  □ TC-CPU-002 CPU 核心/线程正确
  □ TC-CPU-003 内存 ≥ 32GB
  □ TC-DISK-001~004 磁盘空间充足

□ 软件验证
  □ TC-SW-001 Python 版本正确
  □ TC-SW-002 PyTorch 已安装
  □ TC-SW-003 CUDA 可用
  □ TC-SW-004 nvcc 可用（可选）
  □ TC-SW-005 Git 已安装

□ 行动项验证
  □ TC-ACT-001 Python 环境兼容
  □ TC-ACT-002 PyTorch CUDA 版本正确
  □ TC-ACT-003 数据路径已配置
  □ TC-ACT-004 D 盘已清理
  □ TC-ACT-005 CUDA Toolkit 可用
```