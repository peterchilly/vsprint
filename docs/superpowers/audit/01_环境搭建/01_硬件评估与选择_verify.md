# 硬件评估与选择 — 验证测试设计

> 来源任务：docs/superpowers/tasks/01_环境搭建/01_硬件评估与选择.md
> 生成日期：2026-06-02

---

## 1. 验证目标概述

本验证方案针对「硬件评估与选择」任务的三个验收标准，设计具体的验证方法和测试用例：

| 验收标准 | 验证目标 | 验证方法 |
|---------|---------|---------|
| 硬件配置文档完成 | 确认文档存在、内容完整、格式规范 | 文档检查 + 内容审计 |
| GPU型号、显存、CUDA版本确认 | 验证GPU硬件信息准确可查 | 命令行验证 + 脚本检测 |
| 系统资源满足最低要求 | 确认CPU、内存、磁盘满足阈值 | 自动化脚本检测 |

**验证原则：**
- 所有硬件信息必须可复现查询
- 配置文档必须与实际系统状态一致
- 自动化检测优先于人工确认

---

## 2. 测试用例设计

### 2.1 GPU资源评估测试用例

#### TC-GPU-001: GPU显存大小验证
| 属性 | 内容 |
|-----|------|
| **测试项** | GPU显存大小确认 |
| **前置条件** | NVIDIA驱动已安装 |
| **测试步骤** | 1. 执行 `nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits`<br>2. 记录输出值（单位：MB）<br>3. 对比最低要求（8192 MB） |
| **预期结果** | 输出值 ≥ 8192 MB |
| **通过标准** | 显存 ≥ 8GB，否则标记为「不满足训练要求」 |
| **自动化** | 可自动化，建议集成到环境检测脚本 |

#### TC-GPU-002: CUDA版本兼容性验证
| 属性 | 内容 |
|-----|------|
| **测试项** | CUDA版本与PyTorch兼容性 |
| **前置条件** | CUDA Toolkit已安装 |
| **测试步骤** | 1. 执行 `nvcc --version` 获取CUDA版本<br>2. 执行 `nvidia-smi` 确认驱动支持的最高CUDA版本<br>3. 查询PyTorch官方兼容性矩阵<br>4. 确认PyTorch版本与CUDA版本匹配 |
| **预期结果** | CUDA版本在PyTorch支持范围内 |
| **通过标准** | CUDA版本 ≥ PyTorch最低要求，且 ≤ 驱动支持的最高版本 |
| **自动化** | 半自动化（版本查询可自动化，兼容性判断需参考官方文档） |

#### TC-GPU-003: GPU型号与计算能力验证
| 属性 | 内容 |
|-----|------|
| **测试项** | GPU型号和计算能力记录 |
| **前置条件** | NVIDIA驱动已安装 |
| **测试步骤** | 1. 执行 `nvidia-smi --query-gpu=name,compute_cap --format=csv`<br>2. 记录GPU型号（如 RTX 3090）<br>3. 记录计算能力（如 8.6）<br>4. 写入硬件配置文档 |
| **预期结果** | 成功获取型号和计算能力 |
| **通过标准** | 信息完整记录到配置文档，计算能力 ≥ 3.5（PyTorch最低要求） |
| **自动化** | 可自动化 |

### 2.2 训练设备确定测试用例

#### TC-DEVICE-001: 训练设备方案评估
| 属性 | 内容 |
|-----|------|
| **测试项** | 训练设备选项评估记录 |
| **前置条件** | 无 |
| **测试步骤** | 1. 列出可用训练设备选项<br>2. 评估各选项的优缺点<br>3. 记录预算和时间约束<br>4. 确定最终方案并记录理由 |
| **预期结果** | 有明确的设备选择决策记录 |
| **通过标准** | 文档中包含：选项列表、评估依据、最终选择、选择理由 |
| **自动化** | 不可自动化（需人工决策） |

### 2.3 系统资源检查测试用例

#### TC-SYS-001: CPU核心数验证
| 属性 | 内容 |
|-----|------|
| **测试项** | CPU核心数检查 |
| **前置条件** | 无 |
| **测试步骤** | 1. Linux: 执行 `nproc` 或 `lscpu \| grep "CPU(s)"`<br>2. Windows: 执行 `wmic cpu get NumberOfCores` 或 PowerShell `(Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors`<br>3. 记录逻辑核心数 |
| **预期结果** | 核心数 ≥ 8 |
| **通过标准** | 逻辑核心数 ≥ 8，否则标记为「可能影响数据加载效率」 |
| **自动化** | 可自动化 |

#### TC-SYS-002: 内存大小验证
| 属性 | 内容 |
|-----|------|
| **测试项** | 系统内存检查 |
| **前置条件** | 无 |
| **测试步骤** | 1. Linux: 执行 `free -h` 或 `cat /proc/meminfo \| grep MemTotal`<br>2. Windows: 执行 `wmic computersystem get TotalPhysicalMemory`<br>3. 将字节数转换为GB |
| **预期结果** | 内存 ≥ 16 GB |
| **通过标准** | 可用内存 ≥ 16GB，否则标记为「可能需要减小batch size」 |
| **自动化** | 可自动化 |

#### TC-SYS-003: 磁盘空间验证
| 属性 | 内容 |
|-----|------|
| **测试项** | 磁盘空间检查 |
| **前置条件** | 无 |
| **测试步骤** | 1. 确认项目将存储的数据类型（数据集、模型、日志）<br>2. Linux: 执行 `df -h /path/to/project`<br>3. Windows: 执行 `wmic logicaldisk get size,freespace,caption` 或 PowerShell `Get-PSDrive -PSProvider FileSystem`<br>4. 评估可用空间 |
| **预期结果** | 可用空间 ≥ 200 GB |
| **通过标准** | 可用空间 ≥ 200GB，建议SSD，否则标记为「可能影响I/O性能」 |
| **自动化** | 可自动化 |

#### TC-SYS-004: SSD类型验证
| 属性 | 内容 |
|-----|------|
| **测试项** | 磁盘类型检查（SSD/HDD） |
| **前置条件** | 无 |
| **测试步骤** | 1. Linux: 执行 `lsblk -d -o name,rota`（rota=0为SSD）<br>2. Windows: 执行 PowerShell `Get-PhysicalDisk \| Select MediaType`<br>3. 记录磁盘类型 |
| **预期结果** | 磁盘类型为SSD |
| **通过标准** | 为SSD则标记为「推荐」，为HDD则标记为「可能影响训练速度」 |
| **自动化** | 可自动化 |

---

## 3. 验证方法与步骤

### 3.1 自动化验证脚本（推荐）

创建环境检测脚本 `scripts/check_environment.py`，执行全部自动化测试用例：

```python
#!/usr/bin/env python3
"""
环境检测脚本 - 硬件评估验证
运行方式: python scripts/check_environment.py
"""

import subprocess
import sys
import json
from pathlib import Path

def check_gpu_memory():
    """TC-GPU-001: GPU显存验证"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        memory_mb = int(result.stdout.strip())
        passed = memory_mb >= 8192
        return {
            'test': 'TC-GPU-001',
            'name': 'GPU显存大小',
            'value': f'{memory_mb} MB',
            'passed': passed,
            'message': '满足要求' if passed else '不满足最低要求(8GB)'
        }
    except Exception as e:
        return {'test': 'TC-GPU-001', 'name': 'GPU显存大小', 'passed': False, 'message': str(e)}

def check_cuda_version():
    """TC-GPU-002: CUDA版本验证"""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        # 解析版本号
        import re
        match = re.search(r'release (\d+\.\d+)', result.stdout)
        if match:
            version = match.group(1)
            return {
                'test': 'TC-GPU-002',
                'name': 'CUDA版本',
                'value': version,
                'passed': True,
                'message': f'CUDA {version} 已安装'
            }
    except Exception as e:
        return {'test': 'TC-GPU-002', 'name': 'CUDA版本', 'passed': False, 'message': str(e)}

def check_gpu_info():
    """TC-GPU-003: GPU型号与计算能力"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,compute_cap', '--format=csv,noheader'],
            capture_output=True, text=True
        )
        name, compute_cap = result.stdout.strip().split(', ')
        cap_num = float(compute_cap)
        passed = cap_num >= 3.5
        return {
            'test': 'TC-GPU-003',
            'name': 'GPU型号与计算能力',
            'value': f'{name}, 计算能力 {compute_cap}',
            'passed': passed,
            'message': '满足PyTorch要求' if passed else '计算能力过低'
        }
    except Exception as e:
        return {'test': 'TC-GPU-003', 'name': 'GPU型号与计算能力', 'passed': False, 'message': str(e)}

def check_cpu_cores():
    """TC-SYS-001: CPU核心数"""
    import os
    cores = os.cpu_count()
    passed = cores >= 8
    return {
        'test': 'TC-SYS-001',
        'name': 'CPU核心数',
        'value': f'{cores} 核',
        'passed': passed,
        'message': '满足要求' if passed else '建议8核以上'
    }

def check_memory():
    """TC-SYS-002: 内存大小"""
    import psutil
    mem = psutil.virtual_memory()
    mem_gb = mem.total / (1024**3)
    passed = mem_gb >= 16
    return {
        'test': 'TC-SYS-002',
        'name': '系统内存',
        'value': f'{mem_gb:.1f} GB',
        'passed': passed,
        'message': '满足要求' if passed else '建议16GB以上'
    }

def check_disk_space():
    """TC-SYS-003: 磁盘空间"""
    import psutil
    disk = psutil.disk_usage('/')
    free_gb = disk.free / (1024**3)
    passed = free_gb >= 200
    return {
        'test': 'TC-SYS-003',
        'name': '磁盘可用空间',
        'value': f'{free_gb:.1f} GB',
        'passed': passed,
        'message': '满足要求' if passed else '建议200GB以上'
    }

def main():
    print("=" * 60)
    print("硬件环境检测 - ERes2NetV2 训练需求验证")
    print("=" * 60)

    tests = [
        check_gpu_memory(),
        check_cuda_version(),
        check_gpu_info(),
        check_cpu_cores(),
        check_memory(),
        check_disk_space(),
    ]

    passed = sum(1 for t in tests if t['passed'])
    total = len(tests)

    for test in tests:
        status = "✓ PASS" if test['passed'] else "✗ FAIL"
        print(f"\n[{status}] {test['name']}")
        if 'value' in test:
            print(f"  值: {test['value']}")
        print(f"  说明: {test['message']}")

    print("\n" + "=" * 60)
    print(f"检测结果: {passed}/{total} 项通过")
    print("=" * 60)

    # 保存结果到JSON
    result_file = Path('docs/hardware_check_result.json')
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({'tests': tests, 'summary': {'passed': passed, 'total': total}}, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到: {result_file}")

    return 0 if passed == total else 1

if __name__ == '__main__':
    sys.exit(main())
```

### 3.2 手动验证步骤

对于无法自动化的测试项（如 TC-DEVICE-001 训练设备评估），按以下步骤手动验证：

1. **文档存在性检查**
   - 确认 `docs/hardware_config.md` 文件存在
   - 检查文档中是否包含所有必填字段

2. **决策记录审查**
   - 检查是否记录了设备选项对比
   - 确认有明确的最终选择
   - 验证选择理由是否合理

### 3.3 验证执行顺序

```
┌─────────────────────────────────────────────────────────────┐
│ 1. 运行自动化脚本                                            │
│    python scripts/check_environment.py                      │
├─────────────────────────────────────────────────────────────┤
│ 2. 检查脚本输出结果                                          │
│    - 确认所有自动化测试通过或记录失败原因                     │
├─────────────────────────────────────────────────────────────┤
│ 3. 手动完成训练设备评估文档                                  │
│    - 填写 docs/hardware_config.md                           │
├─────────────────────────────────────────────────────────────┤
│ 4. 最终验收                                                  │
│    - 对比验收标准逐项确认                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. 通过标准

### 4.1 各测试用例通过标准汇总

| 测试ID | 测试项 | 通过条件 | 失败影响 |
|--------|-------|---------|---------|
| TC-GPU-001 | GPU显存 | ≥ 8GB | 阻塞：无法训练模型 |
| TC-GPU-002 | CUDA版本 | 与PyTorch兼容 | 阻塞：GPU不可用 |
| TC-GPU-003 | 计算能力 | ≥ 3.5 | 阻塞：PyTorch不支持 |
| TC-DEVICE-001 | 设备评估 | 有决策记录 | 阻塞：无明确方案 |
| TC-SYS-001 | CPU核心 | ≥ 8核 | 警告：影响数据加载 |
| TC-SYS-002 | 内存 | ≥ 16GB | 警告：需减小batch size |
| TC-SYS-003 | 磁盘空间 | ≥ 200GB | 警告：可能空间不足 |

### 4.2 总体验收判定规则

| 结果 | 条件 |
|-----|------|
| **通过** | 所有阻塞项通过，警告项≤1 |
| **有条件通过** | 所有阻塞项通过，警告项>1 |
| **不通过** | 任一阻塞项失败 |

### 4.3 硬件配置文档完整性检查

配置文档 `docs/hardware_config.md` 必须包含：

- [ ] GPU型号
- [ ] GPU显存大小
- [ ] CUDA版本
- [ ] 计算能力
- [ ] CPU型号和核心数
- [ ] 内存大小
- [ ] 磁盘空间和类型
- [ ] 训练设备选择及理由
- [ ] 检测日期和检测人

---

## 5. 自动化验证建议

### 5.1 可完全自动化的测试

| 测试ID | 自动化方式 | 输出 |
|--------|-----------|------|
| TC-GPU-001 | nvidia-smi命令 | 显存MB数 |
| TC-GPU-002 | nvcc --version | CUDA版本号 |
| TC-GPU-003 | nvidia-smi命令 | GPU型号、计算能力 |
| TC-SYS-001 | os.cpu_count() | 逻辑核心数 |
| TC-SYS-002 | psutil.virtual_memory() | 内存GB数 |
| TC-SYS-003 | psutil.disk_usage() | 可用空间GB数 |
| TC-SYS-004 | lsblk/Get-PhysicalDisk | 磁盘类型 |

### 5.2 需手动验证的项目

| 测试ID | 原因 | 手动步骤 |
|--------|------|---------|
| TC-DEVICE-001 | 需要人工决策 | 评估选项、填写文档 |

### 5.3 自动化集成建议

1. **开发环境脚本**：将检测脚本加入项目 `scripts/` 目录
2. **CI/CD 集成**：在训练任务前自动运行检测
3. **定期检测**：环境变更时重新运行验证
4. **结果归档**：将检测结果保存到项目文档

---

## 6. 风险点与注意事项

### 6.1 风险识别

| 风险ID | 风险描述 | 可能性 | 影响 | 缓解措施 |
|--------|---------|--------|------|---------|
| R-001 | GPU显存刚好满足8GB，实际训练可能OOM | 中 | 高 | 预留20%显存余量，准备降低batch size方案 |
| R-002 | CUDA版本与PyTorch版本不匹配 | 中 | 高 | 安装前核对官方兼容性矩阵 |
| R-003 | 多GPU环境检测到错误GPU | 低 | 中 | 明确指定训练用GPU ID |
| R-004 | 磁盘空间在训练过程中耗尽 | 中 | 高 | 预留足够余量，设置空间告警 |
| R-005 | 虚拟环境隔离导致GPU不可见 | 低 | 高 | 验证虚拟环境中的CUDA可用性 |
| R-006 | 系统资源波动（其他进程占用） | 中 | 中 | 在无其他负载时检测，记录基准值 |

### 6.2 注意事项

1. **检测时机**
   - 在系统空闲状态下进行检测
   - 关闭其他占用GPU/内存的程序
   - 多次检测取稳定值

2. **文档维护**
   - 硬件变更后需重新验证
   - 检测结果需记录日期
   - 保留历史检测记录用于对比

3. **边界情况处理**
   - 刚好满足最低要求时，记录为「临界满足」
   - 对于警告项，需在后续阶段监控实际影响
   - 对于多GPU系统，记录每个GPU信息

4. **跨平台兼容**
   - 脚本需支持 Windows/Linux
   - 注意不同平台命令差异
   - 统一输出格式便于解析

### 6.3 后续监控建议

硬件验证通过后，建议在后续训练阶段添加：

1. **显存使用监控**：`nvidia-smi dmon -s u`
2. **内存使用日志**：记录训练时峰值内存
3. **磁盘空间检查**：训练前自动检查剩余空间
4. **性能基准测试**：记录首次训练耗时作为基准

---

## 附录：硬件配置文档模板

```markdown
# 硬件配置记录

> 检测日期：YYYY-MM-DD
> 检测人：姓名

## GPU 配置

| 项目 | 值 |
|-----|-----|
| GPU型号 | |
| 显存大小 | GB |
| CUDA版本 | |
| 计算能力 | |

## 系统配置

| 项目 | 值 |
|-----|-----|
| CPU | 核 |
| 内存 | GB |
| 磁盘 | GB (SSD/HDD) |

## 训练设备选择

**最终方案：** （本地工作站 / 云服务器 / GPU集群）

**选择理由：**
-

## 环境满足情况

- [ ] GPU显存 ≥ 8GB
- [ ] CUDA版本兼容
- [ ] CPU ≥ 8核
- [ ] 内存 ≥ 16GB
- [ ] 磁盘 ≥ 200GB
```