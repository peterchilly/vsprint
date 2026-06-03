# Python 环境配置 — 验证测试设计

> 来源任务：docs/superpowers/tasks/01_环境搭建/02_Python环境配置.md
> 生成日期：2026-06-02

---

## 1. 验证目标概述

本验证方案针对「Python 环境配置」任务的验收标准，设计具体的验证方法和测试用例：

| 验收标准 | 验证目标 | 验证方法 |
|---------|---------|---------|
| 虚拟环境可正常激活和退出 | 验证虚拟环境创建成功且隔离有效 | 命令行验证 + 隔离测试 |
| requirements.txt 文件存在 | 确认依赖文件存在且格式正确 | 文件检查 + 内容审计 |

**任务清单验证覆盖：**

| 任务项 | 对应测试用例 | 验证方式 |
|-------|-------------|---------|
| 安装 Python 3.9–3.11 | TC-PY-001, TC-PY-002 | 自动化 |
| 验证 Python 安装 | TC-PY-003 | 自动化 |
| 选择 conda 或 venv | TC-VENV-001 | 手动确认 |
| 创建虚拟环境并激活 | TC-VENV-002, TC-VENV-003 | 自动化 |
| 验证环境隔离正常 | TC-VENV-004, TC-VENV-005 | 自动化 |
| 生成依赖文件 | TC-DEP-001, TC-DEP-002 | 自动化 |
| 记录精确版本号 | TC-DEP-003 | 自动化 |

**验证原则：**
- 版本号必须精确锁定（==x.x.x 格式）
- 环境隔离必须可验证
- 依赖文件必须可重装成功

---

## 2. 测试用例设计

### 2.1 Python 安装测试用例

#### TC-PY-001: Python 版本范围验证
| 属性 | 内容 |
|-----|------|
| **测试项** | Python 版本在允许范围内 |
| **前置条件** | Python 已安装 |
| **测试步骤** | 1. 执行 `python --version` 或 `python3 --version`<br>2. 解析版本号（主.次.补丁）<br>3. 验证主版本为 3，次版本在 9-11 范围内 |
| **预期结果** | 版本格式为 3.9.x / 3.10.x / 3.11.x |
| **通过标准** | 版本 ∈ [3.9.0, 3.12.0)，推荐 3.10.x |
| **自动化** | 可自动化 |

#### TC-PY-002: Python 推荐版本检查
| 属性 | 内容 |
|-----|------|
| **测试项** | Python 版本为推荐版本 3.10 |
| **前置条件** | Python 已安装 |
| **测试步骤** | 1. 获取 Python 版本<br>2. 判断是否为 3.10.x |
| **预期结果** | 版本以 3.10 开头 |
| **通过标准** | 为 3.10.x 则标记「推荐版本」，否则标记「兼容但非推荐」 |
| **自动化** | 可自动化 |

#### TC-PY-003: Python 可执行文件位置验证
| 属性 | 内容 |
|-----|------|
| **测试项** | Python 可执行文件路径正确 |
| **前置条件** | Python 已安装 |
| **测试步骤** | 1. Linux/Mac: 执行 `which python` 或 `which python3`<br>2. Windows: 执行 `where python` 或 `where python3`<br>3. 记录路径位置 |
| **预期结果** | 返回有效路径 |
| **通过标准** | 路径存在且可执行，记录到配置文档 |
| **自动化** | 可自动化 |

#### TC-PY-004: pip 版本验证
| 属性 | 内容 |
|-----|------|
| **测试项** | pip 已安装且可用 |
| **前置条件** | Python 已安装 |
| **测试步骤** | 1. 执行 `python -m pip --version`<br>2. 解析 pip 版本号 |
| **预期结果** | pip 版本 ≥ 21.0 |
| **通过标准** | pip 可用且版本 ≥ 21.0 |
| **自动化** | 可自动化 |

### 2.2 虚拟环境创建测试用例

#### TC-VENV-001: 虚拟环境工具选择确认
| 属性 | 内容 |
|-----|------|
| **测试项** | 明确选择 conda 或 venv |
| **前置条件** | 无 |
| **测试步骤** | 1. 检查 conda 是否可用：`conda --version`<br>2. 检查 venv 是否可用：`python -m venv --help`<br>3. 确认选择并记录到配置文档 |
| **预期结果** | 至少一种工具可用，并有明确选择 |
| **通过标准** | 文档中记录了选择的工具及理由 |
| **自动化** | 半自动化（检测可用性可自动化，选择需手动） |

#### TC-VENV-002: Conda 虚拟环境创建验证
| 属性 | 内容 |
|-----|------|
| **测试项** | Conda 环境创建成功 |
| **前置条件** | Conda 已安装 |
| **测试步骤** | 1. 执行 `conda create -n vsprint python=3.10 -y`<br>2. 检查返回码是否为 0<br>3. 执行 `conda env list` 确认环境存在 |
| **预期结果** | 环境列表中出现 vsprint 环境 |
| **通过标准** | 环境创建成功，路径正确 |
| **自动化** | 可自动化 |

#### TC-VENV-003: venv 虚拟环境创建验证
| 属性 | 内容 |
|-----|------|
| **测试项** | venv 环境创建成功 |
| **前置条件** | Python 已安装 |
| **测试步骤** | 1. 执行 `python -m venv .venv`<br>2. 检查 `.venv` 目录是否存在<br>3. 检查目录结构（bin/Scripts, lib 等） |
| **预期结果** | .venv 目录存在且包含标准结构 |
| **通过标准** | 目录存在，包含 python/pip 可执行文件 |
| **自动化** | 可自动化 |

#### TC-VENV-004: 虚拟环境激活验证
| 属性 | 内容 |
|-----|------|
| **测试项** | 虚拟环境可正常激活 |
| **前置条件** | 虚拟环境已创建 |
| **测试步骤** | **Conda:**<br>1. 执行 `conda activate vsprint`<br>2. 检查命令提示符变化<br>3. 执行 `which python` 确认路径<br><br>**venv (Linux/Mac):**<br>1. 执行 `source .venv/bin/activate`<br>2. 检查 VIRTUAL_ENV 环境变量<br><br>**venv (Windows):**<br>1. 执行 `.venv\Scripts\activate`<br>2. 检查命令提示符变化 |
| **预期结果** | 环境变量指向虚拟环境路径 |
| **通过标准** | 激活后 Python 路径指向虚拟环境内 |
| **自动化** | 可自动化（通过脚本检测环境变量） |

#### TC-VENV-005: 虚拟环境退出验证
| 属性 | 内容 |
|-----|------|
| **测试项** | 虚拟环境可正常退出 |
| **前置条件** | 虚拟环境已激活 |
| **测试步骤** | **Conda:**<br>1. 执行 `conda deactivate`<br>2. 检查命令提示符恢复<br>3. 执行 `which python` 确认路径变化<br><br>**venv:**<br>1. 执行 `deactivate`<br>2. 检查 VIRTUAL_ENV 环境变量是否清除 |
| **预期结果** | 退出后 Python 路径恢复为系统 Python |
| **通过标准** | 退出成功，环境变量正确恢复 |
| **自动化** | 可自动化 |

#### TC-VENV-006: 环境隔离验证 — 包隔离测试
| 属性 | 内容 |
|-----|------|
| **测试项** | 虚拟环境包隔离正常 |
| **前置条件** | 虚拟环境已激活 |
| **测试步骤** | 1. 在虚拟环境中安装测试包：`pip install pytest-testmon`<br>2. 列出虚拟环境包：`pip list`<br>3. 退出虚拟环境<br>4. 列出系统包：`pip list`<br>5. 确认测试包不在系统环境中 |
| **预期结果** | 测试包仅存在于虚拟环境中 |
| **通过标准** | 系统环境中无测试包，虚拟环境中有 |
| **自动化** | 可自动化 |

#### TC-VENV-007: 环境隔离验证 — Python 版本隔离测试
| 属性 | 内容 |
|-----|------|
| **测试项** | 虚拟环境 Python 版本独立 |
| **前置条件** | 虚拟环境已创建（指定 Python 版本） |
| **测试步骤** | 1. 记录系统 Python 版本<br>2. 激活虚拟环境<br>3. 记录虚拟环境 Python 版本<br>4. 对比两者 |
| **预期结果** | 虚拟环境 Python 版本与创建时指定版本一致 |
| **通过标准** | 版本匹配创建时指定版本（如创建时指定 3.10，则应为 3.10.x） |
| **自动化** | 可自动化 |

### 2.3 依赖锁定测试用例

#### TC-DEP-001: requirements.txt 文件存在验证
| 属性 | 内容 |
|-----|------|
| **测试项** | requirements.txt 文件存在 |
| **前置条件** | 项目目录已创建 |
| **测试步骤** | 1. 检查项目根目录是否存在 `requirements.txt`<br>2. 检查文件是否为空 |
| **预期结果** | 文件存在且非空 |
| **通过标准** | 文件存在，大小 > 0 字节 |
| **自动化** | 可自动化 |

#### TC-DEP-002: requirements.txt 格式验证
| 属性 | 内容 |
|-----|------|
| **测试项** | 依赖文件格式正确 |
| **前置条件** | requirements.txt 存在 |
| **测试步骤** | 1. 逐行读取 requirements.txt<br>2. 验证每行格式：<br>   - 标准格式：`package==x.x.x`<br>   - 允许：`package>=x.x.x,<y.y.y`<br>   - 允许：`package @ url`<br>   - 忽略：空行和 `#` 注释行<br>3. 统计有效依赖行数 |
| **预期结果** | 所有依赖行格式正确 |
| **通过标准** | 格式正确率 100%，无解析错误 |
| **自动化** | 可自动化 |

#### TC-DEP-003: 依赖版本精确锁定验证
| 属性 | 内容 |
|-----|------|
| **测试项** | 所有依赖使用精确版本号 |
| **前置条件** | requirements.txt 存在 |
| **测试步骤** | 1. 解析 requirements.txt<br>2. 统计使用 `==` 锁定的依赖数量<br>3. 统计使用范围版本（`>=`, `>`, `<`, `<=`）的依赖数量 |
| **预期结果** | 所有核心依赖使用 `==` 锁定 |
| **通过标准** | 核心依赖（torch, numpy 等）100% 精确锁定，允许工具类依赖使用范围版本 |
| **自动化** | 可自动化 |

#### TC-DEP-004: requirements.txt 可重装验证
| 属性 | 内容 |
|-----|------|
| **测试项** | requirements.txt 可成功安装 |
| **前置条件** | 新建虚拟环境 |
| **测试步骤** | 1. 创建新的虚拟环境<br>2. 激活环境<br>3. 执行 `pip install -r requirements.txt`<br>4. 检查返回码 |
| **预期结果** | 安装成功，返回码为 0 |
| **通过标准** | 所有依赖安装成功，无错误 |
| **自动化** | 可自动化 |

#### TC-DEP-005: environment.yml 文件验证（Conda）
| 属性 | 内容 |
|-----|------|
| **测试项** | Conda environment.yml 存在且有效 |
| **前置条件** | 选择 Conda 作为环境管理工具 |
| **测试步骤** | 1. 检查 `environment.yml` 文件存在<br>2. 验证 YAML 格式正确<br>3. 检查必要字段（name, dependencies）<br>4. 执行 `conda env create -f environment.yml --dry-run` |
| **预期结果** | 文件格式正确，可被 conda 解析 |
| **通过标准** | 格式正确，包含 name 和 dependencies 字段 |
| **自动化** | 可自动化 |

#### TC-DEP-006: 核心依赖完整性验证
| 属性 | 内容 |
|-----|------|
| **测试项** | 深度学习核心依赖已包含 |
| **前置条件** | requirements.txt 存在 |
| **测试步骤** | 1. 读取 requirements.txt<br>2. 检查是否包含以下核心包：<br>   - torch / pytorch<br>   - numpy<br>   - scipy<br>   - pandas<br>   - scikit-learn<br>   - matplotlib<br>   - tqdm |
| **预期结果** | 核心 ML 包已列出 |
| **通过标准** | 至少包含 torch, numpy 两个核心依赖 |
| **自动化** | 可自动化 |

#### TC-DEP-007: 依赖版本兼容性验证
| 属性 | 内容 |
|-----|------|
| **测试项** | 依赖版本与 Python 版本兼容 |
| **前置条件** | requirements.txt 存在 |
| **测试步骤** | 1. 获取 Python 版本<br>2. 检查每个依赖的版本是否支持当前 Python 版本<br>3. 特别检查 PyTorch 与 CUDA 版本匹配 |
| **预期结果** | 所有依赖版本与 Python 兼容 |
| **通过标准** | 无已知不兼容组合（如 Python 3.11 + 旧版 PyTorch） |
| **自动化** | 半自动化（需查询官方兼容性矩阵） |

---

## 3. 验证方法与步骤

### 3.1 自动化验证脚本（推荐）

创建环境检测脚本 `scripts/verify_python_env.py`，执行全部自动化测试用例：

```python
#!/usr/bin/env python3
"""
Python 环境配置验证脚本
运行方式: python scripts/verify_python_env.py
"""

import subprocess
import sys
import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def run_command(cmd: List[str], check: bool = False) -> Tuple[int, str, str]:
    """执行命令并返回结果"""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=check
        )
        return result.returncode, result.stdout, result.stderr
    except FileNotFoundError:
        return -1, "", f"命令不存在: {cmd[0]}"


def check_python_version() -> Dict:
    """TC-PY-001, TC-PY-002: Python 版本验证"""
    test_result = {
        'test': 'TC-PY-001/002',
        'name': 'Python 版本检查',
        'passed': False,
        'value': None,
        'message': ''
    }

    # 尝试获取 Python 版本
    for cmd in [['python', '--version'], ['python3', '--version']]:
        code, stdout, stderr = run_command(cmd)
        if code == 0:
            # 解析版本号
            output = stdout + stderr
            match = re.search(r'Python (\d+)\.(\d+)\.(\d+)', output)
            if match:
                major, minor, patch = int(match.group(1)), int(match.group(2)), int(match.group(3))
                version_str = f"{major}.{minor}.{patch}"
                test_result['value'] = version_str

                # TC-PY-001: 版本范围检查
                if major == 3 and 9 <= minor <= 11:
                    test_result['passed'] = True
                    # TC-PY-002: 推荐版本检查
                    if minor == 10:
                        test_result['message'] = f'Python {version_str} - 推荐版本'
                    else:
                        test_result['message'] = f'Python {version_str} - 兼容但非推荐（推荐 3.10）'
                else:
                    test_result['message'] = f'Python {version_str} - 版本不在允许范围 [3.9-3.11]'
                break
    else:
        test_result['message'] = '未找到 Python 安装'

    return test_result


def check_python_path() -> Dict:
    """TC-PY-003: Python 可执行文件路径"""
    test_result = {
        'test': 'TC-PY-003',
        'name': 'Python 可执行文件路径',
        'passed': False,
        'value': None,
        'message': ''
    }

    # 根据平台选择命令
    if sys.platform == 'win32':
        cmd = ['where', 'python']
    else:
        cmd = ['which', 'python3']

    code, stdout, stderr = run_command(cmd)
    if code == 0 and stdout.strip():
        path = stdout.strip().split('\n')[0]
        test_result['value'] = path
        test_result['passed'] = True
        test_result['message'] = f'Python 路径: {path}'
    else:
        test_result['message'] = '未找到 Python 可执行文件路径'

    return test_result


def check_pip_version() -> Dict:
    """TC-PY-004: pip 版本验证"""
    test_result = {
        'test': 'TC-PY-004',
        'name': 'pip 版本检查',
        'passed': False,
        'value': None,
        'message': ''
    }

    code, stdout, stderr = run_command(['python', '-m', 'pip', '--version'])
    if code == 0:
        output = stdout + stderr
        match = re.search(r'pip (\d+)\.(\d+)\.(\d+)', output)
        if match:
            version = f"{match.group(1)}.{match.group(2)}.{match.group(3)}"
            major = int(match.group(1))
            test_result['value'] = version
            test_result['passed'] = major >= 21
            test_result['message'] = f'pip {version}' + ('' if major >= 21 else ' - 建议升级到 21.0+')
    else:
        test_result['message'] = 'pip 未安装或不可用'

    return test_result


def check_conda_available() -> Dict:
    """TC-VENV-001: Conda 可用性检查"""
    test_result = {
        'test': 'TC-VENV-001-conda',
        'name': 'Conda 可用性',
        'passed': False,
        'value': None,
        'message': ''
    }

    code, stdout, stderr = run_command(['conda', '--version'])
    if code == 0:
        version = stdout.strip()
        test_result['value'] = version
        test_result['passed'] = True
        test_result['message'] = f'Conda 可用: {version}'
    else:
        test_result['message'] = 'Conda 未安装'

    return test_result


def check_venv_available() -> Dict:
    """TC-VENV-001: venv 可用性检查"""
    test_result = {
        'test': 'TC-VENV-001-venv',
        'name': 'venv 可用性',
        'passed': False,
        'value': None,
        'message': ''
    }

    code, stdout, stderr = run_command(['python', '-m', 'venv', '--help'])
    if code == 0:
        test_result['passed'] = True
        test_result['message'] = 'venv 可用'
    else:
        test_result['message'] = 'venv 不可用'

    return test_result


def check_venv_exists(venv_path: str = '.venv') -> Dict:
    """TC-VENV-003: venv 虚拟环境存在检查"""
    test_result = {
        'test': 'TC-VENV-003',
        'name': '虚拟环境存在检查',
        'passed': False,
        'value': None,
        'message': ''
    }

    venv_dir = Path(venv_path)
    if venv_dir.exists():
        test_result['value'] = str(venv_dir.absolute())

        # 检查目录结构
        if sys.platform == 'win32':
            expected_files = ['Scripts/python.exe', 'Scripts/pip.exe']
        else:
            expected_files = ['bin/python', 'bin/pip']

        all_exist = all((venv_dir / f).exists() for f in expected_files)
        if all_exist:
            test_result['passed'] = True
            test_result['message'] = f'虚拟环境存在且结构正确: {venv_path}'
        else:
            test_result['message'] = f'虚拟环境目录存在但结构不完整: {venv_path}'
    else:
        test_result['message'] = f'虚拟环境不存在: {venv_path}'

    return test_result


def check_venv_active() -> Dict:
    """TC-VENV-004: 虚拟环境激活状态检查"""
    test_result = {
        'test': 'TC-VENV-004',
        'name': '虚拟环境激活检查',
        'passed': False,
        'value': None,
        'message': ''
    }

    # 检查 VIRTUAL_ENV 环境变量
    venv_env = os.environ.get('VIRTUAL_ENV')
    conda_default_env = os.environ.get('CONDA_DEFAULT_ENV')

    if venv_env:
        test_result['value'] = venv_env
        test_result['passed'] = True
        test_result['message'] = f'venv 环境已激活: {venv_env}'
    elif conda_default_env and conda_default_env != 'base':
        test_result['value'] = conda_default_env
        test_result['passed'] = True
        test_result['message'] = f'Conda 环境已激活: {conda_default_env}'
    else:
        test_result['message'] = '未检测到激活的虚拟环境'

    return test_result


def check_requirements_exists() -> Dict:
    """TC-DEP-001: requirements.txt 存在检查"""
    test_result = {
        'test': 'TC-DEP-001',
        'name': 'requirements.txt 存在检查',
        'passed': False,
        'value': None,
        'message': ''
    }

    req_file = Path('requirements.txt')
    if req_file.exists():
        size = req_file.stat().st_size
        test_result['value'] = f'{size} bytes'
        if size > 0:
            test_result['passed'] = True
            test_result['message'] = f'requirements.txt 存在，大小: {size} bytes'
        else:
            test_result['message'] = 'requirements.txt 存在但为空'
    else:
        test_result['message'] = 'requirements.txt 不存在'

    return test_result


def check_requirements_format() -> Dict:
    """TC-DEP-002: requirements.txt 格式验证"""
    test_result = {
        'test': 'TC-DEP-002',
        'name': 'requirements.txt 格式验证',
        'passed': False,
        'value': None,
        'message': ''
    }

    req_file = Path('requirements.txt')
    if not req_file.exists():
        test_result['message'] = 'requirements.txt 不存在'
        return test_result

    # 格式正则
    valid_patterns = [
        r'^[a-zA-Z0-9_-]+==[\d.]+$',  # package==x.x.x
        r'^[a-zA-Z0-9_-+]+[<>=!]+[\d.,]+',  # package>=x.x.x,<y.y.y
        r'^[a-zA-Z0-9_-]+\s*@',  # package @ url
        r'^-r\s+',  # -r another.txt
        r'^#',  # 注释
        r'^$',  # 空行
    ]

    valid_count = 0
    invalid_lines = []
    total_lines = 0

    with open(req_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            is_valid = any(re.match(p, line) for p in valid_patterns)
            if is_valid:
                valid_count += 1
            else:
                invalid_lines.append((i, line))

    test_result['value'] = f'{valid_count}/{total_lines} 有效'
    if valid_count == total_lines:
        test_result['passed'] = True
        test_result['message'] = f'格式正确，共 {total_lines} 个依赖'
    else:
        test_result['message'] = f'格式错误：第 {[l[0] for l in invalid_lines]} 行'

    return test_result


def check_requirements_pinned() -> Dict:
    """TC-DEP-003: 依赖版本精确锁定检查"""
    test_result = {
        'test': 'TC-DEP-003',
        'name': '依赖版本锁定检查',
        'passed': False,
        'value': None,
        'message': ''
    }

    req_file = Path('requirements.txt')
    if not req_file.exists():
        test_result['message'] = 'requirements.txt 不存在'
        return test_result

    pinned_count = 0
    unpinned_count = 0
    unpinned_packages = []

    with open(req_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '==' in line:
                pinned_count += 1
            elif any(op in line for op in ['>=', '<=', '>', '<', '~=']):
                unpinned_count += 1
                unpinned_packages.append(line.split('>=')[0].split('>')[0].split('<')[0].strip())

    total = pinned_count + unpinned_count
    if total == 0:
        test_result['message'] = '无依赖项'
        return test_result

    pinned_ratio = pinned_count / total * 100
    test_result['value'] = f'{pinned_count}/{total} 精确锁定 ({pinned_ratio:.1f}%)'

    if pinned_ratio == 100:
        test_result['passed'] = True
        test_result['message'] = '所有依赖已精确锁定'
    elif pinned_ratio >= 80:
        test_result['passed'] = True
        test_result['message'] = f'大部分依赖已锁定，未锁定: {unpinned_packages}'
    else:
        test_result['message'] = f'锁定比例过低，未锁定: {unpinned_packages}'

    return test_result


def check_core_dependencies() -> Dict:
    """TC-DEP-006: 核心依赖完整性检查"""
    test_result = {
        'test': 'TC-DEP-006',
        'name': '核心依赖检查',
        'passed': False,
        'value': None,
        'message': ''
    }

    req_file = Path('requirements.txt')
    if not req_file.exists():
        test_result['message'] = 'requirements.txt 不存在'
        return test_result

    core_packages = ['torch', 'numpy']
    optional_packages = ['scipy', 'pandas', 'scikit-learn', 'matplotlib', 'tqdm']

    content = req_file.read_text(encoding='utf-8').lower()

    found_core = [pkg for pkg in core_packages if pkg in content]
    found_optional = [pkg for pkg in optional_packages if pkg in content]

    test_result['value'] = f'核心: {found_core}, 可选: {found_optional}'

    if len(found_core) == len(core_packages):
        test_result['passed'] = True
        test_result['message'] = f'核心依赖完整，可选依赖: {found_optional}'
    else:
        missing = [pkg for pkg in core_packages if pkg not in content]
        test_result['message'] = f'缺少核心依赖: {missing}'

    return test_result


def main():
    print("=" * 60)
    print("Python 环境配置验证 - ERes2NetV2 训练环境")
    print("=" * 60)

    tests = [
        check_python_version(),
        check_python_path(),
        check_pip_version(),
        check_conda_available(),
        check_venv_available(),
        check_venv_exists(),
        check_venv_active(),
        check_requirements_exists(),
        check_requirements_format(),
        check_requirements_pinned(),
        check_core_dependencies(),
    ]

    passed = sum(1 for t in tests if t['passed'])
    total = len(tests)

    for test in tests:
        status = "✓ PASS" if test['passed'] else "✗ FAIL"
        print(f"\n[{status}] {test['name']}")
        if test['value']:
            print(f"  值: {test['value']}")
        print(f"  说明: {test['message']}")

    print("\n" + "=" * 60)
    print(f"验证结果: {passed}/{total} 项通过")
    print("=" * 60)

    # 保存结果到 JSON
    result_file = Path('docs/superpowers/audit/01_环境搭建/python_env_check_result.json')
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({'tests': tests, 'summary': {'passed': passed, 'total': total}},
                  f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到: {result_file}")

    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
```

### 3.2 手动验证步骤

对于无法自动化的测试项，按以下步骤手动验证：

#### TC-VENV-001 工具选择确认

1. **检查可用工具**
   ```bash
   # 检查 Conda
   conda --version

   # 检查 venv
   python -m venv --help
   ```

2. **记录选择决策**
   - 在项目文档中记录选择的工具
   - 记录选择理由（团队熟悉度、功能需求等）

#### TC-DEP-004 重装验证（需新环境）

1. **创建测试环境**
   ```bash
   # Conda 方式
   conda create -n test_env python=3.10 -y
   conda activate test_env

   # 或 venv 方式
   python -m venv test_env
   source test_env/bin/activate  # Linux/Mac
   # test_env\Scripts\activate    # Windows
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **验证安装结果**
   ```bash
   pip list
   # 检查所有依赖是否正确安装
   ```

4. **清理测试环境**
   ```bash
   # Conda
   conda deactivate
   conda env remove -n test_env

   # venv
   deactivate
   rm -rf test_env
   ```

#### TC-DEP-007 版本兼容性验证

1. **查询 PyTorch 兼容性**
   - 访问 https://pytorch.org/get-started/previous-versions/
   - 确认 PyTorch 版本与 CUDA 版本匹配

2. **查询 Python 兼容性**
   - 检查 requirements.txt 中各包的 Python 版本要求
   - 特别注意 numba, scipy 等对 Python 版本敏感的包

### 3.3 验证执行顺序

```
┌─────────────────────────────────────────────────────────────┐
│ 阶段 1: Python 安装验证                                      │
│ ├─ TC-PY-001: Python 版本范围检查                           │
│ ├─ TC-PY-002: 推荐版本确认                                  │
│ ├─ TC-PY-003: 可执行文件路径                                │
│ └─ TC-PY-004: pip 版本检查                                  │
├─────────────────────────────────────────────────────────────┤
│ 阶段 2: 虚拟环境验证                                         │
│ ├─ TC-VENV-001: 工具选择确认（手动）                         │
│ ├─ TC-VENV-002/003: 环境创建检查                            │
│ ├─ TC-VENV-004: 环境激活检查                                │
│ ├─ TC-VENV-005: 环境退出检查                                 │
│ └─ TC-VENV-006/007: 隔离性测试                               │
├─────────────────────────────────────────────────────────────┤
│ 阶段 3: 依赖锁定验证                                         │
│ ├─ TC-DEP-001: requirements.txt 存在                        │
│ ├─ TC-DEP-002: 格式验证                                      │
│ ├─ TC-DEP-003: 版本锁定检查                                 │
│ ├─ TC-DEP-004: 重装验证（需新环境）                          │
│ ├─ TC-DEP-005: environment.yml 验证（Conda）               │
│ ├─ TC-DEP-006: 核心依赖完整性                               │
│ └─ TC-DEP-007: 版本兼容性（手动）                            │
├─────────────────────────────────────────────────────────────┤
│ 阶段 4: 最终验收                                             │
│ └─ 对照验收标准逐项确认                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. 通过标准

### 4.1 各测试用例通过标准汇总

| 测试ID | 测试项 | 通过条件 | 失败影响 |
|--------|-------|---------|---------|
| TC-PY-001 | Python 版本范围 | 3.9.x / 3.10.x / 3.11.x | 阻塞：版本不兼容 |
| TC-PY-002 | Python 推荐版本 | 3.10.x | 警告：非最优选择 |
| TC-PY-003 | Python 路径 | 路径存在且可执行 | 阻塞：无法执行 |
| TC-PY-004 | pip 版本 | ≥ 21.0 | 警告：可能缺少新特性 |
| TC-VENV-001 | 工具选择 | 有明确选择并记录 | 阻塞：无明确方案 |
| TC-VENV-002 | Conda 环境创建 | 环境创建成功 | 阻塞：无法创建环境 |
| TC-VENV-003 | venv 环境创建 | 目录结构正确 | 阻塞：无法创建环境 |
| TC-VENV-004 | 环境激活 | 激活后路径正确 | 阻塞：环境不可用 |
| TC-VENV-005 | 环境退出 | 退出后恢复正常 | 阻塞：环境异常 |
| TC-VENV-006 | 包隔离 | 包仅存在于虚拟环境 | 阻塞：隔离失败 |
| TC-VENV-007 | 版本隔离 | Python 版本独立 | 阻塞：隔离失败 |
| TC-DEP-001 | requirements.txt 存在 | 文件存在且非空 | 阻塞：无依赖文件 |
| TC-DEP-002 | 格式验证 | 格式正确率 100% | 阻塞：解析错误 |
| TC-DEP-003 | 版本锁定 | 锁定比例 ≥ 80% | 警告：可复现性差 |
| TC-DEP-004 | 重装验证 | 安装成功无错误 | 阻塞：依赖无法安装 |
| TC-DEP-005 | environment.yml | 格式正确可解析 | 条件：使用 Conda |
| TC-DEP-006 | 核心依赖 | 包含 torch, numpy | 阻塞：缺少核心依赖 |
| TC-DEP-007 | 版本兼容性 | 无已知不兼容组合 | 阻塞：运行时错误 |

### 4.2 总体验收判定规则

| 结果 | 条件 |
|-----|------|
| **通过** | 所有阻塞项通过，警告项≤2 |
| **有条件通过** | 所有阻塞项通过，警告项>2 |
| **不通过** | 任一阻塞项失败 |

### 4.3 验收检查清单

**Python 安装验收：**
- [ ] Python 版本在 3.9-3.11 范围内
- [ ] Python 可执行文件路径正确
- [ ] pip 版本 ≥ 21.0

**虚拟环境验收：**
- [ ] 明确选择 conda 或 venv 并记录
- [ ] 虚拟环境创建成功
- [ ] 可正常激活
- [ ] 可正常退出
- [ ] 环境隔离验证通过

**依赖锁定验收：**
- [ ] requirements.txt 文件存在且非空
- [ ] 格式正确无误
- [ ] 核心依赖（torch, numpy）已包含
- [ ] 版本精确锁定比例 ≥ 80%
- [ ] 在新环境中重装成功

---

## 5. 自动化验证建议

### 5.1 可完全自动化的测试

| 测试ID | 自动化方式 | 输出 |
|--------|-----------|------|
| TC-PY-001 | `python --version` 解析 | 版本号 |
| TC-PY-002 | 版本号判断 | 是否推荐版本 |
| TC-PY-003 | `which`/`where` 命令 | 路径 |
| TC-PY-004 | `python -m pip --version` | pip 版本 |
| TC-VENV-002 | `conda env list` 检查 | 环境列表 |
| TC-VENV-003 | 目录结构检查 | 存在/不存在 |
| TC-VENV-004 | `VIRTUAL_ENV` 环境变量 | 激活状态 |
| TC-VENV-006 | pip list 对比 | 包列表差异 |
| TC-DEP-001 | 文件存在检查 | 存在/不存在 |
| TC-DEP-002 | 正则表达式解析 | 格式正确率 |
| TC-DEP-003 | 行解析统计 | 锁定比例 |
| TC-DEP-006 | 文本匹配 | 依赖列表 |

### 5.2 需手动验证的项目

| 测试ID | 原因 | 手动步骤 |
|--------|------|---------|
| TC-VENV-001 | 需人工决策 | 评估工具、记录选择理由 |
| TC-VENV-005 | 需交互式操作 | 手动激活/退出验证用户感知 |
| TC-DEP-004 | 需新环境 | 在隔离环境中重装测试 |
| TC-DEP-007 | 需查询外部资源 | 查官方兼容性矩阵 |

### 5.3 自动化集成建议

1. **开发环境脚本**
   - 将验证脚本放入 `scripts/verify_python_env.py`
   - 添加到项目 Makefile 或 tasks.py

2. **CI/CD 集成**
   ```yaml
   # .github/workflows/verify-env.yml
   name: Verify Python Environment
   on: [push, pull_request]
   jobs:
     verify:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v3
         - uses: actions/setup-python@v4
           with:
             python-version: '3.10'
         - run: python scripts/verify_python_env.py
   ```

3. **预提交钩子**
   ```bash
   # .git/hooks/pre-commit
   #!/bin/bash
   python scripts/verify_python_env.py || exit 1
   ```

4. **环境配置文档化**
   - 每次验证生成 JSON 结果文件
   - 保存验证记录到项目文档

---

## 6. 风险点与注意事项

### 6.1 风险识别

| 风险ID | 风险描述 | 可能性 | 影响 | 缓解措施 |
|--------|---------|--------|------|---------|
| R-ENV-001 | Python 版本与 PyTorch 版本不兼容 | 中 | 高 | 安装前核对官方兼容性矩阵 |
| R-ENV-002 | 虚拟环境激活脚本执行权限问题 | 低 | 中 | 确保 activate 脚本有执行权限 |
| R-ENV-003 | Windows/Linux 路径差异导致脚本失效 | 中 | 中 | 脚本中处理平台差异 |
| R-ENV-004 | requirements.txt 依赖冲突 | 中 | 高 | 使用 `pip-compile` 锁定完整依赖树 |
| R-ENV-005 | Conda 和 pip 混用导致环境污染 | 中 | 高 | 统一使用一种工具或明确混用规则 |
| R-ENV-006 | 网络问题导致依赖安装失败 | 中 | 中 | 配置镜像源，离线包备份 |
| R-ENV-007 | CUDA 版本与 PyTorch 不匹配 | 高 | 高 | 安装 PyTorch 时指定 CUDA 版本 |
| R-ENV-008 | 系统环境变量干扰虚拟环境 | 低 | 中 | 验证环境变量隔离性 |

### 6.2 注意事项

#### 6.2.1 Python 版本选择

1. **推荐 3.10 的原因**
   - PyTorch 官方支持最完善
   - 科学计算库兼容性最佳
   - 避免新版本潜在的兼容问题

2. **避免 3.12+ 的原因**
   - 部分 C 扩展可能不兼容
   - PyTorch 某些版本未测试

#### 6.2.2 虚拟环境管理

1. **Conda vs venv 选择建议**
   | 场景 | 推荐工具 | 理由 |
   |-----|---------|-----|
   | 纯 Python 项目 | venv | 轻量、内置 |
   | 科学计算项目 | Conda | 管理非 Python 依赖 |
   | 团队协作 | Conda | 环境导出更完整 |
   | CI/CD 环境 | venv | 更快创建速度 |

2. **环境命名规范**
   - 使用项目名或功能名
   - 避免使用特殊字符
   - 建议格式：`项目名_用途`

#### 6.2.3 依赖管理

1. **版本锁定最佳实践**
   ```
   # 推荐：精确锁定
   torch==2.0.1
   numpy==1.24.3

   # 避免：范围版本（除非必要）
   torch>=2.0.0  # 不推荐
   ```

2. **依赖分类建议**
   ```
   # requirements.txt 结构建议
   # 核心依赖（精确锁定）
   torch==2.0.1
   numpy==1.24.3

   # 开发依赖（可单独文件）
   pytest==7.3.1
   black==23.3.0
   ```

3. **使用 pip-tools 管理依赖**
   ```bash
   pip install pip-tools
   # 创建 requirements.in（声明直接依赖）
   # pip-compile 生成 requirements.txt（锁定完整依赖树）
   ```

#### 6.2.4 跨平台兼容

1. **路径处理**
   - 使用 `pathlib.Path` 代替字符串路径
   - 脚本中判断 `sys.platform`

2. **激活命令差异**
   | 平台 | venv 激活命令 |
   |-----|-------------|
   | Linux/Mac | `source .venv/bin/activate` |
   | Windows CMD | `.venv\Scripts\activate.bat` |
   | Windows PowerShell | `.venv\Scripts\Activate.ps1` |

### 6.3 后续监控建议

环境验证通过后，建议在后续阶段添加：

1. **依赖变更追踪**
   - 使用 `pip freeze > requirements-lock.txt` 记录完整依赖树
   - 定期检查依赖安全更新

2. **环境一致性检查**
   - 在 CI/CD 中验证依赖安装
   - 定期在新环境中重装测试

3. **Python 版本监控**
   - 关注 Python 安全更新
   - 关注 PyTorch 对新 Python 版本的支持

---

## 附录 A：环境配置文档模板

```markdown
# Python 环境配置记录

> 配置日期：YYYY-MM-DD
> 配置人：姓名

## Python 版本

| 项目 | 值 |
|-----|-----|
| Python 版本 | 3.10.x |
| Python 路径 | /path/to/python |
| pip 版本 | 23.x |

## 虚拟环境

| 项目 | 值 |
|-----|-----|
| 工具选择 | Conda / venv |
| 环境名称 | vsprint |
| 环境路径 | /path/to/env |

## 依赖文件

- [ ] requirements.txt 已创建
- [ ] environment.yml 已创建（Conda）
- [ ] 依赖已精确锁定

## 验证结果

- [ ] Python 版本检查通过
- [ ] 虚拟环境激活/退出正常
- [ ] 环境隔离验证通过
- [ ] 依赖安装验证通过

## 注意事项

- （记录任何特殊配置或问题）
```

---

## 附录 B：常见问题排查

### B.1 Python 版本问题

**问题：** 系统有多个 Python 版本，命令行版本不对
**解决：**
```bash
# 使用具体版本号
python3.10 --version

# 或创建别名
alias python=python3.10
```

### B.2 虚拟环境激活失败

**问题：** Windows PowerShell 执行策略限制
**解决：**
```powershell
# 临时允许脚本执行
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### B.3 依赖安装失败

**问题：** pip 安装超时
**解决：**
```bash
# 使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**问题：** PyTorch CUDA 版本不匹配
**解决：**
```bash
# 安装指定 CUDA 版本的 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### B.4 Conda 环境问题

**问题：** Conda 环境导出不完整
**解决：**
```bash
# 导出完整环境
conda env export > environment.yml

# 从 environment.yml 创建环境
conda env create -f environment.yml
```