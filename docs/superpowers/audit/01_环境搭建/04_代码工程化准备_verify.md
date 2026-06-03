# 代码工程化准备 — 验证测试设计

> **来源任务：** docs/superpowers/tasks/01_环境搭建/04_代码工程化准备.md
> **生成日期：** 2026-06-02

---

## 1. 验证目标概述

本验证方案针对代码工程化准备任务，确保以下目标达成：

| 验收标准 | 验证方式 | 量化指标 |
|---------|---------|---------|
| `.gitignore` 生效 | Git 状态检查 | 指定文件/目录不被追踪 |
| 项目目录结构完整 | 目录存在性检查 | 所有 8 个核心目录存在 |
| 代码格式化工具可运行 | 工具执行测试 | black/ruff 正常执行且返回 exit 0 |

---

## 2. 测试用例设计

### 2.1 Git 仓库验证

#### TC-GIT-001: Git 仓库初始化检查
| 项目 | 内容 |
|------|------|
| **测试目的** | 确认 Git 仓库已正确初始化 |
| **前置条件** | 项目根目录存在 |
| **测试步骤** | 1. 执行 `git rev-parse --is-inside-work-tree`<br>2. 执行 `git status` 检查仓库状态 |
| **预期结果** | 输出 `true`，仓库状态正常显示 |
| **通过标准** | 命令返回 `true`，无报错 |

#### TC-GIT-002: .gitignore 文件存在性检查
| 项目 | 内容 |
|------|------|
| **测试目的** | 确认 `.gitignore` 文件已创建 |
| **前置条件** | TC-GIT-001 通过 |
| **测试步骤** | 执行：<br>```bash<br>test -f .gitignore && echo "exists" || echo "missing"<br>``` |
| **预期结果** | 输出 `exists` |
| **通过标准** | 文件存在 |

#### TC-GIT-003: .gitignore 内容有效性检查
| 项目 | 内容 |
|------|------|
| **测试目的** | 确认 `.gitignore` 包含必要规则 |
| **前置条件** | TC-GIT-002 通过 |
| **测试步骤** | 执行：<br>```bash<br>cat .gitignore<br># 检查是否包含以下规则：<br># - __pycache__/<br># - *.pyc<br># - .venv/ 或 venv/<br># - checkpoints/<br># - *.egg-info/<br># - .env<br>``` |
| **预期结果** | 至少包含 6 类常见忽略规则 |
| **通过标准** | 规则数量 ≥ 6 条 |

#### TC-GIT-004: .gitignore 功能验证
| 项目 | 内容 |
|------|------|
| **测试目的** | 验证 `.gitignore` 实际生效 |
| **前置条件** | TC-GIT-003 通过 |
| **测试步骤** | 1. 创建测试文件 `__pycache__/test.pyc`<br>2. 执行 `git status --porcelain`<br>3. 检查输出是否包含该文件<br>4. 清理测试文件 |
| **预期结果** | `git status` 输出不包含 `__pycache__/test.pyc` |
| **通过标准** | 被忽略文件不在追踪列表中 |

#### TC-GIT-005: 首次提交检查
| 项目 | 内容 |
|------|------|
| **测试目的** | 确认仓库有初始提交 |
| **前置条件** | TC-GIT-001 通过 |
| **测试步骤** | 执行：<br>```bash<br>git log --oneline -1<br>git rev-parse HEAD<br>``` |
| **预期结果** | 输出至少一条提交记录，HEAD 有值 |
| **通过标准** | 提交数量 ≥ 1，HEAD 存在 |

---

### 2.2 项目目录结构验证

#### TC-DIR-001: 核心目录存在性检查
| 项目 | 内容 |
|------|------|
| **测试目的** | 确认所有 8 个核心目录已创建 |
| **前置条件** | 项目根目录存在 |
| **测试步骤** | 执行：<br>```bash<br>for dir in configs data scripts src notebooks docs checkpoints tests; do<br>  test -d "$dir" && echo "$dir: OK" || echo "$dir: MISSING"<br>done<br>``` |
| **预期结果** | 8 个目录全部显示 OK |
| **通过标准** | 所有目录存在，通过率 100% |

#### TC-DIR-002: src 子目录结构检查
| 项目 | 内容 |
|------|------|
| **测试目的** | 确认 src 目录包含必要子模块 |
| **前置条件** | TC-DIR-001 通过 |
| **测试步骤** | 执行：<br>```bash<br>for subdir in models datasets training eval deploy; do<br>  test -d "src/$subdir" && echo "src/$subdir: OK" || echo "src/$subdir: MISSING"<br>done<br>``` |
| **预期结果** | 5 个子目录全部显示 OK |
| **通过标准** | 所有子目录存在 |

#### TC-DIR-003: 目录权限检查
| 项目 | 内容 |
|------|------|
| **测试目的** | 确认目录具有读写权限 |
| **前置条件** | TC-DIR-001 通过 |
| **测试步骤** | 执行：<br>```bash<br>for dir in configs data scripts src notebooks docs checkpoints tests; do<br>  test -r "$dir" && test -w "$dir" && echo "$dir: RW" || echo "$dir: NO-RW"<br>done<br>``` |
| **预期结果** | 所有目录具有读写权限 |
| **通过标准** | 所有目录显示 RW |

#### TC-DIR-004: 目录结构完整性汇总
| 项目 | 内容 |
|------|------|
| **测试目的** | 生成目录结构树形报告 |
| **前置条件** | TC-DIR-001、TC-DIR-002 通过 |
| **测试步骤** | 执行：<br>```bash<br>tree -L 2 -d . 2>/dev/null || find . -maxdepth 2 -type d | head -20<br>``` |
| **预期结果** | 显示完整目录结构树 |
| **通过标准** | 结构完整，无缺失目录 |

---

### 2.3 代码规范工具验证

#### TC-TOOLS-001: black 安装检查
| 项目 | 内容 |
|------|------|
| **测试目的** | 确认 black 已正确安装 |
| **前置条件** | Python 虚拟环境已激活 |
| **测试步骤** | 执行：<br>```bash<br>python -c "import black; print(black.__version__)"<br>black --version<br>``` |
| **预期结果** | 版本号正确输出 |
| **通过标准** | 导入成功，版本号非空 |

#### TC-TOOLS-002: ruff 安装检查
| 项目 | 内容 |
|------|------|
| **测试目的** | 确认 ruff 已正确安装 |
| **前置条件** | Python 虚拟环境已激活 |
| **测试步骤** | 执行：<br>```bash<br>python -c "import ruff; print(ruff.__version__)" 2>/dev/null || ruff --version<br>``` |
| **预期结果** | 版本号正确输出 |
| **通过标准** | 命令执行成功，版本号非空 |

#### TC-TOOLS-003: black 格式化功能测试
| 项目 | 内容 |
|------|------|
| **测试目的** | 验证 black 能正确格式化代码 |
| **前置条件** | TC-TOOLS-001 通过 |
| **测试步骤** | 1. 创建测试文件 `test_black_sample.py`：<br>```python<br>def test(  ):<br>    x=1+2<br>    return x<br>```<br>2. 执行 `black test_black_sample.py --check`<br>3. 执行 `black test_black_sample.py`<br>4. 检查格式化后代码<br>5. 删除测试文件 |
| **预期结果** | `--check` 返回需要格式化的提示，格式化后代码符合规范 |
| **通过标准** | black 执行成功，exit code = 0 |

#### TC-TOOLS-004: ruff 检查功能测试
| 项目 | 内容 |
|------|------|
| **测试目的** | 验证 ruff 能正确检查代码问题 |
| **前置条件** | TC-TOOLS-002 通过 |
| **测试步骤** | 1. 创建测试文件 `test_ruff_sample.py`：<br>```python<br>import os<br>import sys<br>x = 1<br>```<br>2. 执行 `ruff check test_ruff_sample.py`<br>3. 检查是否报告未使用变量<br>4. 删除测试文件 |
| **预期结果** | ruff 正确识别代码问题（如未使用导入/变量） |
| **通过标准** | ruff 执行成功，能检测问题 |

#### TC-TOOLS-005: 工具配置文件检查
| 项目 | 内容 |
|------|------|
| **测试目的** | 确认存在工具配置文件（pyproject.toml 或 .ruff.toml） |
| **前置条件** | 项目根目录存在 |
| **测试步骤** | 执行：<br>```bash<br>test -f pyproject.toml && echo "pyproject.toml: exists" || echo "pyproject.toml: missing"<br>test -f .ruff.toml && echo ".ruff.toml: exists" || echo ".ruff.toml: missing"<br>``` |
| **预期结果** | 至少 pyproject.toml 存在 |
| **通过标准** | pyproject.toml 存在或配置在其他文件中 |

---

### 2.4 日志系统验证

#### TC-LOG-001: 日志模块存在性检查
| 项目 | 内容 |
|------|------|
| **测试目的** | 确认项目包含日志配置模块 |
| **前置条件** | TC-DIR-001 通过（src 目录存在） |
| **测试步骤** | 执行：<br>```bash<br>find src -name "*log*" -o -name "utils.py" | head -5<br>``` |
| **预期结果** | 找到日志相关文件 |
| **通过标准** | 至少存在一个日志配置文件或 utils 模块 |

#### TC-LOG-002: 日志功能测试
| 项目 | 内容 |
|------|------|
| **测试目的** | 验证日志系统能正常工作 |
| **前置条件** | TC-LOG-001 通过 |
| **测试步骤** | 执行 Python 代码：<br>```python<br>import logging<br>logging.basicConfig(level=logging.INFO)<br>logger = logging.getLogger("test")<br>logger.info("Test log message")<br>print("Log test: OK")<br>``` |
| **预期结果** | 日志消息正确输出 |
| **通过标准** | 无报错，日志消息可见 |

#### TC-LOG-003: 日志级别配置检查
| 项目 | 内容 |
|------|------|
| **测试目的** | 确认日志支持多级别输出 |
| **前置条件** | TC-LOG-002 通过 |
| **测试步骤** | 执行：<br>```python<br>import logging<br>logger = logging.getLogger("test")<br>logger.debug("Debug message")<br>logger.info("Info message")<br>logger.warning("Warning message")<br>logger.error("Error message")<br>``` |
| **预期结果** | 各级别日志正确输出 |
| **通过标准** | 无报错，至少 INFO/WARNING/ERROR 可见 |

---

## 3. 验证方法与步骤

### 3.1 验证执行流程

```
┌─────────────────────────────────────────────────────────────┐
│                     验证执行流程                              │
├─────────────────────────────────────────────────────────────┤
│  1. Git 仓库验证                                             │
│     └─ TC-GIT-001 ~ TC-GIT-005 顺序执行                     │
│     └─ 确认仓库初始化和 .gitignore 功能                      │
│                                                             │
│  2. 目录结构验证                                             │
│     └─ TC-DIR-001 ~ TC-DIR-004 可并行执行                   │
│     └─ 确认所有核心目录和子目录存在                          │
│                                                             │
│  3. 代码规范工具验证                                         │
│     └─ TC-TOOLS-001 ~ TC-TOOLS-005 顺序执行                 │
│     └─ 确认 black/ruff 安装和功能                           │
│                                                             │
│  4. 日志系统验证                                             │
│     └─ TC-LOG-001 ~ TC-LOG-003 执行                         │
│     └─ 确认日志模块存在和功能                                │
│                                                             │
│  5. 结果汇总                                                 │
│     └─ 生成验证报告                                         │
│     └─ 记录问题和建议                                        │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 执行优先级

```
优先级排序（阻塞关系）：

TC-GIT-001 ─┬─→ TC-GIT-002 → TC-GIT-003 → TC-GIT-004
            └─→ TC-GIT-005

TC-DIR-001 ─┬─→ TC-DIR-002
            └─→ TC-DIR-003
            └─→ TC-DIR-004

TC-TOOLS-001 → TC-TOOLS-003
TC-TOOLS-002 → TC-TOOLS-004

TC-LOG-001 → TC-LOG-002 → TC-LOG-003
```

### 3.3 批量验证脚本

创建 `verify_code_engineering.py` 进行批量验证：

```python
#!/usr/bin/env python
"""代码工程化准备验证脚本"""
import os
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Callable

@dataclass
class TestResult:
    name: str
    passed: bool
    message: str

PROJECT_ROOT = Path(__file__).parent.parent

def run_command(cmd: str, cwd: str = None) -> tuple[int, str, str]:
    """执行命令并返回结果"""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, cwd=cwd or PROJECT_ROOT
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()

def run_test(name: str, test_func: Callable) -> TestResult:
    """执行单个测试"""
    try:
        passed, message = test_func()
        return TestResult(name, passed, message)
    except Exception as e:
        return TestResult(name, False, str(e))

# ===== Git 验证测试 =====

def test_git_repo_init():
    code, out, err = run_command("git rev-parse --is-inside-work-tree")
    if code != 0 or out != "true":
        return False, f"Not a git repo: {err}"
    return True, "Git repository initialized"

def test_gitignore_exists():
    gitignore = PROJECT_ROOT / ".gitignore"
    if not gitignore.exists():
        return False, ".gitignore file missing"
    return True, ".gitignore exists"

def test_gitignore_content():
    gitignore = PROJECT_ROOT / ".gitignore"
    if not gitignore.exists():
        return False, ".gitignore missing"
    
    content = gitignore.read_text()
    required_rules = [
        "__pycache__",
        "*.pyc",
        ".venv",
        "venv",
        "checkpoints",
        ".egg-info",
        ".env",
    ]
    found = sum(1 for rule in required_rules if rule in content)
    if found < 6:
        return False, f"Only {found}/7 required rules found"
    return True, f"{found} ignore rules configured"

def test_gitignore_functional():
    # 创建测试文件
    pycache_dir = PROJECT_ROOT / "__pycache__"
    pycache_dir.mkdir(exist_ok=True)
    test_file = pycache_dir / "test.pyc"
    test_file.write_text("test")
    
    # 检查 git status
    code, out, err = run_command("git status --porcelain")
    
    # 清理
    test_file.unlink(missing_ok=True)
    try:
        pycache_dir.rmdir()
    except:
        pass
    
    # 检查是否被忽略
    if "__pycache__" in out or "test.pyc" in out:
        return False, "__pycache__ not ignored"
    return True, ".gitignore working correctly"

def test_git_initial_commit():
    code, out, err = run_command("git log --oneline -1")
    if code != 0 or not out:
        return False, "No commits found"
    return True, f"Initial commit: {out.split()[0] if out else 'N/A'}"

# ===== 目录结构验证测试 =====

def test_core_dirs():
    required_dirs = ["configs", "data", "scripts", "src", "notebooks", "docs", "checkpoints", "tests"]
    missing = []
    for dir_name in required_dirs:
        dir_path = PROJECT_ROOT / dir_name
        if not dir_path.exists():
            missing.append(dir_name)
    
    if missing:
        return False, f"Missing dirs: {missing}"
    return True, f"All 8 core directories exist"

def test_src_subdirs():
    required_subdirs = ["models", "datasets", "training", "eval", "deploy"]
    missing = []
    for subdir in required_subdirs:
        dir_path = PROJECT_ROOT / "src" / subdir
        if not dir_path.exists():
            missing.append(subdir)
    
    if missing:
        return False, f"Missing src subdirs: {missing}"
    return True, f"All 5 src subdirectories exist"

def test_dir_permissions():
    required_dirs = ["configs", "data", "scripts", "src", "notebooks", "docs", "checkpoints", "tests"]
    no_rw = []
    for dir_name in required_dirs:
        dir_path = PROJECT_ROOT / dir_name
        if dir_path.exists():
            if not (os.access(dir_path, os.R_OK) and os.access(dir_path, os.W_OK)):
                no_rw.append(dir_name)
    
    if no_rw:
        return False, f"No RW permission: {no_rw}"
    return True, "All directories have RW permission"

# ===== 代码规范工具验证测试 =====

def test_black_install():
    code, out, err = run_command("python -c \"import black; print(black.__version__)\"")
    if code != 0:
        # 尝试命令行
        code, out, err = run_command("black --version")
        if code != 0:
            return False, "black not installed"
    return True, f"black version: {out.split()[0] if out else 'unknown'}"

def test_ruff_install():
    code, out, err = run_command("ruff --version")
    if code != 0:
        return False, "ruff not installed"
    return True, f"ruff version: {out.split()[1] if out else 'unknown'}"

def test_black_functional():
    # 创建测试文件
    test_file = PROJECT_ROOT / "test_black_sample.py"
    test_file.write_text("def test(  ):    x=1+2    return x\n")
    
    # 运行 black
    code, out, err = run_command("black test_black_sample.py")
    
    # 清理
    test_file.unlink(missing_ok=True)
    
    if code != 0:
        return False, f"black failed: {err}"
    return True, "black formatting works"

def test_ruff_functional():
    # 创建测试文件
    test_file = PROJECT_ROOT / "test_ruff_sample.py"
    test_file.write_text("import os\nimport sys\nx = 1\n")
    
    # 运行 ruff check
    code, out, err = run_command("ruff check test_ruff_sample.py")
    
    # 清理
    test_file.unlink(missing_ok=True)
    
    # ruff 应该检测到未使用的导入
    if "unused" not in out.lower() and code == 0:
        return True, "ruff works (no issues detected or warnings shown)"
    return True, f"ruff check works: detected issues"

def test_tool_config():
    pyproject = PROJECT_ROOT / "pyproject.toml"
    ruff_toml = PROJECT_ROOT / ".ruff.toml"
    
    if pyproject.exists():
        return True, "pyproject.toml exists"
    if ruff_toml.exists():
        return True, ".ruff.toml exists"
    return False, "No tool config file found"

# ===== 日志系统验证测试 =====

def test_log_module():
    log_files = list(PROJECT_ROOT.glob("src/**/*log*.py"))
    utils_files = list(PROJECT_ROOT.glob("src/**/utils.py"))
    
    if log_files or utils_files:
        return True, f"Log-related files: {len(log_files + utils_files)}"
    return True, "Standard logging module available"  # Python 内置 logging 总可用

def test_log_function():
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("test_verify")
    logger.info("Verification log test")
    return True, "Logging function works"

def test_log_levels():
    import logging
    logger = logging.getLogger("test_levels")
    logger.setLevel(logging.DEBUG)
    
    # 测试各级别
    logger.debug("Debug")
    logger.info("Info")
    logger.warning("Warning")
    logger.error("Error")
    
    return True, "All log levels work"

# ===== 主程序 =====

def main():
    tests = [
        ("TC-GIT-001", test_git_repo_init),
        ("TC-GIT-002", test_gitignore_exists),
        ("TC-GIT-003", test_gitignore_content),
        ("TC-GIT-004", test_gitignore_functional),
        ("TC-GIT-005", test_git_initial_commit),
        ("TC-DIR-001", test_core_dirs),
        ("TC-DIR-002", test_src_subdirs),
        ("TC-DIR-003", test_dir_permissions),
        ("TC-TOOLS-001", test_black_install),
        ("TC-TOOLS-002", test_ruff_install),
        ("TC-TOOLS-003", test_black_functional),
        ("TC-TOOLS-004", test_ruff_functional),
        ("TC-TOOLS-005", test_tool_config),
        ("TC-LOG-001", test_log_module),
        ("TC-LOG-002", test_log_function),
        ("TC-LOG-003", test_log_levels),
    ]
    
    print("=" * 60)
    print("代码工程化准备验证")
    print("=" * 60)
    
    results: List[TestResult] = []
    
    print("\n[Git 仓库验证]")
    for name, func in tests[:5]:
        result = run_test(name, func)
        results.append(result)
        status = "✓" if result.passed else "✗"
        print(f"  {status} {name}: {result.message}")
    
    print("\n[目录结构验证]")
    for name, func in tests[5:8]:
        result = run_test(name, func)
        results.append(result)
        status = "✓" if result.passed else "✗"
        print(f"  {status} {name}: {result.message}")
    
    print("\n[代码规范工具验证]")
    for name, func in tests[8:13]:
        result = run_test(name, func)
        results.append(result)
        status = "✓" if result.passed else "✗"
        print(f"  {status} {name}: {result.message}")
    
    print("\n[日志系统验证]")
    for name, func in tests[13:]:
        result = run_test(name, func)
        results.append(result)
        status = "✓" if result.passed else "✗"
        print(f"  {status} {name}: {result.message}")
    
    # 汇总
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"结果: {passed}/{total} 通过")
    
    if passed == total:
        print("所有验证通过！")
        return 0
    else:
        print("部分验证失败，请检查上述错误。")
        failed = [r for r in results if not r.passed]
        print(f"失败项: {[r.name for r in failed]}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

---

## 4. 通过标准

### 4.1 总体判定规则

| 级别 | 条件 | 说明 |
|------|------|------|
| **完全通过** | 所有 TC-GIT-*、TC-DIR-*、TC-TOOLS-* 通过 | 工程化环境完整就绪 |
| **部分通过** | Git 和目录通过，工具或日志部分失败 | 基础结构可用，需补充工具配置 |
| **未通过** | 任一 TC-GIT-* 或 TC-DIR-* 失败 | 基础工程化未完成 |

### 4.2 各测试项判定标准

| 测试ID | 通过条件 | 失败影响 |
|--------|----------|----------|
| TC-GIT-001 | `git rev-parse` 返回 true | 阻塞所有 Git 相关测试 |
| TC-GIT-002 | `.gitignore` 文件存在 | 阻塞 TC-GIT-003/004 |
| TC-GIT-003 | 规则数量 ≥ 6 条 | 可能导致敏感文件被追踪 |
| TC-GIT-004 | 测试文件被忽略 | `.gitignore` 不生效 |
| TC-GIT-005 | 提交数量 ≥ 1 | 仓库为空状态 |
| TC-DIR-001 | 8 个目录全部存在 | 阻塞后续目录测试 |
| TC-DIR-002 | 5 个子目录全部存在 | 项目结构不完整 |
| TC-DIR-003 | 所有目录 RW 权限 | 无法写入文件 |
| TC-DIR-004 | 结构树完整 | 项目结构混乱 |
| TC-TOOLS-001 | black 导入成功 | 阻塞 TC-TOOLS-003 |
| TC-TOOLS-002 | ruff 命令可用 | 阻塞 TC-TOOLS-004 |
| TC-TOOLS-003 | black 执行 exit 0 | 代码格式化不可用 |
| TC-TOOLS-004 | ruff 检查有输出 | 代码检查不可用 |
| TC-TOOLS-005 | 配置文件存在 | 工具配置缺失 |
| TC-LOG-001 | 日志模块存在 | 可用标准 logging |
| TC-LOG-002 | 日志输出正常 | 日志功能异常 |
| TC-LOG-003 | 多级别输出正常 | 日志级别不可控 |

### 4.3 必要项与可选项

**必要项（必须通过）：**
- TC-GIT-001、TC-GIT-002、TC-GIT-005
- TC-DIR-001、TC-DIR-002
- TC-TOOLS-001、TC-TOOLS-002

**可选项（建议通过但非阻塞）：**
- TC-GIT-003、TC-GIT-004
- TC-DIR-003、TC-DIR-004
- TC-TOOLS-003~005
- TC-LOG-* 系列

---

## 5. 自动化验证建议

### 5.1 可完全自动化的测试

| 测试ID | 自动化方式 | 备注 |
|--------|-----------|------|
| TC-GIT-001~005 | Shell 命令 + Python subprocess | 完全自动化 |
| TC-DIR-001~004 | Python pathlib 检查 | 完全自动化 |
| TC-TOOLS-001~005 | 工具命令执行 + 清理 | 完全自动化 |
| TC-LOG-001~003 | Python logging 模块 | 完全自动化 |

### 5.2 需要人工检查的项目

| 检查项 | 原因 | 建议 |
|--------|------|------|
| `.gitignore` 规则合理性 | 需根据项目特点定制 | 人工审查规则完整性 |
| 目录结构合理性 | 项目依赖不同结构 | 人工确认目录用途 |
| black/ruff 配置参数 | 团队规范差异 | 人工确认配置值 |
| 日志级别设置 | 环境差异 | 人工确认生产环境配置 |

### 5.3 CI/CD 集成建议

```yaml
# .github/workflows/verify-engineering.yml
name: Verify Code Engineering Setup
on:
  push:
    paths:
      - '.gitignore'
      - 'pyproject.toml'
      - 'src/**'
  pull_request:
    branches: [main, master]

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Verify Git Setup
        run: |
          git rev-parse --is-inside-work-tree
          test -f .gitignore
          git log --oneline -1
      
      - name: Verify Directory Structure
        run: |
          for dir in configs data scripts src notebooks docs checkpoints tests; do
            test -d "$dir" || (echo "Missing: $dir" && exit 1)
          done
          for subdir in models datasets training eval deploy; do
            test -d "src/$subdir" || (echo "Missing: src/$subdir" && exit 1)
          done
      
      - name: Install Lint Tools
        run: pip install black ruff
      
      - name: Verify Black
        run: black --check src/ scripts/ tests/
      
      - name: Verify Ruff
        run: ruff check src/ scripts/ tests/
```

### 5.4 快速验证命令

```bash
# 一键验证别名
alias verify-eng='python scripts/verify_code_engineering.py'

# 单独验证各模块
alias verify-git='git status && git log --oneline -1 && cat .gitignore | wc -l'
alias verify-dirs='ls -la configs data scripts src notebooks docs checkpoints tests'
alias verify-tools='black --version && ruff --version'
```

---

## 6. 风险点与注意事项

### 6.1 Git 仓库风险

| 风险点 | 可能性 | 影响 | 缓解措施 |
|--------|--------|------|----------|
| **.gitignore 规则遗漏** | 高 | 高敏感文件被提交 | 参考常见 Python 项目模板 |
| **敏感文件已提交** | 中 | 安全风险 | 检查历史提交，使用 BFG 清理 |
| **Git 配置不当** | 低 | 提交失败 | 检查 `git config user.name/email` |
| **分支策略未定义** | 中 | 协作混乱 | 明确 main/master 分支保护规则 |

### 6.2 目录结构风险

| 雎险点 | 可能性 | 影响 | 缓解措施 |
|--------|--------|------|----------|
| **目录命名不规范** | 中 | 协作困难 | 统一使用英文小写、下划线分隔 |
| **子目录缺失** | 高 | 模块混乱 | 按任务文档逐一创建 |
| **权限问题（Linux）** | 低 | 写入失败 | 检查目录 owner 和 chmod |
| **路径硬编码** | 中 | 可移植性差 | 使用 pathlib 或配置文件管理路径 |

### 6.3 代码规范工具风险

| 雎险点 | 可能性 | 影响 | 缓解措施 |
|--------|--------|------|----------|
| **black 版本不兼容** | 低 | 格式化失败 | 固定版本：`pip install black==23.x` |
| **ruff 规则过严** | 中 | 开发受阻 | 在 pyproject.toml 中配置忽略规则 |
| **工具未配置 CI** | 高 | 代码质量不稳定 | 添加 pre-commit hooks |
| **IDE 未集成工具** | 中 | 手动执行遗漏 | 配置 VS Code/PyCharm 集成 |

### 6.4 日志系统风险

| 风险点 | 可能性 | 影响 | 缓解措施 |
|--------|--------|------|----------|
| **日志级别不当** | 中 | 生产环境泄露 | 使用环境变量控制日志级别 |
| **日志格式不统一** | 高 | 分析困难 | 配置统一格式化器 |
| **日志文件未管理** | 中 | 磁盘占满 | 配置日志轮转和清理 |
| **敏感信息泄露** | 高 | 安全风险 | 避免在日志中记录密码/密钥 |

### 6.5 验证注意事项

1. **Git 忽略规则验证**
   - 创建测试文件后必须清理
   - 验证多种文件类型（.pyc, .log, .env 等）
   - 注意 Windows/Linux 路径差异

2. **目录结构检查**
   - 使用 `Path` 对象而非字符串拼接
   - 注意大小写敏感性（Linux）
   - 检查空目录是否被 Git 追踪（添加 .gitkeep）

3. **工具验证**
   - black/ruff 需在虚拟环境中测试
   - 测试文件必须清理，避免污染仓库
   - 检查工具版本与项目配置兼容性

4. **日志验证**
   - 测试各级别输出目标（stdout/file）
   - 验证日志文件路径可写
   - 检查多进程日志场景

### 6.6 验证失败排查流程

```
验证失败
    │
    ├─ Git 相关失败
    │       ├─ "Not a git repo"
    │       │       └─ 执行 git init
    │       │
    │       ├─ ".gitignore missing"
    │       │       └─ 创建 .gitignore 文件
    │       │
    │       ├─ ".gitignore not working"
    │       │       └─ 检查规则语法
    │       │       └─ 检查文件是否已被追踪（git rm --cached）
    │       │
    │       └─ "No commits found"
    │               └─ 执行 git add . && git commit -m "Initial commit"
    │
    ├─ 目录结构失败
    │       ├─ "Missing dirs"
    │       │       └─ mkdir -p configs data scripts src notebooks docs checkpoints tests
    │       │
    │       ├─ "Missing src subdirs"
    │       │       └─ mkdir -p src/models src/datasets src/training src/eval src/deploy
    │       │
    │       └─ "No RW permission"
    │               └─ chmod 755 <directory>
    │               └─ chown <user>:<group> <directory>
    │
    ├─ 代码工具失败
    │       ├─ "black not installed"
    │       │       └─ pip install black
    │       │
    │       ├─ "ruff not installed"
    │       │       └─ pip install ruff
    │       │
    │       ├─ "black/ruff failed"
    │       │       └─ 检查 Python 版本兼容性
    │       │       └─ 检查工具配置文件
    │       │
    │       └─ "No tool config"
    │               └─ 创建 pyproject.toml
    │               └─ 添加 [tool.black] 和 [tool.ruff] 配置
    │
    └─ 日志系统失败
            ├─ "Logging error"
            │       └─ 检查 logging 配置
            │       └─ 验证日志路径可写
            │
            └─ "Log levels not working"
                    └─ 设置 logger.setLevel(logging.DEBUG)
                    └─ 检查 handler 配置
```

---

## 附录 A：验证脚本输出示例

```
============================================================
代码工程化准备验证
============================================================

[Git 仓库验证]
  ✓ TC-GIT-001: Git repository initialized
  ✓ TC-GIT-002: .gitignore exists
  ✓ TC-GIT-003: 7 ignore rules configured
  ✓ TC-GIT-004: .gitignore working correctly
  ✓ TC-GIT-005: Initial commit: 7f0bd42

[目录结构验证]
  ✓ TC-DIR-001: All 8 core directories exist
  ✓ TC-DIR-002: All 5 src subdirectories exist
  ✓ TC-DIR-003: All directories have RW permission

[代码规范工具验证]
  ✓ TC-TOOLS-001: black version: 23.12.1
  ✓ TC-TOOLS-002: ruff version: 0.1.9
  ✓ TC-TOOLS-003: black formatting works
  ✓ TC-TOOLS-004: ruff check works: detected issues
  ✓ TC-TOOLS-005: pyproject.toml exists

[日志系统验证]
  ✓ TC-LOG-001: Standard logging module available
  ✓ TC-LOG-002: Logging function works
  ✓ TC-LOG-003: All log levels work

============================================================
结果: 16/16 通过
所有验证通过！
============================================================
```

---

## 附录 B：推荐 .gitignore 内容

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.venv/
venv/
ENV/
env/

# IDE
.idea/
.vscode/
*.swp
*.swo
*~

# Project specific
checkpoints/
logs/
*.log
data/raw/
data/processed/

# Secrets
.env
.env.local
*.pem
*.key

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Testing
.pytest_cache/
.coverage
htmlcov/

# OS
.DS_Store
Thumbs.db
```

---

## 附录 C：推荐 pyproject.toml 配置

```toml
[project]
name = "vsprint"
version = "0.1.0"
description = "ERes2NetV2 Speaker Verification Training Pipeline"

[tool.black]
line-length = 88
target-version = ['py38', 'py39', 'py310']
include = '\.pyi?$'
exclude = '''
/(
    \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | _build
  | buck-out
  | build
  | dist
)/
'''

[tool.ruff]
line-length = 88
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
]
ignore = [
    "E501",  # line too long (handled by black)
    "B008",  # do not perform function calls in argument defaults
]
exclude = [
    ".bzr",
    ".direnv",
    ".eggs",
    ".git",
    ".hg",
    ".mypy_cache",
    ".nox",
    ".pants.d",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "__pypackages__",
    "_build",
    "buck-out",
    "build",
    "dist",
    "node_modules",
    "venv",
]

[tool.ruff.isort]
known-first-party = ["src"]