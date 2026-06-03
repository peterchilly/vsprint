#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为 tasks 目录下的任务文档生成对应的验证测试设计文档。

用法:
    python scripts/generate_verify.py                           # 处理所有 tasks 下的 md 文档
    python scripts/generate_verify.py tasks/01_环境搭建/01_硬件评估与选择.md  # 处理单个文档
    python scripts/generate_verify.py "C:\\path\\to\\task.md"       # 使用绝对路径

对每个任务文档，调用 claude -p 应用 superpowers verification skill，
生成验证测试设计文档并保存到 audit 目录（命名为 原文件名_verify.md）。
"""

import os
import sys
import subprocess
import glob

# 设置 UTF-8 编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS_DIR = os.path.join(PROJECT_ROOT, 'docs', 'superpowers', 'tasks')
AUDIT_DIR = os.path.join(PROJECT_ROOT, 'docs', 'superpowers', 'audit')

os.makedirs(AUDIT_DIR, exist_ok=True)


def get_verify_output_path(task_path):
    """
    根据 task 文档路径，生成对应的 audit verify 文档路径。
    保持阶段子目录结构。

    例: tasks/01_环境搭建/01_硬件评估与选择.md
      -> audit/01_环境搭建/01_硬件评估与选择_verify.md
    """
    # 如果传入的是绝对路径，转为相对于 TASKS_DIR 的相对路径
    if os.path.isabs(task_path):
        task_path = os.path.abspath(task_path)
        if not task_path.startswith(TASKS_DIR):
            print("错误: 指定的文件不在 tasks 目录下")
            return None
        rel = os.path.relpath(task_path, TASKS_DIR)
    else:
        rel = os.path.relpath(os.path.abspath(task_path), TASKS_DIR)

    parts = os.path.split(rel)
    if len(parts) < 2:
        print("错误: 任务文档必须在阶段子目录下")
        return None

    stage_dir = parts[0]
    filename = parts[1]
    name, ext = os.path.splitext(filename)
    verify_name = name + '_verify' + ext

    audit_stage_dir = os.path.join(AUDIT_DIR, stage_dir)
    os.makedirs(audit_stage_dir, exist_ok=True)

    return os.path.join(audit_stage_dir, verify_name)


def generate_verify(task_path, output_path):
    """
    调用 claude -p 应用 superpowers verification skill，
    读取任务文档，生成验证测试设计文档。
    """
    # 读取任务文档内容
    with open(task_path, 'r', encoding='utf-8') as f:
        task_content = f.read()

    # 构建 prompt
    task_rel = os.path.relpath(task_path, PROJECT_ROOT)
    output_rel = os.path.relpath(output_path, PROJECT_ROOT)

    prompt = (
        u"\u4f60\u6b63\u5728\u4f7f\u7528 superpowers verification skill \u5b8c\u6210\u4e00\u9879\u6d4b\u8bd5\u9a8c\u8bc1\u8bbe\u8ba1\u4efb\u52a1\u3002\n\n"
        u"\u8bf7\u9605\u8bfb\u4ee5\u4e0b\u4efb\u52a1\u6587\u6863\u7684\u5185\u5bb9\uff0c\u7136\u540e\u4e3a\u8be5\u4efb\u52a1\u8bbe\u8ba1\u4e00\u4efd\u5b8c\u6574\u7684\u6d4b\u8bd5\u9a8c\u8bc1\u65b9\u6848\u6587\u6863\u3002\n\n"
        u"## \u4efb\u52a1\u6587\u6863\u8def\u5f84\n" + task_rel + "\n\n"
        u"## \u4efb\u52a1\u6587\u6863\u5185\u5bb9\n" + task_content + "\n\n"
        u"## \u4f60\u7684\u4efb\u52a1\n\n"
        u"\u57fa\u4e8e\u4e0a\u8ff0\u4efb\u52a1\u6587\u6863\uff0c\u8bbe\u8ba1\u5bf9\u5e94\u7684\u6d4b\u8bd5\u9a8c\u8bc1\u65b9\u6848\uff0c\u8981\u6c42\uff1a\n\n"
        u"1. **\u9a8c\u8bc1\u76ee\u6807** \u2014 \u660e\u786e\u8be5\u4efb\u52a1\u6bcf\u4e2a\u9a8c\u6536\u6807\u51c6\u5982\u4f55\u88ab\u9a8c\u8bc1\n"
        u"2. **\u6d4b\u8bd5\u7528\u4f8b** \u2014 \u4e3a\u6bcf\u4e2a\u4efb\u52a1\u6e05\u5355\u9879\u8bbe\u8ba1\u5177\u4f53\u7684\u6d4b\u8bd5\u7528\u4f8b\n"
        u"3. **\u9a8c\u8bc1\u65b9\u6cd5** \u2014 \u8bf4\u660e\u6bcf\u4e2a\u6d4b\u8bd5\u7528\u4f8b\u7684\u6267\u884c\u6b65\u9aa4\u548c\u9884\u671f\u7ed3\u679c\n"
        u"4. **\u901a\u8fc7\u6807\u51c6** \u2014 \u5b9a\u4e49\u6bcf\u4e2a\u9a8c\u8bc1\u9879\u7684\u901a\u8fc7/\u5931\u8d25\u5224\u5b9a\u6761\u4ef6\n"
        u"5. **\u81ea\u52a8\u5316\u5efa\u8bae** \u2014 \u54ea\u4e9b\u9a8c\u8bc1\u53ef\u4ee5\u81ea\u52a8\u5316\uff0c\u54ea\u4e9b\u9700\u8981\u624b\u52a8\n"
        u"6. **\u98ce\u9669\u70b9** \u2014 \u8bc6\u522b\u8be5\u4efb\u52a1\u53ef\u80fd\u7684\u98ce\u9669\u548c\u6ce8\u610f\u4e8b\u9879\n\n"
        u"## \u8f93\u51fa\u8981\u6c42\n\n"
        u"\u5c06\u9a8c\u8bc1\u65b9\u6848\u5199\u5165\u6587\u4ef6\uff1a" + output_path + "\n\n"
        u"\u6587\u6863\u683c\u5f0f\u4e3a Markdown\uff0c\u6807\u9898\u4f7f\u7528\u4ee5\u4e0b\u7ed3\u6784\uff1a\n\n"
        u"# [\u4efb\u52a1\u540d\u79f0] \u2014 \u9a8c\u8bc1\u6d4b\u8bd5\u8bbe\u8ba1\n\n"
        u"> **\u6765\u6e90\u4efb\u52a1\uff1a** [\u4efb\u52a1\u6587\u6863\u8def\u5f84]\n"
        u"> **\u751f\u6210\u65e5\u671f\uff1a** [\u65e5\u671f]\n\n"
        u"---\n\n"
        u"## 1. \u9a8c\u8bc1\u76ee\u6807\u6982\u8ff0\n"
        u"## 2. \u6d4b\u8bd5\u7528\u4f8b\u8bbe\u8ba1\n"
        u"## 3. \u9a8c\u8bc1\u65b9\u6cd5\u4e0e\u6b65\u9aa4\n"
        u"## 4. \u901a\u8fc7\u6807\u51c6\n"
        u"## 5. \u81ea\u52a8\u5316\u9a8c\u8bc1\u5efa\u8bae\n"
        u"## 6. \u98ce\u9669\u70b9\u4e0e\u6ce8\u610f\u4e8b\u9879\n\n"
        u"\u8bf7\u786e\u4fdd\u5185\u5bb9\u5177\u4f53\u3001\u53ef\u6267\u884c\u3001\u53ef\u91cf\u5316\u3002\u4e0d\u8981\u6cdb\u6cdb\u800c\u8c08\u3002\n\n"
        u"\u73b0\u5728\u8bf7\u751f\u6210\u9a8c\u8bc1\u65b9\u6848\u5e76\u5199\u5165\u6587\u4ef6\u3002"
    )

    print("处理: " + task_rel)
    print("输出: " + output_rel)

    # 调用 claude -p
    try:
        # 使用 subprocess 并设置环境变量确保 UTF-8
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'

        result = subprocess.run(
            ['claude', '-p', prompt],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=600,
            encoding='utf-8',
            env=env
        )

        if result.returncode == 0:
            if os.path.exists(output_path):
                print("  [OK] 验证文档已生成: " + output_path)
            else:
                if result.stdout.strip():
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(result.stdout)
                    print("  [OK] 验证文档已从输出提取: " + output_path)
                else:
                    print("  [WARN] 未找到输出内容")

            if result.stderr.strip():
                print("  stderr: " + result.stderr[:500])
        else:
            print("  [FAIL] claude 执行失败 (exit code " + str(result.returncode) + ")")
            if result.stderr.strip():
                print("  stderr: " + result.stderr[:500])
            if result.stdout.strip():
                print("  stdout: " + result.stdout[:500])

    except subprocess.TimeoutExpired:
        print("  [FAIL] claude 执行超时 (600s)")
    except FileNotFoundError:
        print("  [FAIL] 找不到 claude 命令，请先安装 Claude Code")
    except Exception as e:
        print("  [FAIL] 错误: " + str(e))


def main():
    args = sys.argv[1:]

    if args:
        task_files = []
        for arg in args:
            # 支持相对路径和绝对路径
            if os.path.isabs(arg):
                abs_path = os.path.abspath(arg)
            else:
                abs_path = os.path.abspath(os.path.join(PROJECT_ROOT, arg))

            if os.path.isfile(abs_path):
                task_files.append(abs_path)
            else:
                print("警告: 文件不存在: " + arg)
    else:
        # 处理 tasks 目录下所有 md 文档
        task_files = sorted(glob.glob(os.path.join(TASKS_DIR, '**', '*.md'), recursive=True))
        # 排除 README.md
        task_files = [f for f in task_files if not f.endswith('README.md')]

    if not task_files:
        print("没有找到任务文档。")
        sys.exit(1)

    print("找到 " + str(len(task_files)) + " 个任务文档\n")

    for task_path in task_files:
        output_path = get_verify_output_path(task_path)
        if output_path is None:
            continue

        # 跳过已生成的文件
        if os.path.exists(output_path):
            print("跳过 (已存在): " + os.path.relpath(output_path, PROJECT_ROOT))
            continue

        generate_verify(task_path, output_path)
        print()

    print("全部完成！")


if __name__ == '__main__':
    main()
