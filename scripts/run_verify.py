#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为 tasks 目录下的任务文档生成对应的验证测试设计文档。
输出运行日志到文件。

用法:
    python scripts\run_verify.py
"""

import os
import sys
import subprocess
import glob
import json
import datetime
import traceback

# UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

PROJECT_ROOT = r'C:\Users\Administrator\vsprint'
TASKS_DIR = os.path.join(PROJECT_ROOT, 'docs', 'superpowers', 'tasks')
AUDIT_DIR = os.path.join(PROJECT_ROOT, 'docs', 'superpowers', 'audit')
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'scripts')
LOG_FILE = os.path.join(SCRIPTS_DIR, 'verify_run.log')

os.makedirs(AUDIT_DIR, exist_ok=True)

def log(msg):
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = '[' + ts + '] ' + msg
    print(line)
    sys.stdout.flush()
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def get_output_path(task_path):
    rel = os.path.relpath(task_path, TASKS_DIR)
    parts = os.path.split(rel)
    stage_dir = parts[0]
    filename = parts[1]
    name, ext = os.path.splitext(filename)
    verify_name = name + '_verify' + ext
    audit_stage_dir = os.path.join(AUDIT_DIR, stage_dir)
    os.makedirs(audit_stage_dir, exist_ok=True)
    return os.path.join(audit_stage_dir, verify_name), rel

def generate_verify(task_path, output_path, rel):
    with open(task_path, 'r', encoding='utf-8') as f:
        task_content = f.read()

    prompt = (
        u'Please read the following task document and design a complete verification test plan for it.\n\n'
        u'## Task content\n\n' + task_content + '\n\n'
        u'Write the verification plan as a markdown file to: ' + output_path + '\n\n'
        u'Include: verification objectives, test cases, verification methods, pass criteria, '
        u'automation suggestions, and risk points. '
        u'Make it specific, actionable, and quantifiable. '
        u'Write the output directly to the file path specified above.'
    )

    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'

    try:
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
                size = os.path.getsize(output_path)
                log('  OK: ' + rel + ' (' + str(size) + ' bytes)')
                return True
            elif result.stdout.strip():
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(result.stdout)
                size = os.path.getsize(output_path)
                log('  OK (from stdout): ' + rel + ' (' + str(size) + ' bytes)')
                return True
            else:
                log('  FAIL (no output): ' + rel)
                return False
        else:
            err_msg = '  FAIL (code ' + str(result.returncode) + '): ' + rel
            if result.stderr.strip():
                err_msg += ' stderr: ' + result.stderr[:500]
            if result.stdout.strip():
                err_msg += ' stdout: ' + result.stdout[:500]
            log(err_msg)
            return False

    except subprocess.TimeoutExpired:
        log('  TIMEOUT (600s): ' + rel)
        return False
    except Exception as e:
        log('  ERROR: ' + rel + ' - ' + str(e))
        log('  ' + traceback.format_exc())
        return False

def main():
    # Clear old log
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write('=== Verify Generation Run Log ===\n')
        f.write('Started: ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n\n')

    log('Scanning task documents...')

    task_files = sorted(glob.glob(os.path.join(TASKS_DIR, '**', '*.md'), recursive=True))
    task_files = [f for f in task_files if not f.endswith('README.md')]

    log('Found ' + str(len(task_files)) + ' task documents')

    done = 0
    failed = 0
    skipped = 0
    failed_list = []

    for idx, task_path in enumerate(task_files):
        output_path, rel = get_output_path(task_path)

        if os.path.exists(output_path):
            skipped += 1
            log('SKIP (exists): ' + rel)
            continue

        log('Processing [' + str(idx+1) + '/' + str(len(task_files)) + ']: ' + rel)

        success = generate_verify(task_path, output_path, rel)
        if success:
            done += 1
        else:
            failed += 1
            failed_list.append(rel)

        # Small delay between calls
        import time
        time.sleep(1)

    # Summary
    log('')
    log('=' * 60)
    log('SUMMARY')
    log('=' * 60)
    log('Done:    ' + str(done))
    log('Skipped: ' + str(skipped))
    log('Failed:  ' + str(failed))
    if failed_list:
        log('Failed tasks:')
        for f in failed_list:
            log('  - ' + f)
    log('Total:   ' + str(len(task_files)))
    log('=' * 60)

    # Save summary to JSON
    summary = {
        'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'done': done,
        'skipped': skipped,
        'failed': failed,
        'failed_tasks': failed_list,
        'total': len(task_files)
    }
    summary_path = os.path.join(AUDIT_DIR, 'verify_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log('Summary saved to: ' + summary_path)

if __name__ == '__main__':
    main()
