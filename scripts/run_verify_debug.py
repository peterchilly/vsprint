#!/usr/bin/env python
# -*- coding: utf-8 -*-
import subprocess, sys, os, traceback

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

PROJECT_ROOT = r'C:\Users\Administrator\vsprint'
TASKS_DIR = os.path.join(PROJECT_ROOT, 'docs', 'superpowers', 'tasks')
AUDIT_DIR = os.path.join(PROJECT_ROOT, 'docs', 'superpowers', 'audit')

import glob as globmod

task_files = sorted(globmod.glob(os.path.join(TASKS_DIR, '**', '*.md'), recursive=True))
task_files = [f for f in task_files if not f.endswith('README.md')]

os.makedirs(AUDIT_DIR, exist_ok=True)

def get_output(task_path):
    rel = os.path.relpath(task_path, TASKS_DIR)
    parts = os.path.split(rel)
    stage = parts[0]
    fname = parts[1]
    name, ext = os.path.splitext(fname)
    outdir = os.path.join(AUDIT_DIR, stage)
    os.makedirs(outdir, exist_ok=True)
    return os.path.join(outdir, name + '_verify' + ext), rel

done = 0
failed = 0
skipped = 0
errors = []

for tp in task_files:
    outpath, rel = get_output(tp)
    if os.path.exists(outpath):
        skipped += 1
        continue

    # Read task
    try:
        with open(tp, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        msg = 'Read failed: ' + rel + ' - ' + str(e)
        print(msg)
        errors.append(msg)
        failed += 1
        continue

    prompt = ('Please read the following task document and design a complete verification test plan for it.\n\n'
              '## Task content\n\n' + content + '\n\n'
              'Write the verification plan as a markdown file to: ' + outpath + '\n\n'
              'Include: verification objectives, test cases, verification methods, pass criteria, '
              'automation suggestions, and risk points. '
              'Make it specific, actionable, and quantifiable.')

    try:
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'

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
            if os.path.exists(outpath):
                print('OK: ' + rel)
                done += 1
            elif result.stdout.strip():
                with open(outpath, 'w', encoding='utf-8') as f:
                    f.write(result.stdout)
                print('OK (from stdout): ' + rel)
                done += 1
            else:
                msg = 'No output for: ' + rel
                print(msg)
                errors.append(msg)
                failed += 1
        else:
            msg = 'Claude failed for: ' + rel + ' (code ' + str(result.returncode) + ')'
            if result.stderr.strip():
                msg += ' stderr: ' + result.stderr[:300]
            if result.stdout.strip():
                msg += ' stdout: ' + result.stdout[:300]
            print(msg)
            errors.append(msg)
            failed += 1

    except subprocess.TimeoutExpired:
        msg = 'Timeout for: ' + rel
        print(msg)
        errors.append(msg)
        failed += 1
    except Exception as e:
        msg = 'Error for: ' + rel + ' - ' + str(e)
        print(msg)
        errors.append(msg)
        failed += 1

print('\n' + '=' * 60)
print('Done: ' + str(done) + ', Skipped: ' + str(skipped) + ', Failed: ' + str(failed))
if errors:
    print('\nErrors:')
    for e in errors:
        print('  - ' + e)
