import sys, json, http.client
sys.stdout.reconfigure(encoding='utf-8')

token = 't-g10462ezT3QYQN2YXW6WAFOPE6N5PJN4U3DYLFQY'
chat_id = 'ou_3824ea2d19ebc98f7336ec3f0c475d47'

PROJECT_ROOT = r'C:\Users\Administrator\vsprint'
TASKS_DIR = r'C:\Users\Administrator\vsprint\docs\superpowers\tasks'
AUDIT_DIR = r'C:\Users\Administrator\vsprint\docs\superpowers\audit'

# Find the first task file
import os, glob
task_files = sorted(glob.glob(os.path.join(TASKS_DIR, '**', '*.md'), recursive=True))
task_files = [f for f in task_files if not f.endswith('README.md')]

if not task_files:
    print("No task files found.")
    sys.exit(1)

# Process the first one
task_path = task_files[0]
print(f'Found task: {os.path.basename(task_path)}')

# Get output path
rel = os.path.relpath(task_path, TASKS_DIR)
parts = os.path.split(rel)
stage_dir = parts[0]
filename = parts[1]
name, ext = os.path.splitext(filename)
verify_name = name + '_verify' + ext
audit_stage_dir = os.path.join(AUDIT_DIR, stage_dir)
os.makedirs(audit_stage_dir, exist_ok=True)
output_path = os.path.join(audit_stage_dir, verify_name)

# Read task content
with open(task_path, 'r', encoding='utf-8') as f:
    task_content = f.read()

print(f'Task content length: {len(task_content)} chars')

# Build prompt
prompt = f"""你正在使用 superpowers verification skill 完成一项测试验证设计任务。

请阅读以下任务文档的内容，然后为该任务设计一份完整的测试验证方案文档。

## 任务文档内容
{task_content}

## 你的任务

基于上述任务文档，设计对应的测试验证方案，要求：

1. 验证目标 — 明确该任务每个验收标准如何被验证
2. 测试用例 — 为每个任务清单项设计具体的测试用例
3. 验证方法 — 说明每个测试用例的执行步骤和预期结果
4. 通过标准 — 定义每个验证项的通过/失败判定条件
5. 自动化建议 — 哪些验证可以自动化，哪些需要手动
6. 风险点 — 识别该任务可能的风险和注意事项

## 输出要求

将验证方案写入文件：{output_path}

文档格式为 Markdown，标题使用以下结构：

# [任务名称] — 验证测试设计

> 来源任务：[任务文档路径]
> 生成日期：[日期]

---

## 1. 验证目标概述
## 2. 测试用例设计
### 2.1 [用例分类1]
### 2.2 [用例分类2]
## 3. 验证方法与步骤
## 4. 通过标准
## 5. 自动化验证建议
## 6. 风险点与注意事项

请确保内容具体、可执行、可量化。不要泛泛而谈。

现在请生成验证方案并写入文件。"""

print(f'Prompt length: {len(prompt)} chars')
print('Calling claude -p...')

import subprocess
try:
    result = subprocess.run(
        ['claude', '-p', prompt],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        encoding='utf-8'
    )
    
    print(f'claude exit code: {result.returncode}')
    
    if os.path.exists(output_path):
        print(f'Success! Verify doc created: {output_path}')
        size = os.path.getsize(output_path)
        print(f'File size: {size} bytes')
        
        # Send the file to user via Feishu
        file_name = os.path.basename(output_path)
        with open(output_path, 'rb') as f:
            file_content = f.read()
        
        from email.mime.multipart import MIMEMultipart
        from email.mime.base import MIMEBase
        from email.mime.text import MIMEText
        
        msg = MIMEMultipart('form-data')
        
        part1 = MIMEText('stream', 'plain')
        part1.add_header('Content-Disposition', 'form-data', name='file_type')
        msg.attach(part1)
        
        part2 = MIMEText(file_name, 'plain')
        part2.add_header('Content-Disposition', 'form-data', name='file_name')
        msg.attach(part2)
        
        part3 = MIMEBase('application', 'octet-stream')
        part3.set_payload(file_content)
        part3.add_header('Content-Disposition', 'form-data', name='file', filename=file_name)
        part3.add_header('Content-Type', 'application/octet-stream')
        msg.attach(part3)
        
        body_bytes = msg.as_bytes()
        content_type = msg.get('Content-Type')
        
        conn = http.client.HTTPSConnection('open.feishu.cn')
        conn.request('POST', '/open-apis/im/v1/files', body=body_bytes, headers={
            'Authorization': 'Bearer ' + token,
            'Content-Type': content_type
        })
        resp = conn.getresponse()
        resp_body = resp.read().decode('utf-8')
        upload_result = json.loads(resp_body)
        
        if upload_result.get('code') == 0:
            file_key = upload_result.get('data', {}).get('file_key')
            send_data = json.dumps({
                'receive_id': chat_id,
                'msg_type': 'file',
                'content': json.dumps({'file_key': file_key})
            }).encode('utf-8')
            
            conn2 = http.client.HTTPSConnection('open.feishu.cn')
            conn2.request('POST', '/open-apis/im/v1/messages?receive_id_type=open_id', body=send_data, headers={
                'Authorization': 'Bearer ' + token,
                'Content-Type': 'application/json'
            })
            resp2 = conn2.getresponse()
            resp2_body = resp2.read().decode('utf-8')
            print(f'Send result: {resp2_body}')
        else:
            print(f'Upload error: {resp_body}')
    else:
        if result.stdout.strip():
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result.stdout)
            print(f'Verify doc created from stdout: {output_path}')
            print(f'Content preview: {result.stdout[:200]}...')
        else:
            print('No output received')
            if result.stderr.strip():
                print(f'stderr: {result.stderr[:500]}')
    
except subprocess.TimeoutExpired:
    print('claude timed out (600s)')
except FileNotFoundError:
    print('claude command not found')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
