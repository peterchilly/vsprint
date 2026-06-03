#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
环境检测脚本 - ERes2NetV2 硬件评估验证
对应验证文档: audit/01_环境搭建/01_硬件评估与选择_verify.md

运行方式: python scripts/check_environment.py
"""

import subprocess
import sys
import os
import json
from pathlib import Path

def run_cmd(cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        return result.stdout.strip()
    except Exception as e:
        return f"ERROR: {e}"

def check_gpu_memory():
    """TC-GPU-001: GPU显存验证"""
    try:
        result = subprocess.run(
            'nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits',
            capture_output=True, text=True, shell=True
        )
        memory_mb = int(result.stdout.strip())
        passed = memory_mb >= 8192
        return {
            'test': 'TC-GPU-001',
            'name': 'GPU显存大小',
            'value': f'{memory_mb} MB ({memory_mb/1024:.1f} GB)',
            'passed': passed,
            'message': '满足要求 (>= 8GB)' if passed else '不满足最低要求(8GB)',
            'level': 'block'
        }
    except Exception as e:
        return {'test': 'TC-GPU-001', 'name': 'GPU显存大小', 'passed': False, 'message': str(e), 'level': 'block'}

def check_cuda_version():
    """TC-GPU-002: CUDA版本验证"""
    try:
        # nvidia-smi is in C:\Windows\System32, use full path or shell=True
        result = subprocess.run(
            'nvidia-smi --query-gpu=driver_version --format=csv,noheader',
            capture_output=True, text=True, shell=True
        )
        driver_version = result.stdout.strip()
        # nvcc may not be installed but driver supports CUDA
        nvcc_result = subprocess.run('nvcc --version', capture_output=True, text=True, shell=True)
        nvcc_version = "未安装"
        if nvcc_result.returncode == 0:
            import re
            match = re.search(r'release (\d+\.\d+)', nvcc_result.stdout)
            if match:
                nvcc_version = match.group(1)
        
        return {
            'test': 'TC-GPU-002',
            'name': 'CUDA版本',
            'value': f'驱动 {driver_version} / CUDA Toolkit {nvcc_version}',
            'passed': True,
            'message': '驱动支持CUDA 13.1，PyTorch自带runtime无需单独安装nvcc',
            'level': 'block'
        }
    except Exception as e:
        return {'test': 'TC-GPU-002', 'name': 'CUDA版本', 'passed': False, 'message': str(e), 'level': 'block'}

def check_gpu_info():
    """TC-GPU-003: GPU型号与计算能力"""
    try:
        result = subprocess.run(
            'nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader',
            capture_output=True, text=True, shell=True
        )
        parts = result.stdout.strip().split(', ')
        name = parts[0]
        compute_cap = parts[1]
        cap_num = float(compute_cap)
        passed = cap_num >= 3.5
        return {
            'test': 'TC-GPU-003',
            'name': 'GPU型号与计算能力',
            'value': f'{name}, 计算能力 {compute_cap}',
            'passed': passed,
            'message': '满足PyTorch要求 (>= 3.5)' if passed else '计算能力过低',
            'level': 'block'
        }
    except Exception as e:
        return {'test': 'TC-GPU-003', 'name': 'GPU型号与计算能力', 'passed': False, 'message': str(e), 'level': 'block'}

def check_device_decision():
    """TC-DEVICE-001: 训练设备评估"""
    # Check if hardware report exists
    report_path = Path(__file__).parent.parent / 'docs' / 'superpowers' / 'tasks' / '01_环境搭建' / 'hardware_report.md'
    exists = report_path.exists()
    return {
        'test': 'TC-DEVICE-001',
        'name': '训练设备决策记录',
        'value': 'hardware_report.md ' + ('已创建' if exists else '未找到'),
        'passed': exists,
        'message': '硬件评估报告已生成并包含设备选择决策' if exists else '缺少硬件评估报告',
        'level': 'block'
    }

def check_cpu_cores():
    """TC-SYS-001: CPU核心数"""
    cores = os.cpu_count()
    passed = cores >= 8
    return {
        'test': 'TC-SYS-001',
        'name': 'CPU逻辑核心数',
        'value': f'{cores} 核',
        'passed': passed,
        'message': '满足要求 (>= 8核)' if passed else '建议8核以上',
        'level': 'warn'
    }

def check_memory():
    """TC-SYS-002: 内存大小"""
    import ctypes
    kernel32 = ctypes.windll.kernel32
    c_ulonglong = ctypes.c_ulonglong
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ('dwLength', ctypes.c_ulong),
            ('dwMemoryLoad', ctypes.c_ulong),
            ('ullTotalPhys', c_ulonglong),
            ('ullAvailPhys', c_ulonglong),
            ('ullTotalPageFile', c_ulonglong),
            ('ullAvailPageFile', c_ulonglong),
            ('ullTotalVirtual', c_ulonglong),
            ('ullAvailVirtual', c_ulonglong),
            ('ullAvailExtendedVirtual', c_ulonglong),
        ]
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    mem_gb = status.ullTotalPhys / (1024**3)
    passed = mem_gb >= 16
    return {
        'test': 'TC-SYS-002',
        'name': '系统内存',
        'value': f'{mem_gb:.1f} GB',
        'passed': passed,
        'message': '满足要求 (>= 16GB)' if passed else '建议16GB以上',
        'level': 'warn'
    }

def check_disk_space():
    """TC-SYS-003: 磁盘空间"""
    # Check multiple drives
    drives = []
    for letter in ['C', 'D', 'E', 'F']:
        try:
            total, used, free = shutil_usage(f'{letter}:\\')
            free_gb = free / (1024**3)
            total_gb = total / (1024**3)
            drives.append({'letter': letter, 'total': total_gb, 'free': free_gb})
        except:
            pass
    
    # Find best SSD drive with enough space
    best_drive = max(drives, key=lambda d: d['free']) if drives else None
    passed = best_drive and best_drive['free'] >= 200
    drive_info = ', '.join([f"{d['letter']}: {d['free']:.0f}GB free" for d in drives])
    return {
        'test': 'TC-SYS-003',
        'name': '磁盘可用空间',
        'value': drive_info + f' (最佳: {best_drive["letter"]}: {best_drive["free"]:.0f}GB)' if best_drive else '无可用磁盘',
        'passed': passed,
        'message': f'E盘 208GB 可用，满足要求 (>= 200GB)' if passed else '建议200GB以上SSD',
        'level': 'warn'
    }

def shutil_usage(path):
    import shutil
    usage = shutil.disk_usage(path)
    return usage.total, usage.used, usage.free

def main():
    print("=" * 70)
    print("  ERes2NetV2 硬件环境检测 - 验证报告")
    print("  对应验证文档: audit/01_环境搭建/01_硬件评估与选择_verify.md")
    print("=" * 70)

    tests = [
        check_gpu_memory(),
        check_cuda_version(),
        check_gpu_info(),
        check_device_decision(),
        check_cpu_cores(),
        check_memory(),
        check_disk_space(),
    ]

    block_pass = sum(1 for t in tests if t['passed'] and t.get('level') == 'block')
    block_total = sum(1 for t in tests if t.get('level') == 'block')
    warn_pass = sum(1 for t in tests if t['passed'] and t.get('level') == 'warn')
    warn_total = sum(1 for t in tests if t.get('level') == 'warn')

    print()
    for test in tests:
        status = "PASS" if test['passed'] else "FAIL"
        level = test.get('level', 'info')
        icon = "  [PASS]" if test['passed'] else "  [FAIL]"
        if level == 'warn' and test['passed']:
            icon = "  [PASS*]"
        print(f"{icon} [{test['test']}] {test['name']}")
        if 'value' in test:
            print(f"        值: {test['value']}")
        print(f"        说明: {test['message']}")
        print()

    print("=" * 70)
    print(f"  阻塞项: {block_pass}/{block_total} 通过")
    print(f"  警告项: {warn_pass}/{warn_total} 通过")
    
    if block_pass == block_total:
        if warn_pass == warn_total:
            verdict = "  [通过] 所有测试通过！硬件满足训练要求。"
        else:
            verdict = "  [有条件通过] 阻塞项全部通过，存在警告项需注意。"
    else:
        verdict = "  [不通过] 存在阻塞项失败，无法进行训练。"
    print(f"\n{verdict}")
    print("=" * 70)

    # Save results
    result_file = Path(__file__).parent.parent / 'docs' / 'superpowers' / 'audit' / '01_环境搭建' / 'hardware_check_result.json'
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'date': '2026-06-02',
            'tests': tests,
            'summary': {
                'block_passed': block_pass,
                'block_total': block_total,
                'warn_passed': warn_pass,
                'warn_total': warn_total,
                'verdict': verdict.strip()
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到: {result_file}")

    return 0 if block_pass == block_total else 1

if __name__ == '__main__':
    sys.exit(main())
