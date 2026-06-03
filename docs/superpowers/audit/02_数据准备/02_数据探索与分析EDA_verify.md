# 数据探索与分析（EDA）— 验证测试设计

> **来源任务：** docs/superpowers/tasks/02_数据准备/02_数据探索与分析EDA.md
> **生成日期：** 2026/06/02

---

## 1. 验证目标概述

本验证方案针对「数据探索与分析（EDA）」任务，确保以下验收标准得到完整验证：

| 验收标准 | 验证目标 | 验证方式 |
|---------|---------|---------|
| EDA 报告完成（含可视化） | 确认 EDA 报告存在，包含必要的可视化图表，内容完整且有意义 | 文档检查 + 可视化文件验证 |
| 类别不平衡问题已识别 | 确认类别分布统计完成，不平衡问题已量化并记录 | 统计输出验证 + 报告内容审查 |
| 数据集元信息记录完整 | 确认样本数、类别数、平均尺寸等元信息完整记录 | 元信息文件检查 + 数值验证 |

---

## 2. 测试用例设计

### 2.1 类别分布统计验证

| 用例ID | TC-EDA-001 |
|--------|------------|
| **用例名称** | 类别分布统计完整性验证 |
| **测试目标** | 确认所有类别的样本数量已被正确统计 |
| **前置条件** | 数据集已下载并解压完成（前置任务 TC-003 通过） |
| **测试步骤** | 1. 运行类别分布统计脚本<br>2. 检查输出是否包含所有类别<br>3. 验证每个类别的样本计数 > 0<br>4. 确认类别名称与数据集定义一致 |
| **预期结果** | 类别分布统计完整，所有类别均有计数 |
| **通过标准** | 类别覆盖度 = 100%，所有类别计数 > 0 |

### 2.2 类别不平衡量化验证

| 用例ID | TC-EDA-002 |
|--------|------------|
| **用例名称** | 类别不平衡量化验证 |
| **测试目标** | 确认类别不平衡问题已被正确识别和量化 |
| **前置条件** | TC-EDA-001 通过 |
| **测试步骤** | 1. 计算最大类/最小类样本比（Imbalance Ratio）<br>2. 计算类别分布的基尼系数（Gini Coefficient）<br>3. 计算类别熵（Class Entropy）<br>4. 判断不平衡程度：<br>   - IR < 1.5：平衡<br>   - 1.5 ≤ IR < 10：轻度不平衡<br>   - IR ≥ 10：严重不平衡<br>5. 记录不平衡分析结果 |
| **预期结果** | 不平衡指标已计算，问题级别已判定 |
| **通过标准** | IR 值已计算，不平衡级别已判定并记录 |

### 2.3 类别分布可视化验证

| 用例ID | TC-EDA-003 |
|--------|------------|
| **用例名称** | 类别分布可视化验证 |
| **测试目标** | 确认类别分布可视化图表已正确生成 |
| **前置条件** | TC-EDA-001 通过 |
| **测试步骤** | 1. 检查可视化文件存在：<br>   - `eda_reports/class_distribution.png`（柱状图）<br>   - `eda_reports/class_distribution_pie.png`（饼图，可选）<br>2. 验证图像可正常打开<br>3. 验证图像分辨率 ≥ 800×600<br>4. 验证图表包含必要元素：<br>   - 类别标签<br>   - 数值标签<br>   - 图例（如有多个数据系列）<br>   - 标题 |
| **预期结果** | 可视化文件存在，图像质量合格，元素完整 |
| **通过标准** | 文件存在 + 可打开 + 分辨率达标 + 元素完整 |

### 2.4 样本图片可视化验证

| 用例ID | TC-EDA-004 |
|--------|------------|
| **用例名称** | 样本图片可视化验证 |
| **测试目标** | 确认样本图片及标注已正确可视化展示 |
| **前置条件** | 数据集包含图像和标注文件 |
| **测试步骤** | 1. 检查样本可视化目录存在：`eda_reports/samples/`<br>2. 验证每个类别至少有 N 张样本可视化（N ≥ 3）<br>3. 验证标注可视化正确：<br>   - 分类任务：标签显示正确<br>   - 检测任务：边界框绘制正确<br>   - 分割任务：掩码叠加正确<br>4. 验证图像质量：<br>   - 分辨率适中，细节可见<br>   - 标注颜色对比明显<br>   - 无拉伸变形 |
| **预期结果** | 每个类别有代表性样本可视化，标注显示正确 |
| **通过标准** | 样本覆盖所有类别，标注可视化正确，图像质量合格 |

### 2.5 图像尺寸分布分析验证

| 用例ID | TC-EDA-005 |
|--------|------------|
| **用例名称** | 图像尺寸分布分析验证 |
| **测试目标** | 确认图像尺寸分布已被完整分析 |
| **前置条件** | 数据集图像可正常读取 |
| **测试步骤** | 1. 统计所有图像的宽度和高度<br>2. 计算尺寸分布统计量：<br>   - 最小值、最大值<br>   - 平均值、中位数<br>   - 标准差<br>   - 常见尺寸（Mode）<br>3. 生成尺寸分布可视化：<br>   - 散点图（宽度 vs 高度）<br>   - 直方图（宽度分布、高度分布）<br>4. 识别异常尺寸（过小、过大、长宽比极端）<br>5. 记录尺寸分布分析结果 |
| **预期结果** | 尺寸统计完整，可视化图表存在，异常已识别 |
| **通过标准** | 统计量完整 + 可视化存在 + 异常识别完成 |

### 2.6 图像尺寸异常检测验证

| 用例ID | TC-EDA-006 |
|--------|------------|
| **用例名称** | 图像尺寸异常检测验证 |
| **测试目标** | 确认异常尺寸图像已被正确识别 |
| **前置条件** | TC-EDA-005 通过 |
| **测试步骤** | 1. 定义异常尺寸阈值：<br>   - 过小：宽或高 < 32px<br>   - 过大：宽或高 > 4096px<br>   - 极端长宽比：宽/高 > 10 或 < 0.1<br>2. 筛选出所有异常尺寸图像<br>3. 生成异常图像列表文件<br>4. 统计各类异常的数量 |
| **预期结果** | 异常图像列表完整，异常类型已分类统计 |
| **通过标准** | 异常图像列表文件存在，异常数量统计正确 |

### 2.7 数据集元信息记录验证

| 用例ID | TC-EDA-007 |
|--------|------------|
| **用例名称** | 数据集元信息记录完整性验证 |
| **测试目标** | 确认数据集元信息已完整记录 |
| **前置条件** | TC-EDA-001、TC-EDA-005 通过 |
| **测试步骤** | 1. 检查元信息文件存在：`eda_reports/dataset_meta.json`<br>2. 验证必填字段存在且有效：<br>   - [ ] `dataset_name`: 数据集名称<br>   - [ ] `task_type`: 任务类型<br>   - [ ] `total_samples`: 样本总数<br>   - [ ] `num_classes`: 类别数量<br>   - [ ] `splits`: 各划分的样本数（train/val/test）<br>   - [ ] `avg_width`: 平均宽度<br>   - [ ] `avg_height`: 平均高度<br>   - [ ] `class_distribution`: 类别分布字典<br>   - [ ] `imbalance_ratio`: 不平衡比率<br>   - [ ] `analysis_date`: 分析日期<br>3. 验证数值有效性：<br>   - total_samples > 0<br>   - num_classes > 0<br>   - avg_width > 0, avg_height > 0<br>4. 验证一致性：<br>   - sum(splits) = total_samples<br>   - len(class_distribution) = num_classes |
| **预期结果** | 元信息文件存在，所有必填字段完整有效 |
| **通过标准** | 必填字段完整度 = 100%，数值验证通过，一致性验证通过 |

### 2.8 EDA 报告完整性验证

| 用例ID | TC-EDA-008 |
|--------|------------|
| **用例名称** | EDA 报告完整性验证 |
| **测试目标** | 确认 EDA 报告文档完整且格式正确 |
| **前置条件** | 所有前置测试通过 |
| **测试步骤** | 1. 检查报告文件存在：`eda_reports/EDA_report.md`<br>2. 验证报告结构完整：<br>   - [ ] 概述（数据集基本信息）<br>   - [ ] 类别分布分析<br>   - [ ] 类别不平衡分析<br>   - [ ] 图像尺寸分析<br>   - [ ] 样本可视化展示<br>   - [ ] 异常数据记录<br>   - [ ] 结论与建议<br>3. 验证报告引用的可视化文件存在：<br>   - 检查报告中引用的图片路径<br>   - 验证图片文件存在<br>4. 验证报告内容质量：<br>   - 有明确的结论<br>   - 有针对问题的建议<br>   - 数值数据有表格或图表支持 |
| **预期结果** | 报告结构完整，引用正确，内容有质量 |
| **通过标准** | 结构完整度 = 100%，引用验证通过，内容质量合格 |

---

## 3. 验证方法与步骤

### 3.1 自动化验证脚本示例

```python
# scripts/verify_eda.py

import os
import json
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
from PIL import Image


@dataclass
class EDACheckResult:
    """EDA 检查结果"""
    passed: bool
    errors: List[str]
    warnings: List[str]
    details: Dict[str, Any] = None


def verify_class_distribution(eda_dir: Path) -> EDACheckResult:
    """TC-EDA-001, TC-EDA-002: 验证类别分布统计"""
    errors = []
    warnings = []
    
    meta_file = eda_dir / "dataset_meta.json"
    if not meta_file.exists():
        errors.append("元信息文件不存在: dataset_meta.json")
        return EDACheckResult(False, errors, warnings)
    
    with open(meta_file, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    # 检查类别分布字段
    if "class_distribution" not in meta:
        errors.append("缺少 class_distribution 字段")
    else:
        class_dist = meta["class_distribution"]
        if not class_dist:
            errors.append("class_distribution 为空")
        else:
            # 检查所有类别计数 > 0
            zero_classes = [k for k, v in class_dist.items() if v == 0]
            if zero_classes:
                errors.append(f"存在计数为 0 的类别: {zero_classes}")
    
    # 检查不平衡指标
    if "imbalance_ratio" not in meta:
        warnings.append("缺少 imbalance_ratio 字段，未计算不平衡比率")
    else:
        ir = meta["imbalance_ratio"]
        if ir >= 10:
            warnings.append(f"严重类别不平衡: IR = {ir:.2f}")
        elif ir >= 1.5:
            warnings.append(f"轻度类别不平衡: IR = {ir:.2f}")
    
    passed = len(errors) == 0
    return EDACheckResult(passed, errors, warnings, meta)


def verify_visualization_files(eda_dir: Path) -> EDACheckResult:
    """TC-EDA-003: 验证可视化文件"""
    errors = []
    warnings = []
    details = {"files": [], "missing": []}
    
    # 必需的可视化文件
    required_files = [
        "class_distribution.png",
        "size_distribution.png",
    ]
    
    # 可选的可视化文件
    optional_files = [
        "class_distribution_pie.png",
        "size_scatter.png",
    ]
    
    for f in required_files:
        file_path = eda_dir / f
        if file_path.exists():
            # 验证图像可打开
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    if width < 800 or height < 600:
                        warnings.append(f"{f} 分辨率过低: {width}x{height}")
                    details["files"].append({
                        "name": f,
                        "size": file_path.stat().st_size,
                        "resolution": f"{width}x{height}"
                    })
            except Exception as e:
                errors.append(f"{f} 无法打开: {str(e)}")
        else:
            errors.append(f"缺少必需的可视化文件: {f}")
            details["missing"].append(f)
    
    for f in optional_files:
        file_path = eda_dir / f
        if not file_path.exists():
            warnings.append(f"缺少可选的可视化文件: {f}")
    
    passed = len(errors) == 0
    return EDACheckResult(passed, errors, warnings, details)


def verify_sample_visualization(eda_dir: Path, class_names: List[str]) -> EDACheckResult:
    """TC-EDA-004: 验证样本可视化"""
    errors = []
    warnings = []
    details = {"classes_covered": [], "classes_missing": []}
    
    samples_dir = eda_dir / "samples"
    if not samples_dir.exists():
        errors.append("样本可视化目录不存在: samples/")
        return EDACheckResult(False, errors, warnings)
    
    # 检查每个类别的样本
    for class_name in class_names:
        class_dir = samples_dir / class_name
        if class_dir.exists():
            samples = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
            if len(samples) < 3:
                warnings.append(f"类别 {class_name} 样本数不足 3 张: {len(samples)}")
            details["classes_covered"].append(class_name)
        else:
            errors.append(f"类别 {class_name} 缺少样本可视化目录")
            details["classes_missing"].append(class_name)
    
    passed = len(errors) == 0
    return EDACheckResult(passed, errors, warnings, details)


def verify_size_analysis(eda_dir: Path) -> EDACheckResult:
    """TC-EDA-005, TC-EDA-006: 验证尺寸分析"""
    errors = []
    warnings = []
    details = {}
    
    meta_file = eda_dir / "dataset_meta.json"
    with open(meta_file, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    # 检查尺寸统计字段
    required_fields = ["avg_width", "avg_height", "min_width", "min_height", 
                       "max_width", "max_height"]
    for field in required_fields:
        if field not in meta:
            errors.append(f"缺少尺寸统计字段: {field}")
        elif meta[field] <= 0:
            errors.append(f"{field} 值无效: {meta[field]}")
    
    # 检查异常尺寸文件
    anomaly_file = eda_dir / "size_anomalies.json"
    if anomaly_file.exists():
        with open(anomaly_file, 'r', encoding='utf-8') as f:
            anomalies = json.load(f)
        details["anomaly_count"] = len(anomalies.get("images", []))
        if details["anomaly_count"] > 0:
            warnings.append(f"发现 {details['anomaly_count']} 张异常尺寸图像")
    else:
        warnings.append("缺少异常尺寸记录文件: size_anomalies.json")
    
    passed = len(errors) == 0
    return EDACheckResult(passed, errors, warnings, details)


def verify_meta_info(eda_dir: Path) -> EDACheckResult:
    """TC-EDA-007: 验证元信息完整性"""
    errors = []
    warnings = []
    details = {"missing_fields": [], "invalid_values": [], "inconsistencies": []}
    
    meta_file = eda_dir / "dataset_meta.json"
    if not meta_file.exists():
        errors.append("元信息文件不存在: dataset_meta.json")
        return EDACheckResult(False, errors, warnings)
    
    with open(meta_file, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    # 检查必填字段
    required_fields = {
        "dataset_name": str,
        "task_type": str,
        "total_samples": int,
        "num_classes": int,
        "splits": dict,
        "avg_width": (int, float),
        "avg_height": (int, float),
        "class_distribution": dict,
        "analysis_date": str
    }
    
    for field, expected_type in required_fields.items():
        if field not in meta:
            errors.append(f"缺少必填字段: {field}")
            details["missing_fields"].append(field)
        elif not isinstance(meta[field], expected_type):
            errors.append(f"字段 {field} 类型错误: 期望 {expected_type}, 实际 {type(meta[field])}")
            details["invalid_values"].append(field)
    
    # 验证数值有效性
    if "total_samples" in meta and meta["total_samples"] <= 0:
        errors.append("total_samples 必须 > 0")
    
    if "num_classes" in meta and meta["num_classes"] <= 0:
        errors.append("num_classes 必须 > 0")
    
    # 验证一致性
    if "splits" in meta and "total_samples" in meta:
        splits_sum = sum(meta["splits"].values())
        if splits_sum != meta["total_samples"]:
            errors.append(f"splits 总和 ({splits_sum}) 与 total_samples ({meta['total_samples']}) 不一致")
            details["inconsistencies"].append("splits_sum != total_samples")
    
    if "class_distribution" in meta and "num_classes" in meta:
        actual_classes = len(meta["class_distribution"])
        if actual_classes != meta["num_classes"]:
            errors.append(f"class_distribution 类别数 ({actual_classes}) 与 num_classes ({meta['num_classes']}) 不一致")
            details["inconsistencies"].append("class_dist_count != num_classes")
    
    passed = len(errors) == 0
    return EDACheckResult(passed, errors, warnings, details)


def verify_eda_report(eda_dir: Path) -> EDACheckResult:
    """TC-EDA-008: 验证 EDA 报告完整性"""
    errors = []
    warnings = []
    details = {"sections": {}, "broken_links": []}
    
    report_file = eda_dir / "EDA_report.md"
    if not report_file.exists():
        errors.append("EDA 报告不存在: EDA_report.md")
        return EDACheckResult(False, errors, warnings)
    
    with open(report_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查必需的章节
    required_sections = [
        ("概述", ["概述", "数据集概述", "基本信息"]),
        ("类别分布", ["类别分布", "类别分析"]),
        ("不平衡分析", ["不平衡", "类别不平衡"]),
        ("尺寸分析", ["尺寸", "分辨率", "图像尺寸"]),
        ("样本可视化", ["样本", "可视化", "样本展示"]),
        ("结论与建议", ["结论", "建议", "总结"]),
    ]
    
    for section_name, keywords in required_sections:
        found = any(kw in content for kw in keywords)
        details["sections"][section_name] = found
        if not found:
            errors.append(f"报告缺少章节: {section_name}")
    
    # 检查引用的图片链接
    import re
    image_links = re.findall(r'!\[.*?\\]\((.*?)\)', content)
    for link in image_links:
        # 处理相对路径
        if not link.startswith(('http://', 'https://')):
            image_path = eda_dir / link
            if not image_path.exists():
                errors.append(f"报告引用的图片不存在: {link}")
                details["broken_links"].append(link)
    
    passed = len(errors) == 0
    return EDACheckResult(passed, errors, warnings, details)


def run_all_eda_verifications(eda_dir: str) -> Dict[str, Any]:
    """运行所有 EDA 验证"""
    eda_path = Path(eda_dir)
    results = {}
    
    # 获取类别名称（从元信息文件）
    meta_file = eda_path / "dataset_meta.json"
    class_names = []
    if meta_file.exists():
        with open(meta_file, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            class_names = list(meta.get("class_distribution", {}).keys())
    
    # 执行验证
    results["TC-EDA-001_002"] = verify_class_distribution(eda_path)
    results["TC-EDA-003"] = verify_visualization_files(eda_path)
    results["TC-EDA-004"] = verify_sample_visualization(eda_path, class_names)
    results["TC-EDA-005_006"] = verify_size_analysis(eda_path)
    results["TC-EDA-007"] = verify_meta_info(eda_path)
    results["TC-EDA-008"] = verify_eda_report(eda_path)
    
    # 汇总结果
    all_passed = all(r.passed for r in results.values())
    all_errors = []
    all_warnings = []
    for tc_id, result in results.items():
        for err in result.errors:
            all_errors.append(f"[{tc_id}] {err}")
        for warn in result.warnings:
            all_warnings.append(f"[{tc_id}] {warn}")
    
    results["summary"] = {
        "passed": all_passed,
        "total_errors": len(all_errors),
        "total_warnings": len(all_warnings),
        "errors": all_errors,
        "warnings": all_warnings
    }
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EDA 验证脚本")
    parser.add_argument("--eda-dir", default="eda_reports", help="EDA 报告目录")
    args = parser.parse_args()
    
    results = run_all_eda_verifications(args.eda_dir)
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    
    # 返回退出码
    exit(0 if results["summary"]["passed"] else 1)
```

### 3.2 验证执行流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         EDA 验证流程                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────┐                                                   │
│  │ 前置条件检查      │                                                   │
│  │ (数据集已下载)    │                                                   │
│  └────────┬─────────┘                                                   │
│           │                                                             │
│           ▼                                                             │
│  ┌──────────────────┐    ┌──────────────────┐                         │
│  │ TC-EDA-001       │───▶│ TC-EDA-002       │                         │
│  │ 类别分布统计     │    │ 类别不平衡量化   │                         │
│  └────────┬─────────┘    └────────┬─────────┘                         │
│           │                       │                                    │
│           │                       ▼                                    │
│           │              ┌──────────────────┐                        │
│           │              │ TC-EDA-003       │                        │
│           │              │ 分布可视化验证   │                        │
│           │              └────────┬─────────┘                        │
│           │                       │                                    │
│           ▼                       ▼                                    │
│  ┌──────────────────┐    ┌──────────────────┐                         │
│  │ TC-EDA-004       │    │ TC-EDA-005       │                         │
│  │ 样本可视化验证   │    │ 尺寸分布分析     │                         │
│  └────────┬─────────┘    └────────┬─────────┘                         │
│           │                       │                                    │
│           │                       ▼                                    │
│           │              ┌──────────────────┐                        │
│           │              │ TC-EDA-006       │                        │
│           │              │ 尺寸异常检测     │                        │
│           │              └────────┬─────────┘                        │
│           │                       │                                    │
│           └───────────┬───────────┘                                    │
│                       │                                                  │
│                       ▼                                                  │
│              ┌──────────────────┐                                       │
│              │ TC-EDA-007       │                                       │
│              │ 元信息记录验证   │                                       │
│              └────────┬─────────┘                                       │
│                       │                                                  │
│                       ▼                                                  │
│              ┌──────────────────┐                                       │
│              │ TC-EDA-008       │                                       │
│              │ EDA 报告验证     │                                       │
│              └────────┬─────────┘                                       │
│                       │                                                  │
│                       ▼                                                  │
│              ┌──────────────────┐                                       │
│              │ 生成验证报告     │                                       │
│              └──────────────────┘                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 通过标准

### 4.1 各测试用例通过标准

| 用例ID | 通过标准 | 阻塞级别 |
|--------|---------|---------|
| TC-EDA-001 | 类别覆盖度 = 100%，所有类别计数 > 0 | 🔴 阻塞 |
| TC-EDA-002 | IR 值已计算，不平衡级别已判定并记录 | 🔴 阻塞 |
| TC-EDA-003 | 可视化文件存在 + 可打开 + 分辨率 ≥ 800×600 | 🔴 阻塞 |
| TC-EDA-004 | 每个类别 ≥ 3 张样本，标注显示正确 | 🔴 阻塞 |
| TC-EDA-005 | 尺寸统计量完整（6项以上） + 可视化存在 | 🔴 阻塞 |
| TC-EDA-006 | 异常尺寸图像已识别并记录 | 🟡 警告 |
| TC-EDA-007 | 必填字段完整度 = 100%，数值有效，一致性验证通过 | 🔴 阻塞 |
| TC-EDA-008 | 报告结构完整度 = 100%，引用验证通过 | 🔴 阻塞 |

### 4.2 总体通过条件

**必须满足：**
- 所有 🔴 阻塞级别测试用例通过
- TC-EDA-001 ~ TC-EDA-008 全部通过（除 TC-EDA-006 可为警告）

**建议满足：**
- TC-EDA-006 无异常图像，或异常图像有处理建议
- 无警告信息

### 4.3 验证报告模板

```markdown
# EDA 验证报告

**数据集名称：** [名称]
**验证日期：** [日期]
**验证人：** [姓名]

## 验证结果摘要

| 测试用例 | 状态 | 备注 |
||---------|------|------|
| TC-EDA-001 | ✅ 通过 | 类别数：100，覆盖率：100% |
| TC-EDA-002 | ✅ 通过 | IR = 5.23，轻度不平衡 |
| TC-EDA-003 | ✅ 通过 | 2 张分布图已生成 |
| TC-EDA-004 | ✅ 通过 | 每类 5 张样本 |
| TC-EDA-005 | ✅ 通过 | 平均尺寸：224×224 |
| TC-EDA-006 | ⚠️ 警告 | 发现 3 张异常尺寸图像 |
| TC-EDA-007 | ✅ 通过 | 10 项元信息完整 |
| TC-EDA-008 | ✅ 通过 | 报告结构完整 |

## 类别分布统计

| 类别 | 样本数 | 占比 |
|------|--------|------|
| ... | ... | ... |

## 不平衡分析

- 最大类样本数：[N]
- 最小类样本数：[N]
- 不平衡比率（IR）：[值]
- 不平衡级别：[平衡/轻度不平衡/严重不平衡]

## 异常数据

### 尺寸异常图像

| 文件名 | 宽度 | 高度 | 异常类型 |
|--------|------|------|----------|
| ... | ... | ... | 过小/过大/极端长宽比 |

## 问题记录

| 问题编号 | 描述 | 严重程度 | 处理状态 |
|----------|------|----------|----------|
| ... | ... | ... | ... |

## 总体结论

✅ **验证通过** / ❌ **验证失败**

## 建议

1. ...
2. ...
```

---

## 5. 自动化验证建议

### 5.1 可自动化的验证项

| 用例ID | 自动化程度 | 自动化方式 |
|--------|-----------|-----------|
| TC-EDA-001 | ✅ 完全自动化 | Python 脚本遍历标注文件统计 |
| TC-EDA-002 | ✅ 完全自动化 | 计算 IR、基尼系数、熵 |
| TC-EDA-003 | ✅ 完全自动化 | 文件存在性检查 + PIL 图像验证 |
| TC-EDA-004 | 🟡 半自动化 | 自动生成样本可视化，人工抽查质量 |
| TC-EDA-005 | ✅ 完全自动化 | PIL/cv2 读取尺寸统计 |
| TC-EDA-006 | ✅ 完全自动化 | 阈值筛选 + 异常列表生成 |
| TC-EDA-007 | ✅ 完全自动化 | JSON 字段检查 + 数值验证 |
| TC-EDA-008 | 🟡 半自动化 | 章节检测自动，内容质量人工审查 |

### 5.2 需要手动验证的项

| 验证项 | 手动原因 | 建议 |
|--------|---------|------|
| 样本可视化质量 | 需人工判断标注是否正确显示 | 抽查每类 3-5 张 |
| 报告内容质量 | 需人工判断结论是否合理 | 审查报告结论和建议 |
| 类别标签正确性 | 需人工确认标签含义与图像匹配 | 抽查部分样本 |
| 不平衡问题严重性判断 | 需结合业务场景判断 | 与业务方沟通确认 |

### 5.3 CI/CD 集成建议

```yaml
# .github/workflows/verify_eda.yml
name: EDA Verification

on:
  workflow_dispatch:
    inputs:
      dataset_name:
        description: 'Dataset name to verify'
        required: true
      eda_dir:
        description: 'EDA report directory'
        required: false
        default: 'eda_reports'

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install pillow pytest matplotlib numpy

      - name: Run EDA verification
        run: |
          python scripts/verify_eda.py --eda-dir ${{ inputs.eda_dir }}

      - name: Check meta info
        run: |
          python scripts/check_meta_info.py --eda-dir ${{ inputs.eda_dir }}

      - name: Upload verification report
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: eda-verification-report
          path: reports/eda_verification_*.json

      - name: Comment on PR (if applicable)
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const report = JSON.parse(fs.readFileSync('reports/eda_verification_latest.json'));
            const summary = report.summary;
            const body = `## EDA 验证结果\n\n` +
              `**状态**: ${summary.passed ? '✅ 通过' : '❌ 失败'}\n` +
              `**错误数**: ${summary.total_errors}\n` +
              `**警告数**: ${summary.total_warnings}\n`;
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: body
            });
```

---

## 6. 风险点与注意事项

### 6.1 技术风险

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|---------|
| 大数据集统计耗时过长 | 高 | 中 | 使用采样策略，设置超时机制 |
| 图像读取失败导致统计不完整 | 中 | 高 | 异常捕获，记录失败文件，统计失败率 |
| 内存不足导致处理失败 | 中 | 高 | 分批处理，使用生成器而非列表 |
| 可视化生成失败 | 低 | 中 | 捕获异常，生成占位图并记录错误 |
| 类别名称编码问题 | 低 | 中 | 统一使用 UTF-8 编码处理 |

### 6.2 数据风险

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|---------|
| 类别标注错误 | 中 | 高 | 抽查验证，记录可疑样本 |
| 图像格式异常 | 中 | 中 | 支持多种格式，记录异常格式 |
| 标注文件损坏 | 低 | 高 | 备份原始数据，验证解析结果 |
| 类别定义不一致 | 中 | 高 | 与官方类别列表对比验证 |

### 6.3 流程风险

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|---------|
| EDA 报告与实际数据不符 | 中 | 高 | 自动化生成，减少人工干预 |
| 可视化图表误导 | 低 | 高 | 人工审查关键图表 |
| 元信息更新不及时 | 中 | 中 | 数据变更后自动触发 EDA 重新执行 |

### 6.4 注意事项

1. **执行前检查**
   - 确认数据集已完整下载（前置任务 TC-003 通过）
   - 确认有足够的磁盘空间存储可视化结果
   - 确认 Python 环境已安装必要依赖（PIL、matplotlib、numpy）

2. **执行中监控**
   - 监控内存使用，大数据集采用分批处理
   - 记录处理进度，支持断点续传
   - 保存中间结果，避免重复计算

3. **输出质量控制**
   - 可视化图表需有清晰标题、标签、图例
   - 统计数值需保留适当精度
   - 报告中的结论需有数据支撑

4. **特殊情况处理**
   - 单类别数据集：跳过不平衡分析，标注为"不适用"
   - 空数据集：立即报错，不执行后续分析
   - 类别数量过多（>100）：使用 Top-K 展示，其余归类为"其他"

5. **结果存档**
   - EDA 报告应存放在项目文档目录
   - 原始统计数据应保存为 JSON 格式
   - 可视化图表应保存为高分辨率 PNG

---

## 附录：EDA 输出目录结构

```
eda_reports/
├── EDA_report.md                    # 主报告文档
├── dataset_meta.json                # 数据集元信息
├── class_distribution.png           # 类别分布柱状图
├── class_distribution_pie.png       # 类别分布饼图（可选）
├── size_distribution.png            # 尺寸分布直方图
├── size_scatter.png                 # 尺寸散点图
├── size_anomalies.json              # 异常尺寸图像列表
├── samples/                         # 样本可视化目录
│   ├── class_1/
│   │   ├── sample_001.png
│   │   ├── sample_002.png
│   │   └── sample_003.png
│   ├── class_2/
│   │   └── ...
│   └── ...
└── verification/                    # 验证结果（可选）
    ├── verification_report.json
    └── verification_summary.md
```

## 附录：元信息文件模板

```json
{
  "dataset_name": "ExampleDataset",
  "task_type": "image_classification",
  "version": "1.0",
  "analysis_date": "2026-06-02",
  
  "total_samples": 50000,
  "num_classes": 10,
  "splits": {
    "train": 40000,
    "val": 5000,
    "test": 5000
  },
  
  "avg_width": 224.5,
  "avg_height": 224.3,
  "min_width": 32,
  "min_height": 32,
  "max_width": 512,
  "max_height": 512,
  "median_width": 224,
  "median_height": 224,
  
  "class_distribution": {
    "class_0": 5000,
    "class_1": 4800,
    "class_2": 5200,
    "class_3": 5000,
    "class_4": 4900,
    "class_5": 5100,
    "class_6": 5000,
    "class_7": 5050,
    "class_8": 4950,
    "class_9": 5000
  },
  
  "imbalance_ratio": 1.08,
  "gini_coefficient": 0.02,
  "class_entropy": 3.32,
  
  "anomaly_count": {
    "size_too_small": 2,
    "size_too_large": 0,
    "extreme_aspect_ratio": 1
  }
}
```