# 数据集选择与获取 — 验证测试设计

> **来源任务：** docs/superpowers/tasks/02_数据准备/01_数据集选择与获取.md
> **生成日期：** 2026/06/02

---

## 1. 验证目标概述

本验证方案针对「数据集选择与获取」任务，确保以下验收标准得到完整验证：

| 验收标准 | 验证目标 | 验证方式 |
|---------|---------|---------|
| 数据集下载完成 | 确认数据集文件已完整下载到指定目录 | 自动化脚本检查 |
| 文件数量和格式验证通过 | 确认文件数量与官方数据集一致，格式正确 | 自动化脚本 + 校验和验证 |
| 数据目录结构清晰 | 确认目录组织符合规范，说明文档完整 | 结构检查 + 文档审查 |

---

## 2. 测试用例设计

### 2.1 任务类型明确性验证

| 用例ID | TC-001 |
|--------|--------|
| **用例名称** | 任务类型定义验证 |
| **测试目标** | 确认任务类型已明确声明且与后续步骤一致 |
| **前置条件** | 项目需求文档已存在 |
| **测试步骤** | 1. 读取项目配置文件（如 `config.yaml` 或 `README.md`）<br>2. 确认任务类型字段存在且值有效<br>3. 验证任务类型值为以下之一：`image_classification`、`object_detection`、`semantic_segmentation` |
| **预期结果** | 任务类型字段存在，值有效，且与数据集选择一致 |
| **通过标准** | 任务类型明确、值合法、与数据集匹配 |

### 2.2 数据集选择验证

| 用例ID | TC-002 |
|--------|--------|
| **用例名称** | 数据集选择合理性验证 |
| **测试目标** | 确认所选数据集与任务类型匹配 |
| **前置条件** | 任务类型已确定（TC-001 通过） |
| **测试步骤** | 1. 获取已选数据集名称<br>2. 验证数据集与任务类型的映射关系：<br>   - 图像分类 → ImageNet / CIFAR-10 / CIFAR-100<br>   - 目标检测 → COCO / Pascal VOC<br>   - 语义分割 → Cityscapes / ADE20K<br>3. 确认数据集来源为官方或可信镜像 |
| **预期结果** | 数据集与任务类型匹配，来源可靠 |
| **通过标准** | 映射关系正确，来源可追溯 |

### 2.3 数据集下载验证

| 用例ID | TC-003 |
|--------|--------|
| **用例名称** | 数据集下载完整性验证 |
| **测试目标** | 确认数据集文件已完整下载 |
| **前置条件** | 数据集来源已确定 |
| **测试步骤** | 1. 检查 `data/` 目录是否存在<br>2. 验证数据集子目录结构：<br>   ```
     data/
   ├── {dataset_name}/
   │   ├── train/
   │   ├── val/
   │   ├── test/ (如适用)
   │   └── annotations/ (如适用)
   ```<br>3. 统计各目录文件数量<br>4. 与官方公布的数据集规模对比 |
| **预期结果** | 目录结构正确，文件数量与官方一致 |
| **通过标准** | 文件数量误差 ≤ 0%，目录结构符合预期 |

### 2.4 文件格式验证

| 用例ID | TC-004 |
|--------|--------|
| **用例名称** | 数据文件格式验证 |
| **测试目标** | 确认所有数据文件格式正确且可读 |
| **前置条件** | 数据集已下载（TC-003 通过） |
| **测试步骤** | 1. 遍历图像目录，检查文件扩展名<br>2. 验证图像文件可正常解码（使用 PIL/OpenCV）<br>3. 检查标注文件格式（JSON/XML/TXT）<br>4. 验证标注文件可正常解析 |
| **预期结果** | 所有图像文件可解码，标注文件可解析 |
| **通过标准** | 损坏文件数量 = 0，解析成功率 = 100% |

### 2.5 标签一致性验证

| 用例ID | TC-005 |
|--------|--------|
| **用例名称** | 标签一致性验证 |
| **测试目标** | 确认标注与图像一一对应，标签格式一致 |
| **前置条件** | 文件格式验证通过（TC-004 通过） |
| **测试步骤** | 1. 提取所有图像文件名（不含扩展名）<br>2. 提取所有标注中的图像引用<br>3. 对比两个集合，确认一一对应<br>4. 检查标签值范围是否合法<br>5. 验证类别数量与官方一致 |
| **预期结果** | 图像与标注一一对应，标签值合法 |
| **通过标准** | 孤儿图像 = 0，孤儿标注 = 0，类别数量一致 |

### 2.6 数据说明文档验证

| 用例ID | TC-006 |
|--------|--------|
| **用例名称** | 数据说明文档完整性验证 |
| **测试目标** | 确认数据说明文档存在且信息完整 |
| **前置条件** | 数据集已下载（TC-003 通过） |
| **测试步骤** | 1. 检查 `data/{dataset_name}/README.md` 或 `data/DATASET.md` 是否存在<br>2. 验证文档包含以下必填项：<br>   - [ ] 数据集名称<br>   - [ ] 数据集来源/版本<br>   - [ ] 下载日期<br>   - [ ] 文件数量统计<br>   - [ ] 目录结构说明<br>   - [ ] 类别列表（如适用）<br>   - [ ] 许可证信息<br>   - [ ] 引用格式（Citation） |
| **预期结果** | 文档存在，所有必填项完整 |
| **通过标准** | 必填项完整度 = 100% |

### 2.7 校验和验证（可选增强）

| 用例ID | TC-007 |
|--------|--------|
| **用例名称** | 数据集校验和验证 |
| **测试目标** | 通过校验和确保数据完整性 |
| **前置条件** | 数据集提供官方校验和 |
| **测试步骤** | 1. 获取官方 MD5/SHA256 校验和<br>2. 计算本地文件的校验和<br>3. 对比校验和是否一致 |
| **预期结果** | 所有校验和匹配 |
| **通过标准** | 校验和匹配率 = 100% |

---

## 3. 验证方法与步骤

### 3.1 自动化验证脚本示例

```python
# scripts/verify_dataset.py

import os
import hashlib
from pathlib import Path
from PIL import Image
import json

def verify_directory_structure(data_dir: Path, dataset_name: str) -> dict:
    """验证目录结构"""
    result = {"passed": True, "errors": []}

    expected_dirs = ["train", "val"]
    dataset_path = data_dir / dataset_name

    if not dataset_path.exists():
        result["passed"] = False
        result["errors"].append(f"数据集目录不存在: {dataset_path}")
        return result

    for dir_name in expected_dirs:
        if not (dataset_path / dir_name).exists():
            result["passed"] = False
            result["errors"].append(f"缺少目录: {dir_name}")

    return result

def verify_image_files(image_dir: Path) -> dict:
    """验证图像文件可读性"""
    result = {"total": 0, "valid": 0, "corrupted": []}

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [f for f in image_dir.rglob("*")
                   if f.suffix.lower() in valid_extensions]

    result["total"] = len(image_files)

    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                img.verify()
            result["valid"] += 1
        except Exception as e:
            result["corrupted"].append({
                "file": str(img_path),
                "error": str(e)
            })

    return result

def verify_annotation_consistency(image_dir: Path, annotation_file: Path) -> dict:
    """验证标注与图像一致性"""
    result = {"orphans": [], "mismatched": []}

    # 获取图像文件名集合
    image_names = {f.stem for f in image_dir.glob("*.jpg")}
    image_names.update({f.stem for f in image_dir.glob("*.png")})

    # 获取标注中的图像引用
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    annotation_images = {img["file_name"].split('.')[0]
                        for img in annotations.get("images", [])}

    # 找出孤儿
    orphan_images = image_names - annotation_images
    orphan_annotations = annotation_images - image_names

    result["orphan_images"] = list(orphan_images)
    result["orphan_annotations"] = list(orphan_annotations)

    return result

def run_all_verifications(config: dict) -> dict:
    """运行所有验证"""
    results = {}

    data_dir = Path(config["data_dir"])
    dataset_name = config["dataset_name"]

    # TC-003: 目录结构验证
    results["directory_structure"] = verify_directory_structure(data_dir, dataset_name)

    # TC-004: 图像文件验证
    image_dir = data_dir / dataset_name / "train"
    if image_dir.exists():
        results["image_files"] = verify_image_files(image_dir)

    # TC-005: 标注一致性验证
    annotation_file = data_dir / dataset_name / "annotations" / "instances_train.json"
    if annotation_file.exists():
        results["annotation_consistency"] = verify_annotation_consistency(
            image_dir, annotation_file
        )

    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--dataset-name", required=True)
    args = parser.parse_args()

    config = {
        "data_dir": args.data_dir,
        "dataset_name": args.dataset_name
    }

    results = run_all_verifications(config)
    print(json.dumps(results, indent=2, ensure_ascii=False))
```

### 3.2 验证执行流程

```
┌─────────────────────────────────────────────────────────────┐
│                    数据集验证流程                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ TC-001   │───▶│ TC-002   │───▶│ TC-003   │              │
│  │ 任务类型 │    │ 数据集   │    │ 下载完整 │              │
│  │ 定义验证 │    │ 选择验证 │    │ 性验证   │              │
│  └──────────┘    └──────────┘    └────┬─────┘              │
│                                        │                    │
│                         ┌──────────────┴──────────────┐     │
│                         ▼                             ▼     │
│                  ┌──────────┐                  ┌──────────┐ │
│                  │ TC-004   │                  │ TC-006   │ │
│                  │ 文件格式 │                  │ 文档完整 │ │
│                  │ 验证     │                  │ 性验证   │ │
│                  └────┬─────┘                  └──────────┘ │
│                       │                                     │
│                       ▼                                     │
│                  ┌──────────┐                               │
│                  │ TC-005   │                               │
│                  │ 标签一致 │                               │
│                  │ 性验证   │                               │
│                  └────┬─────┘                               │
│                       │                                     │
│                       ▼                                     │
│                  ┌──────────┐                               │
│                  │ TC-007   │ (可选)                        │
│                  │ 校验和   │                               │
│                  │ 验证     │                               │
│                  └──────────┘                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. 通过标准

### 4.1 各测试用例通过标准

| 用例ID | 通过标准 | 阻塞级别 |
|--------|---------|---------|
| TC-001 | 任务类型字段存在且值合法 | 🔴 阻塞 |
| TC-002 | 数据集与任务类型匹配 | 🔴 阻塞 |
| TC-003 | 文件数量 ≥ 官方数量的 100% | 🔴 阻塞 |
| TC-004 | 损坏文件数量 = 0 | 🔴 阻塞 |
| TC-005 | 孤儿图像 = 0，孤儿标注 = 0 | 🔴 阻塞 |
| TC-006 | 必填项完整度 = 100% | 🟡 警告 |
| TC-007 | 校验和匹配率 = 100% | 🟢 可选 |

### 4.2 总体通过条件

**必须满足：**
- 所有 🔴 阻塞级别测试用例通过
- TC-003、TC-004、TC-005 必须通过

**建议满足：**
- TC-006 文档完整性验证通过
- TC-007 校验和验证通过（如官方提供）

### 4.3 验证报告模板

```markdown
# 数据集验证报告

**数据集名称：** [名称]
**验证日期：** [日期]
**验证人：** [姓名]

## 验证结果摘要

| 测试用例 | 状态 | 备注 |
|---------|------|------|
| TC-001 | ✅ 通过 | 任务类型：image_classification |
| TC-002 | ✅ 通过 | 数据集：ImageNet |
| TC-003 | ✅ 通过 | 训练集：1,281,167 张，验证集：50,000 张 |
| TC-004 | ✅ 通过 | 损坏文件：0 |
| TC-005 | ✅ 通过 | 孤儿文件：0 |
| TC-006 | ✅ 通过 | 文档完整 |
| TC-007 | ⏭️ 跳过 | 无官方校验和 |

## 总体结论

✅ **验证通过**

## 问题记录

| 问题编号 | 描述 | 严重程度 | 处理状态 |
|---------|------|---------|---------|
| - | - | - | - |
```

---

## 5. 自动化验证建议

### 5.1 可自动化的验证项

| 用例ID | 自动化程度 | 自动化方式 |
|--------|-----------|-----------|
| TC-001 | ✅ 完全自动化 | 配置文件解析 + 值校验 |
| TC-002 | ✅ 完全自动化 | 映射表查询 |
| TC-003 | ✅ 完全自动化 | 文件计数 + 结构检查 |
| TC-004 | ✅ 完全自动化 | PIL/OpenCV 图像解码测试 |
| TC-005 | ✅ 完全自动化 | 文件名集合运算 |
| TC-006 | 🟡 半自动化 | 模板检查 + 人工审查内容 |
| TC-007 | ✅ 完全自动化 | MD5/SHA256 计算 + 对比 |

### 5.2 需要手动验证的项

| 验证项 | 手动原因 | 建议 |
|--------|---------|------|
| 数据集来源可信度 | 需人工判断 | 建立可信来源白名单 |
| 许可证合规性 | 需法律判断 | 法务审核 |
| 数据质量抽样 | 需人工目视 | 抽样检查 100 张图像 |

### 5.3 CI/CD 集成建议

```yaml
# .github/workflows/verify-dataset.yml
name: Dataset Verification

on:
  workflow_dispatch:
    inputs:
      dataset_name:
        description: 'Dataset name to verify'
        required: true

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
          pip install pillow pytest

      - name: Run dataset verification
        run: |
          python scripts/verify_dataset.py --dataset-name ${{ inputs.dataset_name }}

      - name: Generate report
        if: always()
        run: |
          python scripts/generate_report.py --output reports/dataset_verification.md
```

---

## 6. 风险点与注意事项

### 6.1 技术风险

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|---------|
| 下载中断导致文件不完整 | 中 | 高 | 使用断点续传工具，验证文件数量 |
| 图像损坏无法识别 | 低 | 高 | TC-004 覆盖，需逐个解码验证 |
| 标注格式版本不兼容 | 中 | 高 | 检查标注文件版本，使用兼容解析器 |
| 磁盘空间不足 | 中 | 高 | 下载前检查可用空间，预留 2x 数据集大小 |
| 文件名编码问题 | 低 | 中 | 统一使用 UTF-8 编码处理文件名 |

### 6.2 流程风险

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|---------|
| 数据集许可证限制 | 中 | 高 | 下载前确认许可证，记录许可证信息 |
| 数据集版本更新 | 低 | 中 | 记录数据集版本号，使用固定版本链接 |
| 网络问题导致下载失败 | 高 | 中 | 使用镜像站，支持断点续传 |

### 6.3 数据质量风险

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|---------|
| 标注错误 | 低 | 高 | 抽样验证标注正确性 |
| 类别不平衡 | 中 | 中 | 统计类别分布，记录在文档中 |
| 图像质量差异大 | 中 | 低 | 统计分辨率分布，记录异常值 |

### 6.4 注意事项

1. **下载前**
   - 确认磁盘空间 ≥ 数据集大小的 2 倍
   - 确认网络连接稳定
   - 记录数据集官方 URL 和版本

2. **下载中**
   - 使用支持断点续传的工具（如 `wget -c`、`aria2`）
   - 监控下载进度和速度
   - 保留下载日志

3. **下载后**
   - 立即运行完整验证脚本
   - 生成验证报告并存档
   - 创建数据说明文档

4. **长期维护**
   - 定期检查数据目录是否被意外修改
   - 保留原始数据备份
   - 记录任何数据处理操作

---

## 附录：验证脚本目录结构

```
scripts/
├── verify_dataset.py          # 主验证脚本
├── verify_image.py            # 图像验证模块
├── verify_annotation.py       # 标注验证模块
├── generate_report.py         # 报告生成脚本
└── utils/
    ├── file_utils.py          # 文件操作工具
    └── checksum_utils.py      # 校验和工具

reports/
└── dataset_verification.md    # 验证报告输出
```