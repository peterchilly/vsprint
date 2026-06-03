# 数据预处理与增强 — 验证测试设计

> **来源任务：** docs/superpowers/tasks/02_数据准备/03_数据预处理与增强.md
> **生成日期：** 2026-06-02

---

## 1. 验证目标概述

本文档针对「数据预处理与增强」任务设计完整的测试验证方案，确保以下验收标准得到严格验证：

| 验收标准 | 验证方法 | 验证类型 |
|---------|---------|---------|
| 训练/验证 Transform pipeline 完成 | 单元测试 + 集成测试 | 自动化 |
| 增强效果可视化验证 | 可视化脚本输出检查 | 半自动 |
| 增强参数可通过配置调整 | 配置文件解析测试 | 自动化 |

---

## 2. 测试用例设计

### 2.1 基础处理测试用例

#### TC-BASE-001: 图像尺寸统一验证

| 属性 | 值 |
|-----|-----|
| **测试ID** | TC-BASE-001 |
| **测试项** | 统一图像尺寸/分辨率 |
| **前置条件** | 存在多种尺寸的测试图像集（含非标准比例：正方形、宽图、高图） |
| **输入数据** | 1组含不同尺寸的图像（如 512x512, 1024x768, 800x1200, 256x256） |
| **测试步骤** | 1. 加载测试图像集<br>2. 执行 Resize Transform 到目标尺寸（如 224x224）<br>3. 遍历检查每张输出图像尺寸 |
| **预期结果** | 所有输出图像尺寸均为 (3, 224, 224) 的 Tensor |
| **通过标准** | 100% 图像尺寸正确，Tensor shape 符合预期 |

#### TC-BASE-002: 归一化参数验证

| 属性 | 值 |
|-----|-----|
| **测试ID** | TC-BASE-002 |
| **测试项** | ImageNet mean/std 归一化 |
| **前置条件** | 已定义 ImageNet mean/std 常量 |
| **输入数据** | 单张已知像素值的测试图像（如全 128 灰度图或纯色图） |
| **测试步骤** | 1. 创建已知像素值的测试图像<br>2. 执行归一化 Transform<br>3. 手动计算预期输出：`(pixel/255 - mean) / std`<br>4. 对比实际输出与预期值 |
| **预期结果** | 归一化后的 Tensor 值与手动计算结果误差 < 1e-6 |
| **通过标准** | 最大误差 < 1e-6 |

#### TC-BASE-003: Tensor 转换验证

| 属性 | 值 |
|-----|-----|
| **测试ID** | TC-BASE-003 |
| **测试项** | 图像转换为 Tensor |
| **前置条件** | 无 |
| **输入数据** | PIL Image 或 numpy array 格式图像 |
| **测试步骤** | 1. 加载 PIL/numpy 格式图像<br>2. 执行 ToTensor Transform<br>3. 检查输出类型和属性 |
| **预期结果** | 输出为 torch.Tensor，dtype=torch.float32，值范围 [0.0, 1.0] |
| **通过标准** | 类型正确，值范围符合预期，shape 为 (C, H, W) |

---

### 2.2 训练集增强测试用例

#### TC-AUG-001: 随机水平翻转验证

| 属性 | 值 |
|-----|-----|
| **测试ID** | TC-AUG-001 |
| **测试项** | RandomHorizontalFlip 功能与概率 |
| **前置条件** | 设置固定随机种子 |
| **输入数据** | 已知内容的非对称测试图像（如数字 "6" 或箭头图像） |
| **测试步骤** | 1. 设置 p=0.5，运行 N=1000 次翻转<br>2. 统计翻转次数<br>3. 用 p=1.0 验证必定翻转<br>4. 用 p=0.0 验证不翻转<br>5. 检查翻转后图像内容正确性（左右镜像） |
| **预期结果** | p=0.5 时翻转次数约 500±50（95% 置信区间）；p=1.0 必翻转；p=0.0 必不翻转 |
| **通过标准** | 概率分布符合设定，翻转操作语义正确 |

#### TC-AUG-002: 随机裁剪与 Resize 验证

| 属性 | 值 |
|-----|-----|
| **测试ID** | TC-AUG-002 |
| **测试项** | RandomResizedCrop 功能 |
| **前置条件** | 设置固定随机种子 |
| **输入数据** | 已知尺寸的测试图像（如 512x512） |
| **测试步骤** | 1. 设置 scale=(0.08, 1.0), ratio=(0.75, 1.33)<br>2. 执行 100 次裁剪<br>3. 检查每次输出尺寸<br>4. 验证裁剪区域在原图范围内 |
| **预期结果** | 输出尺寸始终为目标尺寸（如 224x224）；裁剪区域在有效范围内 |
| **通过标准** | 100% 输出尺寸正确，无越界裁剪 |

#### TC-AUG-003: 色彩抖动验证

| 属性 | 值 |
|-----|-----|
| **测试ID** | TC-AUG-003 |
| **测试项** | ColorJitter 功能与参数范围 |
| **前置条件** | 设置固定随机种子 |
| **输入数据** | 纯色测试图像（R=100, G=150, B=200） |
| **测试步骤** | 1. 设置 brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1<br>2. 执行 100 次抖动<br>3. 记录输出图像的色彩统计值<br>4. 验证变化在参数范围内 |
| **预期结果** | 色彩变化在预设范围内，不会产生极端值 |
| **通过标准** | 亮度变化 ±20%，对比度/饱和度变化符合设定，hue 变化在 ±0.1 |

#### TC-AUG-004: 随机旋转验证

| 属性 | 值 |
|-----|-----|
| **测试ID** | TC-AUG-004 |
| **测试项** | RandomRotation 功能与角度范围 |
| **前置条件** | 设置固定随机种子 |
| **输入数据** | 含明显方向特征的测试图像（如箭头） |
| **测试步骤** | 1. 设置 degrees=15<br>2. 执行多次旋转<br>3. 验证旋转角度在 [-15, 15] 范围内<br>4. 检查填充值和填充模式 |
| **预期结果** | 旋转角度在设定范围内，图像边界处理正确 |
| **通过标准** | 所有旋转角度在 [-15, +15] 内，无异常像素值 |

#### TC-AUG-005: MixUp 增强验证（可选）

| 属性 | 值 |
|-----|-----|
| **测试ID** | TC-AUG-005 |
| **测试项** | MixUp 图像混合 |
| **前置条件** | 实现 MixUp 函数 |
| **输入数据** | 两张已知内容的图像和对应标签 |
| **测试步骤** | 1. 设置混合系数 lambda=0.7<br>2. 执行 MixUp<br>3. 验证输出图像 = lambda*img1 + (1-lambda)*img2<br>4. 验证标签混合正确 |
| **预期结果** | 图像混合正确，标签为 lambda 软标签 |
| **通过标准** | 像素值误差 < 1e-6，标签混合系数正确 |

#### TC-AUG-006: CutMix 增强验证（可选）

| 属性 | 值 |
|-----|-----|
| **测试ID** | TC-AUG-006 |
| **测试项** | CutMix 区域裁剪与混合 |
| **前置条件** | 实现 CutMix 函数 |
| **输入数据** | 两张不同颜色的纯色图像（如全红和全蓝） |
| **测试步骤** | 1. 执行 CutMix<br>2. 验证裁剪区域来自第二张图<br>3. 计算裁剪区域面积比<br>4. 验证标签混合比例与面积比一致 |
| **预期结果** | 裁剪区域正确，标签比例与面积比匹配 |
| **通过标准** | 区域边界正确，面积比与标签比误差 < 1% |

---

### 2.3 验证集处理测试用例

#### TC-VAL-001: CenterCrop 验证

| 属性 | 值 |
|-----|-----|
| **测试ID** | TC-VAL-001 |
| **测试项** | CenterCrop 裁剪位置正确性 |
| **前置条件** | 无 |
| **输入数据** | 含明显中心标记的测试图像（如中心为红色十字） |
| **测试步骤** | 1. 创建 512x512 图像，中心 224x224 区域为红色<br>2. 执行 CenterCrop(224)<br>3. 检查输出是否为全红色 |
| **预期结果** | 输出为全红色图像 |
| **通过标准** | 所有像素为红色，裁剪位置居中 |

#### TC-VAL-002: 验证集 Transform 确定性验证

| 属性 | 值 |
|-----|-----|
| **测试ID** | TC-VAL-002 |
| **测试项** | 验证集 Transform 无随机性 |
| **前置条件** | 无 |
| **输入数据** | 同一张测试图像 |
| **测试步骤** | 1. 对同一图像执行验证 Transform 10 次<br>2. 比较所有输出是否完全相同 |
| **预期结果** | 10 次输出完全一致 |
| **通过标准** | 所有输出 Tensor 值完全相同（逐元素比较） |

---

### 2.4 配置化测试用例

#### TC-CFG-001: 增强参数配置解析

| 属性 | 值 |
|-----|-----|
| **测试ID** | TC-CFG-001 |
| **测试项** | YAML/JSON 配置文件解析 |
| **前置条件** | 存在配置文件模板 |
| **输入数据** | 包含增强参数的配置文件 |
| **测试步骤** | 1. 准备配置文件：<br>```yaml<br>augmentation:<br>  horizontal_flip: 0.5<br>  rotation: 15<br>  color_jitter:<br>    brightness: 0.2<br>    contrast: 0.2<br>```<br>2. 加载配置<br>3. 验证参数正确传递到 Transform |
| **预期结果** | 配置值正确解析并应用 |
| **通过标准** | 配置值与 Transform 参数一致 |

#### TC-CFG-002: 默认参数行为

| 属性 | 值 |
|-----|-----|
| **测试ID** | TC-CFG-002 |
| **测试项** | 未配置参数使用默认值 |
| **前置条件** | 存在默认参数定义 |
| **输入数据** | 不完整的配置文件（缺少部分参数） |
| **测试步骤** | 1. 加载不完整配置<br>2. 检查缺失参数是否使用默认值 |
| **预期结果** | 缺失参数采用预定义默认值 |
| **通过标准** | 所有参数都有有效值（配置值或默认值） |

#### TC-CFG-003: 参数范围校验

| 属性 | 值 |
|-----|-----|
| **测试ID** | TC-CFG-003 |
| **测试项** | 非法参数值检测 |
| **前置条件** | 实现参数校验逻辑 |
| **输入数据** | 含非法值的配置文件（如 brightness=-1, rotation=500） |
| **测试步骤** | 1. 加载含非法参数的配置<br>2. 执行参数校验<br>3. 验证是否抛出适当异常或警告 |
| **预期结果** | 拒绝非法参数，抛出明确错误信息 |
| **通过标准** | 非法参数被检测并报告，不会静默接受 |

---

### 2.5 可视化验证测试用例

#### TC-VIS-001: 增强效果可视化脚本

| 属性 | 值 |
|-----|-----|
| **测试ID** | TC-VIS-001 |
| **测试项** | 增强效果可视化输出 |
| **前置条件** | 实现可视化脚本 |
| **输入数据** | 样本图像集 |
| **测试步骤** | 1. 运行可视化脚本<br>2. 检查输出图像/网格<br>3. 人工确认增强效果合理 |
| **预期结果** | 输出展示原图与增强后图像的对比 |
| **通过标准** | 可视化清晰展示增强效果，便于人工审核 |

#### TC-VIS-002: 增强强度对比

| 属性 | 值 |
|-----|-----|
| **测试ID** | TC-VIS-002 |
| **测试项** | 不同参数设置的视觉效果对比 |
| **前置条件** | 实现参数化可视化脚本 |
| **输入数据** | 同一图像，不同增强参数设置 |
| **测试步骤** | 1. 设置弱/中/强三档增强参数<br>2. 生成对比可视化<br>3. 验证效果梯度明显 |
| **预期结果** | 三档增强效果有明显视觉差异 |
| **通过标准** | 人眼可区分不同增强强度 |

---

## 3. 验证方法与步骤

### 3.1 单元测试执行步骤

```bash
# 1. 进入项目根目录
cd /path/to/project

# 2. 运行数据处理模块单元测试
python -m pytest tests/data/test_transforms.py -v

# 3. 运行单个测试用例
python -m pytest tests/data/test_transforms.py::TestBaseTransforms::test_resize -v

# 4. 生成覆盖率报告
python -m pytest tests/data/test_transforms.py --cov=src/data/transforms --cov-report=html
```

### 3.2 集成测试执行步骤

```bash
# 1. 准备测试数据
python scripts/prepare_test_data.py

# 2. 运行 DataLoader 集成测试
python -m pytest tests/data/test_dataloader.py -v

# 3. 验证完整 pipeline
python scripts/verify_data_pipeline.py --config configs/data/default.yaml
```

### 3.3 可视化验证步骤

```bash
# 1. 生成增强效果可视化
python scripts/visualize_augmentation.py \
    --input tests/fixtures/sample_images/ \
    --output docs/augmentation_samples/ \
    --config configs/augmentation/train.yaml

# 2. 人工审核输出图像
# 打开 docs/augmentation_samples/ 目录查看效果

# 3. 生成增强强度对比
python scripts/compare_augmentation_levels.py \
    --input tests/fixtures/sample_images/ \
    --output docs/augmentation_comparison/
```

### 3.4 配置验证步骤

```bash
# 1. 验证配置文件格式
python -c "import yaml; yaml.safe_load(open('configs/augmentation/train.yaml'))"

# 2. 验证配置加载
python scripts/validate_config.py --config configs/augmentation/train.yaml

# 3. 测试参数边界
python scripts/test_config_bounds.py
```

---

## 4. 通过标准

### 4.1 量化通过标准

| 测试类别 | 通过条件 |
|---------|---------|
| 单元测试 | 所有测试用例通过，代码覆盖率 ≥ 90% |
| 概率性增强 | 统计分布在理论值 ±5% 范围内（样本量 ≥ 1000） |
| 数值精度 | 浮点误差 < 1e-6 |
| 配置解析 | 所有配置项正确解析，默认值正确应用 |
| 可视化验证 | 人工审核通过，效果合理 |

### 4.2 必须通过的测试用例

以下测试用例为 **阻塞级**，必须通过：

1. **TC-BASE-001**: 图像尺寸统一 — 核心 Pipeline 基础
2. **TC-BASE-002**: 归一化参数 — 训练稳定性关键
3. **TC-BASE-003**: Tensor 转换 — PyTorch 兼容性基础
4. **TC-VAL-002**: 验证集确定性 — 评估可靠性关键
5. **TC-CFG-001**: 配置解析 — 可配置性要求

### 4.3 建议通过的测试用例

以下测试用例为 **建议级**，建议通过但非阻塞：

1. **TC-AUG-005**: MixUp — 可选功能
2. **TC-AUG-006**: CutMix — 可选功能
3. **TC-VIS-002**: 增强强度对比 — 辅助验证

---

## 5. 自动化验证建议

### 5.1 完全自动化的测试

| 测试类型 | 自动化方式 | CI 集成 |
|---------|-----------|--------|
| 基础 Transform 单元测试 | pytest + 参数化 | ✅ 每次 PR |
| 配置解析测试 | pytest + YAML schema | ✅ 每次 PR |
| 概率性增强测试 | pytest + 统计断言 | ✅ 每次 PR |
| DataLoader 集成测试 | pytest fixtures | ✅ 每次 PR |

### 5.2 半自动化的测试

| 测试类型 | 自动化部分 | 手动部分 |
|---------|-----------|---------|
| 可视化验证 | 自动生成图像 | 人工审核效果 |
| 增强参数调优 | 自动生成对比图 | 人工选择最佳参数 |
| 边界条件测试 | 自动运行测试 | 人工确认异常处理合理 |

### 5.3 自动化脚本示例

```python
# tests/data/test_transforms.py

import pytest
import torch
import torchvision.transforms as T
from src.data.transforms import get_train_transform, get_val_transform

class TestBaseTransforms:
    """基础处理测试"""

    @pytest.mark.parametrize("input_size,target_size", [
        ((512, 512), (224, 224)),
        ((1024, 768), (224, 224)),
        ((800, 1200), (224, 224)),
        ((256, 256), (224, 224)),
    ])
    def test_resize_output_shape(self, input_size, target_size):
        """TC-BASE-001: 验证输出尺寸"""
        transform = T.Compose([T.Resize(target_size[0]), T.ToTensor()])
        img = torch.rand(3, *input_size)
        # ... 测试逻辑

    def test_normalization_values(self):
        """TC-BASE-002: 验证归一化参数"""
        from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD
        expected_mean = [0.485, 0.456, 0.406]
        expected_std = [0.229, 0.224, 0.225]
        assert torch.allclose(IMAGENET_MEAN, torch.tensor(expected_mean), atol=1e-3)
        assert torch.allclose(IMAGENET_STD, torch.tensor(expected_std), atol=1e-3)


class TestAugmentationTransforms:
    """训练集增强测试"""

    def test_horizontal_flip_probability(self):
        """TC-AUG-001: 验证翻转概率"""
        torch.manual_seed(42)
        transform = T.RandomHorizontalFlip(p=0.5)
        flip_count = 0
        for _ in range(1000):
            img = torch.rand(3, 224, 224)
            out = transform(img)
            if not torch.equal(img, out):
                flip_count += 1
        # 95% 置信区间检查
        assert 450 <= flip_count <= 550, f"Flip count {flip_count} out of expected range"

    def test_val_transform_determinism(self):
        """TC-VAL-002: 验证集确定性"""
        transform = get_val_transform(config={'size': 224})
        img = torch.rand(3, 512, 512)

        outputs = [transform(img) for _ in range(10)]
        for i, out in enumerate(outputs[1:], 1):
            assert torch.equal(outputs[0], out), f"Output {i} differs from first output"
```

### 5.4 CI 集成配置

```yaml
# .github/workflows/test-data.yml
name: Data Pipeline Tests

on: [push, pull_request]

jobs:
  test-transforms:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -e .[dev]
      - name: Run transform unit tests
        run: pytest tests/data/test_transforms.py -v --cov=src/data --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v4
```

---

## 6. 风险点与注意事项

### 6.1 概率性测试的不稳定性

**风险描述**: 随机增强操作的测试可能因概率波动而偶发失败。

**缓解措施**:
1. 使用固定随机种子：`torch.manual_seed(42)`
2. 增大样本量：至少 1000 次测试
3. 使用合理的置信区间：±5% 或更宽松
4. 避免硬编码精确值断言

```python
# 不推荐
assert flip_count == 500

# 推荐
assert 470 <= flip_count <= 530  # ±3 sigma
```

### 6.2 图像格式兼容性

**风险描述**: 不同图像格式（JPEG, PNG, TIFF）可能导致读取结果差异。

**缓解措施**:
1. 测试覆盖主流图像格式
2. 统一图像读取方式（如始终用 PIL 或 cv2）
3. 处理 Alpha 通道（RGBA → RGB）

### 6.3 边界条件处理

**风险描述**: 极端尺寸或值可能导致异常。

**需测试的边界情况**:
- 极小图像（如 10x10）
- 极大图像（如 4000x4000）
- 非标准比例（如 1:100）
- 全黑/全白图像
- 含 NaN/Inf 像素值

### 6.4 性能问题

**风险描述**: 增强操作可能成为训练瓶颈。

**关注点**:
1. DataLoader 的 num_workers 设置
2. 内存占用（特别是 CutMix/MixUp）
3. GPU 预处理 vs CPU 预处理

**建议**:
- 基准测试 Transform 耗时：`%timeit transform(img)`
- 监控 DataLoader 吞吐量

### 6.5 配置热更新

**风险描述**: 运行时修改增强配置可能不生效。

**缓解措施**:
1. 配置加载后验证
2. 提供 `get_transform(config)` 工厂函数
3. 避免全局状态

### 6.6 多 GPU 分布式训练

**风险描述**: 分布式训练中增强可能导致数据不一致。

**关注点**:
1. 每个进程独立随机种子
2. 分布式采样器正确性
3. 同步 BN 与增强的交互

**建议**:
```python
# 分布式训练设置种子
seed = base_seed + dist.get_rank()
torch.manual_seed(seed)
```

---

## 附录 A: 测试数据集建议

| 数据类型 | 描述 | 用途 |
|---------|-----|-----|
| 标准尺寸图像 | 224x224, 512x512 正方形 | 基础功能测试 |
| 非标准比例图像 | 16:9, 4:3, 1:10 | Resize/Crop 测试 |
| 边界图像 | 1x1, 10000x10000 | 边界条件测试 |
| 格式测试图像 | JPEG, PNG, BMP, TIFF | 格式兼容测试 |
| 已知内容图像 | 纯色图、渐变图、网格图 | 归一化/Transform 验证 |

---

## 附录 B: 验收检查清单

```markdown
## 数据预处理与增强验收检查清单

### 基础处理
- [ ] TC-BASE-001: 图像尺寸统一验证通过
- [ ] TC-BASE-002: 归一化参数验证通过
- [ ] TC-BASE-003: Tensor 转换验证通过

### 训练集增强
- [ ] TC-AUG-001: 随机水平翻转验证通过
- [ ] TC-AUG-002: 随机裁剪与 Resize 验证通过
- [ ] TC-AUG-003: 色彩抖动验证通过
- [ ] TC-AUG-004: 随机旋转验证通过
- [ ] TC-AUG-005: MixUp 验证通过（如实现）
- [ ] TC-AUG-006: CutMix 验证通过（如实现）

### 验证集处理
- [ ] TC-VAL-001: CenterCrop 验证通过
- [ ] TC-VAL-002: 验证集 Transform 确定性验证通过

### 配置化
- [ ] TC-CFG-001: 增强参数配置解析通过
- [ ] TC-CFG-002: 默认参数行为通过
- [ ] TC-CFG-003: 参数范围校验通过

### 可视化
- [ ] TC-VIS-001: 增强效果可视化脚本完成
- [ ] TC-VIS-002: 增强强度对比验证通过

### 覆盖率
- [ ] 单元测试覆盖率 ≥ 90%

### 文档
- [ ] 增强参数配置说明文档完成
- [ ] 使用示例代码完成
```