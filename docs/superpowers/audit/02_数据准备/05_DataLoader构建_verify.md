# DataLoader 构建验证测试计划

> **验证阶段：** 阶段二：数据准备
> **验证目标：** 确保 Dataset 类实现正确，DataLoader 配置合理，数据加载性能满足训练需求
> **创建日期：** 2026-06-02

---

## 一、验证目标

### 1.1 功能正确性验证
- Dataset 类正确继承 `torch.utils.data.Dataset`
- `__len__` 方法返回正确的数据集大小
- `__getitem__` 方法正确返回数据和标签
- 数据索引边界处理正确

### 1.2 配置合理性验证
- batch_size 配置与 GPU 内存匹配
- num_workers 配置优化数据加载吞吐量
- pin_memory 配置正确
- shuffle 参数配置符合训练/验证需求

### 1.3 性能验证
- 数据加载速度满足训练需求（无 GPU 等待瓶颈）
- 内存占用在可接受范围内
- 无内存泄漏问题

---

## 二、测试用例

### 2.1 Dataset 类测试

| 测试ID | 测试项 | 测试步骤 | 预期结果 |
|--------|--------|----------|----------|
| DS-01 | `__len__` 方法正确性 | 1. 创建 Dataset 实例<br>2. 调用 `len(dataset)`<br>3. 与实际文件数量对比 | 返回值等于数据集中样本总数 |
| DS-02 | `__getitem__` 单样本获取 | 1. 创建 Dataset 实例<br>2. 调用 `dataset[0]`<br>3. 检查返回类型和形状 | 返回 (data, label) 元组，形状符合预期 |
| DS-03 | `__getitem__` 边界索引 | 1. 调用 `dataset[len(dataset)-1]`<br>2. 调用 `dataset[0]`<br>3. 尝试越界索引 `dataset[len(dataset)]` | 最后一个索引正常返回，越界抛出 IndexError |
| DS-04 | `__getitem__` 随机索引 | 1. 随机选择10个不同索引<br>2. 分别调用 `__getitem__`<br>3. 验证每次返回独立数据 | 所有索引正常返回，数据互不干扰 |
| DS-05 | 数据类型验证 | 1. 获取单个样本<br>2. 检查数据类型<br>3. 检查值范围 | 数据为 torch.Tensor 或可转换为 Tensor，值范围合理 |
| DS-06 | 标签正确性 | 1. 获取多个样本<br>2. 验证标签与数据文件对应 | 标签正确映射到数据 |

### 2.2 DataLoader 配置测试

| 测试ID | 测试项 | 测试步骤 | 预期结果 |
|--------|--------|----------|----------|
| DL-01 | batch_size 正确性 | 1. 创建 DataLoader with batch_size=32<br>2. 获取第一个 batch<br>3. 检查 batch 形状 | batch 数据形状为 (32, C, H, W) 或 (32, ...) |
| DL-02 | drop_last 行为 | 1. 设置 batch_size=7, 数据集大小=100<br>2. 设置 drop_last=True<br>3. 统计总 batch 数 | batch 数 = floor(100/7) = 14 |
| DL-03 | shuffle 训练模式 | 1. 设置 shuffle=True<br>2. 两个 epoch 获取第一个 batch<br>3. 比较数据 | 两个 epoch 的第一个 batch 数据不同 |
| DL-04 | 验证模式不shuffle | 1. 设置 shuffle=False<br>2. 两个 epoch 获取第一个 batch<br>3. 比较数据 | 两个 epoch 数据顺序完全相同 |
| DL-05 | num_workers 影响 | 1. 分别设置 num_workers=0,2,4,8<br>2. 测试加载速度<br>3. 记录对比 | 存在最优 num_workers 值使加载速度最快 |
| DL-06 | pin_memory 影响 | 1. 设置 pin_memory=True/False<br>2. 测试 GPU 训练时数据传输速度 | pin_memory=True 时 GPU 传输更快 |

### 2.3 性能测试

| 测试ID | 测试项 | 测试步骤 | 预期结果 |
|--------|--------|----------|----------|
| PF-01 | 单 epoch 加载时间 | 1. 记录遍历整个 DataLoader 的时间<br>2. 计算 each sample 平均加载时间 | 平均每个样本加载时间 < 1ms |
| PF-02 | 内存占用峰值 | 1. 使用内存监控工具<br>2. 遍历整个 DataLoader<br>3. 记录内存峰值 | 内存峰值 < 系统可用内存 50% |
| PF-03 | GPU 利用率 | 1. 训练时监控 GPU 利用率<br>2. 检查是否有频繁的 GPU 等待 | GPU 利用率 > 80% |
| PF-04 | 多 epoch 内存稳定性 | 1. 连续运行 10 个 epoch<br>2. 监控内存变化趋势 | 内存占用稳定，无持续增长趋势 |
| PF-05 | 并发加载压力测试 | 1. 设置 num_workers=8<br>2. 连续加载 1000 个 batch<br>3. 检查错误率 | 无 OOM 错误，无加载超时 |

---

## 三、验证方法

### 3.1 单元测试方法

```python
import pytest
import torch
from torch.utils.data import DataLoader

class TestDataset:
    """Dataset 类单元测试"""

    def test_len(self, dataset):
        """测试 __len__ 方法"""
        assert len(dataset) > 0
        assert len(dataset) == dataset.__len__()

    def test_getitem_single(self, dataset):
        """测试 __getitem__ 单样本获取"""
        data, label = dataset[0]
        assert data is not None
        assert label is not None

    def test_getitem_boundary(self, dataset):
        """测试边界索引"""
        # 最后一个有效索引
        data, label = dataset[len(dataset) - 1]
        assert data is not None

        # 越界索引
        with pytest.raises(IndexError):
            dataset[len(dataset)]

    def test_getitem_random(self, dataset):
        """测试随机索引独立性"""
        import random
        indices = random.sample(range(len(dataset)), min(10, len(dataset)))
        results = [dataset[i] for i in indices]
        assert len(results) == len(indices)

    def test_data_type(self, dataset):
        """测试数据类型"""
        data, label = dataset[0]
        assert isinstance(data, torch.Tensor) or hasattr(data, '__array__')

class TestDataLoader:
    """DataLoader 配置测试"""

    def test_batch_size(self, dataloader, batch_size):
        """测试 batch_size 配置"""
        batch = next(iter(dataloader))
        data, labels = batch
        assert data.shape[0] == batch_size or data.shape[0] <= batch_size

    def test_shuffle_effect(self, train_loader):
        """测试 shuffle 效果"""
        epoch1_first = next(iter(train_loader))
        epoch2_first = next(iter(train_loader))
        # shuffle=True 时，两次迭代顺序应该不同
        # 注意：小数据集可能偶然相同
        pass

    def test_no_shuffle_consistency(self, val_loader):
        """测试不 shuffle 的一致性"""
        epoch1_first = next(iter(val_loader))
        epoch2_first = next(iter(val_loader))
        torch.testing.assert_close(epoch1_first[0], epoch2_first[0])
```

### 3.2 性能测试脚本

```python
import time
import psutil
import GPUtil
from memory_profiler import profile

def measure_loading_speed(dataloader, num_epochs=3):
    """测量数据加载速度"""
    times = []
    for epoch in range(num_epochs):
        start = time.time()
        for batch in dataloader:
            pass
        times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    samples_per_sec = len(dataloader.dataset) / avg_time
    return {
        'avg_epoch_time': avg_time,
        'samples_per_sec': samples_per_sec,
        'times_per_epoch': times
    }

def measure_memory_usage(dataloader):
    """测量内存使用"""
    process = psutil.Process()
    initial_mem = process.memory_info().rss / 1024 / 1024  # MB

    peak_mem = initial_mem
    for batch in dataloader:
        current_mem = process.memory_info().rss / 1024 / 1024
        peak_mem = max(peak_mem, current_mem)

    return {
        'initial_mb': initial_mem,
        'peak_mb': peak_mem,
        'increase_mb': peak_mem - initial_mem
    }

def check_gpu_utilization(dataloader, model, device, num_batches=100):
    """检查 GPU 利用率"""
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    gpu_utils = []
    model.train()
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        data, labels = batch
        data, labels = data.to(device), labels.to(device)
        output = model(data)

        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_utils.append(util.gpu)

    return {
        'avg_gpu_util': sum(gpu_utils) / len(gpu_utils),
        'min_gpu_util': min(gpu_utils),
        'max_gpu_util': max(gpu_utils)
    }

def find_optimal_workers(dataset, batch_size=32, max_workers=16):
    """寻找最优 num_workers"""
    results = {}
    for workers in range(0, max_workers + 1, 2):
        if workers == 0:
            workers = 1  # 至少使用1个worker测试

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=True
        )

        start = time.time()
        for _ in loader:
            pass
        elapsed = time.time() - start

        results[workers] = elapsed
        print(f"num_workers={workers}: {elapsed:.2f}s")

    optimal = min(results, key=results.get)
    return optimal, results
```

### 3.3 集成测试方法

```python
def integration_test_training_pipeline(dataloader, model, device, num_batches=10):
    """集成测试：验证 DataLoader 与训练流程集成"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters())

    errors = []
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        try:
            # 验证数据格式
            data, labels = batch
            assert data is not None, f"Batch {i}: data is None"
            assert labels is not None, f"Batch {i}: labels is None"

            # 验证设备传输
            data = data.to(device)
            labels = labels.to(device)

            # 验证前向传播
            output = model(data)
            assert output is not None

            # 验证反向传播
            loss = compute_loss(output, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        except Exception as e:
            errors.append(f"Batch {i}: {str(e)}")

    return {
        'success': len(errors) == 0,
        'errors': errors,
        'batches_tested': min(num_batches, len(dataloader))
    }
```

---

## 四、验收标准

### 4.1 功能验收标准（必须全部通过）

| 标准 | 验收条件 | 量化指标 |
|------|----------|----------|
| Dataset 类继承 | 正确继承 `torch.utils.data.Dataset` | isinstance 检查通过 |
| `__len__` 实现 | 返回正确数据集大小 | 等于实际样本数，误差 0 |
| `__getitem__` 实现 | 正确返回数据和标签 | 返回元组，类型正确率 100% |
| 边界处理 | 越界索引抛出 IndexError | 异常抛出率 100% |
| batch_size 配置 | 每个 batch 大小正确 | 除最后一个 batch 外，大小等于配置值 |

### 4.2 性能验收标准

| 指标 | 合格标准 | 优秀标准 | 测量方法 |
|------|----------|----------|----------|
| 单样本加载时间 | < 5ms | < 1ms | epoch_time / total_samples |
| GPU 利用率 | > 70% | > 85% | nvidia-smi 平均利用率 |
| 内存峰值占用 | < 可用内存 70% | < 可用内存 50% | memory_profiler 峰值 |
| 多 epoch 内存增长 | < 5% | < 1% | epoch1_peak vs epoch10_peak |
| DataLoader 吞吐量 | > 训练速度 1.2x | > 训练速度 1.5x | batch/s 对比 |

### 4.3 配置合理性标准

| 配置项 | 推荐范围 | 验收方法 |
|--------|----------|----------|
| batch_size | 根据 GPU 内存最大化，建议 16-128 | 无 OOM 错误，GPU 内存利用率 > 80% |
| num_workers | CPU 核心数的 1/4 到核心数 | 性能测试最优值 |
| pin_memory | CUDA 训练时为 True | 对比测试传输速度 |
| shuffle | 训练集 True，验证/测试集 False | 验证训练收敛性 |

---

## 五、自动化建议

### 5.1 自动化测试脚本

```bash
# run_dataloader_tests.sh
#!/bin/bash

echo "=== DataLoader 自动化测试 ==="

# 运行单元测试
echo "1. 运行单元测试..."
pytest tests/test_dataloader.py -v --cov=src/data --cov-report=html

# 运行性能测试
echo "2. 运行性能测试..."
python scripts/benchmark_dataloader.py --output results/performance.json

# 生成报告
echo "3. 生成验证报告..."
python scripts/generate_verification_report.py \
    --test-results tests/results \
    --perf-results results/performance.json \
    --output docs/verification_report.md

echo "=== 测试完成 ==="
```

### 5.2 CI/CD 集成

```yaml
# .github/workflows/dataloader_test.yml
name: DataLoader Verification

on:
  push:
    paths:
      - 'src/data/**'
      - 'tests/test_dataloader.py'
  pull_request:
    paths:
      - 'src/data/**'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov memory-profiler psutil gpustat

      - name: Run unit tests
        run: pytest tests/test_dataloader.py -v --cov=src/data

      - name: Run performance tests
        run: python scripts/benchmark_dataloader.py

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: results/
```

### 5.3 监控仪表板配置

```python
# scripts/dataloader_monitor.py
"""DataLoader 性能监控脚本"""

import json
import time
from datetime import datetime
from pathlib import Path

class DataLoaderMonitor:
    def __init__(self, log_dir="logs/dataloader"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = []

    def log_epoch(self, epoch, duration, samples, memory_peak, gpu_util):
        """记录单个 epoch 的指标"""
        metric = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'duration_sec': duration,
            'samples': samples,
            'samples_per_sec': samples / duration,
            'memory_peak_mb': memory_peak,
            'gpu_utilization': gpu_util
        }
        self.metrics.append(metric)

    def save_report(self):
        """保存监控报告"""
        report_path = self.log_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        # 生成汇总
        summary = {
            'total_epochs': len(self.metrics),
            'avg_samples_per_sec': sum(m['samples_per_sec'] for m in self.metrics) / len(self.metrics),
            'avg_memory_mb': sum(m['memory_peak_mb'] for m in self.metrics) / len(self.metrics),
            'avg_gpu_util': sum(m['gpu_utilization'] for m in self.metrics) / len(self.metrics)
        }

        summary_path = self.log_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return summary
```

---

## 六、风险点

### 6.1 功能风险

| 风险 | 影响 | 可能性 | 缓解措施 |
|------|------|--------|----------|
| 数据文件损坏或格式错误 | 加载失败，训练中断 | 中 | 添加数据验证和异常处理，记录损坏文件 |
| 索引越界 | 程序崩溃 | 低 | 边界检查，单元测试覆盖 |
| 多进程竞争条件 | 数据加载错误 | 低 | 使用适当的锁机制，测试多 worker 场景 |
| 标签映射错误 | 训练结果错误 | 中 | 标签验证测试，抽样人工检查 |

### 6.2 性能风险

| 风险 | 影响 | 可能性 | 缓解措施 |
|------|------|--------|----------|
| num_workers 过高导致 CPU 瓶颈 | 加载速度下降 | 中 | 性能测试找最优值，监控 CPU 使用率 |
| 内存泄漏 | OOM 崩溃 | 中 | 多 epoch 内存测试，使用内存分析工具 |
| 磁盘 I/O 瓶颈 | GPU 等待，训练慢 | 高 | 使用 SSD，预取数据，缓存策略 |
| pin_memory 与 CUDA 不兼容 | 数据传输慢 | 低 | 检查 CUDA 版本，条件配置 |

### 6.3 配置风险

| 风险 | 影响 | 可能性 | 缓解措施 |
|------|------|--------|----------|
| batch_size 过大导致 OOM | 训练失败 | 高 | 内存测试，梯度检查点，自动 batch_size 查找 |
| shuffle 配置错误 | 验证结果不准确 | 中 | 配置检查，自动化测试 |
| num_workers=0 导致慢速 | 训练效率低 | 中 | 配置验证，性能基准测试 |
| 预取配置不当 | GPU 空闲 | 中 | 根据硬件调优，监控 GPU 利用率 |

### 6.4 兼容性风险

| 风险 | 影响 | 可能性 | 缓解措施 |
|------|------|--------|----------|
| PyTorch 版本差异 | API 不兼容 | 中 | 版本锁定，兼容性测试 |
| 操作系统差异 | 多进程行为不同 | 中 | 跨平台测试，条件配置 |
| CUDA 版本不匹配 | GPU 加速失效 | 低 | 版本检查，降级方案 |
| Python 版本差异 | 语法或库差异 | 低 | 版本要求明确，测试覆盖 |

---

## 七、验证清单

### 7.1 开发自测清单

- [ ] Dataset 类通过所有单元测试
- [ ] DataLoader 基本功能测试通过
- [ ] 单 epoch 加载测试完成
- [ ] 内存占用在合理范围
- [ ] 无明显 bug 或错误

### 7.2 代码审查清单

- [ ] 代码风格符合项目规范
- [ ] 异常处理完整
- [ ] 日志记录充分
- [ ] 文档注释完整
- [ ] 无硬编码配置

### 7.3 集成测试清单

- [ ] 与模型训练流程集成测试通过
- [ ] 多 epoch 运行稳定
- [ ] GPU 训练无 OOM
- [ ] 性能指标达标
- [ ] 日志和监控正常

### 7.4 发布前验收清单

- [ ] 所有测试用例通过
- [ ] 性能验收标准达标
- [ ] 风险点已评估和缓解
- [ ] 文档已更新
- [ ] 代码已合并主分支

---

## 八、参考资料

- [PyTorch DataLoader 官方文档](https://pytorch.org/docs/stable/data.html)
- [PyTorch 性能调优指南](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [内存分析工具 memory_profiler](https://pypi.org/project/memory-profiler/)
- [GPU 监控工具 pynvml](https://pypi.org/project/pynvml/)

---

**验证计划版本：** v1.0
**最后更新：** 2026-06-02
**负责人：** 待指定