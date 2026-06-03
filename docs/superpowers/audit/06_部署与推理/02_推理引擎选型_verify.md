# 推理引擎选型验证测试计划

> **文档版本：** v1.0
> **创建日期：** 2026-06-03
> **关联任务：** 阶段六：部署与推理 - 推理引擎选型

---

## 1. 验证目标

### 1.1 主要目标
1. **性能验证**：确保选定的推理引擎在目标硬件上满足性能指标
2. **精度验证**：确保模型转换后精度损失在可接受范围内
3. **兼容性验证**：确保推理引擎与现有技术栈和部署环境兼容
4. **稳定性验证**：确保推理服务在负载条件下稳定运行

### 1.2 关键指标
| 指标类别 | 指标名称 | 目标值 | 优先级 |
|---------|---------|--------|-------|
| 性能 | GPU推理延迟 | < 10ms (batch=1) | P0 |
| 性能 | CPU推理延迟 | < 50ms (batch=1) | P0 |
| 性能 | 吞吐量 (GPU) | > 100 QPS | P1 |
| 性能 | 吞吐量 (CPU) | > 20 QPS | P1 |
| 精度 | 准确率损失 | < 0.5% | P0 |
| 精度 | 特征向量相似度 | > 99.5% | P0 |
| 资源 | GPU显存占用 | < 2GB | P1 |
| 资源 | CPU内存占用 | < 1GB | P1 |
| 稳定性 | 服务可用性 | > 99.9% | P0 |

---

## 2. 测试用例

### 2.1 GPU推理引擎测试

#### TC-GPU-001: TensorRT 导出验证
```yaml
测试名称: TensorRT模型导出验证
前置条件:
  - ERes2NetV2训练模型可用
  - TensorRT 8.6+ 已安装
测试步骤:
  1. 将PyTorch模型导出为ONNX格式
  2. 使用trtexec转换为TensorRT引擎
  3. 验证引擎文件生成成功
  4. 记录引擎文件大小
验证点:
  - ONNX导出无错误
  - TensorRT引擎构建成功
  - 无精度警告或错误
预期结果: 成功生成TensorRT引擎文件
```

#### TC-GPU-002: TensorRT 精度测试
```yaml
测试名称: TensorRT推理精度对比
测试步骤:
  1. 准备100张测试图片
  2. 分别使用PyTorch和TensorRT推理
  3. 比较输出特征向量的余弦相似度
  4. 统计相似度分布
验证点:
  - 特征向量余弦相似度 > 99.5%
  - 无NaN或Inf输出
  - 相似度方差 < 0.01
预期结果: 精度损失 < 0.5%
```

#### TC-GPU-003: TensorRT 性能测试
```yaml
测试名称: TensorRT推理性能基准
测试步骤:
  1. 预热推理引擎 (100次推理)
  2. 执行1000次推理，记录延迟分布
  3. 测试不同batch size (1, 4, 8, 16, 32)
  4. 测试FP32和FP16精度模式
验证点:
  - P50延迟 < 8ms (batch=1, FP16)
  - P99延迟 < 15ms (batch=1, FP16)
  - 吞吐量 > 100 QPS
预期结果: 满足性能目标
```

#### TC-GPU-004: ONNX Runtime GPU 导出验证
```yaml
测试名称: ONNX Runtime GPU模型验证
测试步骤:
  1. 导出ONNX模型
  2. 使用onnxruntime-gpu加载模型
  3. 验证CUDA执行提供程序可用
  4. 执行推理验证输出正确性
验证点:
  - ONNX模型验证通过
  - CUDA EP正确加载
  - 推理输出正确
预期结果: ONNX Runtime GPU可用
```

#### TC-GPU-005: GPU引擎对比测试
```yaml
测试名称: TensorRT vs ONNX Runtime GPU对比
测试步骤:
  1. 使用相同测试数据集
  2. 分别测试TensorRT和ONNX Runtime
  3. 对比指标：延迟、吞吐量、显存、精度
  4. 生成对比报告
验证点:
  - 性能对比表格完整
  - 精度对比表格完整
  - 资源占用对比完整
预期结果: 生成完整对比报告，推荐最优方案
```

### 2.2 CPU推理引擎测试

#### TC-CPU-001: ONNX Runtime CPU 验证
```yaml
测试名称: ONNX Runtime CPU基础验证
测试步骤:
  1. 使用onnxruntime-cpu加载模型
  2. 启用图优化 (ORT optimizations)
  3. 测试不同线程配置 (1, 2, 4, 8 threads)
  4. 记录性能和资源占用
验证点:
  - 模型加载成功
  - 图优化启用成功
  - 性能随线程数合理变化
预期结果: ONNX Runtime CPU可用且优化有效
```

#### TC-CPU-002: OpenVINO 导出验证
```yaml
测试名称: OpenVINO模型转换验证
测试步骤:
  1. 使用模型优化器转换ONNX到IR格式
  2. 验证FP32和FP16模型生成
  3. 使用OpenVINO Runtime加载模型
  4. 验证推理正确性
验证点:
  - IR模型生成成功
  - FP16量化无精度警告
  - 推理输出正确
预期结果: OpenVINO模型转换和加载成功
```

#### TC-CPU-003: CPU引擎性能对比
```yaml
测试名称: ONNX Runtime vs OpenVINO性能对比
测试步骤:
  1. 使用相同测试数据集
  2. 测试不同硬件配置 (Intel/AMD)
  3. 测试不同优化级别
  4. 生成对比报告
验证点:
  - 延迟对比 (P50/P95/P99)
  - 吞吐量对比
  - 内存占用对比
预期结果: 明确CPU推理最优方案
```

### 2.3 边缘设备推理引擎测试

#### TC-EDGE-001: NCNN 导出验证
```yaml
测试名称: NCNN模型转换验证
前置条件:
  - 目标边缘设备确定
测试步骤:
  1. ONNX转NCNN模型
  2. 验证模型结构正确
  3. 在目标设备测试推理
  4. 记录模型大小和性能
验证点:
  - 模型转换成功
  - 无不支持的算子
  - 推理结果正确
预期结果: NCNN模型可用
```

#### TC-EDGE-002: MNN 导出验证
```yaml
测试名称: MNN模型转换验证
测试步骤:
  1. PyTorch/ONNX转MNN模型
  2. 验证模型量化 (INT8)
  3. 在目标设备测试推理
  4. 对比量化前后精度和性能
验证点:
  - 模型转换成功
  - INT8量化精度损失 < 1%
  - 性能提升明显
预期结果: MNN模型可用
```

#### TC-EDGE-003: TFLite 导出验证
```yaml
测试名称: TensorFlow Lite模型验证
测试步骤:
  1. 转换模型为TFLite格式
  2. 应用训练后量化
  3. 在Android/iOS设备测试
  4. 记录性能和电池消耗
验证点:
  - TFLite模型生成成功
  - 量化后精度符合要求
  - 移动端推理正常
预期结果: TFLite模型在移动端可用
```

#### TC-EDGE-004: CoreML 导出验证
```yaml
测试名称: CoreML模型验证 (iOS)
测试步骤:
  1. 转换模型为CoreML格式
  2. 验证Neural Engine加速
  3. 在iPhone设备测试
  4. 记录性能和电池消耗
验证点:
  - CoreML模型生成成功
  - Neural Engine利用率 > 80%
  - 推理延迟 < 30ms
预期结果: CoreML模型在iOS设备可用
```

#### TC-EDGE-005: 边缘引擎对比测试
```yaml
测试名称: 边缘推理引擎综合对比
测试步骤:
  1. 汇总所有边缘引擎测试结果
  2. 按平台分类对比
  3. 评估部署难度和维护成本
  4. 生成推荐方案
验证点:
  - Android平台推荐方案
  - iOS平台推荐方案
  - 嵌入式平台推荐方案
预期结果: 生成边缘设备推理方案推荐表
```

### 2.4 引擎对比综合测试

#### TC-COMP-001: 全引擎性能基准
```yaml
测试名称: 推理引擎性能基准矩阵
测试步骤:
  1. 定义统一测试数据集 (1000张图片)
  2. 所有引擎使用相同输入
  3. 记录完整性能指标
  4. 生成性能热力图
验证指标:
  - 推理延迟 (P50/P95/P99)
  - 吞吐量 (QPS)
  - 首次推理延迟 (冷启动)
  - 内存/显存占用
  - 模型加载时间
预期结果: 生成完整性能对比矩阵
```

#### TC-COMP-002: 精度一致性测试
```yaml
测试名称: 全引擎精度一致性验证
测试步骤:
  1. 以PyTorch输出为基准
  2. 所有引擎计算特征向量余弦相似度
  3. 统计精度分布和方差
  4. 识别精度异常的引擎配置
验证指标:
  - 余弦相似度均值 > 99.5%
  - 余弦相似度最小值 > 99%
  - 无异常值 (> 3σ)
预期结果: 所有引擎精度一致
```

#### TC-COMP-003: 部署复杂度评估
```yaml
测试名称: 部署和维护复杂度评估
评估维度:
  1. 模型转换复杂度 (步骤数、工具链)
  2. 依赖管理复杂度 (库数量、版本冲突)
  3. 跨平台支持度
  4. 社区活跃度和文档质量
  5. 问题排查难度
评分标准:
  - 每维度1-5分
  - 总分 < 10: 高风险
  - 总分 10-15: 中风险
  - 总分 > 15: 低风险
预期结果: 生成部署风险评估表
```

### 2.5 稳定性和可靠性测试

#### TC-STAB-001: 长时间运行测试
```yaml
测试名称: 推理服务稳定性测试
测试步骤:
  1. 启动推理服务
  2. 持续发送推理请求 (24小时)
  3. 监控内存泄漏、延迟变化
  4. 记录错误率和异常
验证指标:
  - 内存增长 < 5%/小时
  - 延迟波动 < 10%
  - 错误率 < 0.01%
预期结果: 服务稳定运行24小时无异常
```

#### TC-STAB-002: 并发压力测试
```yaml
测试名称: 高并发推理测试
测试步骤:
  1. 配置推理服务最大并发数
  2. 逐步增加并发请求 (1→10→50→100)
  3. 记录延迟、吞吐量、错误率
  4. 确定系统瓶颈
验证指标:
  - 线性扩展区吞吐量
  - 饱和点QPS
  - 错误率突增点
预期结果: 明确并发容量和瓶颈
```

#### TC-STAB-003: 异常输入测试
```yaml
测试名称: 异常输入鲁棒性测试
测试用例:
  1. 空图片输入
  2. 超大图片 (10x正常大小)
  3. 错误格式图片
  4. 损坏图片文件
  5. 极端像素值图片
验证指标:
  - 所有异常输入有明确错误码
  - 无崩溃或内存泄漏
  - 正常请求不受影响
预期结果: 服务对异常输入有健壮处理
```

---

## 3. 验证方法

### 3.1 自动化验证脚本

```python
# verify_engine.py - 推理引擎验证脚本框架
import time
import json
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class BenchmarkResult:
    engine: str
    device: str
    latency_p50: float
    latency_p95: float
    latency_p99: float
    throughput: float
    memory_mb: float
    cosine_similarity: float
    pass_tests: bool
    errors: List[str]

class InferenceEngineVerifier:
    def __init__(self, model_path: str, test_images: List[str]):
        self.model_path = model_path
        self.test_images = test_images
        self.baseline_features = None  # PyTorch features as baseline

    def verify_tensorrt(self) -> BenchmarkResult:
        """验证TensorRT引擎"""
        # 1. 模型转换
        # 2. 精度测试
        # 3. 性能测试
        # 4. 返回结果
        pass

    def verify_onnxruntime_gpu(self) -> BenchmarkResult:
        """验证ONNX Runtime GPU"""
        pass

    def verify_onnxruntime_cpu(self) -> BenchmarkResult:
        """验证ONNX Runtime CPU"""
        pass

    def verify_openvino(self) -> BenchmarkResult:
        """验证OpenVINO"""
        pass

    def generate_report(self, results: List[BenchmarkResult]) -> Dict:
        """生成对比报告"""
        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": [r.__dict__ for r in results],
            "recommendation": self._get_recommendation(results)
        }

    def _get_recommendation(self, results: List[BenchmarkResult]) -> Dict:
        """基于测试结果给出推荐"""
        # 综合性能、精度、资源占用评分
        pass
```

### 3.2 测试数据集准备

```python
# prepare_test_dataset.py
def prepare_test_dataset(
    source_dir: str,
    output_dir: str,
    num_images: int = 1000,
    diversity_requirements: Dict = None
):
    """
    准备测试数据集

    多样性要求:
    - 不同分辨率图片占比
    - 不同光照条件
    - 不同人脸角度
    - 不同人脸数量
    - 边界情况 (遮挡、模糊等)
    """
    pass
```

### 3.3 性能测试脚本

```python
# benchmark_inference.py
import time
import statistics
from concurrent.futures import ThreadPoolExecutor

def benchmark_latency(
    engine,
    images: List,
    warmup: int = 100,
    iterations: int = 1000
) -> Dict:
    """延迟基准测试"""
    # Warmup
    for _ in range(warmup):
        engine.infer(images[0])

    # Benchmark
    latencies = []
    for img in images[:iterations]:
        start = time.perf_counter()
        engine.infer(img)
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        "p50": statistics.median(latencies),
        "p95": statistics.quantiles(latencies, n=20)[18],
        "p99": statistics.quantiles(latencies, n=100)[98],
        "mean": statistics.mean(latencies),
        "std": statistics.stdev(latencies)
    }

def benchmark_throughput(
    engine,
    images: List,
    duration_seconds: int = 60,
    num_workers: int = 4
) -> Dict:
    """吞吐量基准测试"""
    completed = 0
    errors = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        while time.time() - start_time < duration_seconds:
            # 提交推理任务
            pass

    return {
        "qps": completed / duration_seconds,
        "errors": errors,
        "duration": duration_seconds
    }
```

### 3.4 精度验证脚本

```python
# verify_accuracy.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def verify_accuracy(
    baseline_features: np.ndarray,
    test_features: np.ndarray,
    threshold: float = 0.995
) -> Dict:
    """
    验证特征向量精度

    Args:
        baseline_features: PyTorch输出的特征向量 (N, D)
        test_features: 转换后引擎输出的特征向量 (N, D)
        threshold: 余弦相似度阈值

    Returns:
        验证结果字典
    """
    # 计算余弦相似度
    similarities = []
    for i in range(len(baseline_features)):
        sim = cosine_similarity(
            baseline_features[i:i+1],
            test_features[i:i+1]
        )[0, 0]
        similarities.append(sim)

    similarities = np.array(similarities)

    return {
        "mean_similarity": float(np.mean(similarities)),
        "min_similarity": float(np.min(similarities)),
        "max_similarity": float(np.max(similarities)),
        "std_similarity": float(np.std(similarities)),
        "pass_rate": float(np.mean(similarities >= threshold)),
        "threshold": threshold,
        "passed": bool(np.mean(similarities) >= threshold)
    }
```

---

## 4. 通过标准

### 4.1 必须通过标准 (P0)

| 测试项 | 通过标准 | 测量方法 |
|-------|---------|---------|
| 模型转换 | 无错误、无警告 | 转换日志检查 |
| 推理正确性 | 输出特征向量非零、非NaN | 输出验证脚本 |
| 精度一致性 | 余弦相似度 > 99.5% | 精度验证脚本 |
| GPU延迟 | P99 < 15ms (batch=1) | 延迟基准测试 |
| CPU延迟 | P99 < 100ms (batch=1) | 延迟基准测试 |
| 内存泄漏 | 24小时增长 < 5% | 长期运行测试 |
| 服务稳定性 | 错误率 < 0.01% | 压力测试 |

### 4.2 建议通过标准 (P1)

| 测试项 | 通过标准 | 测量方法 |
|-------|---------|---------|
| GPU吞吐量 | > 100 QPS | 吞吐量测试 |
| CPU吞吐量 | > 20 QPS | 吞吐量测试 |
| GPU显存 | < 2GB | 资源监控 |
| CPU内存 | < 1GB | 资源监控 |
| 模型大小 | < 500MB (压缩后) | 文件大小检查 |
| 冷启动时间 | < 5s | 首次加载测试 |

### 4.3 评估标准 (P2)

| 维度 | 权重 | 评分标准 |
|-----|-----|---------|
| 性能 | 40% | 延迟、吞吐量得分 |
| 精度 | 30% | 相似度、一致性得分 |
| 资源 | 15% | 内存、显存、模型大小得分 |
| 易用性 | 15% | 部署复杂度、文档质量得分 |

**总分计算：**
```
总分 = 性能得分 × 0.4 + 精度得分 × 0.3 + 资源得分 × 0.15 + 易用性得分 × 0.15
```

**推荐规则：**
- 总分 ≥ 85：强烈推荐
- 总分 70-84：推荐
- 总分 60-69：备选
- 总分 < 60：不推荐

---

## 5. 自动化建议

### 5.1 CI/CD 集成

```yaml
# .github/workflows/inference-engine-benchmark.yml
name: Inference Engine Benchmark

on:
  push:
    paths:
      - 'models/**'
      - 'inference/**'
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * 0'  # 每周执行

jobs:
  benchmark:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v3

      - name: Setup Environment
        run: |
          pip install -r requirements.txt
          pip install tensorrt onnxruntime-gpu openvino

      - name: Run Verification
        run: |
          python scripts/verify_engine.py --engine all --output results/

      - name: Generate Report
        run: |
          python scripts/generate_report.py --input results/ --output report.md

      - name: Upload Results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: results/

      - name: Comment PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            const report = require('./report.md');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: report
            });
```

### 5.2 自动化测试脚本

```bash
#!/bin/bash
# run_verification.sh - 完整验证流程

set -e

echo "=== 推理引擎验证测试 ==="

# 1. 环境检查
echo "[1/6] 检查环境..."
python scripts/check_environment.py

# 2. 准备测试数据
echo "[2/6] 准备测试数据..."
python scripts/prepare_test_dataset.py --output data/test_images/

# 3. GPU引擎测试
echo "[3/6] 测试GPU引擎..."
python scripts/verify_engine.py --engine tensorrt --output results/tensorrt.json
python scripts/verify_engine.py --engine onnxruntime-gpu --output results/onnx_gpu.json

# 4. CPU引擎测试
echo "[4/6] 测试CPU引擎..."
python scripts/verify_engine.py --engine onnxruntime-cpu --output results/onnx_cpu.json
python scripts/verify_engine.py --engine openvino --output results/openvino.json

# 5. 生成对比报告
echo "[5/6] 生成对比报告..."
python scripts/generate_comparison.py \
    --input results/*.json \
    --output docs/benchmark_report.md

# 6. 判定结果
echo "[6/6] 判定测试结果..."
python scripts/check_pass_criteria.py --results results/ --threshold thresholds.json

echo "=== 验证完成 ==="
```

### 5.3 监控指标采集

```python
# collect_metrics.py
import psutil
import pynvml
import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class SystemMetrics:
    timestamp: float
    cpu_percent: float
    memory_mb: float
    gpu_memory_mb: Optional[float]
    gpu_utilization: Optional[float]

class MetricsCollector:
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        if use_gpu:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def collect(self) -> SystemMetrics:
        """采集当前系统指标"""
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=psutil.cpu_percent(),
            memory_mb=psutil.Process().memory_info().rss / 1024 / 1024,
            gpu_memory_mb=self._get_gpu_memory() if self.use_gpu else None,
            gpu_utilization=self._get_gpu_util() if self.use_gpu else None
        )

    def _get_gpu_memory(self) -> float:
        info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
        return info.used / 1024 / 1024

    def _get_gpu_util(self) -> float:
        util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
        return util.gpu
```

---

## 6. 风险点与缓解措施

### 6.1 技术风险

| 风险 | 影响 | 可能性 | 缓解措施 |
|-----|------|-------|---------|
| **TensorRT算子不支持** | 高 | 中 | 提前验证ERes2NetV2所有算子在TensorRT中支持；准备自定义插件方案 |
| **精度损失过大** | 高 | 低 | 保留FP32作为备选；量化前做敏感层分析 |
| **GPU显存不足** | 中 | 低 | 支持动态batch；提供模型切片方案 |
| **跨平台兼容问题** | 中 | 中 | 每个平台独立测试；准备平台特定优化参数 |
| **依赖版本冲突** | 低 | 中 | 使用Docker容器隔离；锁定依赖版本 |

### 6.2 进度风险

| 风险 | 影响 | 可能性 | 缓解措施 |
|-----|------|-------|---------|
| **测试环境准备延迟** | 高 | 中 | 提前2周申请GPU资源；准备云环境备选方案 |
| **边缘设备不可用** | 中 | 低 | 使用模拟器测试；协调设备采购 |
| **模型转换超时** | 中 | 低 | 预留buffer时间；准备多种转换路径 |

### 6.3 质量风险

| 风险 | 影响 | 可能性 | 缓解措施 |
|-----|------|-------|---------|
| **测试覆盖不足** | 高 | 中 | 使用测试用例检查表；代码审查测试脚本 |
| **测试数据偏倚** | 中 | 中 | 确保测试数据多样性；包含边界情况 |
| **性能波动** | 中 | 高 | 多次测试取平均；固定测试环境配置 |

### 6.4 风险应对流程

```
风险发现 → 评估影响 → 确定优先级 → 制定方案 → 执行缓解 → 验证效果 → 更新文档
    ↓
记录到风险跟踪表
```

---

## 7. 验证交付物清单

### 7.1 必须交付

- [ ] **测试报告**
  - 引擎对比测试报告（Markdown/PDF）
  - 性能基准数据（JSON/CSV）
  - 精度验证报告

- [ ] **推荐方案文档**
  - GPU推理方案推荐及理由
  - CPU推理方案推荐及理由
  - 边缘设备推理方案推荐表

- [ ] **验证脚本**
  - 完整验证脚本套件
  - 使用说明文档

### 7.2 建议交付

- [ ] **可视化看板**
  - 性能对比图表
  - 资源占用趋势图

- [ ] **自动化配置**
  - CI/CD配置文件
  - 定时测试配置

- [ ] **问题跟踪**
  - 发现问题及解决方案文档
  - 优化建议文档

---

## 8. 时间计划

| 阶段 | 任务 | 预计时间 | 依赖 |
|-----|------|---------|-----|
| 第1天 | 环境准备、测试数据准备 | 4h | GPU资源到位 |
| 第2天 | GPU引擎测试 (TensorRT + ONNX) | 6h | 第1天完成 |
| 第3天 | CPU引擎测试 (ONNX + OpenVINO) | 4h | 无依赖 |
| 第3天 | 边缘引擎测试 (条件允许) | 4h | 设备可用 |
| 第4天 | 对比分析和报告编写 | 4h | 所有测试完成 |
| 第4天 | 方案推荐和文档整理 | 2h | 分析完成 |

**总预计时间：2-3个工作日**

---

## 9. 附录

### 9.1 测试环境要求

**GPU测试环境：**
- NVIDIA GPU (计算能力 7.0+)
- CUDA 11.8+
- cuDNN 8.6+
- TensorRT 8.6+
- 显存 ≥ 8GB

**CPU测试环境：**
- Intel Core i7/i9 或 AMD Ryzen 7/9
- 内存 ≥ 16GB
- 支持 AVX2/AVX-512

**边缘设备：**
- Android设备：Android 8.0+
- iOS设备：iOS 13.0+
- 嵌入式设备：ARM Cortex-A53+

### 9.2 参考文档

- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/)
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [OpenVINO Documentation](https://docs.openvino.ai/)
- [NCNN GitHub](https://github.com/Tencent/ncnn)
- [MNN Documentation](https://www.yuque.com/mnn/en)

---

**文档维护：**
- 验证完成后更新实际测试结果
- 记录发现的问题和解决方案
- 根据实际情况调整测试用例