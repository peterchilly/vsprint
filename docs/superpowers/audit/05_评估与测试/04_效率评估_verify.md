# 效率评估验证测试计划

> **文档类型：** 验证测试计划
> **关联任务：** 阶段五：评估与测试 - 效率评估
> **创建日期：** 2026/06/03
> **版本：** 1.0

---

## 1. 验证目标

### 1.1 主要目标
1. **验证推理延迟测量准确性** - 确保推理延迟（ms）测量方法正确，结果可复现
2. **验证吞吐量计算正确性** - 确认FPS计算方式合理，包含预热和批量处理
3. **验证FLOPs计算准确性** - 确保计算复杂度评估与理论值一致
4. **验证参数量统计完整性** - 确认参数统计包含所有可训练和不可训练参数
5. **验证显存/内存监控有效性** - 确保峰值内存监控方法可靠
6. **验证基线对比公平性** - 确保对比条件一致，结论可靠

### 1.2 量化指标
| 指标 | 目标值 |
|------|--------|
| 推理延迟测量精度 | ±1ms 或 ±5% |
| 吞吐量测量精度 | ±5 FPS 或 ±5% |
| FLOPs计算误差 | < 3% vs 理论值 |
| 参数量统计完整性 | 100% |
| 显存监控精度 | ±100MB |
| 基线模型对比维度 | ≥ 3个 |

---

## 2. 测试用例

### 2.1 推理延迟测试

#### TC-LATENCY-001: 单样本推理延迟测试
| 属性 | 内容 |
|------|------|
| **前置条件** | 模型已加载，CUDA已初始化，测试数据准备就绪 |
| **测试步骤** | 1. 执行预热推理 (≥10次)<br>2. 记录单样本推理时间 (≥100次取平均)<br>3. 计算均值、标准差、P50/P95/P99<br>4. 分别测试CPU和GPU延迟 |
| **输入数据** | 单个标准输入张量 (如 1x3x224x224) |
| **预期输出** | 延迟统计报告 (mean, std, P50, P95, P99) |
| **通过标准** | 预热后延迟稳定，标准差/均值 < 20% |

#### TC-LATENCY-002: 批量推理延迟测试
| 属性 | 内容 |
|------|------|
| **前置条件** | 模型支持批量推理 |
| **测试步骤** | 1. 测试不同batch size (1, 2, 4, 8, 16, 32)<br>2. 记录每个batch size的总延迟和单样本延迟<br>3. 分析batch size与延迟的关系 |
| **输入数据** | 不同batch size的输入张量 |
| **预期输出** | batch size-延迟曲线，最优batch size建议 |
| **通过标准** | 延迟随batch size增加呈合理增长趋势 |

#### TC-LATENCY-003: 首次推理延迟测试（冷启动）
| 属性 | 内容 |
|------|------|
| **前置条件** | 模型未加载或CUDA未初始化 |
| **测试步骤** | 1. 重新加载模型<br>2. 记录首次推理延迟<br>3. 重复多次取平均 |
| **输入数据** | 标准输入张量 |
| **预期输出** | 冷启动延迟报告 |
| **通过标准** | 冷启动延迟有明确记录，与热启动延迟分开报告 |

#### TC-LATENCY-004: CPU vs GPU延迟对比
| 属性 | 内容 |
|------|------|
| **前置条件** | 具备CPU和GPU测试环境 |
| **测试步骤** | 1. 在相同条件下分别测试CPU和GPU延迟<br>2. 记录硬件配置信息<br>3. 计算加速比 |
| **输入数据** | 标准输入张量 |
| **预期输出** | CPU/GPU延迟对比表，加速比分析 |
| **通过标准** | 加速比合理，与硬件规格匹配 |

### 2.2 吞吐量测试

#### TC-THROUGHPUT-001: 标准吞吐量测试
| 属性 | 内容 |
|------|------|
| **前置条件** | 模型已加载并预热 |
| **测试步骤** | 1. 执行预热 (≥10次)<br>2. 持续推理固定时长 (≥30秒) 或固定样本数 (≥1000)<br>3. 计算FPS = 样本数 / 总时间<br>4. 测试不同batch size |
| **输入数据** | 标准测试集或随机生成数据 |
| **预期输出** | FPS报告，包含不同batch size结果 |
| **通过标准** | FPS值稳定，波动 < 5% |

#### TC-THROUGHPUT-002: 多线程/多进程吞吐量测试
| 属性 | 内容 |
|------|------|
| **前置条件** | 支持并行推理 |
| **测试步骤** | 1. 测试不同并行度 (1, 2, 4, 8 workers)<br>2. 记录每个配置下的吞吐量<br>3. 分析并行效率 |
| **输入数据** | 标准测试数据 |
| **预期输出** | 并行度-FPS曲线，最优并行度建议 |
| **通过标准** | 并行效率合理，无明显性能下降 |

#### TC-THROUGHPUT-003: 端到端吞吐量测试
| 属性 | 内容 |
|------|------|
| **前置条件** | 具备完整推理管道 |
| **测试步骤** | 1. 测量包含预处理的端到端时间<br>2. 测量包含预处理+后处理的端到端时间<br>3. 分析各阶段时间占比 |
| **输入数据** | 原始图像数据 |
| **预期输出** | 端到端FPS，各阶段时间分解 |
| **通过标准** | 预处理时间占比合理 (< 30%) |

### 2.3 FLOPs和参数量测试

#### TC-FLOPS-001: FLOPs计算验证
| 属性 | 内容 |
|------|------|
| **前置条件** | 模型结构确定，输入尺寸确定 |
| **测试步骤** | 1. 使用标准工具计算FLOPs (如fvcore, ptflops, thop)<br>2. 与手工计算/文献值对比<br>3. 记录不同输入尺寸下的FLOPs |
| **输入数据** | 模型定义，标准输入尺寸 |
| **预期输出** | FLOPs报告，包含各层分解 |
| **通过标准** | 与理论值误差 < 5%，与文献值可比较 |

#### TC-FLOPS-002: MACs vs FLOPs区分
| 属性 | 内容 |
|------|------|
| **前置条件** | 了解MACs和FLOPs的区别 |
| **测试步骤** | 1. 分别计算MACs和FLOPs<br>2. 确认使用的定义一致<br>3. 在报告中明确标注 |
| **输入数据** | 模型定义 |
| **预期输出** | MACs和FLOPs报告，定义说明 |
| **通过标准** | 报告明确区分MACs (乘加操作数) 和FLOPs |

#### TC-PARAM-001: 参数量统计验证
| 属性 | 内容 |
|------|------|
| **前置条件** | 模型结构确定 |
| **测试步骤** | 1. 统计总参数量<br>2. 分别统计可训练参数和固定参数<br>3. 与模型定义对比验证 |
| **输入数据** | 模型定义 |
| **预期输出** | 参数量报告 (总参数、可训练参数、固定参数) |
| **通过标准** | 统计结果与model.parameters()一致 |

#### TC-PARAM-002: 模型体积估算
| 属性 | 内容 |
|------|------|
| **前置条件** | 模型可保存 |
| **测试步骤** | 1. 保存模型到磁盘<br>2. 测量实际文件大小<br>3. 与理论大小对比 (参数量 × 精度字节数) |
| **输入数据** | 模型权重 |
| **预期输出** | 模型文件大小报告 |
| **通过标准** | 实际大小与理论大小误差 < 10% |

### 2.4 GPU显存测试

#### TC-MEM-001: 静态显存占用测试
| 属性 | 内容 |
|------|------|
| **前置条件** | GPU环境可用，nvidia-smi可用 |
| **测试步骤** | 1. 记录GPU初始状态<br>2. 加载模型，记录显存增量<br>3. 与理论值对比 (参数量 × 字节数) |
| **输入数据** | 模型定义 |
| **预期输出** | 静态显存占用报告 |
| **通过标准** | 显存占用合理，与参数量匹配 |

#### TC-MEM-002: 动态显存峰值测试
| 属性 | 内容 |
|------|------|
| **前置条件** | GPU环境可用 |
| **测试步骤** | 1. 执行前向推理，记录显存峰值<br>2. 执行前向+反向，记录显存峰值<br>3. 测试不同batch size下的显存峰值 |
| **输入数据** | 不同batch size的输入张量 |
| **预期输出** | 显存峰值报告，batch size-显存曲线 |
| **通过标准** | 峰值显存 < GPU总显存 × 90% |

#### TC-MEM-003: 显存泄漏检测
| 属性 | 内容 |
|------|------|
| **前置条件** | 模型可多次推理 |
| **测试步骤** | 1. 执行大量推理迭代 (≥1000次)<br>2. 监控显存变化趋势<br>3. 检查是否有显存持续增长 |
| **输入数据** | 标准输入 |
| **预期输出** | 显存趋势分析报告 |
| **通过标准** | 显存稳定，无持续增长趋势 |

#### TC-MEM-004: CPU内存峰值测试
| 属性 | 内容 |
|------|------|
| **前置条件** | 内存监控工具可用 |
| **测试步骤** | 1. 使用memory_profiler或psutil监控<br>2. 记录推理过程内存峰值<br>3. 测试不同batch size |
| **输入数据** | 不同batch size的输入 |
| **预期输出** | CPU内存峰值报告 |
| **通过标准** | 内存峰值在合理范围内，无泄漏 |

### 2.5 基线模型对比测试

#### TC-BASELINE-001: 同架构不同规模对比
| 属性 | 内容 |
|------|------|
| **前置条件** | 具备同一架构不同规模的模型变体 |
| **测试步骤** | 1. 测试所有变体的效率指标<br>2. 生成对比表格<br>3. 分析效率-性能权衡 |
| **输入数据** | 多个模型变体，统一测试数据 |
| **预期输出** | 变体对比表，效率-性能曲线 |
| **通过标准** | 对比公平，指标完整 |

#### TC-BASELINE-002: 跨架构对比
| 属性 | 内容 |
|------|------|
| **前置条件** | 具备不同架构的基线模型 |
| **测试步骤** | 1. 确保测试条件一致 (硬件、数据、精度)<br>2. 测试所有模型的效率指标<br>3. 进行统计显著性检验 |
| **输入数据** | 多个模型，统一测试数据 |
| **预期输出** | 跨架构对比表，统计分析报告 |
| **通过标准** | 对比条件一致，结论有统计支撑 |

#### TC-BASELINE-003: 与文献基准对比
| 属性 | 内容 |
|------|------|
| **前置条件** | 收集相关文献的效率数据 |
| **测试步骤** | 1. 整理文献基准数据<br>2. 在相同条件下复现测试<br>3. 分析差异原因 |
| **输入数据** | 文献数据，自测数据 |
| **预期输出** | 文献对比表，差异分析 |
| **通过标准** | 差异可解释，或有明确复现原因 |

---

## 3. 验证方法

### 3.1 自动化验证方法

#### 推理延迟测量脚本
```python
# scripts/measure_latency.py
import torch
import time
import numpy as np
from typing import Tuple, Dict

def measure_latency(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    device: str = 'cuda',
    warmup_runs: int = 10,
    test_runs: int = 100,
    synchronize: bool = True
) -> Dict[str, float]:
    """
    测量模型推理延迟
    
    Args:
        model: 待测模型
        input_shape: 输入张量形状
        device: 设备类型
        warmup_runs: 预热次数
        test_runs: 测试次数
        synchronize: 是否同步CUDA
    
    Returns:
        包含延迟统计信息的字典
    """
    model.eval()
    model.to(device)
    
    # 准备输入
    dummy_input = torch.randn(input_shape, device=device)
    
    # 预热
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
            if synchronize and device == 'cuda':
                torch.cuda.synchronize()
    
    # 测试
    latencies = []
    with torch.no_grad():
        for _ in range(test_runs):
            if synchronize and device == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            _ = model(dummy_input)
            
            if synchronize and device == 'cuda':
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms
    
    latencies = np.array(latencies)
    
    return {
        'mean_ms': float(np.mean(latencies)),
        'std_ms': float(np.std(latencies)),
        'min_ms': float(np.min(latencies)),
        'max_ms': float(np.max(latencies)),
        'p50_ms': float(np.percentile(latencies, 50)),
        'p95_ms': float(np.percentile(latencies, 95)),
        'p99_ms': float(np.percentile(latencies, 99)),
        'cv': float(np.std(latencies) / np.mean(latencies))  # 变异系数
    }

def verify_latency_result(result: Dict[str, float]) -> Tuple[bool, str]:
    """验证延迟测试结果是否合理"""
    # 检查变异系数
    if result['cv'] > 0.3:
        return False, f"变异系数过高: {result['cv']:.2f} > 0.3，延迟不稳定"
    
    # 检查P99和均值的比值
    ratio = result['p99_ms'] / result['mean_ms']
    if ratio > 3:
        return False, f"P99/均值比值过高: {ratio:.2f} > 3，存在异常值"
    
    return True, "延迟测试结果合理"
```

#### FLOPs计算验证脚本
```python
# scripts/measure_flops.py
import torch
from typing import Dict, Tuple

def measure_flops(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    include_backward: bool = False
) -> Dict[str, float]:
    """
    计算模型FLOPs
    
    Args:
        model: 待测模型
        input_shape: 输入张量形状
        include_backward: 是否包含反向传播
    
    Returns:
        包含FLOPs信息的字典
    """
    try:
        from fvcore.nn import FlopCountAnalysis, parameter_count
    except ImportError:
        raise ImportError("请安装fvcore: pip install fvcore")
    
    model.eval()
    dummy_input = torch.randn(input_shape)
    
    # 计算FLOPs
    flop_counter = FlopCountAnalysis(model, dummy_input)
    total_flops = flop_counter.total()
    
    # 按层分解
    flops_by_layer = dict(flop_counter.by_module())
    
    # 参数量
    params = parameter_count(model)
    
    result = {
        'total_flops': total_flops,
        'total_gflops': total_flops / 1e9,
        'flops_by_layer': flops_by_layer,
        'total_params': params[''],
        'trainable_params': params['trainable'],
        'non_trainable_params': params['non_trainable'],
        'model_size_mb': params[''] * 4 / (1024 ** 2)  # 假设float32
    }
    
    if include_backward:
        result['total_flops_with_backward'] = total_flops * 3  # 近似
    
    return result

def verify_flops_result(result: Dict, expected_gflops: float = None) -> Tuple[bool, str]:
    """验证FLOPs结果"""
    if result['total_gflops'] <= 0:
        return False, "FLOPs值无效"
    
    if expected_gflops is not None:
        error = abs(result['total_gflops'] - expected_gflops) / expected_gflops
        if error > 0.1:
            return False, f"FLOPs与预期值误差过大: {error*100:.1f}%"
    
    return True, f"FLOPs计算合理: {result['total_gflops']:.2f}G"
```

#### 显存监控脚本
```python
# scripts/measure_memory.py
import torch
import subprocess
from typing import Dict, Tuple
import gc

def get_gpu_memory_info() -> Dict[str, float]:
    """获取GPU显存信息"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits'],
            capture_output=True, text=True
        )
        used, total = map(float, result.stdout.strip().split('\n')[1].split(','))
        return {
            'used_mb': used,
            'total_mb': total,
            'utilization': used / total * 100
        }
    except Exception as e:
        return {'error': str(e)}

def measure_memory_peak(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    include_backward: bool = False
) -> Dict[str, float]:
    """
    测量推理过程中的显存峰值
    
    Args:
        model: 待测模型
        input_shape: 输入形状
        include_backward: 是否包含反向传播
    
    Returns:
        显存使用信息
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()
    
    # 记录初始状态
    initial_memory = torch.cuda.memory_allocated() / 1024**2
    
    model.to('cuda')
    model.train() if include_backward else model.eval()
    
    dummy_input = torch.randn(input_shape, device='cuda')
    
    if include_backward:
        output = model(dummy_input)
        loss = output.sum()
        loss.backward()
    else:
        with torch.no_grad():
            _ = model(dummy_input)
    
    torch.cuda.synchronize()
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2
    current_memory = torch.cuda.memory_allocated() / 1024**2
    
    return {
        'initial_mb': initial_memory,
        'peak_mb': peak_memory,
        'delta_mb': peak_memory - initial_memory,
        'current_mb': current_memory,
        'include_backward': include_backward
    }

def check_memory_leak(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    iterations: int = 100
) -> Tuple[bool, str]:
    """检查是否存在显存泄漏"""
    model.eval()
    model.to('cuda')
    
    memory_samples = []
    
    for i in range(iterations):
        dummy_input = torch.randn(input_shape, device='cuda')
        with torch.no_grad():
            _ = model(dummy_input)
        
        if i % 10 == 0:
            current_mem = torch.cuda.memory_allocated() / 1024**2
            memory_samples.append(current_mem)
    
    # 检查内存是否持续增长
    if len(memory_samples) >= 3:
        trend = memory_samples[-1] - memory_samples[0]
        if trend > 100:  # 增长超过100MB
            return False, f"可能存在显存泄漏: {memory_samples[0]:.1f}MB -> {memory_samples[-1]:.1f}MB"
    
    return True, "显存稳定，未检测到泄漏"
```

### 3.2 手动验证方法

#### 报告完整性检查清单
- [ ] 推理延迟数据完整（包含均值、标准差、P50/P95/P99）
- [ ] 吞吐量数据完整（包含不同batch size结果）
- [ ] FLOPs/GMACs数据完整，定义明确
- [ ] 参数量数据完整（总参数、可训练参数）
- [ ] GPU显存数据完整（静态、动态峰值）
- [ ] CPU内存数据完整
- [ ] 基线对比表格完整
- [ ] 硬件配置信息完整

#### 结果合理性审查
- [ ] 延迟变异系数 < 20%
- [ ] FPS随batch size变化趋势合理
- [ ] FLOPs与模型复杂度匹配
- [ ] 参数量统计与模型定义一致
- [ ] 显存峰值在GPU容量范围内
- [ ] 基线对比条件一致

### 3.3 交叉验证方法

```python
# 验证工具一致性
def verify_tool_consistency():
    """使用不同工具交叉验证结果"""
    import torch
    from fvcore.nn import FlopCountAnalysis
    from thop import profile
    
    model = get_model()
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # 使用fvcore
    flops_fvcore = FlopCountAnalysis(model, dummy_input).total()
    
    # 使用thop
    flops_thop, _ = profile(model, inputs=(dummy_input,))
    
    # 验证一致性
    error = abs(flops_fvcore - flops_thop) / flops_fvcore
    assert error < 0.05, f"工具间误差过大: {error*100:.1f}%"
    
    return True
```

---

## 4. 通过标准

### 4.1 功能完整性标准
| 检查项 | 通过条件 | 优先级 |
|--------|----------|--------|
| 推理延迟测量 | 包含mean, std, P50, P95, P99 | P0 |
| 吞吐量测量 | 包含多种batch size结果 | P0 |
| FLOPs计算 | 结果合理，定义明确 | P0 |
| 参数量统计 | 总参数、可训练参数均完整 | P0 |
| GPU显存监控 | 静态和动态峰值均测量 | P0 |
| CPU内存监控 | 峰值内存有记录 | P1 |
| 基线对比 | ≥3个对比模型或≥3个指标维度 | P0 |

### 4.2 质量标准
| 检查项 | 通过条件 | 优先级 |
|--------|----------|--------|
| 测量可复现性 | 相同条件下结果差异 < 5% | P0 |
| 延迟稳定性 | 变异系数 < 20% | P0 |
| FLOPs验证 | 与理论值误差 < 5% | P1 |
| 参数验证 | 与model.parameters()一致 | P0 |
| 显存验证 | 无泄漏，峰值合理 | P0 |

### 4.3 报告标准
| 检查项 | 通过条件 | 优先级 |
|--------|----------|--------|
| 硬件配置记录 | GPU型号、CPU型号、内存大小 | P0 |
| 软件环境记录 | PyTorch版本、CUDA版本 | P0 |
| 测试方法说明 | 测量方法、预热策略、统计方法 | P1 |
| 结果可视化 | 包含对比图表 | P1 |
| 结论明确 | 效率分析结论清晰 | P0 |

### 4.4 性能基准参考
| 指标 | 参考范围 | 说明 |
|------|----------|------|
| 单样本推理延迟 | < 100ms (GPU) | 取决于模型复杂度 |
| 吞吐量 | > 100 FPS (GPU, batch=1) | 取决于模型复杂度 |
| FLOPs | 与同类模型比较 | ResNet50 ~4.1G |
| 参数量 | 与同类模型比较 | ResNet50 ~25M |
| GPU显存占用 | < GPU容量的80% | 推理时 |

---

## 5. 自动化建议

### 5.1 CI/CD 集成

```yaml
# .github/workflows/efficiency_eval.yml
name: Efficiency Evaluation

on:
  workflow_dispatch:
  push:
    paths: ['models/**', 'scripts/efficiency/**']
  pull_request:
    paths: ['models/**', 'scripts/efficiency/**']

jobs:
  efficiency-tests:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install fvcore thop memory_profiler

      - name: Check GPU availability
        run: python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

      - name: Run latency tests
        run: python scripts/measure_latency.py --config configs/efficiency.yaml --output results/latency.json

      - name: Run throughput tests
        run: python scripts/measure_throughput.py --config configs/efficiency.yaml --output results/throughput.json

      - name: Run FLOPs tests
        run: python scripts/measure_flops.py --config configs/efficiency.yaml --output results/flops.json

      - name: Run memory tests
        run: python scripts/measure_memory.py --config configs/efficiency.yaml --output results/memory.json

      - name: Generate efficiency report
        run: python scripts/generate_efficiency_report.py --results-dir results/ --output reports/efficiency_report.md

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: efficiency-results
          path: |
            results/
            reports/

      - name: Check thresholds
        run: python scripts/check_efficiency_thresholds.py --results-dir results/
```

### 5.2 效率评估脚本模板

```python
# scripts/evaluate_efficiency.py
"""
效率评估主脚本
整合所有效率指标的测量和验证
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn


class EfficiencyEvaluator:
    """效率评估器"""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.results = {}
    
    def evaluate_all(self) -> Dict[str, Any]:
        """执行所有效率评估"""
        print("=" * 50)
        print("开始效率评估")
        print("=" * 50)
        
        # 1. 参数量
        print("\n[1/5] 参数量统计...")
        self.results['parameters'] = self._evaluate_parameters()
        
        # 2. FLOPs
        print("\n[2/5] FLOPs计算...")
        self.results['flops'] = self._evaluate_flops()
        
        # 3. 推理延迟
        print("\n[3/5] 推理延迟测量...")
        self.results['latency'] = self._evaluate_latency()
        
        # 4. 吞吐量
        print("\n[4/5] 吞吐量测量...")
        self.results['throughput'] = self._evaluate_throughput()
        
        # 5. 显存
        print("\n[5/5] 显存测量...")
        self.results['memory'] = self._evaluate_memory()
        
        # 添加元信息
        self.results['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda,
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'config': self.config
        }
        
        return self.results
    
    def _evaluate_parameters(self) -> Dict[str, int]:
        """评估参数量"""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total': total,
            'trainable': trainable,
            'non_trainable': total - trainable,
            'size_mb': total * 4 / (1024 ** 2)  # float32
        }
    
    def _evaluate_flops(self) -> Dict[str, Any]:
        """评估FLOPs"""
        from fvcore.nn import FlopCountAnalysis
        
        input_shape = self.config.get('input_shape', (1, 3, 224, 224))
        dummy_input = torch.randn(input_shape)
        
        counter = FlopCountAnalysis(self.model, dummy_input)
        
        return {
            'total_flops': counter.total(),
            'gflops': counter.total() / 1e9,
            'input_shape': list(input_shape)
        }
    
    def _evaluate_latency(self) -> Dict[str, float]:
        """评估推理延迟"""
        device = self.config.get('device', 'cuda')
        warmup = self.config.get('warmup_runs', 10)
        runs = self.config.get('test_runs', 100)
        input_shape = self.config.get('input_shape', (1, 3, 224, 224))
        
        self.model.eval()
        self.model.to(device)
        
        dummy_input = torch.randn(input_shape, device=device)
        
        # 预热
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(dummy_input)
                if device == 'cuda':
                    torch.cuda.synchronize()
        
        # 测试
        latencies = []
        with torch.no_grad():
            for _ in range(runs):
                if device == 'cuda':
                    torch.cuda.synchronize()
                start = time.perf_counter()
                _ = self.model(dummy_input)
                if device == 'cuda':
                    torch.cuda.synchronize()
                end = time.perf_counter()
                latencies.append((end - start) * 1000)
        
        import numpy as np
        latencies = np.array(latencies)
        
        return {
            'mean_ms': float(np.mean(latencies)),
            'std_ms': float(np.std(latencies)),
            'p50_ms': float(np.percentile(latencies, 50)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'p99_ms': float(np.percentile(latencies, 99)),
            'device': device
        }
    
    def _evaluate_throughput(self) -> Dict[str, float]:
        """评估吞吐量"""
        device = self.config.get('device', 'cuda')
        input_shape = self.config.get('input_shape', (1, 3, 224, 224))
        batch_sizes = self.config.get('batch_sizes', [1, 2, 4, 8, 16])
        duration_sec = self.config.get('duration_sec', 10)
        
        self.model.eval()
        self.model.to(device)
        
        results = {}
        
        for bs in batch_sizes:
            shape = (bs,) + input_shape[1:]
            dummy_input = torch.randn(shape, device=device)
            
            # 预热
            with torch.no_grad():
                for _ in range(5):
                    _ = self.model(dummy_input)
                    if device == 'cuda':
                        torch.cuda.synchronize()
            
            # 测试
            count = 0
            start = time.perf_counter()
            end_time = start + duration_sec
            
            with torch.no_grad():
                while time.perf_counter() < end_time:
                    _ = self.model(dummy_input)
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    count += bs
            
            elapsed = time.perf_counter() - start
            fps = count / elapsed
            
            results[f'batch_{bs}'] = {
                'fps': fps,
                'samples': count,
                'elapsed_sec': elapsed
            }
        
        return results
    
    def _evaluate_memory(self) -> Dict[str, float]:
        """评估显存"""
        device = self.config.get('device', 'cuda')
        input_shape = self.config.get('input_shape', (1, 3, 224, 224))
        batch_sizes = self.config.get('batch_sizes', [1, 2, 4, 8, 16])
        
        self.model.to(device)
        
        results = {}
        
        for bs in batch_sizes:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
            shape = (bs,) + input_shape[1:]
            dummy_input = torch.randn(shape, device=device)
            
            initial = torch.cuda.memory_allocated() / 1024**2
            
            with torch.no_grad():
                _ = self.model(dummy_input)
            
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1024**2
            
            results[f'batch_{bs}'] = {
                'initial_mb': initial,
                'peak_mb': peak,
                'delta_mb': peak - initial
            }
        
        return results
    
    def save_results(self, output_path: str):
        """保存结果"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n结果已保存到: {path}")


def main():
    parser = argparse.ArgumentParser(description='效率评估')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output', type=str, default='results/efficiency.json')
    args = parser.parse_args()
    
    # 加载配置
    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # 加载模型
    model = load_model(config['model'])  # 需要实现
    model.eval()
    
    # 评估
    evaluator = EfficiencyEvaluator(model, config)
    results = evaluator.evaluate_all()
    
    # 保存
    evaluator.save_results(args.output)
    
    # 验证
    verify_results(results)


def verify_results(results: Dict[str, Any]) -> bool:
    """验证结果是否合理"""
    errors = []
    
    # 检查延迟稳定性
    latency = results.get('latency', {})
    if latency.get('std_ms', 0) / max(latency.get('mean_ms', 1), 1) > 0.3:
        errors.append("延迟变异系数过高")
    
    # 检查FLOPs
    flops = results.get('flops', {})
    if flops.get('gflops', 0) <= 0:
        errors.append("FLOPs值无效")
    
    # 检查参数量
    params = results.get('parameters', {})
    if params.get('total', 0) <= 0:
        errors.append("参数量无效")
    
    if errors:
        print("\n验证失败:")
        for e in errors:
            print(f"  - {e}")
        return False
    
    print("\n验证通过!")
    return True


if __name__ == '__main__':
    main()
```

### 5.3 报告生成脚本

```python
# scripts/generate_efficiency_report.py
"""
生成效率评估报告
"""

import json
import argparse
from pathlib import Path
from datetime import datetime


def generate_report(results_dir: str, output_path: str):
    """生成Markdown格式的效率评估报告"""
    
    # 加载结果
    results = {}
    for file in Path(results_dir).glob('*.json'):
        with open(file) as f:
            results[file.stem] = json.load(f)
    
    # 生成报告
    report = []
    report.append("# 效率评估报告\n")
    report.append(f"\n> **生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. 环境信息
    report.append("\n## 1. 测试环境\n")
    meta = results.get('latency', {}).get('metadata', {})
    report.append(f"| 项目 | 配置 |\n|------|------|\n")
    report.append(f"| PyTorch版本 | {meta.get('pytorch_version', 'N/A')} |\n")
    report.append(f"| CUDA版本 | {meta.get('cuda_version', 'N/A')} |\n")
    report.append(f"| GPU | {meta.get('gpu_name', 'N/A')} |\n")
    
    # 2. 参数量和FLOPs
    report.append("\n## 2. 模型规模\n")
    params = results.get('parameters', {})
    flops = results.get('flops', {})
    report.append(f"| 指标 | 数值 |\n|------|------|\n")
    report.append(f"| 参数量 | {params.get('total', 0)/1e6:.2f}M |\n")
    report.append(f"| 可训练参数 | {params.get('trainable', 0)/1e6:.2f}M |\n")
    report.append(f"| FLOPs | {flops.get('gflops', 0):.2f}G |\n")
    report.append(f"| 模型大小 | {params.get('size_mb', 0):.2f}MB |\n")
    
    # 3. 推理延迟
    report.append("\n## 3. 推理延迟\n")
    latency = results.get('latency', {})
    report.append(f"| 指标 | 数值 |\n|------|------|\n")
    report.append(f"| 平均延迟 | {latency.get('mean_ms', 0):.2f}ms |\n")
    report.append(f"| 标准差 | {latency.get('std_ms', 0):.2f}ms |\n")
    report.append(f"| P50 | {latency.get('p50_ms', 0):.2f}ms |\n")
    report.append(f"| P95 | {latency.get('p95_ms', 0):.2f}ms |\n")
    report.append(f"| P99 | {latency.get('p99_ms', 0):.2f}ms |\n")
    
    # 4. 吞吐量
    report.append("\n## 4. 吞吐量\n")
    throughput = results.get('throughput', {})
    report.append(f"| Batch Size | FPS |\n|------|------|\n")
    for key, value in sorted(throughput.items()):
        if key.startswith('batch_'):
            bs = key.replace('batch_', '')
            report.append(f"| {bs} | {value.get('fps', 0):.1f} |\n")
    
    # 5. 显存占用
    report.append("\n## 5. 显存占用\n")
    memory = results.get('memory', {})
    report.append(f"| Batch Size | 峰值显存 (MB) |\n|------|------|\n")
    for key, value in sorted(memory.items()):
        if key.startswith('batch_'):
            bs = key.replace('batch_', '')
            report.append(f"| {bs} | {value.get('peak_mb', 0):.1f} |\n")
    
    # 6. 基线对比
    # ... (如果有基线数据)
    
    # 写入文件
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(report))
    
    print(f"报告已生成: {output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--output', default='reports/efficiency_report.md')
    args = parser.parse_args()
    
    generate_report(args.results_dir, args.output)
```

---

## 6. 风险点与缓解措施

### 6.1 技术风险

| 风险ID | 风险描述 | 可能性 | 影响 | 缓解措施 |
|--------|---------|--------|-----|---------|
| R-TECH-001 | CUDA预热不充分导致延迟测量不准 | 高 | 高 | 增加预热次数(≥10)，使用CUDA同步 |
| R-TECH-002 | 后台进程干扰导致测量波动 | 中 | 中 | 独占GPU测试，关闭不必要进程 |
| R-TECH-003 | 不同FLOPs计算工具结果不一致 | 高 | 中 | 统一使用fvcore，明确标注工具版本 |
| R-TECH-004 | 显存碎片导致显存占用偏高 | 中 | 低 | 测试前清理显存，多次测量取最小值 |
| R-TECH-005 | Batch size过大导致OOM | 中 | 高 | 逐步增加batch size，捕获OOM异常 |
| R-TECH-006 | 模型在CPU和GPU行为不一致 | 低 | 中 | 分别测试并记录，明确标注设备 |

### 6.2 方法风险

| 风险ID | 风险描述 | 可能性 | 影响 | 缓解措施 |
|--------|---------|--------|-----|---------|
| R-METH-001 | FLOPs定义不统一(MACs vs FLOPs) | 高 | 中 | 明确定义，在报告中说明 |
| R-METH-002 | 延迟测量包含预处理时间 | 中 | 高 | 分离预处理和推理时间测量 |
| R-METH-003 | 基线对比条件不一致 | 中 | 高 | 使用相同硬件、相同数据、相同精度 |
| R-METH-004 | 吞吐量测量受数据加载影响 | 中 | 中 | 使用内存数据，排除IO影响 |
| R-METH-005 | 参数量统计遗漏BN等参数 | 低 | 中 | 使用标准工具，交叉验证 |

### 6.3 环境风险

| 风险ID | 风险描述 | 可能性 | 影响 | 缓解措施 |
|--------|---------|--------|-----|---------|
| R-ENV-001 | GPU型号不同导致结果不可比 | 高 | 高 | 记录详细硬件信息，提供标准化基准 |
| R-ENV-002 | CUDA/PyTorch版本差异 | 中 | 中 | 记录软件版本，提供版本兼容性说明 |
| R-ENV-003 | 测试时间过长影响效率 | 中 | 低 | 并行化测试，增量评估 |
| R-ENV-004 | 显存不足无法完成大batch测试 | 中 | 中 | 实现batch size自动降级 |

### 6.4 风险优先级矩阵

```
                    影响程度
              低         中         高
         ┌─────────┬─────────┬─────────┐
    高   │R-METH-001│R-TECH-002│R-TECH-001│
    可       │R-ENV-001│R-METH-004│R-METH-002│
    能       ├─────────┼─────────┼─────────┤
    性       中   │R-TECH-004│R-TECH-003│R-TECH-005│
         │   │R-ENV-003│R-METH-003│R-ENV-004│
         │   │         │R-ENV-002│         │
         ├─────────┼─────────┼─────────┤
    低   │R-TECH-006│R-METH-005│         │
         │         │         │         │
         └─────────┴─────────┴─────────┘
```

---

## 7. 验收检查清单

### 7.1 交付物检查
- [ ] 效率评估报告 (Markdown/PDF格式)
- [ ] 推理延迟数据 (JSON格式)
- [ ] 吞吐量数据 (JSON格式)
- [ ] FLOPs计算结果 (JSON格式)
- [ ] 参数量统计结果
- [ ] 显存占用数据
- [ ] CPU内存占用数据 (可选)
- [ ] 基线对比表格
- [ ] 评估脚本代码

### 7.2 功能验收
- [ ] 推理延迟测量完成，包含P50/P95/P99
- [ ] 吞吐量测量完成，覆盖多种batch size
- [ ] FLOPs计算完成，定义明确
- [ ] 参数量统计完成，包含可训练参数
- [ ] GPU显存峰值测量完成
- [ ] CPU内存峰值测量完成 (可选)
- [ ] 基线对比完成，至少3个维度

### 7.3 质量验收
- [ ] 延迟测量变异系数 < 20%
- [ ] 吞吐量测量波动 < 5%
- [ ] FLOPs与理论值误差 < 5%
- [ ] 参数量与模型定义一致
- [ ] 显存无泄漏
- [ ] 结果可复现

### 7.4 文档验收
- [ ] 硬件配置完整记录
- [ ] 软件环境完整记录
- [ ] 测试方法清晰说明
- [ ] 结果可视化图表清晰
- [ ] 结论明确，有改进建议

---

## 8. 参考资源

### 8.1 工具库
- `fvcore.nn.FlopCountAnalysis`: Facebook的FLOPs计算工具
- `thop`: PyTorch操作分析器
- `torch.profiler`: PyTorch内置性能分析器
- `memory_profiler`: Python内存分析
- `nvidia-smi`: NVIDIA显存监控
- `py3nvml`: Python NVIDIA监控库

### 8.2 标准与基准
- [PyTorch Model Zoo](https://pytorch.org/vision/stable/models.html): 标准模型基准
- [TorchBench](https://github.com/pytorch/benchmark): PyTorch基准测试
- [ONNX Model Zoo](https://github.com/onnx/models): ONNX模型基准

### 8.3 最佳实践
- CUDA预热至少10次推理
- 延迟测试至少100次取平均
- 使用`torch.cuda.synchronize()`确保精确计时
- 测试前清理显存碎片
- 记录完整的软硬件环境

---

**文档历史**
| 版本 | 日期 | 修改内容 | 作者 |
|------|------|----------|------|
| 1.0 | 2026/06/03 | 初始版本 | Claude |

**审批状态**: 待审批