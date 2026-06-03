# 模型导出与优化 - 验证测试计划

> **文档版本：** v1.0
> **创建日期：** 2026-06-03
> **关联任务：** 阶段六：部署与推理 - 模型导出与优化

---

## 1. 验证目标

### 1.1 核心目标
| 目标ID | 目标描述 | 优先级 |
|--------|----------|--------|
| O-01 | 验证 PyTorch 模型能成功转换为 ONNX 格式 | P0 |
| O-02 | 验证 ONNX 模型能成功转换为 TensorRT 引擎 | P0 |
| O-03 | 验证 FP16 量化后精度损失在可接受范围内 | P0 |
| O-04 | 验证 INT8 量化后精度损失在可接受范围内 | P1 |
| O-05 | 验证量化后模型推理性能达到预期提升 | P0 |
| O-06 | 验证导出模型在各平台的兼容性 | P1 |
| O-07 | 验证通道剪枝后的模型精度（如执行） | P2 |

### 1.2 验证范围
- **输入：** 训练完成的 ERes2NetV2 模型文件（.pt / .pth）
- **输出：** ONNX 模型文件、TensorRT 引擎文件
- **量化方案：** FP16、INT8（PTQ/QAT）
- **可选优化：** 通道剪枝

---

## 2. 测试用例

### 2.1 PyTorch -> ONNX 转换测试

| 用例ID | 测试项 | 前置条件 | 测试步骤 | 预期结果 |
|--------|--------|----------|----------|----------|
| TC-ONNX-01 | 基础模型导出 | 已有训练好的 .pth 模型 | 1. 加载 PyTorch 模型<br>2. 调用 torch.onnx.export()<br>3. 检查输出文件 | 生成有效的 .onnx 文件，无报错 |
| TC-ONNX-02 | 动态输入维度 | 模型支持可变 batch size | 1. 设置动态轴 dynamic_axes<br>2. 导出 ONNX<br>3. 验证不同 batch 输入 | 支持指定范围内的动态输入 |
| TC-ONNX-03 | 算子兼容性 | 模型使用标准算子 | 1. 导出 ONNX<br>2. 使用 onnx.checker 验证<br>3. 检查算子支持列表 | 所有算子在 TensorRT 支持列表内 |
| TC-ONNX-04 | 数值一致性验证 | 成功导出 ONNX | 1. 准备测试输入<br>2. PyTorch 推理<br>3. ONNX Runtime 推理<br>4. 比较输出差异 | 输出差异 < 1e-5 (相对误差) |
| TC-ONNX-05 | 模型简化 | 已有 .onnx 文件 | 1. 使用 onnx-simplifier<br>2. 验证简化后模型 | 模型大小减小，推理正确 |

### 2.2 ONNX -> TensorRT 转换测试

| 用例ID | 测试项 | 前置条件 | 测试步骤 | 预期结果 |
|--------|--------|----------|----------|----------|
| TC-TRT-01 | FP32 引擎构建 | 已有有效 ONNX 模型 | 1. 调用 trtexec 构建 FP32 引擎<br>2. 检查构建日志 | 成功生成 .engine 文件，无报错 |
| TC-TRT-02 | FP16 引擎构建 | 已有有效 ONNX 模型 | 1. 设置 --fp16 标志<br>2. 构建引擎<br>3. 检查构建日志 | 成功生成 FP16 引擎 |
| TC-TRT-03 | INT8 校准 | 有校准数据集 | 1. 准备校准数据<br>2. 运行 INT8 校准<br>3. 生成校准表 | 校准完成，生成 .cache 文件 |
| TC-TRT-04 | INT8 引擎构建 | 已完成 INT8 校准 | 1. 设置 --int8 标志<br>2. 加载校准表<br>3. 构建引擎 | 成功生成 INT8 引擎 |
| TC-TRT-05 | 数值一致性验证 | 已有各精度引擎 | 1. 准备测试输入<br>2. 分别运行 FP32/FP16/INT8<br>3. 比较输出差异 | FP16 差异 < 1e-3, INT8 差异 < 1e-2 |

### 2.3 精度验证测试

| 用例ID | 测试项 | 前置条件 | 测试步骤 | 预期结果 |
|--------|--------|----------|----------|----------|
| TC-ACC-01 | 基准精度测量 | 原始 PyTorch 模型 | 1. 在验证集上推理<br>2. 计算 Top-1/Top-5 准确率<br>3. 记录基准值 | 获得基准精度指标 |
| TC-ACC-02 | FP16 精度评估 | FP16 TensorRT 引擎 | 1. 使用相同验证集<br>2. 计算 Top-1/Top-5 准确率<br>3. 与基准对比 | 准确率损失 < 0.5% |
| TC-ACC-03 | INT8 精度评估 | INT8 TensorRT 引擎 | 1. 使用相同验证集<br>2. 计算 Top-1/Top-5 准确率<br>3. 与基准对比 | 准确率损失 < 2% |
| TC-ACC-04 | 边界情况测试 | 各精度引擎 | 1. 准备极端输入（全零、全一、随机）<br>2. 检查输出稳定性 | 无 NaN/Inf，输出范围合理 |
| TC-ACC-05 | 批量推理精度 | 各精度引擎 | 1. 测试不同 batch size<br>2. 验证批量推理一致性 | 各 batch 输出与单样本一致 |

### 2.4 性能验证测试

| 用例ID | 测试项 | 前置条件 | 测试步骤 | 预期结果 |
|--------|--------|----------|----------|----------|
| TC-PERF-01 | FP32 基准性能 | FP32 TensorRT 引擎 | 1. 预热推理<br>2. 测量延迟和吞吐<br>3. 记录 GPU 显存占用 | 记录基准性能数据 |
| TC-PERF-02 | FP16 性能提升 | FP16 TensorRT 引擎 | 1. 同样测试流程<br>2. 与 FP32 对比 | 延迟降低 20%+，显存减少 50% |
| TC-PERF-03 | INT8 性能提升 | INT8 TensorRT 引擎 | 1. 同样测试流程<br>2. 与 FP32 对比 | 延迟降低 40%+，显存减少 75% |
| TC-PERF-04 | 动态 batch 性能 | 支持动态 batch 引擎 | 1. 测试 batch=1,4,8,16,32<br>2. 绘制性能曲线 | 吞吐随 batch 线性增长 |
| TC-PERF-05 | 并发推理测试 | TensorRT 引擎 | 1. 多线程/多流推理<br>2. 测量并发性能 | GPU 利用率提升，吞吐增加 |

### 2.5 通道剪枝测试（可选）

| 用例ID | 测试项 | 前置条件 | 测试步骤 | 预期结果 |
|--------|--------|----------|----------|----------|
| TC-PRUNE-01 | 剪枝比例确定 | 原始模型 | 1. 分析各层通道重要性<br>2. 确定剪枝比例 | 获得剪枝配置 |
| TC-PRUNE-02 | 模型剪枝执行 | 剪枝配置 | 1. 执行通道剪枝<br>2. 保存剪枝模型 | 成功生成剪枝模型 |
| TC-PRUNE-03 | 剪枝后微调 | 剪枝模型 | 1. 微调训练<br>2. 评估精度 | 精度恢复到原模型 98%+ |
| TC-PRUNE-04 | 剪枝模型导出 | 微调后模型 | 1. 导出 ONNX<br>2. 转换 TensorRT | 成功导出所有格式 |

---

## 3. 验证方法

### 3.1 自动化验证脚本

```bash
# 目录结构
scripts/
├── verify_onnx_export.py      # ONNX 导出验证
├── verify_trt_build.py        # TensorRT 构建验证
├── verify_accuracy.py         # 精度验证
├── verify_performance.py      # 性能验证
└── run_all_verification.sh    # 一键验证脚本
```

### 3.2 验证命令示例

```bash
# ONNX 导出验证
python scripts/verify_onnx_export.py \
    --pytorch_model checkpoints/eres2netv2_best.pth \
    --onnx_model models/eres2netv2.onnx \
    --input_shape 1,3,224,224 \
    --opset_version 17

# TensorRT 构建验证
python scripts/verify_trt_build.py \
    --onnx_model models/eres2netv2.onnx \
    --engine_output models/trt_engines/ \
    --precision fp16,int8 \
    --calibration_data data/calibration/

# 精度验证
python scripts/verify_accuracy.py \
    --original_model checkpoints/eres2netv2_best.pth \
    --trt_engines models/trt_engines/ \
    --val_data data/val/ \
    --batch_size 32 \
    --output accuracy_report.json

# 性能验证
python scripts/verify_performance.py \
    --trt_engines models/trt_engines/ \
    --batch_sizes 1,4,8,16,32 \
    --warmup 100 \
    --iterations 1000 \
    --output performance_report.json
```

### 3.3 数值一致性验证方法

```python
# verify_numerical_consistency.py 核心逻辑
def verify_consistency(pytorch_output, trt_output, tolerance):
    """
    验证 PyTorch 和 TensorRT 输出的数值一致性
    """
    # 相对误差计算
    relative_error = np.abs(pytorch_output - trt_output) / (np.abs(pytorch_output) + 1e-8)

    # 最大相对误差
    max_relative_error = np.max(relative_error)

    # 平均相对误差
    mean_relative_error = np.mean(relative_error)

    # 余弦相似度
    cosine_similarity = np.dot(pytorch_output.flatten(), trt_output.flatten()) / \
                       (np.linalg.norm(pytorch_output) * np.linalg.norm(trt_output))

    return {
        'max_relative_error': max_relative_error,
        'mean_relative_error': mean_relative_error,
        'cosine_similarity': cosine_similarity,
        'passed': max_relative_error < tolerance
    }
```

### 3.4 性能测试方法

```python
# verify_performance.py 核心逻辑
def benchmark_inference(engine, input_shape, warmup=100, iterations=1000):
    """
    TensorRT 推理性能基准测试
    """
    # GPU 预热
    for _ in range(warmup):
        _ = engine.infer(random_input)

    # 同步计时
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iterations):
        _ = engine.infer(random_input)

    torch.cuda.synchronize()
    end = time.perf_counter()

    # 计算指标
    latency_ms = (end - start) / iterations * 1000
    throughput = iterations / (end - start)

    return {
        'latency_ms': latency_ms,
        'throughput_qps': throughput,
        'iterations': iterations
    }
```

---

## 4. 通过标准

### 4.1 模型转换通过标准

| 检查项 | 通过条件 | 阻塞级别 |
|--------|----------|----------|
| ONNX 导出成功 | 无错误，生成有效 .onnx 文件 | 阻塞 |
| ONNX 算子兼容性 | 所有算子被 TensorRT 支持 | 阻塞 |
| TensorRT 引擎构建 | FP32/FP16/INT8 均成功构建 | 阻塞 |
| 数值一致性 (FP32) | 相对误差 < 1e-5 | 阻塞 |
| 数值一致性 (FP16) | 相对误差 < 1e-3 | 阻塞 |
| 数值一致性 (INT8) | 相对误差 < 1e-2 | 阻塞 |

### 4.2 精度损失通过标准

| 量化方案 | Top-1 准确率损失 | Top-5 准确率损失 | 阻塞级别 |
|----------|-----------------|-----------------|----------|
| FP16 | < 0.5% | < 0.3% | 阻塞 |
| INT8 (PTQ) | < 2.0% | < 1.5% | 阻塞 |
| INT8 (QAT) | < 1.0% | < 0.5% | 阻塞 |
| 通道剪枝+微调 | < 2.0% | < 1.5% | 非阻塞 |

### 4.3 性能提升通过标准

| 指标 | FP16 vs FP32 | INT8 vs FP32 | 阻塞级别 |
|------|--------------|--------------|----------|
| 推理延迟降低 | >= 20% | >= 40% | 非阻塞 |
| 显存占用减少 | >= 50% | >= 75% | 非阻塞 |
| 吞吐量提升 | >= 1.2x | >= 1.5x | 非阻塞 |
| GPU 利用率 | >= 80% | >= 85% | 非阻塞 |

### 4.4 稳定性通过标准

| 检查项 | 通过条件 | 阻塞级别 |
|--------|----------|----------|
| 输出无 NaN/Inf | 100% 测试样本通过 | 阻塞 |
| 批量推理一致性 | 各 batch size 结果一致 | 阻塞 |
| 并发推理稳定性 | 10 次运行方差 < 5% | 非阻塞 |
| 长时间运行稳定性 | 10000 次推理无错误 | 非阻塞 |

---

## 5. 自动化建议

### 5.1 CI/CD 集成

```yaml
# .github/workflows/model_export_verification.yml
name: Model Export Verification

on:
  push:
    paths:
      - 'models/**'
      - 'scripts/verify_*.py'
  pull_request:
    branches: [main]
  workflow_dispatch:
    inputs:
      model_path:
        description: 'Path to PyTorch model'
        required: true

jobs:
  verify-onnx:
    runs-on: gpu-runner
    steps:
      - uses: actions/checkout@v3
      - name: Setup environment
        run: |
          pip install -r requirements.txt
          pip install onnx onnxruntime onnx-simplifier

      - name: Verify ONNX Export
        run: |
          python scripts/verify_onnx_export.py \
            --pytorch_model ${{ inputs.model_path || 'checkpoints/eres2netv2_best.pth' }} \
            --output onnx_verification_report.json

      - name: Upload verification report
        uses: actions/upload-artifact@v3
        with:
          name: onnx-verification
          path: onnx_verification_report.json

  verify-tensorrt:
    runs-on: gpu-runner
    needs: verify-onnx
    steps:
      - name: Verify TensorRT Build
        run: |
          python scripts/verify_trt_build.py \
            --onnx_model models/eres2netv2.onnx \
            --precision fp16,int8 \
            --output trt_verification_report.json

      - name: Upload verification report
        uses: actions/upload-artifact@v3
        with:
          name: trt-verification
          path: trt_verification_report.json

  verify-accuracy:
    runs-on: gpu-runner
    needs: verify-tensorrt
    steps:
      - name: Verify Accuracy
        run: |
          python scripts/verify_accuracy.py \
            --trt_engines models/trt_engines/ \
            --val_data data/val/ \
            --output accuracy_report.json

      - name: Check accuracy threshold
        run: |
          python scripts/check_accuracy_threshold.py accuracy_report.json

  verify-performance:
    runs-on: gpu-runner
    needs: verify-tensorrt
    steps:
      - name: Verify Performance
        run: |
          python scripts/verify_performance.py \
            --trt_engines models/trt_engines/ \
            --output performance_report.json
```

### 5.2 一键验证脚本

```bash
#!/bin/bash
# scripts/run_all_verification.sh

set -e

MODEL_PATH=${1:-"checkpoints/eres2netv2_best.pth"}
VAL_DATA=${2:-"data/val/"}
REPORT_DIR="verification_reports/$(date +%Y%m%d_%H%M%S)"

mkdir -p $REPORT_DIR

echo "=========================================="
echo "Model Export & Optimization Verification"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Val Data: $VAL_DATA"
echo "Report Dir: $REPORT_DIR"
echo ""

# Step 1: ONNX Export
echo "[1/5] Verifying ONNX Export..."
python scripts/verify_onnx_export.py \
    --pytorch_model $MODEL_PATH \
    --output $REPORT_DIR/onnx_report.json
echo "✓ ONNX Export Verification Complete"

# Step 2: TensorRT Build
echo "[2/5] Verifying TensorRT Build..."
python scripts/verify_trt_build.py \
    --onnx_model models/eres2netv2.onnx \
    --precision fp16,int8 \
    --output $REPORT_DIR/trt_report.json
echo "✓ TensorRT Build Verification Complete"

# Step 3: Accuracy
echo "[3/5] Verifying Accuracy..."
python scripts/verify_accuracy.py \
    --trt_engines models/trt_engines/ \
    --val_data $VAL_DATA \
    --output $REPORT_DIR/accuracy_report.json
echo "✓ Accuracy Verification Complete"

# Step 4: Performance
echo "[4/5] Verifying Performance..."
python scripts/verify_performance.py \
    --trt_engines models/trt_engines/ \
    --output $REPORT_DIR/performance_report.json
echo "✓ Performance Verification Complete"

# Step 5: Generate Summary
echo "[5/5] Generating Summary Report..."
python scripts/generate_summary_report.py \
    --report_dir $REPORT_DIR \
    --output $REPORT_DIR/summary_report.md

echo ""
echo "=========================================="
echo "Verification Complete!"
echo "=========================================="
echo "Reports saved to: $REPORT_DIR"
cat $REPORT_DIR/summary_report.md
```

### 5.3 验证报告模板

```json
{
  "verification_id": "VERIFY_20260603_001",
  "timestamp": "2026-06-03T10:30:00Z",
  "model_info": {
    "name": "ERes2NetV2",
    "pytorch_path": "checkpoints/eres2netv2_best.pth",
    "onnx_path": "models/eres2netv2.onnx",
    "trt_engines": {
      "fp32": "models/trt_engines/eres2netv2_fp32.engine",
      "fp16": "models/trt_engines/eres2netv2_fp16.engine",
      "int8": "models/trt_engines/eres2netv2_int8.engine"
    }
  },
  "onnx_verification": {
    "status": "PASSED",
    "export_success": true,
    "operator_compatibility": 100,
    "numerical_consistency": {
      "max_relative_error": 3.2e-6,
      "mean_relative_error": 8.7e-7,
      "passed": true
    }
  },
  "trt_verification": {
    "status": "PASSED",
    "fp32_engine": {"status": "PASSED", "build_time_s": 12.5},
    "fp16_engine": {"status": "PASSED", "build_time_s": 15.2},
    "int8_engine": {"status": "PASSED", "build_time_s": 45.8}
  },
  "accuracy_verification": {
    "status": "PASSED",
    "baseline": {"top1": 87.5, "top5": 96.8},
    "fp16": {"top1": 87.3, "top5": 96.7, "loss": {"top1": 0.2, "top5": 0.1}},
    "int8": {"top1": 86.1, "top5": 95.9, "loss": {"top1": 1.4, "top5": 0.9}}
  },
  "performance_verification": {
    "status": "PASSED",
    "fp32": {"latency_ms": 15.2, "throughput_qps": 65.8, "memory_mb": 2048},
    "fp16": {"latency_ms": 8.5, "throughput_qps": 117.6, "memory_mb": 1024},
    "int8": {"latency_ms": 5.8, "throughput_qps": 172.4, "memory_mb": 512}
  },
  "overall_status": "PASSED",
  "summary": {
    "total_tests": 20,
    "passed": 20,
    "failed": 0,
    "warnings": 0
  }
}
```

---

## 6. 风险点

### 6.1 技术风险

| 风险ID | 风险描述 | 可能性 | 影响 | 缓解措施 |
|--------|----------|--------|------|----------|
| R-01 | 自定义算子不支持 ONNX/TensorRT | 中 | 高 | 1. 提前检查算子兼容性<br>2. 准备自定义算子实现<br>3. 考虑替代算子 |
| R-02 | INT8 量化精度损失超预期 | 中 | 高 | 1. 使用 QAT 替代 PTQ<br>2. 增加校准数据量<br>3. 调整量化策略 |
| R-03 | 动态 shape 处理问题 | 低 | 中 | 1. 仔细设计 dynamic_axes<br>2. 充分测试各输入尺寸 |
| R-04 | TensorRT 版本兼容性 | 低 | 中 | 1. 固定 TensorRT 版本<br>2. 记录版本依赖 |
| R-05 | GPU 显存不足 | 低 | 高 | 1. 优化输入尺寸<br>2. 使用模型切片<br>3. 选择合适 GPU 型号 |

### 6.2 流程风险

| 风险ID | 风险描述 | 可能性 | 影响 | 缓解措施 |
|--------|----------|--------|------|----------|
| R-06 | 校准数据集代表性不足 | 中 | 高 | 1. 使用完整训练数据分布<br>2. 包含边界样本<br>3. 多次校准验证 |
| R-07 | 验证集与生产数据分布不一致 | 中 | 中 | 1. 使用生产数据子集验证<br>2. 持续监控线上精度 |
| R-08 | 性能测试环境与生产不一致 | 低 | 中 | 1. 标准化测试环境<br>2. 记录硬件配置<br>3. 预留性能余量 |

### 6.3 回归测试风险

| 风险ID | 风险描述 | 可能性 | 影响 | 缓解措施 |
|--------|----------|--------|------|----------|
| R-09 | 模型更新后未重新验证 | 中 | 高 | 1. CI/CD 自动触发验证<br>2. 模型版本管理<br>3. 验证报告归档 |
| R-10 | 跳过某些测试用例 | 低 | 中 | 1. 强制测试覆盖率检查<br>2. 阻塞级别严格执行 |

---

## 7. 验证检查清单

### 7.1 导出前检查

- [ ] PyTorch 模型加载正常
- [ ] 模型权重完整性校验
- [ ] 输入输出格式确认
- [ ] 算子兼容性预检查
- [ ] 测试数据集准备完毕

### 7.2 ONNX 导出检查

- [ ] ONNX 导出无报错
- [ ] ONNX 文件大小合理
- [ ] onnx.checker 验证通过
- [ ] ONNX Runtime 推理正常
- [ ] 数值一致性测试通过

### 7.3 TensorRT 转换检查

- [ ] FP32 引擎构建成功
- [ ] FP16 引擎构建成功
- [ ] INT8 校准完成
- [ ] INT8 引擎构建成功
- [ ] 各精度数值验证通过

### 7.4 精度验证检查

- [ ] 基准精度已记录
- [ ] FP16 精度损失 < 0.5%
- [ ] INT8 精度损失 < 2%
- [ ] 边界情况测试通过
- [ ] 批量推理一致性验证

### 7.5 性能验证检查

- [ ] 延迟达到预期目标
- [ ] 吞吐量达到预期目标
- [ ] 显存占用合理
- [ ] 并发推理正常
- [ ] 性能报告生成

### 7.6 发布前检查

- [ ] 所有阻塞测试通过
- [ ] 验证报告完整
- [ ] 模型文件版本标记
- [ ] 部署文档更新
- [ ] 回滚方案准备

---

## 8. 附录

### 8.1 参考文档
- [PyTorch ONNX 导出文档](https://pytorch.org/docs/stable/onnx.html)
- [TensorRT 开发者指南](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)

### 8.2 相关脚本
- `scripts/verify_onnx_export.py` - ONNX 导出验证
- `scripts/verify_trt_build.py` - TensorRT 构建验证
- `scripts/verify_accuracy.py` - 精度验证
- `scripts/verify_performance.py` - 性能验证
- `scripts/run_all_verification.sh` - 一键验证

### 8.3 验证数据要求
- 校准数据集：≥ 500 张图像，覆盖训练数据分布
- 验证数据集：≥ 5000 张图像，与生产分布一致
- 边界测试数据：包含极端亮度、对比度、遮挡样本

---

> **文档维护：** 随项目进展更新测试用例和通过标准
> **最后更新：** 2026-06-03