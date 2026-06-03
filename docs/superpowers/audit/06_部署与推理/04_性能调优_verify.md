# 性能调优验证测试计划

> **文档版本：** v1.0
> **创建日期：** 2026-06-03
> **关联任务：** 阶段六：部署与推理 - 性能调优

---

## 1. 验证目标

### 1.1 主要目标

| 目标 | 描述 | 优先级 |
|------|------|--------|
| **吞吐量验证** | 验证推理服务在高并发场景下的 QPS 能力 | P0 |
| **延迟达标** | 确保推理延迟满足生产环境 SLA 要求 | P0 |
| **稳定性验证** | 验证长时间运行下的服务稳定性 | P0 |
| **资源利用** | 验证资源（GPU/CPU/内存）使用效率 | P1 |
| **弹性伸缩** | 验证动态 batch size 和并发策略有效性 | P1 |

### 1.2 性能指标基线

| 指标 | 目标值 | 测量方法 |
|------|--------|----------|
| **平均延迟 (P50)** | ≤ 50ms | 压力测试统计 |
| **P95 延迟** | ≤ 100ms | 压力测试统计 |
| **P99 延迟** | ≤ 200ms | 压力测试统计 |
| **QPS** | ≥ 100 requests/second | 压力测试统计 |
| **错误率** | ≤ 0.1% | 错误计数/总请求数 |
| **GPU 利用率** | ≥ 70% | nvidia-smi 监控 |
| **内存使用** | ≤ 80% 可用内存 | 系统监控 |
| **服务可用性** | ≥ 99.9% | 长时间稳定性测试 |

---

## 2. 测试用例

### 2.1 动态 Batch Size 测试

#### TC-001: 动态 Batch Size 基础功能

| 项目 | 内容 |
|------|------|
| **测试ID** | TC-001 |
| **测试名称** | 动态 Batch Size 基础功能验证 |
| **前置条件** | 推理服务已部署，模型已加载 |
| **测试步骤** | 1. 配置 min_batch_size=1, max_batch_size=32<br>2. 发送 1-50 个并发请求<br>3. 观察实际 batch size 变化<br>4. 记录延迟和吞吐量 |
| **预期结果** | batch size 随并发数动态调整，在 1-32 范围内变化 |
| **验证方法** | 日志分析、监控指标 |
| **优先级** | P0 |

#### TC-002: Batch Size 边界测试

| 项目 | 内容 |
|------|------|
| **测试ID** | TC-002 |
| **测试名称** | Batch Size 边界值验证 |
| **前置条件** | 推理服务已部署 |
| **测试步骤** | 1. 发送单个请求，验证 batch_size=1<br>2. 发送超大量并发（>max_batch_size）<br>3. 验证请求排队和分批处理<br>4. 验证无请求丢失 |
| **预期结果** | 单请求正确处理，超量请求排队分批处理，无丢失 |
| **验证方法** | 请求计数、响应完整性检查 |
| **优先级** | P0 |

#### TC-003: Batch Size 超时处理

| 项目 | 内容 |
|------|------|
| **测试ID** | TC-003 |
| **测试名称** | Batch 组装超时机制验证 |
| **前置条件** | 推理服务已部署，配置 batch_timeout |
| **测试步骤** | 1. 配置 batch_timeout=50ms<br>2. 发送少量请求（未达到 min_batch_size）<br>3. 验证在超时后仍能正确处理<br>4. 记录实际等待时间 |
| **预期结果** | 请求在超时时间内被处理，不无限等待 |
| **验证方法** | 日志时间戳分析 |
| **优先级** | P1 |

---

### 2.2 多线程并行测试

#### TC-004: 多线程并发推理

| 项目 | 内容 |
|------|------|
| **测试ID** | TC-004 |
| **测试名称** | 多线程并行推理验证 |
| **前置条件** | GPU 服务已部署，支持多线程 |
| **测试步骤** | 1. 配置不同线程数（1/2/4/8）<br>2. 每种配置运行压力测试<br>3. 对比 QPS 和延迟<br>4. 确定最优线程配置 |
| **预期结果** | 存在最优线程数配置，QPS 随线程数增加而提升（到某点） |
| **验证方法** | 性能指标对比分析 |
| **优先级** | P0 |

#### TC-005: 线程安全性验证

| 项目 | 内容 |
|------|------|
| **测试ID** | TC-005 |
| **测试名称** | 多线程数据一致性验证 |
| **前置条件** | 多线程推理服务已部署 |
| **测试步骤** | 1. 发送相同输入的并发请求<br>2. 验证所有响应结果一致<br>3. 发送不同输入的并发请求<br>4. 验证响应与输入正确对应 |
| **预期结果** | 所有响应正确，无数据错乱或竞态条件 |
| **验证方法** | 响应内容验证、哈希对比 |
| **优先级** | P0 |

---

### 2.3 缓存策略测试

#### TC-006: 特征缓存功能

| 项目 | 内容 |
|------|------|
| **测试ID** | TC-006 |
| **测试名称** | 特征缓存命中率验证 |
| **前置条件** | 缓存策略已配置 |
| **测试步骤** | 1. 发送重复音频样本请求<br>2. 验证缓存命中<br>3. 发送新音频样本请求<br>4. 验证缓存未命中<br>5. 统计缓存命中率 |
| **预期结果** | 重复请求命中缓存，响应时间显著降低 |
| **验证方法** | 日志分析、响应时间对比 |
| **优先级** | P1 |

#### TC-007: 缓存过期策略

| 项目 | 内容 |
|------|------|
| **测试ID** | TC-007 |
| **测试名称** | 缓存过期清理验证 |
| **前置条件** | 缓存 TTL 已配置 |
| **测试步骤** | 1. 配置缓存 TTL=60s<br>2. 发送请求填充缓存<br>3. 等待 TTL 过期<br>4. 再次发送相同请求<br>5. 验证缓存未命中（重新计算） |
| **预期结果** | 过期缓存被正确清理，重新计算结果 |
| **验证方法** | 缓存状态监控、响应时间分析 |
| **优先级** | P2 |

#### TC-008: 缓存内存限制

| 项目 | 内容 |
|------|------|
| **测试ID** | TC-008 |
| **测试名称** | 缓存内存上限验证 |
| **前置条件** | 缓存内存限制已配置 |
| **测试步骤** | 1. 配置缓存内存上限<br>2. 发送大量不同请求<br>3. 观察缓存淘汰行为<br>4. 验证内存不超限 |
| **预期结果** | 缓存正确执行 LRU 淘汰，内存使用稳定 |
| **验证方法** | 内存监控、缓存状态检查 |
| **优先级** | P2 |

---

### 2.4 模型预热测试

#### TC-009: 冷启动延迟测试

| 项目 | 内容 |
|------|------|
| **测试ID** | TC-009 |
| **测试名称** | 模型冷启动延迟验证 |
| **前置条件** | 服务未启动 |
| **测试步骤** | 1. 启动服务（不预热）<br>2. 立即发送首个请求<br>3. 记录首次推理延迟<br>4. 发送后续请求对比 |
| **预期结果** | 首次推理延迟明显高于后续（基准数据） |
| **验证方法** | 延迟统计对比 |
| **优先级** | P1 |

#### TC-010: 预热效果验证

| 项目 | 内容 |
|------|------|
| **测试ID** | TC-010 |
| **测试名称** | 模型预热效果验证 |
| **前置条件** | 服务支持预热配置 |
| **测试步骤** | 1. 启动服务并执行预热<br>2. 预热完成后发送首个请求<br>3. 记录延迟<br>4. 与 TC-009 结果对比 |
| **预期结果** | 预热后首次请求延迟与稳态延迟接近 |
| **验证方法** | 延迟对比分析 |
| **优先级** | P0 |

#### TC-011: 预热策略配置

| 项目 | 内容 |
|------|------|
| **测试ID** | TC-011 |
| **测试名称** | 不同预热策略对比 |
| **前置条件** | 服务支持多种预热方式 |
| **测试步骤** | 1. 测试预热次数=0（无预热）<br>2. 测试预热次数=1<br>3. 测试预热次数=3<br>4. 测试预热次数=10<br>5. 对比各种配置的首次延迟 |
| **预期结果** | 确定最优预热次数配置 |
| **验证方法** | 延迟数据统计分析 |
| **优先级** | P1 |

---

### 2.5 压力测试

#### TC-012: 基准压力测试

| 项目 | 内容 |
|------|------|
| **测试ID** | TC-012 |
| **测试名称** | 基准 QPS 测试 |
| **前置条件** | 服务已部署并预热 |
| **测试步骤** | 1. 使用 wrk/locust 等工具<br>2. 逐步增加并发数：1/10/50/100/200<br>3. 每阶段持续 5 分钟<br>4. 记录 QPS、延迟、错误率 |
| **预期结果** | QPS ≥ 100，错误率 ≤ 0.1%，P95 延迟 ≤ 100ms |
| **验证方法** | 压测工具统计报告 |
| **优先级** | P0 |

#### TC-013: 长时间稳定性测试

| 项目 | 内容 |
|------|------|
| **测试ID** | TC-013 |
| **测试名称** | 24小时稳定性测试 |
| **前置条件** | 服务已部署 |
| **测试步骤** | 1. 持续运行 24 小时<br>2. 保持 50% 峰值 QPS<br>3. 每 10 分钟采样性能指标<br>4. 监控内存泄漏、GPU 状态 |
| **预期结果** | 性能指标稳定，无内存泄漏，无服务重启 |
| **验证方法** | 监控看板、日志分析 |
| **优先级** | P0 |

#### TC-014: 极限压力测试

| 项目 | 内容 |
|------|------|
| **测试ID** | TC-014 |
| **测试名称** | 系统极限测试 |
| **前置条件** | 服务已部署 |
| **测试步骤** | 1. 持续增加并发直到服务失败<br>2. 记录最大可承受 QPS<br>3. 观察服务降级行为<br>4. 验证服务恢复能力 |
| **预期结果** | 明确系统极限，服务优雅降级而非崩溃 |
| **验证方法** | 极限值记录、服务状态监控 |
| **优先级** | P1 |

#### TC-015: 突发流量测试

| 项目 | 内容 |
|------|------|
| **测试ID** | TC-015 |
| **测试名称** | 突发流量处理验证 |
| **前置条件** | 服务已部署 |
| **测试步骤** | 1. 平稳运行于 10 QPS<br>2. 突然增加至 500 QPS（10 秒内）<br>3. 观察服务响应<br>4. 验证排队机制 |
| **预期结果** | 服务正确排队处理，无请求丢失，最终恢复稳定 |
| **验证方法** | 请求完整性验证、延迟分析 |
| **优先级** | P1 |

---

### 2.6 延迟达标测试

#### TC-016: 延迟分布测试

| 项目 | 内容 |
|------|------|
| **测试ID** | TC-016 |
| **测试名称** | 延迟百分位测试 |
| **前置条件** | 服务已部署并预热 |
| **测试步骤** | 1. 发送 10000 次请求<br>2. 统计 P50/P90/P95/P99 延迟<br>3. 对比目标值<br>4. 分析长尾请求原因 |
| **预期结果** | P50 ≤ 50ms, P95 ≤ 100ms, P99 ≤ 200ms |
| **验证方法** | 统计分析、日志检查 |
| **优先级** | P0 |

#### TC-017: 不同音频长度延迟测试

| 项目 | 内容 |
|------|------|
| **测试ID** | TC-017 |
| **测试名称** | 音频长度对延迟影响 |
| **前置条件** | 服务已部署 |
| **测试步骤** | 1. 准备不同长度音频：1s/5s/10s/30s/60s<br>2. 每种长度测试 100 次<br>3. 统计各组延迟<br>4. 分析延迟与长度的关系 |
| **预期结果** | 延迟增长在可接受范围内，无异常跳变 |
| **验证方法** | 数据分析、图表可视化 |
| **优先级** | P1 |

---

## 3. 验证方法

### 3.1 测试工具

| 工具 | 用途 | 安装命令 |
|------|------|----------|
| **wrk** | HTTP 压力测试 | `apt install wrk` |
| **locust** | 分布式压测 | `pip install locust` |
| **pytest** | 测试框架 | `pip install pytest pytest-benchmark` |
| **nvidia-smi** | GPU 监控 | CUDA 工具包自带 |
| **prometheus** | 指标收集 | Docker 部署 |
| **grafana** | 监控看板 | Docker 部署 |

### 3.2 监控指标采集

```yaml
# prometheus.yml 示例配置
scrape_configs:
  - job_name: 'inference-service'
    static_configs:
      - targets: ['localhost:8000/metrics']
    scrape_interval: 5s
```

### 3.3 压测脚本示例

```python
# locustfile.py
from locust import HttpUser, task, between

class InferenceUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def infer(self):
        # 发送音频推理请求
        with open('test_audio.wav', 'rb') as f:
            self.client.post(
                '/api/v1/infer',
                files={'audio': f},
                timeout=30
            )
```

### 3.4 延迟测量方法

```python
import time
import numpy as np

def measure_latency(client, requests, warmup=10):
    """测量推理延迟分布"""
    # 预热
    for _ in range(warmup):
        client.infer()

    # 正式测量
    latencies = []
    for req in requests:
        start = time.perf_counter()
        client.infer(req)
        latencies.append(time.perf_counter() - start)

    return {
        'p50': np.percentile(latencies, 50) * 1000,  # ms
        'p90': np.percentile(latencies, 90) * 1000,
        'p95': np.percentile(latencies, 95) * 1000,
        'p99': np.percentile(latencies, 99) * 1000,
        'mean': np.mean(latencies) * 1000,
        'std': np.std(latencies) * 1000,
    }
```

---

## 4. 通过标准

### 4.1 硬性标准（必须通过）

| 编号 | 标准 | 验证方式 |
|------|------|----------|
| AC-001 | QPS ≥ 100（P50 延迟 ≤ 50ms 条件下） | TC-012 |
| AC-002 | P95 延迟 ≤ 100ms | TC-016 |
| AC-003 | P99 延迟 ≤ 200ms | TC-016 |
| AC-004 | 错误率 ≤ 0.1% | TC-012, TC-013 |
| AC-005 | 24 小时稳定性测试无崩溃 | TC-013 |
| AC-006 | 模型预热后首次请求延迟 ≤ P95 标准 | TC-010 |
| AC-007 | 多线程并发无数据错乱 | TC-005 |

### 4.2 软性标准（建议通过）

| 编号 | 标准 | 验证方式 |
|------|------|----------|
| SC-001 | GPU 利用率 ≥ 70%（高负载时） | TC-012 |
| SC-002 | 内存使用稳定，无泄漏趋势 | TC-013 |
| SC-003 | 缓存命中率 ≥ 30%（重复请求场景） | TC-006 |
| SC-004 | 突发流量后 30s 内恢复稳定 | TC-015 |

### 4.3 验收检查清单

- [ ] 所有 P0 测试用例通过
- [ ] 性能指标达到或超过目标值
- [ ] 24 小时稳定性测试无异常
- [ ] 监控看板正常工作
- [ ] 性能测试报告已生成
- [ ] 资源使用报告已生成

---

## 5. 自动化建议

### 5.1 CI/CD 集成

```yaml
# .github/workflows/performance-test.yml
name: Performance Tests

on:
  schedule:
    - cron: '0 2 * * *'  # 每日凌晨 2 点
  workflow_dispatch:

jobs:
  performance-test:
    runs-on: self-hosted-gpu
    steps:
      - uses: actions/checkout@v3

      - name: Start Service
        run: docker-compose up -d --build

      - name: Warmup
        run: python scripts/warmup.py

      - name: Run Benchmark
        run: |
          pytest tests/performance/ \
            --benchmark-only \
            --benchmark-json=benchmark.json

      - name: Run Stress Test
        run: locust -f locustfile.py --headless -u 100 -r 10 -t 5m

      - name: Generate Report
        run: python scripts/generate_perf_report.py

      - name: Upload Report
        uses: actions/upload-artifact@v3
        with:
          name: performance-report
          path: reports/
```

### 5.2 自动化测试脚本

```bash
#!/bin/bash
# scripts/run_performance_tests.sh

set -e

echo "=== Starting Performance Test Suite ==="

# 1. 环境检查
echo "[1/5] Checking environment..."
python scripts/check_gpu.py
python scripts/check_service_health.py

# 2. 预热
echo "[2/5] Warming up model..."
python scripts/warmup.py --iterations 10

# 3. 基准测试
echo "[3/5] Running benchmark tests..."
pytest tests/performance/benchmark.py -v --benchmark-json=reports/benchmark.json

# 4. 压力测试
echo "[4/5] Running stress tests..."
locust -f tests/performance/locustfile.py \
    --headless \
    -u 100 -r 10 -t 5m \
    --html reports/stress_test.html

# 5. 生成报告
echo "[5/5] Generating reports..."
python scripts/generate_perf_report.py \
    --benchmark reports/benchmark.json \
    --stress reports/stress_test.html \
    --output reports/performance_report.md

echo "=== Performance Tests Complete ==="
```

### 5.3 监控告警配置

```yaml
# alerting_rules.yml
groups:
  - name: inference_performance
    rules:
      - alert: HighLatencyP95
        expr: histogram_quantile(0.95, inference_latency_bucket) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P95 latency exceeds 100ms"

      - alert: HighErrorRate
        expr: rate(inference_errors_total[5m]) / rate(inference_requests_total[5m]) > 0.001
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Error rate exceeds 0.1%"

      - alert: LowQPS
        expr: rate(inference_requests_total[1m]) < 50
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "QPS below expected threshold"
```

---

## 6. 风险点

### 6.1 技术风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| **GPU 内存不足** | 高并发时 OOM 错误 | 实现内存监控和请求限流 |
| **模型加载延迟** | 服务启动慢，首次请求超时 | 预热机制、懒加载优化 |
| **线程竞争** | 性能下降、数据不一致 | 充分的并发测试、锁优化 |
| **缓存穿透** | 高负载下缓存失效 | 布隆过滤器、空值缓存 |

### 6.2 环境风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| **GPU 驱动版本不兼容** | 服务无法启动 | 版本检查脚本、容器化部署 |
| **网络带宽限制** | 大音频传输延迟高 | 音频压缩、分块传输 |
| **资源竞争** | 其他服务影响性能 | 独立部署、资源隔离 |

### 6.3 测试风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| **测试数据代表性不足** | 性能评估不准确 | 使用真实/多样化测试数据 |
| **环境差异** | 测试结果不可复现 | 环境标准化、记录环境配置 |
| **压测工具瓶颈** | 无法真实评估服务能力 | 分布式压测、工具性能验证 |

### 6.4 风险应对计划

1. **测试前环境检查清单**
   - [ ] GPU 驱动版本确认
   - [ ] CUDA 版本确认
   - [ ] Docker 版本确认
   - [ ] 网络带宽测试
   - [ ] 磁盘 I/O 测试

2. **测试数据准备**
   - 准备覆盖各时长的音频样本
   - 包含边界情况（极短、极长音频）
   - 模拟真实用户输入分布

3. **回滚策略**
   - 保留上一版本部署配置
   - 监控关键指标，异常时自动回滚
   - 记录性能基线，对比验证

---

## 7. 附录

### 7.1 测试数据规格

| 类型 | 规格 | 数量 |
|------|------|------|
| 短音频 | 1-3 秒 | 100 条 |
| 中等音频 | 5-15 秒 | 200 条 |
| 长音频 | 30-60 秒 | 100 条 |
| 边界音频 | <1 秒 / >60 秒 | 各 20 条 |

### 7.2 测试环境配置

```yaml
# 推荐测试环境配置
hardware:
  gpu: NVIDIA A100 40GB / V100 32GB
  cpu: 16+ cores
  memory: 64GB+
  storage: SSD 500GB+

software:
  os: Ubuntu 22.04
  docker: 24.0+
  cuda: 11.8+
  python: 3.10+
```

### 7.3 性能报告模板

```markdown
# 性能测试报告

## 测试概要
- 测试日期：YYYY-MM-DD
- 测试环境：[环境描述]
- 测试版本：[Git Commit]

## 性能指标
| 指标 | 目标值 | 实测值 | 状态 |
|------|--------|--------|------|
| QPS | ≥ 100 | | |
| P50 延迟 | ≤ 50ms | | |
| P95 延迟 | ≤ 100ms | | |
| P99 延迟 | ≤ 200ms | | |
| 错误率 | ≤ 0.1% | | |

## 详细分析
[图表和分析]

## 问题和建议
[发现的问题和优化建议]
```

---

## 8. 验证执行记录

| 日期 | 执行人 | 测试范围 | 结果 | 备注 |
|------|--------|----------|------|------|
| | | | | |
| | | | | |
| | | | | |

---

> **文档维护：** 随着项目进展更新测试用例和通过标准
> **审阅周期：** 每个迭代周期结束时审阅