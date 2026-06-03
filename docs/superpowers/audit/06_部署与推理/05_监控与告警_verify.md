# 监控与告警验证测试计划

> **文档版本：** v1.0
> **创建日期：** 2026-06-03
> **关联阶段：** 阶段六：部署与推理
> **验证目标：** 确保推理服务的监控与告警系统完整、可靠、有效

---

## 1. 验证目标

### 1.1 核心目标
1. **监控覆盖完整性** - 所有关键指标被正确采集和展示
2. **告警准确性** - 告警规则正确触发，无误报/漏报
3. **系统可靠性** - 监控系统本身具备高可用性
4. **可观测性** - 运维人员能够通过看板快速定位问题

### 1.2 验证范围
| 监控维度 | 具体指标 | 数据源 |
|---------|---------|--------|
| 服务可用性 | 健康状态、存活探针 | `/health`, `/ready` 端点 |
| 请求指标 | 请求量、成功率、延迟分布 | 推理 API 网关 |
| 资源指标 | GPU 利用率、显存使用、CPU、内存 | Prometheus Node Exporter + DCGM |
| 模型指标 | 推理耗时、批处理效率 | 应用埋点 |
| 数据质量 | 输入分布漂移、异常输入检测 | 特征监控模块 |

---

## 2. 测试用例

### 2.1 请求指标监控

| 用例ID | 测试场景 | 前置条件 | 测试步骤 | 预期结果 |
|--------|---------|---------|---------|---------|
| TC-MON-001 | 请求量计数准确性 | 服务正常运行 | 1. 发送 100 次推理请求<br>2. 检查监控看板请求量 | 看板显示请求数 = 100 ± 2% |
| TC-MON-002 | 成功率计算 | 服务正常运行 | 1. 发送 80 次正常请求<br>2. 发送 20 次错误请求(无效输入)<br>3. 检查成功率指标 | 成功率显示 80% ± 2% |
| TC-MON-003 | P50 延迟监控 | 服务正常运行 | 1. 记录 100 次请求实际延迟<br>2. 对比看板 P50 值 | 偏差 < 5% |
| TC-MON-004 | P99 延迟监控 | 服务正常运行 | 1. 记录 100 次请求实际延迟<br>2. 对比看板 P99 值 | 偏差 < 5% |
| TC-MON-005 | 延迟直方图 | 服务正常运行 | 1. 检查延迟分布直方图<br>2. 验证各 bucket 边界值 | 直方图 bucket 正确划分 |
| TC-MON-006 | 零流量处理 | 服务启动但无请求 | 1. 停止所有请求<br>2. 等待 5 分钟<br>3. 检查看板 | 请求量显示 0，无错误告警 |
| TC-MON-007 | 高并发压测 | 服务正常运行 | 1. 使用压测工具发送 1000 QPS<br>2. 持续 5 分钟<br>3. 检查指标采集 | 指标无丢失，延迟无异常跳变 |

### 2.2 GPU 资源监控

| 用例ID | 测试场景 | 前置条件 | 测试步骤 | 预期结果 |
|--------|---------|---------|---------|---------|
| TC-GPU-001 | GPU 利用率采集 | GPU 驱动已安装，DCGM 运行 | 1. 运行推理负载<br>2. 对比 `nvidia-smi` 与看板数值 | 偏差 < 3% |
| TC-GPU-002 | 显存使用监控 | GPU 驱动已安装 | 1. 运行批处理推理<br>2. 检查显存曲线 | 显存变化与实际负载匹配 |
| TC-GPU-003 | GPU 温度监控 | GPU 驱动已安装 | 1. 运行高负载推理<br>2. 检查温度指标 | 温度指标正确显示 |
| TC-GPU-004 | 多 GPU 环境 | 多 GPU 服务器 | 1. 检查各 GPU 指标分离<br>2. 验证聚合视图 | 各 GPU 指标独立，聚合正确 |
| TC-GPU-005 | GPU 故障模拟 | GPU 驱动已安装 | 1. 模拟 GPU 掉线(如禁用设备)<br>2. 检查监控状态 | 正确显示离线状态 |

### 2.3 数据漂移检测

| 用例ID | 测试场景 | 前置条件 | 测试步骤 | 预期结果 |
|--------|---------|---------|---------|---------|
| TC-DRIFT-001 | 输入分布监控 | 漂移检测模块已部署 | 1. 发送正常分布的测试数据<br>2. 检查分布统计 | 统计量在基准范围内 |
| TC-DRIFT-002 | 漂移告警触发 | 漂移检测模块已部署 | 1. 发送明显偏离基准的数据<br>2. 检查漂移告警 | 漂移告警正确触发 |
| TC-DRIFT-003 | 基准更新机制 | 存在旧基准 | 1. 触发基准更新<br>2. 验证新基准生效 | 新基准正确保存并应用 |
| TC-DRIFT-004 | 特征缺失检测 | 漂移检测模块已部署 | 1. 发送缺失特征的请求<br>2. 检查异常检测 | 正确识别并记录异常 |

### 2.4 告警机制

| 用例ID | 测试场景 | 前置条件 | 测试步骤 | 预期结果 |
|--------|---------|---------|---------|---------|
| TC-ALERT-001 | 服务不可用告警 | 告警规则已配置 | 1. 停止推理服务<br>2. 等待告警触发时间 | 告警在阈值时间内触发(≤30s) |
| TC-ALERT-002 | 高延迟告警 | 延迟阈值已配置 | 1. 注入延迟(如 sleep)<br>2. 触发 P99 > 阈值 | 延迟告警正确触发 |
| TC-ALERT-003 | 低成功率告警 | 成功率阈值已配置 | 1. 持续发送错误请求<br>2. 使成功率降至阈值以下 | 成功率告警正确触发 |
| TC-ALERT-004 | GPU 显存告警 | 显存阈值已配置 | 1. 分配大量 GPU 内存<br>2. 使显存使用超阈值 | 显存告警正确触发 |
| TC-ALERT-005 | 告警恢复通知 | 已触发告警 | 1. 恢复正常状态<br>2. 等待恢复通知 | 收到告警恢复通知 |
| TC-ALERT-006 | 告警静默期 | 告警规则已配置 | 1. 触发告警<br>2. 在静默期内再次触发 | 仅收到首次告警通知 |
| TC-ALERT-007 | 告警升级机制 | 多级告警配置 | 1. 触发高级别告警<br>2. 验证升级流程 | 按配置升级到对应渠道 |
| TC-ALERT-008 | 告警通知渠道 | 多渠道已配置 | 1. 触发告警<br>2. 检查各通知渠道 | 邮件/钉钉/短信等渠道正常收到 |

### 2.5 监控看板

| 用例ID | 测试场景 | 前置条件 | 测试步骤 | 预期结果 |
|--------|---------|---------|---------|---------|
| TC-DASH-001 | 看板可访问性 | Grafana 已部署 | 1. 访问看板 URL<br>2. 检查加载状态 | 看板正常加载，无错误 |
| TC-DASH-002 | 数据刷新 | 服务正常运行 | 1. 观察看板 5 分钟<br>2. 验证数据自动刷新 | 数据按设定间隔刷新 |
| TC-DASH-003 | 时间范围选择 | 看板可访问 | 1. 切换不同时间范围<br>2. 验证数据正确查询 | 各时间范围数据正确 |
| TC-DASH-004 | 下钻功能 | 看板可访问 | 1. 点击图表元素<br>2. 验证下钻视图 | 下钻视图正确展示 |
| TC-DASH-005 | 导出功能 | 看板可访问 | 1. 导出 PDF/图片<br>2. 验证导出内容 | 导出内容完整清晰 |
| TC-DASH-006 | 多用户并发 | 多用户环境 | 1. 10 用户同时访问看板<br>2. 验证响应时间 | 页面加载时间 < 3s |

---

## 3. 验证方法

### 3.1 测试环境要求

```yaml
环境配置:
  监控栈:
    - Prometheus: v2.45+
    - Grafana: v10.0+
    - AlertManager: v0.26+
    - Node Exporter: v1.6+
    - DCGM Exporter: v3.1+ (GPU 监控)

  推理服务:
    - 模型: ERes2NetV2
    - 部署方式: Docker
    - 副本数: ≥2 (测试高可用)

  测试工具:
    - 压测工具: locust / wrk / hey
    - 数据生成: 测试数据集
    - 告警测试: chaos-mesh / 故障注入
```

### 3.2 测试执行策略

```
阶段 1: 基础功能验证 (Day 1)
├── TC-MON-001 ~ TC-MON-005: 请求指标
├── TC-GPU-001 ~ TC-GPU-003: GPU 基础监控
└── TC-DASH-001 ~ TC-DASH-003: 看板基础

阶段 2: 告警系统验证 (Day 1-2)
├── TC-ALERT-001 ~ TC-ALERT-008: 全部告警场景
└── TC-DRIFT-001 ~ TC-DRIFT-004: 数据漂移

阶段 3: 压力与边界测试 (Day 2)
├── TC-MON-006 ~ TC-MON-007: 高并发
├── TC-GPU-004 ~ TC-GPU-005: GPU 边界
└── TC-DASH-006: 并发访问

阶段 4: 集成验证 (Day 2-3)
├── 端到端故障演练
└── 监控-告警-响应全链路验证
```

### 3.3 测试数据准备

```python
# 测试数据集规格
test_datasets:
  normal_requests:
    count: 10000
    distribution: "匹配训练数据分布"

  drift_requests:
    count: 1000
    distribution: "有意偏移基准分布"

  error_requests:
    count: 500
    types: ["invalid_input", "oversized_input", "missing_features"]

  high_latency_requests:
    count: 200
    latency_injection: "100ms-500ms"
```

---

## 4. 通过标准

### 4.1 功能性标准

| 指标类别 | 通过标准 | 度量方法 |
|---------|---------|---------|
| 指标准确性 | 采集值与实际值偏差 < 3% | 对比监控值与 `nvidia-smi`/日志值 |
| 告警触发时间 | 从故障发生到告警触发 < 30s | 时间戳对比 |
| 告警准确率 | 误报率 < 1%，漏报率 = 0 | 统计告警事件与实际故障 |
| 看板可用性 | 99.9% 可用 | 健康检查成功率 |
| 看板响应时间 | 页面加载 < 3s | 浏览器 DevTools 测量 |

### 4.2 非功能性标准

```
性能标准:
  - Prometheus 查询延迟 P99 < 1s
  - 指标采集周期: 15s (可配置)
  - 数据保留期: ≥ 30 天
  - 并发看板访问: 支持 50+ 用户

可靠性标准:
  - 监控系统单点故障不影响数据采集
  - AlertManager 集群部署，支持故障转移
  - 告警通知重试机制: 最多 3 次

安全标准:
  - 看板访问需要认证
  - 敏感数据脱敏展示
  - 审计日志记录访问
```

### 4.3 单项测试判定规则

| 结果 | 条件 |
|-----|------|
| ✅ 通过 | 所有预期结果满足，无偏差 |
| ⚠️ 条件通过 | 主要功能正常，存在可接受的轻微偏差 |
| ❌ 失败 | 关键预期未满足，需要修复后重测 |
| ⏭️ 跳过 | 前置条件不满足，记录原因 |

---

## 5. 自动化建议

### 5.1 自动化测试脚本

```yaml
# automated_tests.yaml
test_suites:
  smoke_test:
    schedule: "每 6 小时"
    tests:
      - TC-MON-001  # 请求量计数
      - TC-GPU-001  # GPU 利用率
      - TC-DASH-001 # 看板可访问性
    runner: "pytest tests/smoke/ --junit-xml=report.xml"

  regression_test:
    schedule: "每日"
    tests:
      - TC-MON-001 ~ TC-MON-007
      - TC-GPU-001 ~ TC-GPU-005
      - TC-ALERT-001 ~ TC-ALERT-008
    runner: "pytest tests/regression/ --junit-xml=report.xml"

  drift_test:
    schedule: "每周"
    tests:
      - TC-DRIFT-001 ~ TC-DRIFT-004
    runner: "pytest tests/drift/ --junit-xml=report.xml"
```

### 5.2 CI/CD 集成

```yaml
# .github/workflows/monitoring-test.yml
name: Monitoring Test

on:
  deployment_status:
    types: [success]

jobs:
  verify-monitoring:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Wait for service ready
        run: sleep 60

      - name: Run monitoring tests
        run: |
          pip install -r tests/requirements.txt
          pytest tests/monitoring/ -v --junit-xml=report.xml

      - name: Publish test results
        uses: dorny/test-reporter@v1
        with:
          name: Monitoring Tests
          path: report.xml
          reporter: java-junit
```

### 5.3 监控自监控

```yaml
# self_monitoring.yaml
health_checks:
  prometheus:
    endpoint: "/-/healthy"
    interval: "30s"
    alert_on_failure: true

  alertmanager:
    endpoint: "/-/healthy"
    interval: "30s"
    alert_on_failure: true

  grafana:
    endpoint: "/api/health"
    interval: "60s"
    alert_on_failure: true

metrics_to_track:
  - prometheus_tsdb_head_series  # 指标数量
  - prometheus_rule_evaluation_failures_total  # 规则评估失败
  - alertmanager_alerts_invalid_total  # 无效告警
  - grafana_stat_total_dashboard  # 看板数量
```

### 5.4 自动化验证脚本示例

```python
# tests/monitoring/test_metrics.py
import pytest
import requests
from prometheus_api_client import PrometheusConnect

class TestRequestMetrics:

    @pytest.fixture
    def prometheus(self):
        return PrometheusConnect(url="http://prometheus:9090")

    def test_request_count_accuracy(self, prometheus):
        """TC-MON-001: 验证请求计数准确性"""
        # 发送已知数量请求
        expected_count = 100
        for _ in range(expected_count):
            requests.post("http://inference-service/predict",
                         json={"input": "test"})

        # 等待采集周期
        import time; time.sleep(30)

        # 查询 Prometheus
        result = prometheus.custom_query(
            'sum(inference_requests_total)'
        )

        actual_count = float(result[0]["value"][1])
        assert abs(actual_count - expected_count) / expected_count < 0.02

    def test_success_rate_calculation(self, prometheus):
        """TC-MON-002: 验证成功率计算"""
        # 发送混合请求
        for i in range(100):
            if i < 80:
                requests.post("http://inference-service/predict",
                             json={"input": "valid"})
            else:
                requests.post("http://inference-service/predict",
                             json={"invalid": "data"})

        time.sleep(30)

        result = prometheus.custom_query(
            'sum(rate(inference_requests_total[5m])) / '
            'sum(rate(inference_requests_total[5m]))'
        )

        success_rate = float(result[0]["value"][1]) * 100
        assert 78 <= success_rate <= 82  # 允许 ±2%
```

---

## 6. 风险点

### 6.1 技术风险

| 风险 | 可能性 | 影响 | 缓解措施 |
|-----|-------|------|---------|
| Prometheus 存储满导致数据丢失 | 中 | 高 | 配置数据保留策略，增加存储，设置存储告警 |
| 告警风暴淹没重要告警 | 中 | 中 | 配置告警分组、静默规则、分级告警 |
| GPU 监控驱动兼容性问题 | 低 | 高 | 测试 DCGM 与 GPU 驱动版本兼容性 |
| 高负载下监控开销影响服务 | 低 | 中 | 使用 sidecar 模式，限制采集频率 |
| 时钟不同步导致指标异常 | 低 | 中 | 部署 NTP 同步，监控时钟偏移 |

### 6.2 运维风险

| 风险 | 可能性 | 影响 | 缓解措施 |
|-----|-------|------|---------|
| 告警通知渠道不可用 | 中 | 高 | 配置多渠道备份，定期测试通知 |
| 看板配置丢失 | 低 | 高 | 版本控制看板配置，定期备份 |
| 告警规则误配置 | 中 | 高 | Code Review 告警规则，灰度发布 |
| 监控系统无人维护 | 中 | 中 | 建立 On-call 轮值，定期巡检 |

### 6.3 业务风险

| 风险 | 可能性 | 影响 | 缓解措施 |
|-----|-------|------|---------|
| 漂移检测阈值敏感度不当 | 中 | 中 | 渐进调整阈值，A/B 验证 |
| 正常流量波动触发误报 | 中 | 中 | 设置合理阈值边界，引入动态基线 |
| 监控指标过多影响性能 | 低 | 低 | 精简指标，按需采集 |

### 6.4 测试风险

| 风险 | 可能性 | 影响 | 缓解措施 |
|-----|-------|------|---------|
| 测试环境与生产不一致 | 中 | 高 | 使用基础设施即代码，环境对齐检查 |
| 故障注入影响其他服务 | 中 | 中 | 使用独立测试环境，限制爆炸半径 |
| 测试数据覆盖不完整 | 中 | 中 | 使用生产数据采样，持续补充边界用例 |

---

## 7. 测试检查清单

### 7.1 测试前准备

- [ ] 监控组件全部部署并运行正常
- [ ] 推理服务已部署并可正常访问
- [ ] 测试数据集已准备
- [ ] 测试脚本已编写并通过 review
- [ ] 告警通知渠道已配置并验证可用
- [ ] 测试环境与生产环境配置对齐

### 7.2 测试执行

- [ ] 按阶段执行测试用例
- [ ] 记录每个用例的实际结果
- [ ] 对失败用例记录详细日志
- [ ] 收集监控截图作为证据
- [ ] 记录告警触发时间和内容

### 7.3 测试后收尾

- [ ] 汇总测试结果，计算通过率
- [ ] 对失败用例分析根因
- [ ] 提出改进建议
- [ ] 归档测试证据
- [ ] 编写测试报告

---

## 8. 附录

### 8.1 Prometheus 查询示例

```promql
# 请求量 (QPS)
sum(rate(inference_requests_total[5m]))

# 成功率
sum(rate(inference_requests_total{status="success"}[5m]))
/ sum(rate(inference_requests_total[5m]))

# P99 延迟
histogram_quantile(0.99,
  sum(rate(inference_request_duration_seconds_bucket[5m]))
  by (le)
)

# GPU 利用率
DCGM_FI_DEV_GPU_UTIL

# GPU 显存使用率
DCGM_FI_DEV_MEM_COPY_UTIL
```

### 8.2 告警规则示例

```yaml
groups:
  - name: inference_service
    rules:
      - alert: ServiceDown
        expr: up{job="inference-service"} == 0
        for: 30s
        labels:
          severity: critical
        annotations:
          summary: "推理服务不可用"
          description: "服务 {{ $labels.instance }} 已宕机超过 30 秒"

      - alert: HighLatency
        expr: |
          histogram_quantile(0.99,
            sum(rate(inference_request_duration_seconds_bucket[5m]))
            by (le)
          ) > 0.5
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "P99 延迟过高"
          description: "P99 延迟 {{ $value | humanizeDuration }} 超过 500ms 阈值"

      - alert: LowSuccessRate
        expr: |
          sum(rate(inference_requests_total{status="success"}[5m]))
          / sum(rate(inference_requests_total[5m])) < 0.95
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "成功率过低"
          description: "成功率 {{ $value | humanizePercentage }} 低于 95%"

      - alert: HighGPUMemory
        expr: DCGM_FI_DEV_MEM_COPY_UTIL > 90
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU 显存使用过高"
          description: "GPU {{ $labels.gpu }} 显存使用率 {{ $value }}%"
```

### 8.3 参考文档

- [Prometheus 最佳实践](https://prometheus.io/docs/practices/)
- [Grafana 看板设计指南](https://grafana.com/docs/grafana/latest/best-practices/)
- [NVIDIA DCGM 文档](https://docs.nvidia.com/datacenter/dcgm/latest/)
- [SRE 监控指南](https://sre.google/sre-book/monitoring-distributed-systems/)

---

**文档状态：** 待评审
**下一步：** 评审通过后纳入测试执行计划