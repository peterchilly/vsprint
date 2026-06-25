# ERes2NetV2 说话人识别部署指南

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
cd /path/to/vsprint

# 激活虚拟环境
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart onnx onnxruntime
```

### 2. 模型推理

```python
from src.deploy.inference import SpeakerEncoder, SpeakerVerifier, SpeakerIdentifier

# 提取说话人 embedding
encoder = SpeakerEncoder(
    config_path="configs/train_config.yaml",
    checkpoint_path="checkpoints/best_model.pth",
)
embedding = encoder.extract_from_file("audio.wav")
print(f"Embedding shape: {embedding.shape}")  # (192,)

# 说话人验证（1:1）
verifier = SpeakerVerifier(
    config_path="configs/train_config.yaml",
    checkpoint_path="checkpoints/best_model.pth",
    threshold=0.5,
)
score, is_same = verifier.verify("enroll.wav", "test.wav")
print(f"相似度: {score:.4f}, 同一说话人: {is_same}")

# 说话人识别（1:N）
identifier = SpeakerIdentifier(
    config_path="configs/train_config.yaml",
    checkpoint_path="checkpoints/best_model.pth",
)
identifier.enroll("Alice", "alice1.wav")
identifier.enroll("Bob", "bob1.wav")
results = identifier.identify("test.wav", top_k=3)
for name, score in results:
    print(f"  {name}: {score:.4f}")
```

### 3. REST API 服务

```bash
# 启动 API 服务
uvicorn src.deploy.api_server:app --host 0.0.0.0 --port 8000

# 或直接运行
python -m src.deploy.api_server
```

### 4. Docker 部署

```bash
# 构建镜像
docker build -t vsprint-api .

# 启动容器
docker compose up -d

# 查看日志
docker compose logs -f

# 停止
docker compose down
```

## API 使用示例

### 健康检查

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model": "ERes2NetV2-34",
  "device": "cuda",
  "num_enrolled": 0
}
```

### 注册说话人

```bash
curl -X POST "http://localhost:8000/enroll?speaker_name=alice" \
  -F "audio=@alice.wav"
```

```json
{
  "speaker_id": "alice",
  "status": "ok",
  "message": "说话人 'alice' 注册成功"
}
```

### 说话人验证

```bash
curl -X POST "http://localhost:8000/verify" \
  -F "enroll_audio=@enroll.wav" \
  -F "test_audio=@test.wav"
```

```json
{
  "score": 0.8234,
  "is_same": true,
  "threshold": 0.5
}
```

### 说话人识别

```bash
curl -X POST "http://localhost:8000/identify?top_k=5" \
  -F "audio=@test.wav"
```

```json
{
  "top_matches": [
    {"speaker": "alice", "score": 0.82},
    {"speaker": "bob", "score": 0.45}
  ],
  "num_enrolled": 2
}
```

### 管理已注册说话人

```bash
# 列出所有说话人
curl http://localhost:8000/speakers

# 移除说话人
curl -X DELETE http://localhost:8000/speakers/alice

# 清空所有说话人
curl -X DELETE http://localhost:8000/speakers
```

## 配置说明

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VSPRINT_CONFIG` | configs/train_config.yaml | 配置文件路径 |
| `VSPRINT_CHECKPOINT` | checkpoints/best_model.pth | 模型 checkpoint |
| `VSPRINT_THRESHOLD` | 0.5 | 验证阈值 |
| `VSPRINT_HOST` | 0.0.0.0 | API 监听地址 |
| `VSPRINT_PORT` | 8000 | API 监听端口 |
| `VSPRINT_UPLOAD_DIR` | uploads | 上传文件临时目录 |

### 阈值调整

验证阈值（threshold）控制假正率（FAR）和假负率（FRR）的平衡：

| 阈值 | FAR | FRR | 适用场景 |
|------|-----|-----|----------|
| 0.3 | 高 | 低 | 宽松场景（优先召回） |
| 0.5 | 中 | 中 | 通用场景（平衡） |
| 0.7 | 低 | 高 | 严格场景（优先精度） |

基于 EER 阈值（~0.5）是理论最优平衡点。实际部署建议根据业务需求调整。

### 音频要求

| 参数 | 值 |
|------|------|
| 格式 | WAV / MP3 / FLAC（librosa 支持） |
| 采样率 | 16000 Hz（自动重采样） |
| 声道 | 单声道（自动混合） |
| 最短时长 | ≥2 秒（建议 ≥3 秒） |
| 特征 | 80 维 FBank |

## 模型导出

### 导出 ONNX

```bash
python scripts/export_model.py \
  --config configs/train_config.yaml \
  --checkpoint checkpoints/best_model.pth \
  --format onnx
```

### 导出 TorchScript

```bash
python scripts/export_model.py \
  --config configs/train_config.yaml \
  --checkpoint checkpoints/best_model.pth \
  --format torchscript
```

### 导出全部格式

```bash
python scripts/export_model.py \
  --config configs/train_config.yaml \
  --checkpoint checkpoints/best_model.pth \
  --format all
```

## 性能基准

### 推理延迟

| 设备 | Batch=1 | Batch=32 | 说明 |
|------|---------|----------|------|
| RTX 5070 Ti | ~3 ms | ~15 ms | GPU 推理 |
| CPU (i7) | ~20 ms | ~150 ms | CPU 推理 |

### 显存占用

| 场景 | 显存 |
|------|------|
| 模型加载 | ~50 MB |
| 推理 (batch=1) | ~200 MB |
| 推理 (batch=32) | ~500 MB |

### 模型大小

| 格式 | 大小 |
|------|------|
| PyTorch (.pth) | ~6 MB |
| ONNX (.onnx) | ~6 MB |
| TorchScript (.pt) | ~6 MB |

## 评估

### 运行评估脚本

```bash
python scripts/evaluate.py \
  --config configs/train_config.yaml \
  --checkpoint checkpoints/best_model.pth
```

### 评估输出

```
reports/
├── evaluation_report.md    # 评估报告
├── evaluation_data.json    # 完整评估数据
├── roc_curve.json           # ROC 曲线数据
└── det_curve.json           # DET 曲线数据
```

## Troubleshooting

### 常见问题

#### 1. CUDA 不可用

```
[WARN] CUDA 不可用，使用 CPU 推理
```

**解决方案**：
- 检查 NVIDIA 驱动是否正确安装：`nvidia-smi`
- 检查 PyTorch CUDA 版本：`python -c "import torch; print(torch.version.cuda)"`
- 安装匹配的 CUDA Toolkit

#### 2. 音频加载失败

```
RuntimeError: Failed to load audio file
```

**解决方案**：
- 检查文件格式是否受支持（WAV/MP3/FLAC）
- 安装 ffmpeg：`apt-get install ffmpeg` 或 `conda install ffmpeg`
- 检查文件是否损坏

#### 3. 内存不足

```
CUDA out of memory
```

**解决方案**：
- 减小 batch size
- 使用 CPU 推理：`device=cpu`
- 关闭其他占用 GPU 的进程

#### 4. 验证准确率低

**可能原因及解决方案**：
- **音频过短**：确保音频长度 ≥2 秒
- **噪声环境**：进行噪声抑制预处理
- **阈值不当**：调整 `VSPRINT_THRESHOLD` 或使用 EER 阈值
- **说话人未注册**：确保已通过 `/enroll` 注册

#### 5. Docker 构建失败

```
ERROR: Could not install dependencies
```

**解决方案**：
- 检查网络连接
- 使用国内镜像源：
  ```dockerfile
  RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```

#### 6. 模型加载失败

```
RuntimeError: Error(s) in loading state_dict
```

**解决方案**：
- 确保使用正确的 checkpoint 文件（`checkpoints/best_model.pth`）
- 检查配置文件中的模型参数是否与训练时一致
- 如使用迁移学习，确保 `num_speakers` 匹配

### 日志查看

#### API 服务日志

```bash
# Docker
docker compose logs -f vsprint-api

# 直接运行
# 日志输出到 stdout
```

#### 评估日志

```bash
# 评估数据保存在 reports/ 目录
cat reports/evaluation_report.md
cat reports/evaluation_data.json | python -m json.tool
```

## 生产部署建议

### 性能优化

1. **使用 GPU**：推理延迟从 ~20ms 降至 ~3ms
2. **Batch 推理**：使用 `extract_from_files()` 批量提取
3. **模型导出**：使用 ONNX Runtime 推理，减少 Python 开销
4. **连接池**：使用 Gunicorn + Uvicorn workers

### 安全建议

1. **API 认证**：添加 API Key 或 JWT 认证
2. **HTTPS**：使用 Nginx 反向代理配置 SSL
3. **速率限制**：限制单 IP 请求频率
4. **数据加密**：加密存储说话人 embedding
5. **日志脱敏**：避免在日志中记录音频内容

### 监控

1. **健康检查**：定期调用 `/health` 接口
2. **性能监控**：记录推理延迟和错误率
3. **资源监控**：监控 GPU 显存和 CPU 使用率
4. **告警**：设置异常告警规则

---

*本部署指南由 vsprint 项目自动生成*
