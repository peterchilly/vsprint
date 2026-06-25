# ERes2NetV2 说话人识别 Docker 镜像
# 基于 Python 3.12

FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 安装额外部署依赖
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    onnx \
    onnxruntime

# 复制项目代码
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY checkpoints/ ./checkpoints/

# 创建上传和报告目录
RUN mkdir -p uploads reports exports

# 设置环境变量
ENV VSPRINT_CONFIG=configs/train_config.yaml
ENV VSPRINT_CHECKPOINT=checkpoints/best_model.pth
ENV VSPRINT_THRESHOLD=0.5
ENV VSPRINT_HOST=0.0.0.0
ENV VSPRINT_PORT=8000

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# 启动命令
CMD ["python", "-m", "uvicorn", "src.deploy.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
