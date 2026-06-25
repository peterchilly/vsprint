"""
ERes2NetV2 说话人识别 REST API 服务

接口:
    POST /enroll   — 上传音频，注册说话人
    POST /verify   — 上传两个音频文件，返回相似度分数
    POST /identify — 上传音频，从已注册说话人中识别
    GET  /health   — 健康检查
    GET  /speakers — 列出已注册说话人

用法:
    uvicorn src.deploy.api_server:app --host 0.0.0.0 --port 8000
    python -m src.deploy.api_server  # 直接运行
"""

import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from pydantic import BaseModel

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.deploy.inference import SpeakerEncoder, SpeakerVerifier, SpeakerIdentifier


# ──────────────────────────────────────────────
# 响应模型
# ──────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    num_enrolled: int


class EnrollResponse(BaseModel):
    speaker_id: str
    status: str
    message: str


class VerifyResponse(BaseModel):
    score: float
    is_same: bool
    threshold: float
    enroll_duration: Optional[float] = None
    test_duration: Optional[float] = None


class IdentifyResponse(BaseModel):
    top_matches: List[Dict]
    num_enrolled: int


class SpeakerListResponse(BaseModel):
    speakers: List[str]
    count: int


# ──────────────────────────────────────────────
# 全局状态
# ──────────────────────────────────────────────

CONFIG_PATH = os.environ.get("VSPRINT_CONFIG", "configs/train_config.yaml")
CHECKPOINT_PATH = os.environ.get("VSPRINT_CHECKPOINT", "checkpoints/best_model.pth")
THRESHOLD = float(os.environ.get("VSPRINT_THRESHOLD", "0.5"))
UPLOAD_DIR = Path(os.environ.get("VSPRINT_UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 初始化模型（懒加载）
_encoder: Optional[SpeakerEncoder] = None
_verifier: Optional[SpeakerVerifier] = None
_identifier: Optional[SpeakerIdentifier] = None


def get_identifier() -> SpeakerIdentifier:
    """懒加载初始化说话人识别器"""
    global _identifier
    if _identifier is None:
        print(f"[INIT] 加载模型: {CHECKPOINT_PATH}")
        _identifier = SpeakerIdentifier(
            config_path=CONFIG_PATH,
            checkpoint_path=CHECKPOINT_PATH,
        )
        print(f"[INIT] 模型加载完成，设备: {_identifier.encoder.device}")
    return _identifier


def get_verifier() -> SpeakerVerifier:
    """懒加载初始化说话人验证器"""
    global _verifier
    if _verifier is None:
        _verifier = SpeakerVerifier(
            config_path=CONFIG_PATH,
            checkpoint_path=CHECKPOINT_PATH,
            threshold=THRESHOLD,
        )
    return _verifier


def get_encoder() -> SpeakerEncoder:
    """懒加载初始化编码器"""
    global _encoder
    if _encoder is None:
        _encoder = SpeakerEncoder(
            config_path=CONFIG_PATH,
            checkpoint_path=CHECKPOINT_PATH,
        )
    return _encoder


# ──────────────────────────────────────────────
# FastAPI 应用
# ──────────────────────────────────────────────

app = FastAPI(
    title="ERes2NetV2 说话人识别 API",
    description="基于 ERes2NetV2 的说话人识别 REST API，支持注册、验证、识别",
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event():
    """启动时预加载模型"""
    print("[STARTUP] 预加载模型...")
    get_identifier()
    print("[STARTUP] 就绪")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    identifier = get_identifier()
    return HealthResponse(
        status="ok",
        model=f"ERes2NetV2-{identifier.encoder.config['model']['variant']}",
        device=str(identifier.encoder.device),
        num_enrolled=len(identifier.enrolled_speakers),
    )


@app.post("/enroll", response_model=EnrollResponse)
async def enroll(
    speaker_name: str,
    audio: UploadFile = File(...),
):
    """
    注册说话人

    参数:
        speaker_name: 说话人名称
        audio: 音频文件（wav/mp3/flac 等）
    """
    identifier = get_identifier()

    # 检查文件类型
    if not audio.filename:
        raise HTTPException(status_code=400, detail="缺少文件名")

    # 保存上传的文件
    file_ext = Path(audio.filename).suffix or ".wav"
    temp_path = UPLOAD_DIR / f"enroll_{uuid.uuid4().hex}{file_ext}"

    try:
        content = await audio.read()
        temp_path.write_bytes(content)

        # 提取 embedding 并注册
        identifier.enroll(speaker_name, str(temp_path))

        return EnrollResponse(
            speaker_id=speaker_name,
            status="ok",
            message=f"说话人 '{speaker_name}' 注册成功",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"注册失败: {str(e)}")
    finally:
        # 清理临时文件
        if temp_path.exists():
            temp_path.unlink()


@app.post("/verify", response_model=VerifyResponse)
async def verify(
    enroll_audio: UploadFile = File(...),
    test_audio: UploadFile = File(...),
    threshold: Optional[float] = None,
):
    """
    验证两个音频是否来自同一说话人

    参数:
        enroll_audio: 注册音频文件
        test_audio: 测试音频文件
        threshold: 自定义阈值（可选）
    """
    verifier = get_verifier()

    # 设置阈值
    if threshold is not None:
        verifier.set_threshold(threshold)

    # 保存上传文件
    enroll_path = UPLOAD_DIR / f"verify_enroll_{uuid.uuid4().hex}.wav"
    test_path = UPLOAD_DIR / f"verify_test_{uuid.uuid4().hex}.wav"

    try:
        enroll_content = await enroll_audio.read()
        test_content = await test_audio.read()
        enroll_path.write_bytes(enroll_content)
        test_path.write_bytes(test_content)

        # 执行验证
        score, is_same = verifier.verify(str(enroll_path), str(test_path))

        return VerifyResponse(
            score=score,
            is_same=is_same,
            threshold=verifier.threshold,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"验证失败: {str(e)}")
    finally:
        if enroll_path.exists():
            enroll_path.unlink()
        if test_path.exists():
            test_path.unlink()


@app.post("/identify", response_model=IdentifyResponse)
async def identify(
    audio: UploadFile = File(...),
    top_k: int = 5,
):
    """
    从已注册说话人中识别音频中的说话人

    参数:
        audio: 待识别音频文件
        top_k: 返回最相似的 top_k 个说话人
    """
    identifier = get_identifier()

    if not identifier.enrolled_speakers:
        raise HTTPException(status_code=400, detail="尚未注册任何说话人，请先调用 /enroll")

    # 保存上传文件
    temp_path = UPLOAD_DIR / f"identify_{uuid.uuid4().hex}.wav"

    try:
        content = await audio.read()
        temp_path.write_bytes(content)

        # 执行识别
        results = identifier.identify(str(temp_path), top_k=top_k)

        return IdentifyResponse(
            top_matches=[
                {"speaker": name, "score": score}
                for name, score in results
            ],
            num_enrolled=len(identifier.enrolled_speakers),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"识别失败: {str(e)}")
    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.get("/speakers", response_model=SpeakerListResponse)
async def list_speakers():
    """列出所有已注册的说话人"""
    identifier = get_identifier()
    speakers = identifier.list_speakers()
    return SpeakerListResponse(speakers=speakers, count=len(speakers))


@app.delete("/speakers/{speaker_name}")
async def remove_speaker(speaker_name: str):
    """移除已注册的说话人"""
    identifier = get_identifier()
    if identifier.remove_speaker(speaker_name):
        return {"status": "ok", "message": f"说话人 '{speaker_name}' 已移除"}
    raise HTTPException(status_code=404, detail=f"说话人 '{speaker_name}' 不存在")


@app.delete("/speakers")
async def clear_speakers():
    """清空所有已注册的说话人"""
    identifier = get_identifier()
    identifier.clear()
    return {"status": "ok", "message": "所有说话人已清空"}


# ──────────────────────────────────────────────
# 直接运行
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("VSPRINT_HOST", "0.0.0.0")
    port = int(os.environ.get("VSPRINT_PORT", "8000"))

    print("=" * 60)
    print(" ERes2NetV2 说话人识别 API 服务")
    print("=" * 60)
    print(f"   配置: {CONFIG_PATH}")
    print(f"   Checkpoint: {CHECKPOINT_PATH}")
    print(f"   阈值: {THRESHOLD}")
    print(f"   监听: {host}:{port}")
    print("=" * 60)

    uvicorn.run(
        "src.deploy.api_server:app",
        host=host,
        port=port,
        reload=False,
    )
