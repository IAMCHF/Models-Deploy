#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 服务 - microsoft/VibeVoice-ASR-HF
任务: 语音转文本ASR(50+语言)
模态: audio

模型卡来源: https://hf-mirror.com/microsoft/VibeVoice-ASR-HF
使用 VibeVoiceAsrForConditionalGeneration + AutoProcessor (transformers>=5.3.0)。
"""

import os
import base64
import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

# ============================================================
# 镜像站优先：确保模型加载时若需下载额外配置/tokenizer 优先走镜像站
# ============================================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("microsoft-vibevoice-asr-hf")

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

app = FastAPI(title="microsoft-vibevoice-asr-hf", version="1.0.0")

# ============================================================
# 模型加载区域 — 使用 VibeVoiceAsrForConditionalGeneration (来自模型卡 Loading model)
# 参考: https://hf-mirror.com/microsoft/VibeVoice-ASR-HF
# ============================================================
import torch
from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

MODEL_ID = "microsoft/VibeVoice-ASR-HF"
processor = None
model = None


def load_model():
    """加载 VibeVoice-ASR 模型"""
    global processor, model
    if model is not None:
        return
    # 优先使用本地权重目录
    model_path = str(WEIGHTS_DIR) if WEIGHTS_DIR.exists() and any(WEIGHTS_DIR.iterdir()) else MODEL_ID
    logger.info("加载模型: %s", model_path)

    processor = AutoProcessor.from_pretrained(model_path)
    model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
        model_path, device_map="auto"
    )
    logger.info("VibeVoice-ASR 模型加载完成, device=%s, dtype=%s", model.device, model.dtype)


class PredictRequest(BaseModel):
    """预测请求：data 为 base64 编码输入"""
    data: str


class PredictResponse(BaseModel):
    """预测响应：result 为 base64 编码输出"""
    result: str


@app.on_event("startup")
async def startup_event():
    logger.info("服务启动，权重目录: %s", WEIGHTS_DIR)
    load_model()
    if not WEIGHTS_DIR.exists():
        logger.warning("权重目录不存在，模型将从 HuggingFace 自动下载")


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok"}


@app.post("/predict")
async def predict(req: PredictRequest):
    """
    预测接口
    - 输入: req.data (base64 编码的 audio 数据)
    - 输出: result (base64 编码的转写文本)
    """
    import time
    t0 = time.time()

    # 解码输入
    try:
        raw_input = base64.b64decode(req.data)
    except Exception as e:
        logger.error("base64 解码失败: %s", e)
        return PredictResponse(result=base64.b64encode(b"error: invalid base64").decode())

    # ============================================================
    # 推理区域 — 使用 processor.apply_transcription_request + model.generate (来自模型卡)
    # ============================================================
    tmp_path = None
    try:
        # 将音频字节写入临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(raw_input)
            tmp_path = tmp.name

        # 准备输入 (模型卡示例: processor.apply_transcription_request)
        inputs = processor.apply_transcription_request(
            audio=tmp_path,
        ).to(model.device, model.dtype)

        # 运行推理 (模型卡示例: model.generate)
        with torch.no_grad():
            output_ids = model.generate(**inputs)

        # 解码输出 (模型卡示例: processor.decode with return_format)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        transcription = processor.decode(generated_ids, return_format="transcription_only")[0]

        output_bytes = transcription.encode("utf-8")
    except Exception as e:
        logger.error("推理失败: %s", e)
        output_bytes = f"error: inference failed - {e}".encode("utf-8")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # 编码输出
    result = base64.b64encode(output_bytes).decode()
    elapsed = (time.time() - t0) * 1000
    logger.info("推理完成，耗时 %.1f ms", elapsed)
    return PredictResponse(result=result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
