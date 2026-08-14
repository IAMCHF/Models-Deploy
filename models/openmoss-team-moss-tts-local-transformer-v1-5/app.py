#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 服务 - OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5
任务: 零样本语音克隆TTS(31种语言)
模态: audio (text-to-speech)

模型卡来源: https://hf-mirror.com/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5
使用 AutoModel + AutoProcessor (trust_remote_code=True) 加载，参考模型卡 Basic Usage。
输入为待合成文本，输出为 48kHz 立体声 WAV 音频。
"""

import os
import io
import json
import base64
import importlib.util
import logging
from pathlib import Path

import torch
import torchaudio
from transformers import AutoModel, AutoProcessor
from fastapi import FastAPI
from pydantic import BaseModel

# ============================================================
# 镜像站优先：确保模型加载时若需下载额外配置/tokenizer 优先走镜像站
# ============================================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("openmoss-team-moss-tts-local-transformer-v1-5")

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

app = FastAPI(title="openmoss-team-moss-tts-local-transformer-v1-5", version="1.0.0")

# ============================================================
# 模型加载区域 — 使用 AutoModel + AutoProcessor (来自模型卡 Basic Usage)
# 参考: https://hf-mirror.com/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5
# 该仓库使用 HuggingFace remote code，需 trust_remote_code=True。
# 音频编解码使用 MOSS-Audio-Tokenizer-v2，输出 48kHz 立体声。
# ============================================================
MODEL_ID = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
processor = None
model = None
device = None
dtype = None


def _resolve_attn_implementation() -> str:
    """选择 attention 实现 (来自模型卡示例)"""
    if (
        device == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"
    if device == "cuda":
        return "sdpa"
    return "eager"


def load_model():
    """加载 MOSS-TTS-Local-Transformer-v1.5 模型"""
    global processor, model, device, dtype
    if model is not None:
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # 关闭某些 CUDA/PyTorch 组合下损坏的 cuDNN SDPA 后端 (模型卡示例)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    # 优先使用本地权重目录，否则使用模型名称(自动下载)
    model_path = str(WEIGHTS_DIR) if WEIGHTS_DIR.exists() and any(WEIGHTS_DIR.iterdir()) else MODEL_ID
    attn_implementation = _resolve_attn_implementation()
    logger.info("加载模型: %s, device=%s, dtype=%s, attn=%s", model_path, device, dtype, attn_implementation)

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.audio_tokenizer = processor.audio_tokenizer.to(device)

    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    logger.info("MOSS-TTS-Local-Transformer-v1.5 模型加载完成")


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
        logger.warning("权重目录不存在，模型将通过 HuggingFace 自动下载")


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok"}


@app.post("/predict")
async def predict(req: PredictRequest):
    """
    预测接口
    - 输入: req.data (base64 编码的待合成文本)
      支持两种格式:
        1) 纯文本: 直接解码为 UTF-8 文本，默认语言 Chinese
        2) JSON: {"text": "待合成文本", "language": "Chinese"}
    - 输出: result (base64 编码的 48kHz 立体声 WAV 音频)
    """
    import time
    t0 = time.time()

    # 解码输入
    try:
        raw_text = base64.b64decode(req.data).decode("utf-8")
    except Exception as e:
        logger.error("base64 解码失败: %s", e)
        return PredictResponse(result=base64.b64encode(b"error: invalid base64").decode())

    # ============================================================
    # 推理区域 — 使用 processor.build_user_message + model.generate (来自模型卡)
    # ============================================================
    try:
        # 解析输入: 支持 JSON {"text","language"} 或纯文本
        text = raw_text
        language = "Chinese"
        try:
            payload = json.loads(raw_text)
            if isinstance(payload, dict) and "text" in payload:
                text = payload["text"]
                language = payload.get("language", "Chinese")
        except (json.JSONDecodeError, ValueError):
            # 非 JSON，按纯文本处理
            pass

        logger.info("TTS 合成: text=%r, language=%s", text[:60], language)

        # 构建用户消息 (模型卡示例: processor.build_user_message)
        conversations = [[processor.build_user_message(text=text, language=language)]]

        # 处理输入 (模型卡示例: processor(batch, mode="generation"))
        batch = processor(conversations, mode="generation")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # 运行推理 (模型卡示例: model.generate, v1.5 推荐参数)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=2048,
                do_sample=True,
                audio_temperature=1.7,
                audio_top_p=0.8,
                audio_top_k=25,
                audio_repetition_penalty=1.0,
            )

        # 解码输出 (模型卡示例: processor.decode -> message.audio_codes_list[0])
        # v1.5 codec 输出立体声 [channels, samples]
        audio_tensor = None
        for message in processor.decode(outputs):
            if message is None:
                continue
            audio_tensor = message.audio_codes_list[0]
            break

        if audio_tensor is None:
            output_bytes = b"error: no audio generated"
        else:
            # 写入 WAV 字节流 (模型卡示例: torchaudio.save, 采样率来自 processor.model_config)
            sampling_rate = processor.model_config.sampling_rate
            buf = io.BytesIO()
            torchaudio.save(buf, audio_tensor, sampling_rate, format="wav")
            output_bytes = buf.getvalue()
    except Exception as e:
        logger.error("推理失败: %s", e)
        output_bytes = f"error: inference failed - {e}".encode("utf-8")

    # 编码输出
    result = base64.b64encode(output_bytes).decode()
    elapsed = (time.time() - t0) * 1000
    logger.info("推理完成，耗时 %.1f ms", elapsed)
    return PredictResponse(result=result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
