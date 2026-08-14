#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 服务 - ibm-granite/granite-speech-4.1-2b
任务: 多语言ASR/语音翻译
模态: audio

模型卡来源: https://hf-mirror.com/ibm-granite/granite-speech-4.1-2b
使用 AutoModelForSpeechSeq2Seq + AutoProcessor (transformers>=4.52.1)。
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
logger = logging.getLogger("ibm-granite-granite-speech-4-1-2b")

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

app = FastAPI(title="ibm-granite-granite-speech-4-1-2b", version="1.0.0")

# ============================================================
# 模型加载区域 — 使用 AutoModelForSpeechSeq2Seq (来自模型卡 Usage with transformers)
# 参考: https://hf-mirror.com/ibm-granite/granite-speech-4.1-2b
# ============================================================
import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

MODEL_ID = "ibm-granite/granite-speech-4.1-2b"
processor = None
tokenizer = None
model = None
device = None


def load_model():
    """加载 Granite Speech 4.1 2B 模型"""
    global processor, tokenizer, model, device
    if model is not None:
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = str(WEIGHTS_DIR) if WEIGHTS_DIR.exists() and any(WEIGHTS_DIR.iterdir()) else MODEL_ID
    logger.info("加载模型: %s, device=%s", model_path, device)

    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    # device_map="cuda" 曾导致参数被拆分到 CPU/GPU 引发 device mismatch,
    # 改为整体加载后手动 .to(device)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    ).to(device)
    logger.info("Granite Speech 4.1 2B 模型加载完成")


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
    - 输入: req.data (base64 编码的 audio 数据，需为 16kHz 单声道 WAV)
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
    # 推理区域 — 使用 torchaudio + processor + model.generate (来自模型卡)
    # ============================================================
    tmp_path = None
    try:
        # 将音频字节写入临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(raw_input)
            tmp_path = tmp.name

        # 加载音频 (模型卡示例: torchaudio.load, 需 mono 16kHz)
        wav, sr = torchaudio.load(tmp_path, normalize=True)
        # 确保单声道
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        # 重采样到 16kHz (如需要)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            wav = resampler(wav)
            sr = 16000

        # 创建文本提示 (模型卡示例: <|audio|> 前缀 + 转写指令)
        user_prompt = "<|audio|>transcribe the speech with proper punctuation and capitalization."
        chat = [
            {"role": "user", "content": user_prompt},
        ]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        # 运行 processor + model (模型卡示例)
        model_inputs = processor(prompt, wav, device=device, return_tensors="pt").to(device)
        with torch.no_grad():
            model_outputs = model.generate(
                **model_inputs, max_new_tokens=200, do_sample=False, num_beams=1
            )

        # 解码输出 (模型卡示例: 截取新生成的 token)
        num_input_tokens = model_inputs["input_ids"].shape[-1]
        new_tokens = model_outputs[0, num_input_tokens:].unsqueeze(0)
        output_text = tokenizer.batch_decode(
            new_tokens, add_special_tokens=False, skip_special_tokens=True
        )

        transcription = output_text[0] if output_text else ""
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
