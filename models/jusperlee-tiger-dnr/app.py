#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 服务 - JusperLee/TIGER-DnR
任务: 语音分离/音频恢复
模态: audio

[重要] 模型加载代码需从 HuggingFace 模型页面获取真实部署代码后填入下方 TODO 区域。
  - 镜像站优先: https://hf-mirror.com/JusperLee/TIGER-DnR
  - 官方兜底:   https://huggingface.co/JusperLee/TIGER-DnR
  - 解析页面中 "Use in Transformers" / "Use in vLLM" / "How to use" 等代码片段
  - 加载优先级: transformers 加载 > vLLM 加载
"""

import os
import base64
import logging
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

# ============================================================
# 镜像站优先：确保模型加载时若需下载额外配置/tokenizer 优先走镜像站
# ============================================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("jusperlee-tiger-dnr")

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

app = FastAPI(title="jusperlee-tiger-dnr", version="1.0.0")

# ============================================================
# 模型加载区域 — 基于 TIGER 官方 inference_dnr.py 代码适配
# 参考: https://github.com/JusperLee/TIGER/blob/main/inference_dnr.py
# ============================================================
import io
import tempfile
import torch
import torchaudio
import soundfile as sf
import numpy as np
import look2hear.models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = look2hear.models.TIGERDNR.from_pretrained(str(WEIGHTS_DIR))
model.to(device)
model.eval()


class PredictRequest(BaseModel):
    """预测请求：data 为 base64 编码输入"""
    data: str


class PredictResponse(BaseModel):
    """预测响应：result 为 base64 编码输出"""
    result: str


@app.on_event("startup")
async def startup_event():
    logger.info("服务启动，权重目录: %s", WEIGHTS_DIR)
    # TODO: 在此触发模型加载（如需启动时预加载）
    if not WEIGHTS_DIR.exists():
        logger.warning("权重目录不存在，请先运行 download_weights.py 下载权重")


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok"}


@app.post("/predict")
async def predict(req: PredictRequest):
    """
    预测接口
    - 输入: req.data (base64 编码的 audio 数据)
    - 输出: result (base64 编码的推理结果)
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
    # 推理区域 — 基于 TIGER 官方 inference_dnr.py 代码适配
    # 输入: base64 编码的 WAV 音频 → 语音分离(dialog/effect/music)
    # 输出: 3声道 WAV (dialog, effect, music) 的 base64 编码
    # ============================================================
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(raw_input)
        tmp_path = tmp.name
    try:
        audio = torchaudio.load(tmp_path)[0].to(device)
        with torch.no_grad():
            all_target_dialog, all_target_effect, all_target_music = model(audio[None])
        combined = torch.stack([
            all_target_dialog.squeeze(0),
            all_target_effect.squeeze(0),
            all_target_music.squeeze(0)
        ], dim=0)
        combined_np = combined.cpu().numpy().T  # [T, 3]
        out_buf = io.BytesIO()
        sf.write(out_buf, combined_np, 44100, format='WAV')
        output_bytes = out_buf.getvalue()
    finally:
        os.unlink(tmp_path)

    # 编码输出
    result = base64.b64encode(output_bytes).decode()
    elapsed = (time.time() - t0) * 1000
    logger.info("推理完成，耗时 %.1f ms", elapsed)
    return PredictResponse(result=result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
