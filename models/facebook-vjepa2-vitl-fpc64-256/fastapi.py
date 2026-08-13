#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 服务 - facebook/vjepa2-vitl-fpc64-256
任务: 视频理解自监督预训练(分类/检索)
模态: video

[重要] 模型加载代码需从 HuggingFace 模型页面获取真实部署代码后填入下方 TODO 区域。
  - 镜像站优先: https://hf-mirror.com/facebook/vjepa2-vitl-fpc64-256
  - 官方兜底:   https://huggingface.co/facebook/vjepa2-vitl-fpc64-256
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
logger = logging.getLogger("facebook-vjepa2-vitl-fpc64-256")

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

app = FastAPI(title="facebook-vjepa2-vitl-fpc64-256", version="1.0.0")

# ============================================================
# 模型加载区域 — 基于 V-JEPA 2 官方模型卡代码适配
# 参考: https://hf-mirror.com/facebook/vjepa2-vitl-fpc64-256
# ============================================================
import io
import tempfile
import numpy as np
import torch
from transformers import AutoVideoProcessor, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(str(WEIGHTS_DIR)).to(device)
processor = AutoVideoProcessor.from_pretrained(str(WEIGHTS_DIR))


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
    - 输入: req.data (base64 编码的 video 数据)
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
    # 推理区域 — 基于 V-JEPA 2 官方模型卡代码适配
    # 输入: base64 编码的视频 → 采样64帧 → 提取视频特征嵌入
    # 输出: numpy .npy 格式的嵌入向量 base64 编码
    # ============================================================
    from torchcodec.decoders import VideoDecoder

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(raw_input)
        tmp_path = tmp.name
    try:
        vr = VideoDecoder(tmp_path)
        frame_idx = np.arange(0, 64)  # 采样64帧 (fpc64)
        video = vr.get_frames_at(indices=frame_idx).data  # T x C x H x W
        video = processor(video, return_tensors="pt").to(model.device)
        with torch.no_grad():
            video_embeddings = model.get_vision_features(**video)
        out_buf = io.BytesIO()
        np.save(out_buf, video_embeddings.cpu().numpy())
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
