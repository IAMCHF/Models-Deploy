#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 服务 - OpenGVLab/VideoMAEv2-Base
任务: 视频特征提取自监督模型
模态: video

[重要] 模型加载代码需从 HuggingFace 模型页面获取真实部署代码后填入下方 TODO 区域。
  - 镜像站优先: https://hf-mirror.com/OpenGVLab/VideoMAEv2-Base
  - 官方兜底:   https://huggingface.co/OpenGVLab/VideoMAEv2-Base
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
logger = logging.getLogger("opengvlab-videomaev2-base")

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

app = FastAPI(title="opengvlab-videomaev2-base", version="1.0.0")

# ============================================================
# 模型加载区域 — 基于 VideoMAEv2 官方模型卡 How to use 代码适配
# 参考: https://hf-mirror.com/OpenGVLab/VideoMAEv2-Base
# ============================================================
import io
import tempfile
import numpy as np
import torch
import decord
from transformers import VideoMAEImageProcessor, AutoModel, AutoConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = AutoConfig.from_pretrained(str(WEIGHTS_DIR), trust_remote_code=True)
processor = VideoMAEImageProcessor.from_pretrained(str(WEIGHTS_DIR))
model = AutoModel.from_pretrained(str(WEIGHTS_DIR), config=config, trust_remote_code=True)
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
    # 推理区域 — 基于 VideoMAEv2 官方模型卡 How to use 代码适配
    # 输入: base64 编码的视频 → 采样16帧 → 提取视频特征
    # 输出: last_hidden_state 的 numpy .npy 格式 base64 编码
    # ============================================================
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(raw_input)
        tmp_path = tmp.name
    try:
        vr = decord.VideoReader(tmp_path)
        total_frames = len(vr)
        indices = np.linspace(0, total_frames - 1, 16, dtype=int)
        frames = vr.get_batch(indices).asnumpy()  # [16, H, W, 3]
        video = [frames[i] for i in range(16)]
        inputs = processor(video, return_tensors="pt")
        # B, T, C, H, W -> B, C, T, H, W (VideoMAEv2 期望格式)
        inputs['pixel_values'] = inputs['pixel_values'].permute(0, 2, 1, 3, 4)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        out_buf = io.BytesIO()
        np.save(out_buf, outputs.last_hidden_state.cpu().numpy())
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
