#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 服务 - google/videoprism-lvt-base-f16r288
任务: 视频文本编码检索(零样本)
模态: video

[重要] 模型加载代码需从 HuggingFace 模型页面获取真实部署代码后填入下方 TODO 区域。
  - 镜像站优先: https://hf-mirror.com/google/videoprism-lvt-base-f16r288
  - 官方兜底:   https://huggingface.co/google/videoprism-lvt-base-f16r288
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
logger = logging.getLogger("google-videoprism-lvt-base-f16r288")

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

app = FastAPI(title="google-videoprism-lvt-base-f16r288", version="1.0.0")

# ============================================================
# 模型加载区域 — 基于 VideoPrism 官方 GitHub 代码适配
# HF 页面指向 GitHub 仓库: https://github.com/google-deepmind/videoprism
# 模型使用 JAX/Flax 框架（非 transformers），library_name: videoprism
# ============================================================
import io
import tempfile
import numpy as np
import jax
import jax.numpy as jnp
import mediapy
from videoprism import models as vp

MODEL_NAME = "videoprism_lvt_public_v1_base"
CHECKPOINT_FILE = "flax_lvt_base_f16r288_repeated.npz"
NUM_FRAMES = 16
FRAME_SIZE = 288

flax_model = vp.get_model(MODEL_NAME)

# 优先从本地权重目录加载，若不存在则从 HF 镜像下载
_local_ckpt = WEIGHTS_DIR / CHECKPOINT_FILE
if _local_ckpt.exists():
    loaded_state = vp.load_pretrained_weights(
        MODEL_NAME, checkpoint_path=str(_local_ckpt)
    )
else:
    logger.info("本地权重不存在，从 HF 镜像下载: %s", MODEL_NAME)
    loaded_state = vp.load_pretrained_weights(MODEL_NAME)


@jax.jit
def forward_video_only(inputs):
    """仅提取视频嵌入（text 输入传 None 跳过文本编码）"""
    video_embeddings, _, _ = flax_model.apply(
        loaded_state, inputs, None, None, train=False
    )
    return video_embeddings


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
    # 推理区域 — 基于 VideoPrism 官方 Colab 代码适配
    # 输入: base64 编码的视频 → 采样16帧 → 缩放至288x288 → 归一化至[0,1]
    # 输出: 视频全局嵌入向量的 numpy .npy 格式 base64 编码
    # ============================================================
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(raw_input)
        tmp_path = tmp.name
    try:
        # 读取视频
        frames = mediapy.read_video(tmp_path)
        # 采样 16 帧
        frame_indices = np.linspace(
            0, len(frames), num=NUM_FRAMES, endpoint=False, dtype=np.int32
        )
        frames = np.array([frames[i] for i in frame_indices])
        # 缩放至 288x288
        frames = mediapy.resize_video(frames, shape=(FRAME_SIZE, FRAME_SIZE))
        # 归一化至 [0.0, 1.0]
        frames = mediapy.to_float01(frames)
        # 添加 batch 维度并转为 JAX 数组
        video_inputs = jnp.asarray(frames[None, ...])

        # 推理：提取视频嵌入
        video_embeddings = forward_video_only(video_inputs)

        out_buf = io.BytesIO()
        np.save(out_buf, np.asarray(video_embeddings))
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
