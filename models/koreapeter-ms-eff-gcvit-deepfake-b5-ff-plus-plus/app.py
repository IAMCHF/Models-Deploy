#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 服务 - KoreaPeter/ms-eff-gcvit-deepfake-b5-ff-plus-plus
任务: 深度伪造检测(Pro,384x384)
模态: image

模型卡来源: https://hf-mirror.com/KoreaPeter/ms-eff-gcvit-deepfake-b5-ff-plus-plus
模型卡 "Model Usage" 使用 transformers `pipeline("image-classification", ...)`，
加载时需 `trust_remote_code=True`（仓库自带 modeling/processor 远程代码）。
模型卡提示 `pip install deepguard`（提供 YOLO 人脸检测后端，pipeline 远程代码会调用）。
输出为 [{'label': 'fake'/'real', 'score': float}, ...]，higher score = more likely fake。
"""

import os
import io
import json
import base64
import logging
from pathlib import Path

from transformers import pipeline
from fastapi import FastAPI
from pydantic import BaseModel

# ============================================================
# 镜像站优先：确保模型加载时若需下载额外配置/tokenizer 优先走镜像站
# ============================================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("koreapeter-ms-eff-gcvit-deepfake-b5-ff-plus-plus")

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
MODEL_ID = "KoreaPeter/ms-eff-gcvit-deepfake-b5-ff-plus-plus"

app = FastAPI(title="koreapeter-ms-eff-gcvit-deepfake-b5-ff-plus-plus", version="1.0.0")

# ============================================================
# 模型加载区域 — transformers image-classification pipeline (来自模型卡 Model Usage)
# 参考: https://hf-mirror.com/KoreaPeter/ms-eff-gcvit-deepfake-b5-ff-plus-plus
#   clf = pipeline("image-classification",
#                  model="KoreaPeter/ms-eff-gcvit-deepfake-b5-ff-plus-plus",
#                  trust_remote_code=True)
# 该仓库使用 HuggingFace remote code，需 trust_remote_code=True。
# ============================================================
clf = None
_device = None


def load_model():
    """加载 MS-EffGCViT 深度伪造检测 pipeline (image-classification, Pro b5)"""
    global clf, _device
    if clf is not None:
        return
    try:
        import torch
        _device = 0 if torch.cuda.is_available() else -1
    except Exception:
        _device = -1

    # 优先使用本地权重目录，否则使用模型名称(走镜像站自动下载)
    model_path = str(WEIGHTS_DIR) if WEIGHTS_DIR.exists() and any(WEIGHTS_DIR.iterdir()) else MODEL_ID
    logger.info("加载 image-classification pipeline: %s, device=%s", model_path, _device)
    clf = pipeline(
        "image-classification",
        model=model_path,
        trust_remote_code=True,
        device=_device,
    )
    logger.info("MS-EffGCViT-b5 pipeline 加载完成")


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
        logger.warning("权重目录不存在，模型将通过 HuggingFace 镜像站自动下载")


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok"}


@app.post("/predict")
async def predict(req: PredictRequest):
    """
    预测接口
    - 输入: req.data (base64 编码的 image 数据, 支持 jpg/png 等常见格式)
    - 输出: result (base64 编码的 JSON 推理结果)
      JSON 格式: [{"label": "fake", "score": 0.97}, {"label": "real", "score": 0.03}]
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
    # 推理区域 — 使用 image-classification pipeline (来自模型卡)
    # 模型卡示例: result = clf("face.jpg")
    # 这里将 base64 解码后的字节流包装为 PIL.Image 传给 pipeline。
    # ============================================================
    try:
        from PIL import Image
        image = Image.open(io.BytesIO(raw_input)).convert("RGB")

        # 模型卡自定义参数 (margin_ratio/conf_thres/min_face_ratio/tta_hflip/top_k)
        # 这里使用默认人脸检测参数，top_k=2 返回 real/fake 两个概率
        results = clf(image, top_k=2)

        # 标准化输出 (pipeline 可能返回 dict 或 list)
        if isinstance(results, dict):
            results = [results]
        norm_results = [
            {"label": str(r.get("label")), "score": float(r.get("score"))}
            for r in results
        ]
        output_bytes = json.dumps(norm_results, ensure_ascii=False).encode("utf-8")
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
