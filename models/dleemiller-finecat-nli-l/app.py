#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 服务 - dleemiller/finecat-nli-l
任务: 自然语言推理分类(三分类: entailment / neutral / contradiction)
模态: text

模型加载与推理代码已根据 HuggingFace 模型页面官方示例适配：
  - 镜像站优先: https://hf-mirror.com/dleemiller/finecat-nli-l
  - 官方兜底:   https://huggingface.co/dleemiller/finecat-nli-l
  - 参考页面 "Direct Usage (Sentence Transformers)" 代码片段
"""

import os
import base64
import json
import logging
from pathlib import Path

# 离线容器缺 gcc 编译环境, 禁用 torch.compile (sentence-transformers 新版默认编译 embeddings)
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

# ============================================================
# 镜像站优先：确保模型加载时若需下载额外配置/tokenizer 优先走镜像站
# ============================================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dleemiller-finecat-nli-l")

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="dleemiller-finecat-nli-l", version="1.0.0")

# ============================================================
# 模型加载区域 — 适配自模型页面官方示例
# 官方示例:
#   from sentence_transformers import CrossEncoder
#   model = CrossEncoder("dleemiller/finecat-nli-l")
#   id2label = model.model.config.id2label  # {0:'entailment', 1:'neutral', 2:'contradiction'}
#   pairs = [("premise", "hypothesis"), ...]
#   logits = model.predict(pairs)  # shape: (N, 3)
#   pred_idx = int(np.argmax(row))
#   pred = id2label[pred_idx]
# 本地部署改为从 weights/ 目录加载。
# ============================================================
model = CrossEncoder(str(WEIGHTS_DIR), device=DEVICE)
# 标签映射: entailment=0, neutral=1, contradiction=2
ID2LABEL = {int(k): v for k, v in model.model.config.id2label.items()}


class PredictRequest(BaseModel):
    """预测请求：data 为 base64 编码输入(JSON {"premise","hypothesis"} 的 UTF-8 字节)"""
    data: str


class PredictResponse(BaseModel):
    """预测响应：result 为 base64 编码输出(推理结果 JSON 的 UTF-8 字节)"""
    result: str


@app.on_event("startup")
async def startup_event():
    logger.info("服务启动，权重目录: %s，设备: %s", WEIGHTS_DIR, DEVICE)
    if not WEIGHTS_DIR.exists():
        logger.warning("权重目录不存在，请先下载权重")


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok"}


@app.post("/predict")
async def predict(req: PredictRequest):
    """
    预测接口
    - 输入: req.data (base64 编码的 JSON，形如 {"premise": "...", "hypothesis": "..."})
    - 输出: result (base64 编码的 JSON，含 label / predicted_class_id / logits)
    """
    import time
    t0 = time.time()

    # 解码输入
    try:
        raw_input = base64.b64decode(req.data)
        payload = json.loads(raw_input.decode("utf-8"))
        premise = payload["premise"]
        hypothesis = payload["hypothesis"]
    except Exception as e:
        logger.error("输入解码失败(需 JSON {premise, hypothesis}): %s", e)
        return PredictResponse(result=base64.b64encode(b"error: invalid input (expect json {premise,hypothesis})").decode())

    # ============================================================
    # 推理区域 — 自然语言推理(三分类)
    # 参考页面 "Direct Usage (Sentence Transformers)" 示例:
    #   logits = model.predict([(premise, hypothesis)])
    #   pred_idx = int(np.argmax(logits[0]))
    #   pred = id2label[pred_idx]
    # ============================================================
    try:
        logits = model.predict([(premise, hypothesis)])
        pred_idx = int(np.argmax(logits[0]))
        pred = ID2LABEL.get(pred_idx, str(pred_idx))
        result_payload = {
            "label": pred,
            "predicted_class_id": pred_idx,
            "logits": logits[0].tolist(),
        }
        output_bytes = json.dumps(result_payload, ensure_ascii=False).encode("utf-8")
    except Exception as e:
        logger.exception("推理失败: %s", e)
        output_bytes = f"error: inference failed: {e}".encode("utf-8")

    # 编码输出
    result = base64.b64encode(output_bytes).decode()
    elapsed = (time.time() - t0) * 1000
    logger.info("推理完成，耗时 %.1f ms", elapsed)
    return PredictResponse(result=result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
