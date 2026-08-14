#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 服务 - jhu-clsp/mmBERT-base
任务: 多语言编码器(1800+语言)
模态: text

模型加载与推理代码已根据 HuggingFace 模型页面官方示例适配：
  - 镜像站优先: https://hf-mirror.com/jhu-clsp/mmBERT-base
  - 官方兜底:   https://huggingface.co/jhu-clsp/mmBERT-base
  - 参考页面 "Quick Start" / "Cross-lingual Embeddings" 代码片段
"""

import os
import base64
import json
import logging
from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel
try:
    # 容器无 python3-dev, Triton JIT 编译会失败; 出错时回退 eager 而非崩溃
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass
from transformers import AutoTokenizer, AutoModel

# ============================================================
# 镜像站优先：确保模型加载时若需下载额外配置/tokenizer 优先走镜像站
# ============================================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("jhu-clsp-mmbert-base")

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="jhu-clsp-mmbert-base", version="1.0.0")

# ============================================================
# 模型加载区域 — 适配自模型页面官方示例
# 官方示例:
#   tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
#   model = AutoModel.from_pretrained("jhu-clsp/mmBERT-base")
#   inputs = tokenizer("Hello world", return_tensors="pt")
#   outputs = model(**inputs)
# 本地部署改为从 weights/ 目录加载。
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(str(WEIGHTS_DIR))
model = AutoModel.from_pretrained(str(WEIGHTS_DIR))
model.to(DEVICE)
model.eval()


class PredictRequest(BaseModel):
    """预测请求：data 为 base64 编码输入(text 的 UTF-8 字节)"""
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
    - 输入: req.data (base64 编码的 text 数据)
    - 输出: result (base64 编码的 JSON，含 mean-pooled 跨语言嵌入)
    """
    import time
    t0 = time.time()

    # 解码输入
    try:
        raw_input = base64.b64decode(req.data)
        text = raw_input.decode("utf-8")
    except Exception as e:
        logger.error("输入解码失败: %s", e)
        return PredictResponse(result=base64.b64encode(b"error: invalid base64/utf-8").decode())

    # ============================================================
    # 推理区域 — 对输入文本生成跨语言嵌入(mean pooling)
    # 参考页面 "Cross-lingual Embeddings" 示例:
    #   outputs = model(**inputs)
    #   embeddings = outputs.last_hidden_state.mean(dim=1)
    # ============================================================
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192,
        ).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            # mean pooling over last_hidden_state
            embeddings = outputs.last_hidden_state.mean(dim=1)
        result_payload = {
            "embedding": embeddings[0].cpu().tolist(),
            "shape": list(embeddings.shape),
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
