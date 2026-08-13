#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 服务 - MongoDB/mdbr-leaf-ir
任务: 信息检索文本嵌入(BEIR<=100M第1名)
模态: text

模型加载与推理代码已根据 HuggingFace 模型页面官方示例适配：
  - 镜像站优先: https://hf-mirror.com/MongoDB/mdbr-leaf-ir
  - 官方兜底:   https://huggingface.co/MongoDB/mdbr-leaf-ir
  - 参考页面 "Sentence Transformers" 代码片段
"""

import os
import base64
import json
import logging
from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ============================================================
# 镜像站优先：确保模型加载时若需下载额外配置/tokenizer 优先走镜像站
# ============================================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mongodb-mdbr-leaf-ir")

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="mongodb-mdbr-leaf-ir", version="1.0.0")

# ============================================================
# 模型加载区域 — 适配自模型页面官方示例
# 官方示例:
#   from sentence_transformers import SentenceTransformer
#   model = SentenceTransformer("MongoDB/mdbr-leaf-ir")
#   query_embeddings = model.encode(queries, prompt_name="query")
#   document_embeddings = model.encode(documents)
# 本地部署改为从 weights/ 目录加载。
# ============================================================
model = SentenceTransformer(str(WEIGHTS_DIR), device=DEVICE)


class PredictRequest(BaseModel):
    """预测请求：data 为 base64 编码输入(纯文本 或 JSON {"text","mode"})"""
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
        * 纯文本: 默认按 query 编码(prompt_name="query")
        * JSON  : {"text": "...", "mode": "query"|"document"}
    - 输出: result (base64 编码的 JSON，含 embedding 与 mode)
    """
    import time
    t0 = time.time()

    # 解码输入
    try:
        raw_input = base64.b64decode(req.data)
        decoded = raw_input.decode("utf-8")
    except Exception as e:
        logger.error("输入解码失败: %s", e)
        return PredictResponse(result=base64.b64encode(b"error: invalid base64/utf-8").decode())

    # ============================================================
    # 推理区域 — 生成文本嵌入
    # 参考页面 "Sentence Transformers" 示例:
    #   query_embeddings = model.encode(queries, prompt_name="query")
    #   document_embeddings = model.encode(documents)
    # ============================================================
    try:
        # 支持 JSON {"text": "...", "mode": "query"|"document"} 或纯文本
        mode = "query"
        text = decoded
        try:
            payload = json.loads(decoded)
            if isinstance(payload, dict) and "text" in payload:
                text = payload["text"]
                mode = payload.get("mode", "query")
        except (json.JSONDecodeError, ValueError):
            pass

        encode_kwargs = {}
        if mode == "query":
            encode_kwargs["prompt_name"] = "query"
        embedding = model.encode([text], **encode_kwargs)
        result_payload = {
            "embedding": embedding[0].tolist(),
            "mode": mode,
            "dim": int(embedding.shape[1]),
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
