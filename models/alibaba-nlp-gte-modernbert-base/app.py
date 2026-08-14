#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 服务 - Alibaba-NLP/gte-modernbert-base
任务: 文本嵌入语义检索(8192上下文)
模态: text

模型加载与推理代码已根据 HuggingFace 模型页面官方示例适配：
  - 镜像站优先: https://hf-mirror.com/Alibaba-NLP/gte-modernbert-base
  - 官方兜底:   https://huggingface.co/Alibaba-NLP/gte-modernbert-base
  - 参考页面 "Usage" 中 "Use with transformers" 代码片段
"""

import os
import base64
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
try:
    # ModernBERT 推理可能触发 Triton JIT 编译, 容器无 python3-dev 时会失败;
    # 出错回退 eager 而非返回 500
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

# ============================================================
# 镜像站优先：确保模型加载时若需下载额外配置/tokenizer 优先走镜像站
# ============================================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("alibaba-nlp-gte-modernbert-base")

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="alibaba-nlp-gte-modernbert-base", version="1.0.0")

# ============================================================
# 模型加载区域 — 适配自模型页面官方示例
# 官方示例:
#   tokenizer = AutoTokenizer.from_pretrained(model_path)
#   model = AutoModel.from_pretrained(model_path)
#   batch_dict = tokenizer(input_texts, max_length=8192, padding=True,
#                          truncation=True, return_tensors='pt')
#   outputs = model(**batch_dict)
#   embeddings = outputs.last_hidden_state[:, 0]
#   embeddings = F.normalize(embeddings, p=2, dim=1)
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
    - 输出: result (base64 编码的 JSON，含 CLS pooling + L2 归一化后的嵌入)
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
    # 推理区域 — CLS pooling + L2 归一化生成文本嵌入
    # 参考页面 "Use with transformers" 示例:
    #   embeddings = outputs.last_hidden_state[:, 0]
    #   embeddings = F.normalize(embeddings, p=2, dim=1)
    # ============================================================
    try:
        batch_dict = tokenizer(
            [text],
            max_length=8192,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(DEVICE)
        with torch.no_grad():
            outputs = model(**batch_dict)
            embeddings = outputs.last_hidden_state[:, 0]
            embeddings = F.normalize(embeddings, p=2, dim=1)
        result_payload = {
            "embedding": embeddings[0].cpu().tolist(),
            "dim": int(embeddings.shape[1]),
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
