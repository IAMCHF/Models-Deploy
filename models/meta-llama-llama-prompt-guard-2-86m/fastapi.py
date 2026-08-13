#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 服务 - meta-llama/Llama-Prompt-Guard-2-86M
任务: 提示注入/越狱检测
模态: text

模型加载与推理代码已根据 HuggingFace 模型页面官方示例适配：
  - 镜像站优先: https://hf-mirror.com/meta-llama/Llama-Prompt-Guard-2-86M
  - 官方兜底:   https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M
  - 参考页面 "Usage" 中 AutoTokenizer + AutoModelForSequenceClassification 代码片段
"""

import os
import base64
import json
import logging
from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================
# 镜像站优先：确保模型加载时若需下载额外配置/tokenizer 优先走镜像站
# ============================================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("meta-llama-llama-prompt-guard-2-86m")

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="meta-llama-llama-prompt-guard-2-86m", version="1.0.0")

# ============================================================
# 模型加载区域 — 适配自模型页面官方示例
# 官方示例:
#   tokenizer = AutoTokenizer.from_pretrained(model_id)
#   model = AutoModelForSequenceClassification.from_pretrained(model_id)
#   inputs = tokenizer(text, return_tensors="pt")
#   with torch.no_grad():
#       logits = model(**inputs).logits
#   predicted_class_id = logits.argmax().item()
#   print(model.config.id2label[predicted_class_id])  # MALICIOUS
# 本地部署改为从 weights/ 目录加载。
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(str(WEIGHTS_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(WEIGHTS_DIR))
model.to(DEVICE)
model.eval()
# 归一化 id2label 的键为 int(兼容 str/int 两种配置)
ID2LABEL = {int(k): v for k, v in model.config.id2label.items()}


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
    - 输出: result (base64 编码的 JSON，含 label / predicted_class_id / probabilities)
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
    # 推理区域 — 提示注入/越狱检测(二分类: benign / malicious)
    # 参考页面 "Usage" 示例:
    #   with torch.no_grad():
    #       logits = model(**inputs).logits
    #   predicted_class_id = logits.argmax().item()
    #   model.config.id2label[predicted_class_id]
    # ============================================================
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        predicted_class_id = int(logits.argmax().item())
        label = ID2LABEL.get(predicted_class_id, str(predicted_class_id))
        num_labels = probs.shape[1]
        result_payload = {
            "label": label,
            "predicted_class_id": predicted_class_id,
            "probabilities": {
                ID2LABEL.get(i, str(i)): float(probs[0][i].item()) for i in range(num_labels)
            },
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
