#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 服务 - Skywork/Skywork-Reward-V2-Qwen3-0.6B
任务: 奖励模型/偏好评分
模态: text

模型加载与推理代码已根据 HuggingFace 模型页面官方示例适配：
  - 镜像站优先: https://hf-mirror.com/Skywork/Skywork-Reward-V2-Qwen3-0.6B
  - 官方兜底:   https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-0.6B
  - 参考页面 "Simple Example in transformers" 代码片段
"""

import os
import base64
import json
import logging
from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ============================================================
# 镜像站优先：确保模型加载时若需下载额外配置/tokenizer 优先走镜像站
# ============================================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("skywork-skywork-reward-v2-qwen3-0-6b")

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# GPU 使用 bfloat16，CPU 使用 float32 以保证兼容性
TORCH_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

app = FastAPI(title="skywork-skywork-reward-v2-qwen3-0-6b", version="1.0.0")

# ============================================================
# 模型加载区域 — 适配自模型页面官方示例
# 官方示例(以 Llama-3.1-8B 演示，本仓库为 Qwen3-0.6B 变体):
#   rm = AutoModelForSequenceClassification.from_pretrained(
#       model_name, torch_dtype=torch.bfloat16, device_map=device,
#       attn_implementation="flash_attention_2", num_labels=1)
#   tokenizer = AutoTokenizer.from_pretrained(model_name)
#   conv1_formatted = tokenizer.apply_chat_template(conv1, tokenize=False)
#   if tokenizer.bos_token is not None and conv1_formatted.startswith(tokenizer.bos_token):
#       conv1_formatted = conv1_formatted[len(tokenizer.bos_token):]
#   conv1_tokenized = tokenizer(conv1_formatted, return_tensors="pt").to(device)
#   with torch.no_grad():
#       score1 = rm(**conv1_tokenized).logits[0][0].item()
# 说明: Qwen3-0.6B 未强制 flash_attention_2，故省略 attn_implementation 以避免额外依赖。
# 本地部署改为从 weights/ 目录加载。
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(str(WEIGHTS_DIR))
model = AutoModelForSequenceClassification.from_pretrained(
    str(WEIGHTS_DIR),
    torch_dtype=TORCH_DTYPE,
    num_labels=1,
)
model.to(DEVICE)
model.eval()


class PredictRequest(BaseModel):
    """预测请求：data 为 base64 编码输入(JSON {"prompt","response"} 的 UTF-8 字节)"""
    data: str


class PredictResponse(BaseModel):
    """预测响应：result 为 base64 编码输出(推理结果 JSON 的 UTF-8 字节)"""
    result: str


@app.on_event("startup")
async def startup_event():
    logger.info("服务启动，权重目录: %s，设备: %s，dtype: %s", WEIGHTS_DIR, DEVICE, TORCH_DTYPE)
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
    - 输入: req.data (base64 编码的 JSON，形如 {"prompt": "...", "response": "..."})
    - 输出: result (base64 编码的 JSON，含 reward score)
    注意: 官方建议不使用 system prompt；输入长度建议 <= 16384 tokens。
    """
    import time
    t0 = time.time()

    # 解码输入
    try:
        raw_input = base64.b64decode(req.data)
        payload = json.loads(raw_input.decode("utf-8"))
        prompt = payload["prompt"]
        response = payload["response"]
    except Exception as e:
        logger.error("输入解码失败(需 JSON {prompt, response}): %s", e)
        return PredictResponse(result=base64.b64encode(b"error: invalid input (expect json {prompt,response})").decode())

    # ============================================================
    # 推理区域 — 计算对话的奖励分数
    # 参考页面 "Simple Example in transformers" 示例:
    #   conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    #   conv_formatted = tokenizer.apply_chat_template(conv, tokenize=False)
    #   score = rm(**conv_tokenized).logits[0][0].item()
    # ============================================================
    try:
        conv = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        conv_formatted = tokenizer.apply_chat_template(conv, tokenize=False)
        # 去除可能的重复 bos token
        if tokenizer.bos_token is not None and conv_formatted.startswith(tokenizer.bos_token):
            conv_formatted = conv_formatted[len(tokenizer.bos_token):]
        conv_tokenized = tokenizer(
            conv_formatted,
            return_tensors="pt",
            truncation=True,
            max_length=16384,
        ).to(DEVICE)
        with torch.no_grad():
            score = model(**conv_tokenized).logits[0][0].item()
        result_payload = {"score": float(score)}
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
