#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 服务 - voyageai/voyage-4-nano
任务: 多语言文本嵌入(32K上下文,MRL)
模态: text

[重要] 模型加载代码需从 HuggingFace 模型页面获取真实部署代码后填入下方 TODO 区域。
  - 镜像站优先: https://hf-mirror.com/voyageai/voyage-4-nano
  - 官方兜底:   https://huggingface.co/voyageai/voyage-4-nano
  - 解析页面中 "Use in Transformers" / "Use in vLLM" / "How to use" 等代码片段
  - 加载优先级: transformers 加载 > vLLM 加载
"""

import os
import base64
import json
import logging
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

# ============================================================
# 镜像站优先：确保模型加载时若需下载额外配置/tokenizer 优先走镜像站
# ============================================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("voyageai-voyage-4-nano")

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

app = FastAPI(title="voyageai-voyage-4-nano", version="1.0.0")

# ============================================================
# 模型加载区域 — 基于 voyage-4-nano 官方 "Via Transformers" 代码适配
# 参考: https://hf-mirror.com/voyageai/voyage-4-nano
# ============================================================
import io
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer


def mean_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """对 last_hidden_state 做注意力掩码加权平均池化"""
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    )
    sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    return sum_embeddings / sum_mask


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# 优先使用 flash_attention_2，不可用时回退到 sdpa
try:
    model = AutoModel.from_pretrained(
        str(WEIGHTS_DIR),
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=_dtype,
    ).to(device)
except Exception:
    model = AutoModel.from_pretrained(
        str(WEIGHTS_DIR),
        trust_remote_code=True,
        torch_dtype=_dtype,
    ).to(device)

tokenizer = AutoTokenizer.from_pretrained(str(WEIGHTS_DIR))
model.eval()

# 查询提示词（来自官方示例）
_QUERY_PROMPT = "Represent the query for retrieving supporting documents: "


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
    - 输入: req.data (base64 编码的 text 数据)
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
    # 推理区域 — 基于 voyage-4-nano 官方 "Via Transformers" 代码适配
    # 输入: base64 编码的文本 → 解码为字符串 → 添加查询提示词 → 提取嵌入
    # 输出: JSON {embedding: [...], dim: int} 的 UTF-8 字节 base64 编码
    # ============================================================
    try:
        text = raw_input.decode("utf-8")
    except Exception as e:
        logger.error("文本解码失败: %s", e)
        return PredictResponse(
            result=base64.b64encode(b"error: invalid utf-8").decode()
        )

    try:
        inputs = tokenizer(
            _QUERY_PROMPT + text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32768,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.forward(**inputs)
        embeddings = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        emb_list = embeddings[0].cpu().tolist()
        output_bytes = json.dumps(
            {"embedding": emb_list, "dim": len(emb_list)}
        ).encode("utf-8")
    except Exception as e:
        logger.error("推理失败: %s", e)
        output_bytes = json.dumps({"error": str(e)}).encode("utf-8")

    # 编码输出
    result = base64.b64encode(output_bytes).decode()
    elapsed = (time.time() - t0) * 1000
    logger.info("推理完成，耗时 %.1f ms", elapsed)
    return PredictResponse(result=result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
