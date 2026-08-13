#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 服务 - numind/NuExtract3-FP8
任务: 文档信息抽取结构化
模态: text

[重要] 模型加载代码需从 HuggingFace 模型页面获取真实部署代码后填入下方 TODO 区域。
  - 镜像站优先: https://hf-mirror.com/numind/NuExtract3-FP8
  - 官方兜底:   https://huggingface.co/numind/NuExtract3-FP8
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
logger = logging.getLogger("numind-nuextract3-fp8")

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

app = FastAPI(title="numind-nuextract3-fp8", version="1.0.0")

# ============================================================
# TODO [模型加载区域] — 必须从模型页面获取真实部署代码填入此处
# 禁止使用通用 AutoModel/pipeline 模板，须参考模型官方示例代码适配
# ============================================================
# model = ...  # 从 weights/ 加载模型
# tokenizer = ...  # 加载 tokenizer / processor
# 示例参考（需替换为模型页面真实代码）:
#   from transformers import AutoModelForXxx, AutoTokenizer
#   model = AutoModelForXxx.from_pretrained(str(WEIGHTS_DIR), ...)
#   tokenizer = AutoTokenizer.from_pretrained(str(WEIGHTS_DIR))


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
    # TODO [推理区域] — 根据模型页面示例代码实现推理逻辑
    # raw_input 为解码后的原始字节，需根据模态(text)进一步处理
    # ============================================================
    # output_bytes = run_inference(raw_input)
    output_bytes = raw_input  # TODO: 替换为真实推理结果

    # 编码输出
    result = base64.b64encode(output_bytes).decode()
    elapsed = (time.time() - t0) * 1000
    logger.info("推理完成，耗时 %.1f ms", elapsed)
    return PredictResponse(result=result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
