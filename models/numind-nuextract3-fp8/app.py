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
import json
import logging
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
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
# 模型加载区域 — 基于 numind/NuExtract3-FP8 官方示例代码适配
# 使用 transformers AutoProcessor + AutoModelForImageTextToText
# 源码参考: https://huggingface.co/numind/NuExtract3
# FP8 量化配置已内嵌于模型 config.json，无需手动指定 dtype
# ============================================================
_model = None
_processor = None


def load_model():
    """加载 NuExtract3-FP8 模型与 processor"""
    global _model, _processor
    if _model is not None:
        return
    logger.info("正在加载 NuExtract3-FP8 模型...")
    _processor = AutoProcessor.from_pretrained(
        str(WEIGHTS_DIR),
        trust_remote_code=True,
    )
    _model = AutoModelForImageTextToText.from_pretrained(
        str(WEIGHTS_DIR),
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    logger.info("NuExtract3-FP8 模型加载完成")


class PredictRequest(BaseModel):
    """预测请求：data 为 base64 编码输入"""
    data: str


class PredictResponse(BaseModel):
    """预测响应：result 为 base64 编码输出"""
    result: str


@app.on_event("startup")
async def startup_event():
    logger.info("服务启动，权重目录: %s", WEIGHTS_DIR)
    if not WEIGHTS_DIR.exists():
        logger.warning("权重目录不存在，请先运行 download_weights.py 下载权重")
    try:
        load_model()
    except Exception as e:
        logger.error("模型加载失败: %s", e)


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
    # 推理区域 — 基于 NuExtract3 官方 transformers 推理代码适配
    # 支持: 1) JSON 输入 {"text":..., "template":{...}} 结构化抽取
    #       2) 纯文本输入 → Markdown 转换
    # ============================================================
    if _model is None or _processor is None:
        logger.error("模型未加载，无法推理")
        return PredictResponse(result=base64.b64encode(b"error: model not loaded").decode())

    try:
        text_input = raw_input.decode("utf-8")
    except UnicodeDecodeError as e:
        logger.error("UTF-8 解码失败: %s", e)
        return PredictResponse(result=base64.b64encode(b"error: invalid utf-8").decode())

    # 尝试解析 JSON 以支持结构化抽取 (含 template)
    template = None
    try:
        input_data = json.loads(text_input)
        if isinstance(input_data, dict) and "text" in input_data:
            text_content = input_data["text"]
            template = input_data.get("template")
        else:
            text_content = text_input
    except (json.JSONDecodeError, TypeError):
        text_content = text_input

    # 构建消息
    messages = [{"role": "user", "content": [{"type": "text", "text": text_content}]}]

    # 应用 chat template (非思考模式, temperature=0.2)
    chat_template_kwargs = {"enable_thinking": False}
    if template:
        chat_template_kwargs["template"] = json.dumps(template)

    inputs = _processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        chat_template_kwargs=chat_template_kwargs,
    ).to(_model.device)

    # 生成
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=True,
            temperature=0.2,
        )

    # 解码输出 (仅取新生成的 token)
    output_text = _processor.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    output_bytes = output_text.encode("utf-8")

    # 编码输出
    result = base64.b64encode(output_bytes).decode()
    elapsed = (time.time() - t0) * 1000
    logger.info("推理完成，耗时 %.1f ms", elapsed)
    return PredictResponse(result=result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
