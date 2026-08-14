#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 服务 - mldi-lab/Kairos_23m
任务: 自适应时序预测(动态patching+DRoPE)
模态: tabular

[重要] 模型加载代码需从 HuggingFace 模型页面获取真实部署代码后填入下方 TODO 区域。
  - 镜像站优先: https://hf-mirror.com/mldi-lab/Kairos_23m
  - 官方兜底:   https://huggingface.co/mldi-lab/Kairos_23m
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
logger = logging.getLogger("mldi-lab-kairos-23m")

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

app = FastAPI(title="mldi-lab-kairos-23m", version="1.0.0")

# ============================================================
# 模型加载区域 — 从 HuggingFace 模型页面获取的真实部署代码
# 来源: https://hf-mirror.com/mldi-lab/Kairos_23m
# 官方仓库: https://github.com/foundation-model-research/Kairos
# ============================================================
import json
import torch
from tsfm.model.kairos import AutoModel

model = None


def load_model():
    """加载 Kairos-23M 自适应时序预测模型"""
    global model
    if model is None:
        logger.info("正在加载 Kairos-23M 模型，权重目录: %s", WEIGHTS_DIR)
        model = AutoModel.from_pretrained(str(WEIGHTS_DIR), trust_remote_code=True)
        logger.info("Kairos-23M 模型加载完成")


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
    load_model()


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok"}


@app.post("/predict")
async def predict(req: PredictRequest):
    """
    预测接口
    - 输入: req.data (base64 编码的 tabular 数据)
    - 输出: result (base64 编码的推理结果)
    """
    import time
    t0 = time.time()

    # 解码输入
    try:
        raw_input = base64.b64decode(req.data)
        payload = json.loads(raw_input.decode("utf-8"))
    except Exception as e:
        logger.error("输入解码失败: %s", e)
        return PredictResponse(result=base64.b64encode(b"error: invalid input").decode())

    # ============================================================
    # 推理区域 — 基于 Kairos-23M 官方示例代码适配
    # 输入 JSON 格式: {"past_target": [[...]], "prediction_length": 96}
    #   past_target: (batch, context_length) 历史时序数据
    #   prediction_length: 预测步数
    # 输出 JSON 格式: {"forecast": [...]}
    #   forecast: prediction_outputs 预测结果
    # ============================================================
    try:
        past_target = torch.tensor(payload["past_target"], dtype=torch.float32)
        prediction_length = payload.get("prediction_length", 96)

        with torch.no_grad():
            forecast = model(
                past_target=past_target,
                prediction_length=prediction_length,
                generation=True,
                preserve_positivity=True,
                average_with_flipped_input=True,
            )

        result_data = {"forecast": forecast["prediction_outputs"].tolist()}
        output_bytes = json.dumps(result_data).encode("utf-8")
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
