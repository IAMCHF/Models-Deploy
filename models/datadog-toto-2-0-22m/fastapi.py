#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 服务 - Datadog/Toto-2.0-22m
任务: 可观测性时序预测(高效默认版)
模态: tabular

[重要] 模型加载代码需从 HuggingFace 模型页面获取真实部署代码后填入下方 TODO 区域。
  - 镜像站优先: https://hf-mirror.com/Datadog/Toto-2.0-22m
  - 官方兜底:   https://huggingface.co/Datadog/Toto-2.0-22m
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
logger = logging.getLogger("datadog-toto-2-0-22m")

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

app = FastAPI(title="datadog-toto-2-0-22m", version="1.0.0")

# ============================================================
# 模型加载区域 — 从 HuggingFace 模型页面获取的真实部署代码
# 来源: https://hf-mirror.com/Datadog/Toto-2.0-22m
# ============================================================
import torch
import json
from toto2 import Toto2Model

model = None


def load_model():
    """加载 Toto 2.0 时间序列预测模型"""
    global model
    if model is None:
        logger.info("正在加载 Toto 2.0 模型，权重目录: %s", WEIGHTS_DIR)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Toto2Model.from_pretrained(str(WEIGHTS_DIR))
        model = model.to(device).eval()
        logger.info("Toto 2.0 模型加载完成，设备: %s", device)


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
    # 推理区域 — 基于 Toto 2.0 官方示例代码适配
    # 输入 JSON 格式: {"target": [[[...]]], "horizon": 96}
    #   target: (batch, n_variates, time_steps) 三维数组
    #   horizon: 预测步数
    # 输出 JSON 格式: {"quantiles": [...], "quantile_levels": [...]}
    #   quantiles: (9, batch, n_variates, horizon) 分位数预测
    # ============================================================
    try:
        device = next(model.parameters()).device
        target = torch.tensor(payload["target"], dtype=torch.float32, device=device)
        target_mask = torch.ones_like(target, dtype=torch.bool)
        series_ids = torch.zeros(target.shape[0], target.shape[1], dtype=torch.long, device=device)
        horizon = payload.get("horizon", 96)

        with torch.no_grad():
            quantiles = model.forecast(
                {"target": target, "target_mask": target_mask, "series_ids": series_ids},
                horizon=horizon,
                decode_block_size=768,
                has_missing_values=False,
            )

        result_data = {
            "quantiles": quantiles.tolist(),
            "quantile_levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        }
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
