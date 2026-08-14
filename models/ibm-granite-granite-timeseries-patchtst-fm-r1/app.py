#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 服务 - ibm-granite/granite-timeseries-patchtst-fm-r1
任务: 时序预测PatchTST(8192上下文)
模态: tabular

[重要] 模型加载代码需从 HuggingFace 模型页面获取真实部署代码后填入下方 TODO 区域。
  - 镜像站优先: https://hf-mirror.com/ibm-granite/granite-timeseries-patchtst-fm-r1
  - 官方兜底:   https://huggingface.co/ibm-granite/granite-timeseries-patchtst-fm-r1
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
logger = logging.getLogger("ibm-granite-granite-timeseries-patchtst-fm-r1")

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

app = FastAPI(title="ibm-granite-granite-timeseries-patchtst-fm-r1", version="1.0.0")

# ============================================================
# 模型加载区域 — 从 HuggingFace 模型页面及 IBM TSFM 仓库获取的真实部署代码
# 来源: https://hf-mirror.com/ibm-granite/granite-timeseries-patchtst-fm-r1
# 官方库: pip install "granite-tsfm[notebooks] @ git+https://github.com/ibm-granite/granite-tsfm.git"
# ============================================================
import json
import torch
import pandas as pd
from tsfm_public import PatchTSTFMForPrediction, TimeSeriesForecastingPipeline

model = None


def load_model():
    """加载 PatchTST-FM 时间序列预测模型"""
    global model
    if model is None:
        logger.info("正在加载 PatchTST-FM 模型，权重目录: %s", WEIGHTS_DIR)
        model = PatchTSTFMForPrediction.from_pretrained(str(WEIGHTS_DIR))
        logger.info("PatchTST-FM 模型加载完成")


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
    # 推理区域 — 基于 PatchTST-FM 官方示例代码适配
    # 输入 JSON 格式: {"target_values": [...], "prediction_length": 96,
    #                  "target_column": "target", "timestamp_column": "date"}
    # 输出 JSON 格式: {"forecast": {...}}
    # ============================================================
    try:
        target_values = payload["target_values"]
        prediction_length = payload.get("prediction_length", 96)
        target_column = payload.get("target_column", "target")
        timestamp_column = payload.get("timestamp_column", "date")

        # 构造 DataFrame
        df = pd.DataFrame({
            timestamp_column: pd.date_range(start="2020-01-01", periods=len(target_values), freq="h"),
            target_column: target_values,
        })

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = TimeSeriesForecastingPipeline(
            model=model,
            id_columns=[],
            timestamp_column=timestamp_column,
            target_columns=[target_column],
            max_context_length=model.config.context_length,
            context_length=min(len(target_values), model.config.context_length),
            prediction_length=prediction_length,
            batch_size=1,
            impute_method=None,
            device=device,
            quantile_levels=[0.1, 0.5, 0.9],
        )

        forecast_df = pipe(df)
        result_data = {"forecast": forecast_df.to_dict(orient="records")}
        output_bytes = json.dumps(result_data, default=str).encode("utf-8")
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
