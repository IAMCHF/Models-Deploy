#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 服务 - autogluon/chronos-2
任务: 时序预测(单变量/多变量/协变量)
模态: tabular

模型卡来源: https://hf-mirror.com/autogluon/chronos-2
模型卡 "Running the model locally" 使用 `chronos` 包的 `Chronos2Pipeline`：
    from chronos import Chronos2Pipeline
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
    pred_df = pipeline.predict_df(context_df, future_df=..., prediction_length=24,
                                  quantile_levels=[0.1, 0.5, 0.9],
                                  id_column="id", timestamp_column="timestamp", target="target")
依赖: `pip install "chronos-forecasting>=2.0"` (见 requirements.txt)。
输入为 pandas DataFrame (历史时序 + 可选协变量)，输出为分位数预测 DataFrame。
"""

import os
import io
import json
import base64
import logging
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ============================================================
# 镜像站优先：确保模型加载时若需下载额外配置/tokenizer 优先走镜像站
# ============================================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("autogluon-chronos-2")

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
# 模型卡示例使用 "amazon/chronos-2"，本仓库对应 "autogluon/chronos-2"
MODEL_ID = "autogluon/chronos-2"

app = FastAPI(title="autogluon-chronos-2", version="1.0.0")

# ============================================================
# 模型加载区域 — Chronos2Pipeline (来自模型卡 Running the model locally)
# 参考: https://hf-mirror.com/autogluon/chronos-2
# ============================================================
pipeline = None
_device_map = None


def load_model():
    """加载 Chronos-2 时序预测 pipeline"""
    global pipeline, _device_map
    if pipeline is not None:
        return
    from chronos import Chronos2Pipeline
    try:
        import torch
        _device_map = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        _device_map = "cpu"

    # 优先使用本地权重目录，否则使用模型名称(走镜像站自动下载)
    model_path = str(WEIGHTS_DIR) if WEIGHTS_DIR.exists() and any(WEIGHTS_DIR.iterdir()) else MODEL_ID
    logger.info("加载 Chronos2Pipeline: %s, device_map=%s", model_path, _device_map)
    pipeline = Chronos2Pipeline.from_pretrained(model_path, device_map=_device_map)
    logger.info("Chronos-2 pipeline 加载完成")


class PredictRequest(BaseModel):
    """预测请求：data 为 base64 编码输入"""
    data: str


class PredictResponse(BaseModel):
    """预测响应：result 为 base64 编码输出"""
    result: str


@app.on_event("startup")
async def startup_event():
    logger.info("服务启动，权重目录: %s", WEIGHTS_DIR)
    load_model()
    if not WEIGHTS_DIR.exists():
        logger.warning("权重目录不存在，模型将通过 HuggingFace 镜像站自动下载")


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok"}


@app.post("/predict")
async def predict(req: PredictRequest):
    """
    预测接口
    - 输入: req.data (base64 编码的 JSON 载荷)
      JSON 字段:
        context: str            # 历史时序 CSV 字符串 (含 id/timestamp/target 列)
        future: str (可选)      # 未来协变量 CSV 字符串 (不含 target)
        prediction_length: int  # 预测步数 (默认 24)
        quantile_levels: list   # 分位数 (默认 [0.1, 0.5, 0.9])
        id_column: str          # 默认 "id"
        timestamp_column: str   # 默认 "timestamp"
        target: str             # 默认 "target"
    - 输出: result (base64 编码的预测结果 CSV 字符串)
    """
    import time
    t0 = time.time()

    # 解码输入
    try:
        raw_input = base64.b64decode(req.data).decode("utf-8")
    except Exception as e:
        logger.error("base64 解码失败: %s", e)
        return PredictResponse(result=base64.b64encode(b"error: invalid base64").decode())

    # ============================================================
    # 推理区域 — 使用 Chronos2Pipeline.predict_df (来自模型卡)
    # ============================================================
    try:
        payload = json.loads(raw_input)
        context_csv = payload["context"]
        context_df = pd.read_csv(io.StringIO(context_csv))

        future_df = None
        if payload.get("future"):
            future_df = pd.read_csv(io.StringIO(payload["future"]))

        prediction_length = int(payload.get("prediction_length", 24))
        quantile_levels = payload.get("quantile_levels", [0.1, 0.5, 0.9])
        id_column = payload.get("id_column", "id")
        timestamp_column = payload.get("timestamp_column", "timestamp")
        target = payload.get("target", "target")

        logger.info("Chronos-2 预测: prediction_length=%s, context_shape=%s",
                    prediction_length, context_df.shape)

        # 模型卡示例: pipeline.predict_df(context_df, future_df=..., prediction_length=...,
        #                                 quantile_levels=..., id_column=..., timestamp_column=...,
        #                                 target=...)
        pred_df = pipeline.predict_df(
            context_df,
            future_df=future_df,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            id_column=id_column,
            timestamp_column=timestamp_column,
            target=target,
        )

        # 输出为 CSV 字符串
        buf = io.StringIO()
        pred_df.to_csv(buf, index=False)
        output_bytes = buf.getvalue().encode("utf-8")
    except Exception as e:
        logger.error("推理失败: %s", e)
        output_bytes = f"error: inference failed - {e}".encode("utf-8")

    # 编码输出
    result = base64.b64encode(output_bytes).decode()
    elapsed = (time.time() - t0) * 1000
    logger.info("推理完成，耗时 %.1f ms", elapsed)
    return PredictResponse(result=result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
