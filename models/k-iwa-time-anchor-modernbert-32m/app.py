#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 服务 - K-Iwa/time-anchor-modernbert-32m
任务: 时序预测(ModernBERT编码器,分位数预测)
模态: tabular

[重要] 模型加载代码需从 HuggingFace 模型页面获取真实部署代码后填入下方 TODO 区域。
  - 镜像站优先: https://hf-mirror.com/K-Iwa/time-anchor-modernbert-32m
  - 官方兜底:   https://huggingface.co/K-Iwa/time-anchor-modernbert-32m
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
logger = logging.getLogger("k-iwa-time-anchor-modernbert-32m")

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

app = FastAPI(title="k-iwa-time-anchor-modernbert-32m", version="1.0.0")

# ============================================================
# 模型加载区域 — 从 HuggingFace 模型页面获取的真实部署代码
# 来源: https://hf-mirror.com/K-Iwa/time-anchor-modernbert-32m
# 官方库: pip install time-anchor
# 注意: time_anchor 的 predict_time_anchor 函数内部管理模型加载，
#       此处仅导入函数并设置 checkpoint 路径
# ============================================================
import json
import numpy as np
try:
    # ModernBERT 推理可能触发 Triton JIT 编译, 容器无 python3-dev 时会失败;
    # 出错回退 eager 而非返回 500
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass
from time_anchor import predict_time_anchor

# checkpoint 路径：优先本地 weights 目录，回退到 HF Hub model id
CHECKPOINT = str(WEIGHTS_DIR) if WEIGHTS_DIR.exists() else "K-Iwa/time-anchor-modernbert-32m"

model = None  # predict_time_anchor 内部管理模型加载


def load_model():
    """验证 time_anchor 包可用（模型在推理时按需加载）"""
    global model
    if model is None:
        logger.info("time_anchor 包已导入，checkpoint 路径: %s", CHECKPOINT)
        model = True  # 标记为已初始化


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
    # 推理区域 — 基于 time-anchor 官方示例代码适配
    # 输入 JSON 格式: {"target_context": [...], "prediction_length": 64,
    #                  "quantile_levels": [0.1, 0.5, 0.9]}
    # 输出 JSON 格式: {"forecast_rows": [...]}
    # ============================================================
    try:
        target_context = np.array(payload["target_context"], dtype=np.float32)
        prediction_length = payload.get("prediction_length", 64)
        quantile_levels = tuple(payload.get("quantile_levels", [0.1, 0.5, 0.9]))

        result = predict_time_anchor(
            CHECKPOINT,
            target_context=target_context,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
        )

        forecast_rows = result.forecast_rows
        # 将 forecast_rows 转为可序列化的 list of dict
        serializable_rows = []
        for row in forecast_rows:
            serializable_rows.append({
                k: (v.tolist() if hasattr(v, "tolist") else v)
                for k, v in row.items()
            })

        result_data = {"forecast_rows": serializable_rows}
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
