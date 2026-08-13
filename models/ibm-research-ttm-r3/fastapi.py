#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 服务 - ibm-research/ttm-r3
任务: 时序预测TinyTimeMixer R3(MoE)
模态: tabular

模型卡来源: https://hf-mirror.com/ibm-research/ttm-r3
[注意] 该模型卡 "Example recipes and notebooks" 标注为 "[To be released]"，未提供使用代码。
下方实现依据 IBM 官方 TTM 教程 (developer.ibm.com) 与 sktime 集成 (sktime forecasting/ttm.py
中 `model_path="ibm-research/ttm-r3"`) 的通用 TTM 加载/推理方式适配：
    from transformers import TinyTimeMixerForPrediction
    model = TinyTimeMixerForPrediction.from_pretrained("ibm-research/ttm-r3")
    outputs = model(past_values=context)          # context: [batch, num_channels, context_length]
    predictions = outputs.prediction_outputs      # [batch, num_target_vars, prediction_length]
TTM 为 native transformers `tinytimemixer` 架构，无需 trust_remote_code。
依赖: 仅需基础镜像 (torch, transformers)，无额外依赖 (见 requirements.txt)。
"""

import os
import io
import json
import base64
import logging
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# ============================================================
# 镜像站优先：确保模型加载时若需下载额外配置/tokenizer 优先走镜像站
# ============================================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ibm-research-ttm-r3")

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
MODEL_ID = "ibm-research/ttm-r3"

app = FastAPI(title="ibm-research-ttm-r3", version="1.0.0")

# ============================================================
# 模型加载区域 — TinyTimeMixerForPrediction (transformers, 官方 TTM 加载方式)
# 参考: https://hf-mirror.com/ibm-research/ttm-r3
#   (模型卡无示例代码；依据 IBM 教程与 sktime 集成适配)
# ============================================================
model = None
_device = None


def load_model():
    """加载 TTM-R3 TinyTimeMixer 预测模型"""
    global model, _device
    if model is not None:
        return
    import torch
    from transformers import TinyTimeMixerForPrediction
    _device = "cuda" if torch.cuda.is_available() else "cpu"

    # 优先使用本地权重目录，否则使用模型名称(走镜像站自动下载)
    model_path = str(WEIGHTS_DIR) if WEIGHTS_DIR.exists() and any(WEIGHTS_DIR.iterdir()) else MODEL_ID
    logger.info("加载 TinyTimeMixerForPrediction: %s, device=%s", model_path, _device)
    model = TinyTimeMixerForPrediction.from_pretrained(model_path)
    model.to(_device)
    model.eval()
    logger.info("TTM-R3 模型加载完成, prediction_length=%s",
                getattr(getattr(model, "config", None), "prediction_length", "unknown"))


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
        past_values: list       # 历史时序，形状 [batch, num_channels, context_length]
                                 # 或 [num_channels, context_length] (自动扩为 batch=1)
        prediction_length: int (可选) # 截取的预测步数 (默认返回模型原生 prediction_length)
    - 输出: result (base64 编码的 JSON)
      JSON 字段: {"predictions": [...], "shape": [...], "prediction_length": int}
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
    # 推理区域 — TinyTimeMixerForPrediction forward (官方 TTM 推理方式)
    # past_values: [batch, num_channels, context_length]
    # 输出: outputs.prediction_outputs [batch, num_target_vars, prediction_length]
    # ============================================================
    try:
        import torch
        payload = json.loads(raw_input)
        past_values = np.asarray(payload["past_values"], dtype=np.float32)

        # 统一为 3D: [batch, num_channels, context_length]
        if past_values.ndim == 2:
            past_values = past_values[None, ...]
        elif past_values.ndim != 3:
            raise ValueError(f"past_values 应为 2D 或 3D, 实际 ndim={past_values.ndim}")

        desired_pred_len = payload.get("prediction_length")
        logger.info("TTM-R3 预测: past_values shape=%s", past_values.shape)

        with torch.no_grad():
            past_tensor = torch.from_numpy(past_values).to(_device)
            outputs = model(past_values=past_tensor)

        # TTM 输出: prediction_outputs (点预测)
        if hasattr(outputs, "prediction_outputs") and outputs.prediction_outputs is not None:
            preds = outputs.prediction_outputs
        elif hasattr(outputs, "regression_outputs") and outputs.regression_outputs is not None:
            preds = outputs.regression_outputs
        else:
            preds = outputs[0]

        preds_np = preds.detach().cpu().numpy()
        # 可选截取预测步数
        if desired_pred_len is not None and preds_np.shape[-1] >= int(desired_pred_len):
            preds_np = preds_np[..., :int(desired_pred_len)]

        out_payload = {
            "predictions": preds_np.tolist(),
            "shape": list(preds_np.shape),
            "prediction_length": int(preds_np.shape[-1]),
        }
        output_bytes = json.dumps(out_payload).encode("utf-8")
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
