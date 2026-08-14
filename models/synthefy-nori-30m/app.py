#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 服务 - Synthefy/Nori-30M
任务: 表格回归基础模型(ICL)
模态: tabular

模型卡来源: https://hf-mirror.com/Synthefy/Nori-30M
模型卡 "Usage" 使用 `synthefy_nori` 库的 NoriRegressor (in-context learning 回归)：
    from synthefy_nori import NoriRegressor
    model = NoriRegressor(model="nori-30m")   # 或 model_path="path/to/nori.pt"
    model.fit(X_train, y_train)               # 仅缓存上下文行，不训练
    pred = model.predict(X_test)              # 单次前向推理
依赖: `pip install synthefy-nori` (注意: 当前 requirements.txt 标注无额外依赖，需补充此项)。
模型首次使用时自动从 Hub 下载权重 (nori.pt)。本服务优先使用 weights/nori.pt 本地权重。
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
logger = logging.getLogger("synthefy-nori-30m")

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
MODEL_NAME = "nori-30m"
NORI_PT = WEIGHTS_DIR / "nori.pt"

app = FastAPI(title="synthefy-nori-30m", version="1.0.0")

# ============================================================
# 模型加载区域 — NoriRegressor (来自模型卡 Usage)
# 参考: https://hf-mirror.com/Synthefy/Nori-30M
#   model = NoriRegressor(model="nori-30m")  或  NoriRegressor(model_path="path/to/nori.pt")
# Nori 为 in-context learning 回归模型，fit() 仅缓存上下文行 (无训练循环)，
# 模型实例本身可复用，但每次预测需重新 fit 不同上下文。
# ============================================================
_regressor_cls = None


def load_model():
    """加载 NoriRegressor 类 (实例在每次请求中按上下文创建)"""
    global _regressor_cls
    if _regressor_cls is not None:
        return
    from synthefy_nori import NoriRegressor
    _regressor_cls = NoriRegressor

    # 优先本地权重 nori.pt，否则使用模型名 (首次 predict 时自动从镜像站下载)
    if NORI_PT.exists():
        logger.info("检测到本地权重: %s", NORI_PT)
    else:
        logger.info("未发现本地 nori.pt，将使用 model=%s (首次自动下载)", MODEL_NAME)
    logger.info("NoriRegressor 类加载完成")


def _build_regressor():
    """构建 NoriRegressor 实例 (来自模型卡: NoriRegressor(model=...) / (model_path=...))"""
    if NORI_PT.exists():
        return _regressor_cls(model_path=str(NORI_PT))
    return _regressor_cls(model=MODEL_NAME)


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
    if not NORI_PT.exists():
        logger.warning("未发现本地 nori.pt，模型将在首次请求时从 HuggingFace 镜像站自动下载")


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok"}


@app.post("/predict")
async def predict(req: PredictRequest):
    """
    预测接口 (in-context learning: 每次请求提供带标签训练数据)
    - 输入: req.data (base64 编码的 JSON 载荷)
      JSON 字段:
        X_train: list        # 训练特征 (二维数值数组)
        y_train: list        # 训练标签 (一维数值数组)
        X_test: list         # 测试特征 (二维数值数组)
        output_type: str (可选) # "mean"(默认)/"median"/"mode" (来自模型卡 predict 契约)
        quantiles: list (可选)  # 如 [0.1,0.5,0.9] 时返回预测区间
    - 输出: result (base64 编码的 JSON)
      JSON 字段: {"predictions": [...]} 或 {"lo":[...],"mid":[...],"hi":[...]} (quantiles 模式)
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
    # 推理区域 — NoriRegressor.fit + predict (来自模型卡 Usage)
    # ============================================================
    try:
        payload = json.loads(raw_input)
        X_train = np.asarray(payload["X_train"], dtype=np.float32)
        y_train = np.asarray(payload["y_train"], dtype=np.float32)
        X_test = np.asarray(payload["X_test"], dtype=np.float32)
        output_type = payload.get("output_type", "mean")
        quantiles = payload.get("quantiles")

        logger.info("Nori 回归: X_train=%s, y_train=%s, X_test=%s",
                    X_train.shape, y_train.shape, X_test.shape)

        # 模型卡示例: model = NoriRegressor(model="nori-30m")
        reg = _build_regressor()
        # 模型卡示例: model.fit(X_train, y_train) (仅缓存上下文)
        reg.fit(X_train, y_train)

        if quantiles:
            # 模型卡示例: lo, mid, hi = reg.predict(X_test, output_type="quantiles",
            #                                       quantiles=[0.1, 0.5, 0.9])
            res = reg.predict(X_test, output_type="quantiles", quantiles=quantiles)
            # res 为按 quantiles 顺序的数组序列
            res_list = [np.asarray(r, dtype=float).tolist() for r in res]
            out = {f"q{q}": v for q, v in zip(quantiles, res_list)}
            output_bytes = json.dumps(out).encode("utf-8")
        else:
            # 模型卡示例: pred = model.predict(X_test)
            pred = reg.predict(X_test, output_type=output_type)
            pred_np = np.asarray(pred, dtype=float)
            output_bytes = json.dumps({"predictions": pred_np.tolist()}).encode("utf-8")
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
