#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 服务 - NeoQuasar/Kronos-base
任务: 金融时序预测(K线基础模型)
模态: tabular

模型卡来源: https://hf-mirror.com/NeoQuasar/Kronos-base
模型卡 "Getting Started: Making Forecasts" 使用 GitHub 仓库 (`shiyu-coder/Kronos`) 提供的
`model` 模块中的 Kronos / KronosTokenizer / KronosPredictor 类：
    from model import Kronos, KronosTokenizer, KronosPredictor
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)
    pred_df = predictor.predict(df=x_df, x_timestamp=x_timestamp,
                                y_timestamp=y_timestamp, pred_len=pred_len,
                                T=1.0, top_p=0.9, sample_count=1)
注意:
  - 该模型依赖 GitHub 仓库的 `model.py`，需先 `git clone https://github.com/shiyu-coder/Kronos`
    并将其加入 PYTHONPATH (或 `pip install -r requirements.txt` 后置于工作目录)。
  - 模型与分词器分属两个仓库: NeoQuasar/Kronos-base (模型) + NeoQuasar/Kronos-Tokenizer-base (分词器)。
  - max_context 对于 Kronos-base 为 512。
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
logger = logging.getLogger("neoquasar-kronos-base")

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
# 分词器权重目录 (模型与分词器分属两个仓库)
TOKENIZER_DIR = Path(__file__).resolve().parent / "weights_tokenizer"
MODEL_ID = "NeoQuasar/Kronos-base"
TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
MAX_CONTEXT = 512  # Kronos-base 的最大上下文长度 (来自模型卡 Model Zoo)

app = FastAPI(title="neoquasar-kronos-base", version="1.0.0")

# ============================================================
# 模型加载区域 — Kronos + KronosTokenizer + KronosPredictor (来自模型卡)
# 参考: https://hf-mirror.com/NeoQuasar/Kronos-base
# ============================================================
predictor = None
_device = None


def load_model():
    """加载 Kronos-base 金融时序预测模型"""
    global predictor, _device
    if predictor is not None:
        return
    # model 模块来自 GitHub 仓库 shiyu-coder/Kronos
    from model import Kronos, KronosTokenizer, KronosPredictor
    try:
        import torch
        _device = "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        _device = "cpu"

    # 模型路径: 优先本地 weights/，否则使用 HF 模型名(走镜像站自动下载)
    model_path = str(WEIGHTS_DIR) if WEIGHTS_DIR.exists() and any(WEIGHTS_DIR.iterdir()) else MODEL_ID
    # 分词器路径: 优先本地 weights_tokenizer/，否则使用 HF 模型名
    tok_path = str(TOKENIZER_DIR) if TOKENIZER_DIR.exists() and any(TOKENIZER_DIR.iterdir()) else TOKENIZER_ID
    logger.info("加载 Kronos 模型: %s, 分词器: %s, device=%s", model_path, tok_path, _device)

    tokenizer = KronosTokenizer.from_pretrained(tok_path)
    model = Kronos.from_pretrained(model_path)
    # 模型卡示例: KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)
    predictor = KronosPredictor(model, tokenizer, device=_device, max_context=MAX_CONTEXT)
    logger.info("Kronos-base predictor 加载完成")


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


def _infer_freq(ts_series: pd.Series):
    """从时间序列推断频率，用于自动生成未来时间戳"""
    try:
        inferred = pd.infer_freq(pd.to_datetime(ts_series))
        if inferred:
            return inferred
    except Exception:
        pass
    # 退化为取中位时间差
    try:
        diffs = pd.to_datetime(ts_series).diff().dropna()
        if len(diffs) > 0:
            return diffs.median()
    except Exception:
        pass
    return "h"  # 最终兜底: 1小时


@app.post("/predict")
async def predict(req: PredictRequest):
    """
    预测接口
    - 输入: req.data (base64 编码的 JSON 载荷)
      JSON 字段:
        df: str               # K线 CSV 字符串, 列含 open/high/low/close[,volume,amount][,timestamps]
        pred_len: int         # 预测步数 (默认 24)
        lookback: int (可选)  # 历史窗口长度 (默认 min(len(df), 512))
        y_timestamp: list (可选) # 未来时间戳列表 (字符串)；不提供则自动推断生成
        T: float              # 采样温度 (默认 1.0)
        top_p: float          # 核采样概率 (默认 0.9)
        sample_count: int     # 采样路径数 (默认 1)
    - 输出: result (base64 编码的预测结果 CSV 字符串, 含 open/high/low/close/volume/amount)
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
    # 推理区域 — 使用 KronosPredictor.predict (来自模型卡)
    # ============================================================
    try:
        payload = json.loads(raw_input)
        df = pd.read_csv(io.StringIO(payload["df"]))

        pred_len = int(payload.get("pred_len", 24))
        lookback = int(payload.get("lookback", min(len(df), MAX_CONTEXT)))
        lookback = max(1, min(lookback, MAX_CONTEXT, len(df)))
        T = float(payload.get("T", 1.0))
        top_p = float(payload.get("top_p", 0.9))
        sample_count = int(payload.get("sample_count", 1))

        # 提取 OHLCV 列 (模型卡: ['open','high','low','close','volume','amount'])
        ohlcv_cols = [c for c in ["open", "high", "low", "close", "volume", "amount"] if c in df.columns]
        if not {"open", "high", "low", "close"}.issubset(set(ohlcv_cols)):
            raise ValueError("CSV 必须包含 open/high/low/close 列")

        # 时间戳列
        if "timestamps" in df.columns:
            ts = pd.Series(pd.to_datetime(df["timestamps"]))
        else:
            ts = pd.Series(pd.to_datetime(pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq="h")))

        x_df = df.iloc[:lookback][ohlcv_cols].reset_index(drop=True)
        x_timestamp = ts.iloc[:lookback].reset_index(drop=True)

        # 未来时间戳
        if payload.get("y_timestamp"):
            y_timestamp = pd.Series(pd.to_datetime(payload["y_timestamp"]))
        else:
            freq = _infer_freq(ts.iloc[:lookback])
            try:
                y_index = pd.date_range(start=ts.iloc[lookback - 1], periods=pred_len + 1, freq=freq)[1:]
            except Exception:
                y_index = pd.date_range(start=ts.iloc[lookback - 1], periods=pred_len + 1, freq="h")[1:]
            y_timestamp = pd.Series(y_index)

        logger.info("Kronos 预测: lookback=%s, pred_len=%s, cols=%s", lookback, pred_len, ohlcv_cols)

        # 模型卡示例: predictor.predict(df=x_df, x_timestamp=x_timestamp,
        #                               y_timestamp=y_timestamp, pred_len=pred_len,
        #                               T=1.0, top_p=0.9, sample_count=1)
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=T,
            top_p=top_p,
            sample_count=sample_count,
        )

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
