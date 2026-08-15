#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 服务 - autogluon/mitra-classifier
任务: 表格分类基础模型
模态: tabular

模型卡来源: https://hf-mirror.com/autogluon/mitra-classifier
模型卡 "Usage" 使用 AutoGluon 的 TabularPredictor + 'MITRA' 超参数 (in-context learning)：
    from autogluon.tabular import TabularDataset, TabularPredictor
    mitra_predictor = TabularPredictor(label='target')
    mitra_predictor.fit(train_data, hyperparameters={'MITRA': {'fine_tune': False}})
    predictions = mitra_predictor.predict(test_data)
    proba = mitra_predictor.predict_proba(test_data)
依赖: `pip install autogluon.tabular[mitra]` (见 requirements.txt: autogluon)。
Mitra 为 in-context learning 表格基础模型，每次请求需提供带标签的训练数据 (fit 仅缓存上下文，不真正训练)。
"""

import os
import io
import json
import base64
import logging
import tempfile
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ============================================================
# 镜像站优先：确保模型加载时若需下载额外配置/tokenizer 优先走镜像站
# ============================================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 基础模型缓存放到模型目录内(随挂载持久化), 内网离线时直接走本地缓存
os.environ["HF_HOME"] = str(Path(__file__).resolve().parent / "hf_cache")
os.environ["HF_HUB_OFFLINE"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("autogluon-mitra-classifier")

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
MODEL_ID = "autogluon/mitra-classifier"

app = FastAPI(title="autogluon-mitra-classifier", version="1.0.0")

# ============================================================
# 模型加载区域 — Mitra 通过 AutoGluon TabularPredictor 使用 (来自模型卡 Usage)
# 参考: https://hf-mirror.com/autogluon/mitra-classifier
# Mitra 为 in-context learning 模型，无独立持久化权重加载步骤；
# 其基础模型权重由 AutoGluon 在 fit() 时自动下载/加载。
# 此处在启动时校验 autogluon 可用性。
# ============================================================
_autogluon_ready = False


def load_model():
    """校验 AutoGluon (Mitra) 可用性；实际模型在每次请求中通过 fit 创建"""
    global _autogluon_ready
    if _autogluon_ready:
        return
    try:
        from autogluon.tabular import TabularPredictor  # noqa: F401
        _autogluon_ready = True
        logger.info("AutoGluon (Mitra) 可用性校验通过")
    except Exception as e:
        logger.error("AutoGluon 导入失败，请安装 autogluon.tabular[mitra]: %s", e)
        _autogluon_ready = False


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
        train: str        # 训练 CSV 字符串 (含 label 列)
        test: str         # 测试 CSV 字符串 (可含/不含 label 列)
        label: str        # 标签列名 (默认 "target")
        fine_tune: bool   # 是否微调 (默认 False)
        fine_tune_steps: int # 微调步数 (默认 10)
        return_proba: bool   # 是否返回概率 (默认 True)
    - 输出: result (base64 编码的 JSON)
      JSON 字段: {"predictions": [...], "predict_proba": {...}|null}
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
    # 推理区域 — TabularPredictor.fit + predict (来自模型卡 Usage)
    # ============================================================
    tmpdir = None
    try:
        from autogluon.tabular import TabularDataset, TabularPredictor
        payload = json.loads(raw_input)
        label = payload.get("label", "target")
        fine_tune = bool(payload.get("fine_tune", False))
        fine_tune_steps = int(payload.get("fine_tune_steps", 10))
        return_proba = bool(payload.get("return_proba", True))

        train_df = TabularDataset(pd.read_csv(io.StringIO(payload["train"])))
        test_df = TabularDataset(pd.read_csv(io.StringIO(payload["test"])))

        logger.info("Mitra 分类: train_shape=%s, test_shape=%s, label=%s, fine_tune=%s",
                    train_df.shape, test_df.shape, label, fine_tune)

        # 模型卡示例: TabularPredictor(label=...).fit(..., hyperparameters={'MITRA': {...}})
        tmpdir = tempfile.mkdtemp(prefix="mitra_ag_")
        mitra_predictor = TabularPredictor(label=label, path=tmpdir)
        fit_hyper = {"MITRA": {"fine_tune": fine_tune}}
        if fine_tune:
            fit_hyper["MITRA"]["fine_tune_steps"] = fine_tune_steps
        mitra_predictor.fit(train_df, hyperparameters=fit_hyper)

        # 模型卡示例: mitra_predictor.predict(test_data)
        predictions = mitra_predictor.predict(test_df)
        out = {"predictions": predictions.astype(object).tolist()}

        # 模型卡示例: mitra_predictor.predict_proba(test_data)
        if return_proba:
            try:
                proba = mitra_predictor.predict_proba(test_df)
                out["predict_proba"] = proba.astype(object).to_dict(orient="list")
            except Exception as pe:
                out["predict_proba"] = None
                logger.warning("predict_proba 失败: %s", pe)

        output_bytes = json.dumps(out, ensure_ascii=False).encode("utf-8")
    except Exception as e:
        logger.error("推理失败: %s", e)
        output_bytes = f"error: inference failed - {e}".encode("utf-8")
    finally:
        # 清理临时目录
        if tmpdir:
            try:
                import shutil
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass

    # 编码输出
    result = base64.b64encode(output_bytes).decode()
    elapsed = (time.time() - t0) * 1000
    logger.info("推理完成，耗时 %.1f ms", elapsed)
    return PredictResponse(result=result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
