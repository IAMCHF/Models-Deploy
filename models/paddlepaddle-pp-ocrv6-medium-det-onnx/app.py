#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 服务 - PaddlePaddle/PP-OCRv6_medium_det_onnx
任务: OCR文本检测(中精度)
模态: image

[重要] 模型加载代码需从 HuggingFace 模型页面获取真实部署代码后填入下方 TODO 区域。
  - 镜像站优先: https://hf-mirror.com/PaddlePaddle/PP-OCRv6_medium_det_onnx
  - 官方兜底:   https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det_onnx
  - 解析页面中 "Use in Transformers" / "Use in vLLM" / "How to use" 等代码片段
  - 加载优先级: transformers 加载 > vLLM 加载
"""

import os
import base64
import json
import tempfile
import logging
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

# ============================================================
# 镜像站优先：确保模型加载时若需下载额外配置/tokenizer 优先走镜像站
# ============================================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("paddlepaddle-pp-ocrv6-medium-det-onnx")

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

app = FastAPI(title="paddlepaddle-pp-ocrv6-medium-det-onnx", version="1.0.0")

# ============================================================
# 模型加载区域 — 基于 PaddlePaddle/PP-OCRv6_medium_det_onnx 官方示例代码适配
# 使用 paddleocr.TextDetection + onnxruntime 引擎
# 源码参考: https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det_onnx
# ============================================================
_model = None


def load_model():
    """加载 PP-OCRv6_medium_det 文本检测模型 (ONNX)"""
    global _model
    if _model is not None:
        return
    from paddleocr import TextDetection
    logger.info("正在加载 PP-OCRv6_medium_det 模型 (ONNX)...")
    _has_local_weights = (
        WEIGHTS_DIR.exists()
        and any(f for f in WEIGHTS_DIR.iterdir() if f.name != ".gitkeep")
    )
    if _has_local_weights:
        _model = TextDetection(model_name="PP-OCRv6_medium_det", model_dir=str(WEIGHTS_DIR), engine="onnxruntime")
    else:
        _model = TextDetection(model_name="PP-OCRv6_medium_det", engine="onnxruntime")
    logger.info("PP-OCRv6_medium_det 模型加载完成")


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
    - 输入: req.data (base64 编码的 image 数据)
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
    # 推理区域 — 基于 PaddleOCR TextDetection 官方示例代码适配
    # 输入: base64 编码的图像 → 文本检测框
    # 输出: 检测结果 JSON (base64 编码)
    # ============================================================
    if _model is None:
        logger.error("模型未加载，无法推理")
        return PredictResponse(result=base64.b64encode(b"error: model not loaded").decode())

    # 将 base64 解码后的图像写入临时文件
    tmp_path = tempfile.mktemp(suffix=".png")
    try:
        with open(tmp_path, "wb") as f:
            f.write(raw_input)

        # 运行文本检测
        output = _model.predict(input=tmp_path, batch_size=1)

        # 收集结果: 只提取检测框与置信度, 避免整图像素数组
        results = []
        for res in output:
            try:
                item = {}
                for k in ("dt_polys", "dt_scores"):
                    try:
                        v = res[k]
                        item[k] = v.tolist() if hasattr(v, "tolist") else v
                    except Exception:
                        pass
                results.append(item or {"raw": str(res)[:2000]})
            except Exception as e:
                logger.warning("结果序列化失败: %s", e)
                results.append({"raw": str(res)[:2000]})

        output_bytes = json.dumps(results, ensure_ascii=False).encode("utf-8")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # 编码输出
    result = base64.b64encode(output_bytes).decode()
    elapsed = (time.time() - t0) * 1000
    logger.info("推理完成，耗时 %.1f ms", elapsed)
    return PredictResponse(result=result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
