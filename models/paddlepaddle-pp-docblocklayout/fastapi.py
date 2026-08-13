#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 服务 - PaddlePaddle/PP-DocBlockLayout
任务: 文档版块布局定位
模态: image

[重要] 模型加载代码需从 HuggingFace 模型页面获取真实部署代码后填入下方 TODO 区域。
  - 镜像站优先: https://hf-mirror.com/PaddlePaddle/PP-DocBlockLayout
  - 官方兜底:   https://huggingface.co/PaddlePaddle/PP-DocBlockLayout
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
logger = logging.getLogger("paddlepaddle-pp-docblocklayout")

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

app = FastAPI(title="paddlepaddle-pp-docblocklayout", version="1.0.0")

# ============================================================
# 模型加载区域 — 基于 PaddlePaddle/PP-DocBlockLayout 官方示例代码适配
# 使用 paddleocr.LayoutDetection (PaddlePaddle 引擎)
# 源码参考: https://huggingface.co/PaddlePaddle/PP-DocBlockLayout
# ============================================================
_model = None


def load_model():
    """加载 PP-DocBlockLayout 文档版块布局检测模型"""
    global _model
    if _model is not None:
        return
    from paddleocr import LayoutDetection
    logger.info("正在加载 PP-DocBlockLayout 模型...")
    _has_local_weights = (
        WEIGHTS_DIR.exists()
        and any(f for f in WEIGHTS_DIR.iterdir() if f.name != ".gitkeep")
    )
    if _has_local_weights:
        _model = LayoutDetection(model_dir=str(WEIGHTS_DIR))
    else:
        _model = LayoutDetection(model_name="PP-DocBlockLayout")
    logger.info("PP-DocBlockLayout 模型加载完成")


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
    # 推理区域 — 基于 PaddleOCR LayoutDetection 官方示例代码适配
    # 输入: base64 编码的图像 → 文档版块布局检测
    # 输出: 布局检测结果 JSON (base64 编码)
    # ============================================================
    if _model is None:
        logger.error("模型未加载，无法推理")
        return PredictResponse(result=base64.b64encode(b"error: model not loaded").decode())

    # 将 base64 解码后的图像写入临时文件
    tmp_path = tempfile.mktemp()
    try:
        with open(tmp_path, "wb") as f:
            f.write(raw_input)

        # 运行布局检测 (启用 layout_nms)
        output = _model.predict(input=tmp_path, batch_size=1, layout_nms=True)

        # 收集结果
        results = []
        for res in output:
            try:
                if hasattr(res, "json") and res.json:
                    results.append(json.loads(res.json))
                elif hasattr(res, "to_dict"):
                    results.append(res.to_dict())
                else:
                    results.append({"raw": str(res)})
            except Exception as e:
                logger.warning("结果序列化失败: %s", e)
                results.append({"raw": str(res)})

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
