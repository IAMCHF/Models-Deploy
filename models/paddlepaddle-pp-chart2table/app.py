#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 服务 - PaddlePaddle/PP-Chart2Table
任务: 图表转表格
模态: image

模型卡来源: https://hf-mirror.com/PaddlePaddle/PP-Chart2Table
使用 PaddleX create_model API 加载模型。
"""

import os
import base64
import json
import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

# ============================================================
# 镜像站优先：确保模型加载时若需下载额外配置/tokenizer 优先走镜像站
# ============================================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("paddlepaddle-pp-chart2table")

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

app = FastAPI(title="paddlepaddle-pp-chart2table", version="1.0.0")

# ============================================================
# 模型加载区域 — 使用 PaddleX create_model API (来自模型卡 Quick Start)
# 参考: https://hf-mirror.com/PaddlePaddle/PP-Chart2Table
# ============================================================
from paddlex import create_model

model = None


def load_model():
    """加载 PP-Chart2Table 模型"""
    global model
    if model is not None:
        return
    # 优先使用本地权重目录，否则使用模型名称(自动下载)
    if WEIGHTS_DIR.exists() and any(WEIGHTS_DIR.iterdir()):
        logger.info("从本地权重目录加载模型: %s", WEIGHTS_DIR)
        try:
            model = create_model(str(WEIGHTS_DIR))
        except Exception as e:
            logger.warning("本地加载失败(%s)，回退到模型名称加载", e)
            model = create_model("PP-Chart2Table")
    else:
        logger.info("使用模型名称 PP-Chart2Table 加载(自动下载)")
        model = create_model("PP-Chart2Table")
    logger.info("PP-Chart2Table 模型加载完成")


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
        logger.warning("权重目录不存在，模型将通过 PaddleX 自动下载")


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok"}


@app.post("/predict")
async def predict(req: PredictRequest):
    """
    预测接口
    - 输入: req.data (base64 编码的 image 数据)
    - 输出: result (base64 编码的推理结果，JSON 格式的表格数据)
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
    # 推理区域 — 使用 PaddleX model.predict (来自模型卡 Model Usage)
    # ============================================================
    tmp_path = None
    tmp_json_path = None
    try:
        # 将图片字节写入临时文件
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(raw_input)
            tmp_path = tmp.name

        # 运行推理 (模型卡示例代码)
        results = model.predict(
            input={"image": tmp_path},
            batch_size=1
        )

        # 提取结果 (模型卡示例: res.save_to_json)
        output_text = ""
        for res in results:
            # 保存到临时 JSON 文件并读取
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp_json:
                tmp_json_path = tmp_json.name
            res.save_to_json(tmp_json_path)
            with open(tmp_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 模型卡输出格式: {'res': {'image': '...', 'result': '表格文本'}}
            result_dict = data.get("res", data) if isinstance(data, dict) else {}
            output_text = result_dict.get("result", json.dumps(data, ensure_ascii=False))

        output_bytes = output_text.encode("utf-8")
    except Exception as e:
        logger.error("推理失败: %s", e)
        output_bytes = f"error: inference failed - {e}".encode("utf-8")
    finally:
        # 清理临时文件
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if tmp_json_path and os.path.exists(tmp_json_path):
            os.unlink(tmp_json_path)

    # 编码输出
    result = base64.b64encode(output_bytes).decode()
    elapsed = (time.time() - t0) * 1000
    logger.info("推理完成，耗时 %.1f ms", elapsed)
    return PredictResponse(result=result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
