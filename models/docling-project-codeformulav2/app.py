#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 服务 - docling-project/CodeFormulaV2
任务: 公式/代码识别输出LaTeX
模态: image

模型卡来源: https://hf-mirror.com/docling-project/CodeFormulaV2
模型卡未提供独立使用代码，使用 Docling 官方示例中的 DocumentConverter 管道加载 CodeFormulaV2。
参考: https://github.com/docling-project/docling/blob/main/docs/examples/code_formula_granite_docling.py
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
logger = logging.getLogger("docling-project-codeformulav2")

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

app = FastAPI(title="docling-project-codeformulav2", version="1.0.0")

# ============================================================
# 模型加载区域 — 使用 Docling DocumentConverter + CodeFormulaVlmOptions
# 参考: docling 官方示例 code_formula_granite_docling.py
# 模型卡未提供独立代码片段，此处使用 Docling 管道 API。
# ============================================================
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    CodeFormulaVlmOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import CodeItem, FormulaItem

converter = None


def load_model():
    """加载 CodeFormulaV2 模型 (通过 Docling 管道)"""
    global converter
    if converter is not None:
        return
    # 使用 CodeFormulaV2 预设
    code_formula_options = CodeFormulaVlmOptions.from_preset("codeformulav2")
    logger.info("CodeFormulaV2 预设: model=%s", code_formula_options.model_spec.name)

    pipeline_options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=False,
        do_code_enrichment=True,
        do_formula_enrichment=True,
        code_formula_options=code_formula_options,
    )
    # 离线部署：artifacts_path 指向本地预下载的模型快照
    # 目录结构: weights/docling-project--CodeFormulaV2 + weights/docling-project--docling-layout-heron
    pipeline_options.artifacts_path = str(WEIGHTS_DIR)
    logger.info("artifacts_path(离线本地模型): %s", WEIGHTS_DIR)

    converter = DocumentConverter(
        format_options={
            InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options),
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )
    logger.info("Docling DocumentConverter (CodeFormulaV2) 加载完成")


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
        logger.warning("权重目录不存在，模型将通过 Docling/HuggingFace 自动下载")


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok"}


@app.post("/predict")
async def predict(req: PredictRequest):
    """
    预测接口
    - 输入: req.data (base64 编码的 image 数据，包含公式或代码截图)
    - 输出: result (base64 编码的识别结果，LaTeX 代码或带语言标签的代码文本)
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
    # 推理区域 — 使用 Docling DocumentConverter 提取代码/公式
    # ============================================================
    tmp_path = None
    try:
        # 将图片字节写入临时文件
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(raw_input)
            tmp_path = tmp.name

        # 运行文档转换 (Docling 管道自动检测并提取代码/公式)
        result = converter.convert(tmp_path)
        doc = result.document

        # 提取代码块 (格式: <_language_> code_content)
        code_blocks = [
            item for item, _ in doc.iterate_items() if isinstance(item, CodeItem)
        ]
        # 提取公式 (LaTeX 格式)
        formulas = [
            item for item, _ in doc.iterate_items() if isinstance(item, FormulaItem)
        ]

        # 组装输出
        parts = []
        for item in code_blocks:
            lang = item.code_language if hasattr(item, "code_language") else "unknown"
            parts.append(f"[Code: {lang}]\n{item.text}")
        for item in formulas:
            parts.append(f"[Formula]\n{item.text}")

        if parts:
            output_text = "\n\n".join(parts)
        else:
            # 如果没有检测到代码/公式，导出整个文档文本
            output_text = doc.export_to_markdown() if hasattr(doc, "export_to_markdown") else str(doc)

        output_bytes = output_text.encode("utf-8")
    except Exception as e:
        logger.error("推理失败: %s", e)
        output_bytes = f"error: inference failed - {e}".encode("utf-8")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # 编码输出
    result_b64 = base64.b64encode(output_bytes).decode()
    elapsed = (time.time() - t0) * 1000
    logger.info("推理完成，耗时 %.1f ms", elapsed)
    return PredictResponse(result=result_b64)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
