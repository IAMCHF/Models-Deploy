#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本 - paddlepaddle-pp-ocrv6-small-det-onnx
- 检查服务是否运行在 localhost:8080
- 测试 /health 断言 status=ok
- 测试 /predict：读取同目录下的样例文件，base64 编码后发送，验证返回非空
"""

import os
import sys
import base64
import logging
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_paddlepaddle-pp-ocrv6-small-det-onnx")

BASE_URL = os.environ.get("SERVICE_URL", "http://localhost:8080")
# 测试数据放在 test.py 同级目录
TEST_DIR = Path(__file__).resolve().parent


def test_health():
    """测试 /health 端点"""
    logger.info("测试 /health ...")
    resp = requests.get(f"{BASE_URL}/health", timeout=10)
    assert resp.status_code == 200, f"/health 返回状态码 {resp.status_code}"
    data = resp.json()
    assert data.get("status") == "ok", f"/health 返回: {data}"
    logger.info("/health 通过: %s", data)


def get_test_data():
    """获取测试数据（base64 编码），从 test.py 同级目录查找样例文件"""
    # 查找同目录下的样例文件（排除 .py 和 .gitkeep）
    sample_files = [f for f in TEST_DIR.iterdir()
                    if f.is_file() and not f.name.startswith(".") and f.suffix != ".py"]

    if sample_files:
        sample = sample_files[0]
        logger.info("使用测试数据: %s", sample.name)
        with open(sample, "rb") as f:
            return base64.b64encode(f.read()).decode()
    else:
        # 根据模态生成随机数据
        logger.warning("无测试数据文件，根据模态(%s)生成随机数据", "image")
        if "image" == "text":
            raw = "这是一条测试文本。This is a test text.".encode("utf-8")
        elif "image" in ("image", "video"):
            import random
            raw = bytes([random.randint(0, 255) for _ in range(1024)])
        elif "image" == "audio":
            import struct, random
            raw = b"".join(struct.pack("<h", random.randint(-32768, 32767)) for _ in range(8000))
        else:  # tabular
            raw = b"1.0,2.0,3.0\n4.0,5.0,6.0"
        return base64.b64encode(raw).decode()


def test_predict():
    """测试 /predict 端点"""
    logger.info("测试 /predict ...")
    data_b64 = get_test_data()
    payload = {"data": data_b64}
    resp = requests.post(f"{BASE_URL}/predict", json=payload, timeout=60)
    assert resp.status_code == 200, f"/predict 返回状态码 {resp.status_code}, body: {resp.text[:500]}"
    result = resp.json()
    assert "result" in result, f"/predict 返回缺少 result 字段: {result}"
    assert result["result"], "/predict 返回 result 为空"
    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    logger.info("/predict 通过，结果长度: %d bytes", len(decoded))


def main():
    logger.info("=" * 50)
    logger.info("开始测试: paddlepaddle-pp-ocrv6-small-det-onnx @ %s", BASE_URL)
    logger.info("=" * 50)
    try:
        test_health()
    except Exception as e:
        logger.error("/health 测试失败: %s", e)
        sys.exit(1)
    try:
        test_predict()
    except Exception as e:
        logger.error("/predict 测试失败: %s", e)
        sys.exit(1)
    logger.info("=" * 50)
    logger.info("全部测试通过!")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
