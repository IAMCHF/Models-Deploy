#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本 - mldi-lab-kairos-23m
- /health 断言 status=ok
- /predict 契约: 输入=JSON {past_target: [[..]], prediction_length}(base64), 输出=JSON {forecast}
- 结果校验: 检测服务端 error 前缀, 校验业务字段, 展示真实输出
- 仅依赖标准库 + requests (宿主机无需 numpy)
"""

import os
import sys
import csv
import io
import json
import time
import base64
import logging
import struct
import array
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_mldi-lab-kairos-23m")

BASE_URL = os.environ.get("SERVICE_URL", "http://localhost:8080")
TEST_DIR = Path(__file__).resolve().parent
TIMEOUT = 300


def test_health():
    """测试 /health 端点"""
    logger.info("测试 /health ...")
    resp = requests.get(f"{BASE_URL}/health", timeout=10)
    assert resp.status_code == 200, f"/health 返回状态码 {resp.status_code}"
    data = resp.json()
    assert data.get("status") == "ok", f"/health 返回: {data}"
    logger.info("/health 通过: %s", data)


def check_service_error(decoded: bytes):
    """服务端约定: 推理失败时返回 b'error: ...' 文本, 必须判为 FAIL"""
    head = decoded[:80].decode("utf-8", errors="replace").strip()
    assert not head.startswith("error"), f"服务端返回错误: {head[:300]}"


def load_bytes(name: str) -> bytes:
    """读取固定名称的测试样例文件(不扫描目录, 避免 result.* 污染)"""
    path = TEST_DIR / name
    assert path.exists(), f"测试样例不存在: {path}"
    return path.read_bytes()


def read_csv_series(name: str):
    """读取两列时序 CSV (timestamp, value), 返回 (header, 时间列表, 数值列表)"""
    with open(TEST_DIR / name, "r", encoding="utf-8") as f:
        rows = [r for r in csv.reader(f) if r]
    header = rows[0]
    ts = [r[0] for r in rows[1:]]
    vals = [float(r[1]) for r in rows[1:]]
    return header, ts, vals


def wav_info(path):
    """用标准库 wave 读取 WAV 元信息"""
    import wave
    try:
        with wave.open(str(path), "rb") as w:
            return w.getframerate(), w.getnchannels(), w.getnframes() / max(w.getframerate(), 1)
    except Exception:
        return 0, 0, 0.0


def wav_channel_rms(path):
    """计算每个声道 RMS 能量(标准库)"""
    import wave
    out = []
    try:
        with wave.open(str(path), "rb") as w:
            nch, sw, n = w.getnchannels(), w.getsampwidth(), w.getnframes()
            data = w.readframes(n)
        if sw != 2:
            return out
        samples = array.array("h")
        samples.frombytes(data[: n * nch * 2])
        for c in range(nch):
            ch = samples[c::nch]
            out.append((sum(v * v for v in ch) / max(len(ch), 1)) ** 0.5)
    except Exception:
        pass
    return out


def parse_npy(data: bytes):
    """纯标准库解析 .npy 头部与数值(支持 f4/f8/i4/i2/u1)"""
    assert data[:6] == b"\x93NUMPY", "结果不是 .npy 格式"
    major = data[6]
    if major == 1:
        hlen = struct.unpack_from("<H", data, 8)[0]
        off = 10
    else:
        hlen = struct.unpack_from("<I", data, 8)[0]
        off = 12
    hdr_txt = data[off: off + hlen].decode("latin1")
    hdr_txt = hdr_txt.replace("True", "true").replace("False", "false").replace("None", "null").replace("'", '"')
    hdr = json.loads(hdr_txt)
    shape, descr = hdr["shape"], hdr["descr"]
    n = 1
    for d in shape:
        n *= d
    tmap = {"f4": "f", "f8": "d", "i4": "i", "i2": "h", "u1": "B", "i8": "q"}
    tc = descr.lstrip("<>|=")
    values = None
    if tc in tmap:
        try:
            arr = array.array(tmap[tc])
            arr.frombytes(data[off + hlen: off + hlen + n * arr.itemsize])
            if descr.startswith(">"):
                arr.byteswap()
            values = list(arr)
        except Exception:
            values = None
    return {"shape": shape, "dtype": descr, "n": n, "values": values}


def build_payload():
    """构造 /predict 请求, 返回 base64 数据字符串"""
    header, ts, vals = read_csv_series("sample.csv")
    assert len(vals) >= 50, f"样例数据不足: {len(vals)} 行"
    inner = json.dumps({"past_target": [vals], "prediction_length": 96})
    logger.info("使用测试数据: %s (%d 点), 信封字段: %s", "sample.csv", len(vals), list(json.loads(inner).keys()))
    return base64.b64encode(inner.encode("utf-8")).decode()


def show_result(decoded: bytes):
    """校验并展示服务返回的真实结果"""
    payload = json.loads(decoded.decode("utf-8"))
    assert "error" not in payload, f"服务端推理错误: {str(payload.get('error'))[:300]}"
    fc = payload.get("forecast")
    assert fc, f"无预测: {str(payload)[:200]}"
    series = fc[0] if fc and isinstance(fc[0], list) else fc
    logger.info("Kairos 预测 %d 步, 前8步: %s", len(series), [round(v, 3) for v in series[:8]])


def test_predict():
    logger.info("测试 /predict ...")
    data_b64 = build_payload()
    t0 = time.time()
    resp = requests.post(f"{BASE_URL}/predict", json={"data": data_b64}, timeout=TIMEOUT)
    assert resp.status_code == 200, f"/predict 返回状态码 {resp.status_code}, body: {resp.text[:300]}"
    result = resp.json()
    assert result.get("result"), f"/predict 返回缺少 result 字段: {result}"
    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    check_service_error(decoded)
    show_result(decoded)
    logger.info("推理耗时 %.1f s", time.time() - t0)


def main():
    logger.info("=" * 50)
    logger.info("开始测试: mldi-lab-kairos-23m @ %s", BASE_URL)
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
