#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为 39 个模型的 test.py 打补丁：在 /predict 通过后，展示/保存实际返回结果。
- 音频模型: 保存 result.wav + 打印采样率/时长
- 向量模型: 保存 result.npy + 打印 shape
- JSON 模型: 打印关键字段
- CSV 模型: 打印前几行
- 文本模型: 直接打印文本

用法: python patch_test_py.py
"""
import re
from pathlib import Path

MODELS_ROOT = Path(__file__).resolve().parent / "models"

OLD_BLOCK = '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    logger.info("/predict 通过，结果长度: %d bytes", len(decoded))'''

# 每个模型 -> 替换后的展示代码
SNIPPETS = {
    # ============ 音频模型: 保存 .wav + wave 信息 ============
    "aratako-miocodec-25hz-44-1khz-v2": '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    out_path = TEST_DIR / "result.wav"
    out_path.write_bytes(decoded)
    try:
        import wave
        with wave.open(str(out_path), "rb") as wf:
            sr = wf.getframerate(); ch = wf.getnchannels(); n = wf.getnframes()
            dur = n / sr if sr else 0
        logger.info("/predict 通过，重建音频已保存: %s", out_path)
        logger.info("音频信息: 采样率=%dHz 声道=%d 时长=%.2fs 大小=%d bytes", sr, ch, dur, len(decoded))
    except Exception:
        logger.info("/predict 通过，音频已保存: %s (%d bytes)", out_path, len(decoded))''',

    "jusperlee-tiger-dnr": '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    out_path = TEST_DIR / "result.wav"
    out_path.write_bytes(decoded)
    try:
        import wave
        with wave.open(str(out_path), "rb") as wf:
            sr = wf.getframerate(); ch = wf.getnchannels(); n = wf.getnframes()
            dur = n / sr if sr else 0
        logger.info("/predict 通过，分离音频已保存: %s", out_path)
        logger.info("音频信息: 采样率=%dHz 声道=%d 时长=%.2fs 大小=%d bytes (3声道: 人声/音效/音乐)", sr, ch, dur, len(decoded))
    except Exception:
        logger.info("/predict 通过，音频已保存: %s (%d bytes)", out_path, len(decoded))''',

    "openmoss-team-moss-tts-local-transformer-v1-5": '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    out_path = TEST_DIR / "result.wav"
    out_path.write_bytes(decoded)
    try:
        import wave
        with wave.open(str(out_path), "rb") as wf:
            sr = wf.getframerate(); ch = wf.getnchannels(); n = wf.getnframes()
            dur = n / sr if sr else 0
        logger.info("/predict 通过，TTS合成音频已保存: %s", out_path)
        logger.info("音频信息: 采样率=%dHz 声道=%d 时长=%.2fs 大小=%d bytes", sr, ch, dur, len(decoded))
    except Exception:
        logger.info("/predict 通过，音频已保存: %s (%d bytes)", out_path, len(decoded))''',

    "openmoss-team-moss-voicegenerator": '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    out_path = TEST_DIR / "result.wav"
    out_path.write_bytes(decoded)
    try:
        import wave
        with wave.open(str(out_path), "rb") as wf:
            sr = wf.getframerate(); ch = wf.getnchannels(); n = wf.getnframes()
            dur = n / sr if sr else 0
        logger.info("/predict 通过，语音生成音频已保存: %s", out_path)
        logger.info("音频信息: 采样率=%dHz 声道=%d 时长=%.2fs 大小=%d bytes", sr, ch, dur, len(decoded))
    except Exception:
        logger.info("/predict 通过，音频已保存: %s (%d bytes)", out_path, len(decoded))''',

    # ============ 向量模型: 保存 .npy + shape ============
    "facebook-vjepa2-vitl-fpc64-256": '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    out_path = TEST_DIR / "result.npy"
    out_path.write_bytes(decoded)
    import io
    import numpy as np
    arr = np.load(io.BytesIO(decoded))
    logger.info("/predict 通过，视频嵌入已保存: %s", out_path)
    logger.info("嵌入信息: shape=%s dtype=%s 数值范围=[%.4f, %.4f]",
                arr.shape, arr.dtype, float(arr.min()), float(arr.max()))''',

    "google-videoprism-lvt-base-f16r288": '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    out_path = TEST_DIR / "result.npy"
    out_path.write_bytes(decoded)
    import io
    import numpy as np
    arr = np.load(io.BytesIO(decoded))
    logger.info("/predict 通过，视频嵌入已保存: %s", out_path)
    logger.info("嵌入信息: shape=%s dtype=%s 数值范围=[%.4f, %.4f]",
                arr.shape, arr.dtype, float(arr.min()), float(arr.max()))''',

    "opengvlab-videomaev2-base": '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    out_path = TEST_DIR / "result.npy"
    out_path.write_bytes(decoded)
    import io
    import numpy as np
    arr = np.load(io.BytesIO(decoded))
    logger.info("/predict 通过，视频特征已保存: %s", out_path)
    logger.info("特征信息: shape=%s dtype=%s 数值范围=[%.4f, %.4f]",
                arr.shape, arr.dtype, float(arr.min()), float(arr.max()))''',

    "voyageai-voyage-4-nano": '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    out_path = TEST_DIR / "result.npy"
    out_path.write_bytes(decoded)
    import io
    import numpy as np
    arr = np.load(io.BytesIO(decoded))
    logger.info("/predict 通过，文本嵌入已保存: %s", out_path)
    logger.info("嵌入信息: shape=%s dtype=%s 前5个值=%s",
                arr.shape, arr.dtype, arr.flatten()[:5].tolist())''',

    # ============ CSV 模型: 打印前几行 ============
    "autogluon-chronos-2": '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    text = decoded.decode("utf-8", errors="replace")
    lines = text.strip().splitlines()
    logger.info("/predict 通过，预测结果(CSV) 共%d行，前8行:", len(lines))
    for ln in lines[:8]:
        logger.info("  %s", ln)''',

    "neoquasar-kronos-base": '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    text = decoded.decode("utf-8", errors="replace")
    lines = text.strip().splitlines()
    logger.info("/predict 通过，K线预测(CSV) 共%d行，前8行:", len(lines))
    for ln in lines[:8]:
        logger.info("  %s", ln)''',

    # ============ 文本模型: 直接打印 ============
    "docling-project-codeformulav2": '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    text = decoded.decode("utf-8", errors="replace")
    logger.info("/predict 通过，识别文本(%d 字符):", len(text))
    logger.info("%s", text[:2000])''',

    "ibm-granite-granite-speech-4-1-2b": '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    text = decoded.decode("utf-8", errors="replace")
    logger.info("/predict 通过，ASR转写文本(%d 字符):", len(text))
    logger.info("%s", text[:2000])''',

    "microsoft-vibevoice-asr-hf": '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    text = decoded.decode("utf-8", errors="replace")
    logger.info("/predict 通过，ASR转写文本(%d 字符):", len(text))
    logger.info("%s", text[:2000])''',

    "numind-nuextract3-fp8": '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    text = decoded.decode("utf-8", errors="replace")
    logger.info("/predict 通过，抽取结果(%d 字符):", len(text))
    logger.info("%s", text[:2000])''',

    "paddlepaddle-pp-chart2table": '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    text = decoded.decode("utf-8", errors="replace")
    logger.info("/predict 通过，图表转表格(%d 字符):", len(text))
    logger.info("%s", text[:2000])''',

    # ============ JSON 模型: 打印关键字段 ============
    "alibaba-nlp-gte-modernbert-base": '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    import json
    payload = json.loads(decoded.decode("utf-8"))
    logger.info("/predict 通过，文本嵌入 dim=%s", payload.get("dim"))
    logger.info("embedding 前5个值: %s", payload.get("embedding", [])[:5])''',

    "autogluon-mitra-classifier": '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    import json
    payload = json.loads(decoded.decode("utf-8"))
    logger.info("/predict 通过，分类预测: %s", payload.get("predictions"))
    logger.info("概率: %s", payload.get("predict_proba"))''',

    "bytedance-research-timer-s1": '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    import json
    payload = json.loads(decoded.decode("utf-8"))
    q = payload.get("quantiles")
    logger.info("/predict 通过，时序分位数预测 levels=%s", payload.get("quantile_levels"))
    logger.info("quantiles shape=%s 首行=%s",
                [len(x) if isinstance(x, list) else "?" for x in (q or [])],
                (q[0][0] if isinstance(q, list) and q and isinstance(q[0], list) else q))''',

    "dleemiller-finecat-nli-l": '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    import json
    payload = json.loads(decoded.decode("utf-8"))
    logger.info("/predict 通过，NLI判断: label=%s (class_id=%s)", payload.get("label"), payload.get("predicted_class_id"))
    logger.info("logits: %s", payload.get("logits"))''',

    "google-timesfm-2-5-200m-transformers": '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    import json
    payload = json.loads(decoded.decode("utf-8"))
    logger.info("/predict 通过，时序预测 mean_predictions=%s", payload.get("mean_predictions"))''',

    "ibm-granite-granite-timeseries-patchtst-fm-r1": '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    import json
    payload = json.loads(decoded.decode("utf-8"))
    fc = payload.get("forecast")
    logger.info("/predict 通过，时序预测 forecast 共%d条，前3条:", len(fc) if isinstance(fc, list) else 0)
    for rec in (fc or [])[:3]:
        logger.info("  %s", rec)''',

    "ibm-research-ttm-r3": '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    import json
    payload = json.loads(decoded.decode("utf-8"))
    logger.info("/predict 通过，时序预测 shape=%s prediction_length=%s",
                payload.get("shape"), payload.get("prediction_length"))
    logger.info("predictions 前10个: %s", payload.get("predictions", [])[:10])''',

    "jhu-clsp-mmbert-base": '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    import json
    payload = json.loads(decoded.decode("utf-8"))
    logger.info("/predict 通过，跨语言嵌入 shape=%s", payload.get("shape"))
    logger.info("embedding 前5个值: %s", payload.get("embedding", [])[:5])''',

    "k-iwa-time-anchor-modernbert-32m": '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    import json
    payload = json.loads(decoded.decode("utf-8"))
    rows = payload.get("forecast_rows")
    logger.info("/predict 通过，时序预测 forecast_rows 共%d条，前3条:", len(rows) if isinstance(rows, list) else 0)
    for rec in (rows or [])[:3]:
        logger.info("  %s", rec)''',

    "koreapeter-ms-eff-gcvit-deepfake-b0-ff-plus-plus": '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    import json
    payload = json.loads(decoded.decode("utf-8"))
    logger.info("/predict 通过，深度伪造检测结果:")
    for item in payload:
        logger.info("  label=%s score=%.4f", item.get("label"), item.get("score"))''',

    "koreapeter-ms-eff-gcvit-deepfake-b5-ff-plus-plus": '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    import json
    payload = json.loads(decoded.decode("utf-8"))
    logger.info("/predict 通过，深度伪造检测结果:")
    for item in payload:
        logger.info("  label=%s score=%.4f", item.get("label"), item.get("score"))''',

    "mldi-lab-kairos-23m": '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    import json
    payload = json.loads(decoded.decode("utf-8"))
    fc = payload.get("forecast")
    logger.info("/predict 通过，时序预测 forecast 共%d条，前3条:", len(fc) if isinstance(fc, list) else 0)
    for rec in (fc or [])[:3]:
        logger.info("  %s", rec)''',

    "mongodb-mdbr-leaf-ir": '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    import json
    payload = json.loads(decoded.decode("utf-8"))
    logger.info("/predict 通过，检索嵌入 mode=%s dim=%s", payload.get("mode"), payload.get("dim"))
    logger.info("embedding 前5个值: %s", payload.get("embedding", [])[:5])''',

    "paddlepaddle-pp-docblocklayout": '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    import json
    payload = json.loads(decoded.decode("utf-8"))
    boxes = payload if isinstance(payload, list) else payload.get("boxes", payload.get("results", []))
    logger.info("/predict 通过，文档布局检测框共%d个，前3个:", len(boxes) if isinstance(boxes, list) else 0)
    for b in (boxes if isinstance(boxes, list) else [])[:3]:
        logger.info("  %s", b)''',

    "paddlepaddle-pp-doclayout-plus-l": '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    import json
    payload = json.loads(decoded.decode("utf-8"))
    boxes = payload if isinstance(payload, list) else payload.get("boxes", payload.get("results", []))
    logger.info("/predict 通过，文档布局检测框共%d个，前3个:", len(boxes) if isinstance(boxes, list) else 0)
    for b in (boxes if isinstance(boxes, list) else [])[:3]:
        logger.info("  %s", b)''',

    "paddlepaddle-pp-ocrv6-medium-det-onnx": '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    import json
    payload = json.loads(decoded.decode("utf-8"))
    boxes = payload if isinstance(payload, list) else payload.get("boxes", payload.get("results", []))
    logger.info("/predict 通过，文本检测框共%d个，前3个:", len(boxes) if isinstance(boxes, list) else 0)
    for b in (boxes if isinstance(boxes, list) else [])[:3]:
        logger.info("  %s", b)''',

    "paddlepaddle-pp-ocrv6-small-det-onnx": '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    import json
    payload = json.loads(decoded.decode("utf-8"))
    boxes = payload if isinstance(payload, list) else payload.get("boxes", payload.get("results", []))
    logger.info("/predict 通过，文本检测框共%d个，前3个:", len(boxes) if isinstance(boxes, list) else 0)
    for b in (boxes if isinstance(boxes, list) else [])[:3]:
        logger.info("  %s", b)''',

    "paddlepaddle-pp-ocrv6-small-rec-onnx": '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    import json
    payload = json.loads(decoded.decode("utf-8"))
    texts = payload if isinstance(payload, list) else payload.get("texts", payload.get("results", []))
    logger.info("/predict 通过，OCR识别文本共%d条:", len(texts) if isinstance(texts, list) else 0)
    for t in (texts if isinstance(texts, list) else [])[:10]:
        logger.info("  %s", t)''',

    "prior-labs-tabpfn-v2-clf": '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    import json
    payload = json.loads(decoded.decode("utf-8"))
    logger.info("/predict 通过，表格分类预测: %s", payload.get("predictions"))
    logger.info("概率矩阵行数=%s", len(payload.get("probabilities", [])))''',

    "prior-labs-tabpfn-v2-reg": '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    import json
    payload = json.loads(decoded.decode("utf-8"))
    logger.info("/predict 通过，表格回归预测: %s", payload.get("predictions"))''',

    "skywork-skywork-reward-v2-qwen3-0-6b": '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    import json
    payload = json.loads(decoded.decode("utf-8"))
    logger.info("/predict 通过，奖励模型 score=%s", payload.get("score"))''',

    "synthefy-nori-30m": '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    import json
    payload = json.loads(decoded.decode("utf-8"))
    if "predictions" in payload:
        logger.info("/predict 通过，时序预测 predictions=%s", payload.get("predictions"))
    else:
        logger.info("/predict 通过，分位数预测: %s", {k: v[:3] for k, v in payload.items() if isinstance(v, list)})''',

    "weborganizer-topicclassifier-nourl": '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    import json
    payload = json.loads(decoded.decode("utf-8"))
    logger.info("/predict 通过，主题分类: label=%s (class_id=%s)", payload.get("label"), payload.get("predicted_class_id"))
    probs = payload.get("probabilities", {})
    top = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3] if isinstance(probs, dict) else []
    logger.info("Top3概率: %s", top)''',

    "yuchenshen-fomo-0d": '''    decoded = base64.b64decode(result["result"])
    assert len(decoded) > 0, "/predict 返回 result 解码后为空"
    import json
    payload = json.loads(decoded.decode("utf-8"))
    preds = payload.get("predictions")
    logger.info("/predict 通过，异常检测 predictions shape=%s", [len(preds), len(preds[0]), len(preds[0][0])] if isinstance(preds, list) and preds and isinstance(preds[0], list) and preds[0] and isinstance(preds[0][0], list) else "?")
    logger.info("首条得分: %s", preds[0][0][:3] if isinstance(preds, list) and preds and isinstance(preds[0], list) and preds[0] and isinstance(preds[0][0], list) else preds)''',
}


def patch_file(model: str, snippet: str) -> bool:
    tp = MODELS_ROOT / model / "test" / "test.py"
    if not tp.exists():
        print(f"[SKIP] {model}: test.py not found")
        return False
    text = tp.read_text(encoding="utf-8")
    if OLD_BLOCK not in text:
        print(f"[SKIP] {model}: OLD_BLOCK not found (already patched?)")
        return False
    new_text = text.replace(OLD_BLOCK, snippet, 1)
    tp.write_text(new_text, encoding="utf-8")
    print(f"[OK] {model}")
    return True


if __name__ == "__main__":
    ok = fail = 0
    for model, snippet in SNIPPETS.items():
        if patch_file(model, snippet):
            ok += 1
        else:
            fail += 1
    print(f"\nPatched {ok} files, skipped/failed {fail} (total {len(SNIPPETS)})")
