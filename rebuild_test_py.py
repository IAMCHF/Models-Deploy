#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rebuild_test_py.py - 按服务契约重写全部 39 个模型的 test/test.py

设计目标:
1. 每个模型按其 app.py 的真实 /predict 契约构造请求(纯文本/JSON信封/音频/图像/视频)
2. 解码结果后先检测 "error:" 前缀与 JSON error 键, 杜绝假通过
3. 仅用标准库(requests 除外), 宿主机无 numpy 也能跑
4. 固定样例文件名, 不再扫描目录(避免 result.* 污染输入)
5. 展示真实输出: 保存音频/嵌入文件, 打印文本/预测值
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# =====================================================================
# 各模型 payload 构造 / 结果展示 代码片段
# =====================================================================

# ---- 通用片段 ----
PAYLOAD_RAW = [
    'data = load_bytes("@SAMPLE@")',
    'logger.info("使用测试数据: @SAMPLE@ (%d bytes)", len(data))',
    'return base64.b64encode(data).decode()',
]

def payload_raw(sample):
    return [l.replace("@SAMPLE@", sample) for l in PAYLOAD_RAW]

DISPLAY_TEXT = [
    'text = decoded.decode("utf-8", errors="replace").strip()',
    'assert len(text) >= 10, f"返回文本过短({len(text)}字符): {text!r}"',
    '(TEST_DIR / "result.txt").write_text(text, encoding="utf-8")',
    'logger.info("@WHAT@(共 %d 字符), 已保存 result.txt, 预览:", len(text))',
    'for ln in text.splitlines()[:15]:',
    '    logger.info("  %s", ln[:150])',
]

def display_text(what):
    return [l.replace("@WHAT@", what) for l in DISPLAY_TEXT]

DISPLAY_EMBED_JSON = [
    'payload = json.loads(decoded.decode("utf-8"))',
    'emb = payload.get("embedding")',
    'assert isinstance(emb, list) and len(emb) > 0, f"无 embedding: {str(payload)[:200]}"',
    'dim = payload.get("dim", len(emb))',
    'assert len(emb) == dim, f"维度不一致: len={len(emb)} dim={dim}"',
    'assert any(abs(x) > 1e-6 for x in emb), "embedding 全零, 推理可能无效"',
    'logger.info("嵌入 dim=%d, 前5个值: %s", dim, [round(x, 5) for x in emb[:5]])',
    'if payload.get("mode"):',
    '    logger.info("编码模式: %s", payload["mode"])',
]

DISPLAY_WAV = [
    'out = TEST_DIR / "result.wav"',
    'out.write_bytes(decoded)',
    'sr, ch, dur = wav_info(out)',
    'assert sr > 0 and dur > 0.5, f"音频无效: sr={sr} dur={dur}"',
    'logger.info("音频已保存: %s (采样率=%dHz, 声道=%d, 时长=%.2fs, %d bytes)", out, sr, ch, dur, len(decoded))',
]

DISPLAY_NPY = [
    'out = TEST_DIR / "result.npy"',
    'out.write_bytes(decoded)',
    'info = parse_npy(decoded)',
    'assert info["n"] > 0, f"嵌入为空: {info}"',
    'vals = info["values"]',
    'if vals:',
    '    assert max(abs(v) for v in vals) > 1e-6, "嵌入全零, 推理可能无效"',
    '    logger.info("嵌入已保存: %s, shape=%s dtype=%s, range=[%.4f, %.4f]", out, info["shape"], info["dtype"], min(vals), max(vals))',
    'else:',
    '    logger.info("嵌入已保存: %s, shape=%s dtype=%s", out, info["shape"], info["dtype"])',
]

def json_display(main_key, human, take=6, is_rows=False):
    """通用 JSON 预测结果展示: 校验无 error 键且主字段非空"""
    lines = [
        'payload = json.loads(decoded.decode("utf-8"))',
        "assert \"error\" not in payload, f\"服务端推理错误: {str(payload.get('error'))[:300]}\"",
        f'main = payload.get("{main_key}")',
        f'assert main, f"返回缺少 {main_key}: {{str(payload)[:200]}}"',
        f'logger.info("{human}, 共 %d 项:", len(main))',
    ]
    if is_rows:
        lines += [
            'for item in main[:%d]:' % take,
            '    logger.info("  %s", json.dumps(item, ensure_ascii=False)[:200])',
        ]
    else:
        lines += [
            'logger.info("  前%d项: %%s", %d, main[:%d])' % (take, take, take),
        ]
    return lines

def csv_values_envelope(sample, inner_fields, context_len=None):
    """读 CSV 时序 -> values 数组 -> JSON 信封"""
    lines = [
        f'header, ts, vals = read_csv_series("{sample}")',
        'assert len(vals) >= %d, f"样例数据不足: {len(vals)} 行"' % (context_len or 50),
        'inner = json.dumps({' + inner_fields + '})',
        'logger.info("使用测试数据: %s (%d 点), 信封字段: %s", "' + sample + '", len(vals), list(json.loads(inner).keys()))',
        'return base64.b64encode(inner.encode("utf-8")).decode()',
    ]
    return lines

# =====================================================================
# 39 个模型规格
# =====================================================================
SPECS = []

def spec(name, timeout, contract, payload_lines, display_lines):
    SPECS.append(dict(name=name, timeout=timeout, contract=contract,
                      payload=payload_lines, display=display_lines))

# ---------- 1. 文本嵌入 ----------
spec("alibaba-nlp-gte-modernbert-base", 120,
     "输入=纯文本(base64), 输出=JSON {embedding, dim}",
     payload_raw("sample.txt"), DISPLAY_EMBED_JSON)

spec("jhu-clsp-mmbert-base", 120,
     "输入=纯文本(base64), 输出=JSON {embedding, shape}",
     payload_raw("sample.txt"), DISPLAY_EMBED_JSON)

spec("mongodb-mdbr-leaf-ir", 120,
     "输入=纯文本(base64), 输出=JSON {embedding, mode, dim}",
     payload_raw("sample.txt"), DISPLAY_EMBED_JSON)

spec("voyageai-voyage-4-nano", 120,
     "输入=纯文本(base64), 输出=JSON {embedding, dim}",
     payload_raw("sample.txt"), DISPLAY_EMBED_JSON)

# ---------- 2. 文本分类 / NLI / 奖励 ----------
spec("dleemiller-finecat-nli-l", 120,
     "输入=JSON {premise, hypothesis}(base64), 输出=JSON {label, logits}",
     payload_raw("sample.json"), [
         'payload = json.loads(decoded.decode("utf-8"))',
         'label = payload.get("label")',
         'assert label in ("entailment", "neutral", "contradiction"), f"非法标签: {label}"',
         'logger.info("NLI 判定: %s (logits=%s)", label, payload.get("logits"))',
         'logger.info("前提: %s", json.loads(load_bytes("sample.json"))["premise"][:80])',
     ])

spec("skywork-skywork-reward-v2-qwen3-0-6b", 300,
     "输入=JSON {prompt, response}(base64), 输出=JSON {score}",
     payload_raw("sample.json"), [
         'payload = json.loads(decoded.decode("utf-8"))',
         'score = float(payload["score"])',
         'logger.info("奖励模型评分: %.4f", score)',
         'assert -100 < score < 100, f"分数异常: {score}"',
     ])

spec("weborganizer-topicclassifier-nourl", 120,
     "输入=网页正文纯文本(base64), 输出=JSON {label, probabilities}",
     payload_raw("sample.txt"), [
         'payload = json.loads(decoded.decode("utf-8"))',
         'label = payload.get("label")',
         'assert label, f"无分类标签: {str(payload)[:200]}"',
         'probs = payload.get("probabilities") or {}',
         'top3 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]',
         'logger.info("主题分类结果: %s", label)',
         'for k, v in top3:',
         '    logger.info("  %s: %.4f", k, v)',
     ])

spec("numind-nuextract3-fp8", 600,
     "输入=JSON {text, template}(base64), 输出=抽取结果纯文本",
     payload_raw("sample.json"), [
         'text = decoded.decode("utf-8", errors="replace").strip()',
         'assert len(text) >= 5, f"抽取结果过短: {text!r}"',
         '(TEST_DIR / "result.txt").write_text(text, encoding="utf-8")',
         'logger.info("信息抽取结果(%d 字符), 已保存 result.txt:", len(text))',
         'try:',
         '    parsed = json.loads(text)',
         '    logger.info("  %s", json.dumps(parsed, ensure_ascii=False, indent=1)[:600])',
         'except Exception:',
         '    for ln in text.splitlines()[:10]:',
         '        logger.info("  %s", ln[:150])',
     ])

# ---------- 3. 表格/时序 (JSON 信封) ----------
spec("autogluon-chronos-2", 300,
     "输入=JSON {context: CSV(id,timestamp,target), prediction_length}(base64), 输出=预测CSV",
     [
         'header, ts, vals = read_csv_series("sample.csv")',
         'buf = io.StringIO()',
         'w = csv.writer(buf)',
         'w.writerow(["id", "timestamp", "target"])',
         'for t, v in zip(ts, vals):',
         '    w.writerow(["series_1", t, v])',
         'inner = json.dumps({"context": buf.getvalue(), "prediction_length": 24,',
         '                     "id_column": "id", "timestamp_column": "timestamp", "target": "target"})',
         'logger.info("使用测试数据: sample.csv (%d 点, 重命名 value->target 并补 id 列)", len(vals))',
         'return base64.b64encode(inner.encode("utf-8")).decode()',
     ], [
         'text = decoded.decode("utf-8", errors="replace").strip()',
         'lines = [ln for ln in text.splitlines() if ln.strip()]',
         'assert len(lines) >= 25, f"预测行数不足: {len(lines)}, 内容: {text[:200]}"',
         '(TEST_DIR / "result.csv").write_text(text, encoding="utf-8")',
         'logger.info("时序预测CSV(共 %d 行, 24 步), 已保存 result.csv, 前8行:", len(lines))',
         'for ln in lines[:8]:',
         '    logger.info("  %s", ln[:150])',
     ])

spec("autogluon-mitra-classifier", 600,
     "输入=JSON {train: CSV, test: CSV, label}(base64), 输出=JSON {predictions}",
     [
         'lines = (TEST_DIR / "sample.csv").read_text(encoding="utf-8").strip().splitlines()',
         'header, rows = lines[0], lines[1:]',
         'n_train = int(len(rows) * 0.8)',
         'train_csv = "\\n".join([header] + rows[:n_train])',
         'test_csv = "\\n".join([header] + rows[n_train:])',
         'inner = json.dumps({"train": train_csv, "test": test_csv, "label": "label"})',
         'logger.info("使用测试数据: sample.csv (train=%d 行, test=%d 行, 标签列=label)", n_train, len(rows) - n_train)',
         'return base64.b64encode(inner.encode("utf-8")).decode()',
     ], [
         'payload = json.loads(decoded.decode("utf-8"))',
         'assert "error" not in payload, f"服务端推理错误: {str(payload.get(\'error\'))[:300]}"',
         'preds = payload.get("predictions")',
         'assert isinstance(preds, list) and len(preds) == 20, f"预测数量异常: {len(preds) if isinstance(preds, list) else preds}"',
         'from collections import Counter',
         'cnt = Counter(str(p) for p in preds)',
         'logger.info("表格分类预测(20 行): %s", dict(cnt))',
         'proba = payload.get("predict_proba")',
         'if proba:',
         '    logger.info("前3行概率: %s", [dict((k, round(v, 3)) for k, v in row.items()) for row in proba[:3]])',
     ])

spec("google-timesfm-2-5-200m-transformers", 300,
     "输入=JSON {past_values: [[..]], forecast_context_len}(base64), 输出=JSON {mean_predictions}",
     csv_values_envelope("sample.csv", '"past_values": [vals], "forecast_context_len": 512'),
     json_display("mean_predictions", "TimesFM 点预测(第一条序列)")[0:3] + [
         'first = main[0] if main and isinstance(main[0], list) else main',
         'assert first, f"预测为空: {str(payload)[:200]}"',
         'logger.info("预测 %d 步, 前8步: %s", len(first), [round(v, 3) for v in first[:8]])',
     ])

spec("ibm-granite-granite-timeseries-patchtst-fm-r1", 300,
     "输入=JSON {target_values: [..], prediction_length}(base64), 输出=JSON {forecast}",
     csv_values_envelope("sample.csv", '"target_values": vals, "prediction_length": 24'),
     json_display("forecast", "PatchTST 分位数预测", take=3, is_rows=True))

spec("ibm-research-ttm-r3", 300,
     "输入=JSON {past_values: [[..]]}(base64), 输出=JSON {predictions, shape}",
     csv_values_envelope("sample.csv", '"past_values": [vals]'),
     [
         'payload = json.loads(decoded.decode("utf-8"))',
         'assert "error" not in payload, f"服务端推理错误: {str(payload.get(\'error\'))[:300]}"',
         'preds = payload.get("predictions")',
         'shape = payload.get("shape")',
         'assert preds, f"无预测: {str(payload)[:200]}"',
         'first = preds[0] if isinstance(preds[0], list) else preds',
         'logger.info("TTM 预测 shape=%s, 前8步: %s", shape, [round(v, 3) for v in first[:8]])',
     ])

spec("neoquasar-kronos-base", 300,
     "输入=JSON {df: K线CSV, pred_len}(base64), 输出=预测CSV",
     [
         'csv_text = (TEST_DIR / "sample.csv").read_text(encoding="utf-8")',
         'inner = json.dumps({"df": csv_text, "pred_len": 24})',
         'logger.info("使用测试数据: sample.csv (OHLCV K线)")',
         'return base64.b64encode(inner.encode("utf-8")).decode()',
     ], [
         'text = decoded.decode("utf-8", errors="replace").strip()',
         'lines = [ln for ln in text.splitlines() if ln.strip()]',
         'assert len(lines) >= 5, f"预测行数不足: {len(lines)}, 内容: {text[:200]}"',
         '(TEST_DIR / "result.csv").write_text(text, encoding="utf-8")',
         'logger.info("K线预测CSV(共 %d 行), 已保存 result.csv, 前8行:", len(lines))',
         'for ln in lines[:8]:',
         '    logger.info("  %s", ln[:150])',
     ])

spec("bytedance-research-timer-s1", 900,
     "输入=JSON {seqs: [[..]], forecast_length}(base64), 输出=JSON {quantiles}",
     csv_values_envelope("sample.csv", '"seqs": [vals], "forecast_length": 32'),
     [
         'payload = json.loads(decoded.decode("utf-8"))',
         'assert "error" not in payload, f"服务端推理错误: {str(payload.get(\'error\'))[:300]}"',
         'qs = payload.get("quantiles")',
         'levels = payload.get("quantile_levels")',
         'assert qs, f"无分位数预测: {str(payload)[:200]}"',
         'mid = qs[len(qs) // 2] if isinstance(qs, list) else qs',
         'series = mid[0] if mid and isinstance(mid[0], list) else mid',
         'logger.info("Timer-S1 分位数层: %s", levels)',
         'logger.info("中位数预测前8步: %s", [round(v, 3) for v in series[:8]])',
     ])

spec("k-iwa-time-anchor-modernbert-32m", 300,
     "输入=JSON {target_context: [..], prediction_length, quantile_levels}(base64), 输出=JSON {forecast_rows}",
     csv_values_envelope("sample.csv",
         '"target_context": vals, "prediction_length": 24, "quantile_levels": [0.1, 0.5, 0.9]'),
     json_display("forecast_rows", "TimeAnchor 预测", take=3, is_rows=True))

spec("mldi-lab-kairos-23m", 300,
     "输入=JSON {past_target: [[..]], prediction_length}(base64), 输出=JSON {forecast}",
     csv_values_envelope("sample.csv", '"past_target": [vals], "prediction_length": 96'),
     [
         'payload = json.loads(decoded.decode("utf-8"))',
         'assert "error" not in payload, f"服务端推理错误: {str(payload.get(\'error\'))[:300]}"',
         'fc = payload.get("forecast")',
         'assert fc, f"无预测: {str(payload)[:200]}"',
         'series = fc[0] if fc and isinstance(fc[0], list) else fc',
         'logger.info("Kairos 预测 %d 步, 前8步: %s", len(series), [round(v, 3) for v in series[:8]])',
     ])

spec("prior-labs-tabpfn-v2-clf", 300,
     "输入=JSON {X_train, y_train, X_test}(base64 文件原文), 输出=JSON {predictions, probabilities}",
     payload_raw("sample.json"), [
         'payload = json.loads(decoded.decode("utf-8"))',
         'assert "error" not in payload, f"服务端推理错误: {str(payload.get(\'error\'))[:300]}"',
         'preds = payload.get("predictions")',
         'assert isinstance(preds, list) and len(preds) == 20, f"预测数量异常: {preds}"',
         'from collections import Counter',
         'logger.info("TabPFN-v2 分类预测(20 样本): %s", dict(Counter(str(p) for p in preds)))',
         'proba = payload.get("probabilities")',
         'if proba:',
         '    logger.info("第1个样本概率: %s", [round(v, 3) for v in proba[0]])',
     ])

spec("prior-labs-tabpfn-v2-reg", 300,
     "输入=JSON {X_train, y_train, X_test}(base64 文件原文), 输出=JSON {predictions}",
     payload_raw("sample.json"),
     json_display("predictions", "TabPFN-v2 回归预测"))

spec("synthefy-nori-30m", 300,
     "输入=JSON {X_train, y_train, X_test}(base64 文件原文), 输出=JSON {predictions}",
     payload_raw("sample.json"),
     json_display("predictions", "NoRI 预测"))

spec("yuchenshen-fomo-0d", 300,
     "输入=JSON {train_x, test_x}(base64 文件原文), 输出=JSON {predictions}",
     payload_raw("sample.json"), [
         'payload = json.loads(decoded.decode("utf-8"))',
         'assert "error" not in payload, f"服务端推理错误: {str(payload.get(\'error\'))[:300]}"',
         'preds = payload.get("predictions")',
         'assert preds, f"无预测: {str(payload)[:200]}"',
         'flat = preds[0] if isinstance(preds[0], list) else preds',
         'vals2 = [v for row in flat for v in (row if isinstance(row, list) else [row])] if isinstance(flat[0], list) else flat',
         'logger.info("FoMo 异常检测: %d 个序列 x %d 步得分, 第1序列前8步: %s",',
         '            len(preds), len(flat), [round(float(v), 3) for v in (flat[0] if isinstance(flat[0], list) else flat)[:8]])',
     ])

# ---------- 4. 音频 ----------
spec("aratako-miocodec-25hz-44-1khz-v2", 300,
     "输入=WAV(base64), 输出=重建WAV",
     payload_raw("sample.wav"), DISPLAY_WAV)

spec("ibm-granite-granite-speech-4-1-2b", 300,
     "输入=16kHz 语音WAV(base64), 输出=转写文本",
     payload_raw("sample.wav"), [
         'text = decoded.decode("utf-8", errors="replace").strip()',
         'assert len(text) >= 4, f"转写文本过短: {text!r}"',
         'assert "[Noise]" not in text and "[noise]" not in text, f"转写为噪声标记, 语音输入无效: {text!r}"',
         'logger.info("ASR 转写结果(%d 字符): %s", len(text), text[:500])',
     ])

spec("microsoft-vibevoice-asr-hf", 900,
     "输入=语音WAV(base64), 输出=转写文本",
     payload_raw("sample.wav"), [
         'text = decoded.decode("utf-8", errors="replace").strip()',
         'assert len(text) >= 4, f"转写文本过短: {text!r}"',
         'assert "[Noise]" not in text and "[noise]" not in text, f"转写为噪声标记, 语音输入无效: {text!r}"',
         'logger.info("ASR 转写结果(%d 字符): %s", len(text), text[:500])',
     ])

spec("openmoss-team-moss-tts-local-transformer-v1-5", 600,
     "输入=中文文本(base64), 输出=合成语音WAV",
     payload_raw("sample.txt"), DISPLAY_WAV)

spec("openmoss-team-moss-voicegenerator", 600,
     "输入=JSON {text, instruction}(base64), 输出=合成语音WAV",
     payload_raw("sample.json"), DISPLAY_WAV)

spec("jusperlee-tiger-dnr", 300,
     "输入=44.1kHz 人声+音乐混合WAV(base64), 输出=3声道分离WAV(人声/音效/音乐)",
     payload_raw("sample.wav"), [
         'out = TEST_DIR / "result.wav"',
         'out.write_bytes(decoded)',
         'sr, ch, dur = wav_info(out)',
         'assert sr > 0 and dur > 0.5, f"音频无效: sr={sr} dur={dur}"',
         'assert ch == 3, f"应输出3声道(人声/音效/音乐), 实际 {ch} 声道"',
         'rms = wav_channel_rms(out)',
         'logger.info("分离结果已保存: %s (采样率=%dHz, 3声道, 时长=%.2fs)", out, sr, dur)',
         'names = ["人声", "音效", "音乐"]',
         'for i, r in enumerate(rms):',
         '    logger.info("  声道%d(%s) RMS 能量: %.1f", i, names[i] if i < 3 else "?", r)',
         'assert any(r > 0 for r in rms), "分离结果全静音"',
     ])

# ---------- 5. 图像 / 视频 ----------
spec("docling-project-codeformulav2", 600,
     "输入=公式/代码截图PNG(base64), 输出=识别文本([Code]/[Formula])",
     payload_raw("sample.png"), display_text("公式/代码识别结果"))

spec("facebook-vjepa2-vitl-fpc64-256", 300,
     "输入=256x256 mp4 视频(base64), 输出=.npy 视频嵌入",
     payload_raw("sample.mp4"), DISPLAY_NPY)

spec("google-videoprism-lvt-base-f16r288", 300,
     "输入=288x288 mp4 视频(base64), 输出=.npy 视频嵌入",
     payload_raw("sample.mp4"), DISPLAY_NPY)

spec("opengvlab-videomaev2-base", 300,
     "输入=224x224 mp4 视频(base64), 输出=.npy 视频嵌入",
     payload_raw("sample.mp4"), DISPLAY_NPY)

def deepfake_display():
    return [
        'preds = json.loads(decoded.decode("utf-8"))',
        'assert isinstance(preds, list) and preds, f"返回非列表: {str(preds)[:200]}"',
        'for p in preds:',
        '    logger.info("  样本1(女性肖像): label=%s score=%.4f", p["label"], p["score"])',
        'male_b64 = base64.b64encode(load_bytes("sample_male.jpg")).decode()',
        'r2 = requests.post(f"{BASE_URL}/predict", json={"data": male_b64}, timeout=TIMEOUT)',
        'assert r2.status_code == 200, f"第二个样本请求失败: {r2.status_code}"',
        'd2 = base64.b64decode(r2.json()["result"])',
        'check_service_error(d2)',
        'preds2 = json.loads(d2.decode("utf-8"))',
        'for p in preds2:',
        '    logger.info("  样本2(男性肖像): label=%s score=%.4f", p["label"], p["score"])',
        'labels = {p["label"] for p in preds} | {p["label"] for p in preds2}',
        'assert labels <= {"fake", "real"}, f"非法标签: {labels}"',
        'logger.info("深度伪造检测: 两张高分辨率人脸均给出有效判定 (AI生成图像预期判 fake)")',
    ]

spec("koreapeter-ms-eff-gcvit-deepfake-b0-ff-plus-plus", 300,
     "输入=人脸JPG(base64, 两张), 输出=JSON [{label, score}]",
     payload_raw("sample_female.jpg"), deepfake_display())

spec("koreapeter-ms-eff-gcvit-deepfake-b5-ff-plus-plus", 300,
     "输入=人脸JPG(base64, 两张), 输出=JSON [{label, score}]",
     payload_raw("sample_female.jpg"), deepfake_display())

spec("paddlepaddle-pp-chart2table", 300,
     "输入=图表PNG(base64), 输出=图表转表格文本",
     payload_raw("sample.png"), display_text("图表转表格结果"))

DISPLAY_BOXES = [
    'payload = json.loads(decoded.decode("utf-8"))',
    'boxes = payload if isinstance(payload, list) else (payload.get("boxes") or payload.get("results") or [])',
    'assert isinstance(boxes, list) and len(boxes) > 0, f"未检测到任何区域: {str(payload)[:200]}"',
    'logger.info("版面分析: 检测到 %d 个区域, 前3个:", len(boxes))',
    'for b in boxes[:3]:',
    '    logger.info("  %s", json.dumps(b, ensure_ascii=False)[:200])',
]

DISPLAY_TEXTS = [
    'payload = json.loads(decoded.decode("utf-8"))',
    'texts = payload if isinstance(payload, list) else (payload.get("texts") or payload.get("results") or [])',
    'assert isinstance(texts, list) and len(texts) > 0, f"未识别到文字: {str(payload)[:200]}"',
    'logger.info("OCR 识别到 %d 条文本:", len(texts))',
    'for t in texts[:10]:',
    '    logger.info("  %s", json.dumps(t, ensure_ascii=False)[:200])',
]

spec("paddlepaddle-pp-docblocklayout", 300,
     "输入=文档页PNG(base64), 输出=JSON 布局区域列表",
     payload_raw("sample.png"), DISPLAY_BOXES)

spec("paddlepaddle-pp-doclayout-plus-l", 300,
     "输入=文档页PNG(base64), 输出=JSON 布局区域列表",
     payload_raw("sample.png"), DISPLAY_BOXES)

spec("paddlepaddle-pp-ocrv6-medium-det-onnx", 300,
     "输入=文档页PNG(base64), 输出=JSON 文本检测框列表",
     payload_raw("sample.png"), DISPLAY_BOXES)

spec("paddlepaddle-pp-ocrv6-small-det-onnx", 300,
     "输入=文档页PNG(base64), 输出=JSON 文本检测框列表",
     payload_raw("sample.png"), DISPLAY_BOXES)

spec("paddlepaddle-pp-ocrv6-small-rec-onnx", 300,
     "输入=文字行PNG(base64), 输出=JSON 识别文本列表",
     payload_raw("sample.png"), DISPLAY_TEXTS)

# =====================================================================
# 模板
# =====================================================================
TEMPLATE = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本 - @@NAME@@
- /health 断言 status=ok
- /predict 契约: @@CONTRACT@@
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
logger = logging.getLogger("test_@@NAME@@")

BASE_URL = os.environ.get("SERVICE_URL", "http://localhost:8080")
TEST_DIR = Path(__file__).resolve().parent
TIMEOUT = @@TIMEOUT@@


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
    assert data[:6] == b"\\x93NUMPY", "结果不是 .npy 格式"
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
@@PAYLOAD@@


def show_result(decoded: bytes):
    """校验并展示服务返回的真实结果"""
@@DISPLAY@@


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
    logger.info("开始测试: @@NAME@@ @ %s", BASE_URL)
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
'''


def indent(lines, n=4):
    pad = " " * n
    return "\n".join(pad + l if l.strip() else l for l in lines)


def main():
    names = [d.name for d in (ROOT / "models").iterdir() if d.is_dir()]
    written, missing = [], []
    for s in SPECS:
        content = (TEMPLATE
                   .replace("@@NAME@@", s["name"])
                   .replace("@@TIMEOUT@@", str(s["timeout"]))
                   .replace("@@CONTRACT@@", s["contract"])
                   .replace("@@PAYLOAD@@", indent(s["payload"]))
                   .replace("@@DISPLAY@@", indent(s["display"])))
        out = ROOT / "models" / s["name"] / "test" / "test.py"
        out.write_text(content, encoding="utf-8")
        written.append(s["name"])
    spec_names = {s["name"] for s in SPECS}
    for n in names:
        if n not in spec_names:
            missing.append(n)
    print(f"已重写 {len(written)} 个 test.py")
    if missing:
        print("!! 以下模型目录没有对应 spec:", missing)
        sys.exit(1)


if __name__ == "__main__":
    main()
