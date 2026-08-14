#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 openapi.json 层注入中文描述（不改任何代码，仅修改文档 JSON）
- info.description: 从 app.py docstring 提取"任务: XXX"
- PredictRequest.data.description: 中文说明
- /predict.description: 若无中文则补充调用说明
"""
import json
import re
from pathlib import Path

MODELS_DIR = Path("/workspace/models")
SKIP = {"datadog-toto-2-0-22m"}

DATA_DESC = "请求输入数据（base64 编码），具体格式见下方接口说明"
PREDICT_DESC = "预测接口：请求体为 PredictRequest（data 字段，base64 编码输入），返回 result（base64 编码输出）。具体输入输出格式见下方接口说明。"


def extract_meta(text):
    """从 docstring 提取 任务/模态/说明"""
    task = None
    modal = None
    m = re.search(r"任务[:：]\s*(.+)", text)
    if m:
        task = m.group(1).strip()
    m = re.search(r"模态[:：]\s*(.+)", text)
    if m:
        modal = m.group(1).strip()
    return task, modal


def has_chinese(s):
    return bool(re.search(r"[\u4e00-\u9fff]", s or ""))


def process(model_dir):
    app_path = model_dir / "app.py"
    json_path = model_dir / "openapi.json"
    if not app_path.exists() or not json_path.exists():
        return None

    text = app_path.read_text(encoding="utf-8")
    task, modal = extract_meta(text)
    d = json.loads(json_path.read_text(encoding="utf-8"))
    changed = []

    # 1. info.description
    if task:
        desc = f"任务: {task}"
        if modal:
            desc += f"；模态: {modal}"
        if d["info"].get("description") != desc:
            d["info"]["description"] = desc
            changed.append("info.description")

    # 2. PredictRequest.data.description
    schemas = d.get("components", {}).get("schemas", {})
    pr = schemas.get("PredictRequest")
    if pr:
        props = pr.get("properties", {})
        if "data" in props:
            if props["data"].get("description") != DATA_DESC:
                props["data"]["description"] = DATA_DESC
                changed.append("data.description")
        if not pr.get("description"):
            pr["description"] = "预测请求：data 为 base64 编码输入"
            changed.append("PredictRequest.description")

    # 3. /predict.description 若无中文则补充
    paths = d.get("paths", {})
    predict = paths.get("/predict", {}).get("post")
    if predict and not has_chinese(predict.get("description", "")):
        predict["description"] = PREDICT_DESC
        changed.append("predict.description")

    json_path.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    return changed


ok = 0
for d in sorted(MODELS_DIR.iterdir()):
    if not d.is_dir() or d.name in SKIP:
        continue
    changed = process(d)
    if changed is None:
        print(f"  [SKIP] {d.name}")
    else:
        print(f"  [OK] {d.name}: {', '.join(changed)}")
        ok += 1

print(f"\n完成: {ok} 个模型 openapi.json 已注入中文描述")
