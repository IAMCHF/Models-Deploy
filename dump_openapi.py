#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导入模型 app 模块并导出 openapi.json（不触发 startup 事件）"""
import importlib.util
import json
import os
import sys

model_dir = sys.argv[1]
out_path = sys.argv[2]

# 模拟 python app.py 的行为：将脚本所在目录加入 sys.path
# （部分模型的依赖如 look2hear/tsfm 直接放在模型目录下）
sys.path.insert(0, model_dir)

spec = importlib.util.spec_from_file_location("app", os.path.join(model_dir, "app.py"))
mod = importlib.util.module_from_spec(spec)
sys.modules["app"] = mod
spec.loader.exec_module(mod)

if not hasattr(mod, "app"):
    print("ERROR: no FastAPI app in module")
    sys.exit(1)

schema = mod.app.openapi()
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(schema, f, indent=2, ensure_ascii=False)
print("OK paths:", list(schema.get("paths", {}).keys()))
