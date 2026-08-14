import json
import re
import sys

def has_chinese(s):
    return bool(re.search(r"[\u4e00-\u9fff]", s or ""))

ok = 0
bad = []
for m in sys.argv[1:]:
    p = f"/workspace/models/{m}/openapi.json"
    try:
        d = json.load(open(p))
        desc = d["info"].get("description", "")
        paths = list(d["paths"].keys())
        schemas = list(d.get("components", {}).get("schemas", {}).keys())
        issues = []
        if not has_chinese(desc):
            issues.append("info.description无中文")
        if "/health" not in paths or "/predict" not in paths:
            issues.append(f"路径缺失: {paths}")
        if "PredictRequest" not in schemas:
            issues.append("缺少PredictRequest schema")
        # check data/result field descriptions in PredictRequest
        pr = d.get("components", {}).get("schemas", {}).get("PredictRequest", {})
        for fname in ("data", "result"):
            if fname in pr.get("properties", {}):
                fdesc = pr["properties"][fname].get("description", "")
                if not has_chinese(fdesc):
                    issues.append(f"字段{fname}无中文描述")
        if issues:
            bad.append((m, issues))
        else:
            ok += 1
    except Exception as e:
        bad.append((m, [f"ERROR: {e}"]))

print(f"通过: {ok}/{len(sys.argv[1:])}")
for m, issues in bad:
    print(f"  {m}: {issues}")
