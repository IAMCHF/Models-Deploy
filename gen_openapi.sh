#!/bin/bash
# 为所有通过测试的模型生成 openapi.json（导入app模块，不触发startup）
# 用法: bash gen_openapi.sh
SKIP_MODELS="datadog-toto-2-0-22m"
PASS=0
FAIL=0
FAILED_LIST=""

for model_dir in /workspace/models/*/; do
    model=$(basename "$model_dir")
    # 跳过未通过测试的模型
    if echo "$SKIP_MODELS" | grep -q "$model"; then
        echo "[SKIP] $model (未通过测试)"
        continue
    fi
    # 跳过没有 app.py 的目录
    if [ ! -f "$model_dir/app.py" ]; then
        echo "[SKIP] $model (无 app.py)"
        continue
    fi
    venv_py="$model_dir/venv/bin/python"
    if [ ! -x "$venv_py" ]; then
        echo "[FAIL] $model (无 venv)"
        FAIL=$((FAIL+1)); FAILED_LIST="$FAILED_LIST $model"
        continue
    fi
    out="$model_dir/openapi.json"
    echo ">>> $model"
    if timeout 300 "$venv_py" /tmp/dump_openapi.py "$model_dir" "$out" > /tmp/gen_${model}.log 2>&1; then
        size=$(stat -c%s "$out" 2>/dev/null || echo 0)
        echo "    [PASS] openapi.json ($size bytes)"
        PASS=$((PASS+1))
    else
        echo "    [FAIL] $(tail -2 /tmp/gen_${model}.log | head -1)"
        FAIL=$((FAIL+1)); FAILED_LIST="$FAILED_LIST $model"
    fi
done

echo ""
echo "========== 汇总 =========="
echo "通过: $PASS, 失败: $FAIL"
if [ -n "$FAILED_LIST" ]; then
    echo "失败列表:$FAILED_LIST"
fi
