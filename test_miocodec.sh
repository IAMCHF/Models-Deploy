#!/bin/bash
PORT=8080
MAX_WAIT=300
MODELS=("aratako-miocodec-25hz-44-1khz-v2")

for model in "${MODELS[@]}"; do
    MODEL_DIR="/workspace/models/${model}"
    echo ">>> 测试: $model"
    pkill -f "app.py" 2>/dev/null
    sleep 1
    for i in $(seq 1 5); do
        if ! ss -tlnp 2>/dev/null | grep -q ":${PORT} " ; then break; fi
        sleep 1; pkill -9 -f "app.py" 2>/dev/null
    done
    cd "${MODEL_DIR}"
    bash start.sh > /tmp/${model}_test.log 2>&1 &
    START_PID=$!
    health_ok=false
    for i in $(seq 1 $MAX_WAIT); do
        resp=$(curl -s -m 2 "http://localhost:${PORT}/health" 2>/dev/null)
        if echo "$resp" | grep -q '"ok"'; then health_ok=true; echo "  健康检查通过 (${i}秒)"; break; fi
        if ! kill -0 $START_PID 2>/dev/null; then sleep 2; if ! kill -0 $START_PID 2>/dev/null; then echo "  [FAIL] 进程已退出"; break; fi; fi
        sleep 1
    done
    if [ "$health_ok" = false ]; then
        echo "  [FAIL] 健康检查失败"; tail -25 /tmp/${model}_test.log 2>/dev/null
        kill -9 $START_PID 2>/dev/null; pkill -f "app.py" 2>/dev/null; continue
    fi
    cd "${MODEL_DIR}/test"
    if python test.py > /tmp/${model}_test_result.log 2>&1; then
        echo "  [PASS] 测试通过"
    else
        echo "  [FAIL] test.py 失败"; tail -15 /tmp/${model}_test_result.log 2>/dev/null
    fi
    kill -9 $START_PID 2>/dev/null; pkill -f "app.py" 2>/dev/null; sleep 2
done
