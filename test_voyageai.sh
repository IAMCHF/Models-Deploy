#!/bin/bash
# Retest voyageai after forward fix
PORT=8080
MAX_WAIT=240

model="voyageai-voyage-4-nano"
MODEL_DIR="/workspace/models/${model}"
echo ">>> 测试: $model"

pkill -f "app.py" 2>/dev/null
sleep 2
for i in $(seq 1 5); do
    if ! ss -tlnp 2>/dev/null | grep -q ":${PORT} " ; then
        break
    fi
    sleep 1
    pkill -9 -f "app.py" 2>/dev/null
done

echo "  启动服务..."
cd "${MODEL_DIR}"
bash start.sh > /tmp/${model}_retest4.log 2>&1 &
START_PID=$!

echo "  等待健康检查 (最多${MAX_WAIT}秒)..."
health_ok=false
for i in $(seq 1 $MAX_WAIT); do
    resp=$(curl -s -m 2 "http://localhost:${PORT}/health" 2>/dev/null)
    if echo "$resp" | grep -q '"ok"'; then
        health_ok=true
        echo "  健康检查通过 (${i}秒)"
        break
    fi
    if ! kill -0 $START_PID 2>/dev/null; then
        sleep 2
        if ! kill -0 $START_PID 2>/dev/null; then
            echo "  [FAIL] 进程已退出"
            break
        fi
    fi
    sleep 1
done

if [ "$health_ok" = false ]; then
    echo "  [FAIL] 健康检查失败"
    tail -20 /tmp/${model}_retest4.log 2>/dev/null
    kill -9 $START_PID 2>/dev/null
    pkill -f "app.py" 2>/dev/null
    exit 1
fi

echo "  运行 test.py..."
cd "${MODEL_DIR}/test"
if python test.py > /tmp/${model}_retest4_result.log 2>&1; then
    echo "  [PASS] 测试通过"
else
    echo "  [FAIL] test.py 失败"
    cat /tmp/${model}_retest4_result.log 2>/dev/null | tail -10
    echo "  --- 服务日志 ---"
    tail -15 /tmp/${model}_retest4.log 2>/dev/null
fi

kill -9 $START_PID 2>/dev/null
pkill -f "app.py" 2>/dev/null
