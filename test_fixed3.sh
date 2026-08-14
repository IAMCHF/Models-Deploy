#!/bin/bash
# Retest batch 3 fixes

PORT=8080
MAX_WAIT=240
PASS_LIST=()
FAIL_LIST=()

FIXED_MODELS=(
  "opengvlab-videomaev2-base"
  "paddlepaddle-pp-chart2table"
  "voyageai-voyage-4-nano"
  "openmoss-team-moss-tts-local-transformer-v1-5"
)

echo "============================================"
echo "重测批次3 - $(date)"
echo "============================================"
echo ""

for model in "${FIXED_MODELS[@]}"; do
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
    bash start.sh > /tmp/${model}_retest3.log 2>&1 &
    START_PID=$!

    echo "  等待健康检查 (最多${MAX_WAIT}秒)..."
    health_ok=false
    HEALTH_URL="http://localhost:${PORT}/health"
    for i in $(seq 1 $MAX_WAIT); do
        resp=$(curl -s -m 2 "$HEALTH_URL" 2>/dev/null)
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
        echo "  --- 最后20行日志 ---"
        tail -20 /tmp/${model}_retest3.log 2>/dev/null
        FAIL_LIST+=("$model")
        kill -9 $START_PID 2>/dev/null
        pkill -f "app.py" 2>/dev/null
        echo ""
        continue
    fi

    echo "  运行 test.py..."
    cd "${MODEL_DIR}/test"
    if python test.py > /tmp/${model}_retest3_result.log 2>&1; then
        echo "  [PASS] 测试通过"
        PASS_LIST+=("$model")
    else
        echo "  [FAIL] test.py 失败"
        echo "  --- 测试输出 ---"
        cat /tmp/${model}_retest3_result.log 2>/dev/null | tail -10
        echo "  --- 服务日志最后10行 ---"
        tail -10 /tmp/${model}_retest3.log 2>/dev/null
        FAIL_LIST+=("$model")
    fi

    kill -9 $START_PID 2>/dev/null
    pkill -f "app.py" 2>/dev/null
    sleep 2
    echo ""
done

echo "============================================"
echo "重测批次3完成 - $(date)"
echo "============================================"
echo ""
echo "通过: ${#PASS_LIST[@]}"
for m in "${PASS_LIST[@]}"; do echo "  ✓ $m"; done
echo ""
echo "失败: ${#FAIL_LIST[@]}"
for m in "${FAIL_LIST[@]}"; do echo "  ✗ $m"; done
