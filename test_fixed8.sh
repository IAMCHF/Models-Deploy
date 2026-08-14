#!/bin/bash
# ============================================================
# test_fixed8.sh - 顺序测试修复后的8个模型
# ============================================================

PORT=8080
MAX_WAIT=300
PASS_LIST=()
FAIL_LIST=()

MODELS=(
  "aratako-miocodec-25hz-44-1khz-v2"
  "google-videoprism-lvt-base-f16r288"
  "ibm-granite-granite-timeseries-patchtst-fm-r1"
  "ibm-research-ttm-r3"
  "neoquasar-kronos-base"
  "yuchenshen-fomo-0d"
  "openmoss-team-moss-voicegenerator"
  "datadog-toto-2-0-22m"
)

echo "============================================"
echo "开始测试修复后的模型 - $(date)"
echo "============================================"

for model in "${MODELS[@]}"; do
    MODEL_DIR="/workspace/models/${model}"
    echo ""
    echo ">>> 测试: $model"

    if [ ! -d "${MODEL_DIR}/venv" ]; then
        echo "  [SKIP] 缺少 venv"
        FAIL_LIST+=("$model (no venv)")
        continue
    fi

    # 清理端口
    pkill -f "app.py" 2>/dev/null
    sleep 1
    for i in $(seq 1 5); do
        if ! ss -tlnp 2>/dev/null | grep -q ":${PORT} " ; then
            break
        fi
        sleep 1
        pkill -9 -f "app.py" 2>/dev/null
    done

    # 启动模型
    echo "  启动服务..."
    cd "${MODEL_DIR}"
    bash start.sh > /tmp/${model}_test.log 2>&1 &
    START_PID=$!

    # 等待健康检查
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
        echo "  [FAIL] 健康检查超时或进程退出"
        echo "  --- 最后25行日志 ---"
        tail -25 /tmp/${model}_test.log 2>/dev/null
        FAIL_LIST+=("$model (health check failed)")
        kill -9 $START_PID 2>/dev/null
        pkill -f "app.py" 2>/dev/null
        continue
    fi

    # 运行测试
    echo "  运行 test.py..."
    cd "${MODEL_DIR}/test"
    if python test.py > /tmp/${model}_test_result.log 2>&1; then
        echo "  [PASS] 测试通过"
        PASS_LIST+=("$model")
    else
        echo "  [FAIL] test.py 失败"
        echo "  --- 测试输出 ---"
        tail -15 /tmp/${model}_test_result.log 2>/dev/null
        FAIL_LIST+=("$model (test.py failed)")
    fi

    # 停止服务
    kill -9 $START_PID 2>/dev/null
    pkill -f "app.py" 2>/dev/null
    sleep 2
done

echo ""
echo "============================================"
echo "测试完成 - $(date)"
echo "============================================"
echo ""
echo "通过: ${#PASS_LIST[@]}"
for m in "${PASS_LIST[@]}"; do echo "  PASS $m"; done
echo ""
echo "失败: ${#FAIL_LIST[@]}"
for m in "${FAIL_LIST[@]}"; do echo "  FAIL $m"; done
