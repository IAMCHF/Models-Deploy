#!/bin/bash
# ============================================================
# test_sequential.sh - 顺序启动/测试/停止所有模型
# 完全模拟内网部署：start.sh 启动 → /health 就绪 → test.py 测试 → 停止
# 所有服务使用 8080 端口，一个测完再测下一个
#
# 用法:
#   bash test_sequential.sh                   # 测试全部模型
#   bash test_sequential.sh 模型名1 模型名2    # 只测试指定模型
# ============================================================

PORT=8080
BASE_URL="http://localhost:${PORT}"
MAX_WAIT=600          # 等待服务就绪最大秒数（大模型加载慢）
HEALTH_INTERVAL=2     # 健康检查间隔秒数
LOG_DIR="/tmp/test_sequential"

mkdir -p "$LOG_DIR"

# 获取模型列表
if [ $# -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=($(ls -d /workspace/models/*/ 2>/dev/null | xargs -n1 basename | sort))
fi

PASS=0
FAIL=0
FAILED_LIST=""
SKIP_LIST=""

echo "============================================"
echo "顺序测试开始 - $(date)"
echo "模型数量: ${#MODELS[@]}"
echo "============================================"

for model in "${MODELS[@]}"; do
    MODEL_DIR="/workspace/models/${model}"
    if [ ! -d "$MODEL_DIR" ]; then
        echo "[SKIP] $model: 目录不存在"
        SKIP_LIST="$SKIP_LIST $model"
        continue
    fi
    if [ ! -f "$MODEL_DIR/start.sh" ]; then
        echo "[SKIP] $model: 无 start.sh"
        SKIP_LIST="$SKIP_LIST $model"
        continue
    fi
    if [ ! -f "$MODEL_DIR/test/test.py" ]; then
        echo "[SKIP] $model: 无 test/test.py"
        SKIP_LIST="$SKIP_LIST $model"
        continue
    fi

    echo ""
    echo "--------------------------------------------"
    echo ">>> [$model] 启动服务 (start.sh)"
    cd "$MODEL_DIR"
    ./start.sh > "$LOG_DIR/${model}_service.log" 2>&1 &
    SERVICE_PID=$!

    # 等待 /health 就绪
    READY=0
    for ((i=1; i<=MAX_WAIT; i+=HEALTH_INTERVAL)); do
        sleep $HEALTH_INTERVAL
        if curl -s -m 2 "${BASE_URL}/health" 2>/dev/null | grep -q '"ok"'; then
            READY=1
            echo "  [OK] 服务就绪 (约 ${i}s)"
            break
        fi
        # 进程是否存活
        if ! kill -0 $SERVICE_PID 2>/dev/null; then
            echo "  [FAIL] 服务进程已退出"
            break
        fi
    done

    if [ $READY -eq 1 ]; then
        echo ">>> [$model] 运行测试 (test.py)"
        cd "$MODEL_DIR/test"
        if python test.py > "$LOG_DIR/${model}_test.log" 2>&1; then
            echo "  [PASS] $model"
            PASS=$((PASS+1))
        else
            echo "  [FAIL] $model 测试失败"
            FAIL=$((FAIL+1))
            FAILED_LIST="$FAILED_LIST $model"
        fi
    else
        echo "  [FAIL] $model 服务启动超时"
        FAIL=$((FAIL+1))
        FAILED_LIST="$FAILED_LIST $model"
    fi

    # 停止服务
    echo ">>> [$model] 停止服务"
    pkill -f "app.py" 2>/dev/null
    kill $SERVICE_PID 2>/dev/null
    sleep 3
    # 确认端口释放
    for ((i=1; i<=10; i++)); do
        if ! curl -s -m 1 "${BASE_URL}/health" >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done
done

echo ""
echo "============================================"
echo "测试完成 - $(date)"
echo "通过: $PASS, 失败: $FAIL"
if [ -n "$FAILED_LIST" ]; then
    echo "失败列表: $FAILED_LIST"
fi
if [ -n "$SKIP_LIST" ]; then
    echo "跳过列表: $SKIP_LIST"
fi
echo "日志目录: $LOG_DIR"
echo "============================================"
