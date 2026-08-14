#!/bin/bash
# ============================================================
# test_all_models.sh - 顺序测试所有40个模型
# 每个模型：启动 → 健康检查 → 测试predict → 停止
# ============================================================

PORT=8080
MAX_WAIT=180  # 单个模型最大等待秒数
PASS_LIST=()
FAIL_LIST=()
SKIP_LIST=()

# 获取所有模型目录
MODELS_DIR="/workspace/models"
ALL_MODELS=$(ls -d ${MODELS_DIR}/*/ 2>/dev/null | xargs -n1 basename | sort)

echo "============================================"
echo "开始全量测试 - $(date)"
echo "============================================"
echo ""

for model in $ALL_MODELS; do
    MODEL_DIR="${MODELS_DIR}/${model}"
    echo ">>> 测试: $model"

    # 检查必要文件
    if [ ! -f "${MODEL_DIR}/start.sh" ]; then
        echo "  [SKIP] 缺少 start.sh"
        SKIP_LIST+=("$model (no start.sh)")
        echo ""
        continue
    fi
    if [ ! -f "${MODEL_DIR}/test/test.py" ]; then
        echo "  [SKIP] 缺少 test/test.py"
        SKIP_LIST+=("$model (no test.py)")
        echo ""
        continue
    fi
    if [ ! -d "${MODEL_DIR}/venv" ]; then
        echo "  [SKIP] 缺少 venv"
        SKIP_LIST+=("$model (no venv)")
        echo ""
        continue
    fi

    # 清理端口
    pkill -f "app.py" 2>/dev/null
    sleep 1
    # 确保端口释放
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
    HEALTH_URL="http://localhost:${PORT}/health"
    for i in $(seq 1 $MAX_WAIT); do
        resp=$(curl -s -m 2 "$HEALTH_URL" 2>/dev/null)
        if echo "$resp" | grep -q '"ok"'; then
            health_ok=true
            echo "  健康检查通过 (${i}秒)"
            break
        fi
        # 检查进程是否还活着
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
        echo "  --- 最后20行日志 ---"
        tail -20 /tmp/${model}_test.log 2>/dev/null
        FAIL_LIST+=("$model (health check failed)")
        # 清理
        kill -9 $START_PID 2>/dev/null
        pkill -f "app.py" 2>/dev/null
        echo ""
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
        cat /tmp/${model}_test_result.log 2>/dev/null | tail -10
        FAIL_LIST+=("$model (test.py failed)")
    fi

    # 停止服务
    kill -9 $START_PID 2>/dev/null
    pkill -f "app.py" 2>/dev/null
    sleep 2

    echo ""
done

# 汇总
echo "============================================"
echo "测试完成 - $(date)"
echo "============================================"
echo ""
echo "通过: ${#PASS_LIST[@]}"
for m in "${PASS_LIST[@]}"; do echo "  ✓ $m"; done
echo ""
echo "失败: ${#FAIL_LIST[@]}"
for m in "${FAIL_LIST[@]}"; do echo "  ✗ $m"; done
echo ""
echo "跳过: ${#SKIP_LIST[@]}"
for m in "${SKIP_LIST[@]}"; do echo "  - $m"; done
echo ""
echo "总计: $(( ${#PASS_LIST[@]} + ${#FAIL_LIST[@]} + ${#SKIP_LIST[@]} ))"
