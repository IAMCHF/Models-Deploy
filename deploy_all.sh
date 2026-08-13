#!/bin/bash
# ============================================================
# deploy_all.sh - 一键部署脚本（内网执行，支持GPU资源管理）
#
# 用法：
#   ./deploy_all.sh env      # 批量创建虚拟环境（顺序，不占GPU）
#   ./deploy_all.sh start    # 启动服务（按资源分级，小模型批量/大模型顺序）
#   ./deploy_all.sh test     # 批量运行测试（需先 start）
#   ./deploy_all.sh stop     # 停止所有服务
#   ./deploy_all.sh status   # 查看运行中的服务及GPU状态
#   ./deploy_all.sh all      # 全部执行：env → 分级部署测试（自动启停）
#   ./deploy_all.sh list     # 列出所有模型及资源分级
#
# GPU资源管理（默认12GB显存）：
#   tiny  (<0.5GB权重):  4个并行部署，测试后统一停止
#   small (0.5-2GB):     2个并行部署，测试后统一停止
#   medium(2-5GB):       顺序部署（start→test→stop）
#   large (5-9GB):       顺序部署+VRAM警告（start→test→stop）
#   ultra (>9GB):        跳过，12GB显存无法加载
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export HF_ENDPOINT="https://hf-mirror.com"
MODELS_DIR="./models"
ACTION="${1:-list}"
BASE_PORT=8080

# ============================================================
# GPU 资源配置
# ============================================================
GPU_VRAM_GB=${GPU_VRAM_GB:-12}

# 资源分级阈值（按权重目录大小，字节）
# 估算VRAM占用 ≈ 权重大小 × 1.5（含激活值和CUDA上下文）
TINY_MAX_BYTES=536870912       # 0.5GB
SMALL_MAX_BYTES=2147483648     # 2GB
MEDIUM_MAX_BYTES=5368709120    # 5GB
LARGE_MAX_BYTES=9663676416     # 9GB

TINY_BATCH=4
SMALL_BATCH=2

# 虚拟环境创建并行数（不占GPU，可并行执行）
PARALLEL_ENV=${PARALLEL_ENV:-4}

# 健康检查最大等待秒数
HEALTH_TIMEOUT=180

# ============================================================
# 工具函数
# ============================================================

get_models() {
    find "$MODELS_DIR" -maxdepth 1 -mindepth 1 -type d | sort
}

get_weight_bytes() {
    local dir="$1/weights"
    if [ ! -d "$dir" ]; then
        echo 0
        return
    fi
    du -sb "$dir" 2>/dev/null | awk '{print $1}'
}

get_tier() {
    local bytes=$1
    if [ "$bytes" -lt "$TINY_MAX_BYTES" ]; then
        echo "tiny"
    elif [ "$bytes" -lt "$SMALL_MAX_BYTES" ]; then
        echo "small"
    elif [ "$bytes" -lt "$MEDIUM_MAX_BYTES" ]; then
        echo "medium"
    elif [ "$bytes" -lt "$LARGE_MAX_BYTES" ]; then
        echo "large"
    else
        echo "ultra"
    fi
}

format_size() {
    local bytes=$1
    local gb=$((bytes / 1073741824))
    local mb=$(( (bytes % 1073741824) / 1048576 ))
    if [ "$gb" -gt 0 ]; then
        printf "%d.%02dGB" "$gb" "$mb"
    else
        printf "%dMB" "$mb"
    fi
}

# 估算VRAM占用（权重 × 1.5）
est_vram_gb() {
    local bytes=$1
    echo "scale=1; $bytes * 3 / 2 / 1073741824" | bc 2>/dev/null || \
    awk "BEGIN {printf \"%.1f\", $bytes * 1.5 / 1073741824}"
}

# 检查GPU显存
check_gpu() {
    if ! command -v nvidia-smi &>/dev/null; then
        echo "[GPU] nvidia-smi 不可用，无法检查显存"
        return 0
    fi
    local free_mb total_mb
    free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    total_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo "[GPU] 总显存: ${total_mb}MB, 空闲: ${free_mb}MB"
}

# 启动单个模型（临时修改端口）
start_model() {
    local dir="$1"
    local port="$2"
    local name
    name=$(basename "$dir")
    cd "$dir"

    sed -i "s/port=8080/port=$port/" fastapi.py
    PORT=$port nohup bash start.sh > service.log 2>&1 &
    echo $! > .service_pid
    echo "[start] $name (PID: $!, 端口: $port)"
    cd "$SCRIPT_DIR"
}

# 等待健康检查
wait_health() {
    local port="$1"
    local name="$2"
    local waited=0
    while [ "$waited" -lt "$HEALTH_TIMEOUT" ]; do
        if curl -sf "http://localhost:$port/health" 2>/dev/null | grep -q "ok"; then
            echo "[health] $name 就绪 (${waited}s)"
            return 0
        fi
        # 检查进程是否已退出
        if ! kill -0 "$(cat "$3/.service_pid" 2>/dev/null)" 2>/dev/null; then
            echo "[health] $name 进程已退出，查看 service.log"
            return 1
        fi
        sleep 3
        waited=$((waited + 3))
        [ $((waited % 15)) -eq 0 ] && echo "[health] $name 等待中... (${waited}s)"
    done
    echo "[health] $name 超时 (${HEALTH_TIMEOUT}s)"
    return 1
}

# 停止单个模型
stop_model() {
    local dir="$1"
    local port="$2"
    local name
    name=$(basename "$dir")
    cd "$dir"

    if [ -f ".service_pid" ]; then
        local pid
        pid=$(cat .service_pid)
        kill "$pid" 2>/dev/null || true
        sleep 2
        kill -9 "$pid" 2>/dev/null || true
        rm -f .service_pid
    fi

    sed -i "s/port=$port/port=8080/" fastapi.py
    echo "[stop] $name 已停止"
    cd "$SCRIPT_DIR"
}

# 测试单个模型
test_model() {
    local dir="$1"
    local port="$2"
    local name
    name=$(basename "$dir")
    cd "$dir"

    if [ -d "venv" ] && [ -f "test/test.py" ]; then
        source ./venv/bin/activate
        if SERVICE_URL="http://localhost:$port" python test/test.py; then
            deactivate 2>/dev/null || true
            cd "$SCRIPT_DIR"
            return 0
        else
            deactivate 2>/dev/null || true
            cd "$SCRIPT_DIR"
            return 1
        fi
    else
        echo "[skip] $name 无虚拟环境或测试脚本"
        cd "$SCRIPT_DIR"
        return 1
    fi
}

# 顺序部署+测试单个模型
deploy_test_sequential() {
    local dir="$1"
    local port="$2"
    local name
    name=$(basename "$dir")

    echo ">>> [$name] 启动 (端口 $port)"
    start_model "$dir" "$port"

    local test_result=1
    if wait_health "$port" "$name" "$dir"; then
        test_model "$dir" "$port"
        test_result=$?
    fi

    stop_model "$dir" "$port"
    return $test_result
}

# 批量部署+测试多个模型（并行启动，统一测试，统一停止）
deploy_test_parallel() {
    local port="$1"
    shift
    local models=("$@")
    local count=${#models[@]}
    local pass=0
    local fail=0

    # 并行启动
    for dir in "${models[@]}"; do
        start_model "$dir" "$port"
        port=$((port + 1))
        sleep 2
    done

    # 等待并测试
    port=$1
    for dir in "${models[@]}"; do
        local name
        name=$(basename "$dir")
        if wait_health "$port" "$name" "$dir"; then
            if test_model "$dir" "$port"; then
                pass=$((pass + 1))
            else
                fail=$((fail + 1))
                echo "[失败] $name"
            fi
        else
            fail=$((fail + 1))
            echo "[失败] $name (健康检查未通过)"
        fi
        port=$((port + 1))
    done

    # 统一停止
    port=$1
    for dir in "${models[@]}"; do
        stop_model "$dir" "$port"
        port=$((port + 1))
    done

    echo "  批次结果: 通过 $pass / 失败 $fail"
    return $fail
}

# ============================================================
# 主命令
# ============================================================

list_models() {
    echo "=========================================="
    echo "模型列表及资源分级（目标GPU: ${GPU_VRAM_GB}GB）"
    echo "=========================================="
    local i=1
    local tier_summary=""
    for dir in $(get_models); do
        local name
        name=$(basename "$dir")
        local bytes
        bytes=$(get_weight_bytes "$dir")
        local tier
        tier=$(get_tier "$bytes")
        local size
        size=$(format_size "$bytes")
        local vram
        vram=$(est_vram_gb "$bytes")
        local flag=""
        [ -f "$dir/.large_model" ] && flag=" *"
        printf "  %2d. [%-6s] %-45s %8s (VRAM~%sGB)%s\n" "$i" "$tier" "$name" "$size" "$vram" "$flag"
        tier_summary="$tier_summary$tier\n"
        i=$((i + 1))
    done
    echo "=========================================="
    echo "分级说明: tiny=4并行 small=2并行 medium=顺序 large=顺序+警告 ultra=跳过"
}

batch_env() {
    echo "=========================================="
    echo "批量创建虚拟环境（${PARALLEL_ENV}个并行，不占用GPU）"
    echo "=========================================="
    local tmpdir
    tmpdir=$(mktemp -d)
    local models=()
    for dir in $(get_models); do
        models+=("$dir")
    done
    local total=${#models[@]}

    for dir in "${models[@]}"; do
        local name
        name=$(basename "$dir")

        # 等待空位
        while [ "$(jobs -rp 2>/dev/null | wc -l)" -ge "$PARALLEL_ENV" ]; do
            sleep 2
        done

        echo ">>> [并行] 创建环境: $name"
        (
            cd "$dir"
            if [ -f "create_env.sh" ]; then
                if bash create_env.sh > create_env.log 2>&1; then
                    echo "OK" > "$tmpdir/$name"
                else
                    echo "FAIL" > "$tmpdir/$name"
                fi
            else
                echo "SKIP" > "$tmpdir/$name"
            fi
        ) &
    done

    # 等待所有完成
    wait

    # 统计结果
    local ok=0 fail=0 skip=0
    for dir in "${models[@]}"; do
        local name
        name=$(basename "$dir")
        local result
        result=$(cat "$tmpdir/$name" 2>/dev/null || echo "UNKNOWN")
        case "$result" in
            OK)      ok=$((ok + 1)) ;;
            FAIL)    fail=$((fail + 1)); echo "  [FAIL] $name (查看 models/$name/create_env.log)" ;;
            SKIP)    skip=$((skip + 1)) ;;
            *)       fail=$((fail + 1)); echo "  [UNKNOWN] $name" ;;
        esac
    done
    rm -rf "$tmpdir"

    echo ""
    echo "=========================================="
    echo "环境创建完成: 成功 $ok / 失败 $fail / 跳过 $skip (共 $total)"
    echo "=========================================="
}

batch_start() {
    check_gpu
    echo "=========================================="
    echo "启动服务（按资源分级）"
    echo "=========================================="
    local port=$BASE_PORT

    # 分类模型
    local tiny_models=()
    local small_models=()
    local medium_models=()
    local large_models=()
    local ultra_models=()

    for dir in $(get_models); do
        local bytes
        bytes=$(get_weight_bytes "$dir")
        local tier
        tier=$(get_tier "$bytes")
        case "$tier" in
            tiny)   tiny_models+=("$dir") ;;
            small)  small_models+=("$dir") ;;
            medium) medium_models+=("$dir") ;;
            large)  large_models+=("$dir") ;;
            ultra)  ultra_models+=("$dir") ;;
        esac
    done

    # ultra: 跳过
    for dir in "${ultra_models[@]}"; do
        local name
        name=$(basename "$dir")
        echo "[跳过] $name - 超过${GPU_VRAM_GB}GB显存限制"
    done

    # large: 顺序启动
    for dir in "${large_models[@]}"; do
        local name
        name=$(basename "$dir")
        local bytes
        bytes=$(get_weight_bytes "$dir")
        local vram
        vram=$(est_vram_gb "$bytes")
        echo ""
        echo "[large] $name (VRAM ~${vram}GB)"
        echo "  ⚠ 此模型可能占用大量显存，建议单独测试"
        start_model "$dir" "$port"
        echo "  等待启动..."
        wait_health "$port" "$name" "$dir" || echo "  [警告] $name 启动失败"
        port=$((port + 1))
    done

    # medium: 顺序启动
    for dir in "${medium_models[@]}"; do
        start_model "$dir" "$port"
        wait_health "$port" "$(basename "$dir")" "$dir" || echo "  [警告] 启动失败"
        port=$((port + 1))
        sleep 2
    done

    # small: 2个并行
    local batch=()
    for dir in "${small_models[@]}"; do
        batch+=("$dir")
        if [ ${#batch[@]} -eq "$SMALL_BATCH" ]; then
            for b in "${batch[@]}"; do
                start_model "$b" "$port"
                port=$((port + 1))
                sleep 2
            done
            batch=()
        fi
    done
    for b in "${batch[@]}"; do
        start_model "$b" "$port"
        port=$((port + 1))
        sleep 2
    done

    # tiny: 4个并行
    batch=()
    for dir in "${tiny_models[@]}"; do
        batch+=("$dir")
        if [ ${#batch[@]} -eq "$TINY_BATCH" ]; then
            for b in "${batch[@]}"; do
                start_model "$b" "$port"
                port=$((port + 1))
                sleep 1
            done
            batch=()
        fi
    done
    for b in "${batch[@]}"; do
        start_model "$b" "$port"
        port=$((port + 1))
        sleep 1
    done

    echo ""
    echo "服务已启动，端口范围: $BASE_PORT - $((port - 1))"
    echo "查看日志: tail -f models/*/service.log"
    echo "停止服务: ./deploy_all.sh stop"
}

batch_test() {
    echo "=========================================="
    echo "批量运行测试"
    echo "=========================================="
    local pass=0
    local fail=0
    local port=$BASE_PORT
    for dir in $(get_models); do
        local name
        name=$(basename "$dir")
        echo ""
        echo ">>> 测试: $name (端口 $port)"
        cd "$dir"
        if [ -d "venv" ] && [ -f "test/test.py" ]; then
            source ./venv/bin/activate
            if SERVICE_URL="http://localhost:$port" python test/test.py; then
                pass=$((pass + 1))
            else
                fail=$((fail + 1))
                echo "[失败] $name"
            fi
            deactivate 2>/dev/null || true
        else
            echo "[跳过] $name 无虚拟环境或测试脚本"
            fail=$((fail + 1))
        fi
        cd "$SCRIPT_DIR"
        port=$((port + 1))
    done
    echo ""
    echo "=========================================="
    echo "测试结果: 通过 $pass / 失败 $fail"
    echo "=========================================="
}

batch_stop() {
    echo "=========================================="
    echo "停止所有服务"
    echo "=========================================="
    local stopped=0
    for dir in $(get_models); do
        if [ -f "$dir/.service_pid" ]; then
            local name
            name=$(basename "$dir")
            local pid
            pid=$(cat "$dir/.service_pid")
            kill "$pid" 2>/dev/null || true
            sleep 1
            kill -9 "$pid" 2>/dev/null || true
            rm -f "$dir/.service_pid"
            # 恢复端口
            sed -i "s/port=[0-9]*/port=8080/" "$dir/fastapi.py"
            echo "[stop] $name (PID: $pid)"
            stopped=$((stopped + 1))
        fi
    done
    echo "已停止 $stopped 个服务"
}

show_status() {
    echo "=========================================="
    echo "服务状态"
    echo "=========================================="
    check_gpu
    echo ""
    local running=0
    local port=$BASE_PORT
    for dir in $(get_models); do
        local name
        name=$(basename "$dir")
        if [ -f "$dir/.service_pid" ]; then
            local pid
            pid=$(cat "$dir/.service_pid")
            if kill -0 "$pid" 2>/dev/null; then
                local health="?"
                curl -sf "http://localhost:$port/health" 2>/dev/null | grep -q "ok" && health="OK"
                printf "  [运行] %-45s PID:%-8s 端口:%d 健康度:%s\n" "$name" "$pid" "$port" "$health"
                running=$((running + 1))
            fi
        fi
        port=$((port + 1))
    done
    [ "$running" -eq 0 ] && echo "  无运行中的服务"
    echo ""
    echo "运行中: $running 个"
}

deploy_all() {
    echo "=========================================="
    echo "一键部署（env → 分级部署测试）"
    echo "目标GPU: ${GPU_VRAM_GB}GB"
    echo "=========================================="

    # 1. 创建环境
    batch_env

    # 2. 分类模型
    check_gpu
    local tiny_models=()
    local small_models=()
    local medium_models=()
    local large_models=()
    local ultra_models=()

    for dir in $(get_models); do
        local bytes
        bytes=$(get_weight_bytes "$dir")
        local tier
        tier=$(get_tier "$bytes")
        case "$tier" in
            tiny)   tiny_models+=("$dir") ;;
            small)  small_models+=("$dir") ;;
            medium) medium_models+=("$dir") ;;
            large)  large_models+=("$dir") ;;
            ultra)  ultra_models+=("$dir") ;;
        esac
    done

    local total_pass=0
    local total_fail=0
    local total_skip=0

    # 3. Ultra large: 跳过
    if [ ${#ultra_models[@]} -gt 0 ]; then
        echo ""
        echo "=========================================="
        echo "[ultra] 超大模型跳过（${GPU_VRAM_GB}GB显存无法加载）"
        echo "=========================================="
        for dir in "${ultra_models[@]}"; do
            local name
            name=$(basename "$dir")
            local bytes
            bytes=$(get_weight_bytes "$dir")
            local size
            size=$(format_size "$bytes")
            local vram
            vram=$(est_vram_gb "$bytes")
            echo "  [跳过] $name (${size}, VRAM~${vram}GB)"
            total_skip=$((total_skip + 1))
        done
    fi

    # 4. Large: 顺序部署（含VRAM警告）
    if [ ${#large_models[@]} -gt 0 ]; then
        echo ""
        echo "=========================================="
        echo "[large] 大模型顺序部署（5-9GB权重，单模型VRAM 7-14GB）"
        echo "=========================================="
    fi
    for dir in "${large_models[@]}"; do
        local name
        name=$(basename "$dir")
        local bytes
        bytes=$(get_weight_bytes "$dir")
        local vram
        vram=$(est_vram_gb "$bytes")
        echo ""
        echo ">>> [large] $name (VRAM ~${vram}GB)"
        echo "  ⚠ 预计占用大量显存，如OOM请降低batch_size或使用CPU"
        if deploy_test_sequential "$dir" "$BASE_PORT"; then
            total_pass=$((total_pass + 1))
        else
            total_fail=$((total_fail + 1))
        fi
    done

    # 5. Medium: 顺序部署
    if [ ${#medium_models[@]} -gt 0 ]; then
        echo ""
        echo "=========================================="
        echo "[medium] 中等模型顺序部署（2-5GB权重）"
        echo "=========================================="
    fi
    for dir in "${medium_models[@]}"; do
        echo ""
        if deploy_test_sequential "$dir" "$BASE_PORT"; then
            total_pass=$((total_pass + 1))
        else
            total_fail=$((total_fail + 1))
        fi
    done

    # 6. Small: 2个并行
    if [ ${#small_models[@]} -gt 0 ]; then
        echo ""
        echo "=========================================="
        echo "[small] 小模型批量部署（0.5-2GB权重，${SMALL_BATCH}个并行）"
        echo "=========================================="
    fi
    local batch=()
    for dir in "${small_models[@]}"; do
        batch+=("$dir")
        if [ ${#batch[@]} -eq "$SMALL_BATCH" ]; then
            echo ""
            echo "--- 批次 (${#batch[@]}个模型) ---"
            if deploy_test_parallel "$BASE_PORT" "${batch[@]}"; then
                total_pass=$((total_pass + ${#batch[@]}))
            else
                local failed=$?
                total_pass=$((total_pass + ${#batch[@]} - failed))
                total_fail=$((total_fail + failed))
            fi
            batch=()
        fi
    done
    if [ ${#batch[@]} -gt 0 ]; then
        echo ""
        echo "--- 批次 (${#batch[@]}个模型) ---"
        if deploy_test_parallel "$BASE_PORT" "${batch[@]}"; then
            total_pass=$((total_pass + ${#batch[@]}))
        else
            local failed=$?
            total_pass=$((total_pass + ${#batch[@]} - failed))
            total_fail=$((total_fail + failed))
        fi
    fi

    # 7. Tiny: 4个并行
    if [ ${#tiny_models[@]} -gt 0 ]; then
        echo ""
        echo "=========================================="
        echo "[tiny] 微型模型批量部署（<0.5GB权重，${TINY_BATCH}个并行）"
        echo "=========================================="
    fi
    batch=()
    for dir in "${tiny_models[@]}"; do
        batch+=("$dir")
        if [ ${#batch[@]} -eq "$TINY_BATCH" ]; then
            echo ""
            echo "--- 批次 (${#batch[@]}个模型) ---"
            if deploy_test_parallel "$BASE_PORT" "${batch[@]}"; then
                total_pass=$((total_pass + ${#batch[@]}))
            else
                local failed=$?
                total_pass=$((total_pass + ${#batch[@]} - failed))
                total_fail=$((total_fail + failed))
            fi
            batch=()
        fi
    done
    if [ ${#batch[@]} -gt 0 ]; then
        echo ""
        echo "--- 批次 (${#batch[@]}个模型) ---"
        if deploy_test_parallel "$BASE_PORT" "${batch[@]}"; then
            total_pass=$((total_pass + ${#batch[@]}))
        else
            local failed=$?
            total_pass=$((total_pass + ${#batch[@]} - failed))
            total_fail=$((total_fail + failed))
        fi
    fi

    # 8. 汇总
    echo ""
    echo "=========================================="
    echo "部署测试完成"
    echo "  通过: $total_pass"
    echo "  失败: $total_fail"
    echo "  跳过: $total_skip (超大模型)"
    echo "=========================================="
    if [ "$total_fail" -gt 0 ]; then
        echo "失败模型请查看对应 models/*/service.log"
    fi
}

# ============================================================
# 入口
# ============================================================
case "$ACTION" in
    list)   list_models ;;
    env)    batch_env ;;
    start)  batch_start ;;
    test)   batch_test ;;
    stop)   batch_stop ;;
    status) show_status ;;
    all)    deploy_all ;;
    *)      echo "用法: $0 {list|env|start|test|stop|status|all}"; exit 1 ;;
esac
