#!/bin/bash
# ============================================================
# deploy_all.sh - 一键部署脚本（内网执行）
# 功能：批量创建虚拟环境 + 启动服务 + 运行测试
# 用法：
#   ./deploy_all.sh env      # 批量创建虚拟环境
#   ./deploy_all.sh start    # 批量启动服务
#   ./deploy_all.sh test     # 批量运行测试
#   ./deploy_all.sh all      # 全部执行（env -> start -> test）
#   ./deploy_all.sh list     # 列出所有模型
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 镜像站优先
export HF_ENDPOINT="https://hf-mirror.com"

MODELS_DIR="./models"
ACTION="${1:-list}"

# 获取所有模型目录
get_models() {
    find "$MODELS_DIR" -maxdepth 1 -mindepth 1 -type d | sort
}

list_models() {
    echo "=========================================="
    echo "模型列表（共 $(get_models | wc -l) 个）"
    echo "=========================================="
    local i=1
    for dir in $(get_models); do
        local name=$(basename "$dir")
        local large=""
        [ -f "$dir/.large_model" ] && large=" [大模型>=10GB]"
        echo "  $i. $name$large"
        i=$((i + 1))
    done
}

batch_env() {
    echo "=========================================="
    echo "批量创建虚拟环境"
    echo "=========================================="
    for dir in $(get_models); do
        local name=$(basename "$dir")
        echo ""
        echo ">>> 创建环境: $name"
        cd "$dir"
        if [ -f "create_env.sh" ]; then
            bash create_env.sh || echo "[警告] $name 环境创建失败，跳过"
        else
            echo "[跳过] $name 无 create_env.sh"
        fi
        cd "$SCRIPT_DIR"
    done
}

batch_start() {
    echo "=========================================="
    echo "批量启动服务（每个模型后台启动，端口从 8080 递增）"
    echo "=========================================="
    local port=8080
    for dir in $(get_models); do
        local name=$(basename "$dir")
        echo ">>> 启动: $name (端口 $port)"
        cd "$dir"
        PORT=$port nohup bash start.sh > service.log 2>&1 &
        echo "  PID: $! -> $dir/service.log"
        cd "$SCRIPT_DIR"
        port=$((port + 1))
        sleep 2
    done
    echo ""
    echo "所有服务已后台启动，查看日志: tail -f models/*/service.log"
}

batch_test() {
    echo "=========================================="
    echo "批量运行测试"
    echo "=========================================="
    local pass=0
    local fail=0
    for dir in $(get_models); do
        local name=$(basename "$dir")
        echo ""
        echo ">>> 测试: $name"
        cd "$dir"
        if [ -d "venv" ] && [ -f "test/test.py" ]; then
            source ./venv/bin/activate
            python test/test.py && pass=$((pass + 1)) || { fail=$((fail + 1)); echo "[失败] $name"; }
            deactivate 2>/dev/null || true
        else
            echo "[跳过] $name 无虚拟环境或测试脚本"
        fi
        cd "$SCRIPT_DIR"
    done
    echo ""
    echo "=========================================="
    echo "测试结果: 通过 $pass / 失败 $fail"
    echo "=========================================="
}

case "$ACTION" in
    list)  list_models ;;
    env)   batch_env ;;
    start) batch_start ;;
    test)  batch_test ;;
    all)   batch_env; batch_start; batch_test ;;
    *)     echo "用法: $0 {list|env|start|test|all}"; exit 1 ;;
esac
