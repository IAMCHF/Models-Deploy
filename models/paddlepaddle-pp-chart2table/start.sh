#!/bin/bash
# ============================================================
# start.sh - 启动 FastAPI 服务（端口 8080）
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 确保python3指向3.10（系统包安装在3.10下）
if [ -x /usr/bin/python3.10 ]; then
    ln -sf /usr/bin/python3.10 /usr/local/bin/python3
    ln -sf /usr/bin/python3.10 /usr/local/bin/python
fi

# 激活虚拟环境
if [ -d "venv" ]; then
    source ./venv/bin/activate
else
    echo "[start] 虚拟环境不存在，请先运行 ./create_env.sh"
    exit 1
fi

# 镜像站优先
export HF_ENDPOINT="https://hf-mirror.com"

echo "[start] 启动服务: $(basename $SCRIPT_DIR) (端口 8080)"
cd /tmp && python "$SCRIPT_DIR/app.py"
