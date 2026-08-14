#!/bin/bash
# ============================================================
# create_env.sh - 创建虚拟环境（仅内网本地执行）
# 模型: PaddlePaddle/PP-DocLayout_plus-L
# 策略: PaddlePaddle 模型
# PyPI 镜像源: https://pypi.mirrors.ustc.edu.cn/simple（中科大，测速最快）
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "[env] 开始创建虚拟环境: paddlepaddle-pp-doclayout-plus-l"
echo "=========================================="

# ------------------------------------------------------------
# 1. 配置 PyPI 镜像源（中科大，已测速最快）
# ------------------------------------------------------------
mkdir -p ~/.config/pip
cat > ~/.config/pip/pip.conf << 'PIPEOF'
[global]
index-url = https://pypi.mirrors.ustc.edu.cn/simple
trusted-host = pypi.mirrors.ustc.edu.cn
PIPEOF
echo "[env] PyPI 镜像源: https://pypi.mirrors.ustc.edu.cn/simple"

# ------------------------------------------------------------
# 2. 启用缓存
# ------------------------------------------------------------
export PIP_CACHE_DIR="$(pwd)/../.pip_cache"
mkdir -p "$PIP_CACHE_DIR"

# ------------------------------------------------------------
# 3. 创建虚拟环境（python3）
# ------------------------------------------------------------
if [ -d "venv" ]; then
    echo "[env] 虚拟环境已存在，跳过创建。如需重建请先删除 venv/"
else
    echo "[env] 创建虚拟环境: python3 -m venv venv"
    python3 -m venv venv
fi

source ./venv/bin/activate
pip install --upgrade pip

# ------------------------------------------------------------
# 4. 如需覆盖基础版本，先卸载旧版本
# ------------------------------------------------------------

# ------------------------------------------------------------
# 5. 安装覆盖版本依赖（特殊模型）
# ------------------------------------------------------------
echo "[env] 安装覆盖版本依赖"
    pip install paddlepaddle-gpu==3.0.0 --index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/
    pip install paddleocr

# ------------------------------------------------------------
# 6. 安装 requirements.txt 模型特有依赖
# ------------------------------------------------------------
echo "[env] 安装 requirements.txt 模型特有依赖"
    pip install -r requirements.txt

# ------------------------------------------------------------
# 7. 记录环境到 env_info.txt
# ------------------------------------------------------------
echo "[env] 记录环境信息"
{
    echo "model_id: PaddlePaddle/PP-DocLayout_plus-L"
    echo "folder_name: paddlepaddle-pp-doclayout-plus-l"
    echo "python: $(python3 --version 2>&1)"
    echo "pip_mirror: https://pypi.mirrors.ustc.edu.cn/simple"
    echo "created_at: $(date)"
    echo "--- pip freeze ---"
    pip freeze
} > env_info.txt

echo "=========================================="
echo "[env] 虚拟环境创建完成: paddlepaddle-pp-doclayout-plus-l"
echo "[env] 环境信息已写入: env_info.txt"
echo "=========================================="
