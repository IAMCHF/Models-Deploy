#!/bin/bash
# ============================================================
# create_env.sh - 创建虚拟环境（仅内网本地执行）
# 模型: mldi-lab/Kairos_23m
# 策略: 使用基础镜像默认环境
# PyPI 镜像源: https://pypi.mirrors.ustc.edu.cn/simple（中科大，测速最快）
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "[env] 开始创建虚拟环境: mldi-lab-kairos-23m"
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
    echo "[env] 创建虚拟环境: python3 -m venv --system-site-packages venv"
    python3 -m venv --system-site-packages venv
fi

source ./venv/bin/activate
pip install --upgrade pip

# ------------------------------------------------------------
# 4. 如需覆盖基础版本，先卸载旧版本
# ------------------------------------------------------------

# ------------------------------------------------------------
# 5. 安装覆盖版本依赖（特殊模型）
# ------------------------------------------------------------
echo "[env] 升级 transformers 至 >=4.56（Kairos 要求）"
    pip install "transformers>=4.56,<5.0"

# ------------------------------------------------------------
# 6. 克隆 Kairos 仓库并复制 tsfm 包
# ------------------------------------------------------------
echo "[env] 克隆 Kairos 仓库并复制 tsfm 包"
if [ ! -d "tsfm" ]; then
    git clone https://github.com/foundation-model-research/Kairos.git /tmp/kairos_repo
    cp -r /tmp/kairos_repo/tsfm "$SCRIPT_DIR/"
    rm -rf /tmp/kairos_repo
    echo "[env] tsfm 包已复制到 $SCRIPT_DIR/tsfm"
else
    echo "[env] tsfm 目录已存在，跳过克隆"
fi

# ------------------------------------------------------------
# 7. 安装 Kairos 推理所需额外依赖
# ------------------------------------------------------------
echo "[env] 安装 Kairos 推理所需额外依赖"
    pip install \
        "einops>=0.8,<0.9" "jaxtyping>=0.3,<0.4" \
        "matplotlib>=3.10,<4.0"

# ------------------------------------------------------------
# 8. 记录环境到 env_info.txt
# ------------------------------------------------------------
echo "[env] 记录环境信息"
{
    echo "model_id: mldi-lab/Kairos_23m"
    echo "folder_name: mldi-lab-kairos-23m"
    echo "python: $(python3 --version 2>&1)"
    echo "pip_mirror: https://pypi.mirrors.ustc.edu.cn/simple"
    echo "created_at: $(date)"
    echo "--- pip freeze ---"
    pip freeze
} > env_info.txt

echo "=========================================="
echo "[env] 虚拟环境创建完成: mldi-lab-kairos-23m"
echo "[env] 环境信息已写入: env_info.txt"
echo "=========================================="
