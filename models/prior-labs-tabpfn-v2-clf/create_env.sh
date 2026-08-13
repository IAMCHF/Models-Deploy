#!/bin/bash
# ============================================================
# create_env.sh - 创建虚拟环境（仅内网本地执行，非 GitHub Actions）
# 模型: Prior-Labs/TabPFN-v2-clf
# 策略: 使用基础镜像默认环境
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "[env] 开始创建虚拟环境: prior-labs-tabpfn-v2-clf"
echo "=========================================="

# ------------------------------------------------------------
# 1. 换源：测试清华、阿里、中科大源速度，选择最快的一个
# ------------------------------------------------------------
echo "[env] 测试国内 PyPI 镜像源速度..."
BEST_SOURCE=""
for SOURCE in \
    "https://pypi.tuna.tsinghua.edu.cn/simple" \
    "https://mirrors.aliyun.com/pypi/simple" \
    "https://pypi.mirrors.ustc.edu.cn/simple"; do
    TIME=$(curl -s -o /dev/null -w "%{time_total}" --connect-timeout 3 "$SOURCE" 2>/dev/null || echo "999")
    echo "  $SOURCE -> ${TIME}s"
    if [ -z "$BEST_SOURCE" ] || (( $(echo "$TIME < $BEST_TIME" | bc -l 2>/dev/null || echo 0) )); then
        BEST_SOURCE="$SOURCE"
        BEST_TIME="$TIME"
    fi
done
echo "[env] 选择最快源: $BEST_SOURCE"
mkdir -p ~/.config/pip
cat > ~/.config/pip/pip.conf << EOF
[global]
index-url = $BEST_SOURCE
trusted-host = $(echo $BEST_SOURCE | sed 's|https://||; s|/simple||')
EOF

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
    echo "model_id: Prior-Labs/TabPFN-v2-clf"
    echo "folder_name: prior-labs-tabpfn-v2-clf"
    echo "python: $(python3 --version 2>&1)"
    echo "created_at: $(date)"
    echo "--- pip freeze ---"
    pip freeze
} > env_info.txt

echo "=========================================="
echo "[env] 虚拟环境创建完成: prior-labs-tabpfn-v2-clf"
echo "[env] 环境信息已写入: env_info.txt"
echo "=========================================="
