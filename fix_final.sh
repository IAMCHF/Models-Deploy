#!/bin/bash
# fix_final.sh - 收尾: 补装 small-det 的 paddleocr + 克隆 fomo_hub
set -u
# 关键: 先做 start.sh 同款软链, 保证包装进 python3.10
ln -sf /usr/bin/python3.10 /usr/local/bin/python3
ln -sf /usr/bin/python3.10 /usr/local/bin/python
mkdir -p ~/.config/pip
cat > ~/.config/pip/pip.conf << 'PIPEOF'
[global]
index-url = https://pypi.mirrors.ustc.edu.cn/simple
trusted-host: pypi.mirrors.ustc.edu.cn
PIPEOF
export PIP_CACHE_DIR=/workspace/models/.pip_cache

echo "=== 1. small-det paddleocr ==="
cd /workspace/models/paddlepaddle-pp-ocrv6-small-det-onnx
./venv/bin/pip install --quiet paddleocr==3.7.0 2>&1 | tail -2
sp=venv/lib/python3.10/site-packages
for p in paddleocr paddlex; do
    found=$(ls $sp | grep -i "^${p}-[0-9]" | head -1)
    echo "[$([ -n "$found" ] && echo OK || echo FAIL)] $p: ${found:-未落盘}"
done

echo "=== 2. fomo_hub 克隆 ==="
if [ ! -d /workspace/models/yuchenshen-fomo-0d/fomo_hub ]; then
    rm -rf /tmp/FoMo-0D
    if git clone --depth 1 https://github.com/A-Chicharito-S/FoMo-0D /tmp/FoMo-0D 2>&1 | tail -1; then
        SRC=$(find /tmp/FoMo-0D -maxdepth 3 -type d -name "fomo_hub" | head -1)
        if [ -n "$SRC" ]; then
            cp -r "$SRC" /workspace/models/yuchenshen-fomo-0d/fomo_hub
            rm -rf /workspace/models/yuchenshen-fomo-0d/fomo_hub/.git
            echo "[OK] fomo_hub <- $SRC: $(ls /workspace/models/yuchenshen-fomo-0d/fomo_hub | head -5 | tr '\n' ' ')"
        else
            echo "[FAIL] 仓库内无 fomo_hub 目录, 仓库结构:"
            find /tmp/FoMo-0D -maxdepth 2 | head -20
        fi
    else
        echo "[FAIL] git clone 失败"
    fi
else
    echo "[SKIP] 已存在"
fi
echo "=== DONE ==="
