#!/bin/bash
# cleanup_311.sh - 删除所有 venv 中的 python3.11 污染目录
# 依据: 39 个 pyvenv.cfg 无一是 3.11(37x3.10 + 2x3.12), 3.10 解释器不读 python3.11 目录
echo "=== 清理前确认无安装进程 ==="
if pgrep -f "pip install" > /dev/null 2>&1; then
    echo "[中止] 仍有 pip 在运行"; exit 1
fi
removed=0
for m in $(ls /workspace/models); do
    d=/workspace/models/$m/venv/lib/python3.11
    if [ -d "$d" ]; then
        n=$(ls $d/site-packages 2>/dev/null | grep -c dist-info)
        rm -rf "$d"
        echo "[清理] $m: python3.11 目录($n 个包)已删除"
        removed=$((removed+1))
    fi
done
echo "共清理 $removed 个 venv"
echo "=== 复核: 全部 venv 只剩合法 lib 目录 ==="
for m in $(ls /workspace/models); do
    libs=$(ls /workspace/models/$m/venv/lib 2>/dev/null | grep python | tr '\n' ',')
    ver=$(grep '^version' /workspace/models/$m/venv/pyvenv.cfg 2>/dev/null | awk '{print $3}')
    echo "$m: venv=$ver lib=$libs"
done
