#!/bin/bash
# 检查各 venv bin/ 下 python 链接的实际指向
for m in $(ls /workspace/models); do
    V=/workspace/models/$m/venv
    [ -d "$V/bin" ] || continue
    has311=0
    [ -d "$V/lib/python3.11" ] && has311=$(ls "$V/lib/python3.11/site-packages" 2>/dev/null | grep -c dist-info)
    echo "--- $m (3.11包数=$has311) ---"
    for f in python python3 python3.10 python3.11; do
        if [ -e "$V/bin/$f" ]; then
            echo "  $f -> $(readlink $V/bin/$f)  ver=$($V/bin/$f --version 2>&1)"
        fi
    done
done
