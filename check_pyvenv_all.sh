#!/bin/bash
# 汇总: 每个模型的 pyvenv.cfg 版本 + start.sh 是否有软链逻辑 + lib 目录
echo "模型名 | venv版本 | lib目录 | start.sh软链目标"
for m in $(ls /workspace/models); do
    V=/workspace/models/$m/venv
    [ -d "$V" ] || continue
    ver=$(grep '^version' $V/pyvenv.cfg 2>/dev/null | awk '{print $3}')
    home=$(grep '^home' $V/pyvenv.cfg 2>/dev/null | awk '{print $3}')
    libs=$(ls $V/lib 2>/dev/null | grep python | tr '\n' ',')
    ln_target=""
    if grep -q 'python3.10' $m/start.sh 2>/dev/null; then ln_target="3.10";
    elif grep -q 'python3.11' $m/start.sh 2>/dev/null; then ln_target="3.11";
    elif grep -q 'python3.12' $m/start.sh 2>/dev/null; then ln_target="3.12";
    fi
    echo "$m | $ver | $libs | $ln_target | home=$home"
done
