#!/bin/bash
for m in $(ls /workspace/models); do
    V=/workspace/models/$m/venv
    [ -d "$V" ] || continue
    d311="$V/lib/python3.11/site-packages"
    d310="$V/lib/python3.10/site-packages"
    has311=0; [ -d "$d311" ] && has311=$(ls "$d311" 2>/dev/null | grep -c dist-info)
    n310=0; [ -d "$d310" ] && n310=$(ls "$d310" 2>/dev/null | grep -c dist-info)
    if [ "$has311" -gt 0 ] || [ ! -d "$d310" ]; then
        echo "=== $m: 3.10包数=$n310  3.11包数=$has311 ==="
        if [ "$has311" -gt 0 ]; then
            echo "  [3.11 里的包]:"
            ls "$d311" | grep dist-info | sed 's/^/    /'
        fi
    fi
done
echo "--- 完成 ---"
