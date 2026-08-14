#!/bin/bash
# check_env.sh - 核实容器内 Python 环境真实状态
echo "=== 1. /usr/local/bin python 软链 ==="
ls -la /usr/local/bin/python /usr/local/bin/python3 /usr/local/bin/python3.1* 2>/dev/null

echo
echo "=== 2. /usr/bin 下所有 python ==="
ls /usr/bin/python* 2>/dev/null

echo
echo "=== 3. 各解释器版本 ==="
for p in /usr/local/bin/python3 /usr/bin/python3 /usr/bin/python3.10 /usr/bin/python3.11 /usr/bin/python3.12; do
    [ -x "$p" ] && echo "$p -> $($p --version 2>&1)"
done

echo
echo "=== 4. which python / python3 / pip ==="
which python python3 pip pip3 2>/dev/null

echo
echo "=== 5. 系统 Python 已装关键包(3.11 site-packages) ==="
/usr/bin/python3 -c "import sys; print('sys python:', sys.version)" 2>/dev/null
for pkg in torch transformers huggingface_hub paddlepaddle paddleocr sentence_transformers; do
    /usr/bin/python3 -c "import $pkg; print('$pkg', getattr($pkg, '__version__', '?'))" 2>/dev/null || echo "$pkg: 未装(系统)"
done

echo
echo "=== 6. 若存在 python3.10, 查它的包 ==="
if [ -x /usr/bin/python3.10 ]; then
    for pkg in torch transformers huggingface_hub paddlepaddle paddleocr; do
        /usr/bin/python3.10 -c "import $pkg; print('$pkg', getattr($pkg, '__version__', '?'))" 2>/dev/null || echo "$pkg: 未装(3.10)"
    done
fi

echo
echo "=== 7. 抽查 venv: python 指向 + pip shebang + site-packages ==="
for m in numind-nuextract3-fp8 paddlepaddle-pp-ocrv6-medium-det-onnx yuchenshen-fomo-0d k-iwa-time-anchor-modernbert-32m; do
    V=/workspace/models/$m/venv
    echo "--- $m ---"
    if [ -e "$V/bin/python" ]; then
        echo "python -> $(readlink -f $V/bin/python 2>/dev/null)"
        echo "shebang: $(head -1 $V/bin/pip 2>/dev/null)"
        echo "lib dirs: $(ls -d $V/lib/python* 2>/dev/null | tr '\n' ' ')"
    else
        echo "venv 不存在"
    fi
done

echo
echo "=== 8. venv/bin/python 实际解析版本(nuextract) ==="
/workspace/models/numind-nuextract3-fp8/venv/bin/python --version 2>&1
/workspace/models/numind-nuextract3-fp8/venv/bin/python -c "import sys; print('prefix:', sys.prefix); print('base:', sys.base_prefix)" 2>&1
