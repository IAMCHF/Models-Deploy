#!/bin/bash
echo "=== ocrv6 三件套 3.10 落盘 ==="
for m in paddlepaddle-pp-ocrv6-medium-det-onnx paddlepaddle-pp-ocrv6-small-det-onnx paddlepaddle-pp-ocrv6-small-rec-onnx; do
    sp=/workspace/models/$m/venv/lib/python3.10/site-packages
    pg=$(ls $sp 2>/dev/null | grep -i "^paddlepaddle_gpu-[0-9]" | head -1)
    po=$(ls $sp 2>/dev/null | grep -i "^paddleocr-[0-9]" | head -1)
    px=$(ls $sp 2>/dev/null | grep -i "^paddlex-[0-9]" | head -1)
    echo "$m:"
    echo "  paddlepaddle_gpu: ${pg:-缺}"
    echo "  paddleocr:        ${po:-缺}"
    echo "  paddlex:          ${px:-缺}"
done
echo
echo "=== fomo_hub ==="
if [ -d /workspace/models/yuchenshen-fomo-0d/fomo_hub ]; then
    echo "[OK] $(ls /workspace/models/yuchenshen-fomo-0d/fomo_hub | head -8 | tr '\n' ' ')"
else
    echo "[缺] 不存在"
fi
echo
echo "=== 3.11 污染目录现状 ==="
for m in $(ls /workspace/models); do
    d=/workspace/models/$m/venv/lib/python3.11
    [ -d "$d" ] && echo "$m: $(ls $d/site-packages 2>/dev/null | grep -c dist-info) 个包"
done
