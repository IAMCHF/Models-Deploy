#!/bin/bash
# fix_ocrv6_onnx.sh - 卸载 ocrv6 三 venv 的 CPU onnxruntime, 回落系统层 onnxruntime-gpu
set -u
ln -sf /usr/bin/python3.10 /usr/local/bin/python3
ln -sf /usr/bin/python3.10 /usr/local/bin/python
for m in paddlepaddle-pp-ocrv6-medium-det-onnx paddlepaddle-pp-ocrv6-small-det-onnx paddlepaddle-pp-ocrv6-small-rec-onnx; do
    V=/workspace/models/$m/venv
    echo "--- $m ---"
    $V/bin/pip uninstall -y -q onnxruntime 2>&1 | tail -1
    $V/bin/python -c "
import onnxruntime as o
import onnxruntime
print('解析位置:', onnxruntime.__file__)
print('providers:', o.get_available_providers())
" 2>&1 | grep -vE "Warning"
done
echo "=== DONE ==="
