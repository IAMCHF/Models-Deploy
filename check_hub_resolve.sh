#!/bin/bash
for m in facebook-vjepa2-vitl-fpc64-256 koreapeter-ms-eff-gcvit-deepfake-b0-ff-plus-plus koreapeter-ms-eff-gcvit-deepfake-b5-ff-plus-plus opengvlab-videomaev2-base; do
    sp=/workspace/models/$m/venv/lib/python3.10/site-packages
    echo "--- $m ---"
    ls $sp 2>/dev/null | grep -i 'huggingface' | sed 's/^/  venv3.10: /' || echo "  venv3.10: 无hub文件"
    /workspace/models/$m/venv/bin/python -c "import huggingface_hub as h; print('  解析版本:', h.__version__); print('  位置:', h.__file__)" 2>/dev/null || echo "  (导入失败)"
done
echo "=== 系统层 ==="
/usr/bin/python3.10 -c "import huggingface_hub as h; print('系统 hub:', h.__version__, h.__file__)"
