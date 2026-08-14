#!/bin/bash
# fix_nuextract_torch.sh - 移除 nuextract venv 内错误拉入的 torch 2.13+cu130 全家桶
# 让 venv 回落使用系统层配套: torch 2.7.0+cu126 + torchvision 0.22.0+cu126
set -u
ln -sf /usr/bin/python3.10 /usr/local/bin/python3
ln -sf /usr/bin/python3.10 /usr/local/bin/python
V=/workspace/models/numind-nuextract3-fp8/venv
sp=$V/lib/python3.10/site-packages

echo "=== 卸载 venv torch 2.13 ==="
$V/bin/pip uninstall -y -q torch 2>&1 | tail -1

echo "=== 卸载 venv 内 cu130 nvidia 库与 triton(全部, 原始 venv 本就没有这些) ==="
for pkg in $($V/bin/pip list --format=freeze 2>/dev/null | grep -iE "^(nvidia|triton)" | cut -d= -f1); do
    echo "  卸载 $pkg"
    $V/bin/pip uninstall -y -q "$pkg" 2>&1 | tail -1
done

echo "=== 清理 __pycache__ 避免旧 torch 残留 ==="
find $sp -maxdepth 1 -name "torch*" -type d | head -5

echo "=== 验证: venv 解析到的 torch/torchvision ==="
$V/bin/python -c "
import torch, torchvision
print('torch:', torch.__version__, '->', torch.__file__)
print('torchvision:', torchvision.__version__, '->', torchvision.__file__)
" 2>&1 | grep -v Warning

echo "=== 验证: AutoProcessor 可导入 ==="
$V/bin/python -c "
from transformers import AutoProcessor, AutoModelForImageTextToText
import transformers, huggingface_hub, compressed_tensors
print('transformers:', transformers.__version__)
print('huggingface_hub:', huggingface_hub.__version__)
print('compressed_tensors:', compressed_tensors.__version__)
print('AutoProcessor/AutoModelForImageTextToText 导入 OK')
" 2>&1 | grep -viE "^.*warning" | tail -6

echo "=== venv 瘦身后大小 ==="
du -sh /workspace/models/numind-nuextract3-fp8/venv 2>/dev/null
echo "=== DONE ==="
