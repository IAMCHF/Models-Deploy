#!/bin/bash
# 验证各修复包是否已正确落在 python3.10 目录
chk() {
    local model=$1 pkg=$2
    local sp=/workspace/models/$model/venv/lib/python3.10/site-packages
    local found=$(ls $sp 2>/dev/null | grep -i "^${pkg}[-_][0-9].*dist-info" | head -1)
    [ -z "$found" ] && found=$(ls $sp 2>/dev/null | grep -i "^${pkg}[-_]([0-9]).*dist-info" | head -1)
    if [ -z "$found" ]; then
        echo "[缺] $model: $pkg (3.10 目录没有)"
    else
        echo "[OK] $model: $found"
    fi
}
echo "== 1. hub 0.36.2 x4 =="
for m in facebook-vjepa2-vitl-fpc64-256 koreapeter-ms-eff-gcvit-deepfake-b0-ff-plus-plus koreapeter-ms-eff-gcvit-deepfake-b5-ff-plus-plus opengvlab-videomaev2-base; do
    chk $m huggingface_hub
done
echo "== 2. nuextract =="
chk numind-nuextract3-fp8 transformers
chk numind-nuextract3-fp8 huggingface_hub
chk numind-nuextract3-fp8 compressed_tensors
echo "== 3. ocrv6 x3 =="
for m in paddlepaddle-pp-ocrv6-medium-det-onnx paddlepaddle-pp-ocrv6-small-det-onnx paddlepaddle-pp-ocrv6-small-rec-onnx; do
    chk $m paddlepaddle_gpu
    chk $m paddleocr
done
echo "== 4. fomo_hub =="
if [ -d /workspace/models/yuchenshen-fomo-0d/fomo_hub ]; then
    echo "[OK] fomo_hub 目录存在: $(ls /workspace/models/yuchenshen-fomo-0d/fomo_hub | head -5 | tr '\n' ' ')"
else
    echo "[缺] fomo_hub 目录不存在"
fi
echo "== 5. tensorflow 装哪了(job-2b642c83) =="
for m in $(ls /workspace/models); do
    sp311=/workspace/models/$m/venv/lib/python3.11/site-packages
    [ -d "$sp311" ] && ls $sp311 | grep -qi tensorflow && echo "  $m: tensorflow 在 3.11(错)"
    sp310=/workspace/models/$m/venv/lib/python3.10/site-packages
    [ -d "$sp310" ] && ls $sp310 | grep -qi "^tensorflow-[0-9]" && echo "  $m: tensorflow 在 3.10(对)"
done
