#!/bin/bash
# fix_venv_deps.sh - 批量修复 venv 依赖(容器内执行, 挂载项目根目录)
# 关键: 必须先把 /usr/local/bin/python 软链到 3.10(同 start.sh),
#       否则 venv/bin/pip 的 shebang 会解析到 python3.11, 包装错环境
set -u
WS=/workspace/models

if [ -x /usr/bin/python3.10 ]; then
    ln -sf /usr/bin/python3.10 /usr/local/bin/python3
    ln -sf /usr/bin/python3.10 /usr/local/bin/python
fi
echo "[env] venv python -> $(readlink -f $WS/../models/numind-nuextract3-fp8/venv/bin/python 2>/dev/null || echo ?)"

mkdir -p ~/.config/pip
cat > ~/.config/pip/pip.conf << 'PIPEOF'
[global]
index-url = https://pypi.mirrors.ustc.edu.cn/simple
trusted-host: pypi.mirrors.ustc.edu.cn
PIPEOF
export PIP_CACHE_DIR=/workspace/.pip_cache

report=""
# install <model> <desc> <pip args...>
install() {
    local model=$1 desc=$2; shift 2
    local sp="$WS/$model/venv/lib/python3.10/site-packages"
    echo ">>> [$model] $desc"
    if (cd $WS/$model && ./venv/bin/pip install --quiet "$@") 2>&1 | tail -2; then :; fi
    # 验证: 用最后一个包名(去掉版本约束)查 dist-info
    local last="${@: -1}"; local pkg=$(echo "$last" | cut -d= -f1 | tr 'A-Z_' 'a-z-')
    if ls "$sp" | grep -qi "^$(echo $pkg | tr '-' '_')-[0-9].*dist-info"; then
        report="$report
[OK] $model: $desc ($(ls $sp | grep -i "^$(echo $pkg | tr '-' '_')-[0-9]" | head -1))"
    else
        report="$report
[FAIL] $model: $desc (venv 中未找到 $pkg)"
    fi
}

# 1. hub<1.0 x4 (transformers 4.49/4.57 需要旧 hub, 覆盖系统 1.27)
for m in facebook-vjepa2-vitl-fpc64-256 \
         koreapeter-ms-eff-gcvit-deepfake-b0-ff-plus-plus \
         koreapeter-ms-eff-gcvit-deepfake-b5-ff-plus-plus \
         opengvlab-videomaev2-base; do
    install $m "huggingface_hub 0.36.2" "huggingface_hub==0.36.2"
done

# 2. nuextract: qwen3_5 架构需要 transformers 5.x + hub 1.x
install numind-nuextract3-fp8 "transformers 5.15 + hub 1.27" \
    "transformers==5.15.0" "huggingface_hub==1.27.0"

# 3. paddleocr x3 (与 docblocklayout 相同版本组合)
for m in paddlepaddle-pp-ocrv6-medium-det-onnx \
         paddlepaddle-pp-ocrv6-small-det-onnx \
         paddlepaddle-pp-ocrv6-small-rec-onnx; do
    echo ">>> [$m] paddlepaddle-gpu 3.0.0 (paddle 官方源)"
    if (cd $WS/$m && ./venv/bin/pip install --quiet paddlepaddle-gpu==3.0.0 \
        --index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/) 2>&1 | tail -2; then :; fi
    if ls $WS/$m/venv/lib/python3.10/site-packages | grep -qi "^paddlepaddle_gpu-[0-9]"; then
        install $m "paddleocr 3.7.0" "paddleocr==3.7.0"
    else
        report="$report
[FAIL] $m: paddlepaddle-gpu 未落盘"
    fi
done

# 4. fomo_hub 源码 (github 克隆到模型目录, 同 tsfm/look2hear 模式)
echo ">>> [yuchenshen-fomo-0d] clone fomo_hub from github"
if [ ! -d $WS/yuchenshen-fomo-0d/fomo_hub ]; then
    rm -rf /tmp/FoMo-0D
    if git clone --depth 1 https://github.com/A-Chicharito-S/FoMo-0D /tmp/FoMo-0D 2>&1 | tail -1; then
        SRC=$(find /tmp/FoMo-0D -maxdepth 2 -type d -name "fomo_hub" | head -1)
        if [ -n "$SRC" ]; then
            cp -r "$SRC" $WS/yuchenshen-fomo-0d/fomo_hub
            rm -rf $WS/yuchenshen-fomo-0d/fomo_hub/.git
            report="$report
[OK] yuchenshen-fomo-0d: fomo_hub <- $SRC"
        else
            report="$report
[FAIL] yuchenshen-fomo-0d: 仓库无 fomo_hub 目录: $(find /tmp/FoMo-0D -maxdepth 1 | tr '\n' ' ')"
        fi
    else
        report="$report
[FAIL] yuchenshen-fomo-0d: git clone 失败"
    fi
else
    report="$report
[SKIP] yuchenshen-fomo-0d: fomo_hub 已存在"
fi

echo "============================================"
echo "$report"
