#!/bin/bash
ln -sf /usr/bin/python3.10 /usr/local/bin/python3
ln -sf /usr/bin/python3.10 /usr/local/bin/python
echo "=== 特殊依赖 venv 内可导入性验证 ==="
declare -A deps=(
    [dleemiller-finecat-nli-l]="sentence_transformers"
    [mongodb-mdbr-leaf-ir]="sentence_transformers"
    [k-iwa-time-anchor-modernbert-32m]="time_anchor"
    [yuchenshen-fomo-0d]="fomo_hub"
    [voyageai-voyage-4-nano]="voyageai"
)
for m in "${!deps[@]}"; do
    p=${deps[$m]}
    v=/workspace/models/$m/venv/bin/python
    if [ ! -e "$v" ]; then echo "[SKIP] $m: venv 不存在"; continue; fi
    loc=$($v -c "import $p, os; print(os.path.dirname($p.__file__))" 2>/dev/null)
    ver=$($v -c "import $p; print(getattr($p,'__version__','?'))" 2>/dev/null)
    if [ -n "$loc" ]; then
        tag="venv"
        case "$loc" in *venv*) tag="venv";; *) tag="系统";; esac
        echo "[OK] $m: $p $ver ($tag)"
    else
        err=$($v -c "import $p" 2>&1 | tail -1)
        echo "[FAIL] $m: $p 不可导入 ($err)"
    fi
done
