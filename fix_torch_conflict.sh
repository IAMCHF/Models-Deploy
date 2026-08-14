#!/bin/bash
# 修复 venv torch 2.13.0+cu130 与系统 torchvision 0.22.0+cu126 的冲突
# 解决方案: 卸载 venv 中的 torch, 回退到系统 torch 2.7.0+cu126

PIP_INDEX="https://pypi.mirrors.stc.edu.cn/simple"
PIP_TRUSTED="pypi.mirrors.ustc.edu.cn"
export PIP_CACHE_DIR="/workspace/models/.pip_cache"

MODELS_TO_FIX=(
  "alibaba-nlp-gte-modernbert-base"
  "koreapeter-ms-eff-gcvit-deepfake-b0-ff-plus-plus"
  "koreapeter-ms-eff-gcvit-deepfake-b5-ff-plus-plus"
  "numind-nuextract3-fp8"
  "opengvlab-videomaev2-base"
  "skywork-skywork-reward-v2-qwen3-0-6b"
)

for model in "${MODELS_TO_FIX[@]}"; do
  echo "=== $model ==="
  cd /workspace/models/$model
  source venv/bin/activate

  # 检查 venv torch 版本
  venv_torch=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
  echo "  venv torch: $venv_torch"

  if [ -d "venv/lib/python3.10/site-packages/torch" ]; then
    echo "  卸载 venv torch..."
    pip uninstall torch -y 2>&1 | tail -2
  fi

  # 同时卸载 venv torchvision (如果有的话)
  if [ -d "venv/lib/python3.10/site-packages/torchvision" ]; then
    echo "  卸载 venv torchvision..."
    pip uninstall torchvision -y 2>&1 | tail -2
  fi

  # 验证回退到系统版本
  sys_torch=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
  sys_tv=$(python -c "import torchvision; print(torchvision.__version__)" 2>/dev/null)
  echo "  after: torch=$sys_torch torchvision=$sys_tv"
  echo ""
  deactivate
done

echo "=== 修复完成 ==="
