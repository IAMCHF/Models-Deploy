#!/bin/bash
# Fix all remaining torch/torchvision conflicts
# Uninstall venv torch/torchvision to use system versions (2.7.0+cu126/0.22.0+cu126)

MODELS_TO_FIX_TORCH=(
  "autogluon-mitra-classifier"
  "docling-project-codeformulav2"
  "jusperlee-tiger-dnr"
  "prior-labs-tabpfn-v2-clf"
  "prior-labs-tabpfn-v2-reg"
  "skywork-skywork-reward-v2-qwen3-0-6b"
)

echo "=== Fixing torch conflicts ==="
for model in "${MODELS_TO_FIX_TORCH[@]}"; do
  echo "--- $model ---"
  cd /workspace/models/$model
  source venv/bin/activate

  # Check and uninstall venv torch
  if [ -d "venv/lib/python3.10/site-packages/torch" ]; then
    echo "  venv torch found, uninstalling..."
    pip uninstall torch -y 2>&1 | tail -2
  else
    echo "  no venv torch"
  fi

  # Check and uninstall venv torchvision
  if [ -d "venv/lib/python3.10/site-packages/torchvision" ]; then
    echo "  venv torchvision found, uninstalling..."
    pip uninstall torchvision -y 2>&1 | tail -2
  else
    echo "  no venv torchvision"
  fi

  # Verify
  sys_torch=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
  sys_tv=$(python -c "import torchvision; print(torchvision.__version__)" 2>/dev/null)
  echo "  after: torch=$sys_torch torchvision=$sys_tv"

  deactivate
  echo ""
done

echo "=== All torch/torchvision fixes complete ==="
