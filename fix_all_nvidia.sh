#!/bin/bash
# Fix nvidia package conflicts in ALL model venvs
# Venv nvidia packages (various CUDA versions) conflict with system torch 2.7.0+cu126

NVIDIA_PKGS="nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-nvtx-cu12 nvidia-cufile-cu12 nvidia-cuda-nvcc-cu12 nvidia-nvshmem3-cu12 nvidia-cufile-cu12"

echo "=== Fixing nvidia conflicts for all models ==="
fixed=0
skipped=0
for d in /workspace/models/*/venv; do
  model=$(basename $(dirname $d))
  has_nvidia=$(ls "$d/lib/python3.10/site-packages/" 2>/dev/null | grep -c "nvidia_.*\.dist-info")
  if [ "$has_nvidia" -gt 0 ]; then
    cd /workspace/models/$model
    source venv/bin/activate

    # Check if torch works before fix
    torch_ok=$(python -c "import torch; print('ok')" 2>/dev/null)
    if [ "$torch_ok" != "ok" ]; then
      echo "FIXING $model (torch import failed)"
      pip uninstall $NVIDIA_PKGS -y 2>&1 | grep -E "(Successfully|Not uninstalling|Can't)" | head -3
      torch_after=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
      echo "  after: torch=$torch_after"
      fixed=$((fixed + 1))
    else
      skipped=$((skipped + 1))
    fi
    deactivate
  fi
done

echo ""
echo "=== Summary: fixed=$fixed, already_ok=$skipped ==="
