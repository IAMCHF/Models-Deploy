#!/bin/bash
# Fix nvidia package conflicts in venvs
# The venv has old CUDA 12.4 nvidia packages that conflict with system CUDA 12.6

echo "=== Checking all models for nvidia package conflicts ==="
for d in /workspace/models/*/venv; do
  model=$(basename $(dirname $d))
  nvidia_dir="$d/lib/python3.10/site-packages/nvidia"
  if [ -d "$nvidia_dir" ]; then
    # Check if there are dist-info files indicating venv-installed nvidia packages
    count=$(ls "$d/lib/python3.10/site-packages/" | grep -c "nvidia_.*\.dist-info" 2>/dev/null)
    if [ "$count" -gt 0 ]; then
      echo "$model: has $count venv nvidia packages"
    fi
  fi
done

echo ""
echo "=== Fixing jusperlee-tiger-dnr ==="
cd /workspace/models/jusperlee-tiger-dnr
source venv/bin/activate
pip uninstall nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-nvtx-cu12 nvidia-cufile-cu12 -y 2>&1 | tail -5

echo ""
echo "Verify torch works:"
python -c "import torch; print('torch:', torch.__version__)" 2>&1
python -c "import torchvision; print('torchvision:', torchvision.__version__)" 2>&1
deactivate
