#!/bin/bash
echo "=== Checking jusperlee-tiger-dnr venv config ==="
cd /workspace/models/jusperlee-tiger-dnr
cat venv/pyvenv.cfg
echo ""
echo "=== Checking if system torch is accessible ==="
source venv/bin/activate
python -c "import torch; print('torch:', torch.__version__)" 2>&1
python -c "import torchvision; print('torchvision:', torchvision.__version__)" 2>&1
python -c "import sys; print('python paths:'); [print(p) for p in sys.path]" 2>&1
deactivate
