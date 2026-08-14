#!/bin/bash
cd /workspace/models/jusperlee-tiger-dnr
echo "=== nvidia packages in venv ==="
ls venv/lib/python3.10/site-packages/ | grep -i nvidia
echo ""
echo "=== nvidia packages in system ==="
ls /usr/local/lib/python3.10/dist-packages/ | grep -i nvidia
echo ""
echo "=== Check system torch directly ==="
python3 -c "import torch; print('system torch:', torch.__version__)" 2>&1
echo ""
echo "=== Check cusparse/nvjitlink in system ==="
pip3 show nvidia-cusparse-cu12 2>/dev/null | head -5
pip3 show nvidia-nvjitlink-cu12 2>/dev/null | head -5
echo ""
echo "=== Check venv nvidia packages ==="
source venv/bin/activate
pip list 2>/dev/null | grep -i nvidia
deactivate
