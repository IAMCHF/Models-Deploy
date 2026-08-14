#!/bin/bash
# Check if system has paddlepaddle
echo "=== Check system paddlepaddle ==="
python3 -c "import paddle; print('system paddle:', paddle.__version__)" 2>&1

# Try upgrading paddlepaddle-gpu in chart2table venv to get fused_rms_norm_ext
echo ""
echo "=== Try upgrading paddlepaddle-gpu for chart2table ==="
cd /workspace/models/paddlepaddle-pp-chart2table
source venv/bin/activate
# Check if newer paddlepaddle-gpu has fused_rms_norm_ext
pip install 'paddlepaddle-gpu>=3.1.0' 2>&1 | tail -8
python -c "from paddle.incubate.nn.functional import fused_rms_norm_ext; print('fused_rms_norm_ext OK')" 2>&1
deactivate

# Fix doclayout-plus-l: add .png extension to temp file
echo ""
echo "=== Fix doclayout-plus-l temp file extension ==="
cd /workspace/models/paddlepaddle-pp-doclayout-plus-l
# Check current tempfile usage
grep "tempfile" app.py
echo ""
# Also try downgrading paddlex to 3.3.0 for this model
source venv/bin/activate
pip install 'paddlex==3.3.0' 2>&1 | tail -5
deactivate
