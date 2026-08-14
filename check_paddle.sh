#!/bin/bash
echo "=== paddlepaddle-pp-chart2table ==="
cd /workspace/models/paddlepaddle-pp-chart2table
source venv/bin/activate
echo "--- paddle version ---"
pip show paddlepaddle-gpu 2>/dev/null | head -3 || pip show paddlepaddle 2>/dev/null | head -3
echo "--- paddlex version ---"
pip show paddlex 2>/dev/null | head -3
echo "--- check fused_rms_norm_ext ---"
python -c "from paddle.incubate.nn.functional import fused_rms_norm_ext; print('OK')" 2>&1
echo "--- check available functions ---"
python -c "import paddle.incubate.nn.functional as F; print([x for x in dir(F) if 'rms' in x.lower() or 'norm' in x.lower()])" 2>&1
deactivate

echo ""
echo "=== paddlepaddle-pp-doclayout-plus-l ==="
cd /workspace/models/paddlepaddle-pp-doclayout-plus-l
echo "--- app.py predict function ---"
grep -A 30 "def predict" app.py 2>/dev/null | head -35
echo ""
echo "--- startup log full ---"
cat /tmp/paddlepaddle-pp-doclayout-plus-l_test.log 2>/dev/null | tail -40
echo ""
echo "--- check venv ---"
source venv/bin/activate
pip show paddlepaddle-gpu 2>/dev/null | head -3 || pip show paddlepaddle 2>/dev/null | head -3
pip show paddlex 2>/dev/null | head -3
deactivate
