#!/bin/bash
# Fix paddlepaddle-pp-chart2table - PaddleX 3.6.0 needs fused_rms_norm_ext not in paddle 3.0.0
echo "=== Fix paddlepaddle-pp-chart2table ==="
cd /workspace/models/paddlepaddle-pp-chart2table
source venv/bin/activate
echo "Current paddlex version:"
pip show paddlex 2>/dev/null | head -3
# Try downgrading paddlex to 3.3.x which should work with paddle 3.0.0
pip install 'paddlex==3.3.0' 2>&1 | tail -5
python -c "from paddle.incubate.nn.functional import fused_rms_norm_ext; print('OK')" 2>&1 || echo "fused_rms_norm_ext still missing (expected)"
# Check if the import works now
python -c "from paddlex.inference.models.doc_vlm.modeling.paddleocr_vl._fusion_ops import *; print('fusion_ops OK')" 2>&1
deactivate

# Check paddlepaddle-pp-doclayout-plus-l sample file
echo ""
echo "=== Check paddlepaddle-pp-doclayout-plus-l sample ==="
ls -la /workspace/models/paddlepaddle-pp-doclayout-plus-l/test/
file /workspace/models/paddlepaddle-pp-doclayout-plus-l/test/sample.png 2>/dev/null || echo "no sample.png"
# Check if sample file is valid
python3 -c "
from PIL import Image
import os
f = '/workspace/models/paddlepaddle-pp-doclayout-plus-l/test/sample.png'
if os.path.exists(f):
    img = Image.open(f)
    print(f'Image: {img.size}, {img.mode}')
else:
    print('sample.png not found')
    # List what's there
    d = '/workspace/models/paddlepaddle-pp-doclayout-plus-l/test/'
    for x in os.listdir(d):
        print(f'  {x}: {os.path.getsize(os.path.join(d, x))} bytes')
" 2>&1

echo ""
echo "=== Check doclayout app.py predict more closely ==="
grep -B5 -A50 "def predict" /workspace/models/paddlepaddle-pp-doclayout-plus-l/app.py | head -60
