#!/bin/bash
# Patch fusion_ops to handle missing fused_rms_norm_ext
echo "=== Patching fusion_ops for chart2table ==="
FUSION_OPS_FILE="/workspace/models/paddlepaddle-pp-chart2table/venv/lib/python3.10/site-packages/paddlex/inference/models/doc_vlm/modeling/paddleocr_vl/_fusion_ops/__init__.py"

# Backup original
cp "$FUSION_OPS_FILE" "${FUSION_OPS_FILE}.bak"

# Replace the import line with try/except
sed -i 's/^from paddle.incubate.nn.functional import fused_rms_norm_ext$/try:\n    from paddle.incubate.nn.functional import fused_rms_norm_ext\nexcept ImportError:\n    from paddle.incubate.nn.functional import fused_rms_norm as fused_rms_norm_ext/' "$FUSION_OPS_FILE"

echo "Patched. Testing import..."
cd /workspace/models/paddlepaddle-pp-chart2table
source venv/bin/activate
python -c "from paddlex.inference.models.doc_vlm.modeling.paddleocr_vl._fusion_ops import fused_rms_norm_ext; print('fusion_ops import OK')" 2>&1
deactivate

# Fix doclayout-plus-l: restore paddlex 3.7.2 and fix temp file extension
echo ""
echo "=== Fix doclayout-plus-l ==="
cd /workspace/models/paddlepaddle-pp-doclayout-plus-l
source venv/bin/activate
pip install 'paddlex==3.7.2' 2>&1 | tail -3
deactivate

# Fix temp file to have .png extension
sed -i 's/tmp_path = tempfile.mktemp()/tmp_path = tempfile.mktemp(suffix=".png")/' app.py
echo "Fixed temp file extension:"
grep "tempfile" app.py
