#!/bin/bash
# Restore paddlex 3.6.0 for chart2table and re-apply fusion_ops patch
echo "=== Restore paddlex 3.6.0 for chart2table ==="
cd /workspace/models/paddlepaddle-pp-chart2table
source venv/bin/activate
pip install 'paddlex==3.6.0' 2>&1 | tail -3

# Re-apply fusion_ops patch
FUSION_OPS_FILE="/workspace/models/paddlepaddle-pp-chart2table/venv/lib/python3.10/site-packages/paddlex/inference/models/doc_vlm/modeling/paddleocr_vl/_fusion_ops/__init__.py"
echo ""
echo "=== Re-applying fusion_ops patch ==="
if [ -f "${FUSION_OPS_FILE}.bak" ]; then
  cp "${FUSION_OPS_FILE}.bak" "$FUSION_OPS_FILE"
  echo "Restored from backup"
fi

# Patch fused_rms_norm_ext
sed -i 's/^from paddle.incubate.nn.functional import fused_rms_norm_ext$/try:\n    from paddle.incubate.nn.functional import fused_rms_norm_ext\nexcept ImportError:\n    from paddle.incubate.nn.functional import fused_rms_norm as fused_rms_norm_ext/' "$FUSION_OPS_FILE"

# Patch cal_aux_loss
python3 << 'EOF'
import re
path = "/workspace/models/paddlepaddle-pp-chart2table/venv/lib/python3.10/site-packages/paddlex/inference/models/doc_vlm/modeling/paddleocr_vl/_fusion_ops/__init__.py"
with open(path) as f:
    content = f.read()
old = """else:
    from paddle.incubate.nn.functional import cal_aux_loss"""
new = """else:
    try:
        from paddle.incubate.nn.functional import cal_aux_loss
    except ImportError:
        def cal_aux_loss(*args, **kwargs):
            raise NotImplementedError("cal_aux_loss not available")"""
content = content.replace(old, new)
with open(path, "w") as f:
    f.write(content)
print("cal_aux_loss patched")
EOF

# Test import
echo ""
echo "=== Testing import ==="
python -c "from paddlex.inference.models.doc_vlm.modeling.paddleocr_vl._fusion_ops import fused_rms_norm_ext, cal_aux_loss; print('fusion_ops import OK')" 2>&1
deactivate
