#!/bin/bash
# Patch additional missing imports in fusion_ops
FUSION_OPS_FILE="/workspace/models/paddlepaddle-pp-chart2table/venv/lib/python3.10/site-packages/paddlex/inference/models/doc_vlm/modeling/paddleocr_vl/_fusion_ops/__init__.py"

echo "=== Current file content ==="
cat "$FUSION_OPS_FILE"

echo ""
echo "=== Patching cal_aux_loss import ==="
# Replace the cal_aux_loss import with try/except
sed -i '/^else:$/,/cal_aux_loss/c\else:\n    try:\n        from paddle.incubate.nn.functional import cal_aux_loss\n    except ImportError:\n        def cal_aux_loss(*args, **kwargs):\n            raise NotImplementedError("cal_aux_loss not available in this paddle version")' "$FUSION_OPS_FILE"

echo ""
echo "=== Testing import ==="
cd /workspace/models/paddlepaddle-pp-chart2table
source venv/bin/activate
python -c "from paddlex.inference.models.doc_vlm.modeling.paddleocr_vl._fusion_ops import fused_rms_norm_ext, cal_aux_loss; print('fusion_ops import OK')" 2>&1
deactivate
