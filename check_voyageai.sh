#!/bin/bash
echo "=== System transformers check ==="
python3 -c "import transformers; print('system tf:', transformers.__version__)"
python3 -c "from transformers.masking_utils import create_causal_mask; print('system has create_causal_mask')" 2>&1 | tail -1
ls /usr/local/lib/python3.10/dist-packages/transformers/masking_utils.py 2>/dev/null && echo "masking_utils file exists" || echo "no masking_utils file"

echo ""
echo "=== voyageai venv transformers check ==="
cd /workspace/models/voyageai-voyage-4-nano
source venv/bin/activate
python -c "import transformers; print('venv tf:', transformers.__version__)"
python -c "from transformers.masking_utils import create_causal_mask; print('venv has create_causal_mask')" 2>&1 | tail -1
python -c "from transformers import Qwen3Config; print('Qwen3Config OK')" 2>&1 | tail -1
deactivate
