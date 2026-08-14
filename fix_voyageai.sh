#!/bin/bash
cd /workspace/models/voyageai-voyage-4-nano
source venv/bin/activate
echo "=== Installing transformers 4.51.3 ==="
pip install "transformers==4.51.3" "tokenizers>=0.21" 2>&1 | tail -5
echo ""
echo "=== Check masking_utils ==="
python -c "from transformers.masking_utils import create_causal_mask; print('masking_utils OK')" 2>&1
echo ""
echo "=== Check model loading ==="
python -c "
from transformers import AutoModel
m = AutoModel.from_pretrained('weights', trust_remote_code=True)
print('model loading OK:', type(m).__name__)
" 2>&1 | tail -5
deactivate
