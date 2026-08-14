#!/bin/bash
cd /workspace/models/voyageai-voyage-4-nano
source venv/bin/activate
python << 'EOF'
import inspect
from transformers.masking_utils import create_causal_mask
print("Signature:", inspect.signature(create_causal_mask))
# Show source to understand expected args
src = inspect.getsource(create_causal_mask)
print(src[:2000])
EOF
deactivate
