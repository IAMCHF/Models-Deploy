#!/bin/bash
# Fix voyageai model custom code for transformers 5.15.0 API
cd /workspace/models/voyageai-voyage-4-nano
source venv/bin/activate

# Find the actual cached module file
CACHE_FILE=$(find /root/.cache/huggingface/modules/transformers_modules -name "modeling_qwen3_bidirectional.py" | head -1)
echo "Cached module: $CACHE_FILE"

if [ -z "$CACHE_FILE" ]; then
  echo "ERROR: cached module not found"
  exit 1
fi

# Backup
cp "$CACHE_FILE" "${CACHE_FILE}.bak"

# Fix the create_causal_mask call
python3 << EOF
path = "$CACHE_FILE"
with open(path) as f:
    content = f.read()

# Fix input_embeds -> inputs_embeds
content = content.replace("input_embeds=inputs_embeds", "inputs_embeds=inputs_embeds")

# Remove cache_position argument (deprecated in 5.15.0)
content = content.replace(
    "                cache_position=dummy_cache_position,\n",
    ""
)

with open(path, "w") as f:
    f.write(content)
print("Patched create_causal_mask call")
EOF

echo ""
echo "=== Test model forward ==="
python << 'EOF'
import torch
from transformers import AutoModel, AutoTokenizer
m = AutoModel.from_pretrained('weights', trust_remote_code=True)
tok = AutoTokenizer.from_pretrained('weights', trust_remote_code=True)
m.eval()
m = m.to('cuda' if torch.cuda.is_available() else 'cpu')
inputs = tok("这是一条测试文本", return_tensors="pt")
inputs = {k: v.to(m.device) for k, v in inputs.items()}
with torch.no_grad():
    out = m(**inputs)
print("Forward OK, output type:", type(out).__name__)
if hasattr(out, 'last_hidden_state'):
    print("last_hidden_state shape:", out.last_hidden_state.shape)
EOF
deactivate
