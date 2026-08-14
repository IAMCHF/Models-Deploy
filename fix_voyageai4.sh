#!/bin/bash
cd /workspace/models/voyageai-voyage-4-nano
source venv/bin/activate

echo "=== Finding all cached module copies ==="
find /root/.cache/huggingface/modules/transformers_modules -name "modeling_qwen3_bidirectional.py" 2>/dev/null

echo ""
echo "=== Patching all copies ==="
for CACHE_FILE in $(find /root/.cache/huggingface/modules/transformers_modules -name "modeling_qwen3_bidirectional.py" 2>/dev/null); do
  echo "Patching: $CACHE_FILE"
  cp "$CACHE_FILE" "${CACHE_FILE}.bak"
  python3 << EOF
path = "$CACHE_FILE"
with open(path) as f:
    content = f.read()
content = content.replace("input_embeds=inputs_embeds", "inputs_embeds=inputs_embeds")
content = content.replace(
    "                cache_position=dummy_cache_position,\n",
    ""
)
with open(path, "w") as f:
    f.write(content)
print("  Done")
EOF
done

echo ""
echo "=== Also patch weights copy ==="
WEIGHTS_FILE="/workspace/models/voyageai-voyage-4-nano/weights/modeling_qwen3_bidirectional.py"
cp "$WEIGHTS_FILE" "${WEIGHTS_FILE}.bak"
python3 << EOF
path = "$WEIGHTS_FILE"
with open(path) as f:
    content = f.read()
content = content.replace("input_embeds=inputs_embeds", "inputs_embeds=inputs_embeds")
content = content.replace(
    "                cache_position=dummy_cache_position,\n",
    ""
)
with open(path, "w") as f:
    f.write(content)
print("  Done")
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
