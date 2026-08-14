#!/bin/bash
# Patch voyageai model to add config_class, keep transformers 5.15.0
cd /workspace/models/voyageai-voyage-4-nano
source venv/bin/activate
echo "=== Reinstall transformers 5.15.0 ==="
pip install "transformers==5.15.0" "tokenizers>=0.21" 2>&1 | tail -3

echo ""
echo "=== Patching model custom code ==="
MODEL_FILE="/workspace/models/voyageai-voyage-4-nano/weights/modeling_qwen3_bidirectional.py"
# Backup
cp "$MODEL_FILE" "${MODEL_FILE}.bak"

# Add Qwen3Config import and config_class attribute
python3 << 'EOF'
path = "/workspace/models/voyageai-voyage-4-nano/weights/modeling_qwen3_bidirectional.py"
with open(path) as f:
    content = f.read()

# Add Qwen3Config to imports
content = content.replace(
    "from transformers import PreTrainedModel, Qwen3Model",
    "from transformers import PreTrainedModel, Qwen3Model, Qwen3Config"
)

# Add config_class to the class
content = content.replace(
    "class Qwen3BidirectionalModel(PreTrainedModel):\n    _supports_flash_attn = True",
    "class Qwen3BidirectionalModel(PreTrainedModel):\n    config_class = Qwen3Config\n    _supports_flash_attn = True"
)

with open(path, "w") as f:
    f.write(content)
print("Patched config_class")
EOF

echo ""
echo "=== Test model loading ==="
python -c "
from transformers import AutoModel
m = AutoModel.from_pretrained('weights', trust_remote_code=True)
print('model loading OK:', type(m).__name__)
" 2>&1 | tail -5
deactivate
