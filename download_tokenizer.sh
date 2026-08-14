#!/bin/bash
set -e
cd /workspace/models/neoquasar-kronos-base
mkdir -p weights_tokenizer

echo "=== download tokenizer config ==="
curl -sL -m 30 \
  -H "User-Agent: Mozilla/5.0" \
  "https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base/resolve/main/config.json" \
  -o weights_tokenizer/config.json
echo "config downloaded: $(wc -c < weights_tokenizer/config.json) bytes"

echo "=== download tokenizer weights ==="
curl -sL -m 120 \
  -H "User-Agent: Mozilla/5.0" \
  "https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base/resolve/main/model.safetensors" \
  -o weights_tokenizer/model.safetensors
echo "weights downloaded: $(wc -c < weights_tokenizer/model.safetensors) bytes"

echo "=== test local tokenizer load ==="
./venv/bin/python -c "
from model import KronosTokenizer
tok = KronosTokenizer.from_pretrained('weights_tokenizer')
print('tokenizer OK:', type(tok).__name__)
" 2>&1