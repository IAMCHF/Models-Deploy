#!/bin/bash
# Fix 1: jusperlee-tiger-dnr - uninstall venv torchaudio (incompatible with system torch)
echo "=== Fix 1: jusperlee-tiger-dnr ==="
cd /workspace/models/jusperlee-tiger-dnr
source venv/bin/activate
pip uninstall torchaudio -y 2>&1 | tail -2
# Check if system has torchaudio
python -c "import torchaudio; print('torchaudio:', torchaudio.__version__)" 2>&1
deactivate

# Fix 2: facebook-vjepa2-vitl-fpc64-256 - install torchcodec
echo ""
echo "=== Fix 2: facebook-vjepa2-vitl-fpc64-256 ==="
cd /workspace/models/facebook-vjepa2-vitl-fpc64-256
source venv/bin/activate
pip install torchcodec 2>&1 | tail -5
python -c "from torchcodec.decoders import VideoDecoder; print('torchcodec OK')" 2>&1
deactivate

# Fix 3: voyageai-voyage-4-nano - needs transformers 5.x for masking_utils
echo ""
echo "=== Fix 3: voyageai-voyage-4-nano ==="
cd /workspace/models/voyageai-voyage-4-nano
source venv/bin/activate
# Need transformers 5.x which has masking_utils, plus compatible tokenizers
pip install 'transformers>=5.0' 'tokenizers>=0.21' 2>&1 | tail -5
python -c "from transformers.masking_utils import create_causal_mask; print('masking_utils OK')" 2>&1
deactivate

# Fix 4: koreapeter GCViT models - need older transformers without _initialize_weights
echo ""
echo "=== Fix 4: koreapeter GCViT models ==="
for model in koreapeter-ms-eff-gcvit-deepfake-b0-ff-plus-plus koreapeter-ms-eff-gcvit-deepfake-b5-ff-plus-plus; do
  echo "--- $model ---"
  cd /workspace/models/$model
  source venv/bin/activate
  # Install transformers 4.49.0 which doesn't have _initialize_weights requirement
  pip install 'transformers==4.49.0' 2>&1 | tail -3
  deactivate
done

# Fix 5: opengvlab-videomaev2-base - same _initialize_weights issue
echo ""
echo "=== Fix 5: opengvlab-videomaev2-base ==="
cd /workspace/models/opengvlab-videomaev2-base
source venv/bin/activate
pip install 'transformers==4.49.0' 2>&1 | tail -3
deactivate

echo ""
echo "=== Batch 1 fixes complete ==="
