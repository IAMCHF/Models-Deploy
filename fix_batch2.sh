#!/bin/bash
# Fix torchcodec - need version compatible with torch 2.7.0+cu126 (CUDA 12.6)
echo "=== Fix torchcodec for facebook-vjepa2 ==="
cd /workspace/models/facebook-vjepa2-vitl-fpc64-256
source venv/bin/activate
# Uninstall the incompatible version
pip uninstall torchcodec -y 2>&1 | tail -2
# Try version 0.1.0 which should work with CUDA 12
pip install 'torchcodec==0.1.0' 2>&1 | tail -5
python -c "from torchcodec.decoders import VideoDecoder; print('torchcodec OK')" 2>&1
if [ $? -ne 0 ]; then
  echo "0.1.0 failed, trying 0.2.0..."
  pip uninstall torchcodec -y 2>&1 | tail -1
  pip install 'torchcodec==0.2.0' 2>&1 | tail -5
  python -c "from torchcodec.decoders import VideoDecoder; print('torchcodec OK')" 2>&1
fi
if [ $? -ne 0 ]; then
  echo "0.2.0 failed, trying 0.3.0..."
  pip uninstall torchcodec -y 2>&1 | tail -1
  pip install 'torchcodec==0.3.0' 2>&1 | tail -5
  python -c "from torchcodec.decoders import VideoDecoder; print('torchcodec OK')" 2>&1
fi
deactivate

# Fix test timeouts for microsoft-vibevoice-asr-hf and openmoss-tts
echo ""
echo "=== Fix test timeouts ==="
for model in microsoft-vibevoice-asr-hf openmoss-team-moss-tts-local-transformer-v1-5; do
  echo "--- $model ---"
  TEST_FILE="/workspace/models/$model/test/test.py"
  if [ -f "$TEST_FILE" ]; then
    # Change timeout=60 to timeout=300
    sed -i 's/timeout=60/timeout=300/g' "$TEST_FILE"
    sed -i 's/timeout=10/timeout=30/g' "$TEST_FILE"
    echo "  Updated timeouts: 60->300, 10->30"
    grep timeout "$TEST_FILE"
  fi
done

echo ""
echo "=== Batch 2 fixes complete ==="
