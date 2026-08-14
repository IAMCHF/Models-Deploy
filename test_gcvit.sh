#!/bin/bash
# Test GCViT model loading after config copy
cd /workspace/models/koreapeter-ms-eff-gcvit-deepfake-b0-ff-plus-plus
source venv/bin/activate
echo "=== Test model loading ==="
python << 'EOF'
from transformers import pipeline
import torch
print("Loading pipeline...")
clf = pipeline(
    "image-classification",
    model="weights",
    trust_remote_code=True,
    device=0 if torch.cuda.is_available() else -1,
)
print("Pipeline loaded OK:", type(clf.model).__name__)
EOF
deactivate
