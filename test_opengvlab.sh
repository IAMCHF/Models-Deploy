#!/bin/bash
cd /workspace/models/opengvlab-videomaev2-base
source venv/bin/activate
python3 << 'EOF'
import torch, numpy as np, decord
from transformers import AutoModel, AutoConfig
from pathlib import Path

WEIGHTS_DIR = Path("/workspace/models/opengvlab-videomaev2-base/weights")
print("Loading config...")
config = AutoConfig.from_pretrained(str(WEIGHTS_DIR), trust_remote_code=True)
print("Config class:", type(config).__name__)
print("Loading model...")
model = AutoModel.from_pretrained(str(WEIGHTS_DIR), config=config, trust_remote_code=True)
print("Model class:", type(model).__name__)
print("Model forward signature:", model.forward.__code__.co_varnames[:10])

# Check processor
from transformers import AutoImageProcessor
try:
    processor = AutoImageProcessor.from_pretrained(str(WEIGHTS_DIR), trust_remote_code=True)
    print("Processor:", type(processor).__name__)
except Exception as e:
    print("Processor error:", e)

# Test with dummy input
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
# Create dummy video
import tempfile, os
tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
tmp.close()
# Generate a small video with decord-compatible format
import subprocess
subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=1:size=224x224:rate=16", "-pix_fmt", "yuv420p", tmp.name], capture_output=True)
vr = decord.VideoReader(tmp.name)
total = len(vr)
indices = np.linspace(0, total-1, 16, dtype=int)
frames = vr.get_batch(indices).asnumpy()
video = [frames[i] for i in range(16)]
inputs = processor(video, return_tensors="pt")
inputs['pixel_values'] = inputs['pixel_values'].permute(0, 2, 1, 3, 4)
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)
print("Output type:", type(outputs))
if hasattr(outputs, "last_hidden_state"):
    print("Has last_hidden_state, shape:", outputs.last_hidden_state.shape)
elif isinstance(outputs, torch.Tensor):
    print("Is Tensor, shape:", outputs.shape)
elif isinstance(outputs, tuple):
    print("Is tuple, len:", len(outputs))
    for i, o in enumerate(outputs):
        print(f"  [{i}] type={type(o).__name__}", getattr(o, 'shape', ''))
os.unlink(tmp.name)
EOF
deactivate
