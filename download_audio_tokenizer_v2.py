import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import snapshot_download

target = r"d:\ssd-projects\Models-Deploy\models\openmoss-team-moss-tts-local-transformer-v1-5\weights_audio_tokenizer"
os.makedirs(target, exist_ok=True)

snapshot_download(
    repo_id="OpenMOSS-Team/MOSS-Audio-Tokenizer-v2",
    local_dir=target,
    allow_patterns=[
        "config.json",
        "configuration_moss_audio_tokenizer.py",
        "modeling_moss_audio_tokenizer.py",
        "model.safetensors.index.json",
        "model-*.safetensors",
    ],
)
print("DOWNLOAD_DONE")
