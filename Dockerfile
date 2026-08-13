# ============================================================
# 基础镜像：内网模型批量部署 - 基础层
# 覆盖 ~30 个通用模型默认需求，剩余 ~11 个模型通过各自 create_env.sh 在 venv 中覆盖版本
# ============================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# 避免 tzdata 等交互式提问
ENV DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------------------
# 系统工具层
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl vim ffmpeg software-properties-common \
        ca-certificates gnupg build-essential \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Python 双版本：3.11（默认 python3）+ 3.12（deadsnakes PPA）
# ------------------------------------------------------------
RUN add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
        python3.12 python3.12-venv python3.12-dev \
        python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --upgrade pip

# python3 / pip3 默认指向 3.11
RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python

# ------------------------------------------------------------
# PyTorch 2.4.0 (cu124) — 仅安装于 Python 3.11
# ------------------------------------------------------------
RUN pip3 install --no-cache-dir \
        torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu124

# ------------------------------------------------------------
# 基础 ML 依赖层（所有模型共享，仅 Python 3.11）
# ------------------------------------------------------------
RUN pip3 install --no-cache-dir \
        transformers==4.52.0 \
        vllm \
        onnxruntime-gpu \
        huggingface_hub \
        accelerate \
        safetensors \
        sentencepiece \
        soundfile \
        librosa \
        numpy \
        Pillow \
        fastapi \
        uvicorn[standard] \
        pydantic \
        requests

# ------------------------------------------------------------
# 工作目录
# ------------------------------------------------------------
WORKDIR /workspace

# 预留 HuggingFace 镜像站环境变量（容器内默认走镜像站）
ENV HF_ENDPOINT="https://hf-mirror.com"

CMD ["bash"]
