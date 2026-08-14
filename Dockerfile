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
        git wget curl vim nano ffmpeg software-properties-common \
        ca-certificates gnupg build-essential \
        zip unzip tar gzip bzip2 xz-utils file rsync dos2unix \
        htop tmux less tree jq socat \
        net-tools iproute2 procps psmisc \
        openssh-client \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Python 双版本：3.11（默认 python3）+ 3.12（deadsnakes PPA）
# ------------------------------------------------------------
RUN add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
        python3.12 python3.12-venv python3.12-dev \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# python3 默认指向 3.11，并为 3.11 安装独立的 pip
# 修复：系统 python3-pip 绑定到 3.10，需用 ensurepip 为 3.11 单独安装
RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && python3 -m ensurepip --upgrade \
    && python3 -m pip install --upgrade pip

# ------------------------------------------------------------
# PyTorch 2.5.1 (cu124) — 仅安装于 Python 3.11
# 注：nvidia-cudnn-cu12==9.1.0.70 已从 PyPI 下架，先装可用版本再 --no-deps 装 torch
# ------------------------------------------------------------
RUN python3 -m pip install --no-cache-dir nvidia-cudnn-cu12==9.1.1.17 && \
    python3 -m pip install --no-cache-dir --no-deps \
        torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu124 && \
    python3 -m pip install --no-cache-dir \
        nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-cupti-cu12 \
        nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 \
        nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cusparselt-cu12 \
        nvidia-nccl-cu12 nvidia-nvtx-cu12 nvidia-nvjitlink-cu12 \
        triton==3.1.0 sympy networkx jinja2 fsspec filelock typing-extensions

# ------------------------------------------------------------
# 基础 ML 依赖层（所有模型共享，仅 Python 3.11）
# ------------------------------------------------------------
RUN python3 -m pip install --no-cache-dir \
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
