#!/bin/bash
# ============================================================
# 全局环境变量配置（镜像站优先 + 官方兜底）
# 用法：source ./set_env.sh
# ============================================================

# Hugging Face 镜像站优先，官方源兜底
export HF_ENDPOINT="https://hf-mirror.com"

# PyPI 国内镜像源（仅内网本地 create_env.sh 使用，GitHub Actions 构建时勿 source 此文件）
export PIP_INDEX_URL="https://pypi.mirrors.ustc.edu.cn/simple"
export PIP_TRUSTED_HOST="pypi.mirrors.ustc.edu.cn"

# pip 缓存目录（启用缓存，避免重复下载）
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$(pwd)/.pip_cache}"

# 保留代理兼容性（如内网需走代理）
# export HTTP_PROXY="http://127.0.0.1:7890"
# export HTTPS_PROXY="http://127.0.0.1:7890"

echo "[set_env] HF_ENDPOINT=$HF_ENDPOINT"
echo "[set_env] PIP_INDEX_URL=$PIP_INDEX_URL"
