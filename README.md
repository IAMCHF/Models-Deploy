# 内网模型批量部署工具

> **核心原则**：所有与 Hugging Face 相关的访问（下载权重、查阅示例代码）优先使用国内镜像站 `https://hf-mirror.com`，镜像站不可用时以官方 `huggingface.co` 作为兜底。所有脚本内置镜像站环境变量。

---

## 项目概述

读取模型列表（41 个），自动生成批量下载权重脚本、基础 Docker 镜像、每模型独立的 FastAPI 服务代码、虚拟环境脚本、启动脚本和测试代码。模型权重 ≥10GB 时自动标注提醒内网容器扩容。

## 目录结构

```
项目根目录/
├── models/
│   └── {模型文件夹名}/              # 共 41 个模型目录
│       ├── fastapi.py              # 服务主程序（端口 8080）
│       ├── test/
│       │   ├── test.py             # 测试脚本
│       │   └── test_data/          # 测试数据（需下载或生成）
│       ├── weights/                # 存放模型权重（需下载）
│       ├── create_env.sh           # 创建虚拟环境（含换源）
│       ├── start.sh                # 启动服务
│       ├── requirements.txt        # 模型特有依赖
│       └── env_info.txt            # 环境记录（create_env.sh 生成）
├── download_weights.py             # 批量下载脚本（镜像站优先+官方兜底）
├── models_list.json                # 模型清单（41个，结构化输入）
├── set_env.sh                      # 全局环境变量（镜像站优先）
├── deploy_all.sh                   # 一键部署脚本
├── Dockerfile                      # 基础镜像（CUDA 12.4 + PyTorch 2.4 + vLLM）
├── .github/workflows/build.yml     # GitHub Actions 构建基础镜像
└── README.md
```

## 快速开始

### 1. 外网：下载模型权重

```bash
# 下载全部模型权重（镜像站优先）
python download_weights.py

# 仅下载指定模型
python download_weights.py --only mldi-lab-kairos-23m,autogluon-chronos-2

# 强制重新下载
python download_weights.py --force
```

### 2. 外网：构建基础 Docker 镜像

推送代码到 GitHub，Actions 自动构建基础镜像并产出 `.tar.gz` artifact 供下载。

```bash
# 或手动构建
docker build -t models-deploy-base .
```

### 3. 内网：部署流程

```bash
# 导入基础镜像
docker load < models-deploy-base.tar.gz

# 一键部署（创建环境 -> 启动服务 -> 测试）
./deploy_all.sh all

# 或分步执行
./deploy_all.sh env     # 批量创建虚拟环境
./deploy_all.sh start   # 批量启动服务
./deploy_all.sh test    # 批量运行测试
./deploy_all.sh list    # 列出所有模型
```

## 基础镜像

| 组件 | 版本 |
|------|------|
| 基础镜像 | `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` |
| Python | 3.11（默认）+ 3.12（MOSS-TTS 等使用） |
| PyTorch | 2.4.0 + torchvision 0.19.0 + torchaudio 2.4.0 (cu124) |
| transformers | 4.52.0 |
| vLLM | 最新稳定版 |
| ONNX Runtime | GPU 版 |

**venv 覆盖模型（11 个）**：基础镜像覆盖 ~30 个模型默认需求，以下 11 个通过各自 `create_env.sh` 在 venv 中覆盖版本：

| 模型 | 覆盖项 |
|------|--------|
| MOSS-TTS / MOSS-VoiceGenerator | torch 2.9.1+cu128, transformers>=5.0, Python 3.12 |
| VibeVoice-ASR | transformers>=5.3.0 |
| Timer-S1 | transformers~=4.57.1 |
| TIGER-DnR | torch 2.5.1, transformers 4.47.1, pytorch-lightning 2.0.2 |
| PP-DocBlockLayout / PP-DocLayout_plus-L | paddlepaddle-gpu 3.0.0 |
| PP-Chart2Table | paddlepaddle-gpu 3.0.0 + PaddleX |
| VideoPrism | jax + flax |
| CodeFormulaV2 | docling + onnxruntime |
| FoMo-0D | pytorch-lightning + hydra + wandb |

## 接口规范

每个模型 FastAPI 服务监听 `0.0.0.0:8080`：

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 返回 `{"status": "ok"}` |
| `/predict` | POST | 接收 `{"data": "<base64>"}`，返回 `{"result": "<base64>"}` |

## 大模型标注

权重 ≥10GB 的模型会在模型根目录生成 `.large_model` 标记文件，根目录生成 `large_models_list.txt` 汇总清单。内网部署时需为这些模型扩容容器存储。

## 宿主机要求

- NVIDIA 驱动 ≥ 550.54（支持 CUDA 12.4 及 cu128 wheel 向下兼容）
- 安装 NVIDIA Container Toolkit

## 模型服务代码说明（重要）

当前 `fastapi.py` 为**初始框架骨架**，包含完整的 API 结构（`/health`、`/predict`、base64 编解码）和明确的 TODO 标记区域。**模型加载与推理代码需从 HuggingFace 模型页面获取真实部署代码后填入**：

1. 访问镜像站 `https://hf-mirror.com/{model_id}`（不可用时访问官方 `https://huggingface.co/{model_id}`）
2. 解析页面中 "Use in Transformers" / "Use in vLLM" / "How to use" 等代码片段
3. 适配到 `fastapi.py` 的 TODO 区域
4. 若镜像站和官方均无示例代码，需汇总缺失模型列表询问用户

> **约束**：禁止使用通用 `AutoModel`/`pipeline` 模板生成加载代码，必须参考模型官方示例。
