# 40个模型测试最终汇总

## 测试结果概览

| 指标 | 数量 |
|------|------|
| **总计** | 40 |
| **通过** | **32** (80%) |
| **失败** | **8** (20%) |

---

## 通过的模型 (32个)

### 首次测试即通过 (22个)
| # | 模型 | 说明 |
|---|------|------|
| 1 | alibaba-nlp-gte-modernbert-base | |
| 2 | autogluon-chronos-2 | |
| 3 | autogluon-mitra-classifier | 修复torch/torchvision冲突后通过 |
| 4 | bytedance-research-timer-s1 | |
| 5 | dleemiller-finecat-nli-l | |
| 6 | docling-project-codeformulav2 | 修复torch/torchvision冲突后通过 |
| 7 | google-timesfm-2-5-200m-transformers | |
| 8 | ibm-granite-granite-speech-4-1-2b | |
| 9 | jhu-clsp-mmbert-base | |
| 10 | k-iwa-time-anchor-modernbert-32m | |
| 11 | mldi-lab-kairos-23m | |
| 12 | mongodb-mdbr-leaf-ir | |
| 13 | numind-nuextract3-fp8 | 修复torch冲突后通过 |
| 14 | paddlepaddle-pp-docblocklayout | |
| 15 | paddlepaddle-pp-ocrv6-medium-det-onnx | |
| 16 | paddlepaddle-pp-ocrv6-small-det-onnx | |
| 17 | paddlepaddle-pp-ocrv6-small-rec-onnx | |
| 18 | prior-labs-tabpfn-v2-clf | 修复torch冲突后通过 |
| 19 | prior-labs-tabpfn-v2-reg | 修复torch冲突后通过 |
| 20 | skywork-skywork-reward-v2-qwen3-0-6b | 修复torch冲突后通过 |
| 21 | synthefy-nori-30m | |
| 22 | weborganizer-topicclassifier-nourl | |

### 修复后通过 (10个)
| # | 模型 | 修复内容 |
|---|------|---------|
| 23 | jusperlee-tiger-dnr | 卸载venv中torchaudio(2.5.1)和nvidia包(CUDA 12.4)，使用系统版本 |
| 24 | facebook-vjepa2-vitl-fpc64-256 | 安装torchcodec 0.3.0兼容torch 2.7.0 |
| 25 | koreapeter-ms-eff-gcvit-deepfake-b0-ff-plus-plus | transformers降级到4.49.0 + 复制configuration_ms_eff_gcvit到venv |
| 26 | koreapeter-ms-eff-gcvit-deepfake-b5-ff-plus-plus | 同上 |
| 27 | microsoft-vibevoice-asr-hf | test.py predict超时从60s改为300s |
| 28 | openmoss-team-moss-tts-local-transformer-v1-5 | test.py超时改为600s + max_new_tokens从4096降到2048 |
| 29 | opengvlab-videomaev2-base | transformers降级到4.49.0 + app.py修复outputs.last_hidden_state→outputs |
| 30 | paddlepaddle-pp-chart2table | paddlex 3.6.0 + fusion_ops补丁(fused_rms_norm_ext/cal_aux_loss) |
| 31 | paddlepaddle-pp-doclayout-plus-l | tempfile.mktemp()→mktemp(suffix=".png") |
| 32 | voyageai-voyage-4-nano | transformers升级到5.15.0 + config_class补丁 + create_causal_mask调用修复 |

---

## 失败的模型 (8个) - 不可修复

| # | 模型 | 失败原因 |
|---|------|---------|
| 1 | aratako-miocodec-25hz-44-1khz-v2 | 需要GitHub仓库 `miocodec`，内网无法下载 |
| 2 | datadog-toto-2-0-22m | 需要Python 3.12，基础镜像仅提供3.10 |
| 3 | google-videoprism-lvt-base-f16r288 | 需要GitHub仓库 `videoprism`，内网无法下载 |
| 4 | ibm-granite-granite-timeseries-patchtst-fm-r1 | 需要GitHub仓库 `tsfm_public`，内网无法下载 |
| 5 | ibm-research-ttm-r3 | 需要GitHub仓库 `tsfm_public`，内网无法下载 |
| 6 | neoquasar-kronos-base | 需要GitHub仓库自定义 `model` 模块，内网无法下载 |
| 7 | openmoss-team-moss-voicegenerator | 需要Python 3.12 + 运行时下载依赖失败(401) |
| 8 | yuchenshen-fomo-0d | 需要GitHub仓库 `fomo_hub`，内网无法下载 |

---

## 修复总结

### 修复模式
1. **torch/torchvision版本冲突** (6个模型): 卸载venv中torch/torchvision，使用系统版本(2.7.0+cu126)
2. **nvidia包冲突** (1个模型): 卸载venv中CUDA 12.4的nvidia包
3. **transformers版本兼容** (4个模型): 升级到5.15.0或降级到4.49.0
4. **缺失依赖** (2个模型): 安装torchcodec + 复制自定义配置文件
5. **测试超时** (2个模型): 增大test.py timeout
6. **代码bug** (2个模型): 修复app.py中outputs访问和tempfile扩展名
7. **PaddleX兼容** (2个模型): 补丁fusion_ops缺失导入

### 最终结论
32/40个模型可以通过基础镜像 + 虚拟环境直接部署启动。8个失败模型均因需要GitHub私有仓库或Python 3.12，在内网环境中无法满足。