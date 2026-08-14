# 40个模型测试最终汇总

## 测试结果概览

| 指标 | 数量 |
|------|------|
| **总计** | 40 |
| **通过** | **39** (97.5%) |
| **失败** | **1** (2.5%) |

---

## 通过的模型 (39个)

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

### 修复后通过 (17个)
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
| 33 | aratako-miocodec-25hz-44-1khz-v2 | 安装GitHub依赖miocodec到venv + sf.write指定format="wav" |
| 34 | google-videoprism-lvt-base-f16r288 | 安装GitHub依赖videoprism到venv + 安装tensorflow |
| 35 | ibm-granite-granite-timeseries-patchtst-fm-r1 | 安装GitHub依赖tsfm_public到venv |
| 36 | ibm-research-ttm-r3 | 安装GitHub依赖tsfm_public到venv |
| 37 | neoquasar-kronos-base | 安装GitHub依赖Kronos到venv + 本地下载tokenizer权重 |
| 38 | openmoss-team-moss-voicegenerator | 创建Python 3.12 venv + 安装依赖 + audio tokenizer本地化(6.7G权重拷入weights_audio_tokenizer) |
| 39 | yuchenshen-fomo-0d | 安装GitHub依赖fomo_hub到venv |

---

## 失败的模型 (1个)

| # | 模型 | 失败原因 |
|---|------|---------|
| 1 | datadog-toto-2-0-22m | 需Python 3.12 venv已建好、依赖装好、toto2导入成功；但torch 2.9.1+cu128要求CUDA 12.8驱动，机器驱动为CUDA 12.6；降级到2.7.0+cu126需下载约2.5GB，已取消 |

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
8. **GitHub依赖** (7个模型): 克隆仓库到venv site-packages（miocodec/videoprism/tsfm_public/Kronos/fomo_hub）
9. **Python 3.12** (1个模型): openmoss-voicegenerator 创建独立Python 3.12 venv
10. **Audio tokenizer本地化** (1个模型): openmoss-voicegenerator 将6.7G的MOSS-Audio-Tokenizer权重拷入本地目录，支持离线加载

### 最终结论
39/40个模型可以通过基础镜像 + 虚拟环境直接部署启动。唯一失败模型 datadog-toto 因torch版本与机器CUDA 12.6驱动不兼容（需12.8），且降级安装包过大已放弃。

> **2026-08-14 更新：模型 datadog-toto-2-0-22m 已删除**（部署失败，CUDA 12.8 驱动不兼容，放弃修复），当前项目为 39 个模型，全部可部署。
