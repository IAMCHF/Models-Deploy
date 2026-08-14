# 39 个模型单容器部署命令（手动测试用）

> 每个模型一个容器，只挂载单个模型目录，端口映射到本机 8080。
> 宿主机直接运行 `test.py` 测试（test.py 只依赖 requests，宿主机已装）。

## 重要说明

1. **挂载路径必须是 `/workspace/models/{模型名}`**：venv 的 bin 脚本（activate/pip/入口脚本）硬编码了该路径，挂到别处会导致 venv 失效、服务用基础 Python 运行而报缺包错误。
2. 每条 `docker run` 是**单行命令**（PowerShell / Git Bash 均可直接粘贴）。
3. 每次只测一个模型：测完先 `docker stop` + `docker rm`，再测下一个（8080 端口复用）。
4. 容器启动后等待 `/health` 就绪（大模型加载需 1-3 分钟），再跑宿主机测试。
5. 宿主机测试命令：`python d:/ssd-projects/Models-Deploy/models/{模型名}/test/test.py`

---

## 1. alibaba-nlp-gte-modernbert-base

```bash
docker run -d --name alibaba-nlp-gte-modernbert-base --gpus all -p 8080:8080 -v "d:/ssd-projects/Models-Deploy/models/alibaba-nlp-gte-modernbert-base:/workspace/models/alibaba-nlp-gte-modernbert-base" -w "/workspace/models/alibaba-nlp-gte-modernbert-base" models-deploy-base:latest bash start.sh
```

```powershell
python d:/ssd-projects/Models-Deploy/models/alibaba-nlp-gte-modernbert-base/test/test.py
```

```bash
docker stop alibaba-nlp-gte-modernbert-base && docker rm alibaba-nlp-gte-modernbert-base
```

---

## 2. aratako-miocodec-25hz-44-1khz-v2

```bash
docker run -d --name aratako-miocodec-25hz-44-1khz-v2 --gpus all -p 8080:8080 -v "d:/ssd-projects/Models-Deploy/models/aratako-miocodec-25hz-44-1khz-v2:/workspace/models/aratako-miocodec-25hz-44-1khz-v2" -w "/workspace/models/aratako-miocodec-25hz-44-1khz-v2" models-deploy-base:latest bash start.sh
```

```powershell
python d:/ssd-projects/Models-Deploy/models/aratako-miocodec-25hz-44-1khz-v2/test/test.py
```

```bash
docker stop aratako-miocodec-25hz-44-1khz-v2 && docker rm aratako-miocodec-25hz-44-1khz-v2
```

---

## 3. autogluon-chronos-2

```bash
docker run -d --name autogluon-chronos-2 --gpus all -p 8080:8080 -v "d:/ssd-projects/Models-Deploy/models/autogluon-chronos-2:/workspace/models/autogluon-chronos-2" -w "/workspace/models/autogluon-chronos-2" models-deploy-base:latest bash start.sh
```

```powershell
python d:/ssd-projects/Models-Deploy/models/autogluon-chronos-2/test/test.py
```

```bash
docker stop autogluon-chronos-2 && docker rm autogluon-chronos-2
```

---

## 4. autogluon-mitra-classifier

```bash
docker run -d --name autogluon-mitra-classifier --gpus all -p 8080:8080 -v "d:/ssd-projects/Models-Deploy/models/autogluon-mitra-classifier:/workspace/models/autogluon-mitra-classifier" -w "/workspace/models/autogluon-mitra-classifier" models-deploy-base:latest bash start.sh
```

```powershell
python d:/ssd-projects/Models-Deploy/models/autogluon-mitra-classifier/test/test.py
```

```bash
docker stop autogluon-mitra-classifier && docker rm autogluon-mitra-classifier
```

---

## 5. bytedance-research-timer-s1

```bash
docker run -d --name bytedance-research-timer-s1 --gpus all -p 8080:8080 -v "d:/ssd-projects/Models-Deploy/models/bytedance-research-timer-s1:/workspace/models/bytedance-research-timer-s1" -w "/workspace/models/bytedance-research-timer-s1" models-deploy-base:latest bash start.sh
```

```powershell
python d:/ssd-projects/Models-Deploy/models/bytedance-research-timer-s1/test/test.py
```

```bash
docker stop bytedance-research-timer-s1 && docker rm bytedance-research-timer-s1
```

---

## 6. dleemiller-finecat-nli-l

```bash
docker run -d --name dleemiller-finecat-nli-l --gpus all -p 8080:8080 -v "d:/ssd-projects/Models-Deploy/models/dleemiller-finecat-nli-l:/workspace/models/dleemiller-finecat-nli-l" -w "/workspace/models/dleemiller-finecat-nli-l" models-deploy-base:latest bash start.sh
```

```powershell
python d:/ssd-projects/Models-Deploy/models/dleemiller-finecat-nli-l/test/test.py
```

```bash
docker stop dleemiller-finecat-nli-l && docker rm dleemiller-finecat-nli-l
```

---

## 7. docling-project-codeformulav2

```bash
docker run -d --name docling-project-codeformulav2 --gpus all -p 8080:8080 -v "d:/ssd-projects/Models-Deploy/models/docling-project-codeformulav2:/workspace/models/docling-project-codeformulav2" -w "/workspace/models/docling-project-codeformulav2" models-deploy-base:latest bash start.sh
```

```powershell
python d:/ssd-projects/Models-Deploy/models/docling-project-codeformulav2/test/test.py
```

```bash
docker stop docling-project-codeformulav2 && docker rm docling-project-codeformulav2
```

---

## 8. facebook-vjepa2-vitl-fpc64-256

```bash
docker run -d --name facebook-vjepa2-vitl-fpc64-256 --gpus all -p 8080:8080 -v "d:/ssd-projects/Models-Deploy/models/facebook-vjepa2-vitl-fpc64-256:/workspace/models/facebook-vjepa2-vitl-fpc64-256" -w "/workspace/models/facebook-vjepa2-vitl-fpc64-256" models-deploy-base:latest bash start.sh
```

```powershell
python d:/ssd-projects/Models-Deploy/models/facebook-vjepa2-vitl-fpc64-256/test/test.py
```

```bash
docker stop facebook-vjepa2-vitl-fpc64-256 && docker rm facebook-vjepa2-vitl-fpc64-256
```

---

## 9. google-timesfm-2-5-200m-transformers

```bash
docker run -d --name google-timesfm-2-5-200m-transformers --gpus all -p 8080:8080 -v "d:/ssd-projects/Models-Deploy/models/google-timesfm-2-5-200m-transformers:/workspace/models/google-timesfm-2-5-200m-transformers" -w "/workspace/models/google-timesfm-2-5-200m-transformers" models-deploy-base:latest bash start.sh
```

```powershell
python d:/ssd-projects/Models-Deploy/models/google-timesfm-2-5-200m-transformers/test/test.py
```

```bash
docker stop google-timesfm-2-5-200m-transformers && docker rm google-timesfm-2-5-200m-transformers
```

---

## 10. google-videoprism-lvt-base-f16r288

```bash
docker run -d --name google-videoprism-lvt-base-f16r288 --gpus all -p 8080:8080 -v "d:/ssd-projects/Models-Deploy/models/google-videoprism-lvt-base-f16r288:/workspace/models/google-videoprism-lvt-base-f16r288" -w "/workspace/models/google-videoprism-lvt-base-f16r288" models-deploy-base:latest bash start.sh
```

```powershell
python d:/ssd-projects/Models-Deploy/models/google-videoprism-lvt-base-f16r288/test/test.py
```

```bash
docker stop google-videoprism-lvt-base-f16r288 && docker rm google-videoprism-lvt-base-f16r288
```

---

## 11. ibm-granite-granite-speech-4-1-2b

```bash
docker run -d --name ibm-granite-granite-speech-4-1-2b --gpus all -p 8080:8080 -v "d:/ssd-projects/Models-Deploy/models/ibm-granite-granite-speech-4-1-2b:/workspace/models/ibm-granite-granite-speech-4-1-2b" -w "/workspace/models/ibm-granite-granite-speech-4-1-2b" models-deploy-base:latest bash start.sh
```

```powershell
python d:/ssd-projects/Models-Deploy/models/ibm-granite-granite-speech-4-1-2b/test/test.py
```

```bash
docker stop ibm-granite-granite-speech-4-1-2b && docker rm ibm-granite-granite-speech-4-1-2b
```

---

## 12. ibm-granite-granite-timeseries-patchtst-fm-r1

```bash
docker run -d --name ibm-granite-granite-timeseries-patchtst-fm-r1 --gpus all -p 8080:8080 -v "d:/ssd-projects/Models-Deploy/models/ibm-granite-granite-timeseries-patchtst-fm-r1:/workspace/models/ibm-granite-granite-timeseries-patchtst-fm-r1" -w "/workspace/models/ibm-granite-granite-timeseries-patchtst-fm-r1" models-deploy-base:latest bash start.sh
```

```powershell
python d:/ssd-projects/Models-Deploy/models/ibm-granite-granite-timeseries-patchtst-fm-r1/test/test.py
```

```bash
docker stop ibm-granite-granite-timeseries-patchtst-fm-r1 && docker rm ibm-granite-granite-timeseries-patchtst-fm-r1
```

---

## 13. ibm-research-ttm-r3

```bash
docker run -d --name ibm-research-ttm-r3 --gpus all -p 8080:8080 -v "d:/ssd-projects/Models-Deploy/models/ibm-research-ttm-r3:/workspace/models/ibm-research-ttm-r3" -w "/workspace/models/ibm-research-ttm-r3" models-deploy-base:latest bash start.sh
```

```powershell
python d:/ssd-projects/Models-Deploy/models/ibm-research-ttm-r3/test/test.py
```

```bash
docker stop ibm-research-ttm-r3 && docker rm ibm-research-ttm-r3
```

---

## 14. jhu-clsp-mmbert-base

```bash
docker run -d --name jhu-clsp-mmbert-base --gpus all -p 8080:8080 -v "d:/ssd-projects/Models-Deploy/models/jhu-clsp-mmbert-base:/workspace/models/jhu-clsp-mmbert-base" -w "/workspace/models/jhu-clsp-mmbert-base" models-deploy-base:latest bash start.sh
```

```powershell
python d:/ssd-projects/Models-Deploy/models/jhu-clsp-mmbert-base/test/test.py
```

```bash
docker stop jhu-clsp-mmbert-base && docker rm jhu-clsp-mmbert-base
```

---

## 15. jusperlee-tiger-dnr

```bash
docker run -d --name jusperlee-tiger-dnr --gpus all -p 8080:8080 -v "d:/ssd-projects/Models-Deploy/models/jusperlee-tiger-dnr:/workspace/models/jusperlee-tiger-dnr" -w "/workspace/models/jusperlee-tiger-dnr" models-deploy-base:latest bash start.sh
```

```powershell
python d:/ssd-projects/Models-Deploy/models/jusperlee-tiger-dnr/test/test.py
```

```bash
docker stop jusperlee-tiger-dnr && docker rm jusperlee-tiger-dnr
```

---

## 16. k-iwa-time-anchor-modernbert-32m

```bash
docker run -d --name k-iwa-time-anchor-modernbert-32m --gpus all -p 8080:8080 -v "d:/ssd-projects/Models-Deploy/models/k-iwa-time-anchor-modernbert-32m:/workspace/models/k-iwa-time-anchor-modernbert-32m" -w "/workspace/models/k-iwa-time-anchor-modernbert-32m" models-deploy-base:latest bash start.sh
```

```powershell
python d:/ssd-projects/Models-Deploy/models/k-iwa-time-anchor-modernbert-32m/test/test.py
```

```bash
docker stop k-iwa-time-anchor-modernbert-32m && docker rm k-iwa-time-anchor-modernbert-32m
```

---

## 17. koreapeter-ms-eff-gcvit-deepfake-b0-ff-plus-plus

```bash
docker run -d --name koreapeter-ms-eff-gcvit-deepfake-b0-ff-plus-plus --gpus all -p 8080:8080 -v "d:/ssd-projects/Models-Deploy/models/koreapeter-ms-eff-gcvit-deepfake-b0-ff-plus-plus:/workspace/models/koreapeter-ms-eff-gcvit-deepfake-b0-ff-plus-plus" -w "/workspace/models/koreapeter-ms-eff-gcvit-deepfake-b0-ff-plus-plus" models-deploy-base:latest bash start.sh
```

```powershell
python d:/ssd-projects/Models-Deploy/models/koreapeter-ms-eff-gcvit-deepfake-b0-ff-plus-plus/test/test.py
```

```bash
docker stop koreapeter-ms-eff-gcvit-deepfake-b0-ff-plus-plus && docker rm koreapeter-ms-eff-gcvit-deepfake-b0-ff-plus-plus
```

---

## 18. koreapeter-ms-eff-gcvit-deepfake-b5-ff-plus-plus

```bash
docker run -d --name koreapeter-ms-eff-gcvit-deepfake-b5-ff-plus-plus --gpus all -p 8080:8080 -v "d:/ssd-projects/Models-Deploy/models/koreapeter-ms-eff-gcvit-deepfake-b5-ff-plus-plus:/workspace/models/koreapeter-ms-eff-gcvit-deepfake-b5-ff-plus-plus" -w "/workspace/models/koreapeter-ms-eff-gcvit-deepfake-b5-ff-plus-plus" models-deploy-base:latest bash start.sh
```

```powershell
python d:/ssd-projects/Models-Deploy/models/koreapeter-ms-eff-gcvit-deepfake-b5-ff-plus-plus/test/test.py
```

```bash
docker stop koreapeter-ms-eff-gcvit-deepfake-b5-ff-plus-plus && docker rm koreapeter-ms-eff-gcvit-deepfake-b5-ff-plus-plus
```

---

## 19. microsoft-vibevoice-asr-hf

```bash
docker run -d --name microsoft-vibevoice-asr-hf --gpus all -p 8080:8080 -v "d:/ssd-projects/Models-Deploy/models/microsoft-vibevoice-asr-hf:/workspace/models/microsoft-vibevoice-asr-hf" -w "/workspace/models/microsoft-vibevoice-asr-hf" models-deploy-base:latest bash start.sh
```

```powershell
python d:/ssd-projects/Models-Deploy/models/microsoft-vibevoice-asr-hf/test/test.py
```

```bash
docker stop microsoft-vibevoice-asr-hf && docker rm microsoft-vibevoice-asr-hf
```

---

## 20. mldi-lab-kairos-23m

```bash
docker run -d --name mldi-lab-kairos-23m --gpus all -p 8080:8080 -v "d:/ssd-projects/Models-Deploy/models/mldi-lab-kairos-23m:/workspace/models/mldi-lab-kairos-23m" -w "/workspace/models/mldi-lab-kairos-23m" models-deploy-base:latest bash start.sh
```

```powershell
python d:/ssd-projects/Models-Deploy/models/mldi-lab-kairos-23m/test/test.py
```

```bash
docker stop mldi-lab-kairos-23m && docker rm mldi-lab-kairos-23m
```

---

## 21. mongodb-mdbr-leaf-ir

```bash
docker run -d --name mongodb-mdbr-leaf-ir --gpus all -p 8080:8080 -v "d:/ssd-projects/Models-Deploy/models/mongodb-mdbr-leaf-ir:/workspace/models/mongodb-mdbr-leaf-ir" -w "/workspace/models/mongodb-mdbr-leaf-ir" models-deploy-base:latest bash start.sh
```

```powershell
python d:/ssd-projects/Models-Deploy/models/mongodb-mdbr-leaf-ir/test/test.py
```

```bash
docker stop mongodb-mdbr-leaf-ir && docker rm mongodb-mdbr-leaf-ir
```

---

## 22. neoquasar-kronos-base

```bash
docker run -d --name neoquasar-kronos-base --gpus all -p 8080:8080 -v "d:/ssd-projects/Models-Deploy/models/neoquasar-kronos-base:/workspace/models/neoquasar-kronos-base" -w "/workspace/models/neoquasar-kronos-base" models-deploy-base:latest bash start.sh
```

```powershell
python d:/ssd-projects/Models-Deploy/models/neoquasar-kronos-base/test/test.py
```

```bash
docker stop neoquasar-kronos-base && docker rm neoquasar-kronos-base
```

---

## 23. numind-nuextract3-fp8

```bash
docker run -d --name numind-nuextract3-fp8 --gpus all -p 8080:8080 -v "d:/ssd-projects/Models-Deploy/models/numind-nuextract3-fp8:/workspace/models/numind-nuextract3-fp8" -w "/workspace/models/numind-nuextract3-fp8" models-deploy-base:latest bash start.sh
```

```powershell
python d:/ssd-projects/Models-Deploy/models/numind-nuextract3-fp8/test/test.py
```

```bash
docker stop numind-nuextract3-fp8 && docker rm numind-nuextract3-fp8
```

---

## 24. opengvlab-videomaev2-base

```bash
docker run -d --name opengvlab-videomaev2-base --gpus all -p 8080:8080 -v "d:/ssd-projects/Models-Deploy/models/opengvlab-videomaev2-base:/workspace/models/opengvlab-videomaev2-base" -w "/workspace/models/opengvlab-videomaev2-base" models-deploy-base:latest bash start.sh
```

```powershell
python d:/ssd-projects/Models-Deploy/models/opengvlab-videomaev2-base/test/test.py
```

```bash
docker stop opengvlab-videomaev2-base && docker rm opengvlab-videomaev2-base
```

---

## 25. openmoss-team-moss-tts-local-transformer-v1-5

```bash
docker run -d --name openmoss-team-moss-tts-local-transformer-v1-5 --gpus all -p 8080:8080 -v "d:/ssd-projects/Models-Deploy/models/openmoss-team-moss-tts-local-transformer-v1-5:/workspace/models/openmoss-team-moss-tts-local-transformer-v1-5" -w "/workspace/models/openmoss-team-moss-tts-local-transformer-v1-5" models-deploy-base:latest bash start.sh
```

```powershell
python d:/ssd-projects/Models-Deploy/models/openmoss-team-moss-tts-local-transformer-v1-5/test/test.py
```

```bash
docker stop openmoss-team-moss-tts-local-transformer-v1-5 && docker rm openmoss-team-moss-tts-local-transformer-v1-5
```

---

## 26. openmoss-team-moss-voicegenerator

```bash
docker run -d --name openmoss-team-moss-voicegenerator --gpus all -p 8080:8080 -v "d:/ssd-projects/Models-Deploy/models/openmoss-team-moss-voicegenerator:/workspace/models/openmoss-team-moss-voicegenerator" -w "/workspace/models/openmoss-team-moss-voicegenerator" models-deploy-base:latest bash start.sh
```

```powershell
python d:/ssd-projects/Models-Deploy/models/openmoss-team-moss-voicegenerator/test/test.py
```

```bash
docker stop openmoss-team-moss-voicegenerator && docker rm openmoss-team-moss-voicegenerator
```

---

## 27. paddlepaddle-pp-chart2table

```bash
docker run -d --name paddlepaddle-pp-chart2table --gpus all -p 8080:8080 -v "d:/ssd-projects/Models-Deploy/models/paddlepaddle-pp-chart2table:/workspace/models/paddlepaddle-pp-chart2table" -w "/workspace/models/paddlepaddle-pp-chart2table" models-deploy-base:latest bash start.sh
```

```powershell
python d:/ssd-projects/Models-Deploy/models/paddlepaddle-pp-chart2table/test/test.py
```

```bash
docker stop paddlepaddle-pp-chart2table && docker rm paddlepaddle-pp-chart2table
```

---

## 28. paddlepaddle-pp-docblocklayout

```bash
docker run -d --name paddlepaddle-pp-docblocklayout --gpus all -p 8080:8080 -v "d:/ssd-projects/Models-Deploy/models/paddlepaddle-pp-docblocklayout:/workspace/models/paddlepaddle-pp-docblocklayout" -w "/workspace/models/paddlepaddle-pp-docblocklayout" models-deploy-base:latest bash start.sh
```

```powershell
python d:/ssd-projects/Models-Deploy/models/paddlepaddle-pp-docblocklayout/test/test.py
```

```bash
docker stop paddlepaddle-pp-docblocklayout && docker rm paddlepaddle-pp-docblocklayout
```

---

## 29. paddlepaddle-pp-doclayout-plus-l

```bash
docker run -d --name paddlepaddle-pp-doclayout-plus-l --gpus all -p 8080:8080 -v "d:/ssd-projects/Models-Deploy/models/paddlepaddle-pp-doclayout-plus-l:/workspace/models/paddlepaddle-pp-doclayout-plus-l" -w "/workspace/models/paddlepaddle-pp-doclayout-plus-l" models-deploy-base:latest bash start.sh
```

```powershell
python d:/ssd-projects/Models-Deploy/models/paddlepaddle-pp-doclayout-plus-l/test/test.py
```

```bash
docker stop paddlepaddle-pp-doclayout-plus-l && docker rm paddlepaddle-pp-doclayout-plus-l
```

---

## 30. paddlepaddle-pp-ocrv6-medium-det-onnx

```bash
docker run -d --name paddlepaddle-pp-ocrv6-medium-det-onnx --gpus all -p 8080:8080 -v "d:/ssd-projects/Models-Deploy/models/paddlepaddle-pp-ocrv6-medium-det-onnx:/workspace/models/paddlepaddle-pp-ocrv6-medium-det-onnx" -w "/workspace/models/paddlepaddle-pp-ocrv6-medium-det-onnx" models-deploy-base:latest bash start.sh
```

```powershell
python d:/ssd-projects/Models-Deploy/models/paddlepaddle-pp-ocrv6-medium-det-onnx/test/test.py
```

```bash
docker stop paddlepaddle-pp-ocrv6-medium-det-onnx && docker rm paddlepaddle-pp-ocrv6-medium-det-onnx
```

---

## 31. paddlepaddle-pp-ocrv6-small-det-onnx

```bash
docker run -d --name paddlepaddle-pp-ocrv6-small-det-onnx --gpus all -p 8080:8080 -v "d:/ssd-projects/Models-Deploy/models/paddlepaddle-pp-ocrv6-small-det-onnx:/workspace/models/paddlepaddle-pp-ocrv6-small-det-onnx" -w "/workspace/models/paddlepaddle-pp-ocrv6-small-det-onnx" models-deploy-base:latest bash start.sh
```

```powershell
python d:/ssd-projects/Models-Deploy/models/paddlepaddle-pp-ocrv6-small-det-onnx/test/test.py
```

```bash
docker stop paddlepaddle-pp-ocrv6-small-det-onnx && docker rm paddlepaddle-pp-ocrv6-small-det-onnx
```

---

## 32. paddlepaddle-pp-ocrv6-small-rec-onnx

```bash
docker run -d --name paddlepaddle-pp-ocrv6-small-rec-onnx --gpus all -p 8080:8080 -v "d:/ssd-projects/Models-Deploy/models/paddlepaddle-pp-ocrv6-small-rec-onnx:/workspace/models/paddlepaddle-pp-ocrv6-small-rec-onnx" -w "/workspace/models/paddlepaddle-pp-ocrv6-small-rec-onnx" models-deploy-base:latest bash start.sh
```

```powershell
python d:/ssd-projects/Models-Deploy/models/paddlepaddle-pp-ocrv6-small-rec-onnx/test/test.py
```

```bash
docker stop paddlepaddle-pp-ocrv6-small-rec-onnx && docker rm paddlepaddle-pp-ocrv6-small-rec-onnx
```

---

## 33. prior-labs-tabpfn-v2-clf

```bash
docker run -d --name prior-labs-tabpfn-v2-clf --gpus all -p 8080:8080 -v "d:/ssd-projects/Models-Deploy/models/prior-labs-tabpfn-v2-clf:/workspace/models/prior-labs-tabpfn-v2-clf" -w "/workspace/models/prior-labs-tabpfn-v2-clf" models-deploy-base:latest bash start.sh
```

```powershell
python d:/ssd-projects/Models-Deploy/models/prior-labs-tabpfn-v2-clf/test/test.py
```

```bash
docker stop prior-labs-tabpfn-v2-clf && docker rm prior-labs-tabpfn-v2-clf
```

---

## 34. prior-labs-tabpfn-v2-reg

```bash
docker run -d --name prior-labs-tabpfn-v2-reg --gpus all -p 8080:8080 -v "d:/ssd-projects/Models-Deploy/models/prior-labs-tabpfn-v2-reg:/workspace/models/prior-labs-tabpfn-v2-reg" -w "/workspace/models/prior-labs-tabpfn-v2-reg" models-deploy-base:latest bash start.sh
```

```powershell
python d:/ssd-projects/Models-Deploy/models/prior-labs-tabpfn-v2-reg/test/test.py
```

```bash
docker stop prior-labs-tabpfn-v2-reg && docker rm prior-labs-tabpfn-v2-reg
```

---

## 35. skywork-skywork-reward-v2-qwen3-0-6b

```bash
docker run -d --name skywork-skywork-reward-v2-qwen3-0-6b --gpus all -p 8080:8080 -v "d:/ssd-projects/Models-Deploy/models/skywork-skywork-reward-v2-qwen3-0-6b:/workspace/models/skywork-skywork-reward-v2-qwen3-0-6b" -w "/workspace/models/skywork-skywork-reward-v2-qwen3-0-6b" models-deploy-base:latest bash start.sh
```

```powershell
python d:/ssd-projects/Models-Deploy/models/skywork-skywork-reward-v2-qwen3-0-6b/test/test.py
```

```bash
docker stop skywork-skywork-reward-v2-qwen3-0-6b && docker rm skywork-skywork-reward-v2-qwen3-0-6b
```

---

## 36. synthefy-nori-30m

```bash
docker run -d --name synthefy-nori-30m --gpus all -p 8080:8080 -v "d:/ssd-projects/Models-Deploy/models/synthefy-nori-30m:/workspace/models/synthefy-nori-30m" -w "/workspace/models/synthefy-nori-30m" models-deploy-base:latest bash start.sh
```

```powershell
python d:/ssd-projects/Models-Deploy/models/synthefy-nori-30m/test/test.py
```

```bash
docker stop synthefy-nori-30m && docker rm synthefy-nori-30m
```

---

## 37. voyageai-voyage-4-nano

```bash
docker run -d --name voyageai-voyage-4-nano --gpus all -p 8080:8080 -v "d:/ssd-projects/Models-Deploy/models/voyageai-voyage-4-nano:/workspace/models/voyageai-voyage-4-nano" -w "/workspace/models/voyageai-voyage-4-nano" models-deploy-base:latest bash start.sh
```

```powershell
python d:/ssd-projects/Models-Deploy/models/voyageai-voyage-4-nano/test/test.py
```

```bash
docker stop voyageai-voyage-4-nano && docker rm voyageai-voyage-4-nano
```

---

## 38. weborganizer-topicclassifier-nourl

```bash
docker run -d --name weborganizer-topicclassifier-nourl --gpus all -p 8080:8080 -v "d:/ssd-projects/Models-Deploy/models/weborganizer-topicclassifier-nourl:/workspace/models/weborganizer-topicclassifier-nourl" -w "/workspace/models/weborganizer-topicclassifier-nourl" models-deploy-base:latest bash start.sh
```

```powershell
python d:/ssd-projects/Models-Deploy/models/weborganizer-topicclassifier-nourl/test/test.py
```

```bash
docker stop weborganizer-topicclassifier-nourl && docker rm weborganizer-topicclassifier-nourl
```

---

## 39. yuchenshen-fomo-0d

```bash
docker run -d --name yuchenshen-fomo-0d --gpus all -p 8080:8080 -v "d:/ssd-projects/Models-Deploy/models/yuchenshen-fomo-0d:/workspace/models/yuchenshen-fomo-0d" -w "/workspace/models/yuchenshen-fomo-0d" models-deploy-base:latest bash start.sh
```

```powershell
python d:/ssd-projects/Models-Deploy/models/yuchenshen-fomo-0d/test/test.py
```

```bash
docker stop yuchenshen-fomo-0d && docker rm yuchenshen-fomo-0d
```

---

## 快速健康检查

容器启动后等待就绪，然后验证：

```powershell
# 健康检查
curl http://localhost:8080/health

# 查看容器日志（确认模型加载完成）
docker logs -f {模型名}
```

## 常见问题

- **端口占用**：`docker: Error response from daemon: driver failed programming external connectivity` → 先 `docker stop` 上一个测试容器
- **容器名冲突**：`Conflict. The container name ... is already in use` → 先 `docker rm` 同名容器
- **服务启动失败**：`docker logs {模型名}` 查看报错；若 venv 相关报错，确认挂载路径是 `/workspace/models/{模型名}`（venv 硬编码了该路径）
