# run_package.ps1 — 宿主机入口: 在 models-deploy-base 容器内执行打包
# 用法:
#   .\run_package.ps1                      # 打包全部模型
#   .\run_package.ps1 jhu-clsp-mmbert-base  # 只打包指定模型
# 产物: .\packages\<model>.tar.xz  + .\packages\manifest\
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$modelArgs = ($args -join " ")
docker run --rm `
  -e XZ_THREADS=0 `
  -e XZ_LEVEL=-6 `
  -e OUT_DIR=/packages_out `
  -v "${root}:/workspace" `
  -v "F:\model-packages:/packages_out" `
  models-deploy-base:latest `
  bash -c "dos2unix -q /workspace/package_models.sh 2>/dev/null || true; bash /workspace/package_models.sh $modelArgs"
