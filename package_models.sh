#!/bin/bash
# ============================================================
# package_models.sh — 内网部署打包（在 models-deploy-base 容器内运行, 见 run_package.ps1）
#
# 打包规则:
#   1. 每个模型一个 <model>.tar.xz（xz -9 多线程, 高压缩比）
#   2. 仅剔除 weights/ 目录内的"主模型权重"文件（按扩展名判定,
#      内网可通过 HF 代理重新下载）
#   3. 白名单保护辅助权重, 绝不剔除:
#      - weights/torch_home/... (aratako-miocodec 的 WavLM 特征提取器)
#      - weights/yolov8n-face.pt (koreapeter b0/b5 的人脸检测辅助模型)
#      - weights/docling-project--docling-layout-heron/ (docling 版面分析子模型)
#   4. 模型目录下其余资产全量保留: hf_cache/, weights_tokenizer/,
#      weights_audio_tokenizer/, test/, 自定义代码, tokenizer/配置文件等
#   5. venv/ 完整打包(仅去掉 __pycache__ 与 *.pyc, 部署后可自动重建)
#   6. 每个包在 packages/manifest/ 生成排除清单, 汇总到 PACKAGES.tsv
#
# 用法:
#   bash package_models.sh                # 打包全部 39 个模型
#   bash package_models.sh model-a model-b # 只打包指定模型
# 环境变量:
#   XZ_THREADS=4  XZ_LEVEL=-9  SKIP_DONE=1
# ============================================================
set -euo pipefail

MODELS_ROOT="/workspace/models"
OUT="${OUT_DIR:-/workspace/packages}"   # 产物目录(默认项目内, 可通过 OUT_DIR 指定到其他挂载盘)
MANIFEST="$OUT/manifest"
XZ_THREADS="${XZ_THREADS:-0}"   # 0 = 自动使用全部 CPU 核心
XZ_LEVEL="${XZ_LEVEL:--6}"      # -6 性价比最优(比 -9 小差距 1~3%, 速度快数倍)

mkdir -p "$MANIFEST"
[ -f "$MANIFEST/PACKAGES.tsv" ] || printf 'model\tfiles\texcluded_bytes\tarchive_bytes\tsha256\n' > "$MANIFEST/PACKAGES.tsv"

# weights/ 下视为"主模型权重"的扩展名（剔除, 由内网代理重新下载）
WEIGHT_EXTS_RE='\.(safetensors|bin|pt|pth|ckpt|cpkt|pdparams|pdiparams|onnx|npz|gguf|msgpack|h5)$'

# 辅助权重白名单: 命中则不剔除（参数为 models/ 下的相对路径）
is_aux_keep() {
  case "$1" in
    aratako-miocodec-25hz-44-1khz-v2/weights/torch_home/*) return 0 ;;
    koreapeter-ms-eff-gcvit-deepfake-b0-ff-plus-plus/weights/yolov8n-face.pt) return 0 ;;
    koreapeter-ms-eff-gcvit-deepfake-b5-ff-plus-plus/weights/yolov8n-face.pt) return 0 ;;
    docling-project-codeformulav2/weights/docling-project--docling-layout-heron/*) return 0 ;;
  esac
  return 1
}

is_excluded() {
  # $1: models/ 下相对路径; 返回 0 表示剔除(主权重)
  case "$1" in
    */__pycache__/*|*.pyc|*/.pytest_cache/*|*/.ipynb_checkpoints/*) return 0 ;;
  esac
  if [[ "$1" == */weights/* && "$1" =~ $WEIGHT_EXTS_RE ]] && ! is_aux_keep "$1"; then
    return 0
  fi
  return 1
}

pack_model() {
  local model="$1"
  local src="$MODELS_ROOT/$model"
  if [ ! -d "$src" ]; then echo "[skip] $model: 目录不存在"; return 0; fi
  if [ ! -f "$src/start.sh" ]; then echo "[skip] $model: 无 start.sh, 非模型目录"; return 0; fi

  local archive="$OUT/$model.tar.xz"
  local excl_report="$MANIFEST/$model.excluded.txt"
  if [ "${SKIP_DONE:-1}" = "1" ] && [ -f "$archive.done" ]; then
    echo "[skip] $model: 已打包 ($archive)"
    return 0
  fi

  local list0
  list0=$(mktemp)
  echo "==> 打包 $model"
  : > "$excl_report"

  (
    cd "$MODELS_ROOT"
    find "$model" \( -type f -o -type l \) -print0 | while IFS= read -r -d '' p; do
      if is_excluded "$p"; then
        sz=0
        [ -f "$p" ] && sz=$(stat -c %s "$p" 2>/dev/null || echo 0)
        printf '%12d  %s\n' "$sz" "$p" >> "$excl_report"
        continue
      fi
      printf '%s\0' "$p"
    done > "$list0"
  )

  local n_files esz asz sha
  n_files=$(tr -cd '\0' < "$list0" | wc -c)
  tar -C "$MODELS_ROOT" --null --no-recursion -T "$list0" -cf - \
    | xz "$XZ_LEVEL" -T "$XZ_THREADS" > "$archive"
  rm -f "$list0"

  asz=$(stat -c %s "$archive")
  esz=$(awk '{s+=$1} END {printf "%.0f", s+0}' "$excl_report")
  sha=$(sha256sum "$archive" | cut -d' ' -f1)
  sed -i "/^${model}\t/d" "$MANIFEST/PACKAGES.tsv"
  printf '%s\t%d\t%d\t%d\t%s\n' "$model" "$n_files" "$esz" "$asz" "$sha" >> "$MANIFEST/PACKAGES.tsv"
  : > "$archive.done"
  echo "    完成: $((asz/1024/1024)) MB (剔除主权重 $((esz/1024/1024)) MB, 打包文件数 $n_files)"
}

pack_root_scripts() {
  # 仓库根目录的部署脚本/文档, 打成一个公共包, 解压到项目根目录即可
  local archive="$OUT/00-common-scripts.tar.xz"
  local list0
  list0=$(mktemp)
  (
    cd /workspace
    find . -maxdepth 1 -type f \
      \( -name "*.sh" -o -name "*.py" -o -name "*.md" -o -name "*.json" \
         -o -name "Dockerfile" -o -name ".gitignore" \) -print0 > "$list0"
  )
  tar -C /workspace --null --no-recursion -T "$list0" -cf - \
    | xz "$XZ_LEVEL" -T "$XZ_THREADS" > "$archive"
  local asz sha
  asz=$(stat -c %s "$archive")
  sha=$(sha256sum "$archive" | cut -d' ' -f1)
  rm -f "$list0"
  sed -i "/^00-common-scripts\t/d" "$MANIFEST/PACKAGES.tsv"
  printf '00-common-scripts\t0\t0\t%d\t%s\n' "$asz" "$sha" >> "$MANIFEST/PACKAGES.tsv"
  echo "==> 公共脚本包完成: $((asz/1024/1024)) MB"
}

shopt -s nullglob
if [ "$#" -gt 0 ]; then
  for m in "$@"; do pack_model "$m"; done
else
  pack_root_scripts
  for d in "$MODELS_ROOT"/*/; do
    m=$(basename "$d")
    if [ -f "$d/start.sh" ]; then
      pack_model "$m"
    fi
  done
fi

echo ""
echo "全部完成。产物目录: $OUT"
echo "汇总清单: $MANIFEST/PACKAGES.tsv"
