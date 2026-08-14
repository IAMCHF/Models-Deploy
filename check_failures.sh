#!/bin/bash
# Check failure logs for all failed models

FAILED_MODELS=(
  "facebook-vjepa2-vitl-fpc64-256"
  "jusperlee-tiger-dnr"
  "koreapeter-ms-eff-gcvit-deepfake-b0-ff-plus-plus"
  "microsoft-vibevoice-asr-hf"
  "openmoss-team-moss-tts-local-transformer-v1-5"
  "openmoss-team-moss-voicegenerator"
  "paddlepaddle-pp-chart2table"
  "paddlepaddle-pp-doclayout-plus-l"
  "voyageai-voyage-4-nano"
)

for model in "${FAILED_MODELS[@]}"; do
  echo "========================================"
  echo "=== $model ==="
  echo "========================================"
  echo "--- startup log (last 15 lines) ---"
  tail -15 /tmp/${model}_test.log 2>/dev/null || echo "no startup log"
  echo ""
  echo "--- test result log ---"
  cat /tmp/${model}_test_result.log 2>/dev/null || echo "no test result log"
  echo ""
  echo ""
done
