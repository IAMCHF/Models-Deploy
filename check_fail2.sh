#!/bin/bash
for model in voyageai-voyage-4-nano paddlepaddle-pp-chart2table opengvlab-videomaev2-base; do
  echo "========================================"
  echo "=== $model ==="
  echo "========================================"
  echo "--- startup log (last 20 lines) ---"
  tail -20 /tmp/${model}_retest2.log 2>/dev/null
  echo ""
  echo "--- test result log ---"
  cat /tmp/${model}_retest2_result.log 2>/dev/null
  echo ""
  echo ""
done
