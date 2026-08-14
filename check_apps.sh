#!/bin/bash
for model in neoquasar-kronos-base datadog-toto-2-0-22m ibm-research-ttm-r3 ibm-granite-granite-timeseries-patchtst-fm-r1; do
  echo "========================================"
  echo "=== $model ==="
  echo "========================================"
  echo "--- app.py full imports ---"
  grep -nE "^(from|import) " /workspace/models/$model/app.py 2>/dev/null | head -30
  echo ""
  echo "--- app.py head (first 40 lines) ---"
  head -40 /workspace/models/$model/app.py 2>/dev/null
  echo ""
done
