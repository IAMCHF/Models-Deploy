#!/bin/bash
cd /workspace/models
MODELS=""
for d in */; do
  m="${d%/}"
  if [ -f "$m/openapi.json" ]; then
    MODELS="$MODELS $m"
  fi
done
python3 /tmp/verify_openapi.py $MODELS
