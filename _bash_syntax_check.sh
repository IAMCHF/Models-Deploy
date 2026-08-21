#!/bin/bash
cd /workspace/models
ok=0; bad=0
for f in */start.sh; do
  if bash -n "$f" 2>/dev/null; then ok=$((ok+1)); else echo "语法错误: $f"; bad=$((bad+1)); fi
done
echo "语法OK: $ok, 失败: $bad"
file aratako-miocodec-25hz-44-1khz-v2/start.sh