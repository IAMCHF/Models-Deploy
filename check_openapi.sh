#!/bin/bash
count=0
for f in /workspace/models/*/openapi.json; do
    count=$((count+1))
done
echo "生成文件数: $count"
echo "=== 各文件大小 ==="
ls -la /workspace/models/*/openapi.json 2>/dev/null | awk '{print $5, $9}'
echo "=== 缺失的模型(有app.py但无openapi.json) ==="
for d in /workspace/models/*/; do
    if [ -f "$d/app.py" ] && [ ! -f "$d/openapi.json" ]; then
        basename "$d"
    fi
done
