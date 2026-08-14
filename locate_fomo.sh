#!/bin/bash
ln -sf /usr/bin/python3.10 /usr/local/bin/python3
ln -sf /usr/bin/python3.10 /usr/local/bin/python
echo "=== fomo_hub 可导入性 ==="
/workspace/models/yuchenshen-fomo-0d/venv/bin/python -c "import fomo_hub; print('文件:', fomo_hub.__file__); from fomo_hub import FoMo0DHub; print('FoMo0DHub: OK')" 2>&1 | grep -vE "Warning|warn"
echo "=== app.py 同目录(cwd=模型目录)导入 ==="
cd /workspace/models/yuchenshen-fomo-0d && ./venv/bin/python -c "import fomo_hub; print('cwd导入 OK:', fomo_hub.__file__)" 2>&1 | grep -vE "Warning|warn"
