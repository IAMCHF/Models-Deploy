#!/bin/bash
set -e
if [ -x /usr/bin/python3.10 ]; then
    ln -sf /usr/bin/python3.10 /usr/local/bin/python3
    ln -sf /usr/bin/python3.10 /usr/local/bin/python
fi
export PIP_CACHE_DIR=/workspace/.pip_cache
cd /workspace/models/numind-nuextract3-fp8
./venv/bin/pip install --quiet "compressed-tensors>=0.15.0" 2>&1 | tail -2
./venv/bin/pip show compressed-tensors 2>/dev/null | grep -E "^(Name|Version|Location)"
