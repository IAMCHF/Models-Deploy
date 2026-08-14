#!/bin/bash
echo "=== Models with torch conflicts ==="
for d in /workspace/models/*/venv; do
  model=$(basename $(dirname $d))
  vfile="$d/lib/python3.10/site-packages/torch/version.py"
  if [ -f "$vfile" ]; then
    ver=$(python3 -c "exec(open('$vfile').read()); print(__version__)" 2>/dev/null)
    if [ "$ver" != "2.7.0+cu126" ] && [ -n "$ver" ]; then
      echo "$model: torch=$ver"
    fi
  fi
done

echo ""
echo "=== Models with torchvision conflicts ==="
for d in /workspace/models/*/venv; do
  model=$(basename $(dirname $d))
  vfile="$d/lib/python3.10/site-packages/torchvision/version.py"
  if [ -f "$vfile" ]; then
    ver=$(python3 -c "exec(open('$vfile').read()); print(__version__)" 2>/dev/null)
    if [ "$ver" != "0.22.0+cu126" ] && [ -n "$ver" ]; then
      echo "$model: torchvision=$ver"
    fi
  fi
done

echo ""
echo "=== Checking transformers conflicts ==="
SYSTF=$(python3 -c "import transformers; print(transformers.__version__)" 2>/dev/null)
echo "System transformers: $SYSTF"
for d in /workspace/models/*/venv; do
  model=$(basename $(dirname $d))
  vfile="$d/lib/python3.10/site-packages/transformers/__init__.py"
  if [ -f "$vfile" ]; then
    cd /workspace/models/$model
    source venv/bin/activate
    vtf=$(python -c "import transformers; print(transformers.__version__)" 2>/dev/null)
    if [ -n "$vtf" ] && [ "$vtf" != "$SYSTF" ]; then
      echo "$model: venv transformers=$vtf (system=$SYSTF)"
    fi
    deactivate
  fi
done
