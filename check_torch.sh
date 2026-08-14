#!/bin/bash
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
