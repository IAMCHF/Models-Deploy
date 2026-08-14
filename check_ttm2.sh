#!/bin/bash
echo "=== System transformers version ==="
python3 -c "import transformers; print(transformers.__version__)"
echo "=== Check TinyTimeMixer ==="
python3 -c "from transformers import TinyTimeMixerForPrediction; print('TinyTimeMixer OK')" 2>&1 | tail -1
echo "=== Check available TTM classes ==="
python3 -c "import transformers; print([x for x in dir(transformers) if 'TinyTime' in x])" 2>&1
