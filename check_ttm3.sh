#!/bin/bash
echo "=== ttm-r3 venv transformers ==="
cd /workspace/models/ibm-research-ttm-r3
source venv/bin/activate
python -c "import transformers; print('venv transformers:', transformers.__version__)"
python -c "from transformers import TinyTimeMixerForPrediction; print('TinyTimeMixer OK')" 2>&1 | tail -1
python -c "import transformers; print([x for x in dir(transformers) if 'TinyTime' in x])" 2>&1
deactivate
echo ""
echo "=== granite venv transformers ==="
cd /workspace/models/ibm-granite-granite-timeseries-patchtst-fm-r1
source venv/bin/activate
python -c "import transformers; print('venv transformers:', transformers.__version__)"
python -c "from transformers import PatchTSTForPrediction; print('PatchTST OK')" 2>&1 | tail -1
deactivate
