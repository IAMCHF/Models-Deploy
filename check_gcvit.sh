#!/bin/bash
cd /workspace/models/koreapeter-ms-eff-gcvit-deepfake-b0-ff-plus-plus
source venv/bin/activate
echo "=== Check deepguard ==="
python -c "import deepguard; print('deepguard OK:', deepguard.__file__)" 2>&1 | tail -2
echo ""
echo "=== Check configuration_ms_eff_gcvit ==="
python -c "import sys; sys.path.insert(0, 'weights'); from configuration_ms_eff_gcvit import MsEffGCViTConfig; print('config OK')" 2>&1 | tail -2
echo ""
echo "=== Check transformers version ==="
python -c "import transformers; print('transformers:', transformers.__version__)"
echo ""
echo "=== Check check_imports logic ==="
python << 'EOF'
import inspect
from transformers.dynamic_module_utils import check_imports
src = inspect.getsource(check_imports)
print(src[:2500])
EOF
deactivate
