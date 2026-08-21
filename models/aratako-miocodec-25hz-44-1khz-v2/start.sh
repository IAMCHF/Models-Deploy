#!/bin/bash
# ============================================================
# start.sh - 鍚姩 FastAPI 鏈嶅姟锛堢鍙?8080锛?# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 浣跨敤绯荤粺 Python锛?usr/local/bin/python3锛?# 娉ㄦ剰锛氳櫄鎷熺幆澧冪殑 python 鍙墽琛屾枃浠跺彲鑳芥崯鍧忥紝鐩存帴浣跨敤绯荤粺 Python

# 璁剧疆 PYTHONPATH 鍖呭惈铏氭嫙鐜鐨?site-packages
# 纭繚 python3 鎸囧悜 3.10锛堢郴缁熷寘瀹夎鍦?3.10 涓嬶級
if [ -x /usr/bin/python3.10 ]; then
    ln -sf /usr/bin/python3.10 /usr/local/bin/python3
    ln -sf /usr/bin/python3.10 /usr/local/bin/python
fi

export PYTHONPATH="$SCRIPT_DIR/venv/lib/python3.10/site-packages"

# 闀滃儚绔欎紭鍏?export HF_ENDPOINT="https://hf-mirror.com"

# torchaudio 缂撳瓨鎸囧悜鏈湴鏉冮噸鐩綍锛堢绾垮姞杞?wavlm 绛夐璁粌鏉冮噸锛?export TORCH_HOME="$SCRIPT_DIR/weights/torch_home"

echo "[start] 鍚姩鏈嶅姟锛?(basename $SCRIPT_DIR) (绔彛 8080)"
cd /tmp && /usr/local/bin/python3 "$SCRIPT_DIR/app.py"