#!/bin/bash
# install.sh — One-command setup for dLLM-serve CW-SRPT V2
# Usage: bash install.sh [--sglang-path /path/to/sglang]
set -e

SGLANG_PATH=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --sglang-path) SGLANG_PATH="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Auto-detect SGLang path
if [ -z "$SGLANG_PATH" ]; then
    SGLANG_PATH=$(python -c "import sglang; import os; print(os.path.dirname(sglang.__path__[0]))" 2>/dev/null) || true
    if [ -z "$SGLANG_PATH" ]; then
        echo "ERROR: SGLang not found. Install it first:"
        echo "  pip install sglang[all]"
        echo "Or specify path: bash install.sh --sglang-path /path/to/sglang/python"
        exit 1
    fi
fi

SGLANG_DLLM="$SGLANG_PATH/sglang/srt/dllm"
echo "SGLang dLLM path: $SGLANG_DLLM"

if [ ! -d "$SGLANG_DLLM" ]; then
    echo "ERROR: $SGLANG_DLLM not found. Is SGLang installed with dLLM support?"
    exit 1
fi

# Backup originals
echo "Backing up original files..."
cp "$SGLANG_DLLM/mixin/req.py" "$SGLANG_DLLM/mixin/req.py.bak" 2>/dev/null || true
cp "$SGLANG_DLLM/mixin/scheduler.py" "$SGLANG_DLLM/mixin/scheduler.py.bak" 2>/dev/null || true

# Install CW-SRPT V2 algorithm
echo "Installing CW-SRPT V2 algorithm..."
# Install CW-SRPT V3/V2/V1 algorithms
echo "Installing CW-SRPT algorithms (V3/V2/V1)..."
cp cwsrpt_v2/algorithm/cw_srpt_v3.py "$SGLANG_DLLM/algorithm/"
cp cwsrpt_v2/algorithm/cw_srpt_v2.py "$SGLANG_DLLM/algorithm/"
cp cwsrpt_v2/algorithm/cw_srpt.py "$SGLANG_DLLM/algorithm/"

# Install mixin patches (frontier state + frontier-aware admission + aging)
echo "Installing scheduler patches..."
cp cwsrpt_v2/mixin/req.py "$SGLANG_DLLM/mixin/"
cp cwsrpt_v2/mixin/scheduler.py "$SGLANG_DLLM/mixin/"

echo ""
echo "✅ Installation complete!"
echo ""
echo "Verify:"
echo "  python -c \"from sglang.srt.dllm.algorithm.cw_srpt_v2 import CW_SRPT_V2; print('OK')\""
echo ""
echo "Launch:"
echo "  python -m sglang.launch_server \\"
echo "    --model-path /path/to/LLaDA2.0-mini \\"
echo "    --dllm-algorithm CW_SRPT_V2 \\"
echo "    --max-running-requests 8 \\"
echo "    --disable-radix-cache \\"
echo "    --trust-remote-code \\"
echo "    --port 30100"
