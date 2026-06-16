#!/bin/bash
# uninstall.sh — Restore original SGLang dLLM files
set -e

SGLANG_PATH=$(python -c "import sglang; import os; print(os.path.dirname(sglang.__path__[0]))" 2>/dev/null) || true
if [ -z "$SGLANG_PATH" ]; then
    echo "SGLang not found"; exit 1
fi

SGLANG_DLLM="$SGLANG_PATH/sglang/srt/dllm"

echo "Restoring originals..."
for f in mixin/req.py mixin/scheduler.py; do
    if [ -f "$SGLANG_DLLM/${f}.bak" ]; then
        mv "$SGLANG_DLLM/${f}.bak" "$SGLANG_DLLM/${f}"
        echo "  Restored $f"
    fi
done

# Remove added algorithm files
rm -f "$SGLANG_DLLM/algorithm/cw_srpt_v2.py"
rm -f "$SGLANG_DLLM/algorithm/cw_srpt.py"
echo "  Removed CW-SRPT algorithms"

echo "✅ Uninstall complete. SGLang restored to original state."
