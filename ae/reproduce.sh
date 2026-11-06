#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
VENV=${AE_VENV:-"$ROOT/.venv-ae"}
PROFILE=${1:-smoke}
MODEL=${AE_MODEL:-"$ROOT/ae/models/LLaDA2.0-mini"}
OUTPUT=${AE_OUTPUT:-"$ROOT/ae/results/${PROFILE}-$(date -u +%Y%m%dT%H%M%SZ)"}
if [[ ! -x "$VENV/bin/python" ]]; then
  bash ae/setup.sh
fi
if [[ "${AE_SKIP_PREPARE:-0}" != 1 ]]; then
  if [[ -f "$MODEL/model.safetensors.index.json" ]]; then
    "$VENV/bin/python" ae/prepare.py --local-model --model "$MODEL"
  else
    "$VENV/bin/python" ae/prepare.py --model "$MODEL"
  fi
fi
export PYTHONPATH="$ROOT/python"
"$VENV/bin/python" -m pytest -q ae/tests
run_stage() {
  local stage=$1
  local destination=$2
  case "$stage" in
    trace)
      # 20 req/s over 300 records: below roughly 12 req/s the replay is
      # arrival-limited on A100 and the throughput gain disappears entirely.
      "$VENV/bin/python" ae/trace.py --model "$MODEL" --max-requests 300 --rate 20 --batch 64 \
        --gpus "${AE_GPUS:-0,1}" --tp "${AE_TP:-2}" --port "${AE_PORT:-30174}" --output "$destination" ;;
    sweep)
      "$VENV/bin/python" ae/sweep.py --model "$MODEL" \
        --gpus "${AE_GPUS:-0,1}" --tp "${AE_TP:-2}" --port "${AE_PORT:-30174}" --output "$destination"
      "$VENV/bin/python" ae/plot_sweep.py "$destination" ;;
    quality)
      "$VENV/bin/python" ae/quality.py --model "$MODEL" --samples "${AE_QUALITY_SAMPLES:-200}" \
        --gpus "${AE_GPUS:-0,1}" --tp "${AE_TP:-2}" --port "${AE_PORT:-30174}" --output "$destination" ;;
    smoke|quick|full)
      "$VENV/bin/python" ae/run.py --model "$MODEL" --profile "$stage" \
        --gpus "${AE_GPUS:-0,1}" --tp "${AE_TP:-2}" --port "${AE_PORT:-30174}" --output "$destination" ;;
    *) echo "Unknown profile: $stage (smoke, quick, full, sweep, trace, quality, all)" >&2; exit 2 ;;
  esac
}
if [[ "$PROFILE" == all ]]; then
  for stage in quick sweep trace quality; do
    run_stage "$stage" "$OUTPUT/$stage"
  done
else
  run_stage "$PROFILE" "$OUTPUT"
fi
echo "Results: $OUTPUT"
