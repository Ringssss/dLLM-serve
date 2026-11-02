#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${AE_VENV:-"$ROOT/.venv-ae"}
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install 'uv==0.13.6'
if [[ -f "$ROOT/ae/requirements.lock" ]]; then
  "$VENV/bin/uv" pip sync --python "$VENV/bin/python" "$ROOT/ae/requirements.lock"
else
  "$VENV/bin/uv" pip install --python "$VENV/bin/python" -r "$ROOT/python/pyproject.toml" matplotlib pyyaml pytest
fi
echo "Environment: $VENV"
echo "Use: export PYTHONPATH=$ROOT/python"
