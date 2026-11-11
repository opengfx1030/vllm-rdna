#!/usr/bin/env bash
# LOAD_HOOK adapter; runs clients only against the existing isolated server.
set -euo pipefail
TREE=$(cd "$(dirname "$0")/../../.." && pwd)
VENV=${VENV:-$TREE/.venv}
# LONG_CELLS can select a bounded continuation, e.g. 32k-c1 after a saved 16k run.
cell_args=()
if [[ -n "${LONG_CELLS:-}" ]]; then
    read -r -a selected_cells <<< "$LONG_CELLS"
    cell_args=(--cells "${selected_cells[@]}")
fi
exec "$VENV/bin/python" "$TREE/tools/rdna/mono/qualify_offload_long.py" "${cell_args[@]}" "$@"
