#!/usr/bin/env bash
# Shell adapter for serve_validate.sh's LOAD_HOOK contract.
set -euo pipefail
TREE=$(cd "$(dirname "$0")/../../.." && pwd)
VENV=${VENV:-$TREE/.venv}
OUT=${OUT:-$HOME/w4a8_runs/moe-offload}
mkdir -p "$OUT"
# Controlled transfer diagnostic: remove different restored-prefix lengths as
# an arithmetic confound. Requires --enable-prompt-tokens-details on the server.
# Leave unset for the ordinary, unconstrained offload qualification.
probe_args=()
if [[ ${KV_MATCH_RELOAD_BOUNDARY:-0} == 1 ]]; then
    probe_args+=(--match-reload-boundary)
fi
# Native Mamba lookup subtracts one before block alignment. Optional slack
# compensates its request cap, while the probe still requires exact hit usage.
if [[ ${KV_RELOAD_CAP_SLACK:-0} != 0 ]]; then
    probe_args+=(--reload-cap-slack "$KV_RELOAD_CAP_SLACK")
fi
if "$VENV/bin/python" "$TREE/tools/rdna/mono/probe_offload_kv.py" \
    --base-url "http://127.0.0.1:${PORT:-18766}" \
    --model "${SERVED:-flash-next}" --out "$OUT/kv-proof.json" \
    "${probe_args[@]}" "$@"; then
    echo 'PASS: native KV store/reload and output parity'
else
    echo 'FAIL: native KV store/reload proof'
    exit 1
fi
