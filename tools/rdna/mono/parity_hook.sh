#!/usr/bin/env bash
# LOAD_HOOK: greedy top-5 logprob dump for the off/on parity check, plus the
# mono path decisions from the serve log. Exported: URL PORT SERVED MODEL OUT.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
"${VENV:-$HOME/Apps/vllm/venv-7.14.0_0.31.0-final}/bin/python" "$HERE/logprob_parity.py" \
    dump --url "http://127.0.0.1:$PORT" --model "$SERVED" --out "$OUT/parity.json" || { echo FAIL; exit 1; }
grep -aE "mono (rdna2|k1_rdna2|k2_rdna2|gdn_rdna2)" "$OUT/serve.log" \
    | sed -E 's/^.*(mono (rdna2|k1_rdna2|k2_rdna2|gdn_rdna2))/\1/' | sort | uniq -c | head -60
echo PASS
