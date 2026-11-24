#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Boot the HI2 directory, wait for the server, run the language-only greedy probe and
# print the engine's load/KV/graph lines.  With STOCK_URL set, also compare greedy
# token streams and (RUN_KL=1) the held-out KL against the stock server.
#
#   MODEL=/path/to/bpw3.5 HIP_VISIBLE_DEVICES=0,1 VENV=/path/to/venv \
#       bash tools/inkling_hi2/smoke.sh
#
# Optional: STOCK_URL=http://127.0.0.1:18231 (a stock-model server), RUN_KL=1,
#           IDS=$HOME/w4a8_runs/inkling-hi2/stage4/calib/heldout_mixed_ids.npy,
#           WINDOWS=32, KEEP=1 (leave the server up), MAXLEN/MAXBAT/SEQS/EAGER.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PORT=${PORT:-18230}
URL="http://127.0.0.1:$PORT"
RUN_DIR=${RUN_DIR:-$HOME/w4a8_runs/inkling-hi2-runtime}
LOG=${LOG:-$RUN_DIR/logs/serve-$(date +%Y%m%d-%H%M%S).log}
KEEP=${KEEP:-0}
WAIT_SECONDS=${WAIT_SECONDS:-2400}
mkdir -p "$(dirname "$LOG")"

PORT="$PORT" bash "$HERE/serve.sh" >"$LOG" 2>&1 &
SERVER=$!
cleanup() {
    if [[ $KEEP != 1 ]]; then
        kill "$SERVER" 2>/dev/null || true
        wait "$SERVER" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "smoke.sh: waiting for $URL/health (log $LOG)"
for _ in $(seq 1 $((WAIT_SECONDS / 5))); do
    if ! kill -0 "$SERVER" 2>/dev/null; then
        echo "smoke.sh: the server died during boot; last lines:" >&2
        tail -60 "$LOG" >&2
        exit 1
    fi
    if curl -sf "$URL/health" >/dev/null 2>&1; then
        break
    fi
    sleep 5
done
curl -sf "$URL/health" >/dev/null || {
    echo "smoke.sh: no /health after $WAIT_SECONDS s; last lines:" >&2
    tail -80 "$LOG" >&2
    exit 1
}

python3 - "$URL" <<'PY'
import json, sys, time, urllib.request

url = sys.argv[1]

def call(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=1200) as r:
        return json.loads(r.read())

models = call("/v1/models")
name = models["data"][0]["id"]
print(f"greedy: served={name} max_model_len={models['data'][0].get('max_model_len')}")
t0 = time.time()
out = call(
    "/v1/chat/completions",
    {
        "model": name,
        "messages": [
            {
                "role": "user",
                "content": "What is the capital of France? Reply with the city name.",
            }
        ],
        "max_tokens": 32,
        "temperature": 0.0,
    },
)
text = out["choices"][0]["message"].get("content") or ""
elapsed = time.time() - t0
print(f"greedy: {text!r} ({elapsed:.2f} s, {out['usage']})")
assert "paris" in text.lower(), "the greedy answer does not name Paris"
print("RESULT greedy PASS")
PY

if [[ -n ${STOCK_URL:-} ]]; then
    python3 "$HERE/kl_probe.py" \
        --hi2-url "$URL" --stock-url "$STOCK_URL" \
        --ids "${IDS:-$HOME/w4a8_runs/inkling-hi2/stage4/calib/heldout_mixed_ids.npy}" \
        --windows "${WINDOWS:-32}" --greedy
    if [[ ${RUN_KL:-0} == 1 ]]; then
        python3 "$HERE/kl_probe.py" \
            --hi2-url "$URL" --stock-url "$STOCK_URL" \
            --ids "${IDS:-$HOME/w4a8_runs/inkling-hi2/stage4/calib/heldout_mixed_ids.npy}" \
            --windows "${WINDOWS:-32}" --max-kl "${MAX_KL:-0.15}"
    fi
fi

echo "smoke.sh: engine lines"
grep -E "Model loading took|KV cache|graph|Inkling-2 HIP serving path|workspace" "$LOG" | tail -12 || true
