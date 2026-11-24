#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Boot the HI2 directory on the a2 lease (HIP 0,1; PCI buses d1 d4), wait for the
# server, run the language-only greedy probe and print the engine's load/KV/graph
# lines.  With STOCK_URL set, also compare greedy token streams and (RUN_KL=1) the
# held-out KL against the stock server.
#
#   MODEL=/path/to/bpw3.5 VENV=/path/to/venv bash tools/inkling_hi2/smoke.sh
#   SLOT=a2 HIP_VISIBLE_DEVICES=0,1 EXPECT_PCI_BUSES=d1,d4 ...
#
# SLOT=a2 only (TP=2; b4 is the TP4 lease).  HIP_VISIBLE_DEVICES / EXPECT_PCI_BUSES
# default from SLOT, must be given together, and are resolved HIP -> PCI before the
# boot and before every measured cell.  A measured window touches
# ~/w4a8_runs/lanes/quiet-$SLOT/inkling-hi2-smoke (<= 15 min) and waits for the
# cross-slot boot lock; MEASURE_LOCK stays empty (cross-slot lock off).
#
# Optional: STOCK_URL=http://127.0.0.1:18231 (a stock-model server), RUN_KL=1,
#           IDS=$HOME/w4a8_runs/inkling-hi2/stage4/calib/heldout_mixed_ids.npy,
#           WINDOWS=32, KEEP=1 (leave the server up), MAXLEN/MAXBAT/SEQS/EAGER.
set -euo pipefail

: "${MODEL:?MODEL must point at the HI2 output directory}"
: "${VENV:?set VENV to the box venv with the built vllm}"

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SLOT=${SLOT:-a2}
case "$SLOT" in
    a2) slot_hip=0,1 slot_buses=d1,d4 ;;
    b4)
        echo "smoke.sh: TP=2 belongs on a2 (HIP 0,1; PCI d1 d4); b4 is the TP4 lease" >&2
        exit 2
        ;;
    *)
        echo "smoke.sh: unknown SLOT=$SLOT (a2|b4)" >&2
        exit 2
        ;;
esac
if [[ -n ${HIP_VISIBLE_DEVICES:-} && -z ${EXPECT_PCI_BUSES:-} ]] ||
    [[ -z ${HIP_VISIBLE_DEVICES:-} && -n ${EXPECT_PCI_BUSES:-} ]]; then
    echo "smoke.sh: give HIP_VISIBLE_DEVICES and EXPECT_PCI_BUSES together, or neither (SLOT=$SLOT)" >&2
    exit 2
fi
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-$slot_hip}
EXPECT_PCI_BUSES=${EXPECT_PCI_BUSES:-$slot_buses}
export SLOT HIP_VISIBLE_DEVICES EXPECT_PCI_BUSES
MEASURE_LOCK=${MEASURE_LOCK:-}
export MEASURE_LOCK
PORT=${PORT:-18230}
URL="http://127.0.0.1:$PORT"
RUN_DIR=${RUN_DIR:-$HOME/w4a8_runs/inkling-hi2-runtime}
LOG=${LOG:-$RUN_DIR/logs/serve-$(date +%Y%m%d-%H%M%S).log}
KEEP=${KEEP:-0}
WAIT_SECONDS=${WAIT_SECONDS:-2400}
BOOT_LOCK=${BOOT_LOCK:-$HOME/w4a8_runs/boot.lock}
QUIET_DIR=${QUIET_DIR:-$HOME/w4a8_runs/lanes/quiet-$SLOT}
QUIET_FILE=${QUIET_FILE:-$QUIET_DIR/inkling-hi2-smoke}
mkdir -p "$(dirname "$LOG")" "$(dirname "$BOOT_LOCK")"

check_devices() {
    env HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" \
        "$VENV/bin/python" "$HERE/slot.py" --slot "$SLOT" \
        --hip "$HIP_VISIBLE_DEVICES" --expect-buses "$EXPECT_PCI_BUSES"
}
touch_quiet() {
    mkdir -p "$QUIET_DIR"
    touch "$QUIET_FILE"
}
# The protocol is "touch, then wait until flock -n boot.lock succeeds"; a boot in
# the other slot can take the lock just before the touch, so retry.
wait_boot_lock() {
    for _ in $(seq 1 180); do
        if flock -n "$BOOT_LOCK" true 2>/dev/null; then
            return 0
        fi
        sleep 5
    done
    echo "smoke.sh: boot.lock stayed busy for 900 s" >&2
    return 1
}
measure_begin() {
    touch_quiet
    check_devices
    wait_boot_lock
}

rm -f "$QUIET_FILE"  # a stale marker of a crashed smoke would block our own boot
PORT="$PORT" bash "$HERE/serve.sh" >"$LOG" 2>&1 &
SERVER=$!
cleanup() {
    rm -f "$QUIET_FILE"
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

# The measured window starts here: touch the slot's quiet file, wait for the
# boot lock, and re-check before each cell (AGENTS.md, two-slot note).
measure_begin

HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES EXPECT_PCI_BUSES=$EXPECT_PCI_BUSES \
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
    measure_begin
    HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES EXPECT_PCI_BUSES=$EXPECT_PCI_BUSES \
    python3 "$HERE/kl_probe.py" \
        --hi2-url "$URL" --stock-url "$STOCK_URL" \
        --ids "${IDS:-$HOME/w4a8_runs/inkling-hi2/stage4/calib/heldout_mixed_ids.npy}" \
        --windows "${WINDOWS:-32}" --greedy
    if [[ ${RUN_KL:-0} == 1 ]]; then
        measure_begin
        HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES EXPECT_PCI_BUSES=$EXPECT_PCI_BUSES \
        python3 "$HERE/kl_probe.py" \
            --hi2-url "$URL" --stock-url "$STOCK_URL" \
            --ids "${IDS:-$HOME/w4a8_runs/inkling-hi2/stage4/calib/heldout_mixed_ids.npy}" \
            --windows "${WINDOWS:-32}" --max-kl "${MAX_KL:-0.15}"
    fi
fi

echo "smoke.sh: engine lines"
grep -E "Model loading took|KV cache|graph|Inkling-2 HIP serving path|workspace" "$LOG" | tail -12 || true
