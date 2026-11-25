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
# default from SLOT, must be given together, and are resolved HIP -> PCI with
# exactly the server's environment before the boot and before every measured cell.
# ROCR_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES are cleared (a stale filter must not
# hide or add cards).  A measured window touches
# ~/w4a8_runs/lanes/quiet-$SLOT/inkling-hi2-smoke (re-touched every 5 min while the
# window runs) and waits for the cross-slot boot lock; MEASURE_LOCK stays empty
# (cross-slot lock off).  Another owner's holder in LEASE_DIR refuses the run.
#
# The window also journals junction temperatures (gpu_temps.py in watcher mode:
# any sample >= 95 C or a 5 s mean >= 90 C kills the server tree and exits 3; the
# window is INVALID) and the vLLM tree's KFD eviction counters (kfd_evictions.py;
# no workers/KFD context is INVALID, never a pass).
#
# Optional: STOCK_URL=http://127.0.0.1:18231 (a stock-model server), RUN_KL=1,
#           IDS=$HOME/w4a8_runs/inkling-hi2/stage4/calib/heldout_mixed_ids.npy,
#           WINDOWS=32, GREEDY_TOKENS=160 (France probe; a reasoning prefix counts),
#           CAPSULE_TOKENS=48 (deterministic-repeat probe),
#           STOCK_KL=1 (0: with STOCK_URL, only the France comparison, no KL cell),
#           KEEP=1 (leave the server up), MAXLEN/MAXBAT/SEQS/EAGER,
#           OWNER=HI2InklingRuntime, LEASE_DIR=~/w4a8_runs/lanes/lease,
#           READY_SECONDS=2400 (boot), BOOT_TIMEOUT_SECONDS=900 and
#           CELL_TIMEOUT=900 (s per measured probe; a longer one is split),
#           TEMPS_TOOL / KFD_TOOL (defaults: this tree's tools; tests point them
#           at stubs).
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
OWNER=${OWNER:-HI2InklingRuntime}
LEASE_DIR=${LEASE_DIR:-$HOME/w4a8_runs/lanes/lease}
MEASURE_LOCK=${MEASURE_LOCK:-}
export MEASURE_LOCK
TREE=${TREE:-$(cd "$HERE/../.." && pwd)}
TEMPS_TOOL=${TEMPS_TOOL:-$TREE/tools/rdna/mono/w4a8/gpu_temps.py}
KFD_TOOL=${KFD_TOOL:-$TREE/tools/rdna/port_v031/kfd_evictions.py}
EVICT_TRIES=${EVICT_TRIES:-12}
EVICT_POLL_SECONDS=${EVICT_POLL_SECONDS:-5}
QUIET_TOUCH=${QUIET_TOUCH:-python3}
PORT=${PORT:-18230}
URL="http://127.0.0.1:$PORT"
RUN_DIR=${RUN_DIR:-$HOME/w4a8_runs/inkling-hi2-runtime}
LOG=${LOG:-$RUN_DIR/logs/serve-$(date +%Y%m%d-%H%M%S).log}
KEEP=${KEEP:-0}
WAIT_SECONDS=${WAIT_SECONDS:-2400}
BOOT_TIMEOUT_SECONDS=${BOOT_TIMEOUT_SECONDS:-900}
CELL_TIMEOUT=${CELL_TIMEOUT:-900}
BOOT_LOCK=${BOOT_LOCK:-$HOME/w4a8_runs/boot.lock}
QUIET_DIR=${QUIET_DIR:-$HOME/w4a8_runs/lanes/quiet-$SLOT}
QUIET_FILE=${QUIET_FILE:-$QUIET_DIR/inkling-hi2-smoke}
SERVER_PID_FILE=${SERVER_PID_FILE:-$RUN_DIR/server.pid}
export SERVER_PID_FILE

# Never inherit another workload's device filter; see serve.sh.
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
mkdir -p "$(dirname "$LOG")" "$(dirname "$BOOT_LOCK")" "$RUN_DIR/logs"

check_devices() {
    env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
        HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" \
        "$VENV/bin/python" "$HERE/slot.py" --slot "$SLOT" \
        --hip "$HIP_VISIBLE_DEVICES" --expect-buses "$EXPECT_PCI_BUSES"
}
# No live check here: it would initialize the slot's cards before the lease is
# confirmed and before the quiet/lock wait.  serve.sh checks before its boot and
# measure_begin() re-checks before each measured cell.

# Another holder for this slot means the slot is not ours to boot; the boot of
# serve.sh acquires/writes our own holder (HI2InklingRuntime.a2).
. "$HERE/lease.sh"
lease_refuse_foreign "smoke.sh" || exit 1
touch_quiet() {
    mkdir -p "$QUIET_DIR"
    touch "$QUIET_FILE"
}
# Keep the touch fresh for a measured window (<= 15 min; one touch per 5 min).
# quiet_touch.py sleeps in-process, so killing it cannot leave a stray sleep.
start_quiet_watchdog() {
    "$QUIET_TOUCH" "$HERE/quiet_touch.py" --file "$QUIET_FILE" \
        --interval 300 --count 3 &
    QUIET_WATCHDOG=$!
}
stop_quiet_watchdog() {
    [[ -n ${QUIET_WATCHDOG:-} ]] && kill "$QUIET_WATCHDOG" 2>/dev/null || true
    QUIET_WATCHDOG=
}
# The protocol is "touch, then wait until flock -n boot.lock succeeds"; a boot in
# the other slot can take the lock just before the touch, so retry.
wait_boot_lock() {
    for _ in $(seq 1 $((BOOT_TIMEOUT_SECONDS / 5))); do
        if flock -n "$BOOT_LOCK" true 2>/dev/null; then
            return 0
        fi
        sleep 5
    done
    echo "smoke.sh: boot.lock stayed busy for $BOOT_TIMEOUT_SECONDS s" >&2
    return 1
}
measure_begin() {
    touch_quiet
    check_devices
    wait_boot_lock
}
# Junction-temperature watcher (sysfs only; no GPU context): any sample >= 95 C
# or a 5 s mean >= 90 C kills the server tree and exits 3, and the window is
# INVALID.  The CSV is parsed at the end as the second, independent check.
sample_temps_start() {
    TEMPS=$RUN_DIR/logs/temps-$(date +%Y%m%d-%H%M%S).csv
    local kill_pid=${SERVER_PID:-$SERVER}
    "$VENV/bin/python" "$TEMPS_TOOL" \
        --gpus "$HIP_VISIBLE_DEVICES" --period 5 --count 0 --out "$TEMPS" \
        --kill-pid "$kill_pid" &
    TEMPS_PID=$!
    echo "smoke.sh: junction temperatures -> $TEMPS (watcher kills $kill_pid on a breach)"
}
sample_temps_stop() {
    [[ -n ${TEMPS_PID:-} ]] || return 0
    local pid=$TEMPS_PID rc
    kill "$pid" 2>/dev/null || true
    if wait "$pid"; then rc=0; else rc=$?; fi
    TEMPS_PID=
    python3 - "$TEMPS" "$rc" <<'PY'
import csv, sys
peak = None
with open(sys.argv[1], newline="") as f:
    rows = csv.reader(f)
    next(rows, None)
    for row in rows:
        for cell in row[2:]:
            if cell:
                peak = max(peak or float(cell), float(cell))
shown = "n/a" if peak is None else f"{peak:.1f}"
print(f"smoke.sh: max junction {shown} C ({sys.argv[1]})")
if peak is None:
    sys.exit(1)  # no temperature evidence is not a pass
if sys.argv[2] not in ("0", "143", "130"):  # 3/1 = the watcher tripped
    sys.exit(1)
sys.exit(1 if peak >= 95 else 0)
PY
}
# KFD evictions of the vLLM tree: no workers/KFD context or a jump is INVALID.
snapshot_evictions_start() {
    EVICT_BEFORE=
    EVICT_MISSING=0
    if [[ -z ${SERVER_PID:-} ]]; then
        EVICT_MISSING=1
        echo "smoke.sh: no server pid for the KFD eviction check; the window is INVALID" >&2
        return 1
    fi
    for _ in $(seq 1 "$EVICT_TRIES"); do
        if "$VENV/bin/python" "$KFD_TOOL" "$SERVER_PID" 2>/dev/null |
            "$VENV/bin/python" -c 'import json,sys; sys.exit(0 if json.load(sys.stdin) else 1)'; then
            EVICT_BEFORE=$RUN_DIR/logs/evict-before-$(date +%Y%m%d-%H%M%S).json
            "$VENV/bin/python" "$KFD_TOOL" "$SERVER_PID" --save "$EVICT_BEFORE"
            return 0
        fi
        sleep "$EVICT_POLL_SECONDS"
    done
    EVICT_MISSING=1
    echo "smoke.sh: no vLLM KFD context under pid $SERVER_PID; the window is INVALID" >&2
    return 1
}
evictions_ok() {
    [[ -n ${EVICT_BEFORE:-} ]] || return 1
    local line
    line=$("$VENV/bin/python" "$KFD_TOOL" "$SERVER_PID" --since "$EVICT_BEFORE")
    echo "smoke.sh: $line"
    [[ $line != *INVALID* && $line != *"no vLLM workers"* ]]
}

rm -f "$QUIET_FILE"  # a stale marker of a crashed smoke would block our own boot
PORT="$PORT" bash "$HERE/serve.sh" >"$LOG" 2>&1 &
SERVER=$!
cleanup() {
    stop_quiet_watchdog
    [[ -n ${TEMPS_PID:-} ]] && kill "$TEMPS_PID" 2>/dev/null || true
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
SERVER_PID=$(cat "$SERVER_PID_FILE" 2>/dev/null || true)
snapshot_evictions_start || true

# The measured window starts here: touch the slot's quiet file, wait for the
# boot lock, and re-check before each cell (AGENTS.md, two-slot note).
measure_begin
start_quiet_watchdog
sample_temps_start

HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES EXPECT_PCI_BUSES=$EXPECT_PCI_BUSES \
python3 - "$URL" <<'PY'
import json, os, sys, time, urllib.request

url = sys.argv[1]
stock = os.environ.get("STOCK_URL") or ""
greedy_tokens = int(os.environ.get("GREEDY_TOKENS", "160"))
capsule_tokens = int(os.environ.get("CAPSULE_TOKENS", "48"))
served = {}


def call(base, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=1200) as r:
        return json.loads(r.read())


def model_name(base):
    if base not in served:
        served[base] = call(base, "/v1/models")["data"][0]["id"]
    return served[base]


def chat(base, messages, max_tokens):
    t0 = time.time()
    out = call(
        base,
        "/v1/chat/completions",
        {
            "model": model_name(base),
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
    )
    text = out["choices"][0]["message"].get("content") or ""
    return text, time.time() - t0, out["usage"]


FRANCE = [
    {
        "role": "user",
        "content": "What is the capital of France? Reply with the city name.",
    }
]
hi2_text, hi2_s, hi2_usage = chat(url, FRANCE, greedy_tokens)
print(f"greedy: hi2 {hi2_text!r} ({hi2_s:.2f} s, {hi2_usage})")
if stock:
    stock_text, stock_s, stock_usage = chat(stock, FRANCE, greedy_tokens)
    print(f"greedy: stock {stock_text!r} ({stock_s:.2f} s, {stock_usage})")
    divergence = next(
        (
            i
            for i, (a, b) in enumerate(zip(hi2_text, stock_text))
            if a != b
        ),
        min(len(hi2_text), len(stock_text)),
    )
    print(
        f"greedy: hi2 {len(hi2_text)} chars, stock {len(stock_text)} chars, "
        f"first divergence at char {divergence}"
    )
# A reasoning model may emit its plan before the answer: 160 tokens, and the city
# is accepted anywhere in the reasoning + answer text (lower case).
assert "paris" in hi2_text.lower(), (
    f"the greedy answer does not name Paris within {greedy_tokens} tokens: "
    f"{hi2_text!r}"
)
print("RESULT greedy PASS")

# The capsule does not judge wording: two identical greedy requests must return
# the same non-empty text (deterministic repeat + something decodable).
CAPSULE = [
    {
        "role": "user",
        "content": "Write the numbers 1 to 5 separated by spaces. Output only that.",
    }
]
cap_first, _, _ = chat(url, CAPSULE, capsule_tokens)
cap_second, _, _ = chat(url, CAPSULE, capsule_tokens)
print(f"capsule: first {cap_first!r}")
print(f"capsule: second {cap_second!r}")
assert cap_first.strip(), "the greedy capsule returned an empty answer"
assert sum(ch.isalnum() for ch in cap_first) >= 2, (
    f"the greedy capsule has no decodable token: {cap_first!r}"
)
assert cap_first == cap_second, (
    "the greedy capsule is not a deterministic repeat: "
    f"{cap_first!r} != {cap_second!r}"
)
print("RESULT capsule PASS")
PY

if [[ -n ${STOCK_URL:-} && ${STOCK_KL:-1} == 1 ]]; then
    measure_begin
    HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES EXPECT_PCI_BUSES=$EXPECT_PCI_BUSES \
    timeout "$CELL_TIMEOUT" python3 "$HERE/kl_probe.py" \
        --hi2-url "$URL" --stock-url "$STOCK_URL" \
        --ids "${IDS:-$HOME/w4a8_runs/inkling-hi2/stage4/calib/heldout_mixed_ids.npy}" \
        --windows "${WINDOWS:-32}" --greedy
    if [[ ${RUN_KL:-0} == 1 ]]; then
        measure_begin
        HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES EXPECT_PCI_BUSES=$EXPECT_PCI_BUSES \
        timeout "$CELL_TIMEOUT" python3 "$HERE/kl_probe.py" \
            --hi2-url "$URL" --stock-url "$STOCK_URL" \
            --ids "${IDS:-$HOME/w4a8_runs/inkling-hi2/stage4/calib/heldout_mixed_ids.npy}" \
            --windows "${WINDOWS:-32}" --max-kl "${MAX_KL:-0.15}"
    fi
fi

# The measured window is only valid on a cool, undisturbed slot: temps and KFD
# evictions are checked last so a hot/evicted/unsampled run can never print PASS.
window_ok=1
if ! sample_temps_stop; then
    echo "smoke.sh: the temperature watcher tripped (>= 95 C or >= 90 C mean) or there are no samples; the window is INVALID" >&2
    window_ok=0
fi
if [[ ${EVICT_MISSING:-0} == 1 ]]; then
    echo "smoke.sh: no KFD worker context was found; the window is INVALID" >&2
    window_ok=0
elif ! evictions_ok; then
    echo "smoke.sh: KFD evicted the model during the window; the window is INVALID" >&2
    window_ok=0
fi
if ((!window_ok)); then
    echo "RESULT measured window INVALID"
    exit 1
fi
echo "RESULT measured window PASS"

echo "smoke.sh: engine lines"
grep -E "Model loading took|KV cache|graph|Inkling-2 HIP serving path|workspace" "$LOG" | tail -12 || true
