#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Boot an HI2 Inkling-2-Mini directory on the a2 lease (HIP 0,1; PCI buses d1 d4),
# two V620s (TP=2), language model only.
#
#   MODEL=/path/to/bpw3.5 VENV=/path/to/venv bash tools/inkling_hi2/serve.sh
#   SLOT=a2 HIP_VISIBLE_DEVICES=0,1 EXPECT_PCI_BUSES=d1,d4 ...
#
# Required: MODEL (a directory written by the quantizer; it carries the tokenizer),
#           VENV (the box venv with the built vllm).
# Optional: SLOT=a2 (the only slot this TP=2 recipe runs on; b4 is the TP4 lease
#           and is refused), HIP_VISIBLE_DEVICES / EXPECT_PCI_BUSES (both default
#           from SLOT, must be given together: resolved HIP -> PCI with exactly the
#           server's environment before the boot and refused on a mismatch or a
#           non-V620), OWNER=HI2Inkling (this work's lane holder; another holder
#           file for the slot refuses the boot), LEASE_DIR=~/w4a8_runs/lanes/lease,
#           TREE (default: this worktree), PORT=18230, MAXLEN=8192, MAXBAT=256,
#           SEQS=1, EAGER=0, GMEM=0.92, WAIT_QUIET=1 (wait out the slot's quiet
#           windows), BOOT_LOCK=~/w4a8_runs/boot.lock, QUIET_DIR (default:
#           ~/w4a8_runs/lanes/quiet-$SLOT), MEASURE_LOCK= (empty: cross-slot lock
#           off), READY_SECONDS=1800, SERVER_PID_FILE= (write the server pid here
#           for the window samplers), REGRESSION=1 (the HI2 kernel path; 0 boots
#           with the torch glue reference for an A/B).
#
# Protocol (AGENTS.md, "Two-slot box operation 2026-11-15"): fail-fast lease check,
# wait out the slot's quiet windows, take the cross-slot boot lock, hold it until
# the server answers /health (then release it: the server keeps running), and
# refuse to launch on a moved HIP -> PCI mapping.  ROCR_VISIBLE_DEVICES and
# CUDA_VISIBLE_DEVICES are cleared so a stale filter cannot hide or add cards.
#
# The recipe (inkling2-mini-tp2) owns FULL_AND_PIECEWISE HIP graphs, prefix
# caching, the HIP all-reduce and the TP plan; this wrapper only pins the
# language-only + HI2 settings and the small first-run context.
set -euo pipefail

: "${MODEL:?MODEL must point at the HI2 output directory}"
: "${VENV:?set VENV to the box venv with the built vllm}"

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SLOT=${SLOT:-a2}
case "$SLOT" in
    a2) slot_hip=0,1 slot_buses=d1,d4 ;;
    b4)
        echo "serve.sh: TP=2 belongs on a2 (HIP 0,1; PCI d1 d4); b4 is the TP4 lease" >&2
        exit 2
        ;;
    *)
        echo "serve.sh: unknown SLOT=$SLOT (a2|b4)" >&2
        exit 2
        ;;
esac
if [[ -n ${HIP_VISIBLE_DEVICES:-} && -z ${EXPECT_PCI_BUSES:-} ]] ||
    [[ -z ${HIP_VISIBLE_DEVICES:-} && -n ${EXPECT_PCI_BUSES:-} ]]; then
    echo "serve.sh: give HIP_VISIBLE_DEVICES and EXPECT_PCI_BUSES together, or neither (SLOT=$SLOT)" >&2
    exit 2
fi
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-$slot_hip}
EXPECT_PCI_BUSES=${EXPECT_PCI_BUSES:-$slot_buses}
export SLOT HIP_VISIBLE_DEVICES EXPECT_PCI_BUSES
OWNER=${OWNER:-HI2Inkling}
LEASE_DIR=${LEASE_DIR:-$HOME/w4a8_runs/lanes/lease}
MEASURE_LOCK=${MEASURE_LOCK:-}
export MEASURE_LOCK
TREE=${TREE:-$(cd "$HERE/../.." && pwd)}
PORT=${PORT:-18230}
MAXLEN=${MAXLEN:-8192}
MAXBAT=${MAXBAT:-256}
SEQS=${SEQS:-1}
EAGER=${EAGER:-0}
GMEM=${GMEM:-0.92}
WAIT_QUIET=${WAIT_QUIET:-1}
BOOT_LOCK=${BOOT_LOCK:-$HOME/w4a8_runs/boot.lock}
QUIET_DIR=${QUIET_DIR:-$HOME/w4a8_runs/lanes/quiet-$SLOT}
READY_SECONDS=${READY_SECONDS:-1800}
SERVER_PID_FILE=${SERVER_PID_FILE:-}
REGRESSION=${REGRESSION:-1}

if [[ ! -f $MODEL/model.safetensors.index.json || ! -f $MODEL/tokenizer.json ]]; then
    echo "serve.sh: $MODEL is not a servable HI2 directory (index or tokenizer missing)" >&2
    exit 2
fi
if [[ ! -x $VENV/bin/python ]]; then
    echo "serve.sh: $VENV/bin/python not found" >&2
    exit 2
fi

# Resolve the declared HIP indexes to their live PCI buses and refuse a moved
# enumeration or a non-V620 (AGENTS.md, "Two-slot box operation 2026-11-15").
# Called once now (fail fast) and once immediately before the launch.
check_devices() {
    env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
        HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" \
        "$VENV/bin/python" "$HERE/slot.py" --slot "$SLOT" \
        --hip "$HIP_VISIBLE_DEVICES" --expect-buses "$EXPECT_PCI_BUSES"
}

# The launcher (and smoke.sh) never inherit another workload's device filter: it
# would silently change the set the server sees.  HIP_VISIBLE_DEVICES is the one
# filter this kit sets, and the check below runs with exactly the server's env.
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES

# Another owner's holder file for this slot means somebody else has it leased.
# request/release per the lease README; never delete another agent's file.
lease_check() {
    [[ -d $LEASE_DIR ]] || return 0
    local holder other=0
    for holder in "$LEASE_DIR"/*."$SLOT"; do
        [[ -e $holder ]] || continue
        [[ $(basename "$holder") == "$OWNER.$SLOT" ]] && continue
        echo "serve.sh: $(basename "$holder") holds the $SLOT lease; not booting (OWNER=$OWNER)" >&2
        other=1
    done
    if ((other)); then
        echo "serve.sh: inspect/clear it per $LEASE_DIR/README.md; do not delete another agent's file" >&2
        return 1
    fi
}
lease_check
# The slot is ours per the holder; resolve HIP -> PCI now (fail fast before the
# quiet wait) and again under the boot lock before the launch.
check_devices

# --- quiet windows + one cross-slot boot lock, held until the server is ready ---
mkdir -p "$(dirname "$BOOT_LOCK")" "$QUIET_DIR"
exec 9>"$BOOT_LOCK"
while :; do
    if [[ $WAIT_QUIET == 1 ]]; then
        while q=$(find "$QUIET_DIR" -type f -mmin -6 -print -quit 2>/dev/null) && [[ -n $q ]]; do
            echo "serve.sh: a quiet window in $QUIET_DIR is active ($q); waiting (60 s)"
            sleep 60
        done
    fi
    flock 9
    if [[ $WAIT_QUIET == 1 ]]; then
        q=$(find "$QUIET_DIR" -type f -mmin -6 -print -quit 2>/dev/null)
        if [[ -n $q ]]; then
            echo "serve.sh: quiet window opened under the boot lock ($q); released, retry"
            flock -u 9
            continue
        fi
    fi
    break
done

EXTRA="--limit-mm-per-prompt {\"image\":0} --max-logprobs 128"
if [[ $REGRESSION == 0 ]]; then
    export VLLM_INKLING2_PATH=reference
fi

echo "serve.sh: SLOT=$SLOT HIP=$HIP_VISIBLE_DEVICES PCI=$EXPECT_PCI_BUSES MODEL=$MODEL TP=2 PORT=$PORT MAXLEN=$MAXLEN MAXBAT=$MAXBAT SEQS=$SEQS EAGER=$EAGER OWNER=$OWNER"
check_devices
if [[ -n $SERVER_PID_FILE ]]; then
    mkdir -p "$(dirname "$SERVER_PID_FILE")"
fi
env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
    HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" EXPECT_PCI_BUSES="$EXPECT_PCI_BUSES" \
    bash "$TREE/tools/rdna/serve_rdna.sh" \
    RECIPE=inkling2-mini-tp2 \
    MODEL="$MODEL" \
    VENV="$VENV" \
    HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" \
    PORT="$PORT" \
    DSPARK=0 MTP=0 EAGER="$EAGER" CG_MODE=FULL_AND_PIECEWISE \
    MAXLEN="$MAXLEN" MAXBAT="$MAXBAT" SEQS="$SEQS" GMEM="$GMEM" \
    EXTRA_ARGS="$EXTRA" 9>&- &
SERVER=$!
cleanup() {
    local pid=${SERVER:-}
    [[ -n $pid ]] && kill "$pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'cleanup; exit 143' TERM INT
if [[ -n $SERVER_PID_FILE ]]; then
    echo "$SERVER" >"$SERVER_PID_FILE"
fi

ready=0
for ((i = 0; i < READY_SECONDS / 5; i++)); do
    if ! kill -0 "$SERVER" 2>/dev/null; then
        echo "serve.sh: the server died during boot" >&2
        break
    fi
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 5
done
exec 9>&-
if ((!ready)); then
    echo "serve.sh: no /health after $READY_SECONDS s" >&2
    kill "$SERVER" 2>/dev/null || true
    wait "$SERVER" 2>/dev/null || true
    exit 1
fi

echo "serve.sh: READY http://127.0.0.1:$PORT/health"
wait "$SERVER"
