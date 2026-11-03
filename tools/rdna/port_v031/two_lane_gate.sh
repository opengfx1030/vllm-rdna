#!/usr/bin/env bash
# R-S7-3: one two-lane acceptance run plus the gates the generic driver
# cannot check (KFD budget, runlist oversubscription, strict-mode sync,
# stall gates, probe verdicts).
#
#   RECIPE=flashnext-mtp2-twolane MODEL=/path GPUS=6,7,8,9 TAG=fn-tl16 \
#     LANE_DECODE_WGPS=20 bash tools/rdna/port_v031/two_lane_gate.sh
#
# RECIPE, MODEL, GPUS and TAG are required. CELLS/STALL_ARGS/BENCH_EXTRA
# default to the G2/G3 values in the contract; override them for M3 (27B)
# after the gap gate is set from data. Results land in
# ~/w4a8_runs/port-v031/serve-$TAG/ and verdict.txt beside them.
set -uo pipefail

TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
: "${RECIPE:?RECIPE required}" "${MODEL:?MODEL required}" "${GPUS:?GPUS required}"
TAG=${TAG:?TAG required}
PORT=${PORT:-18120}
OUT=${OUT:-$HOME/w4a8_runs/port-v031/serve-$TAG}
BOOT_LOCK=${BOOT_LOCK:-$HOME/w4a8_runs/boot.lock}
LANE_DECODE_WGPS=${LANE_DECODE_WGPS:-20}
CELLS=${CELLS:-"1024:512:1 1024:512:8 16384:1024:8:32 16384:128:1"}
STALL_ARGS=${STALL_ARGS:---decoders 6 --prefill-lens 16384 --repeats 3 --max-stall-ms 100 --max-stall-worst-ms 150 --max-gap-p99-ms 100 --min-decode-tokens-during-prefill 100 --max-starved 0 --max-ttft-slowdown 2.5}
STALL_TREE=${STALL_TREE:-1}

mkdir -p "$OUT"
VERDICT=$OUT/verdict.txt
: > "$VERDICT"
fail=0
ok() { echo "ok   $*" | tee -a "$VERDICT"; }
bad() { echo "FAIL $*" | tee -a "$VERDICT"; fail=1; }
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$VERDICT"; }

if [[ ${TWO_LANE:-1} != 1 ]]; then
    bad "TWO_LANE must be 1 for the two-lane gate (got ${TWO_LANE})"
fi
if (( ${GPU_MAX_HW_QUEUES:-2} > 2 )); then
    bad "GPU_MAX_HW_QUEUES must be <=2 (got ${GPU_MAX_HW_QUEUES})"
fi

log "boot RECIPE=$RECIPE GPUS=$GPUS split D=$LANE_DECODE_WGPS TAG=$TAG"

RECIPE="$RECIPE" MODEL="$MODEL" GPUS="$GPUS" PORT="$PORT" TAG="$TAG" \
    CELLS="$CELLS" BENCH_EXTRA="--temperature 0" \
    STALL_PROBE=1 STALL_ARGS="$STALL_ARGS" CONC_PROBE=1 WARMUP=0 \
    BOOT_LOCK="$BOOT_LOCK" TWO_LANE=1 GPU_MAX_HW_QUEUES=2 \
    LANE_DECODE_WGPS="$LANE_DECODE_WGPS" VLLM_RDNA_LANE_STRICT=1 \
    bash "$TREE/tools/rdna/port_v031/serve_validate.sh"
rc=$?
if (( rc != 0 )); then
    bad "serve_validate.sh exited $rc"
fi

# --- G5 resources -------------------------------------------------------
if [[ -f $OUT/serve.log ]]; then
    if grep -q 'Runlist is getting oversubscribed' "$OUT/serve.log"; then
        bad "runlist oversubscribed (G5)"
    else
        ok "no runlist oversubscription"
    fi
    if grep -q 'rdna_ar_wedged' "$OUT/serve.log"; then
        bad "rdna_ar_wedged marker (G5)"
    else
        ok "no rdna_ar_wedged marker"
    fi
    if grep -Eq 'device-wide sync|device wide sync' "$OUT/serve.log"; then
        bad "device-wide sync in strict mode (G5)"
    else
        ok "no device-wide sync"
    fi
    queues=$(grep -oE '[0-9]+ KFD queues in this process' "$OUT/serve.log" \
        | grep -oE '^[0-9]+' | sort -n | tail -1)
    if [[ -z ${queues:-} ]]; then
        bad "no KFD queue census in serve.log (G5)"
    elif (( queues > 7 )); then
        bad "KFD queues $queues > 7 (G5)"
    else
        ok "KFD queues at idle $queues <= 7"
    fi
    if grep -q 'recompile' "$OUT/serve.log"; then
        bad "torch.compile recompile at boot (G5)"
    else
        ok "no recompile marker"
    fi
else
    bad "missing $OUT/serve.log"
fi

# --- G1/G2 probes -------------------------------------------------------
if [[ -f $OUT/probe.txt ]]; then
    if grep -q 'FAIL' "$OUT/probe.txt"; then
        bad "probe.txt has FAIL (G1)"
    else
        ok "probe.txt has no FAIL"
    fi
    if grep -q 'PASS' "$OUT/probe.txt"; then
        ok "probe.txt has PASS"
    else
        bad "probe.txt has no PASS (G1)"
    fi
else
    bad "missing $OUT/probe.txt"
fi

if [[ $STALL_TREE == 1 ]]; then
    STALL=$OUT/stall/stall.txt
    if [[ -f $STALL ]]; then
        if grep -vqE '^(ok|PASS)\b' "$STALL"; then
            bad "stall gate line not ok (G2): $(grep -vE '^(ok|PASS)\b' "$STALL" | head -1)"
        else
            ok "stall gates all ok (G2)"
        fi
    else
        bad "missing $STALL (G2)"
    fi
fi

# --- summary ------------------------------------------------------------
if [[ -f $OUT/summary.txt ]] && grep -q 'FAIL' "$OUT/summary.txt"; then
    bad "summary.txt has FAIL"
else
    ok "summary.txt clean"
fi

echo
if (( fail == 0 )); then
    echo "GATE PASS: $TAG" | tee -a "$VERDICT"
    exit 0
fi
echo "GATE FAIL: $TAG (see $VERDICT)" | tee -a "$VERDICT"
exit 1
