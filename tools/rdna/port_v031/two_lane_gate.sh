#!/usr/bin/env bash
# R-S7-3: the two-lane acceptance gate.
#
#   RECIPE=flashnext-mtp2-twolane MODEL=/path GPUS=6,7,8,9 TAG=fn-tl16 \
#     LANE_DECODE_WGPS=20 bash tools/rdna/port_v031/two_lane_gate.sh
#
# Runs one full acceptance arm flag off and one flag on (same recipe, same
# cells, one boot at a time) and checks what the generic driver cannot:
# G5 KFD budget, runlist oversubscription, strict-mode sync, G2 stall gates,
# G1 probe verdicts, and G3 (flag on versus flag off: mixed-load decode ITL,
# solo prefill TTFT). The verdict is one line per gate in verdict.txt.
#
# RECIPE, MODEL, GPUS and TAG are required. CELLS/STALL_ARGS/BENCH_EXTRA
# default to the G3 values in the contract; override them for M3/M4. Results
# land in ~/w4a8_runs/port-v031/serve-$TAG-gate/.
#
# G4 (the TP4 split sweep) is not one boot: `accept/tp_sweep.sh` runs it.
set -uo pipefail

TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
: "${RECIPE:?RECIPE required}" "${MODEL:?MODEL required}" "${GPUS:?GPUS required}"
TAG=${TAG:?TAG required}
PORT=${PORT:-18120}
VENV=${VENV:-$HOME/Apps/vllm/venv-7.14.0_0.32.0}
base=${OUT:-$HOME/w4a8_runs/port-v031/serve-$TAG-gate}
OUT=$base/gate
BOOT_LOCK=${BOOT_LOCK:-$HOME/w4a8_runs/boot.lock}
LANE_DECODE_WGPS=${LANE_DECODE_WGPS:-20}
TP=${TP:-1}
MTP=${MTP:-2}
GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-2}
CELLS=${CELLS:-"1024:512:1 1024:512:8 16384:1024:8:32 16384:128:1"}
STALL_ARGS=${STALL_ARGS:---decoders 6 --prefill-lens 16384 --repeats 3 --max-stall-ms 100 --max-stall-worst-ms 150 --max-gap-p99-ms 100 --min-decode-tokens-during-prefill 100 --max-starved 0 --max-ttft-slowdown 2.5}

PY=${PY:-python3}
mkdir -p "$OUT"
VERDICT=$OUT/verdict.txt
: > "$VERDICT"
fails=0
warns=0
ok() { echo "ok   $*" | tee -a "$VERDICT"; }
bad() { echo "FAIL $*" | tee -a "$VERDICT"; fails=$((fails + 1)); }
note() { echo "WARN $*" | tee -a "$VERDICT"; warns=$((warns + 1)); }
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$VERDICT"; }

if (( GPU_MAX_HW_QUEUES > 2 )); then
    bad "GPU_MAX_HW_QUEUES must be <=2 (got $GPU_MAX_HW_QUEUES)"
fi

# --- one acceptance arm --------------------------------------------------
run_arm() {
    local mode=$1
    local tl=$2
    local arm=$OUT/$mode
    local rc
    mkdir -p "$arm"
    log "arm $mode (TWO_LANE=$tl): boot + probes + stall + bench"
    RECIPE="$RECIPE" MODEL="$MODEL" GPUS="$GPUS" PORT="$PORT" TAG="$TAG-$mode" \
        OUT="$arm" CELLS="$CELLS" BENCH_EXTRA="--temperature 0" \
        STALL_PROBE=1 STALL_ARGS="$STALL_ARGS" CONC_PROBE=1 WARMUP=0 \
        BOOT_LOCK="$BOOT_LOCK" TWO_LANE="$tl" TP="$TP" MTP="$MTP" \
        GPU_MAX_HW_QUEUES="$GPU_MAX_HW_QUEUES" VENV="$VENV" \
        LANE_DECODE_WGPS="$LANE_DECODE_WGPS" VLLM_RDNA_LANE_STRICT=1 \
        bash "$TREE/tools/rdna/port_v031/serve_validate.sh"
    rc=$?
    if (( rc != 0 )); then
        bad "$mode: serve_validate.sh exited $rc (see $arm/summary.txt)"
    fi
    arm_checks "$arm" "$mode"
}

# --- per-arm checks -----------------------------------------------------
arm_checks() {
    local arm=$1 mode=$2

    if [[ ! -f $arm/serve.log ]]; then
        bad "$mode: missing serve.log"
        return
    fi
    if [[ ! -f $arm/summary.txt ]]; then
        bad "$mode: missing summary.txt"
        return
    fi

    if grep -q 'Runlist is getting oversubscribed' "$arm/serve.log"; then
        bad "$mode: runlist oversubscribed (G5)"
    else
        ok "$mode: no runlist oversubscription"
    fi
    if grep -q 'rdna_ar_wedged' "$arm/serve.log"; then
        bad "$mode: rdna_ar_wedged marker (G5)"
    else
        ok "$mode: no rdna_ar_wedged marker"
    fi
    if grep -Eq 'device-wide sync|device wide sync' "$arm/serve.log"; then
        bad "$mode: device-wide sync in strict mode (G5)"
    else
        ok "$mode: no device-wide sync"
    fi
    if grep -q 'recompile' "$arm/serve.log"; then
        bad "$mode: torch.compile recompile marker (G5)"
    else
        ok "$mode: no recompile marker"
    fi

    # The arm must actually run the mode it claims.
    if [[ $mode == on ]]; then
        if grep -q 'two-lane: created' "$arm/serve.log"; then
            ok "on: lane streams created"
        else
            bad "on: no lane-stream marker; the flag did not engage (G0)"
        fi
        if grep -q 'two-lane: lane start' "$arm/serve.log" \
            && grep -q 'partitioned=' "$arm/summary.txt"; then
            ok "on: scheduler partitioned batches (two-lane engine)"
        else
            bad "on: no partitioned-batch marker (G0)"
        fi
    elif grep -q 'two-lane: created' "$arm/serve.log"; then
        bad "off: lane streams created despite TWO_LANE=0 (I9)"
    else
        ok "off: no lane-stream marker (flag-off path)"
    fi

    if [[ -f $arm/probe.txt ]]; then
        if grep -q '^FAIL' "$arm/probe.txt"; then
            bad "$mode: probe.txt has FAIL (G1)"
        else
            ok "$mode: probe.txt has no FAIL"
        fi
        if grep -q 'PASS' "$arm/probe.txt"; then
            ok "$mode: probe.txt has PASS"
        else
            bad "$mode: probe.txt has no PASS (G1)"
        fi
    else
        bad "$mode: missing probe.txt (G1)"
    fi

    # Concurrency-correctness is part of G1 (greedy, seeded and churned
    # streams; long 1.5k-token prompts across a mamba block).
    if grep -q 'CONCURRENCY PASS' "$arm/summary.txt"; then
        ok "$mode: concurrency PASS"
    elif grep -q 'CONCURRENCY FAIL' "$arm/summary.txt"; then
        bad "$mode: concurrency FAIL (G1; see $arm/concurrency.txt)"
    else
        bad "$mode: no concurrency verdict (G1)"
    fi

    # KFD census: lane mode logs it at startup (R-S3-2). Budget 7, hard 8,
    # 9 oversubscribes the runlist (I6), so the gate keeps the 7 target.
    local queues
    queues=$(grep -oE '[0-9]+ KFD queues in this process' "$arm/serve.log" \
        | grep -oE '^[0-9]+' | sort -n | tail -1)
    if [[ $mode == off ]]; then
        if [[ -n ${queues:-} ]] && (( queues > 7 )); then
            bad "off: KFD queues $queues > 7 (G5)"
        else
            ok "off: KFD queues at idle ${queues:-n/a} (baseline)"
        fi
    elif [[ -z ${queues:-} ]]; then
        bad "on: no KFD queue census in serve.log (G5)"
    elif (( queues > 7 )); then
        bad "on: KFD queues $queues > 7 (G5)"
    else
        ok "on: KFD queues at idle $queues <= 7"
    fi

    arm_stall "$arm" "$mode"
}

# G2 (flag on): every stall gate line must be ok. Flag off is the baseline,
# so its stall gates are recorded only (G3 compares them). The probe's absolute
# --min-decode-tokens-during-prefill 100 applies to 16k windows; the contract
# gates a 4k window at 30 tokens, so those periodic:4096 lines are recorded as
# a known probe-scaling artifact, not an acceptance miss.
arm_stall() {
    local arm=$1
    local mode=$2
    local stall=$arm/stall/stall.txt
    local line
    if [[ ! -f $stall ]]; then
        bad "$mode: missing stall/stall.txt (G2)"
        return
    fi
    while IFS= read -r line; do
        case $line in
            ok\ *|PASS*) ;;
            *'min_decode_tokens_during_prefill'*'periodic:4096#'*)
                [[ $mode == on ]] && note "$mode: 4k window uses the 16k token minimum: ${line#FAIL }"
                ;;
            FAIL\ *)
                if [[ $mode == on ]]; then
                    bad "$mode: stall gate FAIL: ${line#FAIL }"
                else
                    log "off: baseline stall gate (expected): ${line#FAIL }"
                fi
                ;;
        esac
    done < "$stall"
    ok "$mode: stall gates checked (G2)"
}

# --- G3: flag on versus flag off ---------------------------------------
# shellcheck disable=SC2329  # invoked from run_arm's caller
compare_g3() {
    local cell=16384x1024-c8
    "$PY" "$TREE/tools/rdna/port_v031/bench_compare.py" \
        "$OUT/off" "$OUT/on" "$cell" | tee -a "$VERDICT"
    if (( PIPESTATUS[0] != 0 )); then
        bad "G3: flag-on numbers miss the gate (see above)"
    fi
}

# shellcheck disable=SC2329  # invoked from run_arm's caller
compare_g3() {
    local cell=16384x1024-c8
    "$PY" "$TREE/tools/rdna/port_v031/bench_compare.py" \
        "$OUT/off" "$OUT/on" "$cell" | tee -a "$VERDICT"
    if (( PIPESTATUS[0] != 0 )); then
        bad "G3: flag-on numbers miss the gate (see above)"
    fi
}

# --- run both arms ------------------------------------------------------
if [[ ! -d $TREE/tools/rdna/port_v031 ]]; then
    bad "tree $TREE has no tools/rdna/port_v031"
elif [[ ! -f $TREE/tools/rdna/port_v031/serve_validate.sh ]]; then
    bad "tree $TREE has no serve_validate.sh"
else
    case ",$GPUS," in
        *,0,*|*,1,*)
            bad "refusing HIP 0/1 (reserved by this allocation): GPUS=$GPUS"
            ;;
        *)
            run_arm off 0
            run_arm on 1
            compare_g3
            ;;
    esac
fi

echo
if (( fails == 0 )); then
    echo "GATE PASS: $TAG ($warns warnings, see $VERDICT)" | tee -a "$VERDICT"
    exit 0
fi
echo "GATE FAIL: $TAG ($fails failing, $warns warnings, see $VERDICT)" \
    | tee -a "$VERDICT"
exit 1
