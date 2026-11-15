#!/usr/bin/env bash
# One GPU session on slot a2 (HIP 0,1 = PCI buses d1,d4) between two HI2 scoreboard
# arms. The arms run under `flock gpu01.lock`, so the session queues on that lock,
# then runs the listed steps in order inside one hold of at most HOLD_S seconds
# (default 2400). Lock order is always gpu01.lock -> measure.lock (each step) ->
# boot.lock (the profile driver).
#   bash tools/rdna/rdna_hint_flex/c8_session.sh STEP [STEP ...]
# Steps:
#   profile-c8      c8_run.sh final-c8 c8           final switch set, c=8 16k/1k workload
#   profile-gemm    c8_run.sh final-gemm fast RDNA_HINT_FLEX_PREFILL_GEMM=hip
#   profile-hipmb2056  c8_run.sh final-hipmb2056 fast RDNA_HINT_FLEX_PREFILL_GEMM=hip
#                   MAXBAT=2056 (mixed steps of 2048 + 7 rows)
#   profile-vec     c8_run.sh final-vec c8 VLLM_FA_RDNA2_GQA_DECODE=2 (decode windows)
#   allreduce       bench_allreduce.py sweep, RCCL defaults and NCCL_P2P_LEVEL=pxb
#                   NCCL_PROTO=Simple (VPUSH=1 adds the 16-byte push instance)
#   overlap         bench_allreduce.py overlap
#   overlap-hp      the same with the collectives on a high-priority NCCL stream
#   attn-check      gqa_decode_check.py with VLLM_FA_RDNA2_GQA_DECODE=1 and 2: bits, timing,
#                   and a split-count sweep (VLLM_FA_RDNA2_DECODE_SPLITS)
#   test-vpush      tests/distributed/test_rdna_ar_vpush_gpu.py (2 GPUs, needs the new _rocm_C)
#   install-so      move $NEW_SO over vllm/_rocm_C.abi3.so (atomic; running servers keep
#                   the old inode)
# Output goes to $OUT (default ~/w4a8_runs/hi2-c8).
set -uo pipefail
TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
HERE=$TREE/tools/rdna/rdna_hint_flex
OUT=${OUT:-$HOME/w4a8_runs/hi2-c8}
GPU_LOCK=${GPU_LOCK:-$HOME/w4a8_runs/hi2-tp2/gpu01.lock}
MLOCK=${MEASURE_LOCK:-$HOME/w4a8_runs/measure.lock}
VENV=${VENV:-$HOME/Apps/vllm/venv-7.14.0_0.32.0}
PY=$VENV/bin/python
GPUS=${GPUS:-0,1}
BUSES=${BUSES:-d1,d4}
QUIET=${QUIET:-$HOME/w4a8_runs/lanes/quiet/HiC8Opt}
NEW_SO=${NEW_SO:-$TREE/build/c8/_rocm_C.abi3.so}
mkdir -p "$OUT" "$(dirname "$QUIET")"
if [ $# -eq 0 ]; then
    sed -n '2,/^set -uo/p' "$0" | sed 's/^# \{0,1\}//;$d'
    exit 2
fi
if [ -z "${C8_HAVE_GPU_LOCK:-}" ]; then
    echo "[$(date +%T)] session $*: waiting for $GPU_LOCK"
    C8_HAVE_GPU_LOCK=1 exec flock "$GPU_LOCK" timeout --kill-after=60 "${HOLD_S:-2400}" \
        bash "$0" "$@"
fi
T0=$(date +%s)
log() { echo "[$(date +%T) +$(($(date +%s) - T0))s] $*"; }
log "session $*: holding $GPU_LOCK (limit ${HOLD_S:-2400}s)"

SL=$LD_LIBRARY_PATH
SP=$VENV/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$SP/_rocm_sdk_libraries/lib:$SP/_rocm_sdk_core/lib/host-math/lib:$SP/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SP/_rocm_sdk_core/lib:$SP/torch/lib${SL:+:$SL}
export PYTHONPATH=$TREE
export VLLM_ROCM_USE_AITER=0 VLLM_RDNA_FORCE_FP16=1 TORCH_BLAS_PREFER_HIPBLASLT=0
export PYTORCH_TUNABLEOP_ENABLED=0 VLLM_WORKER_MULTIPROC_METHOD=spawn GPU_MAX_HW_QUEUES=2
export HSA_FORCE_FINE_GRAIN_PCIE=1 HSA_ENABLE_SDMA=0 OMP_NUM_THREADS=4
export HIP_VISIBLE_DEVICES=$GPUS
export VENV GPUS BUSES

# measured NAME SECONDS OUTFILE CMD...: one quiet window under the measurement mutex
# (waits for it at most 5 min). The quiet file is refreshed every minute, for at most
# 16 minutes whatever happens.
measured() {
    local name=$1 secs=$2 outfile=$3
    shift 3
    log "$name: waiting for $MLOCK"
    # shellcheck disable=SC2016
    flock -w 300 -E 99 "$MLOCK" bash -c '
        q=$0 secs=$1
        shift
        touch "$q"
        ( for _ in $(seq 1 16); do sleep 60; touch "$q"; done ) &
        keep=$!
        timeout --kill-after=30 "$secs" "$@"
        rc=$?
        kill "$keep" 2>/dev/null
        rm -f "$q"
        exit $rc' "$QUIET" "$secs" "$@" > "$outfile" 2>&1
    local rc=$?
    log "$name: rc=$rc -> $outfile"
    return $rc
}

step_allreduce() {
    local variant envs
    for variant in default pxb; do
        envs=()
        [ "$variant" = pxb ] && envs=(NCCL_P2P_LEVEL=pxb NCCL_PROTO=Simple)
        measured "allreduce-$variant" 700 "$OUT/ar_sweep_$variant.log" env "${envs[@]}" "$PY" \
            "$HERE/bench_allreduce.py" sweep --buses "$BUSES" ${VPUSH:+--vpush} \
            --csv "$OUT/ar_sweep_$variant.csv"
        tail -22 "$OUT/ar_sweep_$variant.log"
    done
}

step_overlap() {
    measured overlap 500 "$OUT/ar_overlap.log" "$PY" "$HERE/bench_allreduce.py" overlap \
        --buses "$BUSES" --csv "$OUT/ar_overlap.csv"
    tail -12 "$OUT/ar_overlap.log"
}

step_overlap_hp() {
    measured overlap-hp 500 "$OUT/ar_overlap_hp.log" "$PY" "$HERE/bench_allreduce.py" \
        overlap --buses "$BUSES" --high-priority --csv "$OUT/ar_overlap_hp.csv"
    tail -12 "$OUT/ar_overlap_hp.log"
}

step_attn_check() {
    local mode chk=$TREE/tools/rdna/port_v031/gqa_decode_check.py
    for mode in 1 2; do
        measured "attn-check-$mode" 600 "$OUT/gqa_check_m$mode.log" env \
            VLLM_FA_RDNA2_GQA_DECODE=$mode bash -c "\"$PY\" \"$chk\" --tag m$mode --repeat 2 \
            --out \"$OUT/gqa_m$mode.json\"; \"$PY\" \"$chk\" --tag m$mode --bench \
            --only 27b_tp2_decode --splits 16 12 13 8 6 --out \"$OUT/gqa_bench_m$mode.json\""
        grep -E "SUMMARY|FAIL" "$OUT/gqa_check_m$mode.log" | head -5
    done
    # another split count: not bit-identical to 16 (the combine order changes) but in tolerance
    measured attn-check-s12 400 "$OUT/gqa_check_m2s12.log" env VLLM_FA_RDNA2_GQA_DECODE=2 \
        VLLM_FA_RDNA2_DECODE_SPLITS=12 "$PY" "$chk" --tag m2s12 --quick --repeat 2
    grep -E "SUMMARY|FAIL" "$OUT/gqa_check_m2s12.log" | head -5
    "$PY" "$chk" --compare "$OUT/gqa_m1.json" "$OUT/gqa_m2.json" | tail -5
    "$PY" "$chk" --compare-bench "$OUT/gqa_bench_m1.json" "$OUT/gqa_bench_m2.json"
}

step_test_vpush() {
    (
        cd "$TREE" || exit 2
        measured test-vpush 600 "$OUT/test_vpush.log" "$PY" -m pytest \
            tests/distributed/test_rdna_ar_vpush_gpu.py -x -q -p no:cacheprovider
    )
    tail -8 "$OUT/test_vpush.log"
}

step_install_so() {
    local dst=$TREE/vllm/_rocm_C.abi3.so
    [ -f "$dst.orig" ] || cp -p "$dst" "$dst.orig"
    cp -p "$NEW_SO" "$dst.new.$$" && mv -f "$dst.new.$$" "$dst"
    log "install-so: $(md5sum "$dst" | cut -c1-12) from $NEW_SO"
}

for step in "$@"; do
    log "step $step"
    case $step in
        profile-c8) HOLD_S=1500 bash "$HERE/c8_run.sh" final-c8 c8 ;;
        profile-gemm) HOLD_S=1000 bash "$HERE/c8_run.sh" final-gemm fast RDNA_HINT_FLEX_PREFILL_GEMM=hip ;;
        profile-hipmb2056)
            HOLD_S=1000 bash "$HERE/c8_run.sh" final-hipmb2056 fast \
                RDNA_HINT_FLEX_PREFILL_GEMM=hip MAXBAT=2056
            ;;
        profile-vec)
            C8_ARGS="--windows decode" HOLD_S=1000 bash "$HERE/c8_run.sh" final-vec c8 \
                VLLM_FA_RDNA2_GQA_DECODE=2
            ;;
        allreduce) step_allreduce ;;
        overlap) step_overlap ;;
        overlap-hp) step_overlap_hp ;;
        attn-check) step_attn_check ;;
        test-vpush) step_test_vpush ;;
        install-so) step_install_so ;;
        *) log "unknown step $step" ;;
    esac
    log "step $step done rc=$?"
done
log "session done"
