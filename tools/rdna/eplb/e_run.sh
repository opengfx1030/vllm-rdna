#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# The cross-rank top-k disagreement study (docs/rdna2/eplb-lean-design-2026-11-16.md,
# section 11): which stage reproduces it, which discriminator removes it, and the proof
# of the fix. One lease (b4, HIP 2-5), one boot at a time, stop on a fault marker or a
# KFD eviction. Each boot goes through serve_validate.sh (boot lock, probes, eviction
# check); the replay stages run with the lean counters ON and e_hook.sh (e_replay.py)
# as the LOAD_HOOK, and their CPU verdict is replay_check.py.
#
#   micro    the router GEMM through the shipped rank tables, one process per ordinal,
#            on HIP 2,3,4,5 and again with the visible devices reversed (tunable_bits.py)
#   microfix the same through the agreed tables ($RUN/tunable-agree)
#   verify   lookup hits and per-key timings of the shipped and the agreed rank files,
#            each file on its own GPU (verify_tunableop_lookup.py)
#   vmatrix  every shipped rank file on every GPU: the measured time of each candidate
#            solution of a split key on each card
#   e1       HIP 2,3,4,5, shipped rows: fixed prompts of 1023/1024/1025/2048 tokens x24,
#            1030/2047, an aligned 16k prompt and a text prompt, prefix cache defeated
#            by cache_salt
#   e2ctl    HIP 5,4,3,2 (rank r on the other card) with VLLM_RDNA_AR=0,
#            VLLM_RDNA_AR_HIER=0 and the shared-expert stream off
#   fix      agreed rows ($RUN/tunable-agree: one solution per key in every rank file,
#            tools/rdna2_028/tunableop_agree.py), production flags, the same replay
#   accflat  agreed rows, VLLM_RDNA2_MOE_DETERMINISTIC=1, flat all-reduce: greedy, PPL
#            and concurrency probes (repeated), then a replay
#   acchier  the same with the hierarchical all-reduce forced on two islands of two
#   accbase  the shipped rows with VLLM_RDNA2_MOE_DETERMINISTIC=1, flat all-reduce: the
#            baseline probes and a replay in which the same prompt must give the same
#            disagreeing (pass, layer) cells every time if the cause is the rows
#   benchA / benchB  counters off, bench cells (TTFT, ITL) on the shipped / agreed rows
#                    (benchA2, benchB2 repeat them: STAGES="benchA benchB benchB2 benchA2")
#
#   setsid nohup env STAGES="micro e1" bash tools/rdna/eplb/e_run.sh \
#       > "$RUN/e_run.log" 2>&1 < /dev/null &
#   touch "$RUN/STOP"     # between stages
#
# Every GPU command below sets HIP_VISIBLE_DEVICES itself; CPU commands clear it.
# LEASE_HIP and the lease_* functions come from lease.sh (sourced below).
# shellcheck disable=SC2153
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TREE=${TREE:-$HOME/w4a8_runs/leapquant/tree-eplb}
RUN=${RUN:-$HOME/w4a8_runs/eplb-e}
STAGES=${STAGES:-"micro e1 e2ctl"}
PORT=${PORT:-18491}
LEASE_MIN=${LEASE_MIN:-170}
TEMPS_PERIOD=5
BOOT_LOCK=${BOOT_LOCK:-$HOME/w4a8_runs/boot.lock}
CACHE=${CACHE:-$HOME/w4a8_runs/eplb-p0b/cache/$(cut -c1-12 "$TREE/.sha")}
GROUPS_E1=${GROUPS_E1:-"mk:777:3,g1024:1024:24,g1023:1023:24,g1025:1025:24,g2048:2048:24,g1030:1030:12,g2047:2047:12,g16384:16384:8,t1024:1024:12:txt"}
GROUPS_E2=${GROUPS_E2:-"mk:777:3,g1024:1024:24,g1023:1023:12"}
GROUPS_FIX=${GROUPS_FIX:-"mk:777:3,g1024:1024:24,g1023:1023:24,g1025:1025:12,g2048:2048:12,g1030:1030:12,g16384:16384:4,t1024:1024:12:txt"}
GROUPS_ACC=${GROUPS_ACC:-"mk:777:3,g1023:1023:24,g1024:1024:12"}
GROUPS_BASE=${GROUPS_BASE:-"mk:777:3,g1024:1024:24,g1023:1023:12"}
BENCH_CELLS=${BENCH_CELLS:-"1024:128:1:24 16384:128:1:6"}
CONC_REPEATS=${CONC_REPEATS:-2}

export LEASE=b4 OWNER=${OWNER:-EplbLean}
# shellcheck source=/dev/null
source "$HOME/w4a8_runs/leapquant/lease.sh" # EXIT trap, VENV, MODEL, LD_LIBRARY_PATH
PY=$VENV/bin/python
mkdir -p "$RUN"

OBSERVERS=()
CUR_OUT=""
stop_observers() {
    ((${#OBSERVERS[@]})) && kill "${OBSERVERS[@]}" 2> /dev/null
    OBSERVERS=()
    return 0
}
cleanup() {
    stop_observers
    lease_release
}
trap cleanup EXIT
trap 'exit 143' TERM INT

stopped() { [[ -e $RUN/STOP ]]; }

# Abort rule of the box rules on the 5 s samples of temps.csv (>= 95 C, or two samples
# >= 90 C in a row): stop the running server and the sequence.
temp_guard() {
    local prev="" hot
    while sleep "$TEMPS_PERIOD"; do
        [[ -s $RUN/temps.csv ]] || continue
        hot=$(tail -n 1 "$RUN/temps.csv" | awk -F, '
            NR == 1 { for (i = 3; i <= NF; i++) { t = $i + 0; if (t >= 95) print "!" (i - 3); else if (t >= 90) print (i - 3) } }')
        if [[ $hot == *'!'* ]] || { [[ -n $hot && $hot == "$prev" ]]; }; then
            say "TEMPERATURE ABORT: hot GPU column(s) [$hot] in $(tail -n 1 "$RUN/temps.csv")"
            touch "$RUN/STOP"
            if [[ -n $CUR_OUT && -s $CUR_OUT/serve.pid ]]; then
                kill -- -"$(cut -d' ' -f1 "$CUR_OUT/serve.pid")" 2> /dev/null || true
            fi
            pkill -f "[-]-port $PORT" 2> /dev/null || true
        fi
        prev=$hot
    done
}

start_observers() {
    HIP_VISIBLE_DEVICES=$LEASE_HIP "$PY" "$TREE/tools/rdna/mono/w4a8/gpu_temps.py" \
        --gpus "$LEASE_HIP" --period "$TEMPS_PERIOD" --out "$RUN/temps.csv" &
    OBSERVERS+=($!)
    temp_guard &
    OBSERVERS+=($!)
    say "junction temperatures of HIP $LEASE_HIP every $TEMPS_PERIOD s -> $RUN/temps.csv"
}

# ---- boots ---------------------------------------------------------------------
# stage_config NAME: S_GPUS S_GROUPS S_ENV (words) S_OVERRIDES (recipe KEY=VALUE words)
# S_CONC/S_PPL (serve_validate probes), S_COUNTERS (lean counters + replay hook),
# S_CELLS (bench cells), S_REPEATS (extra concurrency probes inside the hook)
S_GPUS="" S_GROUPS="" S_ENV=() S_OVERRIDES=() S_CONC=0 S_PPL=0 S_COUNTERS=1
S_CELLS=none S_REPEATS=0
AGREED="$RUN/tunable-agree"
stage_config() {
    S_GPUS=$LEASE_HIP S_GROUPS="" S_ENV=() S_OVERRIDES=() S_CONC=0 S_PPL=0
    S_COUNTERS=1 S_CELLS=none S_REPEATS=0
    case $1 in
        e1) S_GROUPS=$GROUPS_E1 ;;
        e2ctl)
            S_GPUS=5,4,3,2
            S_GROUPS=$GROUPS_E2
            S_ENV=(VLLM_RDNA_AR_HIER=0 VLLM_DISABLE_SHARED_EXPERTS_STREAM=1)
            S_OVERRIDES=(VLLM_RDNA_AR=0)
            ;;
        fix)
            S_GROUPS=$GROUPS_FIX
            S_ENV=(TUNABLEOP_ROOT="$AGREED")
            S_CONC=1 S_PPL=1
            ;;
        accbase)
            S_GROUPS=$GROUPS_BASE
            S_ENV=(VLLM_RDNA2_MOE_DETERMINISTIC=1)
            S_CONC=1 S_PPL=1
            ;;
        accflat)
            S_GROUPS=$GROUPS_ACC
            S_ENV=(TUNABLEOP_ROOT="$AGREED" VLLM_RDNA2_MOE_DETERMINISTIC=1)
            S_CONC=1 S_PPL=1 S_REPEATS=$CONC_REPEATS
            ;;
        acchier)
            S_GROUPS=$GROUPS_ACC
            S_ENV=(TUNABLEOP_ROOT="$AGREED" VLLM_RDNA2_MOE_DETERMINISTIC=1
                VLLM_RDNA_AR_HIER=1 "VLLM_RDNA_AR_HIER_ISLANDS=0,1|2,3")
            S_CONC=1 S_PPL=1 S_REPEATS=$CONC_REPEATS
            ;;
        benchA | benchA2) S_COUNTERS=0 S_CELLS=$BENCH_CELLS ;;
        benchB | benchB2)
            S_COUNTERS=0 S_CELLS=$BENCH_CELLS
            S_ENV=(TUNABLEOP_ROOT="$AGREED")
            ;;
        *) return 1 ;;
    esac
}

boot_stage() { # name
    local name=$1 out=$RUN/$1
    stage_config "$name" || { say "unknown stage $name"; return 2; }
    mkdir -p "$out" "$out/eplb"
    CUR_OUT=$out
    say "stage $name: HIP $S_GPUS env=[${S_ENV[*]:-}] overrides=[${S_OVERRIDES[*]:-}]"
    local counters=() hook=()
    if [[ $S_COUNTERS == 1 ]]; then
        counters=(VLLM_RDNA_EPLB_STATS_DIR="$out/eplb" VLLM_RDNA_EPLB_STATS_INTERVAL_S=30
            VLLM_RDNA_EPLB_STATS_EVERY=1 VLLM_RDNA_EPLB_STATS_KERNEL=hip)
        hook=(LOAD_HOOK="$HERE/e_hook.sh" E_GROUPS="$S_GROUPS" E_STATS_DIR="$out/eplb"
            E_OUT="$out/replay" E_TEXT_FILE="$TREE/docs/rdna2/tuning.md"
            E_CONC_REPEATS="$S_REPEATS")
    fi
    (
        cd "$TREE" || exit 2
        env "${counters[@]}" "${S_ENV[@]}" "${hook[@]}" \
            RECIPE=flashnext-mtp2 MODEL="$MODEL" GPUS="$S_GPUS" PORT="$PORT" \
            TAG="eplb-e-$name" OUT="$out" CACHE="$CACHE" VENV="$VENV" CELLS="$S_CELLS" \
            CONC_PROBE="$S_CONC" PPL_PROBE="$S_PPL" PREFIX_PROBE=0 BOOT_LOCK="$BOOT_LOCK" \
            HIP_VISIBLE_DEVICES="$S_GPUS" BENCH_EXTRA="--temperature 0" QUIET_HOLD=0 \
            STOP_GRACE=90 READY_TIMEOUT=3600 \
            timeout 3000 bash tools/rdna/port_v031/serve_validate.sh "${S_OVERRIDES[@]}"
    ) > "$out/stage.log" 2>&1
    local rc=$?
    CUR_OUT=""
    say "stage $name: serve_validate rc=$rc"
    return "$rc"
}

# Why a boot's data cannot be used ("" = fine): fault markers, KFD evictions, no dump.
stage_verdict() { # name
    local out=$RUN/$1 sum=$RUN/$1/summary.txt
    [[ -s $sum ]] || { echo "no summary"; return; }
    local faults
    faults=$(sed -n 's/.*fault markers in serve.log: \([0-9]*\).*/\1/p' "$sum" | tail -1)
    [[ ${faults:-0} == 0 ]] || { echo "$faults fault marker(s) in serve.log"; return; }
    grep -a "evicted_ms since READY" "$sum" | grep -aq -- "-> ok" ||
        { echo "KFD eviction since READY: $(grep -a 'evicted_ms since READY' "$sum" | tail -1)"; return; }
    if grep -aq "INVALID" "$sum"; then echo "a bench cell is INVALID (KFD eviction)"; return; fi
    grep -aq "rdna_ar: WEDGED" "$out/serve.log" && { echo "rdna_ar wedged"; return; }
    [[ $S_COUNTERS == 1 ]] || return 0
    grep -q "LOAD HOOK rc=0" "$sum" || { echo "replay hook failed"; return; }
    [[ -d $out/replay/dump ]] || echo "no dump"
}

check() { # CPU analysis of one boot
    local out=$RUN/$1
    [[ $S_COUNTERS == 1 ]] || return 0
    (
        cd "$TREE" || exit 2
        env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
            PYTHONPATH="$TREE" "$PY" -m tools.rdna.eplb.replay_check "$out/replay" \
            --json "$out/replay_check.json"
    ) 2>&1 | tee "$out/replay_check.txt"
}

# ---- the GEMM micro test ------------------------------------------------------------
# TunableOp keeps one results file per process, picked by the device that is current at
# the first GEMM, so every ordinal runs in its own process (one worker per rank).
micro_stage() { # name rows_dir
    local name=$1 rows=$2 out=$RUN/$1 order tag r
    mkdir -p "$out"
    local fa=() fb=()
    for order in 2,3,4,5 5,4,3,2; do
        tag=${order//,/}
        for r in 0 1 2 3; do
            say "$name: router GEMM, ordinal $r of HIP $order, rank tables from $rows"
            (
                cd "$TREE" || exit 2
                env PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 \
                    PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0 \
                    PYTORCH_TUNABLEOP_FILENAME="$rows/tunableop_results.csv" \
                    PYTHONPATH="$TREE" HIP_VISIBLE_DEVICES="$order" \
                    timeout 600 "$PY" -m tools.rdna.eplb.tunable_bits run \
                    --ordinal "$r" --out "$out/bits-$tag-$r.npz"
            ) > "$out/run-$tag-$r.log" 2>&1 ||
                { say "$name: ordinal $r of HIP $order failed, see $out/run-$tag-$r.log"; return 1; }
            tail -1 "$out/run-$tag-$r.log" | cut -c1-300
            if [[ $order == 2,3,4,5 ]]; then fa+=("$out/bits-$tag-$r.npz"); else fb+=("$out/bits-$tag-$r.npz"); fi
        done
    done
    (
        cd "$TREE" || exit 2
        env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
            PYTHONPATH="$TREE" "$PY" -m tools.rdna.eplb.tunable_bits compare "${fa[@]}" \
            --vs "${fb[@]}"
    ) 2>&1 | tee "$out/compare.txt"
}

# ---- lookup hits and measured time of the rank tables ----------------------------------
# verify_tunableop_lookup.py on each GPU with the shipped and the agreed file of its rank:
# every key must hit (the solver TunableOp records is the one of the row, so the solution
# exists and runs on that card) and the per-key medians (lookup_ms) are the measured
# effect of the rewritten rows on the same card. One process per GPU, in parallel.
verify_stage() { # name
    local name=$1 out=$RUN/$1 which rows g r dt
    mkdir -p "$out"
    for which in shipped agreed; do
        rows=$TREE/tunableop/rocm7.14-rocblas5.5
        [[ $which == agreed ]] && rows=$AGREED/rocm7.14-rocblas5.5
        say "$name: lookup hits of the $which rank files, one process per GPU"
        local pids=()
        for g in 2 3 4 5; do
            r=$((g - 2))
            (
                cd "$TREE" || exit 2
                for dt in Half BFloat16; do
                    env HIP_VISIBLE_DEVICES="$g" PYTHONPATH="$TREE" \
                        "$PY" tools/rdna2_028/verify_tunableop_lookup.py \
                        --rows "$rows/tunableop_results$r.csv" --device 0 --dtype "$dt" \
                        --out "$out/$which-r$r-$dt.json"
                done
            ) > "$out/$which-r$r.log" 2>&1 &
            pids+=($!)
        done
        wait "${pids[@]}"
        grep -h "lookup hits" "$out/$which"-r*.log | cut -c1-200
    done
}

# Every shipped rank file on every GPU (4 x 4 runs, 10 warm-up and 40 timed calls per
# key): the measured time of each candidate solution of a split key on each card, to
# re-select the agreed row by measurement instead of by the recorded tuner time. One
# process per GPU in parallel; each GPU walks the files in a rotated order.
vmatrix_stage() { # name
    local name=$1 out=$RUN/$1 g i c dt
    mkdir -p "$out"
    say "$name: every shipped rank file on every GPU, one process per GPU"
    local pids=()
    for g in 2 3 4 5; do
        (
            cd "$TREE" || exit 2
            for i in 0 1 2 3; do
                c=$(((g - 2 + i) % 4))
                for dt in Half BFloat16; do
                    env HIP_VISIBLE_DEVICES="$g" PYTHONPATH="$TREE" \
                        "$PY" tools/rdna2_028/verify_tunableop_lookup.py \
                        --rows "$TREE/tunableop/rocm7.14-rocblas5.5/tunableop_results$c.csv" \
                        --device 0 --dtype "$dt" --warmup 10 --reps 40 \
                        --out "$out/rows$c-card$g-$dt.json"
                done
            done
        ) > "$out/card$g.log" 2>&1 &
        pids+=($!)
    done
    wait "${pids[@]}"
    grep -h "lookup hits" "$out"/card*.log | sort | uniq -c | cut -c1-200
}

# ---- main ---------------------------------------------------------------------------
say "e_run.sh: stages [$STAGES], tree $(cat "$TREE/.sha"), run dir $RUN"
lease_take "eplb cross-rank top-k replay (E1-E3), stages $STAGES" "$LEASE_MIN"
lease_check_buses
start_observers
for stage in $STAGES; do
    stopped && { say "STOP present"; break; }
    case $stage in
        micro) micro_stage micro "$TREE/tunableop/rocm7.14-rocblas5.5" ;;
        microfix) micro_stage microfix "$AGREED/rocm7.14-rocblas5.5" ;;
        verify) verify_stage verify ;;
        vmatrix) vmatrix_stage vmatrix ;;
        *)
            boot_stage "$stage"
            why=$(stage_verdict "$stage")
            if [[ -n $why ]]; then
                say "stage $stage UNUSABLE: $why"
                touch "$RUN/STOP"
                break
            fi
            check "$stage"
            ;;
    esac
done
say "e_run.sh done"
