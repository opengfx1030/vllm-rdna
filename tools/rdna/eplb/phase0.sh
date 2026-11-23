#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Phase 0 of the lean EPLB (docs/rdna2/eplb-lean-design-2026-11-16.md): measure the
# expert load on the production TP4/EP4 Flash-Next server and change nothing. One exact
# SHA tree, one fresh boot per arm, the lease and the quiet windows of
# ~/w4a8_runs/lanes/lease/README.md (same protocol as tools/rdna/hetero_moe/t10_run.sh).
#
#   A1 A2   flashnext-mtp2, counters OFF                 ITL cells (the baseline)
#   B1 B2   flashnext-mtp2, lean counters ON (HIP)       ITL cells + per layer/rank T and P
#   Z1      flashnext-mtp0, counters OFF                 ITL cells
#   Y1      flashnext-mtp0, lean counters ON             ITL cells + T and P
#   L1 L2   flashnext-mtp2, lean counters ON + prefill   no cells; LOAD_SECONDS of agentic
#           histograms (HIST=512)                        traffic, SEED0 1 / 1001: the real
#                                                        routing, two independent halves
#   G       flashnext-mtp2, the OLD collector (fused Triton, VLLM_EXPERT_STATS_*) with the
#           top-k id trace of G_TRACE_LAYERS            agentic traffic: the id trace that
#                                                        simulate replay scores placements on
#                                                        (needs no rebuild)
#   P8 P1   flashnext-mtp2, lean counters ON (EVERY=1, RING=1M) under rocprofv3 --kernel-trace,
#           CG_MODE=NONE (tools/rdna/prof/serve_profile.sh), decode c=8 / c=1: which load
#           the MoE kernel time follows (objective.py)
#
# Cells (serve_validate.sh CELLS, in:out:conc): c=1,8 x 1k,16k by default; MTP2 = 3 rows
# per request, MTP0 = 1 row. The ITL gate is A against B, Z against Y.
#
# Default order A1 B1 B2 A2 Z1 Y1 L1 L2 G P8 P1 (ABBA first). Unattended; touches no GPU
# before GO exists:
#   setsid nohup env EXPECT_SHA=<sha> RUN=$HOME/w4a8_runs/eplb-p0 \
#       bash tools/rdna/eplb/phase0.sh > "$RUN/phase0.log" 2>&1 < /dev/null &
#   touch "$RUN/GO"      # GO may hold KEY=VALUE lines overriding ARMS, GPUS, PORT, ...
#   touch "$RUN/STOP"    # abort between arms / while waiting for the lease
# DRY=1 prints each arm's configuration (no tree check, no GPU, no lease).
# ANALYZE=1 only runs the CPU analysis of the arms that are DONE (no GPU, no lease).
#
# The lean-counter arms need a tree whose _rocm_C was built with rdna_eplb_stats (checked
# with strings on the .so); the A, Z and G arms run on any tree. Every arm goes through
# serve_validate.sh (boot lock, quiet windows, KFD eviction check per cell, probes, PPL,
# concurrency probe) except P8/P1 (serve_profile.sh: a profiling run is not a timing
# run; it takes the boot lock only). While the lease is held the junction temperatures of
# the leased GPUs are logged every 5 s to $RUN/temps.csv and the running server is
# aborted at >= 95 C, or two consecutive samples >= 90 C.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TREE=${TREE:-$(cd "$HERE/../../.." && pwd)}
RUN=${RUN:-$HOME/w4a8_runs/eplb-p0}
VENV=${VENV:-$HOME/Apps/vllm/venv-7.14.0_0.32.0}
PY=$VENV/bin/python
MODEL=${MODEL:-$HOME/.cache/huggingface/hub/models--wtdcode--Qwen3.8-Flash-Next-AWQ-W4A16/snapshots/0939125b929543a783ce700c90e36dd1a575c00c}
LEASE=${LEASE:-$HOME/w4a8_runs/lanes/lease}
BOOT_LOCK=${BOOT_LOCK:-$HOME/w4a8_runs/boot.lock}
OWNER=${OWNER:-EplbP0}
# Interim topology (README of the lease dir): HIP 2-5 = V620 d9 dc df e2, lease tag b4.
GPUS=${GPUS:-2,3,4,5}
LEASE_TAG=${LEASE_TAG:-b4}
WANT_BUSES=${WANT_BUSES:-d9 dc df e2}
PORT=${PORT:-18490}
ARMS=${ARMS:-A1 B1 B2 A2 Z1 Y1 L1 L2 G P8 P1}
MAX_TRIES=${MAX_TRIES:-2}
EXPECT_SHA=${EXPECT_SHA:-}
DRY=${DRY:-0}
ANALYZE=${ANALYZE:-0}
LOAD_SECONDS=${LOAD_SECONDS:-1800}
PROF_LOAD_S=${PROF_LOAD_S:-60}
CELLS=${CELLS:-"1024:512:1 1024:512:8 16384:1024:1 16384:1024:8"}
G_TRACE_LAYERS=${G_TRACE_LAYERS:-0,12,24,36,47}
G_TRACE_STEPS=${G_TRACE_STEPS:-8192}
EP=${EP:-4}
MEMLOCK_MIN_KB=${MEMLOCK_MIN_KB:-33554432}
LOAD_MAX=${LOAD_MAX:-4}
TEMPS_PERIOD=5

say() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
die() { say "ABORT: $*"; exit "${2:-2}"; }
stopped() { [[ -e $RUN/STOP ]]; }
aer_count() {
    sudo -n dmesg 2> /dev/null | grep -cE 'pcieport.*(Timeout|BadTLP|NonFatalErr)' || true
}

verify_tree() {
    [[ -n $EXPECT_SHA ]] || die "EXPECT_SHA is required"
    [[ $(cat "$TREE/.sha" 2> /dev/null) == "$EXPECT_SHA" ]] ||
        die "tree $TREE is not $EXPECT_SHA ($(cat "$TREE/.sha" 2> /dev/null))"
    local so
    for so in _C _rocm_C; do
        [[ -e $TREE/vllm/$so.abi3.so ]] || die "$TREE/vllm/$so.abi3.so is missing"
    done
    [[ -x $PY ]] || die "no python at $PY"
    [[ -d $MODEL ]] || die "no model at $MODEL"
}

# The lean-counter arms need the HIP ops; boot them only on a tree that has them (a
# silent torch fallback would put ~60 tiny kernels per MoE call into the ITL cells, and
# the arms set VLLM_RDNA_EPLB_STATS_KERNEL=hip, which refuses to boot without the ops).
tree_has_ops() {
    grep -aq rdna_eplb_stats "$TREE/vllm/_rocm_C.abi3.so" 2> /dev/null
}

verify_host() {
    local lim
    lim=$(ulimit -l)
    if [[ $lim != unlimited ]] && ((lim < MEMLOCK_MIN_KB)); then
        die "RLIMIT_MEMLOCK is $lim kB, the workers need >= $MEMLOCK_MIN_KB kB (fresh login shell)"
    fi
}

parse_go() {
    local k v
    while IFS='=' read -r k v; do
        case $k in
            GPUS | ARMS | PORT | MAX_TRIES | LOAD_SECONDS | CELLS | PROF_LOAD_S | G_TRACE_LAYERS)
                printf -v "$k" '%s' "$v"
                export "${k?}"
                say "GO override: $k=$v"
                ;;
        esac
    done < <(grep -E '^[A-Z_0-9]+=' "$RUN/GO" || true)
}

# ---- lease (same protocol as t10_run.sh) -----------------------------------
setup_gpus() {
    local g
    export HIP_VISIBLE_DEVICES=$GPUS
    unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
    for g in ${GPUS//,/ }; do
        [[ $g =~ ^[0-9]+$ ]] || die "GPUS=$GPUS is not a HIP index list"
    done
    REQUEST=$LEASE/$LEASE_TAG.request
    HOLDER=$LEASE/$OWNER.$LEASE_TAG
}

# Each visible GPU's PCI bus must be one of WANT_BUSES (the leased V620s); free GiB each.
probe_gpus() {
    WANT_BUSES="$WANT_BUSES" "$PY" -c '
import os, sys, torch
torch.cuda.init()
want = set(os.environ["WANT_BUSES"].lower().split())
def bus_of(i):
    b = torch.cuda.get_device_properties(i).pci_bus_id
    return f"{b:02x}" if isinstance(b, int) else str(b).split(":")[1].lower()


bus = [bus_of(i) for i in range(torch.cuda.device_count())]
free = [torch.cuda.mem_get_info(i)[0] / 2**30 for i in range(torch.cuda.device_count())]
print("buses", ",".join(bus), "free_gib", " ".join(f"{f:.1f}" for f in free))
sys.exit(0 if len(bus) == len(want) and set(bus) == want and min(free) >= 28 else 1)'
}

others_hold() {
    find "$LEASE" -maxdepth 1 -name "*.$LEASE_TAG" ! -name "$OWNER.*" | grep -q .
}

HOLD=0
take_lease() {
    until [[ ! -e $REQUEST ]] || grep -q "^$OWNER" "$REQUEST"; do
        stopped && die "STOP while waiting for $REQUEST" 3
        say "waiting: $REQUEST is $(head -c 100 "$REQUEST" | tr '\n' ' ')"
        sleep 60
    done
    echo "$OWNER $(date -Is) phase0.sh pid $$" > "$REQUEST"
    until ! others_hold && probe_gpus; do
        stopped && die "STOP while waiting for the lease" 3
        say "waiting: holders $(find "$LEASE" -maxdepth 1 -name '*.slot*' -printf '%f ')"
        sleep 60
    done
    echo "$OWNER $(date -Is) phase0.sh pid $$" > "$HOLDER"
    HOLD=1
    say "lease taken: $(find "$LEASE" -maxdepth 1 -type f -name '*.slot*' -printf '%f ')"
}

OBSERVERS=()
CUR_OUT=""
start_observers() {
    "$PY" "$TREE/tools/rdna/mono/w4a8/gpu_temps.py" --gpus "$GPUS" \
        --period "$TEMPS_PERIOD" --out "$RUN/temps.csv" &
    OBSERVERS+=($!)
    temp_guard &
    OBSERVERS+=($!)
    say "junction temperatures of HIP $GPUS every $TEMPS_PERIOD s -> $RUN/temps.csv"
}
stop_observers() {
    ((${#OBSERVERS[@]})) && kill "${OBSERVERS[@]}" 2> /dev/null
    OBSERVERS=()
    return 0
}

# Abort rule of the box rules, on the 5 s samples of temps.csv.
temp_guard() {
    local prev=""
    while sleep "$TEMPS_PERIOD"; do
        [[ -s $RUN/temps.csv ]] || continue
        local hot
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

# shellcheck disable=SC2329  # invoked by the EXIT trap
release_lease() {
    stop_observers
    if ((HOLD)); then
        if [[ -n $CUR_OUT && -s $CUR_OUT/serve.pid ]]; then
            kill -- -"$(cut -d' ' -f1 "$CUR_OUT/serve.pid")" 2> /dev/null || true
        fi
        rm -f "$HOLDER"
        HOLD=0
        say "lease released"
    fi
    if [[ -n ${REQUEST:-} ]] && grep -q "^$OWNER" "$REQUEST" 2> /dev/null; then
        rm -f "$REQUEST"
    fi
}
trap release_lease EXIT
trap 'exit 130' INT TERM

# ---- arms ------------------------------------------------------------------
# Sets ARM_DESC ARM_RECIPE ARM_KIND (validate|prof) ARM_CELLS ARM_HOOK_SECONDS
# ARM_HOOK_SEED0 ARM_STATS (the arm's environment, KEY=VALUE words) ARM_OVERRIDES
# (recipe KEY=VALUE words) ARM_CDEC (prof arms) and ARM_LEAN (1: needs the HIP ops).
arm_config() {
    local arm=$1 sdir=$RUN/arms/$1
    ARM_STATS=() ARM_OVERRIDES=() ARM_CELLS=$CELLS ARM_HOOK_SECONDS=0 ARM_HOOK_SEED0=1
    ARM_KIND=validate ARM_RECIPE=flashnext-mtp2 ARM_CDEC=0 ARM_LEAN=0
    local lean=(VLLM_RDNA_EPLB_STATS_DIR="$sdir/eplb" VLLM_RDNA_EPLB_STATS_INTERVAL_S=30
        VLLM_RDNA_EPLB_STATS_EVERY=1 VLLM_RDNA_EPLB_STATS_KERNEL=hip)
    case ${arm%[0-9]*} in
        A) ARM_DESC="mtp2, counters OFF" ;;
        B)
            ARM_DESC="mtp2, lean counters ON"
            ARM_STATS=("${lean[@]}")
            ARM_LEAN=1
            ;;
        Z)
            ARM_DESC="mtp0, counters OFF"
            ARM_RECIPE=flashnext-mtp0
            ;;
        Y)
            ARM_DESC="mtp0, lean counters ON"
            ARM_RECIPE=flashnext-mtp0
            ARM_STATS=("${lean[@]}")
            ARM_LEAN=1
            ;;
        L)
            ARM_DESC="mtp2, lean counters ON + prefill histograms, agentic ${LOAD_SECONDS}s"
            ARM_STATS=("${lean[@]}" VLLM_RDNA_EPLB_STATS_HIST=512)
            ARM_LEAN=1
            ARM_CELLS=none
            ARM_HOOK_SECONDS=$LOAD_SECONDS
            if [[ $arm == L2 ]]; then ARM_HOOK_SEED0=1001; fi
            ;;
        G)
            ARM_DESC="mtp2, old collector id trace of layers $G_TRACE_LAYERS, agentic ${LOAD_SECONDS}s"
            ARM_STATS=(VLLM_EXPERT_STATS_DIR="$sdir/stats" VLLM_EXPERT_STATS_INTERVAL_S=30
                VLLM_EXPERT_STATS_KERNEL=fused
                VLLM_EXPERT_STATS_TRACE_LAYERS="$G_TRACE_LAYERS"
                VLLM_EXPERT_STATS_TRACE_STEPS="$G_TRACE_STEPS")
            ARM_CELLS=none
            ARM_HOOK_SECONDS=$LOAD_SECONDS
            ;;
        P)
            ARM_KIND=prof
            ARM_CDEC=${arm#P}
            ARM_DESC="mtp2, lean counters ON under rocprofv3 --kernel-trace, CG_MODE=NONE, decode c=$ARM_CDEC"
            ARM_LEAN=1
            ARM_OVERRIDES=(CG_MODE=NONE)
            ARM_STATS=(VLLM_RDNA_EPLB_STATS_DIR="$sdir/eplb"
                VLLM_RDNA_EPLB_STATS_INTERVAL_S=5 VLLM_RDNA_EPLB_STATS_EVERY=1
                VLLM_RDNA_EPLB_STATS_RING=1048576 VLLM_RDNA_EPLB_STATS_KERNEL=hip)
            ;;
        *) return 1 ;;
    esac
}

stash_attempt() {
    local out=$1 k=1 f
    [[ -e $out/serve.log ]] || return 0
    while [[ -e $out/attempt$k ]]; do k=$((k + 1)); done
    mkdir -p "$out/attempt$k"
    for f in "$out"/*; do
        case ${f##*/} in attempt* | DONE) continue ;; esac
        mv "$f" "$out/attempt$k/"
    done
    say "stashed $out attempt$k"
}

# /metrics poll (log only): one line per 10 s of the counters the token check needs.
poll_metrics() {
    local out=$1
    while sleep 10; do
        curl -sf --max-time 5 "http://127.0.0.1:$PORT/metrics" -o "$out/metrics-last.tmp" || continue
        mv "$out/metrics-last.tmp" "$out/metrics-last.txt"
        {
            printf 'epoch %s\n' "$(date +%s)"
            grep -E '^vllm:(iteration_tokens_total_sum|prompt_tokens_total|generation_tokens_total|prompt_tokens_cached_total|spec_decode_num_[a-z_]+_total)' \
                "$out/metrics-last.txt"
        } >> "$out/metrics-poll.txt"
    done
}

# The arm's own statistics environment and nothing of an earlier arm's.
clean_env() {
    unset VLLM_EXPERT_STATS_DIR VLLM_EXPERT_STATS_INTERVAL_S VLLM_EXPERT_STATS_PAIRS \
        VLLM_EXPERT_STATS_ALL_RANKS VLLM_EXPERT_STATS_KERNEL \
        VLLM_EXPERT_STATS_TRACE_LAYERS VLLM_EXPERT_STATS_TRACE_STEPS \
        VLLM_RDNA_EPLB_STATS_DIR VLLM_RDNA_EPLB_STATS_RING VLLM_RDNA_EPLB_STATS_HIST \
        VLLM_RDNA_EPLB_STATS_EVERY VLLM_RDNA_EPLB_STATS_INTERVAL_S \
        VLLM_RDNA_EPLB_STATS_DECODE_MAX VLLM_RDNA_EPLB_STATS_KERNEL
}

boot_arm() { # arm out cache
    local arm=$1 out=$2 cache=$3 envargs
    envargs=(
        RECIPE="$ARM_RECIPE" MODEL="$MODEL" GPUS="$GPUS" PORT="$PORT"
        TAG="eplb-p0-$arm" OUT="$out" CACHE="$cache" VENV="$VENV" CELLS="$ARM_CELLS"
        CONC_PROBE=1 BOOT_LOCK="$BOOT_LOCK" HIP_VISIBLE_DEVICES="$GPUS"
        BENCH_EXTRA="--temperature 0" QUIET_HOLD=1 QUIET_OWN="$OWNER" STOP_GRACE=90
        READY_TIMEOUT=3600
    )
    if ((ARM_HOOK_SECONDS > 0)); then
        envargs+=(LOAD_HOOK="$HERE/load_hook.sh" LOAD_SECONDS="$ARM_HOOK_SECONDS"
            SEED0="$ARM_HOOK_SEED0")
    fi
    (
        clean_env
        cd "$TREE" &&
            env "${ARM_STATS[@]}" "${envargs[@]}" \
                bash tools/rdna/port_v031/serve_validate.sh "${ARM_OVERRIDES[@]}"
    )
}

# P arms: the profiling launcher. It removes its own output directory first, so the
# counters' directory sits next to it ($RUN/arms/P8.eplb); the kernel trace is
# $RUN/arms/P8/trace/*_kernel_trace.csv, one file per process, named by pid like the
# eplb_stats-rank<R>-pid<P>.npz it is joined with.
boot_prof_arm() { # arm out
    local arm=$1 out=$2 prof=$RUN/prof
    (
        clean_env
        cd "$TREE" &&
            env "${ARM_STATS[@]}" ROCPROFILER_REGISTER_ENABLED=1 \
                OUT_ROOT="$prof" VENV="$VENV" MODEL="$MODEL" BOOT_LOCK="$BOOT_LOCK" \
                CDEC="$ARM_CDEC" DEC_NUM=$((ARM_CDEC * 4)) LOAD_S="$PROF_LOAD_S" INJECT=0 \
                TRACE_OPTS=--kernel-trace \
                bash tools/rdna/prof/serve_profile.sh "$arm" "$GPUS" "$PORT" rdna \
                RECIPE="$ARM_RECIPE" "${ARM_OVERRIDES[@]}" > "$out/prof.log" 2>&1
    )
}

# Why an arm's data cannot be used ("" = fine).
arm_verdict() { # arm out rc aer0 aer1
    local arm=$1 out=$2 rc=$3 aer0=$4 aer1=$5 files slog=$2/serve.log
    [[ $ARM_KIND == prof ]] && slog=$RUN/prof/$arm/serve.log
    if [[ $ARM_KIND == prof ]]; then
        ((rc == 0)) || { echo "profile run failed (rc=$rc)"; return; }
        grep -aq "FINISHED" "$out/prof.log" || { echo "profile run did not finish"; return; }
    else
        if grep -aq "rdna_ar: WEDGED" "$slog"; then
            echo "rdna_ar wedged"
            return
        elif ((rc != 0)); then
            echo "boot failed (rc=$rc)"
            return
        elif ((ARM_HOOK_SECONDS > 0)) && ! grep -q "LOAD HOOK rc=0" "$out/summary.txt"; then
            echo "load hook failed"
            return
        elif ((aer1 > aer0)); then
            echo "PCIe AER: $((aer1 - aer0)) new line(s)"
            return
        elif grep -aq "INVALID" "$out/summary.txt"; then
            echo "a cell had KFD evictions (INVALID)"
            return
        fi
    fi
    case ${ARM_STATS[*]:-} in
        *VLLM_RDNA_EPLB_STATS_DIR*)
            grep -aq "EPLB load counters are ON" "$slog" ||
                { echo "counters ON arm without the startup line"; return; }
            files=$(find "${ARM_STATS[0]#*=}" -name 'eplb_stats-rank*-pid*.npz' \
                2> /dev/null | wc -l)
            ((files == EP)) || { echo "$files counter files, expected $EP"; return; }
            ;;
        *VLLM_EXPERT_STATS_DIR*)
            grep -aq "MoE expert statistics are ON" "$slog" ||
                { echo "old collector ON arm without the startup line"; return; }
            ;;
        *)
            if grep -aq "EPLB load counters are ON\|MoE expert statistics are ON" "$slog"; then
                echo "counters OFF arm shows a collector"
                return
            fi
            ;;
    esac
}

run_arm() {
    local arm=$1 out=$RUN/arms/$1 cache try rc aer0 aer1 why poller
    arm_config "$arm" || { say "unknown arm $arm"; return 2; }
    cache=$RUN/cache/$(cut -c1-12 "$TREE/.sha")
    if [[ -e $out/DONE ]]; then
        say "$arm: already DONE"
        return 0
    fi
    if ((ARM_LEAN)) && ! tree_has_ops; then
        say "$arm: $TREE/vllm/_rocm_C.abi3.so has no rdna_eplb_stats (rebuild first): skipped"
        return 1
    fi
    mkdir -p "$out"
    CUR_OUT=$out
    for try in $(seq 1 "$MAX_TRIES"); do
        stopped && { say "STOP present"; return 3; }
        stash_attempt "$out"
        rm -rf "$cache/vllm" "$out/stats" "$out/eplb" "$RUN/prof/$arm"
        until probe_gpus > "$out/gpus.txt"; do
            stopped && { say "STOP present"; return 3; }
            say "waiting for the GPUs: $(cat "$out/gpus.txt")"
            sleep 30
        done
        # a compile/build on the box (the 1-min load of the 96 threads) would land in
        # the ITL cells: boot only on a quiet host (LOAD_MAX; boots anyway after 2 h)
        t_load=$(date +%s)
        until awk -v m="$LOAD_MAX" '{ exit !($1 < m) }' /proc/loadavg ||
            (($(date +%s) - t_load > 7200)); do
            stopped && { say "STOP present"; return 3; }
            say "host busy: load $(cut -d' ' -f1-3 /proc/loadavg): waiting"
            sleep 60
        done
        ss -ltnH "sport = :$PORT" 2> /dev/null | grep -q . && die "port $PORT is busy"
        aer0=$(aer_count)
        {
            echo "arm=$arm try=$try desc=$ARM_DESC"
            echo "sha=$(cat "$TREE/.sha") tree=$TREE kind=$ARM_KIND recipe=$ARM_RECIPE"
            echo "stats_env=${ARM_STATS[*]:-none} cells=$ARM_CELLS hook_s=$ARM_HOOK_SECONDS"
            echo "load=$(cut -d' ' -f1-3 /proc/loadavg) aer=$aer0 date=$(date -Is)"
            echo "memlock=$(ulimit -l)"
        } >> "$out/arm.txt"
        say "arm $arm try $try start ($ARM_DESC)"
        poll_metrics "$out" &
        poller=$!
        if [[ $ARM_KIND == prof ]]; then
            boot_prof_arm "$arm" "$out"
        else
            boot_arm "$arm" "$out" "$cache"
        fi
        rc=$?
        kill "$poller" 2> /dev/null
        wait "$poller" 2> /dev/null
        aer1=$(aer_count)
        echo "rc=$rc aer_end=$aer1 end=$(date -Is)" >> "$out/arm.txt"
        say "arm $arm try $try rc=$rc:" \
            "$(grep -aE 'LOAD HOOK|PROBE|PPL|PREFIX|CONCURRENCY|READY|BOOT FAILED|INVALID' \
                "$out/summary.txt" 2> /dev/null | tr '\n' ' ')"
        why=$(arm_verdict "$arm" "$out" "$rc" "$aer0" "$aer1")
        if [[ -n $why ]]; then
            say "arm $arm try $try discarded: $why"
            continue
        fi
        date -Is > "$out/DONE"
        say "arm $arm DONE"
        return 0
    done
    say "arm $arm failed after $MAX_TRIES tries"
    return 1
}

# ---- CPU analysis of the arms that are DONE ----------------------------------
counters_dir() { # arm
    echo "$RUN/arms/$1/eplb"
}

analyze() {
    local arm dir out
    export PYTHONPATH=$TREE
    cd "$TREE" || return 1
    for arm in B1 B2 Y1 L1 L2 P8 P1; do
        [[ -e $RUN/arms/$arm/DONE ]] || continue
        dir=$(counters_dir "$arm")
        out=$RUN/arms/$arm
        say "summary $arm"
        "$PY" -m tools.rdna.eplb.summary "$dir" --window 1000 \
            --json "$out/eplb-summary.json" --export-stats "$out/eplb-stats.json" \
            > "$out/eplb-summary.txt" 2>&1 || say "summary $arm: rc=$? (see $out/eplb-summary.txt)"
    done
    if [[ -e $RUN/arms/A1/DONE && -e $RUN/arms/A2/DONE && -e $RUN/arms/B1/DONE && -e $RUN/arms/B2/DONE ]]; then
        "$PY" -m tools.rdna.eplb.itl_gate --off "$RUN/arms/A1" "$RUN/arms/A2" \
            --on "$RUN/arms/B1" "$RUN/arms/B2" --json "$RUN/itl-gate-mtp2.json" \
            > "$RUN/itl-gate-mtp2.txt" 2>&1
        say "ITL gate mtp2: rc=$? ($RUN/itl-gate-mtp2.txt)"
    fi
    if [[ -e $RUN/arms/Z1/DONE && -e $RUN/arms/Y1/DONE ]]; then
        "$PY" -m tools.rdna.eplb.itl_gate --off "$RUN/arms/Z1" --on "$RUN/arms/Y1" \
            --json "$RUN/itl-gate-mtp0.json" > "$RUN/itl-gate-mtp0.txt" 2>&1
        say "ITL gate mtp0 (one off arm: no noise estimate): rc=$? ($RUN/itl-gate-mtp0.txt)"
    fi
    for arm in P8 P1; do
        [[ -e $RUN/arms/$arm/DONE ]] || continue
        "$PY" -m tools.rdna.eplb.objective --stats "$(counters_dir "$arm")" \
            --trace "$RUN/prof/$arm/trace" --rows $((${arm#P} * 3)) \
            > "$RUN/arms/$arm/objective.txt" 2>&1
        say "objective $arm: rc=$? ($RUN/arms/$arm/objective.txt)"
    done
    if [[ -e $RUN/arms/L1/DONE && -e $RUN/arms/L2/DONE ]]; then
        "$PY" -m tools.rdna.eplb.solve --train "$RUN/arms/L1/eplb-stats.json" \
            --test "$RUN/arms/L2/eplb-stats.json" --counters "$(counters_dir L1)" \
            --ep "$EP" --objective touch --rows 24 --out "$RUN/placement-L1.json" \
            --report "$RUN/placement-L1-report.json" > "$RUN/placement-L1.txt" 2>&1
        say "solve (train L1, test L2): rc=$? ($RUN/placement-L1.txt)"
        "$PY" -m tools.rdna.eplb.simulate replay --ep "$EP" --objective pairs \
            --hist "$(find "$(counters_dir L1)" -name 'eplb_stats-rank0-pid*.npz' | head -1)" \
            --folds 2 --json "$RUN/replay-prefill-L1.json" > "$RUN/replay-prefill-L1.txt" 2>&1
        say "prefill replay: rc=$? ($RUN/replay-prefill-L1.txt)"
    fi
    if [[ -e $RUN/arms/G/DONE ]]; then
        local trace
        trace=$(find "$RUN/arms/G/stats" -name '*.trace.npz' | head -1)
        "$PY" -m tools.rdna.eplb.simulate replay --ep "$EP" --objective touch \
            --id-trace "$trace" --folds 2 --json "$RUN/replay-decode-G.json" \
            > "$RUN/replay-decode-G.txt" 2>&1
        say "decode replay: rc=$? ($RUN/replay-decode-G.txt)"
    fi
}

dry_run() {
    local arm
    for arm in $ARMS; do
        arm_config "$arm" || die "unknown arm $arm"
        say "arm $arm: $ARM_DESC; kind $ARM_KIND recipe $ARM_RECIPE cells [$ARM_CELLS]" \
            "hook ${ARM_HOOK_SECONDS}s seed0 $ARM_HOOK_SEED0 lean $ARM_LEAN"
        say "    env: ${ARM_STATS[*]:-none}; recipe overrides: ${ARM_OVERRIDES[*]:-none}"
    done
}

mkdir -p "$RUN/arms" "$RUN/cache"
if [[ $DRY == 1 ]]; then
    dry_run
    exit 0
fi
if [[ $ANALYZE == 1 ]]; then
    analyze
    exit 0
fi
verify_tree
verify_host

say "tree $TREE sha $EXPECT_SHA; waiting for $RUN/GO (arms: $ARMS)"
until [[ -e $RUN/GO ]]; do
    stopped && die "STOP before GO" 3
    sleep 30
done
parse_go
setup_gpus
say "GO: GPUS=$GPUS ARMS=$ARMS PORT=$PORT LOAD_SECONDS=$LOAD_SECONDS CELLS=$CELLS"
take_lease
start_observers

status=0
for arm in $ARMS; do
    run_arm "$arm"
    rc=$?
    CUR_OUT=""
    ((rc == 3)) && { status=3; break; }
    ((rc != 0)) && status=1
done

CUR_OUT=""
stop_observers
analyze
date -Is > "$RUN/DONE"
say "phase 0 finished status=$status; arms in $RUN/arms/, analysis in $RUN/"
exit "$status"
