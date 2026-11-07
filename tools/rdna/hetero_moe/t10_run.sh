#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# T10 of the hetero-MoE spec: the expert-statistics collector on the production
# flashnext-mtp2 TP4 server (V620 slot B, HIP 6-9, compiled mode, no monokernel),
# one exact-SHA tree, one fresh boot per arm. Docs: docs/rdna2/expert-stats-t10-2026-11-07.md.
#
#   A1 A2   stats OFF (VLLM_EXPERT_STATS_DIR unset)            ITL cells
#   B1 B2   stats ON, pairs on (the collection configuration)   ITL cells;
#           B1 then drives LOAD_SECONDS (default 3600) of agentic_bench traffic
#   C       stats ON on EVERY TP rank (VLLM_EXPERT_STATS_ALL_RANKS=1), short load:
#           rank agreement
#
# Default order A1 B1 B2 A2 C (ABBA, then the rank check). Unattended; touches no GPU
# before GO exists:
#   setsid nohup env EXPECT_SHA=<sha> RUN=$HOME/w4a8_runs/expert-stats-t10 \
#       bash tools/rdna/hetero_moe/t10_run.sh > "$RUN/t10.log" 2>&1 < /dev/null &
#   touch "$RUN/GO"      # GO may hold KEY=VALUE lines overriding ARMS, GPUS, PORT, ...
#   touch "$RUN/STOP"    # abort between arms / while waiting for the lease
# DRY=1 verifies the tree and prints each arm's launch line (no GPU, no lease).
#
# It takes the slot B lease (lanes/lease/README.md: slotB.request, then
# $OWNER.slotB once no other holder exists and every GPU has >= 28 GiB free) and
# resolves HIP -> PCI bus first. Every boot goes through serve_validate.sh (boot
# lock, quiet windows, KFD eviction check per cell, probes, PPL, concurrency probe).
# While it holds the lease it logs the junction temperatures every 5 s to
# $RUN/temps.csv and aborts the running server when one GPU reads >= 95 C or two
# consecutive samples of one GPU read >= 90 C.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TREE=${TREE:-$(cd "$HERE/../../.." && pwd)}
RUN=${RUN:-$HOME/w4a8_runs/expert-stats-t10}
VENV=${VENV:-$HOME/Apps/vllm/venv-7.14.0_0.32.0}
PY=$VENV/bin/python
MODEL=${MODEL:-$HOME/.cache/huggingface/hub/models--wtdcode--Qwen3.8-Flash-Next-AWQ-W4A16/snapshots/0939125b929543a783ce700c90e36dd1a575c00c}
LEASE=${LEASE:-$HOME/w4a8_runs/lanes/lease}
BOOT_LOCK=${BOOT_LOCK:-$HOME/w4a8_runs/boot.lock}
OWNER=${OWNER:-ExpertStats}
GPUS=${GPUS:-6,7,8,9}
PORT=${PORT:-18480}
ARMS=${ARMS:-A1 B1 B2 A2 C}
MAX_TRIES=${MAX_TRIES:-2}
EXPECT_SHA=${EXPECT_SHA:-}
DRY=${DRY:-0}
LOAD_SECONDS=${LOAD_SECONDS:-3600}
CELLS=${CELLS:-"1024:512:1 1024:512:6 1024:512:8 16384:1024:1 16384:1024:8"}
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
            GPUS | ARMS | PORT | MAX_TRIES | LOAD_SECONDS | CELLS)
                printf -v "$k" '%s' "$v"
                export "${k?}"
                say "GO override: $k=$v"
                ;;
        esac
    done < <(grep -E '^[A-Z_0-9]+=' "$RUN/GO" || true)
}

# ---- lease -----------------------------------------------------------------
setup_gpus() {
    local g
    [[ $GPUS == 6,7,8,9 ]] || die "GPUS=$GPUS: slot B is HIP 6,7,8,9"
    for g in ${GPUS//,/ }; do
        [[ $g =~ ^[6-9]$ ]] || die "GPU $g is not in 6-9"
    done
    export HIP_VISIBLE_DEVICES=$GPUS
    unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
    REQUEST=$LEASE/slotB.request
    HOLDER=$LEASE/$OWNER.slotB
}

# Each visible GPU's PCI bus must be a slot-B V620 (0a, 0f, 14, 17); free GiB each.
probe_gpus() {
    "$PY" -c '
import sys, torch
want = {"0a", "0f", "14", "17"}
bus = [torch.cuda.get_device_properties(i).pci_bus_id.split(":")[1].lower()
       for i in range(torch.cuda.device_count())]
free = [torch.cuda.mem_get_info(i)[0] / 2**30 for i in range(torch.cuda.device_count())]
print("buses", ",".join(bus), "free_gib", " ".join(f"{f:.1f}" for f in free))
sys.exit(0 if len(bus) == 4 and set(bus) == want and min(free) >= 28 else 1)'
}

others_hold() {
    find "$LEASE" -maxdepth 1 -name '*.slotB' ! -name "$OWNER.*" | grep -q .
}

HOLD=0
take_lease() {
    until [[ ! -e $REQUEST ]] || grep -q "^$OWNER" "$REQUEST"; do
        stopped && die "STOP while waiting for $REQUEST" 3
        say "waiting: $REQUEST is $(head -c 100 "$REQUEST" | tr '\n' ' ')"
        sleep 60
    done
    echo "$OWNER $(date -Is) t10_run.sh pid $$" > "$REQUEST"
    until ! others_hold && probe_gpus; do
        stopped && die "STOP while waiting for the lease" 3
        say "waiting: holders $(find "$LEASE" -maxdepth 1 -name '*.slot*' -printf '%f ')"
        sleep 60
    done
    echo "$OWNER $(date -Is) t10_run.sh pid $$" > "$HOLDER"
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
# Sets ARM_DESC, ARM_STATS (env of the stats collector), ARM_CELLS, ARM_HOOK_SECONDS.
arm_config() {
    ARM_STATS=() ARM_CELLS=$CELLS ARM_HOOK_SECONDS=0
    local sdir=$RUN/arms/$1/stats
    case ${1%[12]} in
        A) ARM_DESC="stats OFF" ;;
        B)
            ARM_DESC="stats ON (pairs)"
            ARM_STATS=(VLLM_EXPERT_STATS_DIR="$sdir" VLLM_EXPERT_STATS_INTERVAL_S=30
                VLLM_EXPERT_STATS_PAIRS=1)
            if [[ $1 == B1 ]]; then ARM_HOOK_SECONDS=$LOAD_SECONDS; fi
            ;;
        C)
            ARM_DESC="stats ON, every TP rank records"
            ARM_STATS=(VLLM_EXPERT_STATS_DIR="$sdir" VLLM_EXPERT_STATS_INTERVAL_S=30
                VLLM_EXPERT_STATS_ALL_RANKS=1)
            ARM_CELLS="1024:512:8 16384:1024:1"
            ARM_HOOK_SECONDS=${C_LOAD_SECONDS:-600}
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
            grep -E '^vllm:(iteration_tokens_total_sum|prompt_tokens_total|generation_tokens_total|prompt_tokens_cached_total|prompt_tokens_recomputed_total|spec_decode_num_[a-z_]+_total)' \
                "$out/metrics-last.txt"
        } >> "$out/metrics-poll.txt"
    done
}

boot_arm() { # arm out cache
    local arm=$1 out=$2 cache=$3 envargs
    envargs=(
        RECIPE=flashnext-mtp2 MODEL="$MODEL" GPUS="$GPUS" PORT="$PORT"
        TAG="t10-$arm" OUT="$out" CACHE="$cache" VENV="$VENV" CELLS="$ARM_CELLS"
        CONC_PROBE=1 BOOT_LOCK="$BOOT_LOCK" HIP_VISIBLE_DEVICES="$GPUS"
        BENCH_EXTRA="--temperature 0" QUIET_HOLD=1 QUIET_OWN="$OWNER" STOP_GRACE=90
        READY_TIMEOUT=3600
    )
    if ((ARM_HOOK_SECONDS > 0)); then
        envargs+=(LOAD_HOOK="$HERE/t10_load_hook.sh" LOAD_SECONDS="$ARM_HOOK_SECONDS")
    fi
    (
        # the arm's stats environment, and nothing of an earlier arm's
        unset VLLM_EXPERT_STATS_DIR VLLM_EXPERT_STATS_INTERVAL_S \
            VLLM_EXPERT_STATS_PAIRS VLLM_EXPERT_STATS_ALL_RANKS
        cd "$TREE" &&
            env "${ARM_STATS[@]}" "${envargs[@]}" \
                bash tools/rdna/port_v031/serve_validate.sh
    )
}

run_arm() {
    local arm=$1 out=$RUN/arms/$1 cache try rc aer0 aer1 why poller
    arm_config "$arm" || { say "unknown arm $arm"; return 2; }
    cache=$RUN/cache/$(cut -c1-12 "$TREE/.sha")
    if [[ -e $out/DONE ]]; then
        say "$arm: already DONE"
        return 0
    fi
    mkdir -p "$out"
    CUR_OUT=$out
    for try in $(seq 1 "$MAX_TRIES"); do
        stopped && { say "STOP present"; return 3; }
        stash_attempt "$out"
        rm -rf "$cache/vllm" "$out/stats"
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
            echo "sha=$(cat "$TREE/.sha") tree=$TREE"
            echo "stats_env=${ARM_STATS[*]:-none} cells=$ARM_CELLS hook_s=$ARM_HOOK_SECONDS"
            echo "load=$(cut -d' ' -f1-3 /proc/loadavg) aer=$aer0 date=$(date -Is)"
            echo "memlock=$(ulimit -l)"
        } >> "$out/arm.txt"
        say "arm $arm try $try start ($ARM_DESC)"
        poll_metrics "$out" &
        poller=$!
        boot_arm "$arm" "$out" "$cache"
        rc=$?
        kill "$poller" 2> /dev/null
        wait "$poller" 2> /dev/null
        aer1=$(aer_count)
        echo "rc=$rc aer_end=$aer1 end=$(date -Is)" >> "$out/arm.txt"
        say "arm $arm try $try rc=$rc:" \
            "$(grep -aE 'LOAD HOOK|PROBE|PPL|PREFIX|CONCURRENCY|READY|BOOT FAILED' \
                "$out/summary.txt" 2> /dev/null | tr '\n' ' ')"
        why=""
        if grep -aq "rdna_ar: WEDGED" "$out/serve.log"; then
            why="rdna_ar wedged"
        elif ((rc != 0)); then
            why="boot failed (rc=$rc)"
        elif ((ARM_HOOK_SECONDS > 0)) && ! grep -q "LOAD HOOK rc=0" "$out/summary.txt"; then
            why="load hook failed"
        elif ((aer1 > aer0)); then
            why="PCIe AER: $((aer1 - aer0)) new line(s)"
        elif grep -aq "INVALID" "$out/summary.txt"; then
            why="a cell had KFD evictions (INVALID)"
        elif [[ ${ARM_STATS[*]:-} == *EXPERT_STATS_DIR* ]] &&
            ! grep -aq "MoE expert statistics are ON" "$out/serve.log"; then
            why="stats ON arm without the collector's startup line"
        elif [[ -z ${ARM_STATS[*]:-} ]] &&
            grep -aq "MoE expert statistics are ON" "$out/serve.log"; then
            why="stats OFF arm shows the collector"
        fi
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

dry_run() {
    local arm
    for arm in $ARMS; do
        arm_config "$arm" || die "unknown arm $arm"
        say "arm $arm: $ARM_DESC; stats env: ${ARM_STATS[*]:-none}; cells: $ARM_CELLS; hook ${ARM_HOOK_SECONDS}s"
    done
}

mkdir -p "$RUN/arms" "$RUN/cache"
verify_tree
if [[ $DRY == 1 ]]; then
    dry_run
    exit 0
fi
verify_host

say "tree $TREE sha $EXPECT_SHA; waiting for $RUN/GO (arms: $ARMS)"
until [[ -e $RUN/GO ]]; do
    stopped && die "STOP before GO" 3
    sleep 30
done
parse_go
setup_gpus
say "GO: GPUS=$GPUS ARMS=$ARMS PORT=$PORT LOAD_SECONDS=$LOAD_SECONDS"
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
date -Is > "$RUN/DONE"
say "t10 finished status=$status; arms in $RUN/arms/"
exit "$status"
