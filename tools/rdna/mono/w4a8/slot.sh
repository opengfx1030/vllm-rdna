#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# GPU slot for --monokernel next to the W4A8 MoE prefill (VLLM_RDNA2_W4A8_SDOT4=1):
# TP4 Flash-Next, recipe flashnext-mtp2, one exact-SHA tree, one boot per arm.
#
#   a   production: W4A8 prefill, no --monokernel (the recipe's 2 HW queues)
#   b   --monokernel, VLLM_RDNA2_W4A8_SDOT4=0 (mono raises the queues to 4)
#   c   --monokernel + W4A8 (4 HW queues)
#   d   --monokernel + W4A8, rocm_mono_decode.hw_queues=0 (the recipe's 2 queues)
#   e   production at GPU_MAX_HW_QUEUES=4 (queue-attribution control, opt-in)
#   a0  production with the recipe's PLE (only differs from a after an OOM fallback)
#   a suffix 2 repeats an arm (c2, a2): the boot-to-boot noise floor.
#
# Unattended. Start it any time, it touches no GPU before GO exists:
#   setsid nohup bash tools/rdna/mono/w4a8/slot.sh > "$RUN/slot.log" 2>&1 \
#       < /dev/null &
#   touch "$RUN/GO"     # the window is open (GO may hold KEY=VALUE lines that
#                       # override GPUS, ARMS, PORT, MAX_TRIES, PLE_MODE, MONO_*)
#   touch "$RUN/STOP"   # abort: between arms and while waiting for the lease
# After GO it takes the lease (lanes/lease/README.md: hipF-L.request, then waits
# for no other holder and >= 28 GiB free per card), runs the arms, writes
# $RUN/results/{tables.md,results.json}, releases the lease, touches
# lanes/hip<F><L>.free and lanes/hip<F><L>.monow4a8.done, and writes $RUN/DONE.
#
# Required: EXPECT_SHA (must equal $TREE/.sha). Arm outputs: $RUN/arms/<arm>/.
# An attempt that wedges rdna_ar, boots badly, fails the hook, evicts, or sees a
# new PCIe AER line is stashed in attempt<k>/ and the arm is measured again
# (MAX_TRIES). DRY=1 verifies the tree, prints each arm's launch line and exits
# (no GPU, no lease).
#
# Start it from a fresh login shell: the workers mlock their pinned PLE table
# (23.84 GiB each) and need RLIMIT_MEMLOCK >= MEMLOCK_MIN_KB (it refuses a lower
# one). While it holds the lease it logs each leased GPU's junction temperature
# every TEMPS_PERIOD s to $RUN/temps.csv and the KFD queues of the running arm's
# workers every QUEUES_PERIOD s to $RUN/queues.jsonl (log only).
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TREE=${TREE:-$(cd "$HERE/../../../.." && pwd)}
RUN=${RUN:-$HOME/w4a8_runs/mono-w4a8}
VENV=${VENV:-$HOME/Apps/vllm/venv-7.14.0_0.32.0}
PY=$VENV/bin/python
MODEL=${MODEL:-$HOME/.cache/huggingface/hub/models--wtdcode--Qwen3.8-Flash-Next-AWQ-W4A16/snapshots/0939125b929543a783ce700c90e36dd1a575c00c}
LEASE=${LEASE:-$HOME/w4a8_runs/lanes/lease}
LANES=${LANES:-$HOME/w4a8_runs/lanes}
BOOT_LOCK=${BOOT_LOCK:-$HOME/w4a8_runs/boot.lock}
OWNER=${OWNER:-MonoW4A8}
GPUS=${GPUS:-2,3,4,5}
PORT=${PORT:-18470}
ARMS=${ARMS:-c a b d c2 a2}
MAX_TRIES=${MAX_TRIES:-2}
PLE_MODE=${PLE_MODE:-}
PLE_QUANT_DIR=${PLE_QUANT_DIR:-$HOME/.cache/huggingface/hub/models--primitive-ai--Qwen3.8-Flash-Next-PLE-quant/snapshots/4f861b63f69e61bfc2e22130ec91ec67f03ec43e/ples_int4}
EXPECT_SHA=${EXPECT_SHA:-}
DRY=${DRY:-0}
TEMPS_PERIOD=${TEMPS_PERIOD:-15}
QUEUES_PERIOD=${QUEUES_PERIOD:-20}
MEMLOCK_MIN_KB=${MEMLOCK_MIN_KB:-33554432}
export MONO_WINDOWS=${MONO_WINDOWS:-$HOME/w4a8_runs/w4a8-quality-2026-11-04/windows.json}
export MONO_PHASES=${MONO_PHASES:-perf,parity,topk}
export MONO_TOPK_LIMIT=${MONO_TOPK_LIMIT:-32}
export MONO_TOPK_REPEAT=${MONO_TOPK_REPEAT:-16}
export MONO_CELLS=${MONO_CELLS:-}
export MONO_CELL_TRIES=${MONO_CELL_TRIES:-3}
export MONO_TP=4 PY

say() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
die() { say "ABORT: $*"; exit "${2:-2}"; }
stopped() { [[ -e $RUN/STOP ]]; }
aer_count() {
    sudo -n dmesg 2>/dev/null | grep -cE 'pcieport.*(Timeout|BadTLP|NonFatalErr)' || true
}

# The workers mlock their pinned PLE table (23.84 GiB per TP4 rank). Under a lower
# RLIMIT_MEMLOCK they serve without the lock, and host page reclaim can evict
# their GPU queues for seconds (docs/rdna/serving.md): the limit limits.d sets
# only exists in a fresh login shell.
verify_host() {
    local lim
    lim=$(ulimit -l)
    if [[ $lim != unlimited ]] && ((lim < MEMLOCK_MIN_KB)); then
        die "RLIMIT_MEMLOCK is $lim kB, the workers need >= $MEMLOCK_MIN_KB kB: start from a fresh login shell (docs/rdna/serving.md)"
    fi
}

verify_tree() {
    [[ -n $EXPECT_SHA ]] || die "EXPECT_SHA is required"
    [[ $(cat "$TREE/.sha" 2>/dev/null) == "$EXPECT_SHA" ]] ||
        die "tree $TREE is not $EXPECT_SHA ($(cat "$TREE/.sha" 2>/dev/null))"
    local so
    for so in _C _rocm_C; do
        [[ -e $TREE/vllm/$so.abi3.so ]] || die "$TREE/vllm/$so.abi3.so is missing"
    done
    [[ -x $PY ]] || die "no python at $PY"
    [[ -d $MODEL ]] || die "no model at $MODEL"
    case ",$MONO_PHASES," in
        *,topk,*) [[ -f $MONO_WINDOWS ]] || die "no windows file $MONO_WINDOWS" ;;
    esac
}

# Overrides from the GO file (KEY=VALUE lines, whitelisted).
parse_go() {
    local k v
    while IFS='=' read -r k v; do
        case $k in
            GPUS | ARMS | PORT | MAX_TRIES | PLE_MODE | PLE_QUANT_DIR | MONO_PHASES | \
                MONO_TOPK_LIMIT | MONO_TOPK_REPEAT | MONO_CELLS | MONO_CELL_TRIES)
                printf -v "$k" '%s' "$v"
                export "${k?}"
                say "GO override: $k=$v"
                ;;
        esac
    done < <(grep -E '^[A-Z_0-9]+=' "$RUN/GO" || true)
}

# ---- lease -----------------------------------------------------------------
setup_group() {
    local g
    for g in ${GPUS//,/ }; do
        [[ $g =~ ^[2-9]$ ]] || die "GPU $g is not in 2-9 (HIP 0/1 are the W7800s)"
    done
    FIRST=${GPUS%%,*}
    LAST=${GPUS##*,}
    REQUEST=$LEASE/hip$FIRST-$LAST.request
    export HIP_VISIBLE_DEVICES=$GPUS
    unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
}

free_gib() {
    "$PY" -c '
import torch
print(" ".join(f"{torch.cuda.mem_get_info(i)[0] / 2**30:.1f}"
               for i in range(torch.cuda.device_count())))'
}

enough_memory() {
    local f g
    f=$(free_gib) || return 1
    say "free GiB on HIP $GPUS: $f"
    for g in $f; do
        awk -v g="$g" 'BEGIN { exit !(g >= 28) }' || return 1
    done
}

port_busy() { ss -ltnH "sport = :$PORT" 2> /dev/null | grep -q .; }

others_hold() {
    local g
    for g in ${GPUS//,/ }; do
        find "$LEASE" -maxdepth 1 -name "*.hip$g" ! -name "$OWNER.*" | grep -q . &&
            return 0
    done
    return 1
}

HOLD=0
take_lease() {
    until [[ ! -e $REQUEST ]] || grep -q "^$OWNER" "$REQUEST"; do
        stopped && die "STOP while waiting for $REQUEST" 3
        say "waiting: $REQUEST is $(head -c 100 "$REQUEST" | tr '\n' ' ')"
        sleep 60
    done
    echo "$OWNER $(date -Is) slot.sh pid $$" > "$REQUEST"
    until ! others_hold && enough_memory; do
        stopped && die "STOP while waiting for the lease" 3
        say "waiting: holders $(find "$LEASE" -maxdepth 1 -name '*.hip*' -printf '%f ')"
        sleep 60
    done
    local g
    for g in ${GPUS//,/ }; do
        echo "$OWNER $(date -Is) slot.sh pid $$" > "$LEASE/$OWNER.hip$g"
    done
    HOLD=1
    say "lease taken: $(find "$LEASE" -maxdepth 1 -type f -printf '%f ')"
}

# Observers for the whole lease (logged, nothing gates on them): the junction
# temperature of the leased GPUs every TEMPS_PERIOD s ($RUN/temps.csv) and the KFD
# queues of the running arm's workers every QUEUES_PERIOD s ($RUN/queues.jsonl).
# analyze.py reads both.
OBSERVER_PIDS=()
start_observers() {
    "$PY" "$HERE/gpu_temps.py" --gpus "$GPUS" --period "$TEMPS_PERIOD" \
        --out "$RUN/temps.csv" &
    OBSERVER_PIDS+=($!)
    "$PY" "$HERE/queue_probe.py" --run "$RUN" --period "$QUEUES_PERIOD" &
    OBSERVER_PIDS+=($!)
    say "junction temperatures of HIP $GPUS every $TEMPS_PERIOD s -> $RUN/temps.csv"
    say "KFD queues of the workers every $QUEUES_PERIOD s -> $RUN/queues.jsonl"
}
stop_observers() {
    ((${#OBSERVER_PIDS[@]})) && kill "${OBSERVER_PIDS[@]}" 2> /dev/null
    OBSERVER_PIDS=()
    return 0
}

CUR_OUT=""
# shellcheck disable=SC2329  # invoked by the EXIT trap
release_lease() {
    local g
    stop_observers
    if ((HOLD)); then
        if [[ -n $CUR_OUT && -s $CUR_OUT/serve.pid ]]; then
            kill -- -"$(cut -d' ' -f1 "$CUR_OUT/serve.pid")" 2> /dev/null || true
        fi
        for g in ${GPUS//,/ }; do rm -f "$LEASE/$OWNER.hip$g"; done
        touch "$LANES/hip$FIRST$LAST.free" "$LANES/hip$FIRST$LAST.monow4a8.done"
        HOLD=0
        say "lease released"
    fi
    # also when the wait for the holders was cut short: the request is ours
    if [[ -n ${REQUEST:-} ]] && grep -q "^$OWNER" "$REQUEST" 2> /dev/null; then
        rm -f "$REQUEST"
    fi
}
trap release_lease EXIT
trap 'exit 130' INT TERM

# ---- arms ------------------------------------------------------------------
# Sets ARM_DESC, ARM_ENV (serve_validate KEY=VALUE overrides), ARM_EXTRA
# (EXTRA_ARGS) and what the logs must show: EXP_MONO, EXP_W4A8, EXP_HWQ.
arm_config() {
    ARM_ENV=() ARM_EXTRA=""
    case ${1%2} in
        a)
            ARM_DESC="production: W4A8 prefill, no --monokernel (2 HW queues)"
            EXP_MONO=0 EXP_W4A8=1 EXP_HWQ=2
            ;;
        a0)
            ARM_DESC="production with the recipe's PLE (2 HW queues)"
            EXP_MONO=0 EXP_W4A8=1 EXP_HWQ=2
            ;;
        b)
            ARM_DESC="--monokernel, W4A8 off (mono raises to 4 HW queues)"
            ARM_ENV=(VLLM_RDNA2_W4A8_SDOT4=0)
            ARM_EXTRA="--monokernel"
            EXP_MONO=1 EXP_W4A8=0 EXP_HWQ=4
            ;;
        c)
            ARM_DESC="--monokernel + W4A8 (4 HW queues)"
            ARM_EXTRA="--monokernel"
            EXP_MONO=1 EXP_W4A8=1 EXP_HWQ=4
            ;;
        d)
            ARM_DESC="--monokernel + W4A8, hw_queues 0 (2 HW queues)"
            ARM_EXTRA='--monokernel --additional-config {"rocm_mono_decode":{"hw_queues":0}}'
            EXP_MONO=1 EXP_W4A8=1 EXP_HWQ=2
            ;;
        e)
            ARM_DESC="production at GPU_MAX_HW_QUEUES=4 (control)"
            ARM_ENV=(GPU_MAX_HW_QUEUES=4)
            EXP_MONO=0 EXP_W4A8=1 EXP_HWQ=4
            ;;
        *) return 1 ;;
    esac
    EXP_PLE=recipe
    if [[ -n $PLE_MODE && ${1%2} != a0 ]]; then
        ARM_ENV+=(PLE="$PLE_MODE" PLE_QUANT_DIR="$PLE_QUANT_DIR")
        EXP_PLE=$PLE_MODE
    fi
}

# Move one boot's files aside (everything but earlier attempts and DONE).
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

boot_arm() { # arm out cache
    local arm=$1 out=$2 cache=$3 envargs
    envargs=(
        RECIPE=flashnext-mtp2 MODEL="$MODEL" GPUS="$GPUS" PORT="$PORT"
        TAG="mono-w4a8-$arm" OUT="$out" CACHE="$cache" VENV="$VENV" CELLS=none
        CONC_PROBE=1 LOAD_HOOK="$HERE/hook.sh" BOOT_LOCK="$BOOT_LOCK"
        HIP_VISIBLE_DEVICES="$GPUS"
    )
    [[ -n $ARM_EXTRA ]] && envargs+=(EXTRA_ARGS="$ARM_EXTRA")
    (
        cd "$TREE" &&
            env "${envargs[@]}" bash tools/rdna/port_v031/serve_validate.sh \
                "${ARM_ENV[@]}"
    )
}

# Returns 0 ok, 1 retry, 2 config mismatch (no retry), 10 OOM with the recipe PLE.
run_arm() {
    local arm=$1 out=$RUN/arms/$1 cache try rc ar aer0 aer1 why
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
        if [[ -e $cache/vllm/rdna_ar_wedged ]]; then
            say "removing wedge marker: $(tr '\n' ' ' < "$cache/vllm/rdna_ar_wedged")"
            rm -f "$cache/vllm/rdna_ar_wedged"
        fi
        until enough_memory; do
            stopped && { say "STOP present"; return 3; }
            sleep 30
        done
        until ! port_busy; do
            stopped && { say "STOP present"; return 3; }
            say "port $PORT is busy: waiting"
            sleep 15
        done
        aer0=$(aer_count)
        {
            echo "arm=$arm try=$try desc=$ARM_DESC"
            echo "sha=$(cat "$TREE/.sha") tree=$TREE ple=${PLE_MODE:-recipe}"
            echo "extra_args=$ARM_EXTRA overrides=${ARM_ENV[*]:-}"
            echo "load=$(cut -d' ' -f1-3 /proc/loadavg) aer=$aer0 date=$(date -Is)"
            # the ps itself ranks first (its CPU time over milliseconds): leave it out
            echo "top=$(ps -eo pcpu,comm --sort=-pcpu |
                awk 'NR > 1 && $2 != "ps" && n++ < 2' | tr -s ' \n' ' ')"
            echo "memlock=$(ulimit -l) compact_unevictable=$(sysctl -n \
                vm.compact_unevictable_allowed 2> /dev/null || echo n/a)"
        } > "$out/arm.txt"
        say "arm $arm try $try start ($ARM_DESC): $(grep -a '^load=' "$out/arm.txt")"
        boot_arm "$arm" "$out" "$cache"
        rc=$?
        aer1=$(aer_count)
        echo "rc=$rc aer_end=$aer1 end=$(date -Is)" >> "$out/arm.txt"
        say "arm $arm try $try rc=$rc:" \
            "$(grep -aE 'LOAD HOOK|PROBE|PPL|PREFIX|CONCURRENCY|READY|BOOT FAILED' \
                "$out/summary.txt" 2> /dev/null | tr '\n' ' ')"
        ar=$(grep -ac "rdna_ar: one-shot all-reduce active" "$out/serve.log")
        say "arm $arm rdna_ar active lines: $ar;" \
            "disabled: $(grep -ac 'rdna_ar: disabled' "$out/serve.log")"
        why=""
        if grep -aq "rdna_ar: WEDGED" "$out/serve.log"; then
            why="rdna_ar wedged"
        elif ((rc != 0)) || ! grep -q "LOAD HOOK rc=0" "$out/summary.txt"; then
            why="boot or hook failed (rc=$rc)"
            if grep -aqiE "out of memory|OutOfMemoryError" "$out/serve.log" &&
                [[ ${ARM_EXTRA} == *--monokernel* && -z $PLE_MODE ]]; then
                say "arm $arm: OOM with the recipe PLE"
                return 10
            fi
        elif ((aer1 > aer0)); then
            why="PCIe AER: $((aer1 - aer0)) new line(s)"
        elif ((ar == 0)) || grep -aq "rdna_ar: disabled" "$out/serve.log"; then
            why="rdna_ar is not active"
        fi
        if [[ -n $why ]]; then
            say "arm $arm try $try discarded: $why"
            continue
        fi
        if ! "$PY" "$HERE/analyze.py" check-arm --arm-dir "$out" --mono "$EXP_MONO" \
            --w4a8 "$EXP_W4A8" --hwq "$EXP_HWQ" --tp 4 --ple "$EXP_PLE" |
            tee "$out/check.txt"; then
            say "arm $arm: markers do not match the arm (see $out/check.txt)"
            return 2
        fi
        date -Is > "$out/DONE"
        say "arm $arm DONE"
        return 0
    done
    say "arm $arm failed after $MAX_TRIES tries"
    return 1
}

# ---- main ------------------------------------------------------------------
# One pass over ARMS. 0 done (arms may have failed: status), 3 STOP, 10 OOM.
status=0
run_all() {
    local arm rc
    for arm in $ARMS; do
        run_arm "$arm"
        rc=$?
        CUR_OUT=""  # no server runs between arms: nothing for the exit trap to kill
        ((rc == 3 || rc == 10)) && return "$rc"
        ((rc != 0)) && status=1
    done
    return 0
}

dry_run() {
    local arm denv
    setup_group
    for arm in $ARMS; do
        arm_config "$arm" || die "unknown arm $arm"
        say "arm $arm: $ARM_DESC; expect mono=$EXP_MONO w4a8=$EXP_W4A8 hwq=$EXP_HWQ"
        denv=(PRINT=1)
        [[ -n $ARM_EXTRA ]] && denv+=(EXTRA_ARGS="$ARM_EXTRA")
        (
            cd "$TREE" &&
                env "${denv[@]}" bash tools/rdna/serve_rdna.sh RECIPE=flashnext-mtp2 \
                    MODEL="$MODEL" VENV="$VENV" PORT="$PORT" "${ARM_ENV[@]}" 2>&1 |
                    grep -E "GPU_MAX_HW_QUEUES|W4A8_SDOT4|^# ple:|'--speculative-config'" |
                    grep -oE "^export [A-Z0-9_]+='[^']*'|^# ple:.*|'--speculative-config'.*"
        )
    done
}

mkdir -p "$RUN/arms" "$RUN/results" "$RUN/cache"
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
setup_group
say "GO: GPUS=$GPUS ARMS=$ARMS PORT=$PORT MAX_TRIES=$MAX_TRIES PLE=${PLE_MODE:-recipe}"
take_lease
start_observers

run_all
rc=$?
if ((rc == 10)); then
    # A mono arm does not fit with the recipe's bf16 PLE: every arm reruns with
    # the int4 PLE (the finished ones move to arms-bf16/) and the production arm
    # with the recipe's PLE is added at the end.
    echo "$(date -Is) OOM with the recipe PLE (mono arm); all arms rerun with" \
        "PLE=int4, a0 = production with the recipe PLE" >> "$RUN/FINDINGS.txt"
    PLE_MODE=int4
    mkdir -p "$RUN/arms-bf16"
    for d in "$RUN"/arms/*; do
        [[ -d $d ]] && mv "$d" "$RUN/arms-bf16/"
    done
    ARMS="$ARMS a0"
    run_all
    rc=$?
fi
((rc == 3)) && status=3

CUR_OUT=""
stop_observers
say "analysis"
"$PY" "$HERE/analyze.py" report --run "$RUN" --tree "$TREE" --arms "$ARMS" \
    > "$RUN/results/analyze.log" 2>&1 || say "analyze.py failed: see results/analyze.log"
date -Is > "$RUN/DONE"
say "slot finished status=$status; results in $RUN/results/"
exit "$status"
