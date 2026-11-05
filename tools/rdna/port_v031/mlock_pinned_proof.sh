#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# GPU proof of fix/mlock-pinned-host: one TP4 Flash-Next server (recipe
# flashnext-mtp2, one exact-SHA tree, one boot per arm) is measured while a
# neighbour does LOADS weight loads, with and without the lock of its pinned host
# mappings (VLLM_ROCM_MLOCK_PINNED_HOST):
#
#   base   =0  the PLE table is locked, nothing else (production before the fix)
#   fix    =1  every pinned mapping is locked after warm-up
#   a suffix 2 repeats an arm (fix2, base2): the boot-to-boot floor of the bench
#   cells. The default order ABBA, "base fix fix2 base2", cancels a drift.
#
# Per arm serve_validate.sh boots and warms the server, runs two bench cells
# (1024:512 at c=1 and c=8: TTFT/ITL, each cell eviction-checked), then the
# LOAD_HOOK (mlock_pinned_hook.sh): an idle window, then LOADS neighbour loads
# with the root-run pinned sampler, the raw KFD event reader and KFD evicted_ms
# sampling in the background and a light streaming load that logs stalls.
# The verdict (pinned_proof.py verdict, results/verdict.md) is PROVEN when the
# baseline evicted (>= BASE_MIN_MS in a worker), no fix arm evicted more than
# EVICT_OK_MS, every worker locked all its candidates, VmLck grew by what the fix
# logged, and TTFT/ITL stay within TOL_PCT % (or the base spread). It is
# INCONCLUSIVE when the baseline did not evict: then the neighbour did not
# provoke the problem and nothing was shown.
#
# Unattended. Start it any time, it touches no GPU before GO exists:
#   setsid nohup bash tools/rdna/port_v031/mlock_pinned_proof.sh \
#       > "$RUN/proof.log" 2>&1 < /dev/null &
#   touch "$RUN/GO"     # the window is open (GO may hold KEY=VALUE lines that
#                       # override GPUS, ARMS, PORT, LOADS, NEIGHBOUR, NEIGHBOUR_CMD,
#                       # PLE_MODE, PLE_QUANT_DIR, MAX_TRIES, SHM_GIB, OVERSHOOT_GIB)
#   touch "$RUN/STOP"   # abort: between arms and while waiting for the lease
# After GO it takes the lease (lanes/lease/README.md: hipF-L.request, then waits for
# no other holder and >= 28 GiB free per card), runs the arms under the quiet-window
# and boot-lock protocol (serve_validate.sh QUIET_DIR), releases the lease,
# touches lanes/hip<F><L>.free and writes $RUN/DONE. Outputs: $RUN/arms/<arm>/
# (proof/ holds the sampler, windows.json, report.md), $RUN/results/verdict.md.
#
# Required: EXPECT_SHA (must equal $TREE/.sha). DRY=1 verifies the tree and prints
# each arm's launch line and the neighbour command (no GPU, no lease).
#
# Start it from a fresh login shell: the workers mlock about 25 GiB each, and the
# hook's replay mlocks SHM_GIB x REPLAY_WORKERS (95 GiB, a TP2 load of the 95.4 GiB
# PLE table), so RLIMIT_MEMLOCK must be >= MEMLOCK_MIN_KB; the proof is
# meaningless unless vm.compact_unevictable_allowed is 0 (it refuses otherwise);
# the sampler and the KFD reader need passwordless `sudo -n`. A server is only ever
# stopped with SIGTERM here (STOP_GRACE, default 300 s, before serve_validate.sh
# falls back to SIGKILL): a TP4 server torn down mid-teardown may wedge KFD.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TREE=${TREE:-$(cd "$HERE/../../.." && pwd)}
RUN=${RUN:-$HOME/w4a8_runs/evictfix-proof}
VENV=${VENV:-$HOME/Apps/vllm/venv-7.14.0_0.32.0}
PY=$VENV/bin/python
MODEL=${MODEL:-$HOME/.cache/huggingface/hub/models--wtdcode--Qwen3.8-Flash-Next-AWQ-W4A16/snapshots/0939125b929543a783ce700c90e36dd1a575c00c}
LEASE=${LEASE:-$HOME/w4a8_runs/lanes/lease}
LANES=${LANES:-$HOME/w4a8_runs/lanes}
BOOT_LOCK=${BOOT_LOCK:-$HOME/w4a8_runs/boot.lock}
QUIET_DIR=${QUIET_DIR:-$LANES/quiet}
OWNER=${OWNER:-EvictFix}
GPUS=${GPUS:-2,3,4,5}
PORT=${PORT:-18480}
ARMS=${ARMS:-base fix fix2 base2}
MAX_TRIES=${MAX_TRIES:-2}
PLE_MODE=${PLE_MODE:-}
PLE_QUANT_DIR=${PLE_QUANT_DIR:-$HOME/.cache/huggingface/hub/models--primitive-ai--Qwen3.8-Flash-Next-PLE-quant/snapshots/4f861b63f69e61bfc2e22130ec91ec67f03ec43e/ples_int4}
EXPECT_SHA=${EXPECT_SHA:-}
DRY=${DRY:-0}
MEMLOCK_MIN_KB=${MEMLOCK_MIN_KB:-104857600}
STOP_GRACE=${STOP_GRACE:-300}
EVICT_OK_MS=${EVICT_OK_MS:-100}
BASE_MIN_MS=${BASE_MIN_MS:-500}
TOL_PCT=${TOL_PCT:-3}
KFD_RAW=${KFD_RAW:-python3 $HOME/w4a8_runs/rdna_ar_wedge/kfd_smi_raw.py}
export LOADS=${LOADS:-3} IDLE_S=${IDLE_S:-60} GAP_S=${GAP_S:-90} SETTLE_S=${SETTLE_S:-45}
export STREAMS=${STREAMS:-4} NEIGHBOUR=${NEIGHBOUR:-replay} NEIGHBOUR_CMD=${NEIGHBOUR_CMD:-}
export SHM_GIB=${SHM_GIB:-47.68} OVERSHOOT_GIB=${OVERSHOOT_GIB:-16} MARGIN_GIB=${MARGIN_GIB:-60}
export HOLD_S=${HOLD_S:-30} REPLAY_WORKERS=${REPLAY_WORKERS:-2} MAX_MINUTES=${MAX_MINUTES:-120}
export KFD_RAW PY BOOT_LOCK QUIET_DIR

say() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
die() { say "ABORT: $*"; exit "${2:-2}"; }
stopped() { [[ -e $RUN/STOP ]]; }
aer_count() {
    sudo -n dmesg 2> /dev/null | grep -cE 'pcieport.*(Timeout|BadTLP|NonFatalErr)' || true
}

verify_host() {
    local lim
    lim=$(ulimit -l)
    if [[ $lim != unlimited ]] && ((lim < MEMLOCK_MIN_KB)); then
        die "RLIMIT_MEMLOCK is $lim kB, the proof needs >= $MEMLOCK_MIN_KB kB: start from a fresh login shell (docs/rdna/serving.md)"
    fi
    local ca
    ca=$(sysctl -n vm.compact_unevictable_allowed 2> /dev/null || echo n/a)
    [[ $ca == 0 ]] || die "vm.compact_unevictable_allowed is $ca: compaction would move the locked pages, the proof needs 0 (docs/rdna/serving.md)"
    sudo -n true 2> /dev/null || die "passwordless sudo is needed for the sampler and the KFD event reader"
    "$PY" -c 'import numpy' 2> /dev/null || die "$PY has no numpy (the sampler needs it)"
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
    [[ -f $TREE/vllm/utils/pinned_host.py ]] || die "$TREE has no vllm/utils/pinned_host.py (not the fix tree)"
    if [[ $NEIGHBOUR == cmd && -z $NEIGHBOUR_CMD ]]; then
        die "NEIGHBOUR=cmd needs NEIGHBOUR_CMD"
    fi
}

# Overrides from the GO file (KEY=VALUE lines, whitelisted).
parse_go() {
    local k v
    while IFS='=' read -r k v; do
        case $k in
            GPUS | ARMS | PORT | LOADS | NEIGHBOUR | NEIGHBOUR_CMD | PLE_MODE | PLE_QUANT_DIR | \
                MAX_TRIES | SHM_GIB | OVERSHOOT_GIB | IDLE_S | GAP_S | SETTLE_S | HOLD_S)
                printf -v "$k" '%s' "$v"
                export "${k?}"
                say "GO override: $k=$v"
                ;;
        esac
    done < <(grep -E '^[A-Z_0-9]+=' "$RUN/GO" || true)
}

# ---- lease (lanes/lease/README.md) -----------------------------------------
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
    echo "$OWNER $(date -Is) mlock_pinned_proof.sh pid $$" > "$REQUEST"
    until ! others_hold && enough_memory; do
        stopped && die "STOP while waiting for the lease" 3
        say "waiting: holders $(find "$LEASE" -maxdepth 1 -name '*.hip*' -printf '%f ')"
        sleep 60
    done
    local g
    for g in ${GPUS//,/ }; do
        echo "$OWNER $(date -Is) mlock_pinned_proof.sh pid $$" > "$LEASE/$OWNER.hip$g"
    done
    HOLD=1
    say "lease taken: $(find "$LEASE" -maxdepth 1 -type f -printf '%f ')"
}

CUR_OUT=""
# shellcheck disable=SC2329  # invoked by the EXIT trap
release_lease() {
    local g
    if ((HOLD)); then
        # SIGTERM only: a server SIGKILLed during its KFD teardown may wedge the box
        if [[ -n $CUR_OUT && -s $CUR_OUT/serve.pid ]]; then
            kill -- -"$(cut -d' ' -f1 "$CUR_OUT/serve.pid")" 2> /dev/null || true
        fi
        for g in ${GPUS//,/ }; do rm -f "$LEASE/$OWNER.hip$g"; done
        touch "$LANES/hip$FIRST$LAST.free"
        rm -f "$QUIET_DIR/$OWNER".*
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

# ---- arms --------------------------------------------------------------------
# Sets ARM_DESC and ARM_MLOCK (the value of VLLM_ROCM_MLOCK_PINNED_HOST).
arm_config() {
    case ${1%%[0-9]*} in
        base)
            ARM_DESC="hook off: only the PLE table is locked"
            ARM_MLOCK=0
            ;;
        fix)
            ARM_DESC="every pinned host mapping locked after warm-up"
            ARM_MLOCK=1
            ;;
        *) return 1 ;;
    esac
    ARM_ENV=()
    if [[ -n $PLE_MODE ]]; then
        ARM_ENV+=(PLE="$PLE_MODE" PLE_QUANT_DIR="$PLE_QUANT_DIR")
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
        TAG="mlock-proof-$arm" OUT="$out" CACHE="$cache" VENV="$VENV"
        CELLS="1024:512:1 1024:512:8" CONC_PROBE=0 PPL_PROBE=0 PREFIX_PROBE=0
        LOAD_HOOK="$HERE/mlock_pinned_hook.sh" BOOT_LOCK="$BOOT_LOCK"
        QUIET_DIR="$QUIET_DIR" QUIET_OWN="$OWNER" QUIET_HOLD=1 STOP_GRACE="$STOP_GRACE"
        HIP_VISIBLE_DEVICES="$GPUS" VLLM_ROCM_MLOCK_PINNED_HOST="$ARM_MLOCK"
    )
    (
        cd "$TREE" &&
            env "${envargs[@]}" bash tools/rdna/port_v031/serve_validate.sh "${ARM_ENV[@]}"
    )
}

# Returns 0 ok, 1 retry, 2 config mismatch (no retry).
run_arm() {
    local arm=$1 out=$RUN/arms/$1 cache try rc ar aer0 aer1 why locks want
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
            echo "VLLM_ROCM_MLOCK_PINNED_HOST=$ARM_MLOCK overrides=${ARM_ENV[*]:-}"
            echo "neighbour=$NEIGHBOUR loads=$LOADS shm_gib=$SHM_GIB overshoot_gib=$OVERSHOOT_GIB"
            echo "load=$(cut -d' ' -f1-3 /proc/loadavg) aer=$aer0 date=$(date -Is)"
            echo "memlock=$(ulimit -l) compact_unevictable=$(sysctl -n \
                vm.compact_unevictable_allowed 2> /dev/null || echo n/a)"
            echo "swappiness=$(sysctl -n vm.swappiness 2> /dev/null || echo n/a)" \
                "min_free_kbytes=$(sysctl -n vm.min_free_kbytes 2> /dev/null || echo n/a)" \
                "watermark_scale_factor=$(sysctl -n vm.watermark_scale_factor 2> /dev/null || echo n/a)"
            echo "meminfo=$(grep -E '^(MemFree|Cached|SwapFree):' /proc/meminfo | tr -s ' \n' ' ')"
        } > "$out/arm.txt"
        say "arm $arm try $try start ($ARM_DESC): $(grep -a '^load=' "$out/arm.txt")"
        boot_arm "$arm" "$out" "$cache"
        rc=$?
        aer1=$(aer_count)
        echo "rc=$rc aer_end=$aer1 end=$(date -Is)" >> "$out/arm.txt"
        say "arm $arm try $try rc=$rc:" \
            "$(grep -aE 'LOAD HOOK|READY|BOOT FAILED' "$out/summary.txt" 2> /dev/null | tr '\n' ' ')"
        ar=$(grep -ac "rdna_ar: one-shot all-reduce active" "$out/serve.log")
        locks=$(grep -ac "pinned host mappings (" "$out/serve.log")
        want=0
        ((ARM_MLOCK == 1)) && want=4
        why=""
        if grep -aq "rdna_ar: WEDGED" "$out/serve.log"; then
            why="rdna_ar wedged"
        elif ((rc != 0)) || ! grep -q "LOAD HOOK rc=0" "$out/summary.txt"; then
            why="boot or hook failed (rc=$rc)"
        elif ((aer1 > aer0)); then
            why="PCIe AER: $((aer1 - aer0)) new line(s)"
        elif ((ar == 0)) || grep -aq "rdna_ar: disabled" "$out/serve.log"; then
            why="rdna_ar is not active"
        fi
        if [[ -z $why ]] && ((locks != want)); then
            # the arm is what was asked for, or it is not an arm: no retry helps
            say "arm $arm: $locks 'Locked N of M' lines, expected $want (VLLM_ROCM_MLOCK_PINNED_HOST=$ARM_MLOCK)"
            return 2
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

status=0
run_all() {
    local arm rc
    for arm in $ARMS; do
        run_arm "$arm"
        rc=$?
        CUR_OUT=""  # no server runs between arms: nothing for the exit trap to kill
        ((rc == 3)) && return 3
        ((rc != 0)) && status=1
    done
    return 0
}

dry_run() {
    local arm
    setup_group
    say "neighbour: $NEIGHBOUR${NEIGHBOUR_CMD:+ ($NEIGHBOUR_CMD)} x $LOADS; idle ${IDLE_S}s gap ${GAP_S}s settle ${SETTLE_S}s"
    for arm in $ARMS; do
        arm_config "$arm" || die "unknown arm $arm"
        say "arm $arm: $ARM_DESC; VLLM_ROCM_MLOCK_PINNED_HOST=$ARM_MLOCK" \
            "overrides=${ARM_ENV[*]:-none}"
        (
            cd "$TREE" &&
                env PRINT=1 bash tools/rdna/serve_rdna.sh RECIPE=flashnext-mtp2 \
                    MODEL="$MODEL" VENV="$VENV" PORT="$PORT" "${ARM_ENV[@]}" 2>&1 |
                    grep -E "GPU_MAX_HW_QUEUES|^# ple:|'--speculative-config'" |
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
verify_tree
say "GO: GPUS=$GPUS ARMS=$ARMS PORT=$PORT LOADS=$LOADS NEIGHBOUR=$NEIGHBOUR PLE=${PLE_MODE:-recipe}"
take_lease

run_all
rc=$?
((rc == 3)) && status=3

CUR_OUT=""
say "verdict"
"$PY" "$HERE/pinned_proof.py" verdict --run "$RUN" --arms "$ARMS" \
    --ok-ms "$EVICT_OK_MS" --base-min-ms "$BASE_MIN_MS" --tol-pct "$TOL_PCT" \
    > "$RUN/results/verdict.log" 2>&1
vrc=$?
say "verdict rc=$vrc: $(head -1 "$RUN/results/verdict.log")"
date -Is > "$RUN/DONE"
say "proof finished status=$status; results in $RUN/results/"
exit "$status"
