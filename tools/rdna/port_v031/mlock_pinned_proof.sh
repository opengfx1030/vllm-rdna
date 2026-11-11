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
# LOAD_HOOK (mlock_pinned_hook.sh): LOADS neighbour loads, each in a quiet
# window of its own (idle, load, settle), with the root-run pinned sampler, the
# raw KFD event reader and KFD evicted_ms sampling in the background and a
# light streaming load that logs stalls.
# The verdict (pinned_proof.py verdict, results/verdict.md) is PROVEN when the
# baseline evicted (>= BASE_MIN_MS in a worker), no fix arm evicted more than
# EVICT_OK_MS, every worker locked all its candidates, VmLck grew by what the fix
# logged, and the bench cells stay within TOL_PCT % (their p99s within three
# times that), or within the base spread. It is
# INCONCLUSIVE when the baseline did not evict: then the neighbour did not
# provoke the problem and nothing was shown.
#
# Unattended. Start it any time, it touches no GPU before GO exists:
#   setsid nohup bash tools/rdna/port_v031/mlock_pinned_proof.sh \
#       > "$RUN/proof.log" 2>&1 < /dev/null &
#   touch "$RUN/GO"     # the slot is ours (GO may hold KEY=VALUE lines that
#                       # override SLOT, GPUS, ARMS, PORT, LOADS, NEIGHBOUR, NEIGHBOUR_CMD,
#                       # PLE_MODE, PLE_QUANT_DIR, MAX_TRIES, SHM_GIB, OVERSHOOT_GIB)
#   touch "$RUN/STOP"   # abort: between arms and while waiting for the lease
# SLOT is the lease README's topology of 2026-11-05 (A: HIP 1,2,3,4 = V620 buses
# ce d3 d8 db; B: HIP 6,7,8,9 = 0a 0f 14 17; HIP 0 and 5 are the W7800s). After GO
# the driver maps the slot's PCI buses to HIP indices (hipDeviceGetPCIBusId, no
# GPU context), refuses unless they are V620s at exactly GPUS, and does so again
# before every arm. It takes the lease (slot<X>.request, then waits for no other
# <owner>.slot<X> holder and >= 28 GiB free per card, then writes $OWNER.slot<X>),
# runs the arms under the quiet-window and boot-lock protocol (serve_validate.sh
# QUIET_DIR: the bench cells are one window, each neighbour load of the hook
# another; every window <= 15 min, >= 2 min between two, none while a docker
# buildx build runs), removes its holder file and writes $RUN/DONE. Outputs:
# $RUN/arms/<arm>/ (proof/ holds the sampler, windows.json, report.md),
# $RUN/results/verdict.md.
# The W7800 buses (cb, 07) are resolved the same way, none may be in the slot, and
# their HIP indices go to serve_validate.sh as REFUSED_GPUS (its default, 0,1, is the
# topology before the reboot). A junction guard per arm (pinned_proof.py temps,
# TEMP_GUARD=1) logs the slot's temperatures to arms/<arm>/temps.log and SIGTERMs
# the server at a sample of 95 C or a mean of five at 90 C: the proof then stops.
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
SLOT=${SLOT:-A}
GPUS=${GPUS:-}
BUS_RESOLVER=${BUS_RESOLVER:-}
PCI_SYSFS=${PCI_SYSFS:-/sys/bus/pci/devices}
TEMP_GUARD=${TEMP_GUARD:-1}
HWMON=${HWMON:-/sys/class/hwmon}
V620_PCI_ID=0x73a1
W7800_BUSES="cb 07"
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
export LOADS=${LOADS:-3} IDLE_S=${IDLE_S:-30} GAP_S=${GAP_S:-120} SETTLE_S=${SETTLE_S:-30}
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
            SLOT | GPUS | ARMS | PORT | LOADS | NEIGHBOUR | NEIGHBOUR_CMD | PLE_MODE | PLE_QUANT_DIR | \
                MAX_TRIES | SHM_GIB | OVERSHOOT_GIB | IDLE_S | GAP_S | SETTLE_S | HOLD_S)
                printf -v "$k" '%s' "$v"
                export "${k?}"
                say "GO override: $k=$v"
                ;;
        esac
    done < <(grep -E '^[A-Z_0-9]+=' "$RUN/GO" || true)
}

# ---- slot and lease (lanes/lease/README.md, the topology of 2026-11-05 15:35) --
slot_hip() {
    case $1 in A) echo 1,2,3,4 ;; B) echo 6,7,8,9 ;; *) return 1 ;; esac
}
slot_buses() {
    case $1 in A) echo "ce d3 d8 db" ;; B) echo "0a 0f 14 17" ;; *) return 1 ;; esac
}

setup_group() {
    local want
    want=$(slot_hip "$SLOT") || die "unknown SLOT '$SLOT' (A or B)"
    [[ -n $GPUS ]] || GPUS=$want
    [[ $GPUS == "$want" ]] ||
        die "GPUS=$GPUS is not slot $SLOT (HIP $want; HIP 0 and 5 are the W7800s)"
    REQUEST=$LEASE/slot$SLOT.request
    HOLDER=$LEASE/$OWNER.slot$SLOT
    export HIP_VISIBLE_DEVICES=$GPUS
    unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
}

# The HIP indices of PCI buses BUS... (comma separated, in that order) from
# hipDeviceGetPCIBusId over every device: the query creates no context.
hip_of_buses() {
    if [[ -n $BUS_RESOLVER ]]; then
        "$BUS_RESOLVER" "$@"
        return
    fi
    env -u HIP_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
        "$PY" "$HERE/pinned_proof.py" buses --venv "$VENV" "$@"
}

# The README's check before every launch: every bus of the slot is a V620 and
# resolves to exactly the HIP indices in GPUS, else refuse (the order moved).
verify_slot() {
    local buses b dev now
    buses=$(slot_buses "$SLOT")
    for b in $buses; do
        dev=$(cat "$PCI_SYSFS/0000:$b:00.0/device" 2> /dev/null || echo none)
        [[ $dev == "$V620_PCI_ID" ]] ||
            die "PCI bus $b of slot $SLOT is device $dev, not a V620 ($V620_PCI_ID)"
    done
    # shellcheck disable=SC2086  # one argument per bus
    now=$(hip_of_buses $buses) ||
        die "cannot map the PCI buses of slot $SLOT ($buses) to HIP indices"
    [[ $now == "$GPUS" ]] ||
        die "slot $SLOT (buses $buses) is HIP $now now, not $GPUS: the topology moved"
    say "slot $SLOT verified: HIP $GPUS = V620 buses $buses"
    # serve_validate.sh refuses REFUSED_GPUS (its default, 0,1, is the old
    # topology): the W7800s, by bus, and none of them may be in the slot
    # shellcheck disable=SC2086  # one argument per bus
    REFUSED=$(hip_of_buses $W7800_BUSES) ||
        die "cannot map the W7800 buses ($W7800_BUSES) to HIP indices"
    for dev in ${GPUS//,/ }; do
        case ",$REFUSED," in
            *,"$dev",*) die "HIP $dev of slot $SLOT is a W7800 (buses $W7800_BUSES = HIP $REFUSED)" ;;
        esac
    done
    say "W7800 buses $W7800_BUSES = HIP $REFUSED: refused"
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
    find "$LEASE" -maxdepth 1 -name "*.slot$SLOT" ! -name "$OWNER.*" | grep -q .
}

HOLD=0
HAVE_REQUEST=0
req_is_mine() { grep -q "^$OWNER " "$REQUEST" 2> /dev/null; }
take_lease() {
    # slot<X>.request is created atomically (noclobber): the README's queue
    until (
        set -C
        echo "$OWNER $(date -Is) mlock_pinned_proof.sh pid $$" > "$REQUEST"
    ) 2> /dev/null || req_is_mine; do
        stopped && die "STOP while waiting for $REQUEST" 3
        say "waiting: $REQUEST is $(head -c 100 "$REQUEST" 2> /dev/null | tr '\n' ' ')"
        sleep 60
    done
    HAVE_REQUEST=1
    until ! others_hold && enough_memory; do
        stopped && die "STOP while waiting for the lease" 3
        say "waiting: holders $(find "$LEASE" -maxdepth 1 -name '*.slot*' ! -name '*.request' -printf '%f ')"
        sleep 60
    done
    local end
    end=$(date -Is -d "+${LEASE_MINUTES:-$(($(wc -w <<< "$ARMS") * 45 + 10))} minutes")
    echo "$OWNER $end mlock_pinned_proof.sh pid $$ slot $SLOT HIP $GPUS" > "$HOLDER"
    HOLD=1
    say "lease taken until $end: $(find "$LEASE" -maxdepth 1 -type f -printf '%f ')"
}

CUR_OUT=""
# shellcheck disable=SC2329  # invoked by the EXIT trap
release_lease() {
    if ((HOLD)); then
        # SIGTERM only: a server SIGKILLed during its KFD teardown may wedge the box
        if [[ -n $CUR_OUT && -s $CUR_OUT/serve.pid ]]; then
            kill -- -"$(cut -d' ' -f1 "$CUR_OUT/serve.pid")" 2> /dev/null || true
        fi
        rm -f "$HOLDER"
        rm -f "$QUIET_DIR/$OWNER" "$QUIET_DIR/$OWNER".*
        HOLD=0
        say "lease released"
    fi
    # also when the wait for the holders was cut short: the request is ours
    if ((HAVE_REQUEST)) && req_is_mine; then
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
        REFUSED_GPUS="$REFUSED"
    )
    local watch="" b bdfs=""
    if [[ $TEMP_GUARD == 1 ]]; then
        # the junction guard of the arm: logs every second, SIGTERMs the server at
        # 95 C (a sample) or 90 C (mean of five)
        for b in $(slot_buses "$SLOT"); do bdfs+=${bdfs:+,}$b; done
        "$PY" "$HERE/pinned_proof.py" temps --buses "$bdfs" --hwmon "$HWMON" \
            --log "$out/temps.log" --pidfile "$out/serve.pid" \
            --abort "$out/ABORT_TEMP" 2> "$out/temps.err" &
        watch=$!
        sleep 2
        kill -0 "$watch" 2> /dev/null ||
            say "WARNING: no temperature guard: $(cat "$out/temps.err")"
    fi
    local rc
    (
        cd "$TREE" &&
            env "${envargs[@]}" bash tools/rdna/port_v031/serve_validate.sh "${ARM_ENV[@]}"
    )
    rc=$?
    [[ -n $watch ]] && kill "$watch" 2> /dev/null
    return "$rc"
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
        verify_slot # the README: map HIP index -> PCI bus before every launch
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
        if [[ -e $out/ABORT_TEMP ]]; then
            say "THERMAL ABORT: $(cat "$out/ABORT_TEMP"); the proof stops"
            return 3
        fi
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
    say "slot $SLOT: HIP $GPUS = V620 buses $(slot_buses "$SLOT")"
    say "neighbour: $NEIGHBOUR${NEIGHBOUR_CMD:+ ($NEIGHBOUR_CMD)} x $LOADS; one quiet window per load: idle ${IDLE_S}s settle ${SETTLE_S}s gap ${GAP_S}s (at least 120 s)"
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
verify_slot
say "GO: SLOT=$SLOT GPUS=$GPUS ARMS=$ARMS PORT=$PORT LOADS=$LOADS NEIGHBOUR=$NEIGHBOUR PLE=${PLE_MODE:-recipe}"
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
