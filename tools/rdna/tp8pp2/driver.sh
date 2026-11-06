#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# ONE boot of a flashnext recipe (production compiled mode, FULL_AND_PIECEWISE) on 4 or
# 8 V620s with probes (serve_validate.sh), dmesg/SEL/temperature/memory capture and an
# optional measured hook (hook.sh). Needs the lease holder files of the slots it uses
# (OWNER.slotA / OWNER.slotB) and an exact-SHA tree.
#
#   OWNER=<lease holder> SHA=<40 hex> CFG=tp8 ARM=tp8-a bash driver.sh        (foreground)
#   setsid nohup env OWNER=... SHA=... CFG=tp8 ARM=tp8-a bash driver.sh \
#       > $RUN_DIR/arms/tp8-a.driver.log 2>&1 < /dev/null &     (from a fresh ssh login)
#
# CFG: tp8   TP=8 EP=8 on slots A+B   (HIP 1,2,3,4,6,7,8,9)
#      pp2   TP=4 PP=2 on slots A+B   (stage 0 = slot A, stage 1 = slot B), PP_PART=24,24
#      tp4a  TP=4 on slot A           tp4b  TP=4 on slot B
#      rtp8 / rpp2  recipe-native boots on slots A+B: no TP/PP override (set RECIPE=
#      flashnext-tp8-mtp2 or flashnext-tp4pp2-mtp2); RECIPE_RESOLVES_GPUS=1 makes the
#      recipe's PCI_BUSES (not the slot list) pick the GPUs; serve_wrap.sh compares.
# SHA        the tree $HOME/vllm-tp8pp2-<sha10> (TREE_T overrides) must carry this .sha
# RECIPE     default flashnext-mtp2
# EXTRA_OV="KEY=value ..."  recipe overrides appended after the CFG ones (KV=auto GMEM=0.90)
# SERVE_ENV="K=V K=V"        extra environment for the server (TUNABLEOP_ROOT=<rows dir>)
# RECORD_UNTUNED=1 (opt-in)  TunableOp GEMM-shape census into $OUT/untuned<ordinal>.csv;
#                            torch 2.12 then DISABLES the rows (never a measured arm)
# NCCL_LOG=1                 NCCL_DEBUG=INFO (INIT,ENV,GRAPH,TUNING) into $OUT/nccl.%p.log
# MEMHOOK=1 (default)        per-worker torch allocator sampler (memhook/sitecustomize.py)
# CELLS / PPL_PROBE / PREFIX_PROBE / CONC_PROBE / WARMUP: passed to serve_validate.sh
# HOOK=1                     LOAD_HOOK=hook.sh (LOAD_HOOK=<path> picks another); the
#                            hook reads CELLS_SPEC, CONC_REPEAT, QUALITY, WIN_MAX_S
# PORT, RUN_TIMEOUT, READY_TIMEOUT, POLL_S, CACHE, PP_PART
set -uo pipefail
# shellcheck source=slots.sh
source "$(dirname "${BASH_SOURCE[0]}")/slots.sh"
OWNER=${OWNER:?OWNER=<lease holder name>}
export OWNER LANES BOOT_LOCK RUN_DIR="$R"
SHA=${SHA:?SHA=<40 hex commit of the exact-SHA tree>}
TREE_T=${TREE_T:-$HOME/vllm-tp8pp2-${SHA:0:10}}
HUB=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}
MODEL_FN=${MODEL_FN:-$HUB/models--wtdcode--Qwen3.8-Flash-Next-AWQ-W4A16/snapshots/0939125b929543a783ce700c90e36dd1a575c00c}
CFG=${CFG:?CFG=tp8|pp2|tp4a|tp4b|rtp8|rpp2}
ARM=${ARM:-$CFG}
RECIPE=${RECIPE:-flashnext-mtp2}
PORT=${PORT:-18700}
POLL_S=${POLL_S:-20}
EXTRA_OV=${EXTRA_OV:-}
SERVE_ENV=${SERVE_ENV:-}
CELLS=${CELLS:-none}
READY_TIMEOUT=${READY_TIMEOUT:-2700}
LOAD_HOOK=${LOAD_HOOK:-}
[[ ${HOOK:-0} == 1 ]] && LOAD_HOOK=$H/hook.sh
case $CFG in
    tp8) SLOT=AB OV=(TP=8) ;;
    pp2) SLOT=AB OV=(TP=4 PP=2 "PP_PARTITION=${PP_PART:-24,24}") ;;
    tp4a) SLOT=A OV=(TP=4) ;;
    tp4b) SLOT=B OV=(TP=4) ;;
    rtp8) SLOT=AB OV=() ;;
    rpp2) SLOT=AB OV=() ;;
    dp2) SLOT=A OV=() ;;
    *)
        echo "bad CFG $CFG"
        exit 2
        ;;
esac
read -r -a extra_ov <<< "$EXTRA_OV"
OV+=("${extra_ov[@]}")
# CFG=dp2: two TP4 replicas (slot A on PORT, slot B on PORT+1), one driver each, their
# hooks measure every cell together (BARRIER_DIR / BARRIER_PEER in hook.sh)
if [[ $CFG == dp2 ]]; then
    export BARRIER_DIR=$R/barrier/$ARM
    rm -rf "$BARRIER_DIR"
    mkdir -p "$BARRIER_DIR"
    port_a=$PORT
    env CFG=tp4a ARM="$ARM-a" PORT="$port_a" BARRIER_PEER="$ARM-b" bash "$H/driver.sh" &
    pid_a=$!
    env CFG=tp4b ARM="$ARM-b" PORT=$((port_a + 1)) BARRIER_PEER="$ARM-a" bash "$H/driver.sh" &
    pid_b=$!
    rc=0
    wait "$pid_a" || rc=1
    wait "$pid_b" || rc=1
    exit "$rc"
fi
GPUS=$(slot_hip "$SLOT")
BUSES=$(slot_buses "$SLOT")
NGPU=$(echo "$GPUS" | tr ',' '\n' | wc -l)
OUT=$R/arms/$ARM
CACHE=${CACHE:-$R/cache-$CFG}
STATUS=0
WATCH_PIDS=()

say() { echo "[$(date '+%F %T')] $*"; }
die() {
    say "FATAL: $*"
    STATUS=1
    exit 1
}
need_leases() {
    [[ -e $LEASE/$OWNER.slotA || -e $LEASE/$OWNER.slotB ]] || return 1
    local f x
    for x in A B; do
        [[ $SLOT == *$x* ]] || continue
        for f in "$LEASE"/*.slot"$x"*; do
            [[ -e $f ]] || continue
            [[ ${f##*/} == "$OWNER".* ]] && continue
            say "foreign lease file ${f##*/}"
            return 1
        done
    done
    return 0
}
free_gib() {
    HIP_VISIBLE_DEVICES=$GPUS "$VENV/bin/python" -c '
import torch
print(" ".join(f"{torch.cuda.mem_get_info(i)[0] / 2**30:.1f}" for i in range(torch.cuda.device_count())))' 2> "$R/free_gib.err"
}
enough_memory() {
    local f g n=0
    f=$(free_gib) || {
        say "free_gib failed: $(tail -2 "$R/free_gib.err" | tr '\n' ' ')"
        return 1
    }
    say "free GiB on HIP $GPUS: $f"
    for g in $f; do
        n=$((n + 1))
        awk -v g="$g" 'BEGIN { exit !(g >= 28) }' || return 1
    done
    ((n == NGPU))
}
wait_quiet() {
    until [[ -z $(find "$QUIET" -type f -mmin -6 ! -name "$OWNER" 2> /dev/null) ]]; do
        say "waiting for quiet windows to expire: $(find "$QUIET" -type f -mmin -6 ! -name "$OWNER" | tr '\n' ' ')"
        sleep "$POLL_S"
    done
}
# shellcheck disable=SC2329  # called from on_exit, which is installed with trap
stop_own_server() {
    local pid i
    [[ -s $OUT/serve.pid ]] || return 0
    pid=$(cat "$OUT/serve.pid")
    kill -0 "$pid" 2> /dev/null || return 0
    say "stopping own server pgid $pid (SIGTERM, 180 s grace)"
    kill -TERM -- -"$pid" 2> /dev/null
    for ((i = 0; i < 90; i++)); do
        kill -0 -- -"$pid" 2> /dev/null || return 0
        sleep 2
    done
    kill -KILL -- -"$pid" 2> /dev/null
}
# shellcheck disable=SC2329  # installed with trap
on_exit() {
    local rc=$? p
    for p in "${WATCH_PIDS[@]}"; do kill "$p" 2> /dev/null; done
    stop_own_server
    rm -f "$QUIET/$OWNER"
    echo "$(date -Is) exit=$rc status=$STATUS" > "$OUT/DONE"
    say "driver exit rc=$rc status=$STATUS"
}
sel_info() { sudo -n ipmitool sel info 2>&1 | grep -E 'Entries|Percent Used|Last Add Time|Overflow' | tr '\n' ';'; }
# summary of the kernel log lines added since the boot (dmesg_before = line count)
dmesg_summary() {
    local n=$((DMESG_BEFORE + 1)) new
    new=$(tail -n +$n "$OUT/dmesg.after.txt")
    echo "arm $ARM: dmesg lines before $DMESG_BEFORE, after $(wc -l < "$OUT/dmesg.after.txt")"
    echo "new 'Runlist is getting oversubscribed': $(grep -c 'Runlist is getting oversubscribed' <<< "$new")"
    echo "new amdgpu fault/timeout/reset/hang/ras/ecc/UTCL2: $(grep -iE 'amdgpu' <<< "$new" | grep -iE 'fault|timeout|reset|hang|ras |ecc|UTCL2' | grep -vc 'uses VM inv eng')"
    echo "new AER / PCIe Bus Error / GHES / Hardware Error: $(grep -ciE 'AER:|PCIe Bus Error|GHES|Hardware Error|pcieport.*(error|SERR)' <<< "$new")"
    echo "new IOMMU/AMD-Vi faults: $(grep -ciE 'AMD-Vi|IO_PAGE_FAULT|iommu.*fault' <<< "$new")"
    echo "new KFD/evict lines: $(grep -ciE 'kfd|evict' <<< "$new")"
    echo "new amdgpu lines of any kind: $(grep -ci amdgpu <<< "$new")"
    echo "SEL before: $(cat "$OUT/sel.before.txt")"
    echo "SEL after:  $(cat "$OUT/sel.after.txt")"
}

main() {
    mkdir -p "$OUT" "$OUT/torchmem"
    unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
    export HIP_VISIBLE_DEVICES=
    trap on_exit EXIT
    trap 'say "signal: leaving"; exit 143' TERM INT HUP
    rm -f "$OUT/DONE" "$R/abort_quiet.$ARM" "$R/abort_bus.$ARM" "$R/ABORT_TEMP.$ARM"
    say "driver start pid $$ cfg $CFG arm $ARM slot $SLOT HIP $GPUS buses $BUSES tree $TREE_T overrides ${OV[*]}"
    [[ $(cat "$TREE_T/.sha") == "$SHA" ]] || die "$TREE_T is not ${SHA:0:10}"
    [[ $(ulimit -l) == unlimited ]] || die "ulimit -l is $(ulimit -l): launch from a fresh ssh login"
    [[ -z $(ss -ltn "sport = :$PORT" | tail -n +2) ]] || die "port $PORT is in use"
    need_leases || die "lease check failed (foreign holder or no $OWNER holder)"
    verify_slot "$SLOT" || die "bus verification failed"
    export HIP_VISIBLE_DEVICES=$GPUS
    wait_quiet
    until enough_memory; do sleep "$POLL_S"; done
    T0=$(date '+%F %T')
    echo "$T0" > "$OUT/t0.txt"
    DMESG_BEFORE=$(sudo -n dmesg | tee "$OUT/dmesg.before.txt" | wc -l)
    sel_info > "$OUT/sel.before.txt"
    python3 "$H/tempwatch.py" "$(echo "$BUSES" | sed 's/\([0-9a-f][0-9a-f]\)/0000:\1:00.0/g;s/ /,/g')" \
        "$OUT/temps1s.log" "$OUT/temps15s.log" "$OUT/serve.pid" "$R/ABORT_TEMP.$ARM" &
    WATCH_PIDS+=($!)
    python3 "$H/memwatch.py" "$(echo "$BUSES" | tr ' ' ',')" "$OUT/serve.pid" "$OUT/memwatch.tsv" 2 &
    WATCH_PIDS+=($!)
    local -a senv
    read -r -a senv <<< "$SERVE_ENV"
    if [[ ${NCCL_LOG:-0} == 1 ]]; then
        senv+=(NCCL_DEBUG=INFO "NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH,TUNING" "NCCL_DEBUG_FILE=$OUT/nccl.%h.%p.log")
    fi
    if [[ ${MEMHOOK:-1} == 1 ]]; then
        senv+=("PYTHONPATH=$H/memhook" "RDNA_MEMHOOK_DIR=$OUT/torchmem" RDNA_MEMHOOK_PERIOD=2)
    fi
    # RECORD_UNTUNED=1: census of every GEMM shape the server runs (torch writes
    # untuned<ordinal>.csv; "%d" is NOT expanded in this file name). Torch 2.12 does
    # NOT read the TunableOp rows while it records (tunetest.sh: rows_loaded=0 on
    # every ordinal), so a census arm is never a measured arm.
    if [[ ${RECORD_UNTUNED:-0} == 1 ]]; then
        senv+=(PYTORCH_TUNABLEOP_RECORD_UNTUNED=1 "PYTORCH_TUNABLEOP_UNTUNED_FILENAME=$OUT/untuned.csv")
    fi
    say "serve env: ${senv[*]}"
    (
        cd "$TREE_T" &&
            env "${senv[@]}" ARM="$ARM" SLOT="$SLOT" TREE_T="$TREE_T" ARM_OUT="$OUT" \
                RECIPE="$RECIPE" MODEL="$MODEL_FN" GPUS="$GPUS" PORT="$PORT" \
                TAG="tp8pp2-$ARM" OUT="$OUT" CACHE="$CACHE" VENV="$VENV" READY_TIMEOUT="$READY_TIMEOUT" \
                CELLS="$CELLS" CONC_PROBE="${CONC_PROBE:-1}" PPL_PROBE="${PPL_PROBE:-1}" \
                PREFIX_PROBE="${PREFIX_PROBE:-0}" WARMUP="${WARMUP:-1}" \
                ${LOAD_HOOK:+LOAD_HOOK="$LOAD_HOOK"} QUIET_HOLD=0 QUIET_NAME="$OWNER" QUIET_OWN="$OWNER" \
                BOOT_LOCK="$BOOT_LOCK" REFUSED_GPUS=0,5 STOP_GRACE=180 \
                SERVE_SCRIPT="$H/serve_wrap.sh" BENCH_EXTRA="--temperature 0" \
                timeout --kill-after=60 "${RUN_TIMEOUT:-7200}" bash tools/rdna/port_v031/serve_validate.sh \
                EAGER=0 "${OV[@]}"
    )
    rc=$?
    sudo -n dmesg | tee "$OUT/dmesg.after.txt" > /dev/null 2>&1
    sudo -n journalctl -k --no-pager --since "$T0" 2> /dev/null | tee "$OUT/journal_k.txt" > /dev/null
    sel_info > "$OUT/sel.after.txt"
    dmesg_summary > "$OUT/dmesg_summary.txt"
    cat "$OUT/dmesg_summary.txt"
    say "serve_validate rc=$rc: $(grep -aE 'READY|PROBE|PPL|CONCURRENCY|LOAD HOOK|done|BOOT FAILED' "$OUT/summary.txt" 2> /dev/null | tr '\n' ' ' | cut -c1-700)"
    [[ -e $R/abort_quiet.$ARM ]] && say "boot refused under boot.lock (quiet window): $(cat "$R/abort_quiet.$ARM")"
    [[ -e $R/abort_bus.$ARM ]] && die "bus verification refused the launch: $(cat "$R/abort_bus.$ARM")"
    if [[ -e $R/ABORT_TEMP.$ARM ]]; then
        STATUS=1
        say "THERMAL ABORT: $(cat "$R/ABORT_TEMP.$ARM")"
    fi
    if ! { grep -aq '\] done$' "$OUT/summary.txt" 2> /dev/null && ((rc == 0)); }; then
        STATUS=1
        say "arm $ARM FAILED (rc=$rc)"
    fi
    exit "$STATUS"
}
main "$@"
