#!/usr/bin/env bash
# rdna_ar soak: production flashnext-mtp2 TP4 on slot A for SOAK_MIN minutes of mixed
# load while other agents boot servers on slot B.
#   cd ~/w4a8_runs/rdna_ar_soak_2026-11-06 && \
#     setsid nohup bash driver.sh > driver.log 2>&1 < /dev/null &
# Launch from a fresh ssh login (ulimit -l unlimited for the PLE mlock).
# Order: lease (request, wait for no holder + >= 28 GiB free per GPU) -> bus check ->
# serve_validate.sh (boot.lock + quiet protocol, probes, PPL, concurrency probe,
# warm-up) with soak_hook.sh as the LOAD_HOOK -> probe-after-load -> release.
# The soak holds NO quiet window: neighbour boots are the load under test.
set -uo pipefail
R=${R:-$HOME/w4a8_runs/rdna_ar_soak_2026-11-06}
export R
export VENV=$HOME/Apps/vllm/venv-7.14.0
# shellcheck source=/dev/null
source "$R/slots.sh"
SHA=${SHA:-22483a656efb843f3253cc78ba191cb2ece649e2}
TREE_T=$HOME/vllm-soak-${SHA:0:10}
export TREE_T
HUB=$HOME/.cache/huggingface/hub
MODEL_FN=$HUB/models--wtdcode--Qwen3.8-Flash-Next-AWQ-W4A16/snapshots/0939125b929543a783ce700c90e36dd1a575c00c
LEASE=$HOME/w4a8_runs/lanes/lease
QUIET=$HOME/w4a8_runs/lanes/quiet
BOOT_LOCK=$HOME/w4a8_runs/boot.lock
OWNER=Soak
export OWNER
SLOT=A
export SLOT
REQ=$LEASE/slot$SLOT.request
HOLD=$LEASE/$OWNER.slot$SLOT
GPUS=$(slot_hip "$SLOT")
BUSES=$(slot_buses "$SLOT")
PORT=${PORT:-18700}
export SOAK_MIN=${SOAK_MIN:-135}
LEASE_MINUTES=${LEASE_MINUTES:-$((SOAK_MIN + 60))}
LEASE_WAIT_MAX_S=${LEASE_WAIT_MAX_S:-14400}
POLL_S=${POLL_S:-20}
OUT=$R/run
CACHE=$R/cache
HAVE_REQUEST=0
STATUS=0

say() { echo "[$(date '+%F %T')] $*"; }
die() { say "FATAL: $*"; STATUS=1; exit 1; }
req_is_mine() { grep -q "^$OWNER " "$REQ" 2> /dev/null; }
others_hold() {
    local f
    for f in "$LEASE"/*.slot"$SLOT"; do
        [[ -e $f && $f != "$HOLD" ]] && return 0
    done
    return 1
}
free_gib() {
    HIP_VISIBLE_DEVICES=$GPUS "$VENV/bin/python" -c '
import torch
print(" ".join(f"{torch.cuda.mem_get_info(i)[0] / 2**30:.1f}" for i in range(torch.cuda.device_count())))' 2> "$R/free_gib.err"
}
enough_memory() {
    local f g n=0
    f=$(free_gib) || { say "free_gib failed: $(tail -2 "$R/free_gib.err" | tr '\n' ' ')"; return 1; }
    say "free GiB on HIP $GPUS: $f"
    for g in $f; do
        n=$((n + 1))
        awk -v g="$g" 'BEGIN { exit !(g >= 28) }' || return 1
    done
    ((n == 4))
}
lease_ls() { local f; for f in "$LEASE"/*; do printf '%s ' "${f##*/}"; done; }
take_lease() {
    local t0 end
    t0=$(date +%s)
    until ( set -C; echo "$OWNER $(date -Is) rdna_ar soak slot $SLOT" > "$REQ" ) 2> /dev/null || req_is_mine; do
        say "waiting: foreign request: $(cat "$REQ" 2> /dev/null)"
        (($(date +%s) - t0 > LEASE_WAIT_MAX_S)) && die "no lease within ${LEASE_WAIT_MAX_S}s: foreign request"
        sleep "$POLL_S"
    done
    HAVE_REQUEST=1
    say "request: $(cat "$REQ")"
    until ! others_hold && enough_memory; do
        say "waiting: lease dir: $(lease_ls)"
        (($(date +%s) - t0 > LEASE_WAIT_MAX_S)) && die "no lease within ${LEASE_WAIT_MAX_S}s: holders or memory"
        sleep "$POLL_S"
    done
    end=$(date -Is -d "+$LEASE_MINUTES minutes")
    echo "$OWNER $end rdna_ar soak slot $SLOT HIP $GPUS" > "$HOLD"
    say "lease taken until $end: $(lease_ls)"
}
# shellcheck disable=SC2329  # called from the EXIT trap
release_lease() {
    rm -f "$HOLD"
    if ((HAVE_REQUEST)) && req_is_mine; then rm -f "$REQ"; fi
    rm -f "$QUIET/$OWNER"
    say "lease released (holder file, request, quiet file removed)"
}
# shellcheck disable=SC2329
stop_own_server() {
    local pid
    [[ -s $OUT/serve.pid ]] || return 0
    pid=$(cat "$OUT/serve.pid")
    kill -0 "$pid" 2> /dev/null || return 0
    say "stopping own server pgid $pid (SIGTERM, 180 s grace)"
    kill -TERM -- -"$pid" 2> /dev/null
    local i
    for ((i = 0; i < 90; i++)); do
        kill -0 -- -"$pid" 2> /dev/null || return 0
        sleep 2
    done
    kill -KILL -- -"$pid" 2> /dev/null
}
# shellcheck disable=SC2329
on_exit() {
    local rc=$?
    local f
    for f in "$OUT/monitor.pid" "$OUT/tempwatch.pid"; do
        [[ -s $f ]] && kill "$(cat "$f")" 2> /dev/null
    done
    stop_own_server
    release_lease
    echo "$(date -Is) exit=$rc status=$STATUS" > "$R/DONE"
    say "driver exit rc=$rc status=$STATUS"
}
wait_quiet() {
    until [[ -z $(find "$QUIET" -type f -mmin -6 ! -name "$OWNER" 2> /dev/null) ]]; do
        say "waiting for quiet windows to expire: $(find "$QUIET" -type f -mmin -6 ! -name "$OWNER" | tr '\n' ' ')"
        sleep "$POLL_S"
    done
}
# monitor + junction watcher start once serve.pid exists (the launch happens inside serve_validate)
start_watchers() {
    local i
    for ((i = 0; i < 600; i++)); do
        [[ -s $OUT/serve.pid ]] && break
        sleep 1
    done
    [[ -s $OUT/serve.pid ]] || return 1
    mkdir -p "$OUT/soak"
    python3 "$R/tempwatch.py" "$(echo "$BUSES" | sed 's/\([0-9a-f][0-9a-f]\)/0000:\1:00.0/g;s/ /,/g')" \
        "$OUT/temps1s.log" "$OUT/temps15s.log" "$OUT/serve.pid" "$R/ABORT_TEMP" &
    echo $! > "$OUT/tempwatch.pid"
    "$VENV/bin/python" "$R/monitor.py" "$(cat "$OUT/serve.pid")" "$OUT/serve.log" "$OUT/soak" --family "$$" &
    echo $! > "$OUT/monitor.pid"
}

main() {
    mkdir -p "$R" "$OUT"
    unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
    export HIP_VISIBLE_DEVICES= # CPU children see no card; every GPU command sets its own list
    trap on_exit EXIT
    trap 'say "signal: leaving"; exit 143' TERM INT HUP
    rm -f "$R/DONE" "$R"/ABORT_TEMP "$R"/abort_*
    say "driver start pid $$ slot $SLOT HIP $GPUS buses $BUSES on $(hostname), tree $TREE_T, soak ${SOAK_MIN} min"
    [[ $(cat "$TREE_T/.sha") == "$SHA" ]] || die "$TREE_T is not ${SHA:0:10}"
    [[ $(ulimit -l) == unlimited ]] || die "ulimit -l is $(ulimit -l): launch from a fresh ssh login"
    [[ -z $(ss -ltn "sport = :$PORT" | tail -n +2) ]] || die "port $PORT is in use"
    take_lease
    verify_slot "$SLOT" || die "bus verification failed"
    local try rc
    for try in 1 2 3 4 5 6; do
        wait_quiet
        until enough_memory; do sleep "$POLL_S"; done
        rm -rf "$CACHE"
        rm -f "$R"/abort_*
        say "=== boot attempt $try ==="
        DMESG_BEFORE=$(sudo -n dmesg | tee "$OUT/dmesg.before.txt" | wc -l)
        (start_watchers) &
        (
            export HIP_VISIBLE_DEVICES=$GPUS OWNER SLOT R TREE_T HIP_CLIENT=$GPUS
            export RECIPE=flashnext-mtp2 MODEL=$MODEL_FN GPUS PORT TAG=rdna-ar-soak OUT CACHE VENV
            export READY_TIMEOUT=1800 CELLS=none CONC_PROBE=1 PPL_PROBE=1 PREFIX_PROBE=1 WARMUP=1
            export LOAD_HOOK=$R/soak_hook.sh QUIET_HOLD=0 QUIET_NAME=$OWNER QUIET_OWN=$OWNER
            export BOOT_LOCK REFUSED_GPUS=0,5 STOP_GRACE=180 SERVE_SCRIPT=$R/serve_wrap.sh
            export BENCH_EXTRA="--temperature 0"
            cd "$TREE_T" &&
                timeout --kill-after=60 $((SOAK_MIN * 60 + 5400)) \
                    bash tools/rdna/port_v031/serve_validate.sh EAGER=0
        )
        rc=$?
        wait
        sudo -n dmesg 2> /dev/null | tee "$OUT/dmesg.after.txt" > /dev/null
        {
            echo "dmesg lines before $DMESG_BEFORE, after $(wc -l < "$OUT/dmesg.after.txt")"
            echo "new 'Runlist is getting oversubscribed' lines: $(tail -n +$((DMESG_BEFORE + 1)) "$OUT/dmesg.after.txt" | grep -c 'Runlist is getting oversubscribed')"
            echo "new amdgpu fault/timeout/reset/hang lines: $(tail -n +$((DMESG_BEFORE + 1)) "$OUT/dmesg.after.txt" | grep -iE 'amdgpu' | grep -iE 'fault|timeout|reset|hang|ras |ecc|UTCL2' | grep -vc 'uses VM inv eng')"
            echo "new AER / PCIe Bus Error lines: $(tail -n +$((DMESG_BEFORE + 1)) "$OUT/dmesg.after.txt" | grep -ciE 'AER:|PCIe Bus Error')"
            echo "new amdgpu lines of any kind: $(tail -n +$((DMESG_BEFORE + 1)) "$OUT/dmesg.after.txt" | grep -ci amdgpu)"
        } > "$OUT/dmesg_summary.txt"
        say "serve_validate rc=$rc: $(grep -aE 'READY|PROBE|PPL|CONCURRENCY|LOAD HOOK|done' "$OUT/summary.txt" 2> /dev/null | tr '\n' ' ' | cut -c1-700)"
        if [[ -e $R/abort_quiet ]]; then
            say "boot refused under boot.lock (quiet window): $(cat "$R/abort_quiet"); retrying"
            mv "$OUT" "$OUT.try$try"; mkdir -p "$OUT"
            continue
        fi
        [[ -e $R/abort_bus ]] && die "bus verification refused the launch: $(cat "$R/abort_bus")"
        [[ -e $R/abort_plan ]] && die "plan check refused the launch: $(cat "$R/abort_plan")"
        [[ -e $R/ABORT_TEMP ]] && { STATUS=1; say "THERMAL ABORT: $(cat "$R/ABORT_TEMP")"; }
        if ! { grep -aq '\] done$' "$OUT/summary.txt" 2> /dev/null && ((rc == 0)); }; then
            STATUS=1
            say "run FAILED (rc=$rc)"
        fi
        break
    done
    exit "$STATUS"
}
main "$@"
