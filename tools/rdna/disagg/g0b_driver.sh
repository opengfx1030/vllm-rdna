#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# G0b driver (docs/rdna2/disagg-g0-2026-11-06.md): the transport matrix between HIP 4 (slot A,
# root complex c0) and HIP 6 (slot B, root complex 00). Takes both leases, verifies the slots
# by PCI bus, runs g0b_run.sh once per entry of $RUNS, then releases slot A and either keeps
# slot B (KEEP_B=1, for g0a_driver.sh with LEASE_HELD=1) or releases everything.
#
#   SHA=<spike commit> setsid nohup bash g0b_driver.sh > driver.log 2>&1 < /dev/null &
#   RUNS="smoke nixl_ipc" KEEP_B=1 SHA=... bash g0b_driver.sh
set -uo pipefail
R=${R:-$HOME/w4a8_runs/disagg/g0b}
SHA=${SHA:?SHA=<spike commit> required}
TREE_T=${TREE_T:-$HOME/vllm-disagg-g0-${SHA:0:10}}
SPIKE=$TREE_T/tools/rdna/disagg
export VENV=${VENV:-$HOME/Apps/vllm/venv-7.14.0}
# shellcheck source=tools/rdna/disagg/slots.sh
source "$SPIKE/slots.sh"
OWNER=Disagg
STATUS=0
KEEP_B=${KEEP_B:-0}
LEASE_HELD=${LEASE_HELD:-0} # 1: the caller holds slots A and B (lease_ctl.sh take A B) and releases them
DRY_CPU=${DRY_CPU:-0} # 1: orchestration test without GPUs or leases (probes on CPU memory)
MEM_MODE=gpu
RUNS=${RUNS:-"smoke smoke_default smoke_shared nixl_ipc nixl_default nixl_shared nixl_noipc hipipc_disjoint hipipc_shared hostbounce multi4"}
RUN_TIMEOUT=${RUN_TIMEOUT:-2700}
DOCKER_WAIT_MAX_S=${DOCKER_WAIT_MAX_S:-1200}
mkdir -p "$R"
# shellcheck source=tools/rdna/disagg/lease.sh
source "$SPIKE/lease.sh"

ALL_PAYLOAD="WRITE:contig:249.1M:100,READ:contig:249.1M:100,WRITE:contig:912.1M:100,READ:contig:912.1M:100"
# plan NAME -> ENGINE VISIBLE TLS CELLS UCX_LOG (names are the run directories under $R)
plan() {
    UCX_LOG=
    PAIRS=
    case $1 in
        smoke) # first contact: does a GPU transfer work at all
            ENGINE=nixl VISIBLE=disjoint TLS=ipc_only WINDOW=0
            CELLS="NOTIF:-:0:20,WRITE:contig:1M:3,READ:contig:1M:3,WRITE:scatter:4M:3"
            ;;
        smoke_default) # first contact with UCX's own transport choice (protocol log)
            ENGINE=nixl VISIBLE=disjoint TLS=default WINDOW=0 UCX_LOG=info
            CELLS="NOTIF:-:0:20,WRITE:contig:1M:3,READ:contig:1M:3,WRITE:scatter:4M:3"
            ;;
        smoke_shared)
            ENGINE=nixl VISIBLE=shared TLS=ipc_only WINDOW=0
            CELLS="NOTIF:-:0:20,WRITE:contig:1M:3,READ:contig:1M:3,WRITE:scatter:4M:3"
            ;;
        nixl_ipc) # headline: NIXL/UCX with no host RMA transport (UCX_TLS=rocm_copy,rocm_ipc,tcp,self)
            ENGINE=nixl VISIBLE=disjoint TLS=ipc_only WINDOW=1
            CELLS="NOTIF:-:0:500,WRITE:contig:4K:300,WRITE:contig:64K:300,WRITE:contig:1M:200,$ALL_PAYLOAD,WRITE:scatter:249.1M:100,READ:scatter:249.1M:50,WRITE:scatter:912.1M:100,WRITE:strided:249.1M:50"
            ;;
        nixl_default) # UCX_TLS unset: what UCX selects on its own, with its protocol log
            ENGINE=nixl VISIBLE=disjoint TLS=default WINDOW=1 UCX_LOG=info
            CELLS="NOTIF:-:0:200,WRITE:contig:249.1M:100,READ:contig:249.1M:100,WRITE:contig:912.1M:50"
            ;;
        nixl_shared) # both processes see both GPUs
            ENGINE=nixl VISIBLE=shared TLS=ipc_only WINDOW=1
            CELLS="WRITE:contig:249.1M:100,READ:contig:249.1M:100,WRITE:contig:912.1M:50,READ:contig:912.1M:50"
            ;;
        nixl_noipc) # host bounce through UCX rocm_copy (fallback D inside NIXL)
            ENGINE=nixl VISIBLE=disjoint TLS=no_ipc WINDOW=1
            CELLS="NOTIF:-:0:200,WRITE:contig:249.1M:50,READ:contig:249.1M:50,WRITE:contig:912.1M:20"
            ;;
        hipipc_disjoint) # fallback B: hipIpc + hipMemcpyAsync
            ENGINE=hipipc VISIBLE=disjoint TLS=default WINDOW=1
            CELLS="$ALL_PAYLOAD,WRITE:scatter:249.1M:100"
            ;;
        hipipc_shared)
            ENGINE=hipipc VISIBLE=shared TLS=default WINDOW=1
            CELLS="$ALL_PAYLOAD,WRITE:scatter:249.1M:100"
            ;;
        hostbounce) # fallback D: device -> pinned host -> shm -> pinned host -> device
            ENGINE=hostbounce VISIBLE=disjoint TLS=default WINDOW=1
            CELLS="WRITE:contig:249.1M:50,WRITE:contig:912.1M:20,WRITE:scatter:249.1M:20"
            ;;
        multi4) # the TP4 -> TP4 handoff: four concurrent cross-root pairs (HIP 1-4 -> 6-9)
            ENGINE=nixl VISIBLE=disjoint TLS=ipc_only WINDOW=1
            PAIRS="1:ce:6:0a 2:d3:7:0f 3:d8:8:14 4:db:9:17"
            CELLS="WRITE:contig:249.1M:50,READ:contig:249.1M:30,WRITE:contig:912.1M:30,READ:contig:912.1M:10"
            ;;
        multi2)
            ENGINE=nixl VISIBLE=disjoint TLS=ipc_only WINDOW=1
            PAIRS="1:ce:6:0a 2:d3:7:0f"
            CELLS="WRITE:contig:249.1M:50,READ:contig:249.1M:30,WRITE:contig:912.1M:30"
            ;;
        *) die "unknown run $1" ;;
    esac
}

# shellcheck disable=SC2329 # the EXIT trap
on_exit() {
    local rc=$?
    if [[ $DRY_CPU == 1 || $LEASE_HELD == 1 ]]; then
        say "leases stay with the caller"
    elif [[ $KEEP_B == 1 && $rc == 0 && $STATUS == 0 ]]; then
        say "keeping slot B for the G0a driver"
        rm -f "$QUIET/$OWNER"
    else
        release_all
    fi
    echo "$(date -Is) exit=$rc status=$STATUS" > "$R/DONE"
    say "driver exit rc=$rc status=$STATUS"
}

main() {
    unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
    export HIP_VISIBLE_DEVICES= # CPU children see no card; every GPU command sets its own list
    trap on_exit EXIT
    trap 'say "signal: leaving"; exit 143' TERM INT HUP
    rm -f "$R/DONE"
    [[ -s $TREE_T/.sha && $(cat "$TREE_T/.sha") == "$SHA" ]] || die "$TREE_T is not ${SHA:0:10}"
    say "G0b driver pid $$ on $(hostname), tree $TREE_T, runs: $RUNS"
    if [[ $DRY_CPU == 1 ]]; then
        MEM_MODE=cpu
        say "DRY_CPU: no leases, no GPU, probes on CPU memory"
    else
        if [[ $LEASE_HELD == 1 ]]; then say "slots A and B already leased by the caller"; else take_leases A B; fi
        verify_slot A || die "slot A bus verification failed"
        verify_slot B || die "slot B bus verification failed"
    fi
    local name rc docker_gave_up=0
    local -A smoke_ran=() smoke_ok=() # "TLS/VISIBLE" -> 1: the smoke run of that configuration ran / passed
    for name in $RUNS; do
        plan "$name"
        if [[ $DRY_CPU == 1 ]]; then # small cells
            CELLS="WRITE:contig:1M:5,READ:contig:1M:5"
        fi
        if [[ $name != smoke* && $ENGINE == nixl && ${smoke_ran[$TLS/$VISIBLE]:-0} == 1 && ${smoke_ok[$TLS/$VISIBLE]:-0} != 1 ]]; then
            say "=== run $name skipped: the $TLS/$VISIBLE smoke run failed ==="
            continue
        fi
        say "=== run $name: engine=$ENGINE visible=$VISIBLE tls=$TLS window=$WINDOW ==="
        waited=0
        ((docker_gave_up)) && waited=$DOCKER_WAIT_MAX_S
        protocol=protected
        while [[ $WINDOW == 1 ]] && pgrep -f '^(/usr/sbin/|/usr/bin/)?docker buildx build' > /dev/null; do
            if ((waited >= DOCKER_WAIT_MAX_S)); then
                say "docker buildx build still running after ${waited}s: run $name WITHOUT the measured-window protocol"
                WINDOW=0
                docker_gave_up=1
                protocol="UNPROTECTED (docker buildx build running)"
                break
            fi
            say "docker buildx build running: measured windows wait (${waited}s)"
            sleep 60
            waited=$((waited + 60))
        done
        rdir=$R/$name
        if [[ -n $PAIRS ]]; then
            rdir=$R/$name-p0
            # shellcheck disable=SC2086 # PAIRS is a word list of SRC_HIP:SRC_BUS:DST_HIP:DST_BUS
            env TREE_T="$TREE_T" G0B_ROOT="$R" MEM="$MEM_MODE" WINDOW="$WINDOW" UCX_LOG="$UCX_LOG" OWNER="$OWNER" \
                timeout --kill-after=60 "$RUN_TIMEOUT" setsid -w bash "$SPIKE/g0b_multi.sh" \
                "$name" "$ENGINE" "$TLS" "$CELLS" $PAIRS
        else
            env TREE_T="$TREE_T" G0B_ROOT="$R" MEM="$MEM_MODE" WINDOW="$WINDOW" UCX_LOG="$UCX_LOG" OWNER="$OWNER" \
                timeout --kill-after=60 "$RUN_TIMEOUT" setsid -w bash "$SPIKE/g0b_run.sh" \
                "$name" "$ENGINE" "$VISIBLE" "$TLS" "$CELLS" "driver run $name"
        fi
        rc=$?
        echo "$protocol" > "$rdir/protocol.txt"
        say "run $name rc=$rc: $(cat "$rdir/rc.txt" 2> /dev/null); $(tail -n 3 "$rdir/dmesg_summary.txt" 2> /dev/null | tr '\n' ' ')"
        ((rc == 0)) || say "run $name FAILED rc=$rc: a finding, logs in $rdir"
        [[ -e $R/STOP ]] && { say "STOP file: leaving"; break; }
        if [[ $name == smoke* ]]; then
            smoke_ran[$TLS/$VISIBLE]=1
            ((rc == 0)) && smoke_ok[$TLS/$VISIBLE]=1
        fi
    done
    if [[ $DRY_CPU == 1 || $LEASE_HELD == 1 ]]; then
        say "G0b runs finished; nothing to release here"
    else
        release_slot A
        say "slot A released; ping Tp8Pp2/Soak"
    fi
}
main "$@"
