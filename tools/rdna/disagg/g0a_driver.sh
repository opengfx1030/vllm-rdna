#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# G0a driver (docs/rdna2/disagg-g0-2026-11-06.md): P/D handoff correctness for
# production Flash-Next (TP4, MTP2, compiled FULL_AND_PIECEWISE) on lease slot B.
# One fresh boot per arm; lease, slot verification, boot.lock/quiet protocol,
# junction watcher and dmesg bookkeeping follow the HwqAbba driver.
#
#   SHA=<spike commit> bash g0a_driver.sh                       # default arm list
#   ARMS="base_ds producer consumer" SHA=... bash g0a_driver.sh
#   setsid nohup bash g0a_driver.sh > driver.log 2>&1 < /dev/null &
#
# Arms (all: VLLM_RDNA2_MOE_DETERMINISTIC=1, VLLM_SERVER_DEV_MODE=1):
#   base_ds        no connector, VLLM_SSM_CONV_STATE_LAYOUT=DS; 4 lengths, run twice with a
#                  prefix-cache reset in between (A / A')
#   base_sd        production conv layout (SD), 4 lengths once (layout control)
#   producer       NixlConnector kv_producer on the file-backed fake nixl_rocm: 4 prefills
#                  (do_remote_decode), handshake capture, per-request state dump
#   consumer       NixlConnector kv_consumer on the fake nixl_rocm + handshake replay:
#                  4 decodes from the dumped state, compared with base_ds
#   consumer_skip  consumer with G0A_SKIP_GROUPS=$SKIP_GROUPS (ablation; expected to diverge)
#   producer_real  kv_producer on the real nixl_rocm/UCX build: registration of every state
#                  region and a real handshake payload (no transfer)
set -uo pipefail
R=${R:-$HOME/w4a8_runs/disagg/g0a}
SHA=${SHA:?SHA=<spike commit> required}
TREE_T=${TREE_T:-$HOME/vllm-disagg-g0-${SHA:0:10}}
SO_TREE=${SO_TREE:-$HOME/vllm-tp8pp2-38ab87ed3a}
SPIKE=$TREE_T/tools/rdna/disagg
export VENV=${VENV:-$HOME/Apps/vllm/venv-7.14.0}
# shellcheck source=tools/rdna/disagg/slots.sh
source "$SPIKE/slots.sh"
HUB=$HOME/.cache/huggingface/hub
MODEL_FN=$HUB/models--wtdcode--Qwen3.8-Flash-Next-AWQ-W4A16/snapshots/0939125b929543a783ce700c90e36dd1a575c00c
BOOT_LOCK=$HOME/w4a8_runs/boot.lock
NIXL_ROOT=${NIXL_ROOT:-$HOME/w4a8_runs/disagg/nixl}
OWNER=Disagg
SLOT=B
GPUS=$(slot_hip "$SLOT")
BUSES=$(slot_buses "$SLOT")
PORT=${PORT:-18700}
ARMS=${ARMS:-"base_ds producer consumer base_sd"}
SKIP_GROUPS=${SKIP_GROUPS:-}
LEASE_HELD=${LEASE_HELD:-0} # 1: the caller already holds slot B (g0b_driver.sh hands it over)
WATCH_PID=
STATUS=0
CUR_OUT=
mkdir -p "$R"
# shellcheck source=tools/rdna/disagg/lease.sh
source "$SPIKE/lease.sh"

# shellcheck disable=SC2329 # called from the EXIT trap
stop_own_server() {
    local pid
    [[ -n $CUR_OUT && -s $CUR_OUT/serve.pid ]] || return 0
    pid=$(cat "$CUR_OUT/serve.pid")
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
# shellcheck disable=SC2329 # the EXIT trap
on_exit() {
    local rc=$?
    [[ -n $WATCH_PID ]] && kill "$WATCH_PID" 2> /dev/null
    stop_own_server
    [[ $LEASE_HELD == 1 ]] || release_all
    echo "$(date -Is) exit=$rc status=$STATUS" > "$R/DONE"
    say "driver exit rc=$rc status=$STATUS"
}
wait_quiet() {
    until [[ -z $(find "$QUIET" -type f -mmin -6 ! -name "$OWNER" 2> /dev/null) ]]; do
        say "waiting for quiet windows to expire: $(find "$QUIET" -type f -mmin -6 ! -name "$OWNER" | tr '\n' ' ')"
        sleep "$POLL_S"
    done
}
gpus_clear() { # nothing left in my server's process group, and the slot memory is back
    [[ -n ${LAST_SPID:-} ]] && pgrep -g "$LAST_SPID" > /dev/null 2>&1 && return 1
    enough_memory "$SLOT"
}

# Arm configuration: ARM_ENV (exported for the boot), ARM_EXTRA (vllm args, no spaces
# inside a word), ARM_NIXL (fake | real | none).
arm_setup() {
    local arm=$1
    ARM_ENV=(VLLM_RDNA2_MOE_DETERMINISTIC=1 VLLM_SERVER_DEV_MODE=1 VLLM_SSM_CONV_STATE_LAYOUT=DS)
    ARM_EXTRA=""
    ARM_NIXL=none
    local ext="--worker-extension-cls=g0a_ext.G0aWorkerExtension"
    local kvp='--kv-transfer-config={"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_buffer_device":"cuda","kv_connector_extra_config":{"kv_lease_duration":1800}}'
    local kvc='--kv-transfer-config={"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_buffer_device":"cuda"}'
    case $arm in
        base_ds) ARM_EXTRA="$ext" ;;
        base_sd)
            ARM_ENV=(VLLM_RDNA2_MOE_DETERMINISTIC=1 VLLM_SERVER_DEV_MODE=1)
            ARM_EXTRA="$ext"
            ;;
        producer)
            ARM_NIXL=fake
            ARM_ENV+=("G0A_STORE_DIR=$R/store")
            ARM_EXTRA="$ext $kvp"
            ;;
        producer_real)
            ARM_NIXL=real
            ARM_ENV+=("G0A_STORE_DIR=$R/store_real")
            ARM_EXTRA="$ext $kvp"
            ;;
        consumer | consumer_skip)
            ARM_NIXL=fake
            ARM_ENV+=("G0A_STORE_DIR=$R/store" "G0A_STATS_DIR=$R/arms/$arm/stats")
            [[ $arm == consumer_skip ]] && ARM_ENV+=("G0A_SKIP_GROUPS=$SKIP_GROUPS")
            ARM_EXTRA="$ext $kvc"
            ;;
        *) die "unknown arm $arm" ;;
    esac
    local pp="$SPIKE"
    case $ARM_NIXL in
        fake) pp="$SPIKE:$SPIKE/fake_nixl" ;;
        real)
            pp="$SPIKE:$NIXL_ROOT/site"
            ARM_ENV+=("NIXL_PLUGIN_DIR=$NIXL_ROOT/site/nixl_rocm.libs/nixl" "HSA_ENABLE_IPC_MODE_LEGACY=1"
                "UCX_ROCM_COPY_SIGPOOL_MAX_ELEMS=inf" "UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=inf")
            ;;
    esac
    ARM_ENV+=("PYTHONPATH=$pp" "G0A_ROOT=$R" "G0A_ARM=$arm")
}

run_arm() {
    local arm=$1 out=$R/arms/$1 cache=$R/cache-$1 try rc
    CUR_OUT=$out
    for try in 1 2 3 4 5 6; do
        wait_quiet
        until enough_memory "$SLOT"; do sleep "$POLL_S"; done
        [[ -d $out ]] && mv "$out" "$out.prev-$(date +%H%M%S)"
        mkdir -p "$out/stats"
        rm -rf "$cache"
        rm -f "$R/abort_quiet.$arm" "$R/abort_bus.$arm" "$R/ABORT_TEMP.$arm"
        arm_setup "$arm"
        say "=== arm $arm attempt $try: env ${ARM_ENV[*]} extra ${ARM_EXTRA} ==="
        DMESG_BEFORE=$(sudo -n dmesg | tee "$out/dmesg.before.txt" | wc -l)
        python3 "$SPIKE/tempwatch.py" "${BUSES// /,}" \
            "$out/temps1s.log" "$out/temps15s.log" "$out/serve.pid" "$R/ABORT_TEMP.$arm" &
        WATCH_PID=$!
        (
            cd "$TREE_T" &&
                env "${ARM_ENV[@]}" EXTRA_ARGS="$ARM_EXTRA" \
                    ARM="$arm" TREE_T="$TREE_T" ARM_OUT="$out" R="$R" SPIKE="$SPIKE" \
                    RECIPE=flashnext-mtp2 MODEL="$MODEL_FN" GPUS="$GPUS" PORT="$PORT" \
                    TAG="g0a-$arm" OUT="$out" CACHE="$cache" VENV="$VENV" READY_TIMEOUT=1800 \
                    CELLS=none CONC_PROBE="${CONC_PROBE:-0}" PPL_PROBE=1 PREFIX_PROBE=0 WARMUP=0 \
                    LOAD_HOOK="$SPIKE/g0a_hook.sh" QUIET_HOLD=0 QUIET_NAME="$OWNER" QUIET_OWN="$OWNER" \
                    BOOT_LOCK="$BOOT_LOCK" REFUSED_GPUS=0,5 SO_BUILD_DIR="$SO_TREE" \
                    STOP_GRACE=180 SERVE_SCRIPT="$SPIKE/g0a_serve_wrap.sh" \
                    timeout --kill-after=60 7200 bash tools/rdna/port_v031/serve_validate.sh \
                    EAGER=0
        )
        rc=$?
        LAST_SPID=$(cat "$out/serve.pid" 2> /dev/null)
        kill "$WATCH_PID" 2> /dev/null; WATCH_PID=
        # shellcheck disable=SC2024 # the redirect target is mine; only dmesg needs sudo
        sudo -n dmesg > "$out/dmesg.after.txt" 2> /dev/null
        {
            echo "arm $arm: dmesg lines before $DMESG_BEFORE, after $(wc -l < "$out/dmesg.after.txt")"
            echo "new 'Runlist is getting oversubscribed' lines: $(tail -n +$((DMESG_BEFORE + 1)) "$out/dmesg.after.txt" | grep -c 'Runlist is getting oversubscribed')"
            echo "new amdgpu fault/timeout/reset/hang lines: $(tail -n +$((DMESG_BEFORE + 1)) "$out/dmesg.after.txt" | grep -iE 'amdgpu' | grep -iE 'fault|timeout|reset|hang|ras |ecc|UTCL2' | grep -vc 'uses VM inv eng')"
            echo "new AER / PCIe Bus Error lines: $(tail -n +$((DMESG_BEFORE + 1)) "$out/dmesg.after.txt" | grep -ciE 'AER:|PCIe Bus Error')"
        } > "$out/dmesg_summary.txt"
        say "serve_validate rc=$rc: $(grep -aE 'READY|PROBE|PPL|LOAD HOOK|done' "$out/summary.txt" 2> /dev/null | tr '\n' ' ' | cut -c1-600)"
        if [[ -e $R/abort_quiet.$arm ]]; then
            say "boot refused under boot.lock (quiet window): $(cat "$R/abort_quiet.$arm"); retrying"
            continue
        fi
        [[ -e $R/abort_bus.$arm ]] && die "bus verification refused the launch: $(cat "$R/abort_bus.$arm")"
        [[ -e $R/ABORT_TEMP.$arm ]] && { STATUS=1; say "THERMAL ABORT: $(cat "$R/ABORT_TEMP.$arm")"; }
        if ! { grep -aq '\] done$' "$out/summary.txt" 2> /dev/null && ((rc == 0)); }; then
            STATUS=1
            say "arm $arm FAILED (rc=$rc)"
        fi
        break
    done
    CUR_OUT=
    local t0
    t0=$(date +%s)
    until gpus_clear; do
        say "waiting for the slot to drain"
        (($(date +%s) - t0 > 600)) && { say "slot did not drain in 10 min"; STATUS=1; break; }
        sleep 10
    done
    rm -rf "$cache"
}

main() {
    mkdir -p "$R/arms"
    unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
    export HIP_VISIBLE_DEVICES= # CPU children see no card; every GPU command sets its own list
    trap on_exit EXIT
    trap 'say "signal: leaving"; exit 143' TERM INT HUP
    rm -f "$R/DONE" "$R"/ABORT_TEMP.*
    say "driver start pid $$ slot $SLOT HIP $GPUS buses $BUSES on $(hostname), tree $TREE_T, arms: $ARMS"
    [[ $(cat "$TREE_T/.sha") == "$SHA" ]] || die "$TREE_T is not ${SHA:0:10}"
    [[ $(ulimit -l) == unlimited ]] || die "ulimit -l is $(ulimit -l): launch from a fresh ssh login"
    [[ -z $(ss -ltn "sport = :$PORT" | tail -n +2) ]] || die "port $PORT is in use"
    [[ -s $R/prompts.json ]] || die "$R/prompts.json missing (g0a_requests.py prompts)"
    [[ $LEASE_HELD == 1 ]] || take_leases "$SLOT"
    verify_slot "$SLOT" || die "bus verification failed"
    export HIP_VISIBLE_DEVICES=$GPUS # probes see only the leased cards
    local a
    for a in $ARMS; do
        [[ $a == producer ]] && rm -rf "$R/store"
        [[ $a == producer_real ]] && rm -rf "$R/store_real"
        run_arm "$a"
        [[ -e $R/STOP ]] && { say "STOP file: leaving"; break; }
    done
    exit "$STATUS"
}
main "$@"
