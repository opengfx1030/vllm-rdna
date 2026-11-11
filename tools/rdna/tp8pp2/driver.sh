#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# ONE boot of a flashnext recipe (production compiled mode, FULL_AND_PIECEWISE) on 4 or
# 8 V620s, or a hetero-MoE island (CFG=h1, h1v), with probes (serve_validate.sh),
# dmesg/SEL/temperature/memory capture and an
# optional measured hook (hook.sh). Needs the lease holder files of the slots it uses
# (OWNER.slotA / OWNER.slotB, OWNER.w7800cb for CFG=h1, OWNER.a2 + OWNER.b4 for CFG=h1v)
# and an exact-SHA tree.
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
#      h1    hetero-MoE stage 1 (T14), RECIPE default flashnext-hetero-h1. Slot HA = hub
#            W7800 (HIP 0, bus cb) + slot A (HIP 1-4, buses ce d3 d8 db). The serve command
#            sees HIP 0 ONLY (the hub engine); tools/rdna/rdna_hetero_launch.sh starts the
#            four V620 servers from the recipe's VLLM_HETERO_MOE_SERVERS. HIP 5 refused.
#            Leases: OWNER.slotA AND OWNER.w7800cb, no foreign *.slotA* / *.w7800cb*.
#      h1v   the V620 stand-in (interim layout since the 2026-11-08 case move): slot HV,
#            RECIPE default flashnext-hetero-h1-v620, hub = HIP 1 (bus d4), servers HIP 2-5
#            (buses d9 dc df e2); HIP 0 is not touched. The serve command sees HIP 1 ONLY.
#            Leases: OWNER.a2 AND OWNER.b4 (STANDIN_LEASES), no foreign *.a2 / *.b4.
#      Watchers, the free-memory check (>= 28 GiB) and the bus verification cover the whole
#      slot (hub + servers).
# SHA        the tree $HOME/vllm-tp8pp2-<sha10> (TREE_T overrides) must carry this .sha
# RECIPE     default flashnext-mtp2
# EXTRA_OV="KEY=value ..."  recipe overrides appended after the CFG ones (KV=auto GMEM=0.90)
# SERVE_ENV="K=V K=V"        extra environment for the server (TUNABLEOP_ROOT=<rows dir>)
# PLAN=<json>                CFG=h1/h1v: the hetero plan, passed to the serve command as
#                            VLLM_HETERO_MOE_PLAN, with HETERO_LOG_DIR=$OUT/hetero and
#                            HETERO_PYTHON=$VENV/bin/python (EXTRA_OV wins over all three).
#                            Default h1: the T10 plan of ExpertStats (130 hub slots per
#                            layer), h1v: its V620-hub twin (40 slots per layer, 4 servers);
#                            both in $TREE_T/bench_results. `cli.py stripe` writes a uniform
#                            stripe plan for a hand-made PLAN.
# KFD_EVICT_HETERO=1         exported for h1/h1v: kfd_evictions.py (the evicted_ms windows of
#                            serve_validate.sh and hook.sh) also watches the hub's engine core
#                            and the expert servers (labels EngineCore, hx_<device id>)
# VLLM_HX_DEV_LIB            a variable of the caller's environment reaches the hetero
#                            servers unchanged (the launcher's servers inherit it); SERVE_ENV
#                            may set it as well
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
CFG=${CFG:?CFG=tp8|pp2|tp4a|tp4b|rtp8|rpp2|h1|h1v}
case $CFG in
    h1) RECIPE=${RECIPE:-flashnext-hetero-h1} ;;
    h1v) RECIPE=${RECIPE:-flashnext-hetero-h1-v620} ;;
esac
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
    h1) SLOT=HA OV=() ;;
    h1v) SLOT=HV OV=() ;;
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
# What the serve command sees: the whole slot, except CFG=h1/h1v, where the hub engine
# sees ONLY its own GPU (rdna_hetero_launch.sh starts the V620 servers itself, from the
# recipe's VLLM_HETERO_MOE_SERVERS buses). Layout before the case move: HIP 0 is allowed
# for h1 only. Interim layout (h1v): all six V620s are allowed, HIP 0 is simply not used.
SERVE_GPUS=$GPUS
REFUSED=0,5
case $CFG in
    h1) SERVE_GPUS=$HUB_HIP REFUSED=5 ;;
    h1v) SERVE_GPUS=$STANDIN_HUB_HIP REFUSED= ;;
esac
NGPU=$(echo "$GPUS" | tr ',' '\n' | wc -l)
OUT=$R/arms/$ARM
CACHE=${CACHE:-$R/cache-$CFG}
# CFG=h1/h1v: the hetero plan; the defaults are committed in the tree
case $CFG in
    h1) PLAN=${PLAN:-$TREE_T/bench_results/2026-11-07_expert-stats-t10/H1.W1.plan.json} ;;
    h1v) PLAN=${PLAN:-$TREE_T/bench_results/2026-11-10_hetero-hub-standin/H1v.W1.plan.json} ;;
esac
if [[ $CFG == h1 || $CFG == h1v ]]; then
    # caller overrides (EXTRA_OV) come later in OV and win
    OV=("VLLM_HETERO_MOE_PLAN=$PLAN" "HETERO_LOG_DIR=$OUT/hetero" "HETERO_PYTHON=$VENV/bin/python" "${OV[@]}")
    # the evicted_ms windows (serve_validate.sh, hook.sh) also watch the hub's engine core
    # (TP1: no VLLM::Worker exists) and the expert servers
    export KFD_EVICT_HETERO=1
fi
STATUS=0
WATCH_PIDS=()

say() { echo "[$(date '+%F %T')] $*"; }
die() {
    say "FATAL: $*"
    STATUS=1
    exit 1
}
need_leases() {
    local f x
    local -a standin_leases
    read -r -a standin_leases <<< "${STANDIN_LEASES:-}"
    case $CFG in
        h1) [[ -e $LEASE/$OWNER.slotA && -e $LEASE/$OWNER.w7800cb ]] || return 1 ;;
        h1v)
            for x in "${standin_leases[@]}"; do
                [[ -e $LEASE/$OWNER.$x ]] || return 1
            done
            ;;
        *) [[ -e $LEASE/$OWNER.slotA || -e $LEASE/$OWNER.slotB ]] || return 1 ;;
    esac
    for x in A B; do
        [[ $SLOT == *$x* ]] || continue
        for f in "$LEASE"/*.slot"$x"*; do
            [[ -e $f ]] || continue
            [[ ${f##*/} == "$OWNER".* ]] && continue
            say "foreign lease file ${f##*/}"
            return 1
        done
    done
    if [[ $CFG == h1 ]]; then
        for f in "$LEASE"/*.w7800cb*; do
            [[ -e $f ]] || continue
            [[ ${f##*/} == "$OWNER".* ]] && continue
            say "foreign lease file ${f##*/}"
            return 1
        done
    fi
    if [[ $CFG == h1v ]]; then
        for x in "${standin_leases[@]}"; do
            for f in "$LEASE"/*."$x"; do
                [[ -e $f ]] || continue
                [[ ${f##*/} == "$OWNER.$x" ]] && continue
                say "foreign lease file ${f##*/}"
                return 1
            done
        done
    fi
    return 0
}
# CFG=h1/h1v: the value serve_rdna.sh will see for a recipe key (its precedence: recipe
# default < the caller's environment, SERVE_ENV < the trailing KEY=value arguments, i.e.
# the CFG overrides and EXTRA_OV).
hetero_setting() {
    local key=$1 val kv
    local -a senv
    val=$(grep -E "^$key=" "$TREE_T/tools/rdna/recipes/$RECIPE.env" 2> /dev/null | tail -1)
    val=${val#*=}
    read -r -a senv <<< "$SERVE_ENV"
    for kv in "${senv[@]}" "${OV[@]}"; do
        [[ $kv == "$key="* ]] && val=${kv#*=}
    done
    printf '%s' "${val//\"/}"
}
# CFG=h1/h1v: the plan must load and validate (CPU only: cli.py never imports the vLLM
# stack) before anything boots, and the decode dispatch rule of the arm (row or primary,
# VLLM_HETERO_MOE_DECODE_DISPATCH) must be one the plan can run; its description, with
# the spill K table of that rule, goes to plan_check.log and the driver log
check_plan() {
    [[ $CFG == h1 || $CFG == h1v ]] || return 0
    [[ -r $PLAN ]] || die "PLAN=$PLAN is not a readable file"
    local cli=$TREE_T/vllm/distributed/hetero_moe/cli.py valid rule spill
    local -a spillarg=()
    "$VENV/bin/python" "$cli" validate "$PLAN" \
        > "$OUT/plan_check.log" 2>&1 || die "plan check failed: $(tail -3 "$OUT/plan_check.log" | tr '\n' ' ')"
    valid=$(tail -1 "$OUT/plan_check.log")
    rule=$(hetero_setting VLLM_HETERO_MOE_DECODE_DISPATCH)
    spill=$(hetero_setting VLLM_HETERO_MOE_SPILL_K)
    [[ -n $spill ]] && spillarg=(--spill-k "$spill")
    "$VENV/bin/python" "$cli" dispatch "$PLAN" --rule "${rule:-row}" "${spillarg[@]}" \
        >> "$OUT/plan_check.log" 2>&1 || die "decode dispatch check failed: $(tail -3 "$OUT/plan_check.log" | tr '\n' ' ')"
    say "plan: $PLAN ($valid)"
    say "$(tail -1 "$OUT/plan_check.log")"
}
# CFG=h1/h1v: judge the concurrency probes by transport and determinism (the hx digest of
# the hub and of every server, solo and solo2 bit-identical in every run) and report the
# token flips against the T=3 solo reference as information (conc_gate.py, hx doc 8.9).
# CONC_GATE=probe keeps the probe's own verdict only. The result goes to conc_gate.txt, the
# first line to summary.txt and the driver log; it never changes the exit status of the arm.
hetero_conc_gate() {
    [[ $CFG == h1 || $CFG == h1v ]] || return 0
    [[ ${CONC_GATE:-hetero} == hetero ]] || return 0
    local verdict
    "$VENV/bin/python" "$TREE_T/tools/rdna/hetero_moe/conc_gate.py" "$OUT" \
        > "$OUT/conc_gate.txt" 2>&1
    verdict=$(head -1 "$OUT/conc_gate.txt")
    printf '[%s] %s\n' "$(date +%T)" "$verdict" >> "$OUT/summary.txt"
    say "$verdict"
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
    if [[ $SERVE_GPUS != "$GPUS" ]]; then
        say "hetero: hub engine sees HIP $SERVE_GPUS only, plan $PLAN, servers on the recipe's buses (refused HIP ${REFUSED:-none})"
    fi
    [[ $(cat "$TREE_T/.sha") == "$SHA" ]] || die "$TREE_T is not ${SHA:0:10}"
    [[ $(ulimit -l) == unlimited ]] || die "ulimit -l is $(ulimit -l): launch from a fresh ssh login"
    [[ -z $(ss -ltn "sport = :$PORT" | tail -n +2) ]] || die "port $PORT is in use"
    need_leases || die "lease check failed (foreign holder or no $OWNER holder)"
    verify_slot "$SLOT" || die "bus verification failed"
    check_plan
    export HIP_VISIBLE_DEVICES=$GPUS
    wait_quiet
    until enough_memory; do sleep "$POLL_S"; done
    T0=$(date '+%F %T')
    echo "$T0" > "$OUT/t0.txt"
    DMESG_BEFORE=$(sudo -n dmesg | tee "$OUT/dmesg.before.txt" | wc -l)
    sel_info > "$OUT/sel.before.txt"
    # hetero: the supervisor stops hub, then servers (up to 30 s per stage) before it
    # exits, so the thermal abort must not SIGKILL its group after 15 s
    TEMPWATCH_GRACE=$([[ $CFG == h1 || $CFG == h1v ]] && echo 120 || echo 15) \
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
                RECIPE="$RECIPE" MODEL="$MODEL_FN" GPUS="$SERVE_GPUS" HIP_VISIBLE_DEVICES="$SERVE_GPUS" PORT="$PORT" \
                TAG="tp8pp2-$ARM" OUT="$OUT" CACHE="$CACHE" VENV="$VENV" READY_TIMEOUT="$READY_TIMEOUT" \
                CELLS="$CELLS" CONC_PROBE="${CONC_PROBE:-1}" PPL_PROBE="${PPL_PROBE:-1}" \
                PREFIX_PROBE="${PREFIX_PROBE:-0}" WARMUP="${WARMUP:-1}" \
                ${LOAD_HOOK:+LOAD_HOOK="$LOAD_HOOK"} QUIET_HOLD=0 QUIET_NAME="$OWNER" QUIET_OWN="$OWNER" \
                BOOT_LOCK="$BOOT_LOCK" REFUSED_GPUS="$REFUSED" STOP_GRACE=180 \
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
    hetero_conc_gate
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
# run only when executed: the tests source this file to call need_leases / check_plan
[[ ${BASH_SOURCE[0]} != "$0" ]] || main "$@"
