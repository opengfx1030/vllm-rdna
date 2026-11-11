#!/usr/bin/env bash
# W4A8 MoE quality arms on HIP 2-5, one boot per arm on one exact tree:
#   A = W4A16 (VLLM_RDNA2_W4A8_SDOT4=0), B = W4A8 (flashnext-mtp2 default),
#   A2 = W4A16 repeat, B2 = W4A8 repeat.
# Lease: ~/w4a8_runs/lanes/lease/README.md (waits for other holders, holds
# W4A8Quality.hip2..5 while it runs). Every boot takes BOOT_LOCK and needs
# >= 28 GiB free on each card. Arm outputs: $W4Q/arms/<arm>/ (DONE when finished).
# An attempt that ends in an rdna_ar wedge is discarded and the arm is measured
# again from scratch (MAX_TRIES): its boot files move to <arm>/attempt<k>/ (hook.sh
# moves its data there too), and the wedge marker is removed so the next boot
# keeps rdna_ar (the marker would silently force RCCL).
#   setsid nohup bash run_arms.sh > run_arms.log 2>&1 < /dev/null &
set -uo pipefail
W4Q=${W4Q:-$HOME/w4a8_runs/w4a8-quality-2026-11-04}
TREE=${TREE:-$HOME/vllm-w4a8q-4a3d}
VENV=$HOME/Apps/vllm/venv-7.14.0_0.32.0
MODEL=$HOME/.cache/huggingface/hub/models--wtdcode--Qwen3.8-Flash-Next-AWQ-W4A16/snapshots/0939125b929543a783ce700c90e36dd1a575c00c
LEASE=$HOME/w4a8_runs/lanes/lease
GPUS=2,3,4,5
PORT=${PORT:-18460}
ARMS=${ARMS:-A B A2 B2}
OWNER=${OWNER:-W4A8Quality}
MAX_TRIES=${MAX_TRIES:-3}
MARKER=$W4Q/cache/vllm/rdna_ar_wedged
export W4Q TASKS=${TASKS:-gsm8k,humaneval,mbpp,ifeval,toolcall,needle,decision}
export LIMIT=${LIMIT:-120}
say() { echo "[$(date '+%F %T')] $*"; }

free_gib() {
    HIP_VISIBLE_DEVICES=$GPUS "$VENV/bin/python" -c '
import torch
print(" ".join(f"{torch.cuda.mem_get_info(i)[0] / 2**30:.1f}"
               for i in range(torch.cuda.device_count())))'
}

enough_memory() {
    local f
    f=$(free_gib) || return 1
    say "free GiB on HIP $GPUS: $f"
    for g in $f; do
        awk -v g="$g" 'BEGIN { exit !(g >= 28) }' || return 1
    done
}

others_hold() {
    find "$LEASE" -maxdepth 1 -name '*.hip[2-5]' ! -name "$OWNER.*" | grep -q .
}

# Move one boot's log files aside (hook.sh moves that boot's data next to them).
stash_attempt() {
    local out=$1 k=1 f
    [[ -e $out/serve.log ]] || return 0
    while [[ -e $out/attempt$k ]]; do k=$((k + 1)); done
    mkdir -p "$out/attempt$k"
    for f in serve.log summary.txt load_hook.txt probe.txt ppl.txt prefix.txt \
        warmup.txt models.json serve.pid w4a8_debug.txt bench.log lp.log \
        stall.txt conc.txt; do
        [[ -e $out/$f ]] && mv "$out/$f" "$out/attempt$k/"
    done
    for f in "$out"/metrics-*.txt; do
        [[ -e $f ]] && mv "$f" "$out/attempt$k/"
    done
    say "stashed $out attempt$k"
}

[[ "$(cat "$TREE/.sha")" == 4a3db777aefa28e06036a355b20c4e0cd3edbcd9 ]] ||
    { say "tree $TREE is not 4a3db777ae"; exit 1; }
until ! others_hold && enough_memory; do
    say "waiting: holders $(find "$LEASE" -maxdepth 1 -name '*.hip*' -printf '%f ')"
    sleep 60
done
for h in 2 3 4 5; do
    echo "$OWNER $(date -Is) run_arms.sh pid $$" > "$LEASE/$OWNER.hip$h"
done
say "lease taken: $(find "$LEASE" -maxdepth 1 -type f -printf '%f ')"

status=0
for arm in $ARMS; do
    # A* = W4A16, B* = W4A8. *d = deterministic MoE epilogue (fp32 scratch
    # accumulator): teacher-forced windows only, first 32 per domain, scored twice.
    case $arm in
        A*) knob=(VLLM_RDNA2_W4A8_SDOT4=0) ;;
        B*) knob=() ;;
        *) say "unknown arm $arm"; status=2; break ;;
    esac
    phases=${PHASES:-topk suite tfkld} topk_limit=0 repeat=16
    if [[ $arm == *d ]]; then
        knob+=(VLLM_RDNA2_MOE_FP32_ACCUM=1)
        phases=topk topk_limit=32 repeat=32
    fi
    ref=""
    [[ $arm != A ]] && ref=$W4Q/arms/A/bench.json
    out=$W4Q/arms/$arm
    if [[ -e $out/DONE ]]; then
        say "$arm: already DONE"
        continue
    fi
    ok=0
    for try in $(seq 1 "$MAX_TRIES"); do
        stash_attempt "$out"
        if [[ -e $MARKER ]]; then
            say "removing wedge marker: $(tr '\n' ' ' < "$MARKER")"
            rm -f "$MARKER"
        fi
        until enough_memory; do sleep 30; done
        say "arm $arm try $try start: load $(cut -d' ' -f1-3 /proc/loadavg)" \
            "knob=${knob[*]:-recipe}"
        (
            cd "$TREE" &&
                PHASES=$phases TOPK_LIMIT=$topk_limit REPEAT=$repeat \
                RECIPE=flashnext-mtp2 MODEL=$MODEL GPUS=$GPUS PORT=$PORT \
                TAG=w4q-$arm OUT=$out CACHE=$W4Q/cache VENV=$VENV CELLS=none \
                CONC_PROBE=0 LOAD_HOOK=$W4Q/scripts/hook.sh REF_BENCH=$ref \
                BOOT_LOCK=$HOME/w4a8_runs/boot.lock \
                bash tools/rdna/port_v031/serve_validate.sh \
                VLLM_RDNA2_W4A8_MOE_DEBUG=1 "${knob[@]}"
        )
        rc=$?
        grep -a "W4A8-MOE-DEBUG" "$out/serve.log" | sort | uniq -c > "$out/w4a8_debug.txt"
        say "arm $arm try $try rc=$rc:" \
            "$(grep -aE 'LOAD HOOK|PROBE|PPL|PREFIX' "$out/summary.txt" | tr '\n' ' ')"
        say "arm $arm W4A8 fast-path shape lines: $(wc -l < "$out/w4a8_debug.txt")"
        ar=$(grep -ac "rdna_ar: one-shot all-reduce active" "$out/serve.log")
        say "arm $arm rdna_ar active lines: $ar;" \
            "disabled: $(grep -ac 'rdna_ar: disabled' "$out/serve.log")"
        if grep -aq "rdna_ar: WEDGED" "$out/serve.log"; then
            say "arm $arm try $try: rdna_ar wedged" \
                "($(grep -a 'rdna_ar: WEDGED' "$out/serve.log" | head -1 | cut -c1-160))"
            continue
        fi
        if ((rc == 0)) && grep -q "LOAD HOOK rc=0" "$out/summary.txt" && ((ar > 0)) &&
            ! grep -aq "rdna_ar: disabled" "$out/serve.log"; then
            ok=1
            date -Is > "$out/DONE"
        fi
        break
    done
    if ((ok == 0)); then
        say "arm $arm failed"
        status=1
        break
    fi
done
if [[ ${RELEASE:-1} == 1 ]]; then
    rm -f "$LEASE/$OWNER".hip[2-5]
    say "holder files $OWNER.hip2-5 removed; request file left for the owner to clear"
fi
say "run_arms exit $status"
exit "$status"
