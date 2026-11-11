#!/usr/bin/env bash
# Perf-only rerun of determinism arms whose KFD window was INVALID (P0's
# 16384x256 c1 cell, 2026-11-04 13:52). Runs after run_det_arms.sh exits
# (WAIT_PID), on HIP 2-5 held through Determinism-rerun.hip2..5 holder files
# that were created before run_det_arms.sh released its lease, so the lease
# never lapses in between. Removes those holder files at the end.
#   WAIT_PID=<run_det_arms pid> ARMS="P0r" setsid nohup bash run_perf_rerun.sh \
#       > run_perf_rerun.log 2>&1 < /dev/null &
set -uo pipefail
DET=$HOME/w4a8_runs/determinism
DET_HOOK=$DET/det_hook.sh
PROD=$HOME/vllm-w4a8q-4a3d
FIX=$HOME/vllm-det-d638
VENV=$HOME/Apps/vllm/venv-7.14.0_0.32.0
MODEL=$HOME/.cache/huggingface/hub/models--wtdcode--Qwen3.8-Flash-Next-AWQ-W4A16/snapshots/0939125b929543a783ce700c90e36dd1a575c00c
LEASE=$HOME/w4a8_runs/lanes/lease
GPUS=2,3,4,5
PORT=${PORT:-18470}
ARMS=${ARMS:-P0r}
say() { echo "[$(date '+%F %T')] $*"; }
# shellcheck disable=SC2329  # invoked by the EXIT trap
release() {
    rm -f "$LEASE"/Determinism-rerun.hip[2-5]
    say "rerun holder files removed"
}
trap release EXIT
trap 'exit 143' TERM INT

free_gib() {
    HIP_VISIBLE_DEVICES=$GPUS "$VENV/bin/python" -c '
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

for h in 2 3 4 5; do
    [[ -e $LEASE/Determinism-rerun.hip$h ]] || { say "missing holder hip$h"; exit 1; }
done
if [[ -n ${WAIT_PID:-} ]]; then
    say "waiting for run_det_arms.sh pid $WAIT_PID"
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 10; done
fi
export HIP_VISIBLE_DEVICES=$GPUS
status=0
for arm in $ARMS; do
    case $arm in
        P0r) tree=$PROD; knob=() ;;
        P1r) tree=$FIX; knob=(VLLM_RDNA2_MOE_DETERMINISTIC=1) ;;
        *) say "unknown arm $arm"; status=2; break ;;
    esac
    out=$DET/arms/$arm
    cache=$DET/cache-${arm%r}
    mkdir -p "$out"
    rm -f "$cache/vllm/rdna_ar_wedged"
    until enough_memory; do sleep 30; done
    say "arm $arm start: tree $(cat "$tree/.sha") knob=${knob[*]:-recipe}" \
        "load $(cut -d' ' -f1-3 /proc/loadavg)"
    (
        cd "$tree" &&
            ARM=$arm DET=$DET DET_PHASES=perf RECIPE=flashnext-mtp2 MODEL=$MODEL \
            GPUS=$GPUS PORT=$PORT TAG=det-$arm OUT=$out CACHE=$cache VENV=$VENV \
            CELLS=none CONC_PROBE=0 PPL_PROBE=0 PREFIX_PROBE=0 WARMUP=1 \
            LOAD_HOOK=$DET_HOOK BOOT_LOCK=$HOME/w4a8_runs/boot.lock \
            bash tools/rdna/port_v031/serve_validate.sh EAGER=0 "${knob[@]}"
    )
    rc=$?
    say "arm $arm rc=$rc: $(grep -aE 'READY|LOAD HOOK|PROBE' "$out/summary.txt" | tr '\n' ' ')"
    ((rc == 0)) || status=1
done
say "run_perf_rerun exit $status"
exit "$status"
