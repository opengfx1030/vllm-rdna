#!/usr/bin/env bash
# Determinism arms on HIP 2-5 (lease slot 2c, ~/w4a8_runs/lanes/lease/README.md),
# one boot per arm:
#   T1 production tree 4a3db777ae, eager + dtrace (CAS MoE epilogue)
#   P0 production tree, compiled (flashnext-mtp2 as served): repeat identity + perf
#   T2 fix tree d6384840c2, eager + dtrace, VLLM_RDNA2_MOE_DETERMINISTIC=1
#      (identity, then +-1 ulp MoE noise and a single-ulp perturbation)
#   P1 fix tree, compiled, VLLM_RDNA2_MOE_DETERMINISTIC=1: repeat identity + perf
# Waits until no other job's hip2-5.request exists, then writes its own request,
# waits for other holder files to go and >= 28 GiB free per card, holds
# Determinism.hip2..5 and removes holder + request files at the end.
#   setsid nohup bash run_det_arms.sh > run_det_arms.log 2>&1 < /dev/null &
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
OWNER=Determinism
ARMS=${ARMS:-T1 P0 T2 P1}
IDENT_WINDOWS=mix:24,mix:25,mix:26,mix:27,mix:28,mix:29,mix:30,mix:31,docs:0,docs:1,docs:2,docs:3,docs:4,docs:5,docs:6,docs:7
say() { echo "[$(date '+%F %T')] $*"; }

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
others_hold() {
    find "$LEASE" -maxdepth 1 -name '*.hip[2-5]' ! -name "$OWNER.*" | grep -q .
}
foreign_request() {
    [[ -e $LEASE/hip2-5.request ]] && ! grep -q "^$OWNER " "$LEASE/hip2-5.request"
}

[[ "$(cat "$PROD/.sha")" == 4a3db777aefa28e06036a355b20c4e0cd3edbcd9 ]] ||
    { say "$PROD is not 4a3db777ae"; exit 1; }
[[ "$(cat "$FIX/.sha")" == d6384840c2669df9fa9f14c269c24b1df98b5c2e ]] ||
    { say "$FIX is not d6384840c2"; exit 1; }
while foreign_request; do
    say "waiting: request $(cat "$LEASE/hip2-5.request")"
    sleep 60
done
( set -C; echo "$OWNER $(date -Is) run_det_arms.sh pid $$" > "$LEASE/hip2-5.request" ) 2>/dev/null ||
    grep -q "^$OWNER " "$LEASE/hip2-5.request" ||
    { say "lost the request race: $(cat "$LEASE/hip2-5.request")"; exit 1; }
say "request: $(cat "$LEASE/hip2-5.request")"
until ! others_hold && enough_memory; do
    say "waiting: holders $(find "$LEASE" -maxdepth 1 -name '*.hip*' -printf '%f ')"
    sleep 60
done
for h in 2 3 4 5; do
    echo "$OWNER $(date -Is) run_det_arms.sh pid $$" > "$LEASE/$OWNER.hip$h"
done
say "lease taken: $(find "$LEASE" -maxdepth 1 -type f -printf '%f ')"
# Every child (server, hook clients, KFD sampler) sees only the leased cards.
export HIP_VISIBLE_DEVICES=$GPUS

status=0
for arm in $ARMS; do
    out=$DET/arms/$arm
    if [[ -e $out/DONE ]]; then
        say "$arm: already DONE"
        continue
    fi
    knob=()
    unset DTRACE_DIR
    case $arm in
        T1) tree=$PROD; eager=1; phases="trace" ;;
        T2) tree=$FIX; eager=1; phases="trace noise elem"
            knob=(VLLM_RDNA2_MOE_DETERMINISTIC=1) ;;
        P0) tree=$PROD; eager=0; phases="ident perf" ;;
        P1) tree=$FIX; eager=0; phases="ident perf"
            knob=(VLLM_RDNA2_MOE_DETERMINISTIC=1) ;;
        *) say "unknown arm $arm"; status=2; break ;;
    esac
    mkdir -p "$out"
    cache=$DET/cache-$arm
    rm -f "$cache/vllm/rdna_ar_wedged"
    if ((eager)); then
        export DTRACE_DIR=$out/trace PYTHONPATH=$DET/dtrace DTRACE_TOKENS=0 DTRACE_MIN_TOKENS=256 \
            DTRACE_DUMP_RANKS=0
        mkdir -p "$DTRACE_DIR"
        rm -f "$DTRACE_DIR/dump_names.txt" "$DTRACE_DIR/perturb.txt"
    else
        unset PYTHONPATH DTRACE_TOKENS DTRACE_MIN_TOKENS DTRACE_DUMP_RANKS
    fi
    until enough_memory; do sleep 30; done
    say "arm $arm start: tree $(cat "$tree/.sha") eager=$eager knob=${knob[*]:-recipe}" \
        "load $(cut -d' ' -f1-3 /proc/loadavg)"
    (
        cd "$tree" &&
            ARM=$arm DET=$DET DET_PHASES=$phases IDENT_SEQ=$IDENT_WINDOWS \
            IDENT_REPEATS=3 RECIPE=flashnext-mtp2 MODEL=$MODEL GPUS=$GPUS \
            PORT=$PORT TAG=det-$arm OUT=$out CACHE=$cache VENV=$VENV CELLS=none \
            CONC_PROBE=0 PPL_PROBE=0 PREFIX_PROBE=0 WARMUP=$((1 - eager)) \
            LOAD_HOOK=$DET_HOOK BOOT_LOCK=$HOME/w4a8_runs/boot.lock \
            bash tools/rdna/port_v031/serve_validate.sh "EAGER=$eager" "${knob[@]}"
    )
    rc=$?
    say "arm $arm rc=$rc: $(grep -aE 'READY|LOAD HOOK|PROBE' "$out/summary.txt" | tr '\n' ' ')"
    say "arm $arm rdna_ar active lines: $(grep -ac 'rdna_ar: one-shot all-reduce active' "$out/serve.log");" \
        "wedged: $(grep -ac 'rdna_ar: WEDGED' "$out/serve.log")"
    if ((rc == 0)) && grep -q "LOAD HOOK rc=0" "$out/summary.txt"; then
        date -Is > "$out/DONE"
    else
        say "arm $arm failed"
        status=1
    fi
done
unset DTRACE_DIR PYTHONPATH
if [[ ${RELEASE:-1} == 1 ]]; then
    rm -f "$LEASE/$OWNER".hip[2-5]
    grep -q "^$OWNER " "$LEASE/hip2-5.request" && rm -f "$LEASE/hip2-5.request"
    say "holder files and request removed"
fi
say "run_det_arms exit $status"
exit "$status"
