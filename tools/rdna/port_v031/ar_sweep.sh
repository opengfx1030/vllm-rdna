#!/usr/bin/env bash
# All-reduce microbenchmark sweep over environment settings, one run each.
#
#   OUT=~/w4a8_runs/port-v031/u-arbench GPUS=2,3,4,5 \
#     bash tools/rdna/port_v031/ar_sweep.sh base "X=1" ll "NCCL_PROTO=LL" ...
#
# Each NAME "ENV=v ENV=v" pair runs benchmarks/kernels/benchmark_rdna_allreduce.py
# with that environment and writes $OUT/NAME.{json,log}. The BMC SEL tail and
# new PCIe AER/SERR/PERR dmesg lines are appended to $OUT/sweep.done after each
# run; the sweep stops at the first new PCI error.
set -uo pipefail
TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
VENV=${VENV:-$HOME/Apps/vllm/venv-7.14.0_0.31.0-u}
OUT=${OUT:-$HOME/w4a8_runs/port-v031/u-arbench}
GPUS=${GPUS:-2,3,4,5}
TOK=${TOK:-1,8,16,24,64,128,256,512,1024,2048}
HIDDEN=${HIDDEN:-2048,5120}
WORLD=$(awk -F, '{print NF}' <<< "$GPUS")
case ",$GPUS," in *,0,* | *,1,*) echo "refusing HIP 0/1"; exit 2 ;; esac
mkdir -p "$OUT"
SITE=$VENV/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$SITE/_rocm_sdk_libraries/lib:$SITE/_rocm_sdk_core/lib/host-math/lib:$SITE/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SITE/_rocm_sdk_core/lib/core/lib:$SITE/torch/lib
export HIP_VISIBLE_DEVICES=$GPUS VLLM_RDNA_AR=${VLLM_RDNA_AR:-1}
export HSA_FORCE_FINE_GRAIN_PCIE=1 HSA_ENABLE_SDMA=0 GPU_MAX_HW_QUEUES=2
cd "$OUT"
while (($# >= 2)); do
    name=$1 envs=$2
    shift 2
    # The BMC listing drops entries on flaky reads, so compare the newest
    # PCI SERR/PERR record rather than a count.
    sel0=$(sudo -n ipmitool sel list 2>/dev/null | grep -iE "PCI (SERR|PERR)" | tail -1 | cut -d'|' -f2-3)
    dm0=$(sudo -n dmesg 2>/dev/null | grep -ciE "AER|SERR|PERR")
    # shellcheck disable=SC2086
    env $envs timeout 900 "$VENV/bin/python" \
        "$TREE/benchmarks/kernels/benchmark_rdna_allreduce.py" --world "$WORLD" \
        --tokens "$TOK" --hidden "$HIDDEN" ${BENCH_ARGS:-} --out "$name.json" \
        > "$name.log" 2>&1
    rc=$?
    sel1=$(sudo -n ipmitool sel list 2>/dev/null | grep -iE "PCI (SERR|PERR)" | tail -1 | cut -d'|' -f2-3)
    dm1=$(sudo -n dmesg 2>/dev/null | grep -ciE "AER|SERR|PERR")
    echo "$name rc=$rc newest SEL PCI error [$sel0] -> [$sel1] dmesg_aer $dm0->$dm1" >> sweep.done
    if [[ -n $sel1 && $sel1 != "$sel0" ]] || ((dm1 > dm0)); then
        echo "new PCI error after $name, stopping" >> sweep.done
        break
    fi
done
echo "sweep done" >> sweep.done
