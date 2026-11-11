#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# All-reduce microbenchmark in an exact-SHA tree at world 4 or 8 (vLLM's own TP group:
# pynccl (RCCL) vs rdna_ar one-shot vs the production dispatch), with RCCL debug
# capture.
#
#   OWNER=<holder> SHA=<40 hex> SLOT=AB LABEL=ar8-base bash ar.sh [ENV=VAL ...]
#
# SLOT: A | B | AB (default AB; verified before the run); HIPS overrides the device
# list (a subset of the verified slots, e.g. 1,2,6,7 = two GPUs per root complex);
# LABEL names $RUN_DIR/ar/$LABEL.{json,log,nccl*.log}. TOKENS (default decode + prefill
# sizes at hidden 2560), HIDDEN, EXTRA_ARGS (e.g. --rs-ag). Needs the GPUs idle: take
# the lease and the quiet protocol before running (the driver of a measured window
# does that; this script only checks free memory and the quiet files).
set -uo pipefail
# shellcheck source=slots.sh
source "$(dirname "${BASH_SOURCE[0]}")/slots.sh"
OWNER=${OWNER:?OWNER=<lease holder name>}
SHA=${SHA:?SHA=<40 hex commit of the exact-SHA tree>}
TREE_T=${TREE_T:-$HOME/vllm-tp8pp2-${SHA:0:10}}
SLOT=${SLOT:-AB}
LABEL=${LABEL:?LABEL}
TOKENS=${TOKENS:-1,3,6,12,24,48,96,256,512,1024,2048,4096}
HIDDEN=${HIDDEN:-2560}
OUTD=$R/ar
mkdir -p "$OUTD"
GPUS=${HIPS:-$(slot_hip "$SLOT")} # HIPS=1,2,6,7 picks a subset of the verified slot(s)
WORLD=$(echo "$GPUS" | tr ',' '\n' | wc -l)
[[ ${SHA_CHECK:-1} == 0 || $(cat "$TREE_T/.sha") == "$SHA" ]] || {
    echo "tree sha mismatch"
    exit 1
}
verify_slot "$SLOT" || exit 1
fresh=$(find "$QUIET" -type f -mmin -6 ! -name "$OWNER" 2> /dev/null | tr '\n' ' ')
[[ -z $fresh ]] || {
    echo "quiet window active: $fresh"
    exit 75
}
export HIP_VISIBLE_DEVICES=$GPUS
export VLLM_CACHE_ROOT=$OUTD/cache-$LABEL
rm -rf "$VLLM_CACHE_ROOT"
export PYTHONPATH=$TREE_T
export VLLM_RDNA_AR=1 VLLM_RDNA_AR_MAX_KB=256 VLLM_RDNA_AR_ONESHOT_KB=256 VLLM_RDNA_AR_BLOCKS=0 VLLM_RDNA_AR_PACE=0
export HSA_FORCE_FINE_GRAIN_PCIE=1 HSA_ENABLE_SDMA=0 GPU_MAX_HW_QUEUES=4 OMP_NUM_THREADS=4
export ROCPROFILER_REGISTER_ENABLED=0 VLLM_ROCM_USE_AITER=0 PYTHONFAULTHANDLER=1
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH,TUNING NCCL_DEBUG_FILE=$OUTD/$LABEL.nccl.%h.%p.log
for kv in "$@"; do export "${kv?}"; done
dm0=$(sudo -n dmesg | tee "$OUTD/$LABEL.dmesg.before" | wc -l)
echo "[$(date +%T)] $LABEL world $WORLD HIP $GPUS tokens $TOKENS hidden $HIDDEN env: $*"
read -r -a extra_args <<< "${EXTRA_ARGS:-}"
cd "$TREE_T" && timeout 1500 "$VENV/bin/python" benchmarks/kernels/benchmark_rdna_allreduce.py \
    --world "$WORLD" --hidden "$HIDDEN" --tokens "$TOKENS" "${extra_args[@]}" \
    --out "$OUTD/$LABEL.json" > "$OUTD/$LABEL.log" 2>&1
rc=$?
sudo -n dmesg | tail -n +$((dm0 + 1)) > "$OUTD/$LABEL.dmesg.new" 2> /dev/null
echo "[$(date +%T)] rc=$rc new dmesg lines: $(wc -l < "$OUTD/$LABEL.dmesg.new"); AER/GHES/amdgpu-fault: $(grep -ciE 'AER:|GHES|Hardware Error|amdgpu.*(fault|timeout|reset)' "$OUTD/$LABEL.dmesg.new")"
exit $rc
