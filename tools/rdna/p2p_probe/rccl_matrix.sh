#!/usr/bin/env bash
# RCCL all_reduce_perf over subsets of the par1-cs25 V620s (HIP ids, resolve
# them to PCI buses first). usage: rccl_matrix.sh <outdir> <label> <hip csv> [ENV=VAL ...]
# Production env is NCCL_P2P_LEVEL=pxb NCCL_PROTO=Simple RCCL_P2P_NET_DISABLE=1
# RCCL_P2P_BATCH_ENABLE=1 RCCL_MSCCL_ENABLE=0 (docs/rdna2/tuning.md).
set -euo pipefail
out=$1 label=$2 hip=$3
shift 3
BIN=${ALLREDUCE_PERF:-$HOME/rccl-tests-build/build/all_reduce_perf}
mkdir -p "$out"
n=$(awk -F, '{print NF}' <<<"$hip")
env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
    HIP_VISIBLE_DEVICES="$hip" NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV,P2P \
    "$@" "$BIN" -b 4K -e "${RCCL_MAX:-32M}" -f 2 -g "$n" -n "${RCCL_ITERS:-50}" -w 10 -c 1 \
    >"$out/$label.log" 2>&1
grep -E '^ *[0-9]+ +[0-9]+ +(float|half|bf16)' "$out/$label.log" || true
