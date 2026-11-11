#!/usr/bin/env bash
# RCCL all_reduce_perf placement/P2P-level matrix (par1-cs25 V620s).
# usage: rccl_suite.sh <outdir>; HIP 1-4 = slot A (root c0), 6-9 = slot B (root 00).
set -uo pipefail
out=$1
here=$(dirname "$0")
BASE=(NCCL_PROTO=Simple RCCL_P2P_NET_DISABLE=1 RCCL_P2P_BATCH_ENABLE=1 RCCL_MSCCL_ENABLE=0)
run() { # label hip env...
  local label=$1 hip=$2
  shift 2
  echo "=== $label hip=$hip $*"
  "$here/rccl_matrix.sh" "$out" "$label" "$hip" "${BASE[@]}" "$@"
}
run tp8_pxb 1,2,3,4,6,7,8,9 NCCL_P2P_LEVEL=pxb
run tp8_sys 1,2,3,4,6,7,8,9 NCCL_P2P_LEVEL=sys
run tp8_phb 1,2,3,4,6,7,8,9 NCCL_P2P_LEVEL=phb
run tp2x_pxb 1,6 NCCL_P2P_LEVEL=pxb
run tp2x_sys 1,6 NCCL_P2P_LEVEL=sys
run tp4A_pxb 1,2,3,4 NCCL_P2P_LEVEL=pxb
run tp4x_sys 1,2,6,7 NCCL_P2P_LEVEL=sys
run tp2A_pxb 1,2 NCCL_P2P_LEVEL=pxb
