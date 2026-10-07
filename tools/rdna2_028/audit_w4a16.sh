#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# W4A16 regression-audit driver for the split-selection A/B. All artifacts land
# under $OUT/<tag>/ (persistent, no /tmp). Commands:
#
#   probe <tag>        run the op-level prefill timing/split probe (1 GPU)
#   swap-old           patch compute_split_k back to the pre-W4A8 search
#   swap-fixed         restore the repair (legacy-when-valid) implementation
#   build              incremental ninja rebuild of _rocm_C
#
# Typical sequence:
#   audit_w4a16.sh probe new_build
#   audit_w4a16.sh swap-old   && audit_w4a16.sh build && audit_w4a16.sh probe old_build
#   audit_w4a16.sh swap-fixed && audit_w4a16.sh build && audit_w4a16.sh probe fixed_build
set -uo pipefail

V=${V:-/home/chenco_adm/Apps/vllm/venv-7.14.0_0.28.0}
T=${T:-/home/chenco_adm/vllm-rdna-0.28.0}
OUT=${OUT:-/home/chenco_adm/w4a8_runs}
SRC=${SRC:-$T/csrc/rocm/q_gemm_rdna2_prefill.cu}
BLK=$T/tools/rdna2_028/audit_split

ROCM_SDK_LIB="$V/lib/python3.12/site-packages/_rocm_sdk_libraries/lib"
ROCM_SDK="$V/lib/python3.12/site-packages/_rocm_sdk_core/lib"
export LD_LIBRARY_PATH="$ROCM_SDK_LIB:$ROCM_SDK/host-math/lib:$ROCM_SDK/rocm_sysdeps/lib:$ROCM_SDK/core/lib:$V/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH=$T
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn GPU_MAX_HW_QUEUES=2
export HIP_VISIBLE_DEVICES=${GPUIDS:-0}
export VLLM_ROCM_USE_AITER=0 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE VLLM_RDNA_FORCE_FP16=1
export TORCH_BLAS_PREFER_HIPBLASLT=0 VLLM_BATCH_INVARIANT=0
# Rows live in the fork; never let TunableOp fall back to its default /tmp name.
# shellcheck source=tools/rdna2_028/tunableop_env.sh
source "$T/tools/rdna2_028/tunableop_env.sh"
configure_tunableop "$ROCM_SDK_LIB/librocblas.so.5" "$T/tunableop"
export VLLM_CACHE_ROOT=$T/cache/vllm
export TRITON_CACHE_DIR=$T/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$T/cache/inductor

cmd=${1:-}
tag=${2:-audit}
D=$OUT/$tag
mkdir -p "$D"

case "$cmd" in
  probe)
    VLLM_RDNA2_PREFILL_DEBUG=1 "$V/bin/python" \
      "$T/tools/rdna2_028/probe_prefill_split.py" \
      --out "$D/prefill_timing.csv" --variants awq,gptq --iters 30 \
      >"$D/splits.log" 2>&1
    rc=$?
    echo "[probe] rc=$rc csv=$D/prefill_timing.csv log=$D/splits.log"
    exit $rc
    ;;
  swap-old)
    "$V/bin/python" "$T/tools/rdna2_028/audit_split/audit_split_swap.py" \
      --file "$SRC" --block "$BLK/split_old.txt"
    ;;
  swap-fixed)
    "$V/bin/python" "$T/tools/rdna2_028/audit_split/audit_split_swap.py" \
      --file "$SRC" --block "$BLK/split_fixed.txt"
    ;;
  build)
    bash "$T/tools/rdna2_028/build_rocm_c_incr.sh" 2>&1 | tee "$D/build.log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" = "0" ]; then
      # ninja writes the linked module into the build dir; the editable install
      # imports it from vllm/, so deploy it explicitly.
      cp "$T/build/temp.linux-x86_64-cpython-312/_rocm_C.abi3.so" \
        "$T/vllm/_rocm_C.abi3.so"
      echo "[build] deployed $(ls -la --time-style=+%H:%M "$T/vllm/_rocm_C.abi3.so")"
    fi
    exit $rc
    ;;
  *)
    echo "usage: $0 {probe|swap-old|swap-fixed|build} [tag]" >&2
    exit 2
    ;;
esac
