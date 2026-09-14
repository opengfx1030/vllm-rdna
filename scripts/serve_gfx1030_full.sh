#!/bin/bash
# Production serve: TRUE FULL HIP graphs (FA-RDNA2 + W4A16 + HIP KV + HIP GDN).
# Greedy PASS 3/3 on Qwen3.8-27B-AWQ-INT4, TP=2, 2026-09-09 (FPP19).
# Usage: MODEL=/path/to/model PORT=18094 HIP_VISIBLE_DEVICES=0,1 ./scripts/serve_gfx1030_full.sh
set -euo pipefail
PORT="${PORT:-18094}"
TP="${TP:-2}"
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1}"
VENV="${VENV:-/home/chenco_adm/Apps/vllm/venv-7.14.0}"
MODEL="${MODEL:?set MODEL to the checkpoint path}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-200000}"
if [ "$MAX_MODEL_LEN" -lt 32768 ]; then
  echo "MAX_MODEL_LEN=$MAX_MODEL_LEN is below the 32768 floor; use 200000 in production." >&2
  exit 1
fi

source "$VENV/bin/activate"
ROCM_SDK_LIB="$VENV/lib/python3.12/site-packages/_rocm_sdk_libraries/lib"
ROCM_SDK="$VENV/lib/python3.12/site-packages/_rocm_sdk_core/lib"
export LD_LIBRARY_PATH="$ROCM_SDK_LIB:$ROCM_SDK/host-math/lib:$ROCM_SDK/rocm_sysdeps/lib:$ROCM_SDK/core/lib:$VENV/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CPATH="/opt/rocm/core-7.14/include:${CPATH:-}"
export LIBRARY_PATH="/opt/rocm/core-7.14/lib:${LIBRARY_PATH:-}"
export ROCM_HOME=/opt/rocm/core-7.14
export HIP_PATH=/opt/rocm/core-7.14
export HIP_VISIBLE_DEVICES

export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_RDNA2_FA="${VLLM_USE_RDNA2_FA:-1}"
# Isolation only: set to 1 to skip mixed decode+prefill steps.
# Production keeps mixed decode (prefix cache + align CoW/zeroing).
export VLLM_ROCM_NO_MIXED_BATCH="${VLLM_ROCM_NO_MIXED_BATCH:-0}"
# Do not hash the live last 784-token hybrid GDN+FA page while decode
# is still appending to it. Prefix cache still hashes completed pages.
export VLLM_ROCM_SKIP_LIVE_TAIL_HASH="${VLLM_ROCM_SKIP_LIVE_TAIL_HASH:-1}"
export VLLM_USE_AOT_COMPILE=0
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_ROCM_USE_AITER=0
export VLLM_ROCM_USE_AITER_MOE=0
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export VLLM_RDNA_FORCE_FP16=1
export TORCH_BLAS_PREFER_HIPBLASLT=0
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0
export PYTORCH_TUNABLEOP_FILENAME="${PYTORCH_TUNABLEOP_FILENAME:-$HOME/.cache/tunableop/tunableop_results.csv}"
export VLLM_BATCH_INVARIANT=0
# Mixed 16k skip_compiled hits reserved-unallocated holes next to FULL
# keepalives. expandable_segments:True is required for that hole (serve26
# 1k c=8 8/8). False + a 128 MiB persist floor regressed 1k c=8 to 3/8.
if [ "${VLLM_PLE_CPU_OFFLOAD:-0}" = "1" ]; then
  # PLE offload exports CUDA tensors over IPC; VMM-backed (expandable segment)
  # memory cannot be shared that way on ROCm.
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
else
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
fi
export GPU_MAX_HW_QUEUES=2
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-pix}"
export RCCL_P2P_NET_DISABLE=1
export RCCL_P2P_BATCH_ENABLE=1
export NCCL_PROTO=Simple
export RCCL_MSCCL_ENABLE=0
export HSA_FORCE_FINE_GRAIN_PCIE="${HSA_FORCE_FINE_GRAIN_PCIE:-1}"
# Breakable cudagraphs (2026-09-12): the GDN + FA-RDNA2 attention run eager
# (live data) while the rest of the model executes the FULL_AND_PIECEWISE
# graphs. This is the only TP=4 config where the RDNA2 W4A16 + FA-RDNA2 HIP
# path is correct under cudagraphs (verified: 16k/1k c=8 = 31.3 tok/s,
# coherent; the non-breakable TP=4 replay NaNs at the GDN).
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
# Breakable cudagraphs cannot replay the stock custom all-reduce
# (custom_all_reduce_hip.cuh:167 'invalid argument' at TP=4). Use PYNCCL.
export VLLM_FORCE_CUSTOM_ALL_REDUCE=0
# Opt-in: measured parity at TP=2 and -18% at TP=4 vs the custom allreduce,
# so the one-shot path is not the default (2026-09-12).
export VLLM_RDNA_AR=${VLLM_RDNA_AR:-0}
export VLLM_RDNA_AR_MAX_KB="${VLLM_RDNA_AR_MAX_KB:-20480}"
# GQA multi-head prefill attention (validated -15.9% cold 16k,
# -5.8% 16k/1k c=8). Set off to revert to varlen/splitk.
export VLLM_FA_RDNA2_GQA_MODE="${VLLM_FA_RDNA2_GQA_MODE:-subgroup}"
unset VLLM_ROCM_TRUE_FULL

# Hybrid GDN page is 24.50 MiB (784-token block). One 200k request
# needs 6.27 GiB, so the 200k default pin is 7e9. That left 0 B free
# after FULL keepalives; mixed FA/GDN scratch OOMed or recycled graph
# pages. For 16k/1k benches set MAX_MODEL_LEN=32768 and
# KV_CACHE_MEMORY=6000000000 (233 blocks; 16k c=8 = 232).
KV_CACHE_MEMORY="${KV_CACHE_MEMORY:-7000000000}"

# FULL decode graphs at 1/2/4/8 (and piecewise 16). Prefill-chunk graphs
# at 256/512/1024/2048 are opt-in — capturing 2048 on 32GB TP=2 OOMs
# during warmup. Override with COMPILATION_CONFIG if you have headroom:
#   max_cudagraph_capture_size=2048,
#   cudagraph_capture_sizes=[1,2,4,8,16,256,512,1024,2048]
COMPILATION_CONFIG="${COMPILATION_CONFIG:-{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"compile_ranges_endpoints\":[],\"max_cudagraph_capture_size\":16,\"cudagraph_capture_sizes\":[1,2,4,8,16],\"inductor_compile_config\":{\"combo_kernels\":false}}}"

if [ "${ENABLE_PREFIX_CACHING:-1}" = "0" ]; then
  PREFIX_CACHE_FLAG=""
else
  PREFIX_CACHE_FLAG="--enable-prefix-caching"
fi

cd /tmp
# A rank JIT-compiling Triton during the V2 warmup blocks the others in the
# logits allgather past PyTorch's 600s NCCL timeout; give it room.
DIST_TIMEOUT="${DIST_TIMEOUT:-1800}"
# EXTRA_ARGS e.g. --enforce-eager for isolation cells.
exec python -m vllm.entrypoints.cli.main serve "$MODEL" \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "${MAX_NUM_SEQS:-16}" \
  --distributed-timeout-seconds "$DIST_TIMEOUT" \
  --dtype float16 \
  --gpu-memory-utilization "${GPU_MEM:-0.90}" \
  --kv-cache-memory-bytes "$KV_CACHE_MEMORY" \
  --block-size 16 \
  ${PREFIX_CACHE_FLAG} \
  --language-model-only \
  --skip-mm-profiling \
  --trust-remote-code \
  --compilation-config "$COMPILATION_CONFIG" \
  ${EXTRA_ARGS:-}
