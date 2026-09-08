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
export VLLM_USE_RDNA2_FA=1
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
export GPU_MAX_HW_QUEUES=2
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_LEVEL=pix
export RCCL_P2P_NET_DISABLE=1
export RCCL_P2P_BATCH_ENABLE=1
export NCCL_PROTO=Simple
export RCCL_MSCCL_ENABLE=0
# TRUE FULL is default-on. Custom AR is auto-disabled under this path.
unset VLLM_RDNA_AR
unset VLLM_ROCM_TRUE_FULL

cd /tmp
exec python -m vllm.entrypoints.cli.main serve "$MODEL" \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --max-num-seqs "${MAX_NUM_SEQS:-4}" \
  --dtype float16 \
  --gpu-memory-utilization "${GPU_MEM:-0.80}" \
  --block-size 16 \
  --enable-prefix-caching \
  --language-model-only \
  --skip-mm-profiling \
  --trust-remote-code \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","compile_ranges_endpoints":[],"inductor_compile_config":{"combo_kernels":false}}'
