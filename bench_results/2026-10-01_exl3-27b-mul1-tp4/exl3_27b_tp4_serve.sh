#!/usr/bin/env bash
# EXL3 27B mul1 (Qwen3.8-27B-exl3-3.00bpw) TP=4, full HIP stack.
# Rung 1: EAGER=1 (--enforce-eager); Rung 2: EAGER=0 (FULL_AND_PIECEWISE).
set -uo pipefail
V=/home/chenco_adm/Apps/vllm/venv-7.14.0_0.28.0
source "$V/bin/activate"
ROCM_SDK_LIB="$V/lib/python3.12/site-packages/_rocm_sdk_libraries/lib"
ROCM_SDK="$V/lib/python3.12/site-packages/_rocm_sdk_core/lib"
export LD_LIBRARY_PATH="$ROCM_SDK_LIB:$ROCM_SDK/host-math/lib:$ROCM_SDK/rocm_sysdeps/lib:$ROCM_SDK/core/lib:$V/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

export PYTHONPATH=/home/chenco_adm/vllm-rdna-0.28.0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-4,5,6,7}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export GPU_MAX_HW_QUEUES=2

# Mandatory HIP stack
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_AOT_COMPILE=0 VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_ROCM_USE_AITER=0 VLLM_ROCM_USE_AITER_MOE=0
export VLLM_RDNA_FORCE_FP16=1
export TORCH_BLAS_PREFER_HIPBLASLT=0
export VLLM_BATCH_INVARIANT=0
# Attention FA-RDNA2 (never Triton)
export VLLM_USE_RDNA2_FA=1
unset FLASH_ATTENTION_TRITON_AMD_ENABLE || true
# EXL3
export VLLM_EXL3_DEBUG=1
# TP=4 allreduce: RCCL PXB + Simple (avoid LL/LL128 deadlock, MSCCL, custom-AR PCI-SERR)
export NCCL_P2P_LEVEL=pxb RCCL_P2P_NET_DISABLE=1 RCCL_P2P_BATCH_ENABLE=1 NCCL_PROTO=Simple RCCL_MSCCL_ENABLE=0
export VLLM_FORCE_CUSTOM_ALL_REDUCE=0
# Caches inside the source tree (never /tmp)
export VLLM_CACHE_ROOT=/home/chenco_adm/vllm-rdna-0.28.0/cache/vllm
export TRITON_CACHE_DIR=/home/chenco_adm/vllm-rdna-0.28.0/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/home/chenco_adm/vllm-rdna-0.28.0/cache/inductor
export TORCH_EXTENSIONS_DIR=/home/chenco_adm/vllm-rdna-0.28.0/cache/extensions
export ROCM_HOME=/opt/rocm/core-7.14 ROCM_PATH=/opt/rocm/core-7.14 HIP_PATH=/opt/rocm/core-7.14 HIP_ROOT_DIR=/opt/rocm/core-7.14
export PATH=/opt/rocm/core-7.14/bin:$PATH
# TunableOp: shared fork rows (lookup-only), never /tmp
source /home/chenco_adm/vllm-rdna-0.28.0/tools/rdna2_028/tunableop_env.sh
configure_tunableop "$ROCM_SDK_LIB/librocblas.so.5" /home/chenco_adm/vllm-rdna-0.28.0/tunableop

EAGER=${EAGER:-1}
CG_MODE=${CG_MODE:-FULL_AND_PIECEWISE}
PORT=${PORT:-18105}
MAXLEN=${MAXLEN:-20480}
MAXBAT=${MAXBAT:-2048}
KVCAP=${KVCAP:-6000000000}

if [ "$EAGER" = "1" ]; then
  CG_ARGS=(--enforce-eager)
else
  CG_ARGS=(--compilation-config "{\"cudagraph_mode\":\"$CG_MODE\",\"compile_ranges_endpoints\":[]}")
fi

cd /home/chenco_adm
exec "$V/bin/python" -m vllm.entrypoints.openai.api_server \
  --model /home/chenco_adm/models/Qwen3.8-27B-exl3-3.00bpw \
  --served-model-name exl3-27b-mul1 \
  --host 127.0.0.1 --port "$PORT" \
  --attention-backend RDNA_ATTN \
  --tensor-parallel-size 4 \
  --dtype float16 --max-model-len "$MAXLEN" \
  --max-num-seqs 8 --max-num-batched-tokens "$MAXBAT" \
  --gpu-memory-utilization 0.90 --kv-cache-memory-bytes "$KVCAP" \
  --language-model-only --skip-mm-profiling --trust-remote-code \
  --enable-prefix-caching --mamba-cache-mode align \
  "${CG_ARGS[@]}"
