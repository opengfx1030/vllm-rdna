#!/bin/bash
# Flash-Next production serve on gfx1030 (TP=4, Qwen3.8-Flash-Next-AWQ-W4A16).
# Validated 2026-09-17: PIECEWISE cudagraphs + prefix caching + max_num_seqs 6.
#
#   PIECEWISE: PP 3331 tok/s agg, TG 72.70 tok/s, TTFT 39.3 s at 16k/1k c=8.
#   Correctness: 18/18 sequential, 6/6 c=8, 8/8 16k shared-prefix.
#   FULL_AND_PIECEWISE has an intermittent single-request '!!!!' corruption at
#   c=8 on this model (do not use until the FULL dispatch for Qwen4Exp is fixed).
#
# Requires commit 388a61b6f (the GDN sanitizer fix) for fresh-server long-prompt
# correctness.
#
# Vision is ON by default with the validated pixel cap (2026-09-17). Without
# the cap the mm-profiling dummy image (~24.8M px) makes the vision encoder's
# SDPA math backend materialize a 64 GiB LxL fp32 score matrix and startup
# OOMs on 30 GiB GPUs. NOTE: --limit-mm-per-prompt alone is NOT sufficient
# (the image count was already 1; the size is the driver).
# max_pixels=1605632 keeps images up to ~1424x1424 full-resolution.
# No --max-model-len by default: the checkpoint's native 262144-token context
# is used (the pinned 5 GiB KV pool holds ~313k tokens). Set MAX_MODEL_LEN to
# opt into a cap.
#
# Usage: MODEL=/path/to/flash-next bash scripts/serve_gfx1030_flashnext.sh
set -u
VENV="${VENV:-/home/chenco_adm/Apps/vllm/venv-7.14.0}"
MODEL="${MODEL:-/home/chenco_adm/hfcache/hub/models--wtdcode--Qwen3.8-Flash-Next-AWQ-W4A16/snapshots/0939125b929543a783ce700c90e36dd1a575c00c}"
PORT="${PORT:-18094}"
TP="${TP:-4}"
SERVED_NAME="${SERVED_NAME:-flash-next}"
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3}"
# In-flight cap 6: the Flash-Next corruption threshold is below 8; clients may
# still send 8/10/16 concurrent requests (they queue).
MAX_NUM_SEQS="${MAX_NUM_SEQS:-6}"
KV_CACHE_MEMORY="${KV_CACHE_MEMORY:-7000000000}"
GPU_MEM="${GPU_MEM:-0.90}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
LOG="${LOG:-/tmp/flashnext_server.log}"

# PLE (n-gram sidecar) CPU offload.
export VLLM_PLE_CPU_OFFLOAD=1
export VLLM_PLE_QUANT_DIR="${VLLM_PLE_QUANT_DIR:-/home/chenco_adm/hfcache/hub/models--primitive-ai--Qwen3.8-Flash-Next-PLE-quant/snapshots/4f861b63f69e61bfc2e22130ec91ec67f03ec43e/ples_int4}"
export VLLM_PLE_OFFLOAD_READY_TIMEOUT=3600

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export VLLM_FORCE_CUSTOM_ALL_REDUCE=1
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_USE_AOT_COMPILE=0
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_RDNA_FUSED_HC=0
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
export HSA_FORCE_FINE_GRAIN_PCIE=1

source "$VENV/bin/activate"
ROCM_SDK_LIB="$VENV/lib/python3.12/site-packages/_rocm_sdk_libraries/lib"
ROCM_SDK="$VENV/lib/python3.12/site-packages/_rocm_sdk_core/lib"
export LD_LIBRARY_PATH="$ROCM_SDK_LIB:$ROCM_SDK/host-math/lib:$ROCM_SDK/rocm_sysdeps/lib:$ROCM_SDK/core/lib:$VENV/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CPATH="/opt/rocm/core-7.14/include:${CPATH:-}"
export LIBRARY_PATH="/opt/rocm/core-7.14/lib:${LIBRARY_PATH:-}"
export ROCM_HOME=/opt/rocm/core-7.14
export HIP_PATH=/opt/rocm/core-7.14
export HIP_VISIBLE_DEVICES

# Kill leftovers (workers + EngineCore + the PLE offload sidecar, not just the
# API server) so the next launch does not fail on GPU memory.
pkill -f "entrypoints.cli.main serve" 2>/dev/null || true
pkill -f "VLLM::Worker" 2>/dev/null || true
pkill -f "VLLM::EngineCore" 2>/dev/null || true
pkill -f "PleOffloadWorker" 2>/dev/null || true
sleep 8
pkill -9 -f "entrypoints.cli.main serve" 2>/dev/null || true
pkill -9 -f "VLLM::Worker" 2>/dev/null || true
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
pkill -9 -f "PleOffloadWorker" 2>/dev/null || true
sleep 3

cd /tmp
nohup setsid bash -c "python -m vllm.entrypoints.cli.main serve \"$MODEL\" \
  --served-model-name \"$SERVED_NAME\" \
  --port $PORT --host 0.0.0.0 --tensor-parallel-size $TP \
  ${MAX_MODEL_LEN:+--max-model-len $MAX_MODEL_LEN} --max-num-seqs $MAX_NUM_SEQS \
  --max-num-batched-tokens 2048 \
  --kv-cache-memory-bytes $KV_CACHE_MEMORY --gpu-memory-utilization $GPU_MEM \
  --dtype float16 --trust-remote-code --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --enable-expert-parallel \
  --limit-mm-per-prompt '{\"image\":1}' --mm-processor-kwargs '{\"max_pixels\":1605632}' \
  --distributed-timeout-seconds 1800 \
  ${BLOCK_SIZE:+--block-size $BLOCK_SIZE} \
  --compilation-config '{\"cudagraph_mode\":\"PIECEWISE\",\"compile_ranges_endpoints\":[]}' \
  ${EXTRA_ARGS:-}" > "$LOG" 2>&1 < /dev/null &
disown
echo "launched flash-next server pid $! log=$LOG (TP=$TP, PIECEWISE, max_num_seqs=$MAX_NUM_SEQS)"

for i in $(seq 1 40); do
  sleep 15
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 \
     && grep -q "init engine" "$LOG" 2>/dev/null; then
    echo "READY after ~$((i * 15))s"; exit 0
  fi
  if ! pgrep -f "entrypoints.cli.main serve" >/dev/null; then
    echo "SERVER DIED"; tail -15 "$LOG"; exit 1
  fi
done
echo "TIMEOUT"; tail -15 "$LOG"; exit 1