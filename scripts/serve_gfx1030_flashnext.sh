#!/bin/bash
# Flash-Next production serve on gfx1030 (TP=4, Qwen3.8-Flash-Next-AWQ-W4A16).
# Validated 2026-09-18: FULL_AND_PIECEWISE + prefix caching + max_num_seqs 16.
#
#   PP 3331 tok/s agg, TG 72.70 tok/s, TTFT 39.3 s at 16k/1k c=8 (PIECEWISE).
#   Correctness: 18/18 sequential, 6/6 c=8, 8/8 16k shared-prefix.
#   FULL_AND_PIECEWISE executes as PIECEWISE on ROCm (rocm_full_executes_as_piecewise)
#   — verified identical and correct 2026-09-18; the historical "FULL_AND_PIECEWISE
#   corrupts at c=8" report traced to probe artifacts (reasoning-parser field +
#   reasoning-budget exhaustion), not the graphs.
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
# This host: HIP 4-7 are this session. HIP 0-3 belong to the other agent.
# rocm-smi GPU 0-3 are HIP 4-7; do not read those indexes as HIP indexes.
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-4,5,6,7}"
# In-flight cap 16 (validated 2026-09-18 with capture sizes up to 64:
# identical-8 8/8, 1k c=16 16/16, 16k c=16 16/16, zero garbage). The old cap-6
# corruption-threshold finding traced to probe artifacts. Soak before raising
# further; capture >64 OOMs at this memory layout (256 needs ~6.5 GiB).
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
# 7 GiB at 0.90 leaves the PLE prefill 80 MiB short on these 30 GiB cards.
KV_CACHE_MEMORY="${KV_CACHE_MEMORY:-6400000000}"
GPU_MEM="${GPU_MEM:-0.85}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
LOG="${LOG:-/tmp/flashnext_server.log}"

# PLE (n-gram sidecar) CPU offload.
export VLLM_PLE_CPU_OFFLOAD=1
export VLLM_PLE_QUANT_DIR="${VLLM_PLE_QUANT_DIR:-/home/chenco_adm/hfcache/hub/models--primitive-ai--Qwen3.8-Flash-Next-PLE-quant/snapshots/4f861b63f69e61bfc2e22130ec91ec67f03ec43e/ples_int4}"
export VLLM_PLE_OFFLOAD_READY_TIMEOUT=3600

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
# Stock custom all-reduce PCI-faults on this chassis. The gfx1030 one-shot
# path is the small-message collective. 64 KiB fails its boot self-test and
# silently falls back to RCCL; 20480 KiB is the size that stays active.
unset VLLM_FORCE_CUSTOM_ALL_REDUCE || true
export VLLM_RDNA_AR=1
export VLLM_RDNA_AR_MAX_KB=20480
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
export VLLM_BATCH_INVARIANT=0
export GPU_MAX_HW_QUEUES=2
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_LEVEL=pxb
export RCCL_P2P_NET_DISABLE=1
export RCCL_P2P_BATCH_ENABLE=1
export NCCL_PROTO=Simple
export RCCL_MSCCL_ENABLE=0
export HSA_FORCE_FINE_GRAIN_PCIE=1

source "$VENV/bin/activate"
ROCM_SDK_LIB="$VENV/lib/python3.12/site-packages/_rocm_sdk_libraries/lib"
ROCM_SDK="$VENV/lib/python3.12/site-packages/_rocm_sdk_core/lib"
export LD_LIBRARY_PATH="$ROCM_SDK_LIB:$ROCM_SDK/host-math/lib:$ROCM_SDK/rocm_sysdeps/lib:$ROCM_SDK/core/lib:$VENV/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Lookup-only rocBLAS rows, keyed by the library hash. A miss uses the
# default algorithm. Online search stays off unless PYTORCH_TUNABLEOP_TUNING=1,
# and then it is capped so a new shape cannot stall the request for tens of seconds.
_source_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# The venv's editable install is the older tree. This launcher must run
# the v0.30 sources next to the script.
export PYTHONPATH="$_source_dir${PYTHONPATH:+:$PYTHONPATH}"
# shellcheck disable=SC1091
source "$_source_dir/tools/rdna2/tunableop_env.sh"
if [ "${TUNABLEOP:-1}" = "1" ]; then
  configure_v620_tunableop "$ROCM_SDK_LIB/librocblas.so.5" "$_source_dir/tunableop"
else
  export PYTORCH_TUNABLEOP_ENABLED=0
  export PYTORCH_TUNABLEOP_TUNING=0
  export PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0
  unset PYTORCH_TUNABLEOP_FILENAME
fi
if [ "${PYTORCH_TUNABLEOP_TUNING:-0}" = "1" ]; then
  export PYTORCH_TUNABLEOP_ENABLED=1
  export PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS="${PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS:-30}"
fi
export CPATH="/opt/rocm/core-7.14/include:${CPATH:-}"
export LIBRARY_PATH="/opt/rocm/core-7.14/lib:${LIBRARY_PATH:-}"
export ROCM_HOME=/opt/rocm/core-7.14
export HIP_PATH=/opt/rocm/core-7.14
export HIP_VISIBLE_DEVICES

# Dry path: print the performance env and exit before any process is killed.
if [ "${1:-}" = "--print-env" ]; then
  printf 'PYTORCH_TUNABLEOP_ENABLED=%s\n' "${PYTORCH_TUNABLEOP_ENABLED:-}"
  printf 'PYTORCH_TUNABLEOP_TUNING=%s\n' "${PYTORCH_TUNABLEOP_TUNING:-}"
  printf 'PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=%s\n' "${PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED:-}"
  printf 'PYTORCH_TUNABLEOP_FILENAME=%s\n' "${PYTORCH_TUNABLEOP_FILENAME:-}"
  printf 'VLLM_RDNA_AR=%s\n' "${VLLM_RDNA_AR:-}"
  printf 'VLLM_RDNA_AR_MAX_KB=%s\n' "${VLLM_RDNA_AR_MAX_KB:-}"
  printf 'VLLM_FORCE_CUSTOM_ALL_REDUCE=%s\n' "${VLLM_FORCE_CUSTOM_ALL_REDUCE-unset}"
  printf 'NCCL_P2P_LEVEL=%s\n' "${NCCL_P2P_LEVEL:-}"
  printf 'NCCL_PROTO=%s\n' "${NCCL_PROTO:-}"
  printf 'HIP_VISIBLE_DEVICES=%s\n' "${HIP_VISIBLE_DEVICES:-}"
  exit 0
fi

# Stop only processes pinned to this launcher's HIP set. A global pkill
# would take down the tenant on the other four GPUs.
_stop_ours() {
  local sig="$1" p hip
  for p in $(ps -eo pid,cmd | awk '/vllm.entrypoints|VLLM::Worker|VLLM::EngineCore|PleOffloadWorker/ && !/awk/ {print $1}'); do
    hip=$(tr '\0' '\n' < /proc/"$p"/environ 2>/dev/null | awk -F= '/^HIP_VISIBLE_DEVICES=/ {print $2}')
    if [ "$hip" = "$HIP_VISIBLE_DEVICES" ]; then
      kill "$sig" "$p" 2>/dev/null || true
    fi
  done
}
_stop_ours -TERM
sleep 8
_stop_ours -KILL
sleep 3

cd /tmp
nohup setsid bash -c "python -m vllm.entrypoints.cli.main serve \"$MODEL\" \
  --served-model-name \"$SERVED_NAME\" \
  --port $PORT --host 0.0.0.0 --tensor-parallel-size $TP \
  ${MAX_MODEL_LEN:+--max-model-len $MAX_MODEL_LEN} --max-num-seqs $MAX_NUM_SEQS \
  --max-num-batched-tokens 2048 \
  --kv-cache-memory-bytes $KV_CACHE_MEMORY --gpu-memory-utilization $GPU_MEM \
  --dtype float16 --trust-remote-code --generation-config vllm --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --enable-expert-parallel \
  --limit-mm-per-prompt '{\"image\":1}' --mm-processor-kwargs '{\"max_pixels\":1605632}' \
  --distributed-timeout-seconds 1800 \
  ${BLOCK_SIZE:+--block-size $BLOCK_SIZE} \
  --compilation-config '{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"compile_ranges_endpoints\":[],\"cudagraph_capture_sizes\":[1,2,4,8,16,32,64]}' \
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