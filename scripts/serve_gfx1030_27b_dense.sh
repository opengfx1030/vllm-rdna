#!/usr/bin/env bash
# Qwen3.8-27B AWQ-INT4 (compressed-tensors W4A16, dense hybrid GDN) on gfx1030.
# Sibling of serve_gfx1030_flashnext_mtp.sh, tuned for the dense 27B: no
# --max-model-len (the model's own context is used), the full HIP stack
# (FA-RDNA2 attention + RDNA2 W4A16/W4A8 GEMMs), prefix caching, F&P graphs.
#
# Usage:
#   MTP=0 bash scripts/serve_gfx1030_27b_dense.sh            # plain decode
#   MTP=2 bash scripts/serve_gfx1030_27b_dense.sh            # MTP spec decode
# Env: MTP(0|2), W4A8(0|1), RDNA_AR(0|1), ATTN(fa|triton), TP, PORT, MAXBAT, KV, SEQS,
#      MAXLEN (unset = model default), CG_MODE, CG_SIZES, MODEL, VENV, VLLM_TREE.
set -euo pipefail

source_dir=${VLLM_TREE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
runtime=${VENV:-$HOME/Apps/vllm/venv-7.14.0_0.28.0}
model=${MODEL:-$HOME/.cache/huggingface/hub/models--cyankiwi--Qwen3.8-27B-AWQ-INT4/snapshots/63768c10df38c0395e12ef49edac1bd539eaeeea}
port=${PORT:-18210}
tp=${TP:-4}
attn=${ATTN:-fa}
mtp=${MTP:-0}
w4a8=${W4A8:-0}
rdna_ar=${RDNA_AR:-1}

export PYTHONPATH=$source_dir
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HIP_VISIBLE_DEVICES=${GPUIDS:-0,1,2,3}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export GPU_MAX_HW_QUEUES=2

ROCM_SDK_LIB="$runtime/lib/python3.12/site-packages/_rocm_sdk_libraries/lib"
ROCM_SDK="$runtime/lib/python3.12/site-packages/_rocm_sdk_core/lib"
export LD_LIBRARY_PATH="$ROCM_SDK_LIB:$ROCM_SDK/host-math/lib:$ROCM_SDK/rocm_sysdeps/lib:$ROCM_SDK/core/lib:$runtime/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_RDNA_FUSED_SE=1
export VLLM_RDNA_DENSE_INT8=0 VLLM_RDNA_DENSE_INT8_ONLY=0 VLLM_RDNA_DENSE_GEMV=0
# W4A8 (int4 x int8 sdot4) opt-in dense prefill path.
export VLLM_RDNA2_W4A8_SDOT4=$w4a8
# One-shot up to 64 KiB, RCCL above; the stock custom-AR force flag stays off
# (it PCI-SERRs this chassis, AGENTS.md 2026-09-24).
export VLLM_FORCE_CUSTOM_ALL_REDUCE=0
export VLLM_RDNA_AR=$rdna_ar VLLM_RDNA_AR_MAX_KB=${VLLM_RDNA_AR_MAX_KB:-64} VLLM_RDNA_AR_ONESHOT_KB=${VLLM_RDNA_AR_ONESHOT_KB:-64} VLLM_RDNA_AR_BLOCKS=0 VLLM_RDNA_AR_PACE=0
export HSA_FORCE_FINE_GRAIN_PCIE=1 HSA_ENABLE_SDMA=0 OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false PYTHONFAULTHANDLER=1
export VLLM_CAUSAL_CONV1D_RDNA2_FWD=0 VLLM_CAUSAL_CONV1D_RDNA2_UPDATE=0
export VLLM_ENABLE_STARTUP_PLAN=0 VLLM_ROCM_USE_AITER=0 TORCH_BLAS_PREFER_HIPBLASLT=0
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$source_dir/cache/vllm}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$source_dir/cache/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$source_dir/cache/inductor}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$source_dir/cache/extensions}
export VLLM_TUNED_CONFIG_FOLDER=$source_dir/tuned-moe

source "$source_dir/tools/rdna2_028/tunableop_env.sh"
configure_mtp_tunableop "$ROCM_SDK_LIB/librocblas.so.5" "$source_dir/tunableop"

if [ "$attn" = "fa" ]; then
  export VLLM_USE_RDNA2_FA=1
  attention_backend=RDNA_ATTN
  export VLLM_FA_RDNA2_GQA_DECODE="${VLLM_FA_RDNA2_GQA_DECODE:-1}"
  unset FLASH_ATTENTION_TRITON_AMD_ENABLE || true
else
  export VLLM_USE_RDNA2_FA=0
  attention_backend=TRITON_ATTN
  export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
fi

if [ "$mtp" = "0" ]; then
  spec_args=()
  capture_sizes=${CG_SIZES:-'[1,2,4,8]'}
else
  spec_args=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$mtp,\"use_local_argmax_reduction\":true}")
  decode_width=$((mtp + 1))
  sizes=""
  for mult in 1 2 4 8; do
    sizes="$sizes$((decode_width * mult)),"
  done
  capture_sizes=${CG_SIZES:-"[${sizes%,}]"}
fi

cg_mode=${CG_MODE:-FULL_AND_PIECEWISE}
maxlen_arg=()
[ -n "${MAXLEN:-}" ] && maxlen_arg=(--max-model-len "$MAXLEN")

# EAGER=1 skips torch.compile + cudagraphs. The W4A8 fast path lives in the
# eager model forward, so shapes/acceptance are unaffected; this only trades
# throughput for a boot that cannot be killed by a mid-compile chassis reset.
compile_args=(--compilation-config "{\"cudagraph_mode\":\"$cg_mode\",\"cudagraph_capture_sizes\":$capture_sizes,\"compile_ranges_endpoints\":[]}")
if [ "${EAGER:-0}" = "1" ]; then
  compile_args=(--enforce-eager)
fi

# Kill leftovers scoped to THIS tree's cache root so a co-tenant is never touched.
for _sig in TERM KILL; do
  for _p in $(pgrep -f "entrypoints.cli.main serve|entrypoints.openai.api_server|VLLM::Worker|VLLM::EngineCore|PleOffloadWorker" 2>/dev/null); do
    [ -r "/proc/$_p/environ" ] || continue
    tr '\0' '\n' < "/proc/$_p/environ" 2>/dev/null | grep -q "^VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT}$" && kill -"$_sig" "$_p" 2>/dev/null
  done
  [ "$_sig" = "TERM" ] && sleep 8
done
sleep 3
exec "$runtime/bin/python" -m vllm.entrypoints.cli.main serve \
  --model "$model" --served-model-name q27d \
  --host 127.0.0.1 --port "$port" \
  --attention-backend "$attention_backend" \
  --tensor-parallel-size "$tp" \
  --dtype float16 --block-size 1024 \
  --max-num-seqs "${SEQS:-8}" --max-num-batched-tokens "${MAXBAT:-2048}" \
  --kv-cache-memory-bytes "${KV:-8000000000}" \
  --gpu-memory-utilization "${GMEM:-0.9}" \
  "${compile_args[@]}" \
  "${maxlen_arg[@]}" \
  "${spec_args[@]}" \
  --enable-prefix-caching --mamba-cache-mode align \
  --language-model-only \
  --limit-mm-per-prompt '{"image":1}' --mm-processor-kwargs '{"max_pixels":1605632}'
