#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Baseline launcher matching the active V620 service configuration.
set -euo pipefail

source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
runtime=${V620_RUNTIME:-/home/george/v620-experiments/upstream-20260915/v620-vllm-testing}
model=${V620_MODEL:-/home/george/v620-vllm/models/intel-autoround}
port=${V620_PORT:-${V620_TEST_PORT:-8082}}
host=${V620_HOST:-127.0.0.1}
export PYTHONPATH=$source_dir
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_PLE_CPU_OFFLOAD=1 VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ROCM_MOE_PREFILL=0 VLLM_GDN_HIP_PREFILL=0
export VLLM_RDNA_FUSED_SE=1
export VLLM_TUNED_CONFIG_FOLDER=$source_dir/tuned-moe
if [[ ${V620_ENABLE_RESIDENT:-0} == 1 ]]; then
    export VLLM_RDNA_MOE_RESIDENT=1
fi
if [[ ${V620_ENABLE_SKINNY:-0} == 1 ]]; then
    export VLLM_RDNA_MOE_RESIDENT_SKINNY=1
fi
export VLLM_RDNA_DENSE_INT8=0 VLLM_RDNA_DENSE_INT8_ONLY=0 VLLM_RDNA_DENSE_GEMV=0
export VLLM_RDNA_AR=1 VLLM_RDNA_AR_MAX_KB=64 VLLM_RDNA_AR_BLOCKS=0 VLLM_RDNA_AR_PACE=0
export HSA_FORCE_FINE_GRAIN_PCIE=1 HSA_ENABLE_SDMA=0 OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false PYTHONFAULTHANDLER=1
export VLLM_CAUSAL_CONV1D_RDNA2_FWD=0 VLLM_CAUSAL_CONV1D_RDNA2_UPDATE=0
export VLLM_ENABLE_STARTUP_PLAN=0 VLLM_ROCM_USE_AITER=0 TORCH_BLAS_PREFER_HIPBLASLT=0
cache_dir=${V620_CACHE_DIR:-$source_dir/cache}
export VLLM_CACHE_ROOT=$cache_dir/vllm TRITON_CACHE_DIR=$cache_dir/triton
export TORCHINDUCTOR_CACHE_DIR=$cache_dir/inductor
export TORCH_EXTENSIONS_DIR=$cache_dir/extensions
export PYTORCH_TUNABLEOP_ENABLED=0 PYTORCH_TUNABLEOP_TUNING=0
export PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0
sdk=$runtime/.venv/lib/python3.12/site-packages/_rocm_sdk_core
export PATH="$runtime/.venv/bin:/opt/rocm/core-10.0/bin:/opt/rocm/core-10.0/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$sdk/lib:$sdk/lib/host-math/lib:/opt/rocm/core-10.0/lib"
source "$source_dir/tools/rdna2/tunableop_env.sh"
configure_v620_tunableop "$runtime/.venv/lib/python3.12/site-packages/_rocm_sdk_libraries/lib/librocblas.so.5" "$runtime/tunableop"
command=("$runtime/.venv/bin/python" -m vllm.entrypoints.openai.api_server
    --model "$model" --served-model-name active qwen3.8-flash-next
    --host "$host" --port "$port" --tensor-parallel-size 4
    --pipeline-parallel-size 1 --enable-expert-parallel --enable-ep-weight-filter
    --dtype float16 --max-model-len 262144 --block-size 1024 --max-num-seqs 4
    --max-num-batched-tokens 4096 --kv-cache-memory-bytes 4026531840
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[3,6,12]}'
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
    --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3
    --default-chat-template-kwargs '{"enable_thinking":false}'
    --limit-mm-per-prompt '{"image":255,"video":32}'
    --mm-processor-kwargs '{"max_pixels":602112}'
    --enable-prefix-caching --mamba-cache-mode align
    --kernel-config '{"moe_backend":"triton"}')
if [[ -n ${V620_KV_OFFLOAD_GB:-} ]]; then
    command+=(--kv-offloading-size "$V620_KV_OFFLOAD_GB")
fi
if [[ ${V620_SKIP_MM_PROFILING:-0} == 1 ]]; then
    command+=(--skip-mm-profiling)
fi
if [[ ${1:-} == --dry-run ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
if pgrep -u "$(id -u)" -f 'vllm.entrypoints|VLLM::EngineCore|VLLM::Worker' >/dev/null; then
    printf 'Existing vLLM process present; refusing overlap.\n' >&2
    exit 2
fi
"$runtime/.venv/bin/python" "$source_dir/tools/rdna2/check_v620_tuning.py" \
    --rows-template "$PYTORCH_TUNABLEOP_FILENAME" "${command[@]:1}"
"$runtime/.venv/bin/python" -c 'import vllm; print(vllm.__file__)'
cd "$source_dir"
exec "${command[@]}"
