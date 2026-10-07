#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# V620 PIECEWISE launcher: compiled piecewise CUDA graphs on 4x V620 / gfx1030.
#
# Identical serving parameters to tools/rdna2/serve_v620_baseline.sh; the only
# difference is the compilation policy:
#   --compilation-config {"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE",...}
# On ROCm the FULL decode dispatches replay the captured piecewise graphs
# (see rocm_full_executes_as_piecewise), so decode and prefill both run graph
# replays. See docs/rdna2/V620-PIECEWISE-CUDAGRAPHS.md for the measured
# trade-off: prefill is unchanged, decode is ~2x slower than the mode-0
# FULL_DECODE_ONLY baseline.
#
# Every value can be overridden by exporting it first:
#   V620_VENV, V620_MODEL, V620_PLE_INT4, V620_PORT, V620_HOST,
#   V620_COMPILE_MODE (3), V620_CUDAGRAPH_MODE (FULL_AND_PIECEWISE),
#   V620_CG_SIZES ([3,6,12]), V620_ROCM_LIB (/opt/rocm/core-10.0/lib),
#   V620_TUNABLEOP (1), V620_TUNABLEOP_ROOT, V620_ROCBLAS_LIBRARY,
#   V620_ENABLE_RESIDENT (1), V620_ENABLE_SKINNY (1)
set -euo pipefail

source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
venv=${V620_VENV:-$source_dir/.venv}
model=${V620_MODEL:-$source_dir/models/qwen38-flash-next}
ple_int4=${V620_PLE_INT4:-$source_dir/../vllm-rdna2-qwen/models/qwen38-flash-next-ple/ples_int4}
port=${V620_PORT:-${V620_TEST_PORT:-8083}}
host=${V620_HOST:-127.0.0.1}
compile_mode=${V620_COMPILE_MODE:-3}
cudagraph_mode=${V620_CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}
capture_sizes=${V620_CG_SIZES:-[3,6,12]}
export PYTHONPATH=$source_dir
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_PLE_CPU_OFFLOAD=1 VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_PLE_QUANT_DIR=$ple_int4 VLLM_PLE_OFFLOAD_READY_TIMEOUT=3600
export VLLM_ROCM_MOE_PREFILL=0 VLLM_GDN_HIP_PREFILL=0
export VLLM_RDNA_FUSED_SE=1
export VLLM_TUNED_CONFIG_FOLDER=$source_dir/tuned-moe
if [[ ${V620_ENABLE_RESIDENT:-1} == 1 ]]; then
    export VLLM_RDNA_MOE_RESIDENT=1
fi
if [[ ${V620_ENABLE_SKINNY:-1} == 1 ]]; then
    export VLLM_RDNA_MOE_RESIDENT_SKINNY=1
fi
export VLLM_RDNA_DENSE_INT8=0 VLLM_RDNA_DENSE_INT8_ONLY=0 VLLM_RDNA_DENSE_GEMV=0
export VLLM_RDNA_AR=1 VLLM_RDNA_AR_MAX_KB=64 VLLM_RDNA_AR_BLOCKS=0 VLLM_RDNA_AR_PACE=0
export HSA_FORCE_FINE_GRAIN_PCIE=1 HSA_ENABLE_SDMA=0 OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false PYTHONFAULTHANDLER=1
export VLLM_CAUSAL_CONV1D_RDNA2_FWD=0 VLLM_CAUSAL_CONV1D_RDNA2_UPDATE=0
export VLLM_ENABLE_STARTUP_PLAN=0 VLLM_ROCM_USE_AITER=0 TORCH_BLAS_PREFER_HIPBLASLT=0
export VLLM_CACHE_ROOT=$source_dir/cache/vllm TRITON_CACHE_DIR=$source_dir/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$source_dir/cache/inductor
export TORCH_EXTENSIONS_DIR=$source_dir/cache/extensions
# The Rust frontend rejects --mm-processor-kwargs; this launcher needs it.
export VLLM_USE_RUST_FRONTEND=0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 VLLM_ENGINE_READY_TIMEOUT_S=3600
# ROCm runtime selection. ROCM_LIB pins the whole stack (torch + the vllm
# extensions, which have no RUNPATH and otherwise resolve through
# /opt/rocm/lib). rocr-fix is a 7.14-era patched ROCr: never preload it over a
# ROCm 10 runtime, so the preload stays opt-in.
rocm_lib=${V620_ROCM_LIB:-/opt/rocm/core-10.0/lib}
if [[ -n $rocm_lib ]]; then
    export PATH="$(dirname -- "$rocm_lib")/bin:$venv/bin:$PATH"
    export LD_LIBRARY_PATH="$rocm_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [[ ${V620_PRELOAD_ROCR:-0} == 1 ]]; then
    export LD_PRELOAD=${V620_ROCR_FIX:-/home/wsantos/work/rocr-fix/rocm-systems/build/rocr/lib/libhsa-runtime64.so}
fi
# TunableOp: solution IDs belong to a rocBLAS build, not just a version number.
# The rows root holds one rocblas-<sha256-first-12>/ directory per build.
if [[ ${V620_TUNABLEOP:-1} == 1 ]]; then
    # shellcheck source=tools/rdna2/tunableop_env.sh
    source "$source_dir/tools/rdna2/tunableop_env.sh"
    rows_root=${V620_TUNABLEOP_ROOT:-$source_dir/../pr17-tunableop}
    configure_v620_tunableop \
        "${V620_ROCBLAS_LIBRARY:-$rocm_lib/librocblas.so.5}" "$rows_root"
else
    export PYTORCH_TUNABLEOP_ENABLED=0 PYTORCH_TUNABLEOP_TUNING=0
    export PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0
    unset PYTORCH_TUNABLEOP_FILENAME
fi
command=("$venv/bin/python" -m vllm.entrypoints.openai.api_server
    --model "$model" --served-model-name active qwen3.8-flash-next
    --host "$host" --port "$port" --tensor-parallel-size 4
    --pipeline-parallel-size 1 --enable-expert-parallel --enable-ep-weight-filter
    --dtype float16 --max-model-len 262144 --block-size 1024 --max-num-seqs 4
    --max-num-batched-tokens 4096 --kv-cache-memory-bytes 4026531840
    --compilation-config "{\"mode\":$compile_mode,\"cudagraph_mode\":\"$cudagraph_mode\",\"cudagraph_capture_sizes\":$capture_sizes}"
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
    --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3
    --default-chat-template-kwargs '{"enable_thinking":false}'
    --limit-mm-per-prompt '{"image":255,"video":32}'
    --mm-processor-kwargs '{"max_pixels":602112}'
    --enable-prefix-caching --mamba-cache-mode align
    --kernel-config '{"moe_backend":"triton"}')
if [[ ${1:-} == --dry-run ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
if pgrep -u "$(id -u)" -f 'vllm.entrypoints|VLLM::EngineCore|VLLM::Worker' >/dev/null; then
    printf 'Existing vLLM process present; refusing overlap.\n' >&2
    exit 2
fi
if [[ ! -x $venv/bin/python || ! -f $model/config.json ]]; then
    printf 'Requires a built venv and a local model directory.\n' >&2
    exit 2
fi
if [[ ${PYTORCH_TUNABLEOP_ENABLED:-0} == 1 ]]; then
    "$venv/bin/python" "$source_dir/tools/rdna2/check_v620_tuning.py" \
        --rows-template "$PYTORCH_TUNABLEOP_FILENAME" "${command[@]:1}"
fi
"$venv/bin/python" -c 'import vllm; print(vllm.__file__)'
cd "$source_dir"
exec "${command[@]}" "$@"
