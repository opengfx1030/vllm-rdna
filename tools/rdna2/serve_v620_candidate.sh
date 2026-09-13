#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Isolated candidate launcher. Never stops, updates, or installs a service.
set -euo pipefail

source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
test_root=$source_dir
if [[ $(basename -- "$source_dir") == source ]]; then
    test_root=$(dirname -- "$source_dir")
fi
case $(basename -- "$test_root") in
    vllm-rdna-testing|v620-vllm-testing) ;;
    *) printf 'Refusing launch outside a dedicated V620 testing directory.\n' >&2; exit 2 ;;
esac
venv=$test_root/.venv
model=${V620_MODEL:-/home/george/v620-vllm/models/intel-autoround}
port=${V620_TEST_PORT:-8081}
mtp_tokens=${V620_MTP_TOKENS:-1}
if [[ ! $port =~ ^[0-9]+$ || ! $mtp_tokens =~ ^[0-9]+$ ]]; then
    printf 'Port and MTP token count must be nonnegative integers.\n' >&2
    exit 2
fi

export VLLM_USE_V2_MODEL_RUNNER=1
export HSA_ENABLE_SDMA=${HSA_ENABLE_SDMA:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false
export PYTHONFAULTHANDLER=1
export VLLM_CACHE_ROOT=$test_root/cache/vllm
export TRITON_CACHE_DIR=$test_root/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$test_root/cache/inductor
export TORCH_EXTENSIONS_DIR=$test_root/cache/extensions
export VLLM_RDNA_AR=${VLLM_RDNA_AR:-0}
export VLLM_RDNA_DENSE_INT8=${VLLM_RDNA_DENSE_INT8:-0}
export VLLM_RDNA_DENSE_INT8_ONLY=${VLLM_RDNA_DENSE_INT8_ONLY:-0}
export VLLM_RDNA_DENSE_GEMV=${VLLM_RDNA_DENSE_GEMV:-0}
export VLLM_CAUSAL_CONV1D_RDNA2_FWD=0
export VLLM_CAUSAL_CONV1D_RDNA2_UPDATE=0
export VLLM_ENABLE_STARTUP_PLAN=${VLLM_ENABLE_STARTUP_PLAN:-0}
# TunableOp solution IDs need qualification against this environment's rocBLAS.
export PYTORCH_TUNABLEOP_ENABLED=0
export PYTORCH_TUNABLEOP_TUNING=0
export PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0
export VLLM_ROCM_USE_AITER=0
export TORCH_BLAS_PREFER_HIPBLASLT=0

decode_width=$((mtp_tokens + 1))
capture_sizes="[$decode_width,$((decode_width * 2)),$((decode_width * 4))]"
compilation_config="{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":$capture_sizes}"
if [[ -z ${V620_MM_LIMIT:-} ]]; then
    printf 'Set V620_MM_LIMIT explicitly for this candidate memory test.\n' >&2
    printf 'Use a JSON image/video count matching your intended workload.\n' >&2
    exit 2
fi
mm_limit=$V620_MM_LIMIT
command=("$venv/bin/python" -m vllm.entrypoints.openai.api_server
    --model "$model" --served-model-name active qwen3.8-flash-next
    --host 127.0.0.1 --port "$port"
    --tensor-parallel-size 4 --enable-expert-parallel --enable-ep-weight-filter
    --dtype float16 --max-model-len 262144
    --block-size 1024
    --max-num-seqs 4 --max-num-batched-tokens 2048
    --kv-cache-memory-bytes 4294967296
    --compilation-config "$compilation_config"
    --engram-config '{"cpu_offload":true}'
    --enable-auto-tool-choice --tool-call-parser qwen3_xml
    --reasoning-parser qwen3
    --default-chat-template-kwargs '{"enable_thinking":false}'
    --limit-mm-per-prompt "$mm_limit"
    --mm-processor-kwargs '{"max_pixels":1638400}')
if (( mtp_tokens > 0 )); then
    command+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$mtp_tokens}")
fi
if [[ ${1:-} == --dry-run ]]; then
    shift
    printf 'Source: %s\nEnvironment: %s\nCache: %s\n' "$source_dir" "$venv" "$VLLM_CACHE_ROOT"
    printf '%q ' "${command[@]}" "$@"
    printf '\n'
    exit 0
fi
if [[ $(uname -s) != Linux || ! -x $venv/bin/python || ! -f $model/config.json ]]; then
    printf 'Requires Linux, a separately built testing .venv, and a local model.\n' >&2
    exit 2
fi
# Refuse to overlap another vLLM instance on the four cards.
if pgrep -u "$(id -u)" -f 'vllm.entrypoints|VLLM::EngineCore|VLLM::Worker' >/dev/null; then
    printf 'An existing vLLM process is present. No process was stopped.\n' >&2
    exit 2
fi
site_packages=$("$venv/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
runtime_sdk=$site_packages/_rocm_sdk_core
export PATH="$venv/bin:/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$runtime_sdk/lib:$runtime_sdk/lib/host-math/lib:/opt/rocm/core-10.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ ${V620_TUNABLEOP:-0} == 1 ]]; then
    # V620_ROCBLAS_LIBRARY must identify the library loaded by this wheel SDK.
    # shellcheck source=tools/rdna2/tunableop_env.sh
    source "$source_dir/tools/rdna2/tunableop_env.sh"
    rows_root=${V620_TUNABLEOP_ROOT:-$test_root/tunableop}
    if [[ ! -d $rows_root && -z ${V620_TUNABLEOP_ROOT:-} ]]; then
        rows_root=$source_dir/tunableop
    fi
    configure_v620_tunableop "${V620_ROCBLAS_LIBRARY:-}" "$rows_root"
    if [[ $PYTORCH_TUNABLEOP_ENABLED == 1 ]]; then
        "$venv/bin/python" "$source_dir/tools/rdna2/check_v620_tuning.py" \
            --rows-template "$PYTORCH_TUNABLEOP_FILENAME" "${command[@]:1}" "$@"
    fi
fi
"$venv/bin/python" - "$test_root" "$source_dir" <<'PY'
import sys
from pathlib import Path

root, source = (Path(arg).resolve() for arg in sys.argv[1:])
if not Path(sys.prefix).resolve().is_relative_to(root):
    raise SystemExit("Refusing an environment outside the testing directory")
import vllm

if not Path(vllm.__file__).resolve().is_relative_to(source):
    raise SystemExit("The testing environment imports a different vLLM checkout")
PY
cd "$source_dir"
exec "${command[@]}" "$@"
