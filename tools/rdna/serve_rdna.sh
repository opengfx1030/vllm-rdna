#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Single entry point for serving the RDNA (gfx1030/gfx1100) fork.
#
#   bash tools/rdna/serve_rdna.sh RECIPE=<name> [KEY=value ...]
#   PRINT=1 bash tools/rdna/serve_rdna.sh RECIPE=<name> [KEY=value ...]
#
# A recipe (tools/rdna/recipes/<name>.env) holds ONLY that config's deltas as
# shell KEY=value lines. This script owns the shared machinery: tree/venv/ROCm
# and LD_LIBRARY_PATH resolution (tools/rdna/rdna_launcher_common.sh), TunableOp
# profile selection, GPU selection, per-arm cache root, stale-process teardown,
# the shared vLLM environment block, MTP/capture-ladder logic, ports, and the
# vLLM argument assembly.
#
# Precedence (lowest to highest):
#   recipe defaults < caller environment < trailing KEY=value arguments
#
# PRINT=1 (alias --dry-run) assembles the environment, the TunableOp profile
# that would be selected, and the exact command, then exits without launching.
#
# Nothing in this file or the recipes bakes in a host, user, or IP.
set -euo pipefail

RDSE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RDSE_RECIPES=$RDSE_DIR/recipes

# shellcheck source=tools/rdna/rdna_launcher_common.sh
source "$RDSE_DIR/rdna_launcher_common.sh"

rdse_usage() {
    cat >&2 <<'EOF'
usage: serve_rdna.sh RECIPE=<name> [KEY=value ...]
       PRINT=1 serve_rdna.sh RECIPE=<name> [KEY=value ...]

  RECIPE=<name>   recipe under tools/rdna/recipes/<name>.env (required)
  KEY=value       overrides a recipe default (highest precedence)
  PRINT=1         assemble + print env/command/TunableOp profile, do not launch
                  (alias: --dry-run, -n)
  --help          show this help
EOF
}

rdse_print=0
if [[ ${PRINT:-0} == 1 || ${DRY_RUN:-0} == 1 ]]; then
    rdse_print=1
fi
rdse_recipe=""
declare -a rdse_overrides=()
for arg in "$@"; do
    case $arg in
        PRINT=1 | --dry-run | -n) rdse_print=1 ;;
        RECIPE=*) rdse_recipe=${arg#RECIPE=} ;;
        -h | --help) rdse_usage; exit 0 ;;
        -*) rdse_usage; rdna_die "unknown option '$arg'." ;;
        *=*) rdse_overrides+=("$arg") ;;
        *) rdse_usage; rdna_die "unexpected argument '$arg'." ;;
    esac
done

if [[ -z $rdse_recipe ]]; then
    rdse_usage
    rdna_die 'RECIPE=<name> is required.'
fi
if [[ $rdse_recipe == */* || $rdse_recipe == .* || $rdse_recipe == *..* ]]; then
    rdna_die "RECIPE must be a bare name under tools/rdna/recipes/ (got '$rdse_recipe')."
fi
recipe_file=$RDSE_RECIPES/$rdse_recipe.env
if [[ ! -r $recipe_file ]]; then
    printf 'ERROR: unknown recipe "%s".\n' "$rdse_recipe" >&2
    printf 'Available recipes:\n' >&2
    for f in "$RDSE_RECIPES"/*.env; do
        [[ -e $f ]] || continue
        printf '  - %s\n' "$(basename "$f" .env)" >&2
    done
    exit 2
fi

# Caller-set variables win over recipe defaults, so snapshot every key the
# recipe touches (values only, setness included) and re-apply after sourcing.
declare -A rdse_saved=()
declare -a rdse_keys=()
while IFS= read -r _key; do
    [[ -n $_key ]] || continue
    rdse_keys+=("$_key")
    if [[ -v $_key ]]; then
        rdse_saved[$_key]=${!_key}
    fi
done < <(grep -oE '^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*=' "$recipe_file" | tr -d ' =')

# shellcheck disable=SC1090
source "$recipe_file"

if ((${#rdse_saved[@]})); then
    for _key in "${!rdse_saved[@]}"; do
        printf -v "$_key" '%s' "${rdse_saved[$_key]}"
    done
fi
if ((${#rdse_overrides[@]})); then
    for _ov in "${rdse_overrides[@]}"; do
        _key=${_ov%%=*}
        [[ $_key =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || rdna_die "bad override '$_ov' (expected KEY=value)."
        printf -v "$_key" '%s' "${_ov#*=}"
        rdse_keys+=("$_key")
    done
fi

# Export only environment knobs; recipe config keys (PORT, TP, MTP, ...) stay
# shell variables so the child environment matches the per-model launchers.
if ((${#rdse_keys[@]})); then
    for _key in "${rdse_keys[@]}"; do
        case $_key in
            VLLM_* | NCCL_* | RCCL_* | HSA_* | TORCH_* | PYTORCH_* | OMP_* | PYTHON* | \
                TOKENIZERS_* | FLASH_* | HIP_* | ROCM_* | GPU_* | TRITON_* | HF_* | TRANSFORMERS_*)
                if [[ -v $_key ]]; then export "${_key?}"; fi
                ;;
        esac
    done
fi

MODEL=${MODEL:-${MODEL_PATH:-}}
SERVED_NAME=${SERVED_NAME:-}
SEQS=${SEQS:-${MAX_NUM_SEQS:-8}}
MAXBAT=${MAXBAT:-${MAX_BATCHED_TOKENS:-2048}}
MAXLEN=${MAXLEN:-${MAX_MODEL_LEN:-}}
MIN_MAXLEN=${MIN_MAXLEN:-}
KV=${KV:-${KVCAP:-${KV_CACHE_MEMORY:-}}}
GMEM=${GMEM:-${GPU_MEM:-}}
HOST=${HOST-127.0.0.1}
PORT=${PORT:-18094}
TP=${TP:-4}
BLOCK_SIZE=${BLOCK_SIZE:-}
LPTH=${LPTH:-}
PREFILL_INTERVAL=${PREFILL_INTERVAL:-}
DIST_TIMEOUT=${DIST_TIMEOUT:-}
SERVE_ENTRY=${SERVE_ENTRY:-cli}
COMPILE_MODE=${COMPILE_MODE:-}
CG_MODE=${CG_MODE:-FULL_AND_PIECEWISE}
CG_CAPTURE_SIZES=${CG_CAPTURE_SIZES:-1}
CG_EXTRA=${CG_EXTRA:-}
COMPILATION_CONFIG=${COMPILATION_CONFIG:-}
EAGER=${EAGER:-0}
ALLOC_CONF=${ALLOC_CONF:-}
TUNED_CONFIG=${TUNED_CONFIG:-0}
ATTN=${ATTN:-fa}
FEATURES=${FEATURES:-}
REQUIRED_VARS=${REQUIRED_VARS:-}
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-1}
LIMIT_MM_PER_PROMPT=${LIMIT_MM_PER_PROMPT:-'{"image":1}'}
MM_PROCESSOR_KWARGS=${MM_PROCESSOR_KWARGS:-'{"max_pixels":1605632}'}

rdna_require_model
if [[ -n $REQUIRED_VARS ]]; then
    for _rv in $REQUIRED_VARS; do
        rdna_require "$_rv"
    done
fi
[[ -n $KV ]] || rdna_die "no KV budget set; add KV=<bytes> (or KVCAP=) to recipe '$rdse_recipe'."
if [[ -n $MIN_MAXLEN && -n $MAXLEN ]] && ((MAXLEN < MIN_MAXLEN)); then
    rdna_die "MAXLEN=$MAXLEN is below the $MIN_MAXLEN floor required by recipe '$rdse_recipe'."
fi

if ((rdse_print)); then
    RDNA_DRY_RUN=1
fi
rdna_init

if [[ $TUNED_CONFIG == 1 ]]; then
    export VLLM_TUNED_CONFIG_FOLDER=${VLLM_TUNED_CONFIG_FOLDER:-$source_dir/tuned-moe}
fi
if [[ -v W4A8 ]]; then
    export VLLM_RDNA2_W4A8_SDOT4=${VLLM_RDNA2_W4A8_SDOT4:-$W4A8}
fi
if [[ -v RDNA_AR ]]; then
    export VLLM_RDNA_AR=${VLLM_RDNA_AR:-$RDNA_AR}
fi

case $ALLOC_CONF in
    false) export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False} ;;
    true) export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} ;;
    auto)
        if [[ ${VLLM_PLE_CPU_OFFLOAD:-0} == 1 ]]; then
            export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}
        else
            export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
        fi
        ;;
    "") ;;
    *) rdna_die "ALLOC_CONF must be false, true or auto (got '$ALLOC_CONF')." ;;
esac

rdse_profile_line=""
rdse_tunableop_report() {
    if [[ ${TUNABLEOP:-1} != 1 ]]; then
        rdse_profile_line="disabled (TUNABLEOP=0)"
        return
    fi
    # shellcheck source=tools/rdna2_028/tunableop_env.sh
    source "$source_dir/tools/rdna2_028/tunableop_env.sh"
    local lib=$ROCM_SDK_LIB/librocblas.so.5 requested=${TUNABLEOP_PROFILE:-}
    if [[ -n $requested ]]; then
        if _tunableop_profiles "$source_dir/tunableop/profiles.json" |
            awk -F'\t' -v n="$requested" '$1 == n { found = 1 } END { exit !found }'; then
            rdse_profile_line="$requested (requested; validated against the loaded library at launch)"
        else
            rdse_profile_line="$requested (NOT registered — a launch would refuse to start)"
        fi
        return
    fi
    if [[ -f $lib ]] && command -v sha256sum >/dev/null 2>&1; then
        local sha match
        sha=$(sha256sum -- "$lib")
        sha=${sha:0:12}
        match=$(_tunableop_profiles "$source_dir/tunableop/profiles.json" |
            awk -F'\t' -v h="$sha" 'substr($3, 1, 12) == h { print $1; exit }')
        if [[ -n $match ]]; then
            rdse_profile_line="$match (auto-selected for librocblas.so.5 $sha)"
        else
            rdse_profile_line="no profile matches librocblas.so.5 $sha; per-user fallback row set"
        fi
    else
        rdse_profile_line="librocblas.so.5 not found at $lib — auto-selection runs at launch"
    fi
}

if ((rdse_print)); then
    rdse_tunableop_report
else
    rdna_tunableop || exit 1
fi

attention_backend=""
case $ATTN in
    fa | triton) rdna_select_attention "$ATTN" ;;
    none | off) : ;;
    *) rdna_die "ATTN must be fa, triton or none (got '$ATTN')." ;;
esac

rdna_mtp_args "${MTP:-0}"

if [[ $EAGER == 1 ]]; then
    comp_config=""
elif [[ -n $COMPILATION_CONFIG ]]; then
    comp_config=$COMPILATION_CONFIG
else
    comp_config="{"
    [[ -n $COMPILE_MODE ]] && comp_config+="\"mode\":$COMPILE_MODE,"
    comp_config+="\"cudagraph_mode\":\"$CG_MODE\""
    [[ $CG_CAPTURE_SIZES == 1 ]] && comp_config+=",\"cudagraph_capture_sizes\":$capture_sizes"
    comp_config+="$CG_EXTRA}"
fi

case $SERVE_ENTRY in
    cli) cmd=("$VENV/bin/python" -m vllm.entrypoints.cli.main serve) ;;
    api) cmd=("$VENV/bin/python" -m vllm.entrypoints.openai.api_server) ;;
    *) rdna_die "SERVE_ENTRY must be 'cli' or 'api' (got '$SERVE_ENTRY')." ;;
esac

cmd+=(--model "$MODEL")
[[ -n $SERVED_NAME ]] && cmd+=(--served-model-name "$SERVED_NAME")
[[ -n $HOST ]] && cmd+=(--host "$HOST")
cmd+=(--port "$PORT")
[[ -n $attention_backend ]] && cmd+=(--attention-backend "$attention_backend")
cmd+=(--tensor-parallel-size "$TP" --dtype float16)
[[ -n $BLOCK_SIZE ]] && cmd+=(--block-size "$BLOCK_SIZE")
[[ -n $MAXLEN ]] && cmd+=(--max-model-len "$MAXLEN")
cmd+=(--max-num-seqs "$SEQS" --max-num-batched-tokens "$MAXBAT")
[[ -n $LPTH ]] && cmd+=(--long-prefill-token-threshold "$LPTH")
[[ -n $PREFILL_INTERVAL ]] && cmd+=(--prefill-schedule-interval "$PREFILL_INTERVAL")
[[ -n $DIST_TIMEOUT ]] && cmd+=(--distributed-timeout-seconds "$DIST_TIMEOUT")
[[ -n $GMEM ]] && cmd+=(--gpu-memory-utilization "$GMEM")
cmd+=(--kv-cache-memory-bytes "$KV")
if [[ $EAGER == 1 ]]; then
    cmd+=(--enforce-eager)
elif [[ -n $comp_config ]]; then
    cmd+=(--compilation-config "$comp_config")
fi
((${#spec_args[@]})) && cmd+=("${spec_args[@]}")

for feat in $FEATURES; do
    case $feat in
        trust-remote-code) cmd+=(--trust-remote-code) ;;
        prefix-caching)
            [[ $ENABLE_PREFIX_CACHING != 0 ]] && cmd+=(--enable-prefix-caching)
            ;;
        prompt-tokens-details) cmd+=(--enable-prompt-tokens-details) ;;
        auto-tool) cmd+=(--enable-auto-tool-choice --tool-call-parser qwen3_coder) ;;
        reasoning) cmd+=(--reasoning-parser qwen3) ;;
        expert-parallel) cmd+=(--enable-expert-parallel) ;;
        language-model-only) cmd+=(--language-model-only) ;;
        skip-mm-profiling) cmd+=(--skip-mm-profiling) ;;
        mamba-align) cmd+=(--mamba-cache-mode align) ;;
        vision-cap)
            cmd+=(--limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT" --mm-processor-kwargs "$MM_PROCESSOR_KWARGS")
            ;;
        *) rdna_die "unknown FEATURES token '$feat' in recipe '$rdse_recipe'." ;;
    esac
done

if [[ -n ${EXTRA_ARGS:-} ]]; then
    read -r -a rdse_extra <<<"$EXTRA_ARGS"
    cmd+=("${rdse_extra[@]}")
fi

rdse_quote() {
    local s=$1
    s=${s//\'/\'\\\'\'}
    printf "'%s'" "$s"
}

RDSE_ENV_KEYS=(
    LD_LIBRARY_PATH ROCM_HOME ROCM_PATH HIP_PATH HIP_ROOT_DIR
    PYTHONPATH HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
    HIP_VISIBLE_DEVICES VLLM_WORKER_MULTIPROC_METHOD GPU_MAX_HW_QUEUES
    VLLM_ROCM_USE_AITER VLLM_ROCM_USE_AITER_MOE VLLM_RDNA_FORCE_FP16
    TORCH_BLAS_PREFER_HIPBLASLT VLLM_BATCH_INVARIANT
    VLLM_CACHE_ROOT TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR TORCH_EXTENSIONS_DIR
    PYTORCH_TUNABLEOP_ENABLED PYTORCH_TUNABLEOP_TUNING
    PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED PYTORCH_TUNABLEOP_FILENAME
    PYTORCH_TUNABLEOP_RECORD_UNTUNED PYTORCH_TUNABLEOP_UNTUNED_FILENAME
    VLLM_USE_RDNA2_FA VLLM_FA_RDNA2_GQA_DECODE VLLM_FA_RDNA2_GQA_MODE
    FLASH_ATTENTION_TRITON_AMD_ENABLE
    VLLM_USE_V2_MODEL_RUNNER VLLM_USE_AOT_COMPILE VLLM_DISABLE_COMPILE_CACHE
    VLLM_USE_BREAKABLE_CUDAGRAPH VLLM_RDNA_FUSED_HC
    VLLM_ROCM_MOE_PREFILL VLLM_GDN_HIP_PREFILL VLLM_RDNA_FUSED_SE
    VLLM_RDNA_DENSE_INT8 VLLM_RDNA_DENSE_INT8_ONLY VLLM_RDNA_DENSE_GEMV
    VLLM_RDNA2_W4A8_SDOT4 VLLM_RDNA2_W4A8_DEBUG
    VLLM_FORCE_CUSTOM_ALL_REDUCE VLLM_RDNA_AR VLLM_RDNA_AR_MAX_KB
    VLLM_RDNA_AR_ONESHOT_KB VLLM_RDNA_AR_BLOCKS VLLM_RDNA_AR_PACE
    VLLM_CAUSAL_CONV1D_RDNA2_FWD VLLM_CAUSAL_CONV1D_RDNA2_UPDATE
    VLLM_ENABLE_STARTUP_PLAN VLLM_TUNED_CONFIG_FOLDER
    VLLM_ROCM_SKIP_LIVE_TAIL_HASH
    VLLM_PLE_CPU_OFFLOAD VLLM_PLE_OFFLOAD_READY_TIMEOUT VLLM_PLE_QUANT_DIR
    VLLM_EXL3_DEBUG
    NCCL_P2P_LEVEL RCCL_P2P_NET_DISABLE RCCL_P2P_BATCH_ENABLE NCCL_PROTO
    RCCL_MSCCL_ENABLE
    HSA_FORCE_FINE_GRAIN_PCIE HSA_ENABLE_SDMA OMP_NUM_THREADS
    TOKENIZERS_PARALLELISM PYTHONFAULTHANDLER PYTORCH_CUDA_ALLOC_CONF
)

rdse_print_plan() {
    printf '# recipe:    %s (%s)\n' "$rdse_recipe" "$recipe_file"
    printf '# tunableop: %s\n' "$rdse_profile_line"
    printf '# entry:     %s\n' "$SERVE_ENTRY"
    printf '\n# --- environment ---\n'
    for _key in "${RDSE_ENV_KEYS[@]}"; do
        if [[ -v $_key ]]; then
            printf 'export %s=%s\n' "$_key" "$(rdse_quote "${!_key}")"
        fi
    done
    printf '\n# --- command ---\n'
    local -a quoted=()
    for _a in "${cmd[@]}"; do
        quoted+=("$(rdse_quote "$_a")")
    done
    printf '%s\n' "${quoted[*]}"
}

if ((rdse_print)); then
    rdse_print_plan
    exit 0
fi

rdna_kill_stale
rdna_run_cwd
exec "${cmd[@]}"
