#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Shared launcher harness for the gfx1030 serve scripts.
#
# Source it, set your knobs, then call the helpers. Nothing here bakes in a
# host-specific path: the tree is derived from this file's location, the venv
# comes from $VENV (or an activated $VIRTUAL_ENV), and the checkpoint comes
# from $MODEL.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/rdna_launcher_common.sh"
#   MODEL=/path/to/ckpt VENV=/path/to/venv
#   rdna_init                 # tree, venv, ROCm SDK, shared vLLM env, caches
#   rdna_tunableop            # honours TUNABLEOP=0/1 and TUNABLEOP_PROFILE
#   rdna_select_attention fa  # fa | triton  (sets $attention_backend)
#   rdna_mtp_args 2           # sets $spec_args and $capture_sizes
#   rdna_kill_stale           # scoped to $VLLM_CACHE_ROOT
#
# Env knobs (all optional unless noted):
#   VLLM_TREE   repo root        (default: two levels above tools/rdna/)
#   VENV        python venv      (default: active $VIRTUAL_ENV; else error)
#   MODEL       checkpoint path  (launchers that serve a model: required)
#   HIP_VISIBLE_DEVICES / GPUIDS (default: 0,1,2,3)
#   ROCM_SDK_ROOT                (default: /opt/rocm/core-7.14)
#   TUNABLEOP=0|1                (default: 1)
#   TUNABLEOP_PROFILE=<name>     explicit TunableOp profile (validated)
#   VLLM_CACHE_ROOT / TRITON_CACHE_DIR / TORCHINDUCTOR_CACHE_DIR /
#   TORCH_EXTENSIONS_DIR         (default: <tree>/cache/*)
#   RUN_CWD                      (default: <tree>/cache/run; never /tmp)

_RDNA_COMMON_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

rdna_die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

rdna_require() {
    local name=$1 hint=${2:-}
    if [[ -z ${!name:-} ]]; then
        printf 'ERROR: %s is not set.%s\n' "$name" "${hint:+ $hint}" >&2
        exit 1
    fi
}

rdna_require_model() {
    rdna_require MODEL "Pass MODEL=/path/to/checkpoint (the launcher has no baked-in default)."
}

rdna_init() {
    source_dir=${VLLM_TREE:-$(cd "$_RDNA_COMMON_DIR/../.." && pwd)}
    export source_dir VLLM_TREE=$source_dir

    runtime=${VENV:-${VIRTUAL_ENV:-}}
    [[ -n $runtime ]] || rdna_die 'set VENV=/path/to/python-venv (or activate one so $VIRTUAL_ENV is set).'
    # RDNA_DRY_RUN lets serve_rdna.sh PRINT the plan for a fork user before the
    # venv exists; it never skips a check on a real launch.
    if [[ ${RDNA_DRY_RUN:-0} != 1 ]]; then
        [[ -x $runtime/bin/python ]] || rdna_die "VENV=$runtime has no bin/python."
    fi
    VENV=$runtime
    export VENV

    ROCM_SDK_ROOT=${ROCM_SDK_ROOT:-/opt/rocm/core-7.14}
    ROCM_SDK_LIB="$VENV/lib/python3.12/site-packages/_rocm_sdk_libraries/lib"
    ROCM_SDK="$VENV/lib/python3.12/site-packages/_rocm_sdk_core/lib"
    export ROCM_SDK_ROOT ROCM_SDK_LIB ROCM_SDK
    export LD_LIBRARY_PATH="$ROCM_SDK_LIB:$ROCM_SDK/host-math/lib:$ROCM_SDK/rocm_sysdeps/lib:$ROCM_SDK/core/lib:$VENV/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export ROCM_HOME=${ROCM_HOME:-$ROCM_SDK_ROOT} ROCM_PATH=${ROCM_PATH:-$ROCM_SDK_ROOT}
    export HIP_PATH=${HIP_PATH:-$ROCM_SDK_ROOT} HIP_ROOT_DIR=${HIP_ROOT_DIR:-$ROCM_SDK_ROOT}
    [[ -d $ROCM_SDK_ROOT/include ]] && export CPATH="$ROCM_SDK_ROOT/include${CPATH:+:$CPATH}"
    [[ -d $ROCM_SDK_ROOT/lib ]] && export LIBRARY_PATH="$ROCM_SDK_ROOT/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
    [[ -d $ROCM_SDK_ROOT/bin ]] && export PATH="$ROCM_SDK_ROOT/bin:$PATH"

    export PYTHONPATH="$source_dir${PYTHONPATH:+:$PYTHONPATH}"
    export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
    export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-${GPUIDS:-0,1,2,3}}
    export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
    export GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-2}
    export VLLM_ROCM_USE_AITER=${VLLM_ROCM_USE_AITER:-0}
    export VLLM_ROCM_USE_AITER_MOE=${VLLM_ROCM_USE_AITER_MOE:-0}
    export VLLM_RDNA_FORCE_FP16=${VLLM_RDNA_FORCE_FP16:-1}
    export TORCH_BLAS_PREFER_HIPBLASLT=${TORCH_BLAS_PREFER_HIPBLASLT:-0}
    export VLLM_BATCH_INVARIANT=${VLLM_BATCH_INVARIANT:-0}

    export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$source_dir/cache/vllm}
    export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$source_dir/cache/triton}
    export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$source_dir/cache/inductor}
    export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$source_dir/cache/extensions}
}

rdna_tunableop() {
    # shellcheck source=tools/rdna2_028/tunableop_env.sh
    source "$source_dir/tools/rdna2_028/tunableop_env.sh"
    if [[ ${TUNABLEOP:-1} == 1 ]]; then
        configure_tunableop "$ROCM_SDK_LIB/librocblas.so.5" "$source_dir/tunableop" || return 1
    else
        export PYTORCH_TUNABLEOP_ENABLED=0 PYTORCH_TUNABLEOP_TUNING=0
        export PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0
    fi
}

rdna_select_attention() {
    local mode=${1:-fa}
    if [[ $mode == fa ]]; then
        export VLLM_USE_RDNA2_FA=1
        attention_backend=RDNA_ATTN
        export VLLM_FA_RDNA2_GQA_DECODE=${VLLM_FA_RDNA2_GQA_DECODE:-1}
        unset FLASH_ATTENTION_TRITON_AMD_ENABLE || true
    else
        export VLLM_USE_RDNA2_FA=0
        attention_backend=TRITON_ATTN
        export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
    fi
    export attention_backend
}

rdna_mtp_args() {
    local mtp=${1:-0} decode_width sizes mult
    if [[ $mtp == 0 ]]; then
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
}

rdna_run_cwd() {
    local dir=${RUN_CWD:-$source_dir/cache/run}
    mkdir -p "$dir"
    cd "$dir"
}

rdna_kill_stale() {
    local _sig _p
    for _sig in TERM KILL; do
        for _p in $(pgrep -f "entrypoints.cli.main serve|entrypoints.openai.api_server|VLLM::Worker|VLLM::EngineCore|PleOffloadWorker" 2>/dev/null); do
            [ -r "/proc/$_p/environ" ] || continue
            tr '\0' '\n' < "/proc/$_p/environ" 2>/dev/null | grep -q "^VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT}$" && kill -"$_sig" "$_p" 2>/dev/null
        done
        [ "$_sig" = TERM ] && sleep 8
    done
    sleep 3
}
