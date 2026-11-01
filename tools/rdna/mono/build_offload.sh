#!/usr/bin/env bash
# Isolated editable build; BASE_VENV is read-only dependency storage.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-6,7}
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
TREE=$(cd "$(dirname "$0")/../../.." && pwd)
: "${BASE_VENV:?Set BASE_VENV to the existing ROCm 7.14 environment}"
ROCM_HOME=${ROCM_HOME:-/opt/rocm/core-7.14}
RUN=${RUN:-$HOME/w4a8_runs/moe-offload}
mkdir -p "$RUN"
mkdir -p "$RUN/build-tmp"
export TMPDIR=$RUN/build-tmp
cd "$TREE"
if [[ ! -x .venv/bin/python ]]; then
    uv venv --python "$BASE_VENV/bin/python" --seed .venv
fi
SITE=$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
BASE_SITE=$("$BASE_VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
printf '%s\n' "$BASE_SITE" > "$SITE/rocm_build_dependencies.pth"
for package in torch _rocm_sdk_libraries _rocm_sdk_core; do
    if [[ -d $BASE_SITE/$package && ! -e $SITE/$package ]]; then
        ln -s "$BASE_SITE/$package" "$SITE/$package"
    fi
done
export VLLM_TARGET_DEVICE=rocm PYTORCH_ROCM_ARCH=gfx1030
export ROCM_HOME ROCM_PATH=$ROCM_HOME HIP_PATH=$ROCM_HOME HIP_ROOT_DIR=$ROCM_HOME
export CMAKE_HIP_COMPILER=$ROCM_HOME/bin/hipcc
export CMAKE_HIP_COMPILER_ROCM_ROOT=$ROCM_HOME
export VLLM_PYTHON_EXECUTABLE=$TREE/.venv/bin/python
export MAX_JOBS=${MAX_JOBS:-8} CMAKE_BUILD_TYPE=RelWithDebInfo
export PATH=$ROCM_HOME/bin:$PATH
export PYTHONPATH=$TREE
export LD_LIBRARY_PATH=$BASE_SITE/_rocm_sdk_libraries/lib:$BASE_SITE/_rocm_sdk_core/lib/host-math/lib:$BASE_SITE/_rocm_sdk_core/lib/rocm_sysdeps/lib:$BASE_SITE/_rocm_sdk_core/lib/core/lib:$BASE_SITE/torch/lib
export VLLM_ROCM_USE_AITER=0 TORCH_BLAS_PREFER_HIPBLASLT=0
# Ignore the inherited editable distribution; never uninstall from BASE_VENV.
if [[ ${INCREMENTAL:-0} == 1 ]]; then
    # An existing editable install can retain its CMake objects across iterations.
    .venv/bin/python setup.py build_ext --inplace \
        --build-temp "$RUN/native-build" > "$RUN/build.log" 2>&1
else
    .venv/bin/python -m pip install -e . --no-build-isolation --no-deps \
        --ignore-installed > "$RUN/build.log" 2>&1
fi
.venv/bin/python - <<'PY'
import torch
import vllm._custom_ops

schemas = [str(schema) for schema in torch._C._jit_get_all_schemas()]
rocm = [schema for schema in schemas if schema.startswith("_rocm_C::")]
assert sum("rdna" in schema for schema in rocm) > 10, rocm
assert any("all_reduce" in schema for schema in rocm), rocm
for name in (
    "mono_k2_offload_rdna2",
    "moe_fixedpoint_to_half_rdna2",
    "rdna2_expert_cache_copy",
):
    assert hasattr(torch.ops._rocm_C, name), name
for name, argument in (
    ("gptq_gemm_rdna2", "deterministic"),
    ("gptq_gemm_rdna2_prefill", "deterministic"),
    ("qsa_expand_indices_rdna2", "canonical_order"),
):
    schema = str(getattr(torch.ops._rocm_C, name).default._schema)
    assert argument in schema, f"Stale extension schema: {schema}"
print(f"Registered {len(rocm)} ROCm ops, including expert-cache kernels")
PY
printf 'Build log: %s/build.log\n' "$RUN"
