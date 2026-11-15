#!/usr/bin/env bash
# Build only the _rocm_C extension of a tree, incrementally, and install it into
# <tree>/vllm/. The cmake build directory persists between runs, so after the
# first (full) build ninja recompiles just the translation units whose sources
# changed (rsync keeps mtimes: touch nothing else). One compile job pool of at
# most 8 under the shared CPU-build lock, nice 19 (AGENTS.md, "CPU builds").
#
#   TREE=~/w4a8_runs/leapquant/tree-hi2g BLD=~/w4a8_runs/hi2-gemm2/build \
#     bash tools/rdna/rdna_hint_flex/gemm_sweep/build_rocm_c.sh
#
# TREE needs its .deps (FetchContent cache, copied from another tree) so that
# configure needs no network. VENV, JOBS, LOCK override the defaults.
set -euo pipefail
TREE=${TREE:?TREE=<source tree>}
BLD=${BLD:?BLD=<persistent cmake build dir>}
VENV=${VENV:-$HOME/Apps/vllm/venv-7.14.0_0.31.0-final}
JOBS=${JOBS:-8}
LOCK=${LOCK:-$HOME/w4a8_runs/audit/build.lock}
PY=$VENV/bin/python
SP=$VENV/lib/python3.12/site-packages
export VLLM_TARGET_DEVICE=rocm PYTORCH_ROCM_ARCH=gfx1030
export CMAKE_HIP_COMPILER=/opt/rocm/core-7.14/bin/hipcc
export ROCM_HOME=/opt/rocm/core-7.14 ROCM_PATH=/opt/rocm/core-7.14
export HIP_PATH=/opt/rocm/core-7.14 HIP_ROOT_DIR=/opt/rocm/core-7.14
export CMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm/core-7.14
export PATH=/opt/rocm/core-7.14/bin:$VENV/bin:$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$SP/_rocm_sdk_libraries/lib:$SP/_rocm_sdk_core/lib/host-math/lib:$SP/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SP/_rocm_sdk_core/lib:$SP/torch/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$TREE
mkdir -p "$BLD"
cd "$BLD"
t0=$(date +%s)
if [[ ! -f build.ninja ]]; then
  PYPATH=$(cd "$TREE" && "$PY" -c 'import sys; print(":".join(sys.path))')
  cmake "$TREE" -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DVLLM_TARGET_DEVICE=rocm \
    -DVLLM_PYTHON_EXECUTABLE="$PY" -DVLLM_PYTHON_PATH="$PYPATH" \
    -DFETCHCONTENT_BASE_DIR="$TREE/.deps" \
    -DCMAKE_JOB_POOL_COMPILE:STRING=compile -DCMAKE_JOB_POOLS:STRING="compile=$JOBS" \
    -DROCM_PATH=$ROCM_PATH
fi
flock "$LOCK" nice -n 19 cmake --build . -j "$JOBS" --target _rocm_C
cmake --install . --prefix "$TREE" --component _rocm_C
echo "build_exit=0 after $(( $(date +%s) - t0 ))s"
ls -la "$TREE/vllm/_rocm_C.abi3.so"
