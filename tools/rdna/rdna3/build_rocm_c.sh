#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# CPU-only build of the `_rocm_C` extension of one source tree, for one arch list,
# in its own build directory (no GPU is touched). Reproduces the recipe other
# agents use on par1-cs25 (hier_ar/build_rocm.sh) with the arch list as an argument.
#
#   bash tools/rdna/rdna3/build_rocm_c.sh <tree> <arch-list> [jobs] [target]
#   bash tools/rdna/rdna3/build_rocm_c.sh ~/vllm-rdna3-base-27e883f3ce 'gfx1030'
#   bash tools/rdna/rdna3/build_rocm_c.sh ~/vllm-rdna3-1234abc 'gfx1030;gfx1100' 8
#   bash tools/rdna/rdna3/build_rocm_c.sh ~/vllm-rdna3-1234abc 'gfx1030;gfx1100' 8 \
#       CMakeFiles/_rocm_C.dir/csrc/rocm/q_gemm_rdna3.hip.o       # one object only
#
# Rules this script encodes (AGENTS.md "GPU selection on par1-cs25", item 5):
# the shared build lock, <= 8 jobs, nice 19, no /tmp, logs under
# ~/w4a8_runs/rdna3_port/. The build dir is
# ~/w4a8_runs/rdna3_port/build-<tree>-<arches><TAG_SUFFIX>.
#
# Device-code comparison builds (base vs patched): hipcc embeds a per-TU CUID,
# a hash of the source path and of the WHOLE command line, in every code object.
# A different path or flag set changes it, shifts section offsets and rewrites
# PC-relative immediates in kernels whose source did not change. So compare
# builds made in the SAME tree path and build dir: rsync the patched sources
# over the base tree and run this script again; ninja recompiles only the
# changed TUs, with the unchanged command line. (-fuse-cuid=none does not
# work: every TU then defines the same __hip_cuid_ symbol and the link fails.)
#
# Environment:
#   EXTRA_HIP_FLAGS  appended to CMAKE_HIP_FLAGS at configure time.
#   TAG_SUFFIX       keeps differently configured builds of one tree apart.
set -uo pipefail

TREE=${1:?usage: build_rocm_c.sh <tree> <arch-list> [jobs] [target]}
ARCHES=${2:?arch list, e.g. gfx1030 or 'gfx1030;gfx1100'}
JOBS=${3:-8}
TARGET=${4:-_rocm_C}
((JOBS <= 8)) || { echo "refusing JOBS=$JOBS (> 8, box rule)" >&2; exit 2; }

TREE=$(cd "$TREE" && pwd)
VENV=${VENV:-$HOME/Apps/vllm/venv-7.14.0}
ROCM=${ROCM:-/opt/rocm/core-7.14}
LOGDIR=${LOGDIR:-$HOME/w4a8_runs/rdna3_port}
TAG=$(basename "$TREE")-${ARCHES//;/+}${TAG_SUFFIX:-}
BD=$LOGDIR/build-$TAG
ST=$LOGDIR/build-$TAG.status
LOCK=$HOME/w4a8_runs/audit/build.lock
mkdir -p "$LOGDIR" "$BD"

echo "waiting-for-lock $(date +%T)" >"$ST"
exec 9>"$LOCK"
flock 9
echo "running $(date +%T)" >"$ST"

# A CPU-only build never needs a GPU: hide them all.
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export HIP_VISIBLE_DEVICES=
export TRITON_KERNELS_SRC_DIR=${TRITON_KERNELS_SRC_DIR:-$HOME/vllm-rdna-0.32.0-fix/.deps/triton_kernels-src/python/triton_kernels/triton_kernels}
export VLLM_TARGET_DEVICE=rocm PYTORCH_ROCM_ARCH=$ARCHES
export VLLM_PYTHON_EXECUTABLE=$VENV/bin/python
export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-RelWithDebInfo}
export CMAKE_HIP_COMPILER=$ROCM/bin/hipcc
export ROCM_HOME=$ROCM ROCM_PATH=$ROCM HIP_PATH=$ROCM HIP_ROOT_DIR=$ROCM
export CMAKE_HIP_COMPILER_ROCM_ROOT=$ROCM
export HIP_DEVICE_LIB_PATH=$ROCM/lib/llvm/amdgcn/bitcode
export PATH=$VENV/bin:$ROCM/bin:$PATH
SITE=$VENV/lib/python3.12/site-packages
export LD_LIBRARY_PATH="$SITE/_rocm_sdk_libraries/lib:$SITE/_rocm_sdk_core/lib/host-math/lib:$SITE/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SITE/_rocm_sdk_core/lib:$SITE/torch/lib:${LD_LIBRARY_PATH:-}"

cd "$TREE" || exit 1
PYPATH=$("$VENV/bin/python" -c 'import sys; print(":".join(sys.path))')
cd "$BD" || exit 1
if [[ ! -f build.ninja ]]; then
    cmake "$TREE" -G Ninja -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" -DVLLM_TARGET_DEVICE=rocm \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_HIP_COMPILER_LAUNCHER=ccache \
        -DVLLM_PYTHON_EXECUTABLE="$VENV/bin/python" "-DVLLM_PYTHON_PATH=$PYPATH" \
        -DFETCHCONTENT_BASE_DIR="$BD/deps" \
        -DCMAKE_JOB_POOL_COMPILE:STRING=compile "-DCMAKE_JOB_POOLS:STRING=compile=$JOBS" \
        -DROCM_PATH="$ROCM" ${EXTRA_HIP_FLAGS:+"-DCMAKE_HIP_FLAGS=$EXTRA_HIP_FLAGS"} ||
        { echo "configure_exit=1 $(date +%T)" >"$ST"; exit 1; }
fi
nice -n 19 cmake --build . -j "$JOBS" --target "$TARGET"
rc=$?
lc=skipped
if [[ $rc -eq 0 && $TARGET == _rocm_C ]]; then
    # Install next to the tree's python package (same as the editable build).
    cmake --install . --prefix "$TREE" --component _rocm_C
    rc=$?
    if [[ $rc -eq 0 ]]; then
        # dlopen the installed library in a fresh process (CPU only, never vllm):
        # an op registered without its kernel symbol makes the .so unloadable, and
        # vllm then drops every _rocm_C op with only a warning.
        "$VENV/bin/python" "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/rocm_c_load_check.py" \
            check "$TREE/vllm/_rocm_C.abi3.so" --out "$LOGDIR/ops-$TAG.json"
        lc=$?
    fi
fi
echo "build_exit=$rc load_check=$lc $(date +%T)" >"$ST"
exit $rc
