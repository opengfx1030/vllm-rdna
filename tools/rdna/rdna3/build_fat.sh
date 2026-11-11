#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Full in-place build of every native extension of one source tree for
# gfx1030 + gfx1100 (CPU only, no GPU is touched). The tree is meant to be an
# exact-SHA `git archive` (write the SHA to <tree>/.sha); serve or test it with
# PYTHONPATH=<tree> and the venv given here, never through the venv's editable
# install.
#
#   bash tools/rdna/rdna3/build_fat.sh <tree> [jobs<=8] [venv]
#
# Box rules (AGENTS.md): the shared build lock, <= 8 jobs, nice 19, no /tmp,
# logs under ~/w4a8_runs/rdna3_port/. The status line is written to
# ~/w4a8_runs/rdna3_port/fat-<tree>.status as
# `build_exit=<rc> load_check=<rc> <time>`.
set -uo pipefail

TREE=${1:?usage: build_fat.sh <tree> [jobs] [venv]}
JOBS=${2:-8}
VENV=${3:-$HOME/Apps/vllm/venv-7.14.0_0.32.0-fix}
((JOBS <= 8)) || { echo "refusing JOBS=$JOBS (> 8, box rule)" >&2; exit 2; }
TREE=$(cd "$TREE" && pwd)
PY=$VENV/bin/python
[[ -x $PY ]] || { echo "no python at $PY" >&2; exit 2; }

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOGDIR=$HOME/w4a8_runs/rdna3_port
TAG=fat-$(basename "$TREE")
LOG=$LOGDIR/$TAG.log
STATUS=$LOGDIR/$TAG.status
mkdir -p "$LOGDIR/tmp"
export TMPDIR=$LOGDIR/tmp

# CPU only: nothing below may initialise a GPU.
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export HIP_VISIBLE_DEVICES=

export SETUPTOOLS_SCM_PRETEND_VERSION=${SETUPTOOLS_SCM_PRETEND_VERSION:-0.33.0+rdna3}
export VLLM_TARGET_DEVICE=rocm
export PYTORCH_ROCM_ARCH='gfx1030;gfx1100'
export VLLM_PYTHON_EXECUTABLE=$PY MAX_JOBS=$JOBS CMAKE_BUILD_TYPE=RelWithDebInfo
export CMAKE_HIP_COMPILER=/opt/rocm/core-7.14/bin/hipcc
export ROCM_HOME=/opt/rocm/core-7.14 ROCM_PATH=/opt/rocm/core-7.14
export HIP_PATH=/opt/rocm/core-7.14 HIP_ROOT_DIR=/opt/rocm/core-7.14
export CMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm/core-7.14
export PATH=/opt/rocm/core-7.14/bin:$VENV/bin:$PATH
SITE=$VENV/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$SITE/_rocm_sdk_libraries/lib:$SITE/_rocm_sdk_core/lib/host-math/lib:$SITE/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SITE/_rocm_sdk_core/lib:$SITE/torch/lib

cd "$TREE" || exit 2
echo "=== fat build $(date -Is) tree=$TREE venv=$VENV arches=$PYTORCH_ROCM_ARCH jobs=$JOBS" >"$LOG"
t0=$(date +%s)
flock "$HOME/w4a8_runs/audit/build.lock" nice -n 19 "$PY" setup.py build_ext --inplace >>"$LOG" 2>&1
rc=$?
echo "--- build took $(($(date +%s) - t0))s, exit $rc" >>"$LOG"

lc=skipped
if ((rc == 0)); then
    # In a venv with an editable vllm, `import torch` already autoloads vllm and
    # registers its _rocm_C: PYTHONPATH makes that the tree under test, otherwise
    # the second TORCH_LIBRARY registration aborts the process.
    HIP_VISIBLE_DEVICES='' PYTHONPATH="$TREE" "$PY" "$HERE/rocm_c_load_check.py" check \
        "$TREE/vllm/_rocm_C.abi3.so" --out "$LOGDIR/ops-$TAG.json" >>"$LOG" 2>&1
    lc=$?
fi
echo "build_exit=$rc load_check=$lc $(date +%T)" >"$STATUS"
cat "$STATUS"
exit "$rc"
