#!/usr/bin/env bash
# Build this tree's native extensions in place (no pip: the shared venv's
# editable install keeps pointing at its own tree; serve with VLLM_TREE / PYTHONPATH).
#   VENV=~/Apps/vllm/venv-7.14.0_0.31.0-final bash tools/rdna/mono/build_tree.sh
set -euo pipefail
TREE=$(cd "$(dirname "$0")/../../.." && pwd)
VENV=${VENV:-$HOME/Apps/vllm/venv-7.14.0_0.31.0-final}
PY=$VENV/bin/python
cd "$TREE"
if [[ ${CLEAN:-0} == 1 ]]; then rm -rf build/ .deps/; fi
export SETUPTOOLS_SCM_PRETEND_VERSION=0.31.0+rdna
export VLLM_TARGET_DEVICE=rocm PYTORCH_ROCM_ARCH=gfx1030
export VLLM_PYTHON_EXECUTABLE=$PY MAX_JOBS=${MAX_JOBS:-32} CMAKE_BUILD_TYPE=RelWithDebInfo
export CMAKE_HIP_COMPILER=/opt/rocm/core-7.14/bin/hipcc
export ROCM_HOME=/opt/rocm/core-7.14 ROCM_PATH=/opt/rocm/core-7.14
export HIP_PATH=/opt/rocm/core-7.14 HIP_ROOT_DIR=/opt/rocm/core-7.14
export CMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm/core-7.14
export PATH=/opt/rocm/core-7.14/bin:$VENV/bin:$PATH
t0=$(date +%s)
"$PY" setup.py build_ext --inplace
echo "build took $(( $(date +%s) - t0 ))s"
ls -la vllm/_rocm_C.abi3.so
