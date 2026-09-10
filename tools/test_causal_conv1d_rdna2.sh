#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Shell wrapper for test_causal_conv1d_rdna2.py
# Sets up the venv-7.14.0 environment with correct LD_LIBRARY_PATH
# for the ROCm 7.14.0 SDK, then runs the test on GPU specified by
# HIP_VISIBLE_DEVICES.
#
# Usage:
#   source /home/chenco_adm/Apps/vllm/venv-7.14.0/bin/activate && \
#   HIP_VISIBLE_DEVICES=2 bash tools/test_causal_conv1d_rdna2.sh
#
# Or simply:
#   bash tools/test_causal_conv1d_rdna2.sh
#

set -euo pipefail

VENV_ROOT="/home/chenco_adm/Apps/vllm/venv-7.14.0"
VENV_PYTHON="${VENV_ROOT}/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# GPU selection — must be set before launching Python
GPU="${HIP_VISIBLE_DEVICES:-2}"
export HIP_VISIBLE_DEVICES="${GPU}"

# LD_LIBRARY_PATH for ROCm 7.14.0 SDK (venv-7.14.0 quirks, AGENTS.md)
_VENV_SITEPACKAGES="${VENV_ROOT}/lib/python3.12/site-packages"
_ROCM_SDK_CORE="${_VENV_SITEPACKAGES}/_rocm_sdk_core/lib"
_ROCM_SDK_LIBS="${_VENV_SITEPACKAGES}/_rocm_sdk_libraries/lib"
_TORCH_LIB="${_VENV_SITEPACKAGES}/torch/lib"

# Build LD_LIBRARY_PATH: rocm_sdk_libraries FIRST (bundled rocblas 5.5.0),
# then host-math, rocm_sysdeps, core, and torch/lib.
# _rocm_sdk_libraries must come before /opt/rocm/lib to avoid loading the
# older system rocblas 5.2.0 which breaks TunableOp's ROCBLAS_VERSION validator.
export LD_LIBRARY_PATH="${_ROCM_SDK_LIBS}:${_ROCM_SDK_CORE}/host-math/lib:${_ROCM_SDK_CORE}/rocm_sysdeps/lib:${_ROCM_SDK_CORE}/core/lib:${_TORCH_LIB}:${LD_LIBRARY_PATH:-}"

echo "[test_causal_conv1d_rdna2.sh] GPU: ${GPU}"
echo "[test_causal_conv1d_rdna2.sh] LD_LIBRARY_PATH (first 3 entries):"
echo "  ${LD_LIBRARY_PATH%%:*}"
echo "  ${LD_LIBRARY_PATH#*:}"
echo "  ${LD_LIBRARY_PATH#*:*:}"

exec "${VENV_PYTHON}" "${SCRIPT_DIR}/test_causal_conv1d_rdna2.py" "$@"
