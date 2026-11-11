#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# CPU-only pytest for a tree on a box whose GPUs must stay asleep.
#
# Importing vllm.platforms calls amdsmi (init, the handle list, the asic info of a
# handle; HIP_VISIBLE_DEVICES does not hide the GPUs from it) and can wake the cards,
# so the amdsmi module is blocked before pytest starts and no GPU is made visible: the
# platform resolves to CPU and every ROCm arch flag in vllm/platforms/rocm.py is
# False. The arch-dispatch tests fake the architecture themselves.
#
# aiter, imported through flash_attn by the Qwen4Exp AMD modules, runs rocminfo at
# import (GPU_ARCHS does not stop it). The real rocminfo enumerates every GPU of the
# box and aiter raises for an arch it does not list: the gfx1030 of a V620-only box,
# where 35 dispatch tests failed. cpu_stubs/rocminfo, first on PATH, reports one
# gfx1100 agent instead and asks no GPU.
#
#   TREE=$HOME/some-vllm-tree tools/rdna/rdna3/cpu_pytest.sh -q tests/platforms/test_rdna3_dispatch.py
#
# TREE   source tree to import vllm from (required; the venv's editable install
#        points elsewhere). PYTHON overrides the interpreter.
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TREE=${TREE:?TREE=<source tree> is required}
PYTHON=${PYTHON:-$HOME/Apps/vllm/venv-7.14.0/bin/python}

export PYTHONPATH=$TREE
export PATH=$HERE/cpu_stubs:$PATH
exec env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
    "$PYTHON" -c 'import sys; sys.modules["amdsmi"] = None; import pytest; sys.exit(pytest.main(sys.argv[1:]))' "$@"
