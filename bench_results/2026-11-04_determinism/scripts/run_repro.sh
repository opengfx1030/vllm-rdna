#!/usr/bin/env bash
TREE=$HOME/vllm-det-d638
VENV=$HOME/Apps/vllm/venv-7.14.0_0.32.0
SITE=$VENV/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$SITE/_rocm_sdk_libraries/lib:$SITE/_rocm_sdk_core/lib/host-math/lib:$SITE/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SITE/_rocm_sdk_core/lib/core/lib:$SITE/torch/lib
export HIP_VISIBLE_DEVICES=${REPRO_GPU:-2} PYTHONPATH=$TREE
cd "$TREE" && "$VENV/bin/python" "$HOME/w4a8_runs/determinism/repro_moe.py"
