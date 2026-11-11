#!/usr/bin/env bash
# Unit tests of the fix tree on one leased card (HIP 2 by default).
#   setsid nohup bash run_unit.sh > unit.log 2>&1 < /dev/null &
set -uo pipefail
TREE=${TREE:-$HOME/vllm-det-d638}
VENV=$HOME/Apps/vllm/venv-7.14.0_0.32.0
SITE=$VENV/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$SITE/_rocm_sdk_libraries/lib:$SITE/_rocm_sdk_core/lib/host-math/lib:$SITE/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SITE/_rocm_sdk_core/lib/core/lib:$SITE/torch/lib
export HIP_VISIBLE_DEVICES=${UNIT_GPU:-2}
export PYTHONPATH=$TREE
cd "$TREE" || exit 1
echo "tree $(cat .sha) HIP $HIP_VISIBLE_DEVICES"
"$VENV/bin/python" -c 'import torch; print("free GiB", torch.cuda.mem_get_info(0)[0] / 2**30)'
"$VENV/bin/python" -m pytest -q -p no:cacheprovider \
    tests/kernels/quantization/test_rdna2_moe_deterministic.py "$@"
