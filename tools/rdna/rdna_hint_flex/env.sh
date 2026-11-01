#!/usr/bin/env bash
# Environment for running this tree (built in place with
# tools/rdna/mono/build_tree.sh) against the shared venv without touching its
# editable install: PYTHONPATH points at this tree.
#   source tools/rdna/rdna_hint_flex/env.sh <hip_device>
TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
VENV=${VENV:-$HOME/Apps/vllm/venv-7.14.0_0.31.0-final}
SP=$VENV/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$SP/_rocm_sdk_libraries/lib:$SP/_rocm_sdk_core/lib/host-math/lib:$SP/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SP/_rocm_sdk_core/lib/core/lib:$SP/torch/lib
export PYTHONPATH=$TREE
export HIP_VISIBLE_DEVICES=${1:?usage: source env.sh <hip_device(s)>}
export VLLM_ROCM_USE_AITER=0 VLLM_RDNA_FORCE_FP16=1 TORCH_BLAS_PREFER_HIPBLASLT=0
export PYTORCH_TUNABLEOP_ENABLED=0 VLLM_WORKER_MULTIPROC_METHOD=spawn GPU_MAX_HW_QUEUES=2
PY=$VENV/bin/python
# Refuse a visible set that is not a V620 (amdsmi sees no GPUs in this venv, so
# vLLM falls back to device 0's arch).
"$PY" -c "import torch; p=torch.cuda.get_device_properties(0); assert p.gcnArchName.startswith('gfx1030'), p.gcnArchName; print('device', p.gcnArchName, 'bus', hex(p.pci_bus_id))"
