#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
TREE=$(cd "$(dirname "$0")/../../.." && pwd)
GPU=${GPU:-6}
case "$GPU" in 6|7) ;; *) echo 'GPU must be HIP 6 or 7.' >&2; exit 2 ;; esac
export HIP_VISIBLE_DEVICES=$GPU
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONPATH=$TREE
SITE=$TREE/.venv/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$SITE/_rocm_sdk_libraries/lib:$SITE/_rocm_sdk_core/lib/host-math/lib:$SITE/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SITE/_rocm_sdk_core/lib/core/lib:$SITE/torch/lib
export VLLM_ROCM_USE_AITER=0 TORCH_BLAS_PREFER_HIPBLASLT=0
export PYTORCH_TUNABLEOP_ENABLED=0 PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0
export HSA_FORCE_FINE_GRAIN_PCIE=1 HSA_ENABLE_SDMA=0 GPU_MAX_HW_QUEUES=4
RUN=${RUN:-$HOME/w4a8_runs/moe-offload}
mkdir -p "$RUN"
cd "$TREE"
.venv/bin/python -c 'import torch; import vllm._custom_ops; assert all(hasattr(torch.ops._rocm_C, name) for name in ("mono_k2_offload_rdna2", "moe_fixedpoint_to_half_rdna2")); print(torch.cuda.get_device_name(0))'
.venv/bin/python -m pytest --noconftest -v \
    tests/model_executor/layers/fused_moe/test_rdna_expert_cache_policy.py \
    tests/model_executor/layers/fused_moe/test_rdna_expert_cache.py \
    tests/models/qwen4_exp/test_amd_mono_rdna2_offload.py \
    tests/models/qwen4_exp/test_offload_graph_refusal.py \
    tests/models/qwen4_exp/test_offload_kv_probe.py \
    tests/quantization/test_rdna2_moe_asymmetric.py \
    "$@" 2>&1 | tee "$RUN/kernel-tests.log"
