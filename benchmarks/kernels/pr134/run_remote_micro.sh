#!/usr/bin/env bash
# Private PR134 microbench entry point on par1-cs25. No serving or installs.
set -euo pipefail

readonly source_dir=/home/chenco_adm/vllm-pr-int4fp16
readonly runtime=/home/chenco_adm/Apps/vllm/venv-pr134-int4fp16
readonly packages=$runtime/lib/python3.12/site-packages
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-2,3,4,5}
unset CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
export VLLM_ROCM_USE_AITER=0
export TORCH_BLAS_PREFER_HIPBLASLT=0
export PYTORCH_TUNABLEOP_ENABLED=0
export PYTORCH_TUNABLEOP_TUNING=0
export ROCPROFILER_REGISTER_ENABLED=0
export GPU_MAX_HW_QUEUES=2
export TRITON_CACHE_DIR=/home/chenco_adm/.cache/pr134/triton
export TORCHINDUCTOR_CACHE_DIR=/home/chenco_adm/.cache/pr134/inductor
export TORCH_EXTENSIONS_DIR=/home/chenco_adm/.cache/pr134/extensions
export LD_LIBRARY_PATH=$packages/_rocm_sdk_libraries/lib:$packages/_rocm_sdk_core/lib/host-math/lib:$packages/_rocm_sdk_core/lib/rocm_sysdeps/lib:$packages/_rocm_sdk_core/lib/core/lib:$packages/torch/lib
export ROCM_HOME=/opt/rocm/core-7.14
export ROCM_PATH=$ROCM_HOME
export HIP_PATH=$ROCM_HOME
export PATH=$runtime/bin:$ROCM_HOME/bin:$PATH
cd "$source_dir"
printf 'boot=%s\n' "$(cat /proc/sys/kernel/random/boot_id)"
printf 'HIP_VISIBLE_DEVICES=%s\n' "$HIP_VISIBLE_DEVICES"
printf 'command:'
printf ' %q' "$runtime/bin/python" "$@"
printf '\n'
exec "$runtime/bin/python" "$@"
