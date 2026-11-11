#!/usr/bin/env bash
# Isolated _rocm_C build only; never installs packages or touches shared trees.
set -euo pipefail
readonly tree=/home/chenco_adm/vllm-pr-int4fp16
readonly runtime=/home/chenco_adm/Apps/vllm/venv-pr134-int4fp16
readonly packages=$runtime/lib/python3.12/site-packages
readonly build=$tree/build/pr134-core
readonly evidence=/home/chenco_adm/pr134-results/core-integration
[[ ${HIP_VISIBLE_DEVICES:-} == 2,3,4,5 ]] || exit 2
unset CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
export ROCM_HOME=/opt/rocm/core-7.14 ROCM_PATH=/opt/rocm/core-7.14
export HIP_PATH=$ROCM_HOME HIP_ROOT_DIR=$ROCM_HOME
export PATH=$runtime/bin:$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$packages/_rocm_sdk_libraries/lib:$packages/_rocm_sdk_core/lib/host-math/lib:$packages/_rocm_sdk_core/lib/rocm_sysdeps/lib:$packages/_rocm_sdk_core/lib/core/lib:$packages/torch/lib
export PYTORCH_ROCM_ARCH=gfx1030 VLLM_TARGET_DEVICE=rocm
export TORCH_BLAS_PREFER_HIPBLASLT=0 VLLM_ROCM_USE_AITER=0
export PYTORCH_TUNABLEOP_ENABLED=0 ROCPROFILER_REGISTER_ENABLED=0
export GPU_MAX_HW_QUEUES=2
# Copy only the already-frozen dependency's Python sources, excluding its Git
# history. CMake points at this private copy; no FetchContent network operation.
readonly frozen_triton=/home/chenco_adm/vllm-rdna-mtp-mono/.deps/triton_kernels-src/python/triton_kernels/triton_kernels
export TRITON_KERNELS_SRC_DIR=$tree/.deps/pr134-triton-kernels
mkdir -p "$evidence" "$TRITON_KERNELS_SRC_DIR" "$build"
rsync -a --exclude='__pycache__/' --exclude='*.pyc' "$frozen_triton/" "$TRITON_KERNELS_SRC_DIR/"
cat /proc/sys/kernel/random/boot_id > "$evidence/boot-id.before"
sha256sum "$tree"/csrc/rocm/{fa_rdna2.cu,int4_fp16_kv_cache.cu,int4_kv_layout_common.cuh,torch_bindings.cpp,ops.h} \
  "$tree/CMakeLists.txt" > "$evidence/source.before.sha256"
test -e "$evidence/rocm-original.so" || cp "$tree/vllm/_rocm_C.abi3.so" "$evidence/rocm-original.so"
sha256sum "$evidence/rocm-original.so" > "$evidence/original-binary.sha256"
cd "$tree"
cmake -S "$tree" -B "$build" -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DVLLM_TARGET_DEVICE=rocm \
  -DROCM_PATH="$ROCM_HOME" \
  -DVLLM_PYTHON_EXECUTABLE="$runtime/bin/python" \
  -DCMAKE_HIP_COMPILER="$ROCM_HOME/lib/llvm/bin/clang++" \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT="$ROCM_HOME" \
  -DCMAKE_HIP_ARCHITECTURES=gfx1030 \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$build/lib" \
  -DFETCHCONTENT_BASE_DIR="$tree/.deps" \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
  > "$evidence/configure.log" 2>&1
cmake --build "$build" --target _rocm_C -j4 > "$evidence/build.log" 2>&1
sha256sum -c "$evidence/source.before.sha256"
cmp "$evidence/boot-id.before" /proc/sys/kernel/random/boot_id
test -s "$build/lib/_rocm_C.abi3.so"
# Both model arms use this same registered core binary. Keep the prior copy
# above for recovery; replacing only our private source is intentional.
cp "$build/lib/_rocm_C.abi3.so" "$tree/vllm/_rocm_C.abi3.so"
sha256sum "$tree/vllm/_rocm_C.abi3.so" > "$evidence/installed-binary.sha256"
printf 'private core build PASS\n'
