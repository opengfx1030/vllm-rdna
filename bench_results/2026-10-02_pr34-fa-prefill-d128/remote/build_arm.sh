#!/usr/bin/env bash
# PR34 arm builder: apply arm sources, incrementally rebuild _rocm_C via the
# cached cmake/ninja build dir (pip install re-configures and fails on the
# non-standard ROCm 7.14 SDK layout -- see tools/rdna2_028/build_rocm_c_incr.sh).
#   ARM=baseline|patched bash build_arm.sh
set -uo pipefail
ARM=${ARM:?ARM required (baseline|patched)}
T=/home/chenco_adm/vllm-rdna-0.28.0
V=/home/chenco_adm/Apps/vllm/venv-7.14.0_0.28.0
RUN=/home/chenco_adm/w4a8_runs/pr34_validate
SRC=$RUN/src/$ARM
BD=$T/build/temp.linux-x86_64-cpython-312
LOG=$RUN/build_$ARM.log
[ -d "$SRC" ] || { echo "missing $SRC"; exit 1; }
[ -d "$BD" ]   || { echo "missing build dir $BD"; exit 1; }

cp "$SRC/fa_rdna2.cu"         "$T/csrc/rocm/fa_rdna2.cu"
cp "$SRC/rdna_attn.py"        "$T/vllm/v1/attention/backends/rdna_attn.py"
cp "$SRC/fa_rdna2_backend.py" "$T/vllm/v1/attention/ops/fa_rdna2_backend.py"
if [ "$ARM" = patched ]; then
  cp "$SRC/test_fa_rdna2_writer_layout.py" "$T/tests/kernels/attention/test_fa_rdna2_writer_layout.py"
  cp "$SRC/benchmark_fa_rdna2_prefill.py"  "$T/benchmarks/kernels/benchmark_fa_rdna2_prefill.py"
fi

ROCM_SDK_LIB="$V/lib/python3.12/site-packages/_rocm_sdk_libraries/lib"
ROCM_SDK="$V/lib/python3.12/site-packages/_rocm_sdk_core/lib"
export LD_LIBRARY_PATH="$ROCM_SDK_LIB:$ROCM_SDK/host-math/lib:$ROCM_SDK/rocm_sysdeps/lib:$ROCM_SDK/core/lib:$V/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$V/bin:/opt/rocm/core-7.14/bin:$PATH"
export CMAKE_HIP_COMPILER=/opt/rocm/core-7.14/lib/llvm/bin/clang++
export HIP_DEVICE_LIB_PATH=/opt/rocm/core-7.14/lib/llvm/amdgcn/bitcode

RULES="$BD/CMakeFiles/rules.ninja"
if grep -q "/opt/rocm/core-7.14/llvm/bin" "$RULES" 2>/dev/null; then
  sed -i 's#/opt/rocm/core-7.14/llvm/bin#/opt/rocm/core-7.14/lib/llvm/bin#g' "$RULES"
  echo "[build_arm] patched HIP link rule -> lib/llvm/bin" | tee -a "$LOG"
fi

cd "$BD"
echo "[build_arm] start arm=$ARM $(date +%H:%M:%S)" | tee -a "$LOG"
ninja _rocm_C >> "$LOG" 2>&1
rc=$?
echo "BUILD_EXIT=$rc" >> "$LOG"
if [ "$rc" != 0 ]; then
  echo "BUILD_FAILED arm=$ARM"; tail -40 "$LOG"; exit "$rc"
fi
# ninja links into the build dir; vllm/_rocm_C.abi3.so is only refreshed by a
# full `pip install` (which re-configures and fails here), so copy the real
# output to both the arm archive and the runtime location.
cp "$BD/_rocm_C.abi3.so" "$RUN/_rocm_C.$ARM.so"
cp "$BD/_rocm_C.abi3.so" "$T/vllm/_rocm_C.abi3.so"
echo "BUILD_OK arm=$ARM sha=$(sha256sum "$RUN/_rocm_C.$ARM.so" | cut -d' ' -f1)"
