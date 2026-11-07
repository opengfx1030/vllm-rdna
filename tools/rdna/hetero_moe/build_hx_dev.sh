#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Fast development build of the hx ops (one translation unit, about one minute
# instead of a full _rocm_C build): hipify csrc/rocm/hetero_moe/hx_ops.cu, compile
# it for gfx1030 + gfx1100 with the flags of the real _rocm_C target, link it with
# tools/rdna/hetero_moe/hx_dev_register.cpp into hx_dev.so. Load the result with
#   torch.ops.load_library("<out>/hx_dev.so")        (tools/rdna/hetero_moe/hx_dev.py)
# in a process whose vllm._rocm_C predates the hx ops; a _rocm_C that already
# contains them (a full build of this branch) must NOT also load hx_dev.so.
#
# Runs on the box, under the build lock (box rules); nice 19, one job.
#
# Usage: build_hx_dev.sh [--tree DIR] [--venv DIR] [--out DIR] [--no-lock]
set -euo pipefail

TREE="${HOME}/vllm-hx-wip"
VENV="${HOME}/Apps/vllm/venv-7.14.0"
OUT=""
LOCK="${HOME}/w4a8_runs/audit/build.lock"
ROCM=/opt/rocm/core-7.14
ARCHES=(gfx1030 gfx1100)

while [ $# -gt 0 ]; do
  case "$1" in
    --tree) TREE="$2"; shift 2 ;;
    --venv) VENV="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --no-lock) LOCK=""; shift ;;
    -h | --help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[ -f "${TREE}/csrc/rocm/hetero_moe/hx_ops.cu" ] || {
  echo "no hx_ops.cu under ${TREE}" >&2
  exit 1
}
if [ -z "${OUT}" ]; then
  SHA="$(cat "${TREE}/.sha" 2>/dev/null || echo unknown)"
  OUT="${HOME}/w4a8_runs/hx_dev/${SHA}"
fi
mkdir -p "${OUT}"

PY="${VENV}/bin/python"
TP="$("${PY}" -c 'import os, torch; print(os.path.dirname(torch.__file__))')"
PYINC="$("${PY}" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
HIPOUT="${OUT}/hipout"
rm -rf "${HIPOUT}"
(cd "${TREE}" && "${PY}" cmake/hipify.py -p csrc -o "${HIPOUT}" csrc/rocm/hetero_moe/hx_ops.cu) \
  > "${OUT}/hipify.log" 2>&1 || {
  tail -20 "${OUT}/hipify.log" >&2
  exit 1
}
SRC_HIP="$(find "${HIPOUT}" -name hx_ops.hip | head -1)"
REG="${TREE}/tools/rdna/hetero_moe/hx_dev_register.cpp"

arch_flags=()
for a in "${ARCHES[@]}"; do arch_flags+=("--offload-arch=${a}"); done

CXX="${ROCM}/lib/llvm/bin/clang++"
cxxflags=(
  -DHIPBLASLT_USE_ROCROLLER -DPy_LIMITED_API=3 -DTORCH_EXTENSION_NAME=_rocm_C
  -DUSE_C10D_GLOO -DUSE_C10D_NCCL -DUSE_DISTRIBUTED -DUSE_NVSHMEM -DUSE_PROF_API=1
  -DUSE_RPC -DUSE_TENSORPIPE -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1
  -I"${HIPOUT}" -isystem "${PYINC}" -isystem "${TP}/include"
  -isystem "${TP}/include/torch/csrc/api/include"
  -isystem /opt/rocm-7.2.0/core-7.14/include/hiprand
  -isystem /opt/rocm-7.2.0/core-7.14/include/rocrand
  -isystem /opt/rocm-7.2.0/core-7.14/include/hipsparse
  -isystem "${ROCM}/lib/rocm_sysdeps/include" -isystem /usr/include/libdrm
  -Wno-unused-result -Wno-unused-value -O2 -DNDEBUG -std=gnu++20 "${arch_flags[@]}"
  -fPIC -DUSE_ROCM=1 -DHIPBLAS_V2 -DCUDA_HAS_FP16=1
  -D__HIP_NO_HALF_OPERATORS__=1 -D__HIP_NO_HALF_CONVERSIONS__=1
  -DHIP_ENABLE_WARP_SYNC_BUILTINS=1 -DENABLE_FP8 -U__HIP_NO_HALF_CONVERSIONS__
  -U__HIP_NO_HALF_OPERATORS__ -Werror=unused-variable -Wall -fno-gpu-rdc
  -fpermissive -DTORCH_HIP_VERSION=714 -Wno-shift-count-negative
  -Wno-shift-count-overflow -DCAFFE2_USE_MIOPEN
  -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP -x hip
)

run() {
  if [ -n "${LOCK}" ]; then
    flock "${LOCK}" nice -n 19 "$@"
  else
    nice -n 19 "$@"
  fi
}

echo "building hx_dev.so for ${ARCHES[*]} from ${TREE} into ${OUT}"
run "${CXX}" "${cxxflags[@]}" -c "${SRC_HIP}" -o "${OUT}/hx_ops.o"
run "${CXX}" "${cxxflags[@]}" -c "${REG}" -o "${OUT}/hx_dev_register.o"

TL="${TP}/lib"
nice -n 19 "${ROCM}/bin/hipcc_cmake_linker_helper" "${ROCM}/llvm/bin" \
  --hip-link --rtlib=compiler-rt -unwindlib=libgcc \
  "${OUT}/hx_ops.o" "${OUT}/hx_dev_register.o" -o "${OUT}/hx_dev.so" \
  "-Wl,-rpath,${TL}" "${TL}/libc10.so" \
  -Wl,--no-as-needed,"${TL}/libtorch.so" -Wl,--as-needed \
  -Wl,--no-as-needed,"${TL}/libtorch_cpu.so" -Wl,--as-needed \
  -Wl,--no-as-needed,"${TL}/libtorch_hip.so" -Wl,--as-needed \
  "${TL}/libc10_hip.so" -shared
ls -l "${OUT}/hx_dev.so"
echo "load with: torch.ops.load_library('${OUT}/hx_dev.so')"
