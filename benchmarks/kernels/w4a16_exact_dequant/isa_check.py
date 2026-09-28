# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VALU cost of the exact RDNA2 W4A16 dequant, from compiled gfx1030 ISA.

Compiles the real ``csrc/rocm/qdq_4_rdna2.cuh`` and ``q_gemm_rdna2_common.cuh``
twice, with ``VLLM_RDNA2_W4A16_EXACT_DEQUANT`` at 0 and 1, into a micro-kernel
whose loop body is the one of ``gemm_q4_kernel_rdna2``: dequantize one dword
for each of 4 columns, then ``dot22_8_f`` for M rows. Needs only a clang with
the AMDGPU backend; a small shim stands in for <hip/hip_fp16.h>.

    .venv/bin/python -m benchmarks.kernels.w4a16_exact_dequant.isa_check
"""

from __future__ import annotations

import argparse
import collections
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import regex as re

ROOT = Path(__file__).resolve().parents[3]
ROCM_SRC = ROOT / "csrc" / "rocm"
M_COUNTS = (1, 8, 16)

HIP_FP16_SHIM = """#pragma once
#include <cstdint>
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __forceinline__ inline __attribute__((always_inline))
typedef _Float16 half;
typedef _Float16 half2 __attribute__((ext_vector_type(2)));
__device__ inline half __float2half_rn(float f) { return (half)f; }
__device__ inline half __int2half_rn(int i) { return (half)i; }
__device__ inline half __hadd(half a, half b) { return a + b; }
__device__ inline half __hsub(half a, half b) { return a - b; }
__device__ inline half2 __half2half2(half a) { return (half2){a, a}; }
__device__ inline half2 __halves2half2(half a, half b) { return (half2){a, b}; }
__device__ inline half2 __hadd2(half2 a, half2 b) { return a + b; }
__device__ inline half2 __hsub2(half2 a, half2 b) { return a - b; }
__device__ inline half2 __hmul2(half2 a, half2 b) { return a * b; }
__device__ inline half2 __hfma2(half2 a, half2 b, half2 c) {
  return __builtin_elementwise_fma(a, b, c);
}
template <typename T>
__device__ inline T atomicCAS(T* p, T cmp, T val) {
  __atomic_compare_exchange_n(p, &cmp, val, false, __ATOMIC_RELAXED,
                              __ATOMIC_RELAXED);
  return cmp;
}
"""

KERNEL = """#include "q_gemm_rdna2_common.cuh"
using namespace vllm::gptq_rdna2;

template <int M>
__global__ void w4a16_loop(const uint32_t* __restrict__ w,
                           const half* __restrict__ a,
                           const uint32_t* __restrict__ qzeros,
                           const half* __restrict__ scales,
                           float* __restrict__ out, int size_n, int dwords) {
  const int n = (__builtin_amdgcn_workgroup_id_x() * 256 +
                 __builtin_amdgcn_workitem_id_x()) * 4;
  half2 z1z16[4][2], y1y16[4][2];
  refresh_group<4>(0, n, qzeros, scales, size_n, 0, z1z16, y1y16);
  float c[M][4] = {};
  const uint32_t* wp = w + n;
  #pragma unroll 1
  for (int j = 0; j < dwords; ++j) {
    half2 dq[4][4];
    dequant_4bit_8_fp16(wp[0], dq[0], z1z16[0], y1y16[0]);
    dequant_4bit_8_fp16(wp[1], dq[1], z1z16[1], y1y16[1]);
    dequant_4bit_8_fp16(wp[2], dq[2], z1z16[2], y1y16[2]);
    dequant_4bit_8_fp16(wp[3], dq[3], z1z16[3], y1y16[3]);
    wp += size_n;
  #pragma unroll
    for (int m = 0; m < M; ++m) {
  #pragma unroll
      for (int col = 0; col < 4; ++col) {
        c[m][col] += dot22_8_f(dq[col], a + (m * dwords + j) * 8);
      }
    }
  }
  #pragma unroll
  for (int m = 0; m < M; ++m) {
  #pragma unroll
    for (int col = 0; col < 4; ++col) {
      out[m * size_n + n + col] = c[m][col];
    }
  }
}
"""


def find_clang(explicit: str | None) -> str:
    for c in (explicit, shutil.which("clang")):
        if c and Path(c).exists():
            return c
    raise SystemExit("no clang found; pass --clang")


def compile_asm(clang: str, exact: bool) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "hip").mkdir()
        (Path(tmp) / "hip" / "hip_fp16.h").write_text(HIP_FP16_SHIM)
        tu = Path(tmp) / "tu.hip"
        tu.write_text(
            KERNEL
            + "".join(
                f"template __global__ void w4a16_loop<{m}>(const uint32_t*, "
                "const half*, const uint32_t*, const half*, float*, int, int);\n"
                for m in M_COUNTS
            )
        )
        out = Path(tmp) / "tu.s"
        cmd = [clang, "-x", "hip", "--cuda-device-only", "--offload-arch=gfx1030"]
        cmd += ["-nogpulib", "-nogpuinc", "-O3", "-std=c++17", f"-I{tmp}"]
        cmd += [f"-I{ROCM_SRC}", f"-DVLLM_RDNA2_W4A16_EXACT_DEQUANT={int(exact)}"]
        subprocess.run([*cmd, "-S", "-o", str(out), str(tu)], check=True)
        return out.read_text()


def loop_counts(asm: str) -> dict[int, collections.Counter]:
    """VALU mnemonics of the loop block (most v_dot2) per M instantiation."""
    counts: dict[int, collections.Counter] = {}
    kernels = re.split(r"^(_Z\d+w4a16_loopILi(\d+)E\S*):", asm, flags=re.M)
    for i in range(1, len(kernels) - 2, 3):
        m, body = int(kernels[i + 1]), kernels[i + 2]
        blocks = re.split(r"^(?:\.LBB\d+_\d+|; %bb\.\d+):", body, flags=re.M)
        ops = [
            [ln.split()[0] for ln in b.splitlines() if ln.strip().startswith("v_")]
            for b in blocks
        ]
        loop = max(ops, key=lambda b: sum(i.startswith("v_dot2") for i in b))
        counts[m] = collections.Counter(loop)
    return counts


def summarize(c: collections.Counter) -> dict[str, int]:
    pk = {k: v for k, v in c.items() if k.startswith("v_pk_")}
    dot2 = sum(v for k, v in c.items() if k.startswith("v_dot2"))
    return {
        "v_dot2": dot2,
        **pk,
        "other": sum(c.values()) - dot2 - sum(pk.values()),
        "total": sum(c.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--clang", help="clang with the AMDGPU backend")
    args = parser.parse_args()
    clang = find_clang(args.clang)
    runs = {exact: loop_counts(compile_asm(clang, exact)) for exact in (0, 1)}
    print(f"VALU per loop iteration (4 columns x 1 dword), {clang}:\n")
    print("| M | dequant | v_dot2 | packed fp16 | other | total |")
    print("| ---: | --- | ---: | --- | ---: | ---: |")
    for m in M_COUNTS:
        for exact in (0, 1):
            s = summarize(runs[exact][m])
            pk = ", ".join(f"{k} {v}" for k, v in s.items() if k.startswith("v_pk_"))
            name = "exact" if exact else "baked"
            print(
                f"| {m} | {name} | {s['v_dot2']} | {pk} | {s['other']} | {s['total']} |"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
