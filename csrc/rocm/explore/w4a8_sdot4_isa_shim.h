// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Stand-ins for the HIP keywords used by w4a8_sdot4.cuh, so its device code
// can be compiled for gfx1030 by any clang with the AMDGPU backend and no
// ROCm install:
//
//   clang -x hip --cuda-device-only --offload-arch=gfx1030 -nogpulib \
//         -nogpuinc -O3 -S ...
//
// Used only by benchmarks/kernels/w4a8_sdot4_explore/isa_check.py. The real
// build (w4a8_sdot4_capi.cu) includes <hip/hip_runtime.h> instead.

#pragma once

#if !defined(__HIP__)
  #error "compile with -x hip"
#endif

#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __shared__ __attribute__((shared))
#define __forceinline__ inline __attribute__((always_inline))
#define __launch_bounds__(n) __attribute__((amdgpu_flat_work_group_size(1, n)))
