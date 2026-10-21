// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// isa_check: HIP public headers before ROCm 6.4 have no __shfl_xor_sync.
// On AMD the sync form is the plain shuffle; the mask is only validated.
#pragma once
#include <hip/hip_runtime.h>
#if HIP_VERSION_MAJOR * 100 + HIP_VERSION_MINOR < 604
template <typename T>
__device__ inline T __shfl_xor_sync(unsigned long long, T var, int lane_mask,
                                    int width = warpSize) {
  return __shfl_xor(var, lane_mask, width);
}
#endif
