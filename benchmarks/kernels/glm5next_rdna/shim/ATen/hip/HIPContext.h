// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// isa_check stand-in for the ROCm torch header, used only when the torch
// install has no ATen/hip/ (CPU or CUDA wheels). Declares what the launch
// code of csrc/rocm/rdna/glm5next/*.cu touches; the host side is parsed,
// never emitted, in a --cuda-device-only compile.
#pragma once
#include <hip/hip_runtime.h>

namespace c10 {
namespace hip {
struct HIPStreamMasqueradingAsCUDA {
  operator hipStream_t() const { return nullptr; }
  hipStream_t stream() const { return nullptr; }
};
inline HIPStreamMasqueradingAsCUDA getCurrentHIPStreamMasqueradingAsCUDA(
    int = -1) {
  return {};
}
}  // namespace hip
}  // namespace c10

namespace at {
namespace cuda {
using CUDAStream = c10::hip::HIPStreamMasqueradingAsCUDA;
inline CUDAStream getCurrentCUDAStream(int = -1) { return {}; }
inline int current_device() { return 0; }
}  // namespace cuda
namespace hip {
using HIPStream = c10::hip::HIPStreamMasqueradingAsCUDA;
inline HIPStream getCurrentHIPStream(int = -1) { return {}; }
}  // namespace hip
}  // namespace at

#ifndef C10_CUDA_KERNEL_LAUNCH_CHECK
  #define C10_CUDA_KERNEL_LAUNCH_CHECK() ((void)hipGetLastError())
#endif
#ifndef C10_HIP_KERNEL_LAUNCH_CHECK
  #define C10_HIP_KERNEL_LAUNCH_CHECK() ((void)hipGetLastError())
#endif
