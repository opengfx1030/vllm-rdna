// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// isa_check stand-in for c10/hip/HIPGuard.h (see ATen/hip/HIPContext.h).
#pragma once
#include <c10/core/Device.h>

#include <optional>

namespace at {
namespace cuda {
struct OptionalCUDAGuard {
  explicit OptionalCUDAGuard(std::optional<c10::Device>) {}
  explicit OptionalCUDAGuard(c10::Device) {}
};
struct CUDAGuard {
  explicit CUDAGuard(c10::Device) {}
  explicit CUDAGuard(int) {}
};
}  // namespace cuda
}  // namespace at
