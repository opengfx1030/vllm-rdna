// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Host-side planning for the six persistent SoA expert-cache payloads.
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>

#include <array>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <vector>

namespace {
void check_hip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    // Consume sticky state only after a reported failure; retain the original
    // error for diagnostics. The caller must drain any already queued copies.
    (void)hipGetLastError();
    TORCH_CHECK(false, operation, ": ", hipGetErrorString(error), " (", error,
                ")");
  }
}

size_t checked_extent(const torch::Tensor& tensor) {
  const auto elements = static_cast<size_t>(tensor.numel());
  const auto element_size = tensor.element_size();
  TORCH_CHECK(elements <= std::numeric_limits<size_t>::max() / element_size,
              "expert cache tensor byte extent overflow");
  const size_t bytes = elements * element_size;
  const auto offset = static_cast<size_t>(tensor.storage_offset());
  TORCH_CHECK(offset <= std::numeric_limits<size_t>::max() / element_size,
              "expert cache storage offset overflow");
  const size_t offset_bytes = offset * element_size;
  const size_t storage_bytes = tensor.storage().nbytes();
  TORCH_CHECK(
      offset_bytes <= storage_bytes && bytes <= storage_bytes - offset_bytes,
      "expert cache tensor exceeds its storage");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) <=
                  std::numeric_limits<uintptr_t>::max() - bytes,
              "expert cache pointer extent overflow");
  return bytes;
}
}  // namespace

bool rdna2_expert_cache_copy(const std::vector<torch::Tensor>& sources,
                             const std::vector<torch::Tensor>& destinations,
                             const std::vector<int64_t>& experts,
                             const std::vector<int64_t>& slots) {
  constexpr size_t kPayloads = 6;
  TORCH_CHECK(sources.size() == kPayloads && destinations.size() == kPayloads,
              "expert cache copy requires exactly six payload tensors");
  TORCH_CHECK(experts.size() == slots.size(),
              "expert cache expert/slot counts must match");
  TORCH_CHECK(destinations[0].defined() && destinations[0].is_cuda(),
              "expert cache destinations must be HIP tensors");
  const auto device = destinations[0].device();
  const c10::cuda::CUDAGuard guard(device);
  const hipStream_t stream = at::cuda::getCurrentCUDAStream(device.index());
  hipStreamCaptureStatus capture;
  check_hip(hipStreamIsCapturing(stream, &capture), "hipStreamIsCapturing");
  TORCH_CHECK(capture == hipStreamCaptureStatusNone,
              "expert cache copy planning is forbidden during graph capture");
  const auto* properties = at::cuda::getDeviceProperties(device.index());
  TORCH_CHECK(std::strncmp(properties->gcnArchName, "gfx103", 6) == 0,
              "expert cache native copy requires RDNA2 (gfx103x)");

  std::array<size_t, kPayloads> row_bytes;
  int64_t expert_count = -1;
  int64_t capacity = -1;
  for (size_t p = 0; p < kPayloads; ++p) {
    const auto& source = sources[p];
    const auto& destination = destinations[p];
    TORCH_CHECK(
        source.defined() && source.device().is_cpu() &&
            source.layout() == torch::kStrided && source.dim() >= 1 &&
            source.numel() > 0 && source.is_contiguous() && source.is_pinned(),
        "expert cache sources must be nonempty contiguous pinned CPU tensors");
    TORCH_CHECK(
        destination.defined() && destination.device() == device &&
            destination.layout() == torch::kStrided &&
            destination.dim() == source.dim() && destination.numel() > 0 &&
            destination.is_contiguous() &&
            destination.scalar_type() == source.scalar_type(),
        "expert cache destinations must be matching contiguous HIP tensors");
    TORCH_CHECK(source.sizes().slice(1) == destination.sizes().slice(1),
                "expert cache per-row dimensions must match");
    if (p == 0) {
      expert_count = source.size(0);
      capacity = destination.size(0);
    }
    TORCH_CHECK(
        source.size(0) == expert_count && destination.size(0) == capacity,
        "expert cache expert counts/capacities must agree");
    const size_t source_bytes = checked_extent(source);
    const size_t destination_bytes = checked_extent(destination);
    row_bytes[p] = source_bytes / static_cast<size_t>(expert_count);
    TORCH_CHECK(
        destination_bytes / static_cast<size_t>(capacity) == row_bytes[p],
        "expert cache row byte extents must match");
  }
  std::vector<bool> used_slots(static_cast<size_t>(capacity), false);
  for (size_t i = 0; i < experts.size(); ++i) {
    TORCH_CHECK(experts[i] >= 0 && experts[i] < expert_count && slots[i] >= 0 &&
                    slots[i] < capacity,
                "expert cache expert/slot index out of range");
    TORCH_CHECK(!used_slots[slots[i]],
                "expert cache destination slots must be unique");
    used_slots[slots[i]] = true;
  }
  TORCH_CHECK(experts.size() <= std::numeric_limits<size_t>::max() / kPayloads,
              "expert cache copy count overflow");
  const size_t count = experts.size() * kPayloads;
  if (count == 0) return false;
  std::vector<void*> dsts(count), srcs(count);
  std::vector<size_t> sizes(count);
  for (size_t i = 0; i < experts.size(); ++i) {
    for (size_t p = 0; p < kPayloads; ++p) {
      const size_t j = i * kPayloads + p;
      sizes[j] = row_bytes[p];
      srcs[j] = static_cast<char*>(sources[p].data_ptr()) +
                static_cast<size_t>(experts[i]) * sizes[j];
      dsts[j] = static_cast<char*>(destinations[p].data_ptr()) +
                static_cast<size_t>(slots[i]) * sizes[j];
    }
  }
#if HIP_VERSION_MAJOR > 7 || (HIP_VERSION_MAJOR == 7 && HIP_VERSION_MINOR >= 14)
  // Resolve against the already loaded runtime; never load a second HIP DSO.
  using BatchCopy = decltype(&hipMemcpyBatchAsync);
  static const auto batch =
      reinterpret_cast<BatchCopy>(dlsym(RTLD_DEFAULT, "hipMemcpyBatchAsync"));
  if (batch != nullptr) {
    size_t fail_index = std::numeric_limits<size_t>::max();
    const auto error = batch(dsts.data(), srcs.data(), sizes.data(), count,
                             nullptr, nullptr, 0, &fail_index, stream);
    if (error != hipSuccess) {
      (void)hipGetLastError();
      // A batch may have submitted a prefix: never retry it as individual
      // copies.
      TORCH_CHECK(false, "hipMemcpyBatchAsync: ", hipGetErrorString(error),
                  " (", error, "), failIdx=", fail_index);
    }
    return true;
  }
#endif
  for (size_t j = 0; j < count; ++j) {
    check_hip(hipMemcpyAsync(dsts[j], srcs[j], sizes[j], hipMemcpyHostToDevice,
                             stream),
              "hipMemcpyAsync expert cache fallback");
  }
  return false;
}
