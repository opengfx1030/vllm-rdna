#pragma once
// Immortal HIP allocations that never return to the PyTorch caching
// allocator. Mixed 16k prefill torch::zeros({2048,N}) churn recycles
// FULL-graph pages on gfx1030 (c=1 16k PASSes; c=4 then ducts).

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>

// Process-wide, one definition in torch_bindings.cpp. hipStreamIsCapturing
// is a false positive on gfx1030 during eager mixed 16k (and graph replay),
// which made persist_zeros write the FULL capture slot and poison decode.
extern std::atomic<int> g_rdna2_graph_capturing;
extern std::atomic<int> g_rdna2_capture_frozen;

inline bool rdna2_stream_is_capturing() {
  if (g_rdna2_capture_frozen.load(std::memory_order_acquire)) {
    return false;
  }
  return g_rdna2_graph_capturing.load(std::memory_order_acquire) != 0;
}

inline torch::Tensor rdna2_keep_if_capturing(torch::Tensor t) {
  if (t.defined() && rdna2_stream_is_capturing()) {
    static std::vector<torch::Tensor> keep;
    keep.push_back(t);
  }
  return t;
}

// hipMalloc + from_blob with a no-op deleter. Never hipFree: growing a
// persist buffer must not unmap a data_ptr baked into a FULL graph.
inline torch::Tensor rdna2_immortal_zeros(at::IntArrayRef shape,
                                          const torch::TensorOptions& opts) {
  static std::vector<void*> ptrs;
  int64_t numel = 1;
  for (int64_t s : shape) {
    numel *= s;
  }
  const auto st = opts.dtype().toScalarType();
  const size_t nbytes =
      static_cast<size_t>(numel) * at::elementSize(st);
  void* ptr = nullptr;
  TORCH_CHECK(hipMalloc(&ptr, nbytes) == hipSuccess,
              "rdna2_immortal_zeros hipMalloc failed");
  TORCH_CHECK(hipMemset(ptr, 0, nbytes) == hipSuccess,
              "rdna2_immortal_zeros hipMemset failed");
  ptrs.push_back(ptr);
  auto t = torch::from_blob(ptr, shape, [](void*) {}, opts);
  TORCH_CHECK(t.is_contiguous());
  return t;
}

inline torch::Tensor rdna2_keep_always(torch::Tensor t) {
  static std::vector<torch::Tensor> keep;
  if (t.defined()) {
    keep.push_back(t);
  }
  return t;
}

struct Rdna2PersistBuf {
  // Capture and eager MUST be distinct storages. Growing a single buffer
  // for 16k prefill (even with a graveyard) still lets the caching
  // allocator recycle the FULL-graph data_ptr on gfx1030.
  torch::Tensor capture;
  torch::Tensor eager;
};

// 1D numel persist. Prefix view stays contiguous. Capture slot is frozen
// after graph capture; eager 16k grows a different tensor.
inline torch::Tensor rdna2_persist_zeros(Rdna2PersistBuf& buf,
                                         at::IntArrayRef shape,
                                         const torch::TensorOptions& opts) {
  int64_t need = 1;
  for (int64_t s : shape) {
    need *= s;
  }
  const auto st = opts.dtype().toScalarType();
  const bool capturing = rdna2_stream_is_capturing();
  torch::Tensor& slot = capturing ? buf.capture : buf.eager;
  bool grow = !slot.defined() || slot.device() != opts.device() ||
              slot.scalar_type() != st || slot.numel() < need;
  if (grow) {
    int64_t grown = need;
    if (slot.defined() && slot.device() == opts.device() &&
        slot.scalar_type() == st) {
      grown = std::max(grown, slot.numel());
    }
    if (slot.defined()) {
      static std::vector<torch::Tensor> graveyard;
      graveyard.push_back(slot);
    }
    if (capturing) {
      // hipMalloc is illegal during CUDA graph capture.
      slot = torch::zeros({grown}, opts);
      rdna2_keep_if_capturing(slot);
    } else {
      // After FULL freeze only, pin prefill token dim (>=256, <2048) to
      // 2048 so 784/1024/1568 never realloc. Do not pin during piecewise
      // capture: d0=8 → 2048 grew 384 MiB into the graph (serve28 1k
      // c=8 0/8 duct). Capture sizes are 1/2/4/8/16.
      if (g_rdna2_capture_frozen.load(std::memory_order_acquire) &&
          need > 65536 && shape.size() >= 2) {
        const int64_t d0 = shape[0];
        if (d0 >= 256 && d0 < 2048 && need % d0 == 0) {
          grown = std::max(grown, (need / d0) * 2048);
        }
      }
      std::fprintf(stderr,
                   "[rdna2_persist] eager grow need=%lld grown=%lld bytes=%lld\n",
                   static_cast<long long>(need),
                   static_cast<long long>(grown),
                   static_cast<long long>(grown) *
                       at::elementSize(st));
      slot = torch::zeros({grown}, opts);
    }
    rdna2_keep_always(slot);
  }
  auto view = slot.reshape(-1).narrow(0, 0, need).view(shape);
  TORCH_CHECK(view.is_contiguous(), "rdna2 persist view must be contiguous");
  view.zero_();
  return view;
}
