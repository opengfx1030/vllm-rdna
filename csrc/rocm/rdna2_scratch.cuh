#pragma once
// Grow-only scratch for op-internal workspaces (split-K partials) that never
// leave the op. A CUDA graph records the scratch pointer it ran with, so a
// buffer is never freed when it grows: the old one stays alive for the graphs
// that captured it. Every call runs on the current stream, so sharing one
// buffer between graph replays and eager calls is ordered.

#include <vector>

#include <torch/all.h>

struct Rdna2PersistBuf {
  torch::Tensor buf;
};

// Contiguous view of at least `shape` elements. Contents are whatever the
// previous call left: only for buffers the caller overwrites before reading
// (see rdna2_persist_zeros otherwise).
inline torch::Tensor rdna2_persist_empty(Rdna2PersistBuf& b,
                                         at::IntArrayRef shape,
                                         const torch::TensorOptions& opts) {
  int64_t need = 1;
  for (int64_t s : shape) {
    need *= s;
  }
  const auto st = opts.dtype().toScalarType();
  if (!b.buf.defined() || b.buf.device() != opts.device() ||
      b.buf.scalar_type() != st || b.buf.numel() < need) {
    if (b.buf.defined()) {
      static std::vector<torch::Tensor> retired;
      retired.push_back(b.buf);
    }
    // Allocated through the caching allocator, so this is legal during
    // graph capture (the block then comes from the graph pool and stays
    // owned by this buffer).
    b.buf = torch::empty({need}, opts);
  }
  return b.buf.narrow(0, 0, need).view(shape);
}

// rdna2_persist_empty, zero-filled on every call.
inline torch::Tensor rdna2_persist_zeros(Rdna2PersistBuf& b,
                                         at::IntArrayRef shape,
                                         const torch::TensorOptions& opts) {
  auto view = rdna2_persist_empty(b, shape, opts);
  view.zero_();
  return view;
}
