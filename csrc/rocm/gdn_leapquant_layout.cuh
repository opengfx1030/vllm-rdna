// LeapQuant GDN blob layout shared by gdn_leapquant_rdna2.cu and the
// MonoKernel GDN (mono_gdn_rdna2.cu). Must match LeapQuantLayout.offsets in
// vllm/model_executor/layers/mamba/ops/leapquant_gdn.py.
#pragma once

#include <cstdint>

namespace rdna2_lq {

constexpr int LQ_K = 128;  // head_k_dim
constexpr int LQ_ALIGN = 16;

__host__ __device__ __forceinline__ long lq_align(long n) {
  return (n + LQ_ALIGN - 1) / LQ_ALIGN * LQ_ALIGN;
}

// Byte offsets of the blob sections.
struct LqLayout {
  long z, bscale, csmooth, comp_k, comp_u, rec_k, rec_u, rec_g, header, total;
  int V, r, cap, window, power_iters;

  __host__ __device__ void init(int V_, int r_, int cap_, int window_,
                                int iters_) {
    V = V_;
    r = r_;
    cap = cap_;
    window = window_;
    power_iters = iters_;
    long off = 0;
    z = off;
    off = lq_align(off + (long)V * LQ_K);
    bscale = off;
    off = lq_align(off + 4L * V);
    csmooth = off;
    off = lq_align(off + 4L * LQ_K);
    comp_k = off;
    off = lq_align(off + 2L * r * LQ_K);
    comp_u = off;
    off = lq_align(off + 2L * r * V);
    rec_k = off;
    off = lq_align(off + 2L * cap * LQ_K);
    rec_u = off;
    off = lq_align(off + 2L * cap * V);
    rec_g = off;
    off = lq_align(off + 4L * cap);
    header = off;
    off = lq_align(off + 16);
    total = off;
  }
};

template <typename T>
__device__ __forceinline__ T* lq_ptr(uint8_t* base, long off) {
  return reinterpret_cast<T*>(base + off);
}

// Headers of blocks a request has not written yet hold stale bytes; anything
// outside 0 <= cursor, 0 <= pending, cursor + pending <= cap reads as empty
// so no kernel indexes the record ring out of bounds. Such blocks are either
// overwritten by a prefill store or masked by has_initial_state.
__device__ __forceinline__ bool lq_header_ok(int cursor, int pending,
                                             const LqLayout& L) {
  return cursor >= 0 && pending >= 0 && cursor <= L.cap &&
         pending <= L.cap - cursor;
}

}  // namespace rdna2_lq
