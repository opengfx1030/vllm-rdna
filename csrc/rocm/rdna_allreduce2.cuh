// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 BlivionIaG
//
// RDNA all-reduce for 2..8 ranks, one reduction kernel with two transports:
//   P2P on : each rank pushes its contribution into every peer's staging area
//   P2P off: each rank pushes into its own pinned host bounce; peers read that
// Both transports fill the same "per-rank slot" layout, so the reduction below
// is identical and there is a single code path to trust.
//
// Grid-wide barrier (the old kernel's defect): every block's thread 0 increments
// arrive; block 0 waits for the full count and then advances seq; EVERY block
// waits for seq to advance before reading peer data. A block-scoped
// __syncthreads() is never used as a grid-wide barrier.
//
// World size is a runtime parameter; nothing here assumes 4.

#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

#define RDNA_AR2_MAX_WORLD 8

struct RdnaAr2Peers {
  void* stage[RDNA_AR2_MAX_WORLD];  // peer j's staging area ([rank] = ours)
};

// arrive[parity] counts blocks; seq[parity] is the published phase sequence.
struct RdnaAr2Sync {
  unsigned int* arrive;         // [2]
  unsigned int* seq;            // [2]
  unsigned long long* report;   // [1] host-mapped, 0 = healthy
};

// Host staging: one buffer of world slots, each `max_elems` long. With P2P on
// this is device memory; with P2P off it is pinned host memory and the peers
// read it over PCIe with an explicit fence.
struct RdStageLayout {
  long long max_elems;  // elements per slot
  int world;
};

__device__ __forceinline__ void rdna_ar2_pause() { __builtin_amdgcn_s_sleep(8); }

__device__ __forceinline__ void rdna_ar2_abort(unsigned long long* report, unsigned phase,
                                         unsigned peer) {
  const unsigned long long code =
      1ull | ((unsigned long long)(phase & 0xFu) << 8) |
      ((unsigned long long)(peer & 0xFu) << 12);
  atomicMax(report, code);
}

// Grid-wide barrier. Precondition: called by exactly one thread per block, and
// every block calls it. Returns false on spin-cap expiry (recorded in report).
__device__ __forceinline__ bool rdna_ar2_barrier(unsigned int* arrive, unsigned int* seq,
                                           unsigned long long* report, unsigned parity,
                                           int nblocks, unsigned long long cap) {
  __shared__ bool s_ok;
  const int t = threadIdx.x;
  const int b = blockIdx.x;
  if (t == 0) {
    __threadfence_system();
    atomicAdd(&arrive[parity], 1u);
    bool ok = true;
    // Block 0 waits for all blocks, then publishes the new sequence.
    if (b == 0) {
      unsigned long long s = 0;
      while (__hip_atomic_load(&arrive[parity], __ATOMIC_ACQUIRE,
                               __HIP_MEMORY_SCOPE_AGENT) < (unsigned)nblocks) {
        rdna_ar2_pause();
        if (++s > cap) { rdna_ar2_abort(report, parity, 0u); ok = false; break; }
      }
      if (ok) {
        const unsigned int next =
            __hip_atomic_load(&seq[parity], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) + 1u;
        __hip_atomic_store(&seq[parity], next, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        arrive[parity] = 0u;
      }
    } else {
      // Every other block waits for the sequence to advance, so it cannot read
      // peer data before the push phase of all blocks has completed.
      const unsigned int before =
          __hip_atomic_load(&seq[parity], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
      unsigned long long s = 0;
      while (__hip_atomic_load(&seq[parity], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) ==
             before) {
        rdna_ar2_pause();
        if (++s > cap) { rdna_ar2_abort(report, parity, 1u); ok = false; break; }
      }
    }
    s_ok = ok;
  }
  __syncthreads();
  return s_ok;
}

// Same barrier for block 0: it published, so it must not wait on its own store.
__device__ __forceinline__ bool rdna_ar2_barrier_leader(unsigned int* arrive, unsigned int* seq,
                                                  unsigned long long* report,
                                                  unsigned parity, int nblocks,
                                                  unsigned long long cap) {
  __shared__ bool s_ok;
  if (threadIdx.x == 0) {
    __threadfence_system();
    atomicAdd(&arrive[parity], 1u);
    bool ok = true;
    unsigned long long s = 0;
    while (__hip_atomic_load(&arrive[parity], __ATOMIC_ACQUIRE,
                             __HIP_MEMORY_SCOPE_AGENT) < (unsigned)nblocks) {
      rdna_ar2_pause();
      if (++s > cap) { rdna_ar2_abort(report, parity, 0u); ok = false; break; }
    }
    if (ok) {
      const unsigned int next =
          __hip_atomic_load(&seq[parity], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) + 1u;
      __hip_atomic_store(&seq[parity], next, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
      arrive[parity] = 0u;
    }
    s_ok = ok;
  }
  __syncthreads();
  return s_ok;
}

// Strided copy over [0, count). Vectorized when 16-byte aligned; the scalar
// remainder always starts at index 0 (not at an offset derived from gid), so
// every element is covered for any nblocks/threads combination.
template <typename T>
__device__ __forceinline__ void rdna_ar2_copy(T* __restrict__ dst, const T* __restrict__ src,
                                        int count, int gid, int gstride) {
  constexpr int kVec = (int)(16 / sizeof(T));
  const bool vec = kVec > 1 && ((uintptr_t)dst % 16u) == 0 && ((uintptr_t)src % 16u) == 0;
  int base = 0;
  if (vec) {
    const int nvec = count / kVec;
    auto* d4 = reinterpret_cast<uint4*>(dst);
    const auto* s4 = reinterpret_cast<const uint4*>(src);
    for (int i = gid; i < nvec; i += gstride) d4[i] = s4[i];
    base = nvec * kVec;
  }
  for (int i = base + gid; i < count; i += gstride) dst[i] = src[i];
}

// All-reduce. `slots[k]` is rank k's contribution as visible to this rank:
//   P2P on : this rank's own staging buffer, slot k (peers pushed into it)
//   P2P off: the pinned host bounce, slot k
// `in` is only used to fill our own slot.
template <typename T>
__global__ void rdna_ar2_reduce_kernel(const T* __restrict__ in, T* __restrict__ out,
                              RdnaAr2Peers peers, RdnaAr2Sync sync,
                              const T* __restrict__ const* slots,
                              int rank, int world, int n, long long max_elems,
                              int nblocks, unsigned long long cap) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int gstride = nblocks * (int)blockDim.x;
  const unsigned parity = 0u;
  const bool leader = (blockIdx.x == 0);

  // 1. push our contribution into every peer's slot `rank`
  for (int k = 1; k < world; k++) {
    const int j = (rank + k) % world;
    T* dst = reinterpret_cast<T*>(peers.stage[j]) + (long long)rank * max_elems;
    rdna_ar2_copy(dst, in, n, gid, gstride);
  }
  __threadfence_system();

  const bool ok = leader ? rdna_ar2_barrier_leader(sync.arrive, sync.seq, sync.report, parity,
                                             nblocks, cap)
                         : rdna_ar2_barrier(sync.arrive, sync.seq, sync.report, parity,
                                      nblocks, cap);
  if (!ok) return;

  // 2. reduce in fixed rank order: identical fp32 accumulation on every rank
  for (int i = gid; i < n; i += gstride) {
    float acc = 0.f;
    const T* s = nullptr;
    for (int k = 0; k < world; k++) {
      s = slots[k];
      acc += (float)s[i];
    }
    out[i] = (T)acc;
  }
}
