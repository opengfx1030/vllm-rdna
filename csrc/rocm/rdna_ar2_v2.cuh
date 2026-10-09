// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 BlivionIaG
//
// RDNA all-reduce v2 — reduce-scatter + allgather over PCIe P2P.
//
// Byte budget per rank per call, vs the push-all shape:
//   W=4: 1.5*N  vs  3.0*N      W=8: 1.75*N  vs  7.0*N
// RCCL reached 169 GB/s effective at 10 MB, so traffic alone did not explain the
// 5x gap the push-all kernel showed; that kernel also serialized a full push
// behind a full grid barrier. This design removes the barrier: chunk c is reduced
// by rank (c % W) as soon as ITS inputs land, and re-broadcast as soon as it is
// done, so a rank never waits on the whole grid.
//
// Chunk count must be a positive multiple of W so every rank owns CH/W chunks
// (CH=2, W=4 leaves two ranks owning nothing and idling on every call).
//
// Every rank stages its slice of EVERY chunk: the owner of chunk c needs W-1
// remote contributions, and its own slice must be staged too, or the reduction
// silently undercounts by one rank.
//
// Ordering uses per-(sender, chunk, phase) flags in the RECEIVER's memory plus a
// monotonic generation, so repeated calls need no counter reset.

#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

#define RA2_MAX_WORLD 8
#define RA2_MAX_CHUNKS 16

// Flag layout in each rank's own memory. Phase A = reduce-scatter contribution
// arrived; phase B = gathered chunk arrived. Row index is the sender's rank.
#define RA2_ROW_A(w) (0)
#define RA2_ROW_B(w) (w)

struct Ra2Peers {
  void* stage[RA2_MAX_WORLD];  // staging base of peer j; [rank] is our own
  int* flags[RA2_MAX_WORLD];   // flag array of peer j; [rank] is our own
  int owned_slot;              // our row index inside a peer's flag array (== our rank)
};

__device__ __forceinline__ void ra2_pause() { __builtin_amdgcn_s_sleep(8); }

// Grid-stride copy, vectorized when both sides are 16-byte aligned. The scalar
// remainder always starts at index 0, so coverage is complete for any thread
// count (an earlier kernel derived the remainder from gid and skipped elements).
template <typename T>
__device__ __forceinline__ void ra2_copy(T* __restrict__ dst, const T* __restrict__ src,
                                         int count, int gid, int gstride) {
  constexpr int kVec = (int)(16 / sizeof(T));
  int base = 0;
  if (kVec > 1 && ((uintptr_t)dst % 16u) == 0 && ((uintptr_t)src % 16u) == 0) {
    const int nvec = count / kVec;
    auto* d4 = reinterpret_cast<uint4*>(dst);
    const auto* s4 = reinterpret_cast<const uint4*>(src);
    for (int i = gid; i < nvec; i += gstride) d4[i] = s4[i];
    base = nvec * kVec;
  }
  for (int i = base + gid; i < count; i += gstride) dst[i] = src[i];
}

// Poll one flag until it reaches `want`. Bounded, so a peer that dies is reported
// as a wedge marker instead of hanging the engine core forever.
__device__ __forceinline__ bool ra2_wait(int* flag, int want, unsigned long long cap,
                                         unsigned long long* report, unsigned peer,
                                         unsigned chunk, unsigned phase) {
  unsigned long long spins = 0;
  while (__hip_atomic_load(flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < want) {
    ra2_pause();
    if (++spins > cap) {
      if (threadIdx.x == 0 && report != nullptr) {
        const unsigned long long code =
            1ull | ((unsigned long long)(peer & 0xF) << 8) |
            ((unsigned long long)(chunk & 0xFF) << 12) |
            ((unsigned long long)(phase & 0x1) << 20);
        atomicMax(report, code);
      }
      return false;
    }
  }
  return true;
}

// Chunk c is reduced+owned by rank (c % W). Every rank pushes its slice of c to
// the owner's stage row for its own rank, then the owner reduces.
template <typename T>
__global__ void ra2_reduce_scatter(const T* __restrict__ in, T* __restrict__ part,
                                   Ra2Peers peers, int rank, int world, int n,
                                   int chunk_elems, int nchunks, int gen,
                                   unsigned long long cap, unsigned long long* report,
                                   int* fail) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int gstride = (int)(gridDim.x * blockDim.x);

  for (int c = 0; c < nchunks; c++) {
    const int start = c * chunk_elems;
    const int count = min(chunk_elems, n - start);
    if (count <= 0) continue;

    const int owner = c % world;
    const long long slot = (long long)rank * chunk_elems + (long long)c * (long long)n;
    ra2_copy(reinterpret_cast<T*>(peers.stage[owner]) + slot, in + start, count, gid, gstride);
    __threadfence_system();
    __syncthreads();
    if (gid == 0 && owner != rank) {
      __hip_atomic_store(&peers.flags[owner][RA2_ROW_A(world) * RA2_MAX_CHUNKS + c], gen,
                         __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    __syncthreads();

    if (owner == rank) {
      for (int j = 0; j < world; j++) {
        if (j == rank) continue;
        if (threadIdx.x == 0) {
          if (!ra2_wait(&peers.flags[rank][j * RA2_MAX_CHUNKS + c], gen, cap, report,
                        (unsigned)j, (unsigned)c, 0u)) {
            if (fail) *fail = 1;
            return;
          }
        }
        __syncthreads();
      }
      for (int i = gid; i < count; i += gstride) {
        float acc = 0.f;
        for (int j = 0; j < world; j++) {
          const T* sl = reinterpret_cast<const T*>(peers.stage[rank]) +
                        (long long)j * chunk_elems + (long long)c * (long long)n;
          acc += (float)sl[i];
        }
        part[start + i] = (T)acc;
      }
      __threadfence_system();
      __syncthreads();
      if (gid == 0) {
        for (int j = 0; j < world; j++) {
          if (j == rank) continue;
          __hip_atomic_store(&peers.flags[j][RA2_ROW_B(world) * RA2_MAX_CHUNKS + c], gen,
                             __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
      }
    }
    __syncthreads();
  }
}

// Owner of chunk c has flagged phase B; every rank copies the finished chunk out
// of its own stage row for that owner.
template <typename T>
__global__ void ra2_all_gather(T* __restrict__ out, const T* __restrict__ part,
                               Ra2Peers peers, int rank, int world, int n,
                               int chunk_elems, int nchunks, int gen,
                               unsigned long long cap, unsigned long long* report,
                               int* fail) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int gstride = (int)(gridDim.x * blockDim.x);

  for (int c = 0; c < nchunks; c++) {
    if (c % world == rank) continue;
    const int owner = c % world;
    if (threadIdx.x == 0) {
      if (!ra2_wait(&peers.flags[rank][owner * RA2_MAX_CHUNKS + c], gen, cap, report,
                    (unsigned)owner, (unsigned)c, 1u)) {
        if (fail) *fail = 1;
        return;
      }
    }
    __syncthreads();
    const int start = c * chunk_elems;
    const int count = min(chunk_elems, n - start);
    const T* src = reinterpret_cast<const T*>(peers.stage[rank]) +
                   (long long)owner * chunk_elems + (long long)c * (long long)n;
    ra2_copy(out + start, src, count, gid, gstride);
    __syncthreads();
  }
}
