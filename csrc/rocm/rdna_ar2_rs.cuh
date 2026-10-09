// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 BlivionIaG
//
// Kernels for the reduce-scatter + allgather all-reduce. See rdna_ar2_rs.cu for
// the protocol and the byte accounting.
//
// Each peer's staging buffer holds two regions, kept apart so a broadcast can
// never overwrite a contribution another rank is about to read:
//   contrib[sender][chunk]  incoming slices, written by every rank
//   result[chunk]           the reduced chunk, written only by its owner
//
// Staging the owner's OWN slice is required: the owner sums W inputs, and
// skipping its own would leave a silent zero in the sum.
//
// Ordering is per-(sender, chunk, phase) flags living in the RECEIVER's memory,
// compared against a per-call generation. No device-wide barrier, and no counter
// reset between calls.

#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

#define RDNA_ARS_MAX_WORLD 8
#define RDNA_ARS_MAX_CHUNKS 16

struct RsPeers {
  void* stage[RDNA_ARS_MAX_WORLD];  // staging base per peer; [rank] is ours
  int* flags[RDNA_ARS_MAX_WORLD];   // flag array per peer; [rank] is ours
  int row_a;                        // row index holding phase-A flags
  int row_b;                        // row index holding phase-B flags
  int rank;                         // this rank's index
};

__device__ __forceinline__ void rdna_ars_pause() { __builtin_amdgcn_s_sleep(8); }

__device__ __forceinline__ long long rs_contrib_slot(int sender, int chunks, int c,
                                                     int chunk_elems) {
  return ((long long)sender * chunks + c) * (long long)chunk_elems;
}

__device__ __forceinline__ long long rs_result_off(int world, int chunks, int chunk_elems) {
  return (long long)world * chunks * chunk_elems;
}

// Vectorized when both sides are 16-byte aligned; the scalar remainder always
// starts at index 0, so every element is covered for any thread count.
template <typename T>
__device__ __forceinline__ void rdna_ars_copy(T* __restrict__ dst, const T* __restrict__ src,
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

// One thread polls until the flag reaches `gen`. Bounded, so a dead peer is
// reported instead of hanging the engine core.
__device__ __forceinline__ bool rdna_ars_wait(int* flag, int gen, unsigned long long cap) {
  unsigned long long spins = 0;
  while (__hip_atomic_load(flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < gen) {
    rdna_ars_pause();
    if (++spins > cap) return false;
  }
  return true;
}

// Phase A + reduce, chunk by chunk. Every rank stages its slice of EVERY chunk
// to that chunk's owner; the owner then waits for W-1 flags, sums W inputs and
// publishes the result to every peer.
template <typename T>
__global__ void rdna_ars_scatter_kernel(const T* __restrict__ in, RsPeers peers, int world,
                                        int chunks, int chunk_elems, int n, int gen,
                                        unsigned long long cap, int* fail) {
  const int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int gstride = (int)(gridDim.x * blockDim.x);
  const int rank = peers.rank;
  const int flag_row = RDNA_ARS_MAX_WORLD * RDNA_ARS_MAX_CHUNKS;

  for (int c = 0; c < chunks; c++) {
    const int start = c * chunk_elems;
    const int count = min(chunk_elems, n - start);
    if (count <= 0) continue;

    const int owner = c % world;
    const long long cslot = rs_contrib_slot(rank, chunks, c, chunk_elems);

    rdna_ars_copy(reinterpret_cast<T*>(peers.stage[owner]) + cslot, in + start, count, gid,
                  gstride);
    __threadfence_system();
    __syncthreads();
    if (rank != owner && blockIdx.x == 0 && threadIdx.x == 0) {
      __hip_atomic_store(&peers.flags[owner][peers.row_a * flag_row +
                                             rank * RDNA_ARS_MAX_CHUNKS + c],
                         gen, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    __syncthreads();

    if (owner == rank) {
      if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int j = 0; j < world; j++) {
          if (j == rank) continue;
          int* f = &peers.flags[rank][peers.row_a * flag_row + j * RDNA_ARS_MAX_CHUNKS + c];
          if (!rdna_ars_wait(f, gen, cap)) {
            if (fail != nullptr) *fail = 1;
            return;
          }
        }
      }
      __syncthreads();

      const long long roff = rs_result_off(world, chunks, chunk_elems);
      const T* mine = reinterpret_cast<const T*>(peers.stage[rank]);
      T* rbase = reinterpret_cast<T*>(peers.stage[rank]) + roff;
      for (int i = gid; i < count; i += gstride) {
        float acc = 0.f;
        for (int j = 0; j < world; j++) {
          acc += (float)mine[rs_contrib_slot(j, chunks, c, chunk_elems) + i];
        }
        rbase[(long long)c * chunk_elems + i] = (T)acc;
      }
      __threadfence_system();
      __syncthreads();

      for (int j = 0; j < world; j++) {
        if (j == rank) continue;
        rdna_ars_copy(reinterpret_cast<T*>(peers.stage[j]) + roff + (long long)c * chunk_elems,
                      rbase + (long long)c * chunk_elems, count, gid, gstride);
      }
      __threadfence_system();
      __syncthreads();
      if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int j = 0; j < world; j++) {
          if (j == rank) continue;
          __hip_atomic_store(&peers.flags[j][peers.row_b * flag_row +
                                             rank * RDNA_ARS_MAX_CHUNKS + c],
                             gen, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
      }
    }
    __syncthreads();
  }
}

// Phase B: copy each non-owned chunk out of the owner's published result slot.
template <typename T>
__global__ void rdna_ars_gather_kernel(T* __restrict__ out, RsPeers peers, int world,
                                       int chunks, int chunk_elems, int n, int gen,
                                       unsigned long long cap, int* fail) {
  const int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int gstride = (int)(gridDim.x * blockDim.x);
  const int rank = peers.rank;
  const int flag_row = RDNA_ARS_MAX_WORLD * RDNA_ARS_MAX_CHUNKS;
  const long long roff = rs_result_off(world, chunks, chunk_elems);

  for (int c = 0; c < chunks; c++) {
    const int start = c * chunk_elems;
    const int count = min(chunk_elems, n - start);
    if (count <= 0) continue;

    const int owner = c % world;
    if (owner != rank) {
      if (blockIdx.x == 0 && threadIdx.x == 0) {
        int* f = &peers.flags[rank][peers.row_b * flag_row + owner * RDNA_ARS_MAX_CHUNKS + c];
        if (!rdna_ars_wait(f, gen, cap)) {
          if (fail != nullptr) *fail = 1;
          return;
        }
      }
      __syncthreads();
    }
    const T* src = reinterpret_cast<const T*>(peers.stage[rank]) + roff +
                   (long long)c * chunk_elems;
    rdna_ars_copy(out + start, src, count, gid, gstride);
    __syncthreads();
  }
}
