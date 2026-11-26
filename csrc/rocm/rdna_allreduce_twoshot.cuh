// SPDX-License-Identifier: Apache-2.0
// Wave-7 BAR1 PCIe P2P 2-shot mesh all-reduce (Reduce-Scatter + All-Gather)
// for gfx1030 (world = 2..8). Designed to share RdnaArPeers staging and
// parity sequence buffers with rdna_ar_oneshot while pushing 16B-aligned
// uint4 posted TLPs across independent per-CTA channels without grid barriers.
#pragma once
#include "rdna_allreduce.cuh"

#define RDNA_AR_2S_FLAG_BASE 16
#define RDNA_AR_2S_MAX_BLOCKS 32

static_assert(RDNA_AR_2S_FLAG_BASE +
                      2 * RDNA_AR_2S_MAX_BLOCKS * RDNA_AR_MAX_WORLD <=
                  RDNA_AR_FLAG_PAGE / 4,
              "2-shot flags must fit inside RDNA_AR_FLAG_PAGE");

__device__ __forceinline__ int rdna_ar_2s_flag(int phase, int b, int src) {
  return RDNA_AR_2S_FLAG_BASE +
         (phase * RDNA_AR_2S_MAX_BLOCKS + b) * RDNA_AR_MAX_WORLD + src;
}

template <typename T>
__device__ __forceinline__ void rdna_ar_2s_copy_range(T* __restrict__ dst,
                                                      const T* __restrict__ src,
                                                      int lo, int hi) {
  constexpr int kVec = 16 / (int)sizeof(T);
  const int t = threadIdx.x;
  const int nt = blockDim.x;
  const int len = hi - lo;
  if (len <= 0) return;
  if constexpr (kVec >= 2) {
    if (((uintptr_t)(dst + lo) & 15) == 0 &&
        ((uintptr_t)(src + lo) & 15) == 0 && len >= kVec) {
      const int nvec = len / kVec;
      uint4* __restrict__ d4 = reinterpret_cast<uint4*>(dst + lo);
      const uint4* __restrict__ s4 = reinterpret_cast<const uint4*>(src + lo);
      for (int v = t; v < nvec; v += nt) d4[v] = s4[v];
      for (int i = nvec * kVec + t; i < len; i += nt) dst[lo + i] = src[lo + i];
      return;
    }
  }
  for (int i = t; i < len; i += nt) dst[lo + i] = src[lo + i];
}

template <typename T>
__device__ __forceinline__ bool rdna_ar_2s_wait_phase(
    const RdnaArPeers& peers, unsigned* timeout, unsigned long long* report,
    int rank, int world, int b, int phase, int seq, long long wait_ticks,
    long long ticks_per_ms) {
  int* myflags = peers.flags[rank];
  const long long slow_ticks = RDNA_AR_SLOW_MS * ticks_per_ms;
  for (int j = 0; j < world; j++) {
    if (j == rank) continue;
    const int idx = rdna_ar_2s_flag(phase, b, j);
    const long long t0 = wall_clock64();
    long long dt = 0, next = slow_ticks, seen = -1;
    unsigned bits = 0u;
    bool live = false;
    while (rdna_ar_before(__hip_atomic_load(&myflags[idx], __ATOMIC_ACQUIRE,
                                            __HIP_MEMORY_SCOPE_SYSTEM),
                          seq)) {
      RDNA_AR_POLL_PAUSE();
      dt = wall_clock64() - t0;
      if (dt < next && dt <= wait_ticks) continue;
      next = dt + RDNA_AR_PROBE_MS * ticks_per_ms;
      const int f = __hip_atomic_load(&myflags[j], __ATOMIC_ACQUIRE,
                                      __HIP_MEMORY_SCOPE_SYSTEM);
      const bool p_seen = !rdna_ar_before(f, seq);
      if (p_seen && seen < 0) seen = dt;
      if (dt > wait_ticks ||
          (seen >= 0 && dt - seen > RDNA_AR_LOST_MS * ticks_per_ms)) {
        if (p_seen) bits |= RDNA_AR_REC_SEEN;
        rdna_ar_abort_ms(timeout, report, (unsigned)(3 + phase) | bits,
                         (unsigned)j, seq, dt / ticks_per_ms);
        return false;
      }
      if (__hip_atomic_load(timeout, __ATOMIC_RELAXED,
                            __HIP_MEMORY_SCOPE_AGENT) != 0u) {
        return false;
      }
      if (b == 0) {
        live = true;
        *report = rdna_ar_ms_code((unsigned)(3 + phase) | RDNA_AR_REC_LIVE |
                                      (p_seen ? RDNA_AR_REC_SEEN : 0u),
                                  (unsigned)j, seq, dt / ticks_per_ms);
      }
    }
    if (live) {
      report[1] = rdna_ar_ms_code(
          (unsigned)(3 + phase) | (seen >= 0 ? RDNA_AR_REC_SEEN : 0u),
          (unsigned)j, seq, dt / ticks_per_ms);
      *report = 0ull;
    }
  }
  return true;
}

template <typename T>
__global__ void rdna_ar_twoshot_mesh(
    const T* __restrict__ in, T* __restrict__ out, RdnaArPeers peers,
    unsigned int* arrive, int* seqbuf, unsigned* timeout,
    unsigned long long* report, int rank, int world, int n, long long max_elems,
    int nblocks, long long wait_ticks, long long ticks_per_ms) {
#if defined(__gfx1030__) || !defined(__HIP_DEVICE_COMPILE__)
  __shared__ int s_seq;
  __shared__ int s_ok;
  const int t = threadIdx.x, nt = blockDim.x, b = blockIdx.x;
  if (t == 0) {
    s_seq = (int)((unsigned)__hip_atomic_load(seqbuf, __ATOMIC_ACQUIRE,
                                              __HIP_MEMORY_SCOPE_AGENT) +
                  1u);
    s_ok = (__hip_atomic_load(timeout, __ATOMIC_RELAXED,
                              __HIP_MEMORY_SCOPE_AGENT) == 0u)
               ? 1
               : 0;
  }
  __syncthreads();
  const int seq = s_seq;
  const int p = seq & 1;

  if (!s_ok) {
    if (b == 0 && t == 0) {
      __hip_atomic_store(seqbuf, seq, __ATOMIC_RELEASE,
                         __HIP_MEMORY_SCOPE_AGENT);
    }
    for (int i = b * nt + t; i < n; i += nblocks * nt) out[i] = in[i];
    return;
  }

  constexpr int kVec = 16 / (int)sizeof(T);
  const int n_chunks = (n + kVec - 1) / kVec;
  const int ch_lo = (int)((long long)n_chunks * b / nblocks);
  const int ch_hi = (int)((long long)n_chunks * (b + 1) / nblocks);
  const int dc = ch_hi - ch_lo;

  auto shard_bounds = [&](int d, int& lo, int& hi) {
    const int s0 = ch_lo + (int)((long long)dc * d / world);
    const int s1 = ch_lo + (int)((long long)dc * (d + 1) / world);
    lo = min(n, s0 * kVec);
    hi = min(n, s1 * kVec);
  };

  // Phase 1 (Reduce-Scatter): staggered 16B pushes of peer d's shard into
  // peers.stage[d] slot (parity, rank) at [lo_d, hi_d).
  for (int k = 1; k < world; k++) {
    const int d = (rank + k) % world;
    int lo = 0, hi = 0;
    shard_bounds(d, lo, hi);
    T* dst = reinterpret_cast<T*>(peers.stage[d]) +
             ((long long)p * world + rank) * max_elems;
    rdna_ar_2s_copy_range<T>(dst, in, lo, hi);
  }
  __syncthreads();
  if (t == 0) {
    __threadfence_system();
    for (int k = 1; k < world; k++) {
      const int d = (rank + k) % world;
      __hip_atomic_store(&peers.flags[d][rdna_ar_2s_flag(0, b, rank)], seq,
                         __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    if (!rdna_ar_2s_wait_phase<T>(peers, timeout, report, rank, world, b, 0,
                                  seq, wait_ticks, ticks_per_ms)) {
      s_ok = 0;
    }
  }
  __syncthreads();
  if (!s_ok) {
    if (b == 0 && t == 0) {
      __hip_atomic_store(seqbuf, seq, __ATOMIC_RELEASE,
                         __HIP_MEMORY_SCOPE_AGENT);
    }
    for (int i = b * nt + t; i < n; i += nblocks * nt) out[i] = in[i];
    return;
  }

  // Local shard reduction in ascending rank order (0..world-1) in float32.
  int my_lo = 0, my_hi = 0;
  shard_bounds(rank, my_lo, my_hi);
  const T* mine_base = reinterpret_cast<const T*>(peers.stage[rank]) +
                       ((long long)p * world) * max_elems;
  const int my_len = my_hi - my_lo;
  if constexpr (kVec >= 2) {
    if (((uintptr_t)(out + my_lo) & 15) == 0 &&
        ((uintptr_t)(in + my_lo) & 15) == 0 && my_len >= kVec) {
      const int nvec = my_len / kVec;
      uint4* out4 = reinterpret_cast<uint4*>(out + my_lo);
      const uint4* in4 = reinterpret_cast<const uint4*>(in + my_lo);
      for (int v = t; v < nvec; v += nt) {
        float acc[kVec];
  #pragma unroll
        for (int m = 0; m < kVec; m++) acc[m] = 0.0f;
        for (int j = 0; j < world; j++) {
          const uint4* src4 =
              (j == rank) ? in4
                          : reinterpret_cast<const uint4*>(
                                mine_base + (long long)j * max_elems + my_lo);
          union {
            uint4 q;
            T e[kVec];
          } s;
          s.q = src4[v];
  #pragma unroll
          for (int m = 0; m < kVec; m++) acc[m] += rdna_ar_to_f(s.e[m]);
        }
        union {
          uint4 q;
          T e[kVec];
        } r;
  #pragma unroll
        for (int m = 0; m < kVec; m++) rdna_ar_from_f(r.e[m], acc[m]);
        out4[v] = r.q;
      }
      for (int i = my_lo + nvec * kVec + t; i < my_hi; i += nt) {
        float acc = 0.0f;
        for (int j = 0; j < world; j++) {
          const T v =
              (j == rank) ? in[i] : mine_base[(long long)j * max_elems + i];
          acc += rdna_ar_to_f(v);
        }
        rdna_ar_from_f(out[i], acc);
      }
    } else {
      for (int i = my_lo + t; i < my_hi; i += nt) {
        float acc = 0.0f;
        for (int j = 0; j < world; j++) {
          const T v =
              (j == rank) ? in[i] : mine_base[(long long)j * max_elems + i];
          acc += rdna_ar_to_f(v);
        }
        rdna_ar_from_f(out[i], acc);
      }
    }
  }
  __syncthreads();

  // Phase 2 (All-Gather): push reduced my_shard from out[my_lo..my_hi) into
  // peers.stage[d] slot (parity, rank) at [my_lo, my_hi) -- disjoint from RS
  // writes because d != rank.
  for (int k = 1; k < world; k++) {
    const int d = (rank + k) % world;
    T* dst = reinterpret_cast<T*>(peers.stage[d]) +
             ((long long)p * world + rank) * max_elems;
    rdna_ar_2s_copy_range<T>(dst, out, my_lo, my_hi);
  }
  __syncthreads();
  if (t == 0) {
    __threadfence_system();
    for (int k = 1; k < world; k++) {
      const int d = (rank + k) % world;
      __hip_atomic_store(&peers.flags[d][rdna_ar_2s_flag(1, b, rank)], seq,
                         __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    if (!rdna_ar_2s_wait_phase<T>(peers, timeout, report, rank, world, b, 1,
                                  seq, wait_ticks, ticks_per_ms)) {
      s_ok = 0;
    }
  }
  __syncthreads();
  if (!s_ok) {
    if (b == 0 && t == 0) {
      __hip_atomic_store(seqbuf, seq, __ATOMIC_RELEASE,
                         __HIP_MEMORY_SCOPE_AGENT);
    }
    for (int i = b * nt + t; i < n; i += nblocks * nt) out[i] = in[i];
    return;
  }

  for (int k = 1; k < world; k++) {
    const int j = (rank + k) % world;
    int lo = 0, hi = 0;
    shard_bounds(j, lo, hi);
    const T* src = mine_base + (long long)j * max_elems;
    rdna_ar_2s_copy_range<T>(out, src, lo, hi);
  }
  __syncthreads();
  if (t == 0) {
    if (nblocks == 1 || atomicAdd(&arrive[p], 1u) == (unsigned)nblocks - 1u) {
      arrive[p] = 0u;
      arrive[1 - p] = 0u;
      __hip_atomic_store(seqbuf, seq, __ATOMIC_RELEASE,
                         __HIP_MEMORY_SCOPE_AGENT);
    }
  }
#endif
}
