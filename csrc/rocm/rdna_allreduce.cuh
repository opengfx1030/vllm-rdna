// T44 — push-based one-shot all-reduce for small messages on gfx1030, W ranks (2..8).
//
// Ported from leapdragon/vllm-rdna2-qwen T44/T44b (Aron Hsiao). The VRAM-flag
// protocol and abort-record layout are the same as that tree.
//
// Descendant of the TP=2 WS2 kernel (builds/shared/ws2-allreduce/src/allreduce.hip.h), whose
// findings it keeps:
//   * push, never pull        — peer STORE 14.3 GB/s vs peer LOAD 5.7 GB/s across PCIe;
//   * staging is UNCACHED     — hipDeviceMallocUncached; a peer's write to coarse-grained
//                               memory lands in DRAM while the owner keeps reading stale L2;
//   * flags live in each rank's OWN uncached device memory (2026-08-30; they were a
//     host-coherent page before): a peer announces by one posted P2P store into our flag
//     slot and we poll locally — waiting no longer generates PCIe read traffic. With the
//     host page, four GPUs hammered system memory with 32-bit reads across both root
//     complexes for the whole barrier; on this machine that coincided with the SAS HBA's
//     tape drive resetting and, twice, with cards dropping off the PCIe bus. Coarse-grained
//     device memory would not work (a peer's write is invisible to the owner's L2, T18);
//     uncached memory is exactly what the staging buffers already use for the same reason.
//   * the sequence number is derived on-device from a local counter, never passed as a kernel
//     argument (frozen at CUDA-graph capture, so every replay would fall through the wait);
//   * bounded spins that set a sticky abort flag instead of hanging the GPU -- and, since
//     T44b, record which phase/peer/sequence aborted in a host-visible word (see below).
// New here: W-way staging (one slot per source rank, x2 parities), pushes fanned out to all
// peers, a W-flag wait, and a FIXED-ORDER fp32 reduction (rank 0 .. W-1) so every rank produces
// bit-identical output — with TP, ranks that disagree by an ulp diverge from each other.
//
// Layout of a rank's staging buffer (uncached device memory), in elements of T:
//   stage[(parity * W + src) * max_elems + i]
// Peer j receives our contribution at its slot src = our rank. Flags sit in a 4 KB page
// appended to the same uncached allocation, so the announce store cannot overtake the
// payload on that destination (posted-write order).
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#define RDNA_AR_MAX_WORLD 8
#define RDNA_AR_FLAG_PAGE 4096  // bytes appended to each staging buffer for the flag slots
#define RDNA_AR_SPIN_CAP 2000000ull  // default polls before abort (~1 us each -> ~2 s); VLLM_RDNA_AR_SPIN_CAP
// Second flag row for the two-shot allgather. One-shot uses row 0 only.
#define RDNA_AR_FLAG_STRIDE 16
// Messages at or below this stay on one-shot unless VLLM_RDNA_AR_ALGO overrides.
#define RDNA_AR_ONESHOT_MAX 32768

// T44b (2026-09-07) -- abort record. Before this, a collective that hit the spin cap set a bare
// sticky flag that only the boot self-test ever read: mid-serving, every later collective also
// spun to its cap and returned WITHOUT writing its output, so a wedged P2P path looked like
// "one or three GPUs pinned at 99 %, generation stopped" for the 300 s engine timeout (two
// boards reported it). Now the first block to hit the cap claims the device word `timeout`
// (atomicCAS, device scope) and writes one 64-bit code into `report`, a host-mapped mirror the
// Python side polls once per step with a plain load -- no device sync, no PCIe atomics:
//   bits 0-7   1 = aborted          bits 8-11  phase: 1 = own blocks' grid barrier,
//   bits 12-15 peer rank (phase 2)                    2 = a peer's flag never arrived
//   bits 16-31 spins / 1024 (~ms)   bits 32-63 sequence number of the collective
__device__ __forceinline__ void rdna_ar_abort(unsigned* timeout, unsigned long long* report,
                                              unsigned phase, unsigned peer, int seq,
                                              unsigned long long spins) {
  if (atomicCAS(timeout, 0u, 1u) == 0u) {
    unsigned long long code = 1ull | ((unsigned long long)(phase & 0xFu) << 8) |
                              ((unsigned long long)(peer & 0xFu) << 12) |
                              (((spins >> 10) & 0xFFFFull) << 16) |
                              ((unsigned long long)(unsigned)seq << 32);
    __hip_atomic_store(report, code, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ __forceinline__ float rdna_ar_to_f(float x) { return x; }
__device__ __forceinline__ float rdna_ar_to_f(__half x) { return __half2float(x); }
__device__ __forceinline__ float rdna_ar_to_f(__hip_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ void rdna_ar_from_f(float& d, float v) { d = v; }
__device__ __forceinline__ void rdna_ar_from_f(__half& d, float v) { d = __float2half(v); }
__device__ __forceinline__ void rdna_ar_from_f(__hip_bfloat16& d, float v) {
  d = __float2bfloat16(v);
}

// 16-byte stores when the span is aligned and pacing is off. Scalar stores
// otherwise: a 2-byte PCIe TLP is the slow path two-shot is trying to avoid.
template <typename T>
__device__ __forceinline__ void rdna_ar_copy(T* __restrict__ dst, const T* __restrict__ src,
                                             int count, int gid, int gstride, int pace) {
  constexpr int kVec = (int)(16 / sizeof(T));
  const bool vec = pace == 0 && kVec > 1 && ((uintptr_t)dst % 16u) == 0 &&
                   ((uintptr_t)src % 16u) == 0;
  if (vec) {
    const int nvec = count / kVec;
    auto* d4 = reinterpret_cast<uint4*>(dst);
    const auto* s4 = reinterpret_cast<const uint4*>(src);
    for (int i = gid; i < nvec; i += gstride) d4[i] = s4[i];
    for (int i = nvec * kVec + gid; i < count; i += gstride) dst[i] = src[i];
    return;
  }
  for (int i = gid; i < count; i += gstride) {
    dst[i] = src[i];
    for (int q = 0; q < pace; q++) __builtin_amdgcn_s_sleep(1);
  }
}

template <typename T>
__device__ __forceinline__ T* rdna_ar_slot(void* stage, int parity, int src, int world,
                                           long long max_elems) {
  return reinterpret_cast<T*>(stage) +
         ((long long)parity * world + src) * max_elems;
}

struct RdnaArPeers {
  void* stage[RDNA_AR_MAX_WORLD];  // peer j's staging buffer (IPC-mapped); [rank] = ours
  int* flags[RDNA_AR_MAX_WORLD];   // peer j's flag slots [W] (uncached, after its staging)
};

// Backoff between polls: s_sleep(n) idles the wave ~64*n clocks (~0.2 us at n=8) so a
// waiting rank does not saturate its memory path while spinning.
#define RDNA_AR_POLL_PAUSE() __builtin_amdgcn_s_sleep(8)

template <typename T>
__global__ void rdna_ar_oneshot(const T* __restrict__ in, T* __restrict__ out,
                                RdnaArPeers peers,               // stage + flags per rank
                                unsigned int* arrive,             // device, [2] per parity
                                int* seqbuf,                      // device, local seq mirror
                                unsigned* timeout,                // device, sticky abort claim
                                unsigned long long* report,       // host-mapped abort record (T44b)
                                int rank, int world, int n, long long max_elems,
                                int nblocks, int pace, unsigned long long spin_cap) {
  __shared__ int s_seq;
  __shared__ int s_abort;
  const int t = threadIdx.x, nt = blockDim.x, b = blockIdx.x;
  if (t == 0) {
    s_seq = __hip_atomic_load(seqbuf, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) + 1;
    s_abort = 0;
  }
  __syncthreads();
  const int seq = s_seq;
  const int p = seq & 1;
  const int gid = b * nt + t, gstride = nblocks * nt;

  // 1. push our slice into every peer's staging slot for us (posted PCIe writes).
  //    The peer order is staggered by rank, so at any instant each destination is being
  //    written by ONE source instead of all W-1 at once; `pace` idles the wave ~64 clocks per
  //    unit between strided stores to bound the burst rate (0 = off). Both are for the rest of
  //    the machine: these pushes land in the receiving GPU's root complex, where other devices'
  //    DMA completions queue behind them (2026-09-01: the SAS HBA on this box lives there).
  for (int k = 1; k < world; k++) {
    const int j = (rank + k) % world;
    T* dst = reinterpret_cast<T*>(peers.stage[j]) + ((long long)p * world + rank) * max_elems;
    for (int i = gid; i < n; i += gstride) {
      dst[i] = in[i];
      for (int q = 0; q < pace; q++) __builtin_amdgcn_s_sleep(1);
    }
  }
  __syncthreads();

  // 2. grid barrier (all our blocks have pushed), payload-before-flag, announce, wait
  if (t == 0) {
    __threadfence_system();
    atomicAdd(&arrive[p], 1u);
    if (b == 0) {
      unsigned long long s = 0;
      while (__hip_atomic_load(&arrive[p], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) <
             (unsigned)nblocks) {
        RDNA_AR_POLL_PAUSE();
        if (++s > spin_cap) {
          rdna_ar_abort(timeout, report, 1u, (unsigned)rank, seq, s);
          s_abort = 1;
          break;
        }
      }
      if (!s_abort) {
        // Both slots must be clear for a following two-shot, which uses one
        // counter per phase. Every block has already atomicAdded.
        arrive[p] = 0u;
        arrive[1 - p] = 0u;
        __hip_atomic_store(seqbuf, seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        // announce: one posted P2P store into every peer's slot for us (and our own)
        for (int j = 0; j < world; j++)
          __hip_atomic_store(&peers.flags[j][rank], seq, __ATOMIC_RELEASE,
                             __HIP_MEMORY_SCOPE_SYSTEM);
      }
    }
    if (!s_abort) {
      // wait: poll OUR flag slots (local uncached memory, no PCIe traffic)
      int* myflags = peers.flags[rank];
      for (int j = 0; j < world && !s_abort; j++) {
        if (j == rank) continue;
        unsigned long long s = 0;
        while (__hip_atomic_load(&myflags[j], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < seq) {
          RDNA_AR_POLL_PAUSE();
          if (++s > spin_cap) {
            rdna_ar_abort(timeout, report, 2u, (unsigned)j, seq, s);
            s_abort = 1;
            break;
          }
        }
      }
    }
  }
  __syncthreads();
  if (s_abort) return;

  // 3. fixed-order reduction: rank 0 .. W-1, fp32, identical on every rank
  const T* mine = reinterpret_cast<const T*>(peers.stage[rank]) + ((long long)p * world) * max_elems;
  for (int i = gid; i < n; i += gstride) {
    float v = 0.f;
    for (int j = 0; j < world; j++)
      v += (j == rank) ? rdna_ar_to_f(in[i]) : rdna_ar_to_f(mine[(long long)j * max_elems + i]);
    rdna_ar_from_f(out[i], v);
  }
}

// Push reduce-scatter + push allgather. Traffic is ~2*(W-1)/W * N instead of
// one-shot's (W-1)*N, so prefill-sized tensors do not have to fall through to
// vLLM's pull-based custom all-reduce. Same uncached flags and abort record.
// Phase 3 = second grid barrier, phase 4 = a peer's allgather flag.
template <typename T>
__global__ void rdna_ar_twoshot(const T* __restrict__ in, T* __restrict__ out,
                                RdnaArPeers peers, unsigned int* arrive, int* seqbuf,
                                unsigned* timeout, unsigned long long* report,
                                int rank, int world, int n, long long max_elems,
                                int nblocks, int pace, unsigned long long spin_cap) {
  __shared__ int s_seq;
  __shared__ int s_abort;
  const int t = threadIdx.x, nt = blockDim.x, b = blockIdx.x;
  if (t == 0) {
    s_seq = __hip_atomic_load(seqbuf, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) + 1;
    s_abort = 0;
  }
  __syncthreads();
  const int seq = s_seq;
  const int p = seq & 1;
  const int gid = b * nt + t, gstride = nblocks * nt;
  const int part = n / world;

  for (int k = 1; k < world; k++) {
    const int j = (rank + k) % world;
    const int j_start = j * part;
    const int j_end = (j == world - 1) ? n : j_start + part;
    T* dst = rdna_ar_slot<T>(peers.stage[j], p, rank, world, max_elems);
    rdna_ar_copy(dst, in + j_start, j_end - j_start, gid, gstride, pace);
  }
  __syncthreads();

  if (t == 0) {
    __threadfence_system();
    atomicAdd(&arrive[p], 1u);
    if (b == 0) {
      unsigned long long s = 0;
      while (__hip_atomic_load(&arrive[p], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) <
             (unsigned)nblocks) {
        RDNA_AR_POLL_PAUSE();
        if (++s > spin_cap) {
          rdna_ar_abort(timeout, report, 1u, (unsigned)rank, seq, s);
          s_abort = 1;
          break;
        }
      }
      if (!s_abort) {
        __hip_atomic_store(seqbuf, seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        for (int j = 0; j < world; j++)
          __hip_atomic_store(&peers.flags[j][rank], seq, __ATOMIC_RELEASE,
                             __HIP_MEMORY_SCOPE_SYSTEM);
      }
    }
    if (!s_abort) {
      int* myflags = peers.flags[rank];
      for (int j = 0; j < world && !s_abort; j++) {
        if (j == rank) continue;
        unsigned long long s = 0;
        while (__hip_atomic_load(&myflags[j], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) <
               seq) {
          RDNA_AR_POLL_PAUSE();
          if (++s > spin_cap) {
            rdna_ar_abort(timeout, report, 2u, (unsigned)j, seq, s);
            s_abort = 1;
            break;
          }
        }
      }
    }
  }
  __syncthreads();
  if (s_abort) return;

  const int start = rank * part;
  const int end = (rank == world - 1) ? n : start + part;
  const int count = end - start;
  for (int i = gid; i < count; i += gstride) {
    float v = 0.f;
    for (int j = 0; j < world; j++) {
      const T elem = (j == rank)
                         ? in[start + i]
                         : rdna_ar_slot<T>(peers.stage[rank], p, j, world, max_elems)[i];
      v += rdna_ar_to_f(elem);
    }
    rdna_ar_from_f(out[start + i], v);
  }
  __syncthreads();

  const int ag = 1 - p;
  for (int k = 1; k < world; k++) {
    const int j = (rank + k) % world;
    T* dst = rdna_ar_slot<T>(peers.stage[j], ag, rank, world, max_elems);
    rdna_ar_copy(dst, out + start, count, gid, gstride, pace);
  }
  __syncthreads();

  if (t == 0) {
    __threadfence_system();
    atomicAdd(&arrive[ag], 1u);
    if (b == 0) {
      unsigned long long s = 0;
      while (__hip_atomic_load(&arrive[ag], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) <
             (unsigned)nblocks) {
        RDNA_AR_POLL_PAUSE();
        if (++s > spin_cap) {
          rdna_ar_abort(timeout, report, 3u, (unsigned)rank, seq, s);
          s_abort = 1;
          break;
        }
      }
      if (!s_abort) {
        arrive[p] = 0u;
        arrive[ag] = 0u;
        for (int j = 0; j < world; j++)
          __hip_atomic_store(&peers.flags[j][RDNA_AR_FLAG_STRIDE + rank], seq,
                             __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
      }
    }
    if (!s_abort) {
      int* myflags = peers.flags[rank];
      for (int j = 0; j < world && !s_abort; j++) {
        if (j == rank) continue;
        unsigned long long s = 0;
        while (__hip_atomic_load(&myflags[RDNA_AR_FLAG_STRIDE + j], __ATOMIC_ACQUIRE,
                                 __HIP_MEMORY_SCOPE_SYSTEM) < seq) {
          RDNA_AR_POLL_PAUSE();
          if (++s > spin_cap) {
            rdna_ar_abort(timeout, report, 4u, (unsigned)j, seq, s);
            s_abort = 1;
            break;
          }
        }
      }
    }
  }
  __syncthreads();
  if (s_abort) return;

  for (int j = 0; j < world; j++) {
    if (j == rank) continue;
    const int j_start = j * part;
    const int j_end = (j == world - 1) ? n : j_start + part;
    const T* src = rdna_ar_slot<T>(peers.stage[rank], ag, j, world, max_elems);
    rdna_ar_copy(out + j_start, src, j_end - j_start, gid, gstride, 0);
  }
}
