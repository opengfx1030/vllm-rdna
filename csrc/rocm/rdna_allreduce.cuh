// T44 — push-based one-shot all-reduce for small messages on gfx1030, W ranks
// (2..8).
//
// Ported from leapdragon/vllm-rdna2-qwen T44/T44b (Aron Hsiao). The VRAM-flag
// protocol and abort-record layout are the same as that tree.
//
// Descendant of the TP=2 WS2 kernel
// (builds/shared/ws2-allreduce/src/allreduce.hip.h), whose findings it keeps:
//   * push, never pull        — peer STORE 14.3 GB/s vs peer LOAD 5.7 GB/s
//   across PCIe;
//   * staging is UNCACHED     — hipDeviceMallocUncached; a peer's write to
//   coarse-grained
//                               memory lands in DRAM while the owner keeps
//                               reading stale L2;
//   * flags live in each rank's OWN uncached device memory (2026-08-30; they
//   were a
//     host-coherent page before): a peer announces by one posted P2P store into
//     our flag slot and we poll locally — waiting no longer generates PCIe read
//     traffic. With the host page, four GPUs hammered system memory with 32-bit
//     reads across both root complexes for the whole barrier; on this machine
//     that coincided with the SAS HBA's tape drive resetting and, twice, with
//     cards dropping off the PCIe bus. Coarse-grained device memory would not
//     work (a peer's write is invisible to the owner's L2, T18); uncached
//     memory is exactly what the staging buffers already use for the same
//     reason.
//   * the sequence number is derived on-device from a local counter, never
//   passed as a kernel
//     argument (frozen at CUDA-graph capture, so every replay would fall
//     through the wait);
//   * bounded spins that set a sticky abort flag instead of hanging the GPU --
//   and, since
//     T44b, record which phase/peer/sequence aborted in a host-visible word
//     (see below).
// New here: W-way staging (one slot per source rank, x2 parities), pushes
// fanned out to all peers, a W-flag wait, and a FIXED-ORDER fp32 reduction
// (rank 0 .. W-1) so every rank produces bit-identical output — with TP, ranks
// that disagree by an ulp diverge from each other.
//
// Layout of a rank's staging buffer (uncached device memory), in elements of T:
//   stage[(parity * W + src) * max_elems + i]
// Peer j receives our contribution at its slot src = our rank. Flags sit in a 4
// KB page appended to the same uncached allocation, so the announce store
// cannot overtake the payload on that destination (posted-write order).
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#define RDNA_AR_MAX_WORLD 8
#define RDNA_AR_FLAG_PAGE \
  4096  // bytes appended to each staging buffer for the flag slots
// One-shot waits are bounded in wall-clock time (wall_clock64, constant rate
// from hipDeviceAttributeWallClockRate), not in polls: a poll costs ~0.4 us on
// gfx1030, so the old 2e6-poll cap was really ~0.8 s, and its abort record's
// "~ms" was a poll count. The bound is for a lost P2P write, NOT for rank
// skew: a peer legitimately reaches the collective seconds late whenever it
// JIT-compiles a kernel or runs a long host step first (first request after
// boot, a new shape). The old cap aborted exactly then ("peer rank N's flag
// never arrived"), killed the engine and left a marker that kept later boots
// on RCCL. 120 s stays under the engine's execute timeout, so a real loss
// still ends with this clear error. VLLM_RDNA_AR_WAIT_MS overrides.
#define RDNA_AR_WAIT_MS 120000ll
// One-shot waits longer than this that still complete are recorded in
// report[1] (abort-record layout, phase 2) and logged as a late peer.
#define RDNA_AR_SLOW_MS 2000ll
// One-shot gate: larger messages are not eligible and go to RCCL
// (VLLM_RDNA_AR_ONESHOT_KB overrides). The two-shot kernel that used to
// cover them was removed: it returned zeros above this gate and made the
// boot self-test disable the whole backend.
#define RDNA_AR_ONESHOT_MAX 65536

// T44b (2026-09-07) -- abort record. Before this, a collective that hit the
// spin cap set a bare sticky flag that only the boot self-test ever read:
// mid-serving, every later collective also spun to its cap and returned WITHOUT
// writing its output, so a wedged P2P path looked like "one or three GPUs
// pinned at 99 %, generation stopped" for the 300 s engine timeout (two boards
// reported it). The first block to hit the cap claims the device word
// `timeout` (atomicCAS, device scope) and writes one 64-bit code into `report`,
// a host-mapped mirror the Python side polls once per step with a plain load --
// no device sync, no PCIe atomics.
// The record alone did not stop the re-spinning: the host reads it only after
// the step, and a CUDA-graph step holds ~100 collectives, each of which waited
// up to the bound again (the 300 s engine timeout still came first). So the
// claim is sticky in the kernel too (2026-10-18): a collective launched after
// it skips its push and waits, copies `in` to `out` (a finite, wrong local
// value), and still announces its sequence number so peers do not wait on
// this rank. The step then completes and rdna_ar_check() fails it with the
// record. Layout:
//   bits 0-7   1 = aborted          bits 8-11  phase: 1 = own blocks' grid
//   barrier, bits 12-15 peer rank (phase 2)                    2 = a peer's
//   flag never arrived bits 16-31 wait time (see below) bits 32-63 sequence
//   number of the collective
// Abort / late-peer record: bits 16-31 hold the measured wall-clock wait in
// 16 ms units (saturating, ~17 min).
__device__ __forceinline__ unsigned long long rdna_ar_ms_code(unsigned phase,
                                                              unsigned peer,
                                                              int seq,
                                                              long long ms) {
  const unsigned long long u =
      ms / 16 > 0xFFFFll ? 0xFFFFull : (unsigned long long)(ms / 16);
  return 1ull | ((unsigned long long)(phase & 0xFu) << 8) |
         ((unsigned long long)(peer & 0xFu) << 12) | (u << 16) |
         ((unsigned long long)(unsigned)seq << 32);
}

__device__ __forceinline__ void rdna_ar_abort_ms(unsigned* timeout,
                                                 unsigned long long* report,
                                                 unsigned phase, unsigned peer,
                                                 int seq, long long ms) {
  if (atomicCAS(timeout, 0u, 1u) == 0u) {
    __hip_atomic_store(report, rdna_ar_ms_code(phase, peer, seq, ms),
                       __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ __forceinline__ float rdna_ar_to_f(float x) { return x; }
__device__ __forceinline__ float rdna_ar_to_f(__half x) {
  return __half2float(x);
}
__device__ __forceinline__ float rdna_ar_to_f(__hip_bfloat16 x) {
  return __bfloat162float(x);
}
__device__ __forceinline__ void rdna_ar_from_f(float& d, float v) { d = v; }
__device__ __forceinline__ void rdna_ar_from_f(__half& d, float v) {
  d = __float2half(v);
}
__device__ __forceinline__ void rdna_ar_from_f(__hip_bfloat16& d, float v) {
  d = __float2bfloat16(v);
}

// 16-byte stores when the span is aligned and pacing is off. Scalar stores
// otherwise: a 2-byte PCIe TLP is the slow path.
template <typename T>
__device__ __forceinline__ void rdna_ar_copy(T* __restrict__ dst,
                                             const T* __restrict__ src,
                                             int count, int gid, int gstride,
                                             int pace) {
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
__device__ __forceinline__ T* rdna_ar_slot(void* stage, int parity, int src,
                                           int world, long long max_elems) {
  return reinterpret_cast<T*>(stage) +
         ((long long)parity * world + src) * max_elems;
}

// 16-byte loads and stores when every pointer is aligned (the uncached staging
// is transaction-limited, so this is ~2.7x on the reduce phase); the scalar
// fallback is the old loop, and both sum in the same rank order, so the result
// stays bit-identical across ranks.
template <typename T>
__device__ __forceinline__ void rdna_ar_reduce(
    const T* __restrict__ mine, const T* __restrict__ in, T* __restrict__ out,
    int n, int rank, int world, long long max_elems, int gid, int gstride) {
  constexpr int kVec = (int)(16 / sizeof(T));
  const bool vec = kVec > 1 && (max_elems % kVec) == 0 &&
                   ((uintptr_t)in % 16u) == 0 && ((uintptr_t)out % 16u) == 0 &&
                   ((uintptr_t)mine % 16u) == 0;
  if (vec) {
    const int nvec = n / kVec;
    const long long slot_v = max_elems / kVec;
    const uint4* m4 = reinterpret_cast<const uint4*>(mine);
    const uint4* i4 = reinterpret_cast<const uint4*>(in);
    uint4* o4 = reinterpret_cast<uint4*>(out);
    for (int i = gid; i < nvec; i += gstride) {
      float f[kVec];
#pragma unroll
      for (int q = 0; q < kVec; q++) f[q] = 0.f;
      for (int j = 0; j < world; j++) {
        const uint4 v = (j == rank) ? i4[i] : m4[(long long)j * slot_v + i];
        const T* vp = reinterpret_cast<const T*>(&v);
#pragma unroll
        for (int q = 0; q < kVec; q++) f[q] += rdna_ar_to_f(vp[q]);
      }
      uint4 acc;
      T* ap = reinterpret_cast<T*>(&acc);
#pragma unroll
      for (int q = 0; q < kVec; q++) rdna_ar_from_f(ap[q], f[q]);
      o4[i] = acc;
    }
    for (int i = nvec * kVec + gid; i < n; i += gstride) {
      float v = 0.f;
      for (int j = 0; j < world; j++)
        v += (j == rank) ? rdna_ar_to_f(in[i])
                         : rdna_ar_to_f(mine[(long long)j * max_elems + i]);
      rdna_ar_from_f(out[i], v);
    }
    return;
  }
  for (int i = gid; i < n; i += gstride) {
    float v = 0.f;
    for (int j = 0; j < world; j++)
      v += (j == rank) ? rdna_ar_to_f(in[i])
                       : rdna_ar_to_f(mine[(long long)j * max_elems + i]);
    rdna_ar_from_f(out[i], v);
  }
}

struct RdnaArPeers {
  void* stage[RDNA_AR_MAX_WORLD];  // peer j's staging buffer (IPC-mapped);
                                   // [rank] = ours
  int* flags[RDNA_AR_MAX_WORLD];   // peer j's flag slots [W] (uncached, after
                                   // its staging)
};

// Backoff between polls: s_sleep(n) idles the wave ~64*n clocks (~0.2 us at
// n=8) so a waiting rank does not saturate its memory path while spinning.
// What another kernel needs to run a one-shot all-reduce step on the same
// buffers and sequence (MonoKernel Stage D): it must follow the same protocol
// (push to stage[j] slot (parity, rank), bump *seqbuf, publish flags[j][rank],
// wait for flags[rank][j] >= seq, sum the slots in rank order).
struct RdnaArDev {
  RdnaArPeers peers;
  int* seqbuf;
  long long max_bytes;  // per slot
  int rank, world, ready;
};
// Host side (rdna_allreduce.cu): false if the handle is not a ready instance.
bool rdna_ar_device(long long handle, RdnaArDev* out);

#define RDNA_AR_POLL_PAUSE() __builtin_amdgcn_s_sleep(8)

template <typename T>
__global__ void rdna_ar_oneshot(
    const T* __restrict__ in, T* __restrict__ out,
    RdnaArPeers peers,           // stage + flags per rank
    unsigned int* arrive,        // device, [2] per parity
    int* seqbuf,                 // device, local seq mirror
    unsigned* timeout,           // device, sticky abort claim
    unsigned long long* report,  // host-mapped abort record (T44b)
    int rank, int world, int n, long long max_elems, int nblocks, int pace,
    long long wait_ticks,      // wall-clock bound per wait
    long long ticks_per_ms) {  // wall_clock64 rate
  __shared__ int s_seq;
  __shared__ int s_abort;
  const int t = threadIdx.x, nt = blockDim.x, b = blockIdx.x;
  if (t == 0) {
    s_seq =
        __hip_atomic_load(seqbuf, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) +
        1;
    // Sticky abort (see the T44b note): an earlier collective on this
    // instance timed out, so this one must not wait again.
    s_abort = __hip_atomic_load(timeout, __ATOMIC_RELAXED,
                                __HIP_MEMORY_SCOPE_AGENT) != 0u;
  }
  __syncthreads();
  const int seq = s_seq;
  const int p = seq & 1;
  const int gid = b * nt + t, gstride = nblocks * nt;
  if (s_abort) {
    if (b == 0 && t == 0) {
      // Keep the sequence moving and release peers waiting on this rank.
      __hip_atomic_store(seqbuf, seq, __ATOMIC_RELEASE,
                         __HIP_MEMORY_SCOPE_AGENT);
      for (int j = 0; j < world; j++)
        __hip_atomic_store(&peers.flags[j][rank], seq, __ATOMIC_RELEASE,
                           __HIP_MEMORY_SCOPE_SYSTEM);
    }
    rdna_ar_copy(out, in, n, gid, gstride, 0);
    return;
  }

  // 1. push our slice into every peer's staging slot for us (posted PCIe
  // writes).
  //    The peer order is staggered by rank, so at any instant each destination
  //    is being written by ONE source instead of all W-1 at once; `pace` idles
  //    the wave ~64 clocks per unit between strided stores to bound the burst
  //    rate (0 = off). Both are for the rest of the machine: these pushes land
  //    in the receiving GPU's root complex, where other devices' DMA
  //    completions queue behind them (2026-09-01: the SAS HBA on this box lives
  //    there).
  for (int k = 1; k < world; k++) {
    const int j = (rank + k) % world;
    T* dst = reinterpret_cast<T*>(peers.stage[j]) +
             ((long long)p * world + rank) * max_elems;
    // 16-byte stores where the pointers allow: the staging is uncached, so the
    // push is bounded by PCIe transactions, not by bytes.
    rdna_ar_copy(dst, in, n, gid, gstride, pace);
  }
  __syncthreads();

  // 2. grid barrier (all our blocks have pushed), payload-before-flag,
  // announce, wait
  if (t == 0) {
    __threadfence_system();
    atomicAdd(&arrive[p], 1u);
    if (b == 0) {
      const long long t0 = wall_clock64();
      while (__hip_atomic_load(&arrive[p], __ATOMIC_ACQUIRE,
                               __HIP_MEMORY_SCOPE_AGENT) < (unsigned)nblocks) {
        RDNA_AR_POLL_PAUSE();
        const long long dt = wall_clock64() - t0;
        if (dt > wait_ticks) {
          rdna_ar_abort_ms(timeout, report, 1u, (unsigned)rank, seq,
                           dt / ticks_per_ms);
          s_abort = 1;
          break;
        }
      }
      if (!s_abort) {
        // Clear both parity counters. Every block has already atomicAdded.
        arrive[p] = 0u;
        arrive[1 - p] = 0u;
        __hip_atomic_store(seqbuf, seq, __ATOMIC_RELEASE,
                           __HIP_MEMORY_SCOPE_AGENT);
        // announce: one posted P2P store into every peer's slot for us (and our
        // own)
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
        const long long t0 = wall_clock64();
        long long dt = 0;
        while (__hip_atomic_load(&myflags[j], __ATOMIC_ACQUIRE,
                                 __HIP_MEMORY_SCOPE_SYSTEM) < seq) {
          RDNA_AR_POLL_PAUSE();
          dt = wall_clock64() - t0;
          if (dt > wait_ticks) {
            rdna_ar_abort_ms(timeout, report, 2u, (unsigned)j, seq,
                             dt / ticks_per_ms);
            s_abort = 1;
            break;
          }
        }
        if (!s_abort && b == 0 && dt > RDNA_AR_SLOW_MS * ticks_per_ms) {
          __hip_atomic_store(
              report + 1,
              rdna_ar_ms_code(2u, (unsigned)j, seq, dt / ticks_per_ms),
              __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        }
      }
    }
  }
  __syncthreads();
  if (s_abort) {
    // Timed out here: leave a finite (wrong) value, never stale memory.
    rdna_ar_copy(out, in, n, gid, gstride, 0);
    return;
  }

  // 3. fixed-order reduction: rank 0 .. W-1, fp32, identical on every rank
  const T* mine = reinterpret_cast<const T*>(peers.stage[rank]) +
                  ((long long)p * world) * max_elems;
  rdna_ar_reduce(mine, in, out, n, rank, world, max_elems, gid, gstride);
}
