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
//   * spins that set a sticky abort flag instead of hanging the GPU -- and,
//     since T44b, record which phase/peer/sequence aborted in a host-visible
//     word (see below). Since 2026-11-04 only a lost write aborts early; a
//     late peer is waited for (see RDNA_AR_WAIT_MS).
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
// One-shot waits are measured in wall-clock time (wall_clock64, constant rate
// from hipDeviceAttributeWallClockRate), not in polls (a poll costs ~0.4 us on
// gfx1030).
//
// A wait has two possible causes, and only one of them is a fault:
//   * the peer has not reached the collective yet: rank skew. A rank arrives
//     seconds late whenever it JIT-compiles a kernel or runs a long host step
//     (Flash-Next serving, 2026-11-04: one rank 7.3 s late, then every rank
//     stalled ~60 s outside rdna_ar). Aborting does not help -- the engine
//     cannot step without that rank -- so the bound is past the engine's
//     300 s execute timeout: a rank that never comes back ends the engine
//     through that timeout, a late one is waited for. With the old 120 s
//     bound, both wedges since 2026-10-09 (#199523 on 11-03, #593530 on
//     11-04) were one rank's flag missing on every waiting rank at once (the
//     11-03 trace shows that rank had stopped dispatching), reported as a
//     lost P2P write.
//   * the peer HAS reached it (its own flag slot, written last in its
//     announce, shows the sequence number) but its store into our slot never
//     landed: a lost posted P2P write. That is aborted RDNA_AR_LOST_MS after
//     the peer's own slot was seen.
// After RDNA_AR_SLOW_MS of waiting, the waiter reads the peer's own slot over
// P2P every RDNA_AR_PROBE_MS to tell the two apart, and block 0 refreshes a
// live record in report[1] that a host thread can log while the GPU spins.
// VLLM_RDNA_AR_WAIT_MS overrides RDNA_AR_WAIT_MS (tests use a short bound).
#define RDNA_AR_WAIT_MS 600000ll
// One-shot waits longer than this that still complete are recorded in
// report[1] (abort-record layout, phase 2) and logged as a late peer.
#define RDNA_AR_SLOW_MS 2000ll
#define RDNA_AR_PROBE_MS 1000ll
#define RDNA_AR_LOST_MS 10000ll
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
//   bits 0-7   bit 0 = 1 (a record), bit 1 = wait still in progress (live
//              report[1] only), bit 2 = the peer's own flag slot showed this
//              collective (it announced: our slot's write was lost or late)
//   bits 8-11  phase: 1 = own blocks' grid barrier, 2 = a peer's flag
//   bits 12-15 peer rank (phase 2)
//   bits 16-31 measured wall-clock wait in 16 ms units (saturating, ~17 min)
//   bits 32-63 sequence number of the collective
#define RDNA_AR_REC_LIVE 2u
#define RDNA_AR_REC_SEEN 4u
__device__ __forceinline__ unsigned long long rdna_ar_ms_code(
    unsigned phase, unsigned peer, int seq, long long ms, unsigned bits = 0u) {
  const unsigned long long u =
      ms / 16 > 0xFFFFll ? 0xFFFFull : (unsigned long long)(ms / 16);
  return (1ull | bits) | ((unsigned long long)(phase & 0xFu) << 8) |
         ((unsigned long long)(peer & 0xFu) << 12) | (u << 16) |
         ((unsigned long long)(unsigned)seq << 32);
}

__device__ __forceinline__ void rdna_ar_abort_ms(unsigned* timeout,
                                                 unsigned long long* report,
                                                 unsigned phase, unsigned peer,
                                                 int seq, long long ms,
                                                 unsigned bits = 0u) {
  if (atomicCAS(timeout, 0u, 1u) == 0u) {
    __hip_atomic_store(report, rdna_ar_ms_code(phase, peer, seq, ms, bits),
                       __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

// Flags hold sequence numbers; compare them in serial-number order so the
// protocol survives the 32-bit wrap (~19 days of decode at ~1300 collectives
// per second): true when `flag` is a collective before `seq`.
__device__ __forceinline__ bool rdna_ar_before(int flag, int seq) {
  return (int)((unsigned)flag - (unsigned)seq) < 0;
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
// (push to stage[j] slot (parity, rank), bump *seqbuf, rdna_ar_announce --
// flags[j][rank] for every peer j, our own slot last -- wait until
// flags[rank][j] is no longer rdna_ar_before(seq), sum the slots in rank
// order).
struct RdnaArDev {
  RdnaArPeers peers;
  int* seqbuf;
  long long max_bytes;  // per slot
  int rank, world, ready;
};
// Host side (rdna_allreduce.cu): false if the handle is not a ready instance.
bool rdna_ar_device(long long handle, RdnaArDev* out);

// Announce `seq`: one posted P2P store into every peer's slot for us, our own
// slot last. A peer that reads our own slot (the probe in the wait) then knows
// that every store we sent it has already left this GPU.
__device__ __forceinline__ void rdna_ar_announce(const RdnaArPeers& peers,
                                                 int rank, int world, int seq) {
  for (int k = 1; k <= world; k++) {
    const int j = (rank + k) % world;
    __hip_atomic_store(&peers.flags[j][rank], seq, __ATOMIC_RELEASE,
                       __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

// Has peer j announced collective `seq`? One P2P read of its own flag slot;
// issued only on slow waits (past RDNA_AR_SLOW_MS, once per RDNA_AR_PROBE_MS).
__device__ __forceinline__ bool rdna_ar_peer_announced(const RdnaArPeers& peers,
                                                       int j, int seq) {
  return !rdna_ar_before(__hip_atomic_load(&peers.flags[j][j], __ATOMIC_RELAXED,
                                           __HIP_MEMORY_SCOPE_SYSTEM),
                         seq);
}

#define RDNA_AR_POLL_PAUSE() __builtin_amdgcn_s_sleep(8)

// ---------------------------------------------------------------------------
// Phases 1-2 of the protocol, shared by every one-shot collective: push our
// payload into every peer's staging slot, grid barrier, announce, wait for
// every peer. What differs per collective is only the finisher that reads the
// slots afterwards (sum / concatenate / argmax), so it lives in the kernel.
//
// `load(i)` yields element i of OUR payload (a plain read for all-reduce and
// all-gather, a (value, index) pack for the argmax). Returns true when the
// exchange aborted: the caller must leave a finite fallback in its output and
// return. Otherwise `p` is the staging parity the finisher reads from.
// ---------------------------------------------------------------------------
template <typename T, typename Load>
__device__ __forceinline__ bool rdna_ar_exchange(
    const Load& load, const RdnaArPeers& peers, unsigned int* arrive,
    int* seqbuf, unsigned* timeout, unsigned long long* report, int rank,
    int world, int n, long long max_elems, int nblocks, int pace,
    long long wait_ticks, long long ticks_per_ms, int& p) {
  __shared__ int s_seq;
  __shared__ int s_abort;
  const int t = threadIdx.x, nt = blockDim.x, b = blockIdx.x;
  if (t == 0) {
    // Unsigned: the sequence wraps (flags are compared with rdna_ar_before).
    s_seq = (int)((unsigned)__hip_atomic_load(seqbuf, __ATOMIC_ACQUIRE,
                                              __HIP_MEMORY_SCOPE_AGENT) +
                  1u);
    // Sticky abort (see the T44b note): an earlier collective on this
    // instance timed out, so this one must not wait again.
    s_abort = __hip_atomic_load(timeout, __ATOMIC_RELAXED,
                                __HIP_MEMORY_SCOPE_AGENT) != 0u;
  }
  __syncthreads();
  const int seq = s_seq;
  p = seq & 1;
  const int gid = b * nt + t, gstride = nblocks * nt;
  if (s_abort) {
    if (b == 0 && t == 0) {
      // Keep the sequence moving and release peers waiting on this rank.
      __hip_atomic_store(seqbuf, seq, __ATOMIC_RELEASE,
                         __HIP_MEMORY_SCOPE_AGENT);
      rdna_ar_announce(peers, rank, world, seq);
    }
    return true;
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
    for (int i = gid; i < n; i += gstride) {
      dst[i] = load(i);
      for (int q = 0; q < pace; q++) __builtin_amdgcn_s_sleep(1);
    }
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
        rdna_ar_announce(peers, rank, world, seq);
      }
    }
    if (!s_abort) {
      // wait: poll OUR flag slots (local uncached memory, no PCIe traffic).
      // Past RDNA_AR_SLOW_MS, once per RDNA_AR_PROBE_MS: probe whether the
      // peer has announced, give up on a lost write (announced, nothing here
      // after RDNA_AR_LOST_MS) or past wait_ticks, follow an abort claimed by
      // another block, and (block 0) refresh the live record in report[1].
      int* myflags = peers.flags[rank];
      const long long slow_ticks = RDNA_AR_SLOW_MS * ticks_per_ms;
      for (int j = 0; j < world && !s_abort; j++) {
        if (j == rank) continue;
        const long long t0 = wall_clock64();
        long long dt = 0, next = slow_ticks, seen = -1;
        unsigned bits = 0u;
        bool live = false;
        while (rdna_ar_before(__hip_atomic_load(&myflags[j], __ATOMIC_ACQUIRE,
                                                __HIP_MEMORY_SCOPE_SYSTEM),
                              seq)) {
          RDNA_AR_POLL_PAUSE();
          dt = wall_clock64() - t0;
          if (dt < next && dt <= wait_ticks) continue;
          next = dt + RDNA_AR_PROBE_MS * ticks_per_ms;
          if (seen < 0 && rdna_ar_peer_announced(peers, j, seq)) {
            seen = dt;
            bits = RDNA_AR_REC_SEEN;
          }
          const bool lost =
              seen >= 0 && dt - seen > RDNA_AR_LOST_MS * ticks_per_ms;
          if (lost || dt > wait_ticks ||
              __hip_atomic_load(timeout, __ATOMIC_RELAXED,
                                __HIP_MEMORY_SCOPE_AGENT) != 0u) {
            // This wave may have been descheduled since its last poll.
            if (!rdna_ar_before(__hip_atomic_load(&myflags[j], __ATOMIC_ACQUIRE,
                                                  __HIP_MEMORY_SCOPE_SYSTEM),
                                seq))
              break;
            // No-op when another block already claimed the abort.
            rdna_ar_abort_ms(timeout, report, 2u, (unsigned)j, seq,
                             dt / ticks_per_ms, bits);
            s_abort = 1;
            break;
          }
          if (b == 0) {
            __hip_atomic_store(
                report + 1,
                rdna_ar_ms_code(2u, (unsigned)j, seq, dt / ticks_per_ms,
                                RDNA_AR_REC_LIVE | bits),
                __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            live = true;
          }
        }
        if (!s_abort && b == 0 && (live || dt > slow_ticks)) {
          __hip_atomic_store(
              report + 1,
              rdna_ar_ms_code(2u, (unsigned)j, seq, dt / ticks_per_ms, bits),
              __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        }
      }
    }
  }
  __syncthreads();
  return s_abort != 0;
}

template <typename T>
struct RdnaArLoadDirect {
  const T* in;
  __device__ __forceinline__ T operator()(int i) const { return in[i]; }
};

// All-reduce finisher, vector path: one 16-byte load per rank slice, the
// fixed-order fp32 sum (rank 0 .. W-1, starting from 0.f, exactly the scalar
// loop's order, so the result is bit-equal to it and identical on every rank),
// one 16-byte store. W is a template argument so the W uncached loads issue
// back to back: 4 ranks, 120 KB, graph replay behind a 200 us busy kernel:
// scalar finisher 55.1 us, vector finisher 44.8 us (see
// docs/rdna2/allreduce-vector-finisher-2026-11-02.md).
template <typename T, int W>
__device__ __forceinline__ void rdna_ar_sum_vec(
    const T* __restrict__ in, T* __restrict__ out, const T* __restrict__ mine,
    int rank, long long max_elems, int nvec, int gid, int gstride) {
  constexpr int kVec = (int)(16 / sizeof(T));
  const auto* in4 = reinterpret_cast<const uint4*>(in);
  auto* out4 = reinterpret_cast<uint4*>(out);
  for (int v = gid; v < nvec; v += gstride) {
    uint4 x[W];
#pragma unroll
    for (int j = 0; j < W; j++) {
      const uint4* src =
          (j == rank)
              ? in4
              : reinterpret_cast<const uint4*>(mine + (long long)j * max_elems);
      x[j] = src[v];
    }
    float acc[kVec];
#pragma unroll
    for (int q = 0; q < kVec; q++) acc[q] = 0.f;
#pragma unroll
    for (int j = 0; j < W; j++) {
      const T* e = reinterpret_cast<const T*>(&x[j]);
#pragma unroll
      for (int q = 0; q < kVec; q++) acc[q] += rdna_ar_to_f(e[q]);
    }
    uint4 r;
    T* re = reinterpret_cast<T*>(&r);
#pragma unroll
    for (int q = 0; q < kVec; q++) rdna_ar_from_f(re[q], acc[q]);
    out4[v] = r;
  }
}

// All-reduce finisher: fixed-order fp32 sum, identical on every rank.
template <typename T>
__global__ void rdna_ar_oneshot(
    const T* __restrict__ in, T* __restrict__ out,
    RdnaArPeers peers,           // stage + flags per rank
    unsigned int* arrive,        // device, [2] per parity
    int* seqbuf,                 // device, local seq mirror
    unsigned* timeout,           // device, sticky abort claim
    unsigned long long* report,  // host-mapped abort record (T44b)
    int rank, int world, int n, long long max_elems, int nblocks, int pace,
    long long wait_ticks,    // wall-clock bound per wait
    long long ticks_per_ms,  // wall_clock64 rate
    int vec) {               // 0: scalar finisher (VLLM_RDNA_AR_VEC=0)
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int gstride = nblocks * blockDim.x;
  int p = 0;
  if (rdna_ar_exchange<T>(RdnaArLoadDirect<T>{in}, peers, arrive, seqbuf,
                          timeout, report, rank, world, n, max_elems, nblocks,
                          pace, wait_ticks, ticks_per_ms, p)) {
    // Timed out (or sticky abort): leave a finite (wrong) value, never stale
    // memory.
    for (int i = gid; i < n; i += gstride) out[i] = in[i];
    return;
  }

  // 3. fixed-order reduction: rank 0 .. W-1, fp32, identical on every rank
  const T* mine = reinterpret_cast<const T*>(peers.stage[rank]) +
                  ((long long)p * world) * max_elems;
  // Whole 16-byte groups take the vector path (2, 4 and 8 ranks, 16-byte
  // aligned in/out); the remaining elements, and every other case, the scalar
  // loop below. Staging slots are 16-byte aligned by construction.
  constexpr int kVec = (int)(16 / sizeof(T));
  int done = 0;
  if (vec && (((uintptr_t)in | (uintptr_t)out) & 15u) == 0) {
    const int nvec = n / kVec;
    if (world == 4) {
      rdna_ar_sum_vec<T, 4>(in, out, mine, rank, max_elems, nvec, gid, gstride);
      done = nvec * kVec;
    } else if (world == 2) {
      rdna_ar_sum_vec<T, 2>(in, out, mine, rank, max_elems, nvec, gid, gstride);
      done = nvec * kVec;
    } else if (world == 8) {
      rdna_ar_sum_vec<T, 8>(in, out, mine, rank, max_elems, nvec, gid, gstride);
      done = nvec * kVec;
    }
  }
  for (int i = done + gid; i < n; i += gstride) {
    float v = 0.f;
    for (int j = 0; j < world; j++)
      v += (j == rank) ? rdna_ar_to_f(in[i])
                       : rdna_ar_to_f(mine[(long long)j * max_elems + i]);
    rdna_ar_from_f(out[i], v);
  }
}

// All-gather finisher (last dim): `in` is [rows, cols] row-major, `out` is
// [rows, world * cols] with rank j's block at columns [j*cols, (j+1)*cols) --
// the layout all_gather_into_tensor + movedim produces. A pure copy, so the
// result is bit-equal to RCCL's.
template <typename T>
__global__ void rdna_ag_oneshot(const T* __restrict__ in, T* __restrict__ out,
                                RdnaArPeers peers, unsigned int* arrive,
                                int* seqbuf, unsigned* timeout,
                                unsigned long long* report, int rank, int world,
                                int n, int cols, long long max_elems,
                                int nblocks, int pace, long long wait_ticks,
                                long long ticks_per_ms) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int gstride = nblocks * blockDim.x;
  int p = 0;
  const bool aborted = rdna_ar_exchange<T>(
      RdnaArLoadDirect<T>{in}, peers, arrive, seqbuf, timeout, report, rank,
      world, n, max_elems, nblocks, pace, wait_ticks, ticks_per_ms, p);
  const T* mine = reinterpret_cast<const T*>(peers.stage[rank]) +
                  ((long long)p * world) * max_elems;
  for (int i = gid; i < n; i += gstride) {
    const long long row = i / cols;
    const long long col = i - row * cols;
    T* dst = out + row * world * cols + col;
    for (int j = 0; j < world; j++) {
      // Aborted: a finite (wrong) value in every block, never stale memory.
      dst[(long long)j * cols] =
          (aborted || j == rank) ? in[i] : mine[(long long)j * max_elems + i];
    }
  }
}

// Argmax finisher for vocab-parallel sampling: each rank pushes one
// (value, global index) pair per row; every rank picks the rank with the
// largest value. Ties go to the lowest rank, NaN wins -- exactly what
// torch.argmax over the gathered [rows, world] values returns -- so the result
// is bit-equal to the all_gather + argmax + gather it replaces.
struct RdnaArPair {
  float v;
  int i;
};

template <typename V>
struct RdnaArLoadPair {
  const V* vals;
  const int64_t* idx;
  __device__ __forceinline__ RdnaArPair operator()(int i) const {
    return RdnaArPair{rdna_ar_to_f(vals[i]), (int)idx[i]};
  }
};

__device__ __forceinline__ bool rdna_ar_pair_wins(float cand, float best) {
  // torch.argmax semantics: NaN is the maximum; otherwise strictly greater.
  return !isnan(best) && (isnan(cand) || cand > best);
}

template <typename V>
__global__ void rdna_ar_argmax_oneshot(
    const V* __restrict__ vals, const int64_t* __restrict__ idx,
    int64_t* __restrict__ out, RdnaArPeers peers, unsigned int* arrive,
    int* seqbuf, unsigned* timeout, unsigned long long* report, int rank,
    int world, int n, long long max_elems, int nblocks, int pace,
    long long wait_ticks, long long ticks_per_ms) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int gstride = nblocks * blockDim.x;
  const RdnaArLoadPair<V> load{vals, idx};
  int p = 0;
  if (rdna_ar_exchange<RdnaArPair>(load, peers, arrive, seqbuf, timeout, report,
                                   rank, world, n, max_elems, nblocks, pace,
                                   wait_ticks, ticks_per_ms, p)) {
    for (int i = gid; i < n; i += gstride) out[i] = idx[i];
    return;
  }
  const RdnaArPair* mine =
      reinterpret_cast<const RdnaArPair*>(peers.stage[rank]) +
      ((long long)p * world) * max_elems;
  for (int i = gid; i < n; i += gstride) {
    RdnaArPair best = (rank == 0) ? load(i) : mine[i];
    for (int j = 1; j < world; j++) {
      const RdnaArPair cand =
          (j == rank) ? load(i) : mine[(long long)j * max_elems + i];
      if (rdna_ar_pair_wins(cand.v, best.v)) best = cand;
    }
    out[i] = best.i;
  }
}
