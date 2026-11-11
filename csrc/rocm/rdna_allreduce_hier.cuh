// Hierarchical (2-level) push all-reduce for K=2 PCIe switch islands of G
// ranks each (world = 2G, G = 2..4; 2 x 4 on the 8-V620 box: slot A behind
// root complex c0, slot B behind root 00). Built on the rdna_allreduce.cuh
// protocol: uncached IPC staging, posted P2P pushes, flags in each rank's own
// uncached page, wall-clock-bounded waits with the same abort record, sticky
// abort and late-peer record.
//
// Why it exists (2026-11-06 measurements,
// docs/rdna2/rdna-ar-hier-2026-11-06.md): the flat one-shot pushes every rank's
// payload to every other rank, which at world 8 is 16 concurrent cross-root
// flows of the whole payload; the cross path caps at ~27 GB/s per direction and
// collapses with 3 or more independent flows. These kernels move one island's
// partial sum across the root complex instead, in at most G flows of N/G bytes.
//
// Algorithm "hier" (reduce-scatter, cross exchange, all-gather), X <= G
// "owners" per island (local index 0..X-1); the payload is cut in per-block
// chunks and each chunk in X owner ranges:
//   phase 0  every island member pushes owner o's range of its input to o;
//            owner o sums the G contributions in local-index order (fp32) and
//            rounds the island partial to T
//   phase 1  owner o pushes its partial to its partner (same local index in
//            the other island); a flag per (phase, block, source) announces it
//   phase 2  owner o adds the two partials (fp32, rounded to T once more) and
//            pushes the result to its island peers (all-gather)
// Every rank ends with bit-identical output. rdna_ar_oneshot_isl is the flat
// exchange with the same arithmetic (island partials rounded to T, then
// added): it agrees bit for bit with rdna_ar_hier and runs only when
// VLLM_RDNA_AR_HIER_ALGO=flat forces it (diagnostics, tests, benchmarks). The
// size dispatch keeps the legacy flat one-shot below the hier threshold.
//
// Blocks run independent pipelines over disjoint element ranges: no grid
// barrier, per-block flags, the last block to finish bumps the sequence. A
// block only waits for the same block of the peers it exchanged with, so
// blocks that are not resident yet cannot deadlock the others (dispatch is in
// block order on every rank).
//
// Slot reuse: slot(parity, src) of a rank's staging is shared by all phases.
// Within one collective the ranges never overlap (reduce-scatter data from
// src covers the receiver's own range, all-gather data from src covers src's
// range, the partner's partial covers the receiver's range in a slot no
// island peer writes). Across collectives: a rank starts collective c + 2
// (the next one of the same parity) after its collective c + 1 completed, and
// every output depends on every rank's input, so every rank had started
// c + 1, i.e. had finished reading the slots of c. Whatever the algorithms of
// c, c + 1 and c + 2 are, all slots of that parity are free.
#pragma once
#include "rdna_allreduce.cuh"

#define RDNA_AR_MAX_G 4  // ranks per island (two islands, world <= 8)
// Hier flags live after the flat ones in the 4 KB flag page (ints): flat uses
// [0, RDNA_AR_MAX_WORLD); index = base + (phase * blocks + block) * world +
// src.
#define RDNA_AR_HFLAG_BASE 16
#define RDNA_AR_HFLAG_BLOCKS 32  // blocks of a hier launch at most
#define RDNA_AR_HPHASES 3
static_assert(RDNA_AR_HFLAG_BASE + RDNA_AR_HPHASES * RDNA_AR_HFLAG_BLOCKS *
                                       RDNA_AR_MAX_WORLD <=
                  RDNA_AR_FLAG_PAGE / 4,
              "hier flags must fit the flag page");
// Record phase codes (bits 8-11) of the hier waits.
#define RDNA_AR_PH_RS 3u  // phase 0: an island peer's reduce-scatter push
#define RDNA_AR_PH_XR 4u  // phase 1: the partner's cross-root partial
#define RDNA_AR_PH_AG 5u  // phase 2: an owner's all-gather push

// Algorithm ids of rdna_ar_all_reduce_algo / the size dispatch.
#define RDNA_AR_ALGO_FLAT 0      // flat one-shot, rank-order fp32 sum
#define RDNA_AR_ALGO_FLAT_ISL 1  // flat exchange, island-grouped arithmetic
#define RDNA_AR_ALGO_HIER 2      // reduce-scatter, cross exchange, all-gather

struct RdnaArTopo {
  int G;                       // ranks per island
  int X;                       // owners per island (1..G)
  int k, l;                    // this rank's island (0/1) and local index
  int kof[RDNA_AR_MAX_WORLD];  // island of each rank
  int isl[2][RDNA_AR_MAX_G];   // rank at (island, local index), ascending
};

__host__ __device__ __forceinline__ int rdna_ar_hflag(int ph, int b, int src) {
  return RDNA_AR_HFLAG_BASE +
         (ph * RDNA_AR_HFLAG_BLOCKS + b) * RDNA_AR_MAX_WORLD + src;
}

// Roles of rank (k, l) in phase `ph`. Phase 0: every member tells the owners
// (never itself); phase 1: owners tell their partner; phase 2: owners tell all
// island peers. Owners wait for all island peers in phase 0, for the partner
// in phase 1; everyone waits for the owners (but itself) in phase 2. A flag is
// announced exactly where a peer waits for it. No local arrays: a local array
// indexed at run time goes to scratch memory.

// The t-th member of our island but ourselves (t < G - 1).
__device__ __forceinline__ int rdna_ar_hmember(const RdnaArTopo& tp, int t) {
  return tp.isl[tp.k][t < tp.l ? t : t + 1];
}

// The t-th owner of our island but ourselves (t < X, or X - 1 for an owner).
__device__ __forceinline__ int rdna_ar_howner(const RdnaArTopo& tp, int t) {
  return tp.isl[tp.k][(tp.l < tp.X && t >= tp.l) ? t + 1 : t];
}

// The t-th rank we announce `ph` to, or -1 past the last one.
__device__ __forceinline__ int rdna_ar_htarget(const RdnaArTopo& tp, int ph,
                                               int t) {
  const bool owner = tp.l < tp.X;
  if (ph == 0) return t < tp.X - (owner ? 1 : 0) ? rdna_ar_howner(tp, t) : -1;
  if (!owner) return -1;
  if (ph == 1) return t == 0 ? tp.isl[1 - tp.k][tp.l] : -1;
  return t < tp.G - 1 ? rdna_ar_hmember(tp, t) : -1;
}

// The t-th rank we wait for in `ph`, or -1 past the last one.
__device__ __forceinline__ int rdna_ar_hsource(const RdnaArTopo& tp, int ph,
                                               int t) {
  const bool owner = tp.l < tp.X;
  if (ph == 0) return owner && t < tp.G - 1 ? rdna_ar_hmember(tp, t) : -1;
  if (ph == 1) return owner && t == 0 ? tp.isl[1 - tp.k][tp.l] : -1;
  return t < tp.X - (owner ? 1 : 0) ? rdna_ar_howner(tp, t) : -1;
}

// One 16-byte unit = V elements of T. The staging slots always hold whole
// units; `in` and `out` are touched element-wise at a ragged tail and when
// they are not 16-byte aligned.
template <typename T>
__device__ __forceinline__ uint4 rdna_ar_load_unit(const T* in, int u, int n,
                                                   bool aligned) {
  constexpr int V = (int)(16 / sizeof(T));
  if (aligned && (u + 1) * V <= n) return reinterpret_cast<const uint4*>(in)[u];
  uint4 r = {0u, 0u, 0u, 0u};
  T* e = reinterpret_cast<T*>(&r);
  for (int q = 0; q < V; q++) {
    const int i = u * V + q;
    if (i < n) e[q] = in[i];
  }
  return r;
}

template <typename T>
__device__ __forceinline__ void rdna_ar_store_unit(T* out, int u, int n,
                                                   bool aligned,
                                                   const uint4& x) {
  constexpr int V = (int)(16 / sizeof(T));
  if (aligned && (u + 1) * V <= n) {
    reinterpret_cast<uint4*>(out)[u] = x;
    return;
  }
  const T* e = reinterpret_cast<const T*>(&x);
  for (int q = 0; q < V; q++) {
    const int i = u * V + q;
    if (i < n) out[i] = e[q];
  }
}

template <typename T>
__device__ __forceinline__ void rdna_ar_unit_acc(float (&acc)[16 / sizeof(T)],
                                                 const uint4& x) {
  const T* e = reinterpret_cast<const T*>(&x);
#pragma unroll
  for (int q = 0; q < (int)(16 / sizeof(T)); q++) acc[q] += rdna_ar_to_f(e[q]);
}

template <typename T>
__device__ __forceinline__ uint4
rdna_ar_unit_pack(const float (&acc)[16 / sizeof(T)]) {
  uint4 r;
  T* re = reinterpret_cast<T*>(&r);
#pragma unroll
  for (int q = 0; q < (int)(16 / sizeof(T)); q++) rdna_ar_from_f(re[q], acc[q]);
  return r;
}

// out = round_T(f(a) + f(b)) per element.
template <typename T>
__device__ __forceinline__ uint4 rdna_ar_unit_add2(const uint4& a,
                                                   const uint4& b) {
  const T* ea = reinterpret_cast<const T*>(&a);
  const T* eb = reinterpret_cast<const T*>(&b);
  uint4 r;
  T* er = reinterpret_cast<T*>(&r);
#pragma unroll
  for (int q = 0; q < (int)(16 / sizeof(T)); q++)
    rdna_ar_from_f(er[q], rdna_ar_to_f(ea[q]) + rdna_ar_to_f(eb[q]));
  return r;
}

struct RdnaArHCtx {
  unsigned* timeout;
  unsigned long long* report;
  int rank, world, seq, p, b;
  long long max_bytes, wait_ticks, ticks_per_ms;
};

__device__ __forceinline__ uint4* rdna_ar_hslot(const RdnaArPeers& peers,
                                                const RdnaArHCtx& c, int target,
                                                int src) {
  return reinterpret_cast<uint4*>(static_cast<char*>(peers.stage[target]) +
                                  ((long long)c.p * c.world + src) *
                                      c.max_bytes);
}

// Our own staging slot of source `src` (what `src` pushed to us this parity).
__device__ __forceinline__ const uint4* rdna_ar_hmine(const RdnaArPeers& peers,
                                                      const RdnaArHCtx& c,
                                                      int src) {
  return reinterpret_cast<const uint4*>(
      static_cast<const char*>(peers.stage[c.rank]) +
      ((long long)c.p * c.world + src) * c.max_bytes);
}

// Announce phase `ph` of this block (thread 0): the targets' slots first, our
// own slot last, as in rdna_ar_announce. Ranks without a role do nothing.
__device__ __forceinline__ void rdna_ar_hannounce(const RdnaArPeers& peers,
                                                  const RdnaArTopo& tp,
                                                  const RdnaArHCtx& c, int ph) {
  const int idx = rdna_ar_hflag(ph, c.b, c.rank);
  bool any = false;
  for (int t = 0; t < RDNA_AR_MAX_G; t++) {
    const int j = rdna_ar_htarget(tp, ph, t);
    if (j < 0) break;
    __hip_atomic_store(&peers.flags[j][idx], c.seq, __ATOMIC_RELEASE,
                       __HIP_MEMORY_SCOPE_SYSTEM);
    any = true;
  }
  if (any)
    __hip_atomic_store(&peers.flags[c.rank][idx], c.seq, __ATOMIC_RELEASE,
                       __HIP_MEMORY_SCOPE_SYSTEM);
}

// Wait for the flags of phase `ph` of this block. All threads call it; lane t
// of the block polls the flag of the t-th source in our own flag page (local
// uncached memory) with the same bounds, probe and records as the flat wait
// (rdna_ar_exchange), then a barrier publishes s_abort. Block 0 keeps the live
// / late record in report[1].
__device__ __forceinline__ void rdna_ar_hwait(const RdnaArPeers& peers,
                                              const RdnaArTopo& tp,
                                              const RdnaArHCtx& c, int ph,
                                              int* s_abort) {
  const int j =
      threadIdx.x < RDNA_AR_MAX_G ? rdna_ar_hsource(tp, ph, threadIdx.x) : -1;
  if (j >= 0) {
    const int idx = rdna_ar_hflag(ph, c.b, j);
    const int* mine = &peers.flags[c.rank][idx];
    const long long slow_ticks = RDNA_AR_SLOW_MS * c.ticks_per_ms;
    const long long t0 = wall_clock64();
    long long dt = 0, next = slow_ticks, seen = -1;
    unsigned bits = 0u;
    bool live = false;
    while (rdna_ar_before(
        __hip_atomic_load(mine, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM),
        c.seq)) {
      RDNA_AR_POLL_PAUSE();
      dt = wall_clock64() - t0;
      if (dt < next && dt <= c.wait_ticks) continue;
      next = dt + RDNA_AR_PROBE_MS * c.ticks_per_ms;
      if (seen < 0 &&
          !rdna_ar_before(
              __hip_atomic_load(&peers.flags[j][idx], __ATOMIC_RELAXED,
                                __HIP_MEMORY_SCOPE_SYSTEM),
              c.seq)) {
        seen = dt;
        bits = RDNA_AR_REC_SEEN;
      }
      const bool lost =
          seen >= 0 && dt - seen > RDNA_AR_LOST_MS * c.ticks_per_ms;
      if (lost || dt > c.wait_ticks ||
          __hip_atomic_load(c.timeout, __ATOMIC_RELAXED,
                            __HIP_MEMORY_SCOPE_AGENT) != 0u) {
        // This wave may have been descheduled since its last poll.
        if (!rdna_ar_before(__hip_atomic_load(mine, __ATOMIC_ACQUIRE,
                                              __HIP_MEMORY_SCOPE_SYSTEM),
                            c.seq))
          break;
        rdna_ar_abort_ms(c.timeout, c.report, RDNA_AR_PH_RS + (unsigned)ph,
                         (unsigned)j, c.seq, dt / c.ticks_per_ms, bits);
        *s_abort = 1;
        break;
      }
      if (c.b == 0) {
        __hip_atomic_store(
            c.report + 1,
            rdna_ar_ms_code(RDNA_AR_PH_RS + (unsigned)ph, (unsigned)j, c.seq,
                            dt / c.ticks_per_ms, RDNA_AR_REC_LIVE | bits),
            __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        live = true;
      }
    }
    if (!*s_abort && c.b == 0 && (live || dt > slow_ticks)) {
      __hip_atomic_store(
          c.report + 1,
          rdna_ar_ms_code(RDNA_AR_PH_RS + (unsigned)ph, (unsigned)j, c.seq,
                          dt / c.ticks_per_ms, bits),
          __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
  }
  __syncthreads();
}

// Per-block geometry, identical on every rank: the chunk of block b in units
// and the X owner ranges inside it.
struct RdnaArHGeom {
  int lo, hi;  // block chunk [lo, hi) in units
  int sc;      // units per owner range
};

__device__ __forceinline__ RdnaArHGeom rdna_ar_hgeom(int nu, int nblocks, int b,
                                                     int X) {
  const int cu = (nu + nblocks - 1) / nblocks;
  RdnaArHGeom g;
  g.lo = b * cu;
  g.hi = min(g.lo + cu, nu);
  g.sc = (max(g.hi - g.lo, 0) + X - 1) / X;
  return g;
}

__device__ __forceinline__ int rdna_ar_hrange_lo(const RdnaArHGeom& g, int o) {
  return min(g.lo + o * g.sc, g.hi);
}
__device__ __forceinline__ int rdna_ar_hrange_hi(const RdnaArHGeom& g, int o) {
  return min(g.lo + (o + 1) * g.sc, g.hi);
}

// Reduce-scatter, cross exchange, all-gather. See the file comment.
template <typename T, int G>
__global__ void __launch_bounds__(256)
    rdna_ar_hier(const T* __restrict__ in, T* __restrict__ out,
                 RdnaArPeers peers, unsigned int* arrive, int* seqbuf,
                 unsigned* timeout, unsigned long long* report, int rank,
                 int world, int n, long long max_bytes, int nblocks,
                 long long wait_ticks, long long ticks_per_ms, RdnaArTopo tp) {
  constexpr int V = (int)(16 / sizeof(T));
  __shared__ int s_seq;
  __shared__ int s_abort;
  const int t = threadIdx.x, nt = blockDim.x, b = blockIdx.x;
  if (t == 0) {
    s_seq = (int)((unsigned)__hip_atomic_load(seqbuf, __ATOMIC_ACQUIRE,
                                              __HIP_MEMORY_SCOPE_AGENT) +
                  1u);
    // Sticky abort (T44b): an earlier collective timed out; do not wait again.
    s_abort = __hip_atomic_load(timeout, __ATOMIC_RELAXED,
                                __HIP_MEMORY_SCOPE_AGENT) != 0u;
  }
  __syncthreads();
  RdnaArHCtx c;
  c.timeout = timeout;
  c.report = report;
  c.rank = rank;
  c.world = world;
  c.seq = s_seq;
  c.p = s_seq & 1;
  c.b = b;
  c.max_bytes = max_bytes;
  c.wait_ticks = wait_ticks;
  c.ticks_per_ms = ticks_per_ms;

  const int nu = (n + V - 1) / V;
  const RdnaArHGeom gm = rdna_ar_hgeom(nu, nblocks, b, tp.X);
  const bool active = gm.hi > gm.lo;
  const bool owner = tp.l < tp.X;
  const bool in_al = (reinterpret_cast<uintptr_t>(in) & 15u) == 0;
  const bool out_al = (reinterpret_cast<uintptr_t>(out) & 15u) == 0;
  int ann = 0;  // phases announced (thread 0)

  if (active) {
    bool ok = !s_abort;
    // ---- phase 0: reduce-scatter ------------------------------------------
    if (ok) {
      for (int s = 1; s < G; s++) {
        const int q = (tp.l + s) % G;
        if (q >= tp.X) continue;
        uint4* dst = rdna_ar_hslot(peers, c, tp.isl[tp.k][q], rank);
        const int r0 = rdna_ar_hrange_lo(gm, q), r1 = rdna_ar_hrange_hi(gm, q);
        for (int u = r0 + t; u < r1; u += nt)
          dst[u] = rdna_ar_load_unit<T>(in, u, n, in_al);
      }
      __syncthreads();
      if (t == 0) {
        __threadfence_system();
        rdna_ar_hannounce(peers, tp, c, 0);
      }
      ann = 1;
      rdna_ar_hwait(peers, tp, c, 0, &s_abort);
      ok = !s_abort;
    }
    const int m0 = rdna_ar_hrange_lo(gm, tp.l),
              m1 = rdna_ar_hrange_hi(gm, tp.l);
    // ---- phase 1: island partial -> partner -------------------------------
    if (ok) {
      if (owner) {
        uint4* pdst = rdna_ar_hslot(peers, c, tp.isl[1 - tp.k][tp.l], rank);
        for (int u = m0 + t; u < m1; u += nt) {
          uint4 x[G];
#pragma unroll
          for (int q = 0; q < G; q++)
            x[q] = (q == tp.l) ? rdna_ar_load_unit<T>(in, u, n, in_al)
                               : rdna_ar_hmine(peers, c, tp.isl[tp.k][q])[u];
          float acc[V];
#pragma unroll
          for (int q = 0; q < V; q++) acc[q] = 0.f;
#pragma unroll
          for (int q = 0; q < G; q++) rdna_ar_unit_acc<T>(acc, x[q]);
          const uint4 part = rdna_ar_unit_pack<T>(acc);
          pdst[u] = part;
          rdna_ar_store_unit<T>(out, u, n, out_al, part);  // our partial
        }
      }
      __syncthreads();
      if (t == 0 && owner) {
        __threadfence_system();
        rdna_ar_hannounce(peers, tp, c, 1);
      }
      ann = 2;
      rdna_ar_hwait(peers, tp, c, 1, &s_abort);
      ok = !s_abort;
    }
    // ---- phase 2: add the partials, all-gather ----------------------------
    if (ok) {
      if (owner) {
        const uint4* pin = rdna_ar_hmine(peers, c, tp.isl[1 - tp.k][tp.l]);
        for (int u = m0 + t; u < m1; u += nt) {
          const uint4 res = rdna_ar_unit_add2<T>(
              rdna_ar_load_unit<T>(out, u, n, out_al), pin[u]);
          rdna_ar_store_unit<T>(out, u, n, out_al, res);
          for (int s = 1; s < G; s++)
            rdna_ar_hslot(peers, c, tp.isl[tp.k][(tp.l + s) % G], rank)[u] =
                res;
        }
      }
      __syncthreads();
      if (t == 0 && owner) {
        __threadfence_system();
        rdna_ar_hannounce(peers, tp, c, 2);
      }
      ann = 3;
      rdna_ar_hwait(peers, tp, c, 2, &s_abort);
      ok = !s_abort;
    }
    if (ok) {
      for (int o = 0; o < tp.X; o++) {
        if (o == tp.l) continue;
        const uint4* src = rdna_ar_hmine(peers, c, tp.isl[tp.k][o]);
        const int r0 = rdna_ar_hrange_lo(gm, o), r1 = rdna_ar_hrange_hi(gm, o);
        for (int u = r0 + t; u < r1; u += nt)
          rdna_ar_store_unit<T>(out, u, n, out_al, src[u]);
      }
    } else {
      // Aborted (here or earlier): release the peers that wait on the phases
      // we did not announce, and leave a finite (wrong) value, never stale
      // memory. rdna_ar_check() fails the step with the record.
      if (t == 0)
        for (int ph = ann; ph < RDNA_AR_HPHASES; ph++)
          rdna_ar_hannounce(peers, tp, c, ph);
      for (int u = gm.lo + t; u < gm.hi; u += nt)
        rdna_ar_store_unit<T>(out, u, n, out_al,
                              rdna_ar_load_unit<T>(in, u, n, in_al));
    }
  }
  // Every block, including the ones with no chunk, counts toward the end of
  // the collective; the last one moves the sequence on.
  __syncthreads();
  if (t == 0) {
    if (atomicAdd(&arrive[c.p], 1u) == (unsigned)(nblocks - 1)) {
      arrive[c.p] = 0u;
      __hip_atomic_store(seqbuf, c.seq, __ATOMIC_RELEASE,
                         __HIP_MEMORY_SCOPE_AGENT);
    }
  }
}

// Flat exchange with the hier arithmetic: every rank's payload lands in every
// rank's staging (rdna_ar_exchange, grid barrier, one flag per peer), then
// each island's contributions are summed in rank order (fp32), rounded to T,
// and the two island partials are added and rounded once more. Bit-equal to
// rdna_ar_hier for every payload, so a group that switches between the two by
// size gets one result per input. For the sizes where one hop beats three.
template <typename T, int W>
__device__ __forceinline__ void rdna_ar_sum_isl_vec(
    const T* __restrict__ in, T* __restrict__ out, const T* __restrict__ mine,
    int rank, long long max_elems, int nvec, int gid, int gstride,
    const RdnaArTopo& tp) {
  constexpr int V = (int)(16 / sizeof(T));
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
    float a0[V], a1[V];
#pragma unroll
    for (int q = 0; q < V; q++) a0[q] = a1[q] = 0.f;
#pragma unroll
    for (int j = 0; j < W; j++) {
      if (tp.kof[j] == 0)
        rdna_ar_unit_acc<T>(a0, x[j]);
      else
        rdna_ar_unit_acc<T>(a1, x[j]);
    }
    out4[v] = rdna_ar_unit_add2<T>(rdna_ar_unit_pack<T>(a0),
                                   rdna_ar_unit_pack<T>(a1));
  }
}

template <typename T>
__global__ void rdna_ar_oneshot_isl(const T* __restrict__ in,
                                    T* __restrict__ out, RdnaArPeers peers,
                                    unsigned int* arrive, int* seqbuf,
                                    unsigned* timeout,
                                    unsigned long long* report, int rank,
                                    int world, int n, long long max_elems,
                                    int nblocks, int pace, long long wait_ticks,
                                    long long ticks_per_ms, RdnaArTopo tp) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int gstride = nblocks * blockDim.x;
  int p = 0;
  if (rdna_ar_exchange<T>(RdnaArLoadDirect<T>{in}, peers, arrive, seqbuf,
                          timeout, report, rank, world, n, max_elems, nblocks,
                          pace, wait_ticks, ticks_per_ms, p)) {
    for (int i = gid; i < n; i += gstride) out[i] = in[i];
    return;
  }
  const T* mine = reinterpret_cast<const T*>(peers.stage[rank]) +
                  ((long long)p * world) * max_elems;
  constexpr int kVec = (int)(16 / sizeof(T));
  int done = 0;
  if ((((uintptr_t)in | (uintptr_t)out) & 15u) == 0) {
    const int nvec = n / kVec;
    if (world == 8) {
      rdna_ar_sum_isl_vec<T, 8>(in, out, mine, rank, max_elems, nvec, gid,
                                gstride, tp);
      done = nvec * kVec;
    } else if (world == 6) {
      rdna_ar_sum_isl_vec<T, 6>(in, out, mine, rank, max_elems, nvec, gid,
                                gstride, tp);
      done = nvec * kVec;
    } else if (world == 4) {
      rdna_ar_sum_isl_vec<T, 4>(in, out, mine, rank, max_elems, nvec, gid,
                                gstride, tp);
      done = nvec * kVec;
    }
  }
  for (int i = done + gid; i < n; i += gstride) {
    float s[2] = {0.f, 0.f};
    for (int j = 0; j < world; j++)
      s[tp.kof[j]] += (j == rank)
                          ? rdna_ar_to_f(in[i])
                          : rdna_ar_to_f(mine[(long long)j * max_elems + i]);
    T r0, r1;
    rdna_ar_from_f(r0, s[0]);
    rdna_ar_from_f(r1, s[1]);
    rdna_ar_from_f(out[i], rdna_ar_to_f(r0) + rdna_ar_to_f(r1));
  }
}
