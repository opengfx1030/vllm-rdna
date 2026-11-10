// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// hx_order: raw ORDERING RACE DETECTOR for the hx reply path (no torch, plain
// HIP, persistent kernels, no host in the loop). N producer GPUs (hx servers)
// write payload chunks and then a flag into ONE consumer GPU's memory (the hx
// hub); the consumer checks every payload vector right after the flags. A
// round is: the consumer posts go_p = r to every producer, each producer
// writes its payload (every 16 B vector distinct and different every round),
// fences, and publishes flag_p = r; the consumer waits the flags in order and
// plain-loads and verifies all payload. Rounds run back to back inside the
// kernels, so 1e8 to 1e9 rounds per variant are reachable.
//
// Usage (the process sees exactly nprod + 1 GPUs):
//   run as HIP_VISIBLE_DEVICES=C,P1,..,PN hx_order [options]
//   ordinal 0 = consumer (hub role), ordinals 1..N = producers (server role).
//   The caller must hold the leases of ALL these GPUs. The program refuses to
//   start (exit 64) unless HIP_VISIBLE_DEVICES is set and exactly nprod + 1
//   devices are visible; missing peer access exits 4.
//
//   --nprod N         producers, 1..4 (default 4)
//   --rounds R        rounds (default 10000000)
//   --chunks K        256 B chunks per producer region (default 60 = T=3
//                     payload); nprod * K <= 16384
//   --blocks B        producer blocks of 256 threads (default 4)
//   --alloc uncached|coarse   consumer buffer + go words (default uncached,
//                     hipExtMallocWithFlags(hipDeviceMallocUncached) like hx)
//   --flush F         producer/consumer variant bits (as ctx.flush_mode in
//                     csrc/rocm/hetero_moe/hx_core.cuh):
//                       1 producer reads one dword of the consumer control
//                         page back before publishing
//                       2 each producer wave reads back one dword of every
//                         256 B chunk it wrote, before the fence
//                       4 publish by a returning system-scope exchange
//                         (release) instead of a store
//                       8 consumer spins ~4 us after the flags before reading
//   --rd R            consumer payload read: 0 plain, 1 nontemporal, 2 two 8 B
//                     relaxed system atomics (bypass), 3 plain after an
//                     acquire fence (agent) in every thread
//   --records M       mismatch records kept with full classification (16)
//   --timeout-ms T    bound of every device wait (default 2000)
//   --report-s S      report interval (default 10)
//   --seed X          payload hash seed
//   --inject-flag-first EVERY   POSITIVE CONTROL (0 = off): on rounds r with
//                     r % EVERY == 0 each producer publishes the flag BEFORE
//                     the payload (block 0 stores flag = r right after go was
//                     seen; the payload stores of all blocks follow
//                     --inject-delay-us later, default 1 us). Expected:
//                     bad_rounds ~ rounds / EVERY (at nprod 1; with several
//                     producers the flags of the others may arrive later),
//                     bad vectors mostly "late", bypass delay ~1-4 us
//   --inject-stale EVERY        POSITIVE CONTROL (0 = off): on rounds
//                     r % EVERY == 0 each producer skips the stores of the
//                     256 B chunk r % chunks (previous round's data stays).
//                     Expected: persistent, bypass wrong, late_ns=-1,
//                     bad_rounds = rounds / EVERY, 16 bad vectors per
//                     producer per injected round, bad_chunks = distinct
//                     (producer, chunk) pairs hit
//   --inject-delay-us U         payload delay of flag-first rounds (default 1)
//   Injected rounds have a separate code path in the producer (one uniform
//   compare per round); bad_rounds_clean counts bad rounds that were NOT
//   injected (must stay 0 for the controls to be meaningful).
//
// Mismatch classification (consumer, per bad vector): a second plain read
// ~1 us later: "late" if it now equals the expected value, else "persistent".
// For the first --records events: a third read through the bypass path (two
// relaxed system-scope 8 B atomic loads), and the delay until the vector
// equals the expected value, polled via the bypass path for up to 100 us
// from the detection (late_ns=-1: never within 100 us), histogram in 0.1 us.
//
// Output (stdout, line buffered):
//   ORDER start ...
//   ORDER rounds=.. rate=../s bad_vectors=.. bad_chunks=.. bad_rounds=..
//         late=.. persistent=.. bypass_ok=..          every --report-s
//   ORDER done rounds=.. bad_rounds=.. ...
//   ORDER HIST delay_us=.. count=..                    (nonzero bins)
//   ORDER REC round=.. prod=.. vec=.. chunk=.. got=<32 hex> exp=<32 hex>
//         again=<32 hex> bypass=<32 hex> late_ns=.. inject=ff|st|ff+st|-
//   (hex = the four 32-bit words of the vector, word 0 first, %08x each)
//   ORDER abort: ...                         a device wait timed out (exit 3)
// Exit codes: 0 no bad round, 2 a bad round (or a HIP error), 3 timeout abort,
// 4 no peer access, 64 usage. SIGTERM/SIGINT stop the kernels cleanly.
//
// Build (no GPU is touched):
//   flock ~/w4a8_runs/audit/build.lock nice -n 19 /opt/rocm/core-7.14/bin/hipcc
//     --offload-arch=gfx1030 -O3 -std=c++17 -x hip -Wall -Wextra
//     -o ~/w4a8_runs/hxorder/hx_order tools/rdna/hetero_moe/hx_order.cpp
#include <hip/hip_runtime.h>

#include <signal.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

#define CK(x)                                                     \
  do {                                                            \
    hipError_t e_ = (x);                                          \
    if (e_ != hipSuccess) {                                       \
      fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, \
              hipGetErrorString(e_));                             \
      exit(2);                                                    \
    }                                                             \
  } while (0)

namespace {

constexpr int kMaxProd = 4;
constexpr int kMaxRec = 256;
constexpr int kHistBins = 1002;  // 0.1 us steps up to 100 us, + overflow
constexpr int kBitmapWords = 256;
constexpr uint32_t kThreads = 256;
constexpr uint64_t kGoAbort = 1ull << 63;
constexpr uint32_t kFlushCtrl = 1, kFlushPayload = 2, kFlushAtomic = 4,
                   kSettle = 8;
constexpr size_t kPage = 4096;

struct Rec {
  uint64_t round;
  uint32_t prod, vec, chunk, inject;  // inject: bit 0 flag-first, bit 1 stale
  uint32_t got[4], exp[4], again[4], bypass[4];
  int64_t late_ticks;
};

// One per device (uncached device memory): counters, abort state, tickets and
// the consumer's results. The host reads the first 512 bytes while running.
struct Ctl {
  uint64_t rounds;      // consumer: rounds checked; producer: rounds published
  uint32_t abort_word;  // 1 = a device wait timed out (record below)
  uint32_t host_abort;  // written by the host: stop
  uint64_t abort_rec[4];  // phase (1 consumer flag wait, 2 producer go wait),
                          // peer, round, waited ms
  uint64_t bad_vectors, bad_rounds, bad_chunks, late, persistent, bypass_ok;
  uint64_t rec_count;
  uint64_t bad_rounds_clean;  // bad rounds that were not injected
  uint64_t pad0[50];
  uint32_t arrive;  // producer block ticket (offset 512)
  uint32_t pad1[63];
  uint64_t hist[kHistBins];
  uint64_t bitmap[kBitmapWords];
  Rec recs[kMaxRec];
};
static_assert(offsetof(Ctl, arrive) == 512, "header is 512 bytes");

struct ProdArgs {
  uint8_t* region;  // this producer's region in the consumer buffer
  const uint64_t* go;
  Ctl* ctl;
  uint64_t rounds;
  int64_t budget, tpms;
  uint32_t seed, p, nvec, flush, nchunks;
  uint64_t inj_ff, inj_st;
  int64_t inj_delay_ticks;
};
struct ConsArgs {
  uint8_t* base;
  uint64_t* go[kMaxProd];
  Ctl* ctl;
  uint64_t rounds;
  int64_t budget, tpms;
  uint32_t seed, nprod, nvec, flush, stride, nchunks, records;
  uint64_t inj_ff, inj_st;
};

__device__ __forceinline__ uint32_t fm32(uint32_t h) {
  h ^= h >> 16;
  h *= 0x85EBCA6Bu;
  h ^= h >> 13;
  h *= 0xC2B2AE35u;
  h ^= h >> 16;
  return h;
}
// Distinct per (round, producer, vector) and per word; changes every round.
__device__ __forceinline__ uint4 hash4(uint32_t seed, uint64_t r, uint32_t p,
                                       uint32_t v) {
  uint32_t b = fm32(seed + (uint32_t)r * 0x9E3779B1u);
  b = fm32(b ^ ((uint32_t)(r >> 32) * 0x85EBCA77u) ^ ((p + 1u) * 0xC2B2AE3Du));
  b = fm32(b + v * 0x27D4EB2Fu);
  return make_uint4(fm32(b), fm32(b + 0x9E3779B9u), fm32(b + 0x3C6EF372u),
                    fm32(b + 0xDAA66D2Bu));
}
__device__ __forceinline__ bool eq4(const uint4& a, const uint4& b) {
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

__device__ __forceinline__ uint64_t ld_acq(const uint64_t* p) {
  return __hip_atomic_load(const_cast<uint64_t*>(p), __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void st_rel(uint64_t* p, uint64_t v) {
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void st_rlx(uint64_t* p, uint64_t v) {
  __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void add_u64(uint64_t* p, uint64_t v) {
  __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void sink32(uint32_t v) {
  asm volatile("" : : "v"(v));
}

// Same semantics as hx_core.cuh hx_ld_reply (RD 0 plain, 1 nontemporal,
// 2 bypass; 3 is plain after a fence run by every thread).
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
__device__ __forceinline__ uint4 ld_bypass(const uint4* p) {
  const uint64_t* q = reinterpret_cast<const uint64_t*>(p);
  const uint64_t a =
      __hip_atomic_load(q, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  const uint64_t b =
      __hip_atomic_load(q + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  return make_uint4((uint32_t)a, (uint32_t)(a >> 32), (uint32_t)b,
                    (uint32_t)(b >> 32));
}
template <int RD>
__device__ __forceinline__ uint4 ld_reply(const uint4* p) {
  if constexpr (RD == 1) {
    const u32x4 r =
        __builtin_nontemporal_load(reinterpret_cast<const u32x4*>(p));
    return make_uint4(r.x, r.y, r.z, r.w);
  } else if constexpr (RD == 2) {
    return ld_bypass(p);
  } else {
    return *p;
  }
}

__device__ __forceinline__ void spin_ticks(int64_t n) {
  const int64_t until = wall_clock64() + n;
  while (wall_clock64() < until) {
  }
}

__device__ __forceinline__ bool ctl_aborted(const Ctl* c) {
  return (__hip_atomic_load(const_cast<uint32_t*>(&c->abort_word),
                            __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) |
          __hip_atomic_load(const_cast<uint32_t*>(&c->host_abort),
                            __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) != 0u;
}

enum { kWaitOk = 0, kWaitTimeout = 1, kWaitAbort = 2 };

// Wait (one thread) until *flag >= want. A flag with bit 63 set is a stop
// request. The first timeout claims abort_word and writes the record.
__device__ int wait_ge(Ctl* c, const uint64_t* flag, uint64_t want,
                       int64_t budget, int64_t tpms, uint32_t phase,
                       uint32_t peer, uint64_t round) {
  uint64_t v = ld_acq(flag);
  if (v & kGoAbort) return kWaitAbort;
  if (v >= want) return kWaitOk;
  const int64_t t0 = wall_clock64();
  unsigned it = 0;
  for (;;) {
    __builtin_amdgcn_s_sleep(2);
    v = ld_acq(flag);
    if (v & kGoAbort) return kWaitAbort;
    if (v >= want) return kWaitOk;
    const int64_t dt = wall_clock64() - t0;
    if ((++it & 7u) == 0u || dt > budget)
      if (ctl_aborted(c)) return kWaitAbort;
    if (dt > budget) {
      v = ld_acq(flag);
      if (v >= want && !(v & kGoAbort)) return kWaitOk;
      if (atomicCAS(&c->abort_word, 0u, 1u) == 0u) {
        st_rlx(&c->abort_rec[0], phase);
        st_rlx(&c->abort_rec[1], peer);
        st_rlx(&c->abort_rec[2], round);
        st_rlx(&c->abort_rec[3], (uint64_t)(dt / tpms));
      }
      return kWaitTimeout;
    }
  }
}

__global__ void k_wall(uint64_t* out) { *out = (uint64_t)wall_clock64(); }

// ---------------------------------------------------------------------------
// Producer (hx_k_srv_push): wait go, write the payload, fence, last block
// publishes the flag.
// ---------------------------------------------------------------------------
// One producer round after go was seen. INJ = positive-control round (see
// --inject-*): `ff` publishes the flag before the payload is written (block 0
// thread 0 stores it right away; the payload of every block is delayed by
// --inject-delay-us so the flag reliably wins), `st` skips the stores of one
// 256 B chunk (r % chunks) and leaves the previous round's data there. The
// normal instantiation (INJ = false) has none of this code.
template <bool INJ>
__device__ __forceinline__ void prod_round(const ProdArgs& a, uint64_t r,
                                           bool ff, bool st) {
  const uint32_t t = threadIdx.x, nb = gridDim.x;
  const uint32_t gid = blockIdx.x * kThreads + t, gs = nb * kThreads;
  uint4* dst = reinterpret_cast<uint4*>(a.region + kPage);
  uint64_t* flag = reinterpret_cast<uint64_t*>(a.region);
  uint32_t skip_chunk = ~0u;
  if constexpr (INJ) {
    if (ff) {
      if (blockIdx.x == 0 && t == 0) st_rel(flag, r);
      if (a.inj_delay_ticks > 0) spin_ticks(a.inj_delay_ticks);
    }
    if (st) skip_chunk = (uint32_t)(r % a.nchunks);
  }
  for (uint32_t v = gid; v < a.nvec; v += gs) {
    if constexpr (INJ) {
      if ((v >> 4) == skip_chunk) continue;
    }
    dst[v] = hash4(a.seed, r, a.p, v);
  }
  if (a.flush & kFlushPayload) {
    uint32_t sink = 0u;
    for (uint32_t v = gid; v < a.nvec; v += gs)
      if ((v & 15u) == 15u || v == a.nvec - 1u)
        sink += __hip_atomic_load(reinterpret_cast<const uint32_t*>(dst + v),
                                  __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    sink32(sink);
  }
  __threadfence_system();
  __syncthreads();
  if (t == 0) {
    const unsigned ticket = __hip_atomic_fetch_add(
        &a.ctl->arrive, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    if (ticket == nb - 1u) {
      __hip_atomic_store(&a.ctl->arrive, 0u, __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_AGENT);
      bool publish = true;
      if constexpr (INJ) publish = !ff;  // the flag was already stored
      if (publish) {
        if (a.flush & kFlushCtrl)
          sink32(__hip_atomic_load(reinterpret_cast<uint32_t*>(flag) + 2,
                                   __ATOMIC_RELAXED,
                                   __HIP_MEMORY_SCOPE_SYSTEM));
        if (a.flush & kFlushAtomic) {
          const uint64_t old = __hip_atomic_exchange(flag, r, __ATOMIC_RELEASE,
                                                     __HIP_MEMORY_SCOPE_SYSTEM);
          sink32((uint32_t)old ^ (uint32_t)(old >> 32));
        } else {
          st_rel(flag, r);
        }
      }
      if ((r & 15u) == 0u || r == a.rounds) st_rlx(&a.ctl->rounds, r);
    }
  }
}

__global__ void __launch_bounds__(kThreads) k_prod(ProdArgs a) {
  __shared__ int s_ok;
  const uint32_t t = threadIdx.x;
  // One uniform compare per round selects the injection path; 0 = off.
  uint64_t next_ff = a.inj_ff ? a.inj_ff : ~0ull;
  uint64_t next_st = a.inj_st ? a.inj_st : ~0ull;
  for (uint64_t r = 1; r <= a.rounds; r++) {
    if (t == 0)
      s_ok = wait_ge(a.ctl, a.go, r, a.budget, a.tpms, 2, a.p, r) == kWaitOk;
    __syncthreads();
    if (!s_ok) return;
    if (__builtin_expect(r == next_ff || r == next_st, 0)) {
      const bool ff = r == next_ff, st = r == next_st;
      if (ff) next_ff += a.inj_ff;
      if (st) next_st += a.inj_st;
      prod_round<true>(a, r, ff, st);
    } else {
      prod_round<false>(a, r, false, false);
    }
  }
}

// ---------------------------------------------------------------------------
// Consumer (hx_k_combine): post go, wait the flags in order, read and verify.
// ---------------------------------------------------------------------------
__device__ __noinline__ void note_bad(const ConsArgs& a, uint64_t r, uint32_t p,
                                      uint32_t v, const uint4& got,
                                      const uint4& exp, const uint4* ptr,
                                      int* round_bad) {
  Ctl* c = a.ctl;
  const int64_t t_det = wall_clock64();
  // Bit 0: flag-first round, bit 1: stale-chunk round (positive controls).
  const uint32_t inj = (a.inj_ff && r % a.inj_ff == 0 ? 1u : 0u) |
                       (a.inj_st && r % a.inj_st == 0 ? 2u : 0u);
  add_u64(&c->bad_vectors, 1);
  const uint32_t chunk = v >> 4;
  const uint32_t bit = p * a.nchunks + chunk;
  const uint64_t m = 1ull << (bit & 63u);
  const uint64_t old = __hip_atomic_fetch_or(
      &c->bitmap[bit >> 6], m, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  if (!(old & m)) add_u64(&c->bad_chunks, 1);
  if (atomicExch(round_bad, 1) == 0) {
    add_u64(&c->bad_rounds, 1);
    if (!inj) add_u64(&c->bad_rounds_clean, 1);
  }
  const uint64_t slot = __hip_atomic_fetch_add(
      &c->rec_count, (uint64_t)1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  spin_ticks(a.tpms / 1000);  // ~1 us
  asm volatile("" ::: "memory");
  const uint4 again = *ptr;
  if (eq4(again, exp))
    add_u64(&c->late, 1);
  else
    add_u64(&c->persistent, 1);
  if (slot >= a.records) return;
  uint4 byp = ld_bypass(ptr);
  const bool byp_ok = eq4(byp, exp);
  int64_t late_ticks = -1;
  if (byp_ok) {
    late_ticks = wall_clock64() - t_det;
    add_u64(&c->bypass_ok, 1);
  } else {
    const int64_t limit = a.tpms / 10;  // 100 us
    while (wall_clock64() - t_det < limit) {
      const uint4 x = ld_bypass(ptr);
      if (eq4(x, exp)) {
        late_ticks = wall_clock64() - t_det;
        break;
      }
    }
  }
  const int64_t ns = late_ticks < 0 ? -1 : late_ticks * 1000000 / a.tpms;
  uint32_t bin = ns < 0 ? kHistBins - 1 : (uint32_t)(ns / 100);
  if (bin > kHistBins - 1) bin = kHistBins - 1;
  add_u64(&c->hist[bin], 1);
  Rec* rec = &c->recs[slot];
  rec->round = r;
  rec->prod = p;
  rec->vec = v;
  rec->chunk = chunk;
  rec->inject = inj;
  rec->got[0] = got.x, rec->got[1] = got.y, rec->got[2] = got.z,
  rec->got[3] = got.w;
  rec->exp[0] = exp.x, rec->exp[1] = exp.y, rec->exp[2] = exp.z,
  rec->exp[3] = exp.w;
  rec->again[0] = again.x, rec->again[1] = again.y, rec->again[2] = again.z,
  rec->again[3] = again.w;
  rec->bypass[0] = byp.x, rec->bypass[1] = byp.y, rec->bypass[2] = byp.z,
  rec->bypass[3] = byp.w;
  rec->late_ticks = late_ticks;
}

template <int RD>
__global__ void __launch_bounds__(kThreads) k_cons(ConsArgs a) {
  __shared__ int s_ok, s_rb;
  const uint32_t t = threadIdx.x;
  Ctl* ctl = a.ctl;
  const uint32_t n = a.nprod;
  for (uint64_t r = 1; r <= a.rounds; r++) {
    if (t == 0) {
      s_rb = 0;
      int rc = ctl_aborted(ctl) ? kWaitAbort : kWaitOk;
      if (rc == kWaitOk) {
        for (uint32_t p = 0; p < n; p++) st_rel(a.go[p], r);
        for (uint32_t p = 0; p < n && rc == kWaitOk; p++)
          rc = wait_ge(
              ctl,
              reinterpret_cast<const uint64_t*>(a.base + (size_t)p * a.stride),
              r, a.budget, a.tpms, 1, p, r);
        if (rc == kWaitOk && (a.flush & kSettle)) spin_ticks(a.tpms / 250);
      }
      if (rc != kWaitOk) {
        for (uint32_t p = 0; p < n; p++) st_rel(a.go[p], kGoAbort);
        st_rlx(&ctl->rounds, r - 1);
      }
      s_ok = rc == kWaitOk;
    }
    __syncthreads();
    if (!s_ok) return;
    if constexpr (RD == 3) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    for (uint32_t v = t; v < a.nvec; v += kThreads) {
      uint4 raw[kMaxProd];
#pragma unroll
      for (int p = 0; p < kMaxProd; p++) {
        raw[p] = make_uint4(0, 0, 0, 0);
        if ((uint32_t)p < n)
          raw[p] = ld_reply<RD>(reinterpret_cast<const uint4*>(
                                    a.base + (size_t)p * a.stride + kPage) +
                                v);
      }
#pragma unroll
      for (int p = 0; p < kMaxProd; p++) {
        if ((uint32_t)p < n) {
          const uint4 e = hash4(a.seed, r, (uint32_t)p, v);
          if (!eq4(raw[p], e))
            note_bad(a, r, (uint32_t)p, v, raw[p], e,
                     reinterpret_cast<const uint4*>(
                         a.base + (size_t)p * a.stride + kPage) +
                         v,
                     &s_rb);
        }
      }
    }
    __syncthreads();
    if (t == 0 && ((r & 15u) == 0u || r == a.rounds)) st_rlx(&ctl->rounds, r);
  }
}

// ---------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------
volatile sig_atomic_t g_stop = 0;
void on_signal(int) { g_stop = 1; }

struct Opts {
  int nprod = 4;
  uint64_t rounds = 10000000;
  uint32_t chunks = 60, blocks = 4, flush = 0, rd = 0, records = 16;
  bool uncached = true;
  long timeout_ms = 2000;
  double report_s = 10;
  uint32_t seed = 0x5EED1234u;
  uint64_t inject_ff = 0, inject_st = 0;  // 0 = off
  double inject_delay_us = 1.0;
};

[[noreturn]] void usage(const char* why) {
  fprintf(stderr,
          "hx_order: %s\n"
          "run as HIP_VISIBLE_DEVICES=C,P1,..,PN hx_order [--nprod N] "
          "[--rounds R] [--chunks K] [--blocks B] [--alloc uncached|coarse] "
          "[--flush F] [--rd R] [--records M] [--timeout-ms T] [--report-s S] "
          "[--seed X] [--inject-flag-first EVERY] [--inject-stale EVERY] "
          "[--inject-delay-us U]\n"
          "ordinal 0 = consumer, 1..N = producers; the caller must hold the "
          "leases of all of them (see the header of hx_order.cpp)\n",
          why);
  exit(64);
}

Opts parse(int argc, char** argv) {
  Opts o;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) usage(("missing value for " + a).c_str());
      return argv[++i];
    };
    if (a == "--nprod")
      o.nprod = atoi(val().c_str());
    else if (a == "--rounds")
      o.rounds = strtoull(val().c_str(), nullptr, 0);
    else if (a == "--chunks")
      o.chunks = (uint32_t)strtoul(val().c_str(), nullptr, 0);
    else if (a == "--blocks")
      o.blocks = (uint32_t)strtoul(val().c_str(), nullptr, 0);
    else if (a == "--alloc") {
      std::string v = val();
      if (v == "uncached")
        o.uncached = true;
      else if (v == "coarse")
        o.uncached = false;
      else
        usage("--alloc must be uncached|coarse");
    } else if (a == "--flush")
      o.flush = (uint32_t)strtoul(val().c_str(), nullptr, 0);
    else if (a == "--rd")
      o.rd = (uint32_t)strtoul(val().c_str(), nullptr, 0);
    else if (a == "--records")
      o.records = (uint32_t)strtoul(val().c_str(), nullptr, 0);
    else if (a == "--timeout-ms")
      o.timeout_ms = atol(val().c_str());
    else if (a == "--report-s")
      o.report_s = atof(val().c_str());
    else if (a == "--seed")
      o.seed = (uint32_t)strtoul(val().c_str(), nullptr, 0);
    else if (a == "--inject-flag-first")
      o.inject_ff = strtoull(val().c_str(), nullptr, 0);
    else if (a == "--inject-stale")
      o.inject_st = strtoull(val().c_str(), nullptr, 0);
    else if (a == "--inject-delay-us")
      o.inject_delay_us = atof(val().c_str());
    else
      usage(("unknown option " + a).c_str());
  }
  if (o.nprod < 1 || o.nprod > kMaxProd) usage("--nprod must be 1..4");
  if (o.rounds < 1) usage("--rounds must be >= 1");
  if (o.chunks < 1 || o.chunks > 4096 ||
      (uint64_t)o.nprod * o.chunks > (uint64_t)kBitmapWords * 64)
    usage("--chunks must be 1..4096 and nprod * chunks <= 16384");
  if (o.blocks < 1 || o.blocks > 64) usage("--blocks must be 1..64");
  if (o.flush > 15) usage("--flush must be 0..15");
  if (o.rd > 3) usage("--rd must be 0..3");
  if (o.records > (uint32_t)kMaxRec) usage("--records must be <= 256");
  if (o.timeout_ms < 1 || o.report_s <= 0)
    usage("--timeout-ms >= 1, --report-s > 0");
  if (o.inject_delay_us < 0 || o.inject_delay_us > 1000)
    usage("--inject-delay-us must be 0..1000");
  return o;
}

struct Dev {
  int tpms = 0;
  Ctl* ctl = nullptr;
  Ctl* mirror = nullptr;  // pinned host copy
  uint64_t* go = nullptr;
  hipStream_t exec = nullptr, copy = nullptr;
};

std::string bus_of(int d) {
  char b[64] = {0};
  CK(hipDeviceGetPCIBusId(b, sizeof b, d));
  return b;
}

hipError_t alloc_kind(void** p, size_t n, bool uncached) {
  return uncached ? hipExtMallocWithFlags(p, n, hipDeviceMallocUncached)
                  : hipMalloc(p, n);
}

std::string hex16(const uint32_t w[4]) {
  char b[40];
  snprintf(b, sizeof b, "%08x%08x%08x%08x", w[0], w[1], w[2], w[3]);
  return b;
}

// Device ticks per host ms: stamp, host busy wait, stamp (never compares
// devices).
double calibrate(int d, double ms) {
  CK(hipSetDevice(d));
  uint64_t* m = nullptr;
  CK(hipMalloc((void**)&m, 16));
  hipStream_t s;
  CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  k_wall<<<1, 1, 0, s>>>(m);
  CK(hipStreamSynchronize(s));
  k_wall<<<1, 1, 0, s>>>(m);
  CK(hipStreamSynchronize(s));
  const auto h0 = Clock::now();
  const auto end = h0 + std::chrono::microseconds((long)(ms * 1000));
  while (Clock::now() < end) {
  }
  k_wall<<<1, 1, 0, s>>>(m + 1);
  CK(hipStreamSynchronize(s));
  const auto h1 = Clock::now();
  uint64_t v[2];
  CK(hipMemcpy(v, m, 16, hipMemcpyDeviceToHost));
  // v[0] was overwritten by the second launch; it precedes h0 by < 1 ms.
  CK(hipStreamDestroy(s));
  CK(hipFree(m));
  return (double)(v[1] - v[0]) /
         std::chrono::duration<double, std::milli>(h1 - h0).count();
}

}  // namespace

int main(int argc, char** argv) {
  Opts o = parse(argc, argv);
  setvbuf(stdout, nullptr, _IOLBF, 0);
  const char* vis = getenv("HIP_VISIBLE_DEVICES");
  if (!vis || !*vis)
    usage("HIP_VISIBLE_DEVICES is not set (refusing to touch every GPU)");
  int ndev = 0;
  CK(hipGetDeviceCount(&ndev));
  const int N = o.nprod;
  if (ndev != N + 1) {
    fprintf(stderr,
            "hx_order: %d devices visible (HIP_VISIBLE_DEVICES=%s), exactly "
            "%d (nprod + 1) are required\n",
            ndev, vis, N + 1);
    return 64;
  }
  struct sigaction sa;
  memset(&sa, 0, sizeof sa);
  sa.sa_handler = on_signal;
  sigemptyset(&sa.sa_mask);
  sigaction(SIGINT, &sa, nullptr);
  sigaction(SIGTERM, &sa, nullptr);

  // Peer access all ways, before anything is allocated.
  for (int a = 0; a < ndev; a++)
    for (int b = 0; b < ndev; b++) {
      if (a == b) continue;
      int can = 0;
      CK(hipDeviceCanAccessPeer(&can, a, b));
      if (!can) {
        fprintf(stderr,
                "hx_order: no peer access from visible ordinal %d to %d\n", a,
                b);
        return 4;
      }
      CK(hipSetDevice(a));
      hipError_t e = hipDeviceEnablePeerAccess(b, 0);
      if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
        fprintf(stderr,
                "hx_order: hipDeviceEnablePeerAccess(%d -> %d) failed: %s\n", a,
                b, hipGetErrorString(e));
        return 4;
      }
      (void)hipGetLastError();
    }

  std::vector<Dev> dv(ndev);
  std::string buses;
  for (int d = 0; d < ndev; d++) {
    CK(hipSetDevice(d));
    buses += (d ? "," : "") + bus_of(d);
    int khz = 0;
    CK(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, d));
    const double meas = calibrate(d, 50);
    const double err = khz > 0 ? (meas - khz) / khz * 100.0 : 100.0;
    dv[d].tpms = (khz > 0 && err > -5.0 && err < 5.0) ? khz : (int)meas;
    printf(
        "ORDER tick dev=%d attr_ticks_per_ms=%d measured_ticks_per_ms=%.0f "
        "use=%d\n",
        d, khz, meas, dv[d].tpms);
    void* c = nullptr;
    hipError_t e = alloc_kind(&c, sizeof(Ctl), true);
    if (e != hipSuccess) {
      printf(
          "ORDER warn dev=%d uncached control block unavailable (%s), using "
          "hipMalloc\n",
          d, hipGetErrorName(e));
      (void)hipGetLastError();
      CK(hipMalloc(&c, sizeof(Ctl)));
    }
    dv[d].ctl = (Ctl*)c;
    CK(hipMemset(c, 0, sizeof(Ctl)));
    CK(hipHostMalloc((void**)&dv[d].mirror, sizeof(Ctl), hipHostMallocDefault));
    CK(hipStreamCreateWithFlags(&dv[d].exec, hipStreamNonBlocking));
    CK(hipStreamCreateWithFlags(&dv[d].copy, hipStreamNonBlocking));
    if (d > 0) {
      CK(alloc_kind((void**)&dv[d].go, kPage, o.uncached));
      CK(hipMemset(dv[d].go, 0, kPage));
    }
    CK(hipDeviceSynchronize());
  }
  uint32_t* h_one = nullptr;
  CK(hipHostMalloc((void**)&h_one, 4, hipHostMallocDefault));
  *h_one = 1;

  const size_t stride =
      ((kPage + (size_t)o.chunks * 256) + kPage - 1) / kPage * kPage;
  uint8_t* cbuf = nullptr;
  CK(hipSetDevice(0));
  CK(alloc_kind((void**)&cbuf, stride * N, o.uncached));
  CK(hipMemset(cbuf, 0, stride * N));
  CK(hipDeviceSynchronize());

  const uint32_t nvec = o.chunks * 16;
  auto prod_args = [&](int p, uint64_t rounds) {
    ProdArgs a{};
    a.region = cbuf + (size_t)p * stride;
    a.go = dv[p + 1].go;
    a.ctl = dv[p + 1].ctl;
    a.rounds = rounds;
    a.tpms = dv[p + 1].tpms;
    a.budget = (int64_t)o.timeout_ms * a.tpms;
    a.seed = o.seed;
    a.p = (uint32_t)p;
    a.nvec = nvec;
    a.flush = o.flush;
    a.nchunks = o.chunks;
    a.inj_ff = o.inject_ff;
    a.inj_st = o.inject_st;
    a.inj_delay_ticks = (int64_t)(o.inject_delay_us * (double)a.tpms / 1000.0);
    return a;
  };
  auto cons_args = [&](uint64_t rounds) {
    ConsArgs a{};
    a.base = cbuf;
    for (int p = 0; p < N; p++) a.go[p] = dv[p + 1].go;
    a.ctl = dv[0].ctl;
    a.rounds = rounds;
    a.tpms = dv[0].tpms;
    a.budget = (int64_t)o.timeout_ms * a.tpms;
    a.seed = o.seed;
    a.nprod = (uint32_t)N;
    a.nvec = nvec;
    a.flush = o.flush;
    a.stride = (uint32_t)stride;
    a.nchunks = o.chunks;
    a.records = o.records;
    a.inj_ff = o.inject_ff;
    a.inj_st = o.inject_st;
    return a;
  };
  auto launch_cons = [&](uint64_t rounds) {
    CK(hipSetDevice(0));
    ConsArgs a = cons_args(rounds);
    switch (o.rd) {
      case 0:
        k_cons<0><<<1, kThreads, 0, dv[0].exec>>>(a);
        break;
      case 1:
        k_cons<1><<<1, kThreads, 0, dv[0].exec>>>(a);
        break;
      case 2:
        k_cons<2><<<1, kThreads, 0, dv[0].exec>>>(a);
        break;
      default:
        k_cons<3><<<1, kThreads, 0, dv[0].exec>>>(a);
        break;
    }
    CK(hipGetLastError());
  };
  auto launch_prod = [&](int p, uint64_t rounds) {
    CK(hipSetDevice(p + 1));
    k_prod<<<o.blocks, kThreads, 0, dv[p + 1].exec>>>(prod_args(p, rounds));
    CK(hipGetLastError());
  };
  // Warm-up launches (zero rounds) load the code objects before the clock
  // starts; the device timeouts must not include module loading.
  for (int p = 0; p < N; p++) launch_prod(p, 0);
  launch_cons(0);
  for (int d = 0; d < ndev; d++) {
    CK(hipSetDevice(d));
    CK(hipStreamSynchronize(dv[d].exec));
  }

  printf(
      "ORDER start nprod=%d rounds=%llu chunks=%u vectors=%u blocks=%u "
      "alloc=%s flush=%u rd=%u records=%u timeout_ms=%ld "
      "inject_flag_first=%llu inject_stale=%llu inject_delay_us=%.2f "
      "buses=%s\n",
      N, (unsigned long long)o.rounds, o.chunks, nvec, o.blocks,
      o.uncached ? "uncached" : "coarse", o.flush, o.rd, o.records,
      o.timeout_ms, (unsigned long long)o.inject_ff,
      (unsigned long long)o.inject_st, o.inject_delay_us, buses.c_str());

  for (int p = 0; p < N; p++) launch_prod(p, o.rounds);
  launch_cons(o.rounds);

  auto fetch = [&](int d, size_t bytes) {
    CK(hipSetDevice(d));
    CK(hipMemcpyAsync(dv[d].mirror, dv[d].ctl, bytes, hipMemcpyDeviceToHost,
                      dv[d].copy));
    CK(hipStreamSynchronize(dv[d].copy));
  };
  auto poke_all = [&]() {
    for (int d = 0; d < ndev; d++) {
      CK(hipSetDevice(d));
      CK(hipMemcpyAsync(&dv[d].ctl->host_abort, h_one, 4, hipMemcpyHostToDevice,
                        dv[d].copy));
    }
    for (int d = 0; d < ndev; d++) {
      CK(hipSetDevice(d));
      CK(hipStreamSynchronize(dv[d].copy));
    }
  };

  const auto t0 = Clock::now();
  auto t_rep = t0;
  uint64_t rounds_rep = 0;
  bool poked = false;
  auto t_poke = t0;
  auto report = [&](const Ctl* c, double dt) {
    printf(
        "ORDER rounds=%llu rate=%.0f/s bad_vectors=%llu bad_chunks=%llu "
        "bad_rounds=%llu late=%llu persistent=%llu bypass_ok=%llu "
        "bad_rounds_clean=%llu\n",
        (unsigned long long)c->rounds,
        dt > 0 ? (double)(c->rounds - rounds_rep) / dt : 0.0,
        (unsigned long long)c->bad_vectors, (unsigned long long)c->bad_chunks,
        (unsigned long long)c->bad_rounds, (unsigned long long)c->late,
        (unsigned long long)c->persistent, (unsigned long long)c->bypass_ok,
        (unsigned long long)c->bad_rounds_clean);
    rounds_rep = c->rounds;
  };
  for (;;) {
    bool all_done = true, any_abort = false;
    for (int d = 0; d < ndev; d++) {
      CK(hipSetDevice(d));
      hipError_t q = hipStreamQuery(dv[d].exec);
      if (q == hipErrorNotReady)
        all_done = false;
      else if (q != hipSuccess) {
        fprintf(stderr, "hx_order: kernel on ordinal %d failed: %s\n", d,
                hipGetErrorString(q));
        _exit(2);
      }
      fetch(d, 512);
      if (dv[d].mirror->abort_word != 0) any_abort = true;
    }
    const auto now = Clock::now();
    if (!poked && (g_stop || any_abort)) {
      poke_all();
      poked = true;
      t_poke = now;
    }
    if (poked && !all_done && now - t_poke > std::chrono::seconds(15)) {
      printf("ORDER abort: kernels did not stop 15 s after the stop request\n");
      _exit(3);
    }
    const double dt = std::chrono::duration<double>(now - t_rep).count();
    if (dt >= o.report_s) {
      report(dv[0].mirror, dt);
      t_rep = now;
    }
    if (all_done) break;
    usleep(100000);
  }
  const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
  for (int d = 0; d < ndev; d++) fetch(d, sizeof(Ctl));
  const Ctl* c = dv[0].mirror;
  int rc = c->bad_rounds ? 2 : 0;
  bool aborted = false;
  for (int d = 0; d < ndev; d++) {
    const Ctl* x = dv[d].mirror;
    if (x->abort_word != 1) continue;
    aborted = true;
    printf(
        "ORDER abort: dev=%d role=%s phase=%llu (%s) peer=%llu round=%llu "
        "waited_ms=%llu timeout_ms=%ld\n",
        d, d == 0 ? "consumer" : "producer",
        (unsigned long long)x->abort_rec[0],
        x->abort_rec[0] == 1 ? "consumer waiting for a producer flag"
                             : "producer waiting for go",
        (unsigned long long)x->abort_rec[1],
        (unsigned long long)x->abort_rec[2],
        (unsigned long long)x->abort_rec[3], o.timeout_ms);
  }
  printf(
      "ORDER done rounds=%llu bad_rounds=%llu bad_vectors=%llu bad_chunks=%llu "
      "late=%llu persistent=%llu bypass_ok=%llu bad_rounds_clean=%llu "
      "inject_flag_first=%llu inject_stale=%llu expected_ff_rounds=%llu "
      "expected_st_rounds=%llu secs=%.1f rate_avg=%.0f/s%s\n",
      (unsigned long long)c->rounds, (unsigned long long)c->bad_rounds,
      (unsigned long long)c->bad_vectors, (unsigned long long)c->bad_chunks,
      (unsigned long long)c->late, (unsigned long long)c->persistent,
      (unsigned long long)c->bypass_ok, (unsigned long long)c->bad_rounds_clean,
      (unsigned long long)o.inject_ff, (unsigned long long)o.inject_st,
      (unsigned long long)(o.inject_ff ? c->rounds / o.inject_ff : 0),
      (unsigned long long)(o.inject_st ? c->rounds / o.inject_st : 0), secs,
      secs > 0 ? (double)c->rounds / secs : 0.0,
      g_stop ? " stopped=signal" : "");
  for (int b = 0; b < kHistBins; b++)
    if (c->hist[b])
      printf("ORDER HIST delay_us=%s%.1f count=%llu\n",
             b == kHistBins - 1 ? ">=" : "", b * 0.1,
             (unsigned long long)c->hist[b]);
  const uint64_t nrec = std::min<uint64_t>(c->rec_count, o.records);
  for (uint64_t i = 0; i < nrec; i++) {
    const Rec& r = c->recs[i];
    const long long ns =
        r.late_ticks < 0 ? -1 : r.late_ticks * 1000000 / dv[0].tpms;
    static const char* kInj[4] = {"-", "ff", "st", "ff+st"};
    printf(
        "ORDER REC round=%llu prod=%u vec=%u chunk=%u got=%s exp=%s again=%s "
        "bypass=%s late_ns=%lld inject=%s\n",
        (unsigned long long)r.round, r.prod, r.vec, r.chunk,
        hex16(r.got).c_str(), hex16(r.exp).c_str(), hex16(r.again).c_str(),
        hex16(r.bypass).c_str(), ns, kInj[r.inject & 3u]);
  }
  if (aborted) rc = 3;

  for (int d = 0; d < ndev; d++) {
    CK(hipSetDevice(d));
    CK(hipStreamDestroy(dv[d].exec));
    CK(hipStreamDestroy(dv[d].copy));
    CK(hipHostFree(dv[d].mirror));
    CK(hipFree(dv[d].ctl));
    if (dv[d].go) CK(hipFree(dv[d].go));
  }
  CK(hipSetDevice(0));
  CK(hipFree(cbuf));
  CK(hipHostFree(h_one));
  return rc;
}
