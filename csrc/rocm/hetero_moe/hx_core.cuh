// hx: push protocol between a hub GPU (W7800, gfx1100) and V620 expert servers
// (gfx1030) of one PCIe island (docs/rdna2/hetero-hx-2026-11-07.md, spec
// docs/rdna2/hetero-moe-design-2026-11-07.md sections 6.4 and 9.4).
//
// Modelled on the rdna_ar protocol (rdna_allreduce.cuh): payload moves with
// posted P2P stores into the receiver's UNCACHED VRAM (a peer store never
// invalidates the owner's L2), then a system-scope fence, then one release
// store of a sequence-numbered flag that the receiver polls in its OWN VRAM (no
// PCIe reads while waiting). Waits have timeouts counted in device wall-clock
// ticks (rate calibrated against the host clock at init, never a constant), a
// sticky abort word and a first-claimer record, so a lost peer turns into a
// loud NaN/failed step instead of a hung GPU.
//
//   hub:    hx_k_begin_step  epoch++, header into every server inbox
//           hx_k_dispatch    spill rule + per-server masked ids, x/ids/w
//           pushes,
//                            flag after the last block's fence
//           hx_k_combine     wait server flags in fixed order, fp32 sum, fp16
//   server: hx_k_srv_begin   wait the hub's header, check it against the host
//                            descriptor
//           hx_k_srv_wait    wait the request flag, widen ids int16 -> int32,
//                            copy x and w into cached workspaces
//           hx_k_srv_push    y into the hub reply slot, fence, flag
//
// Everything here is arch-neutral HIP (wave32 on gfx1030 and gfx1100) and has
// no torch dependency: hx_ops.cu wraps it into torch ops, the standalone
// loopback test (tools/rdna/hetero_moe/hx_loopback.cpp) calls it directly.
// The constants below are the source of truth next to
// vllm/distributed/hetero_moe/hx_proto.py; a test keeps them equal (one
// `#define HX_<NAME> <literal>` per line, nothing computed).
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Constants (mirrored by hx_proto.py)
// ---------------------------------------------------------------------------
#define HX_VERSION 1
#define HX_MAGIC 0x48585F50524F5431ull
#define HX_OPCODE_MOE 1
#define HX_MAX_LAYERS 64
#define HX_MAX_SERVERS 8
#define HX_MAX_COLD 8
#define HX_MAX_SLOTS 4
#define HX_MAX_EXPERTS 512
#define HX_MAX_TOPK 16
#define HX_PAGE 4096
#define HX_HDR_BYTES 64
#define HX_CTX_WORDS 128
#define HX_FLAG_STRIDE 64
#define HX_CTRL_REQ_FLAG 256
#define HX_CTRL_REPLY_FLAG 0
#define HX_CTRL_HOST_ABORT 3840

// ctx.flush_mode (T5/soak variants, 0 = production). Bit 0: read one word of
// the peer control page back before a flag release. Bit 1: every wave reads
// back one word of each 256 B payload chunk it wrote (same address, behind its
// own posted stores) before the fence and the flag. Bit 2: the flag is
// published by a returning system-scope atomic exchange instead of a plain
// store (a non-posted request cannot pass the posted payload writes). Bit 3
// (consumer): spin about 4 us after the flags were seen before the payload is
// read (separates a late write from a stale read). Bits 4-7: how the hub
// combine reads the reply payload: 0 plain, 1 nontemporal, 2 cache-bypassing
// 8 B atomics, 3 plain loads after an acquire fence run by every wave.
#define HX_FLUSH_CTRL 1
#define HX_FLUSH_PAYLOAD 2
#define HX_FLUSH_ATOMIC 4
#define HX_SETTLE 8
#define HX_RD_SHIFT 4
#define HX_RD_MASK 15

// Bit 8 (set on BOTH roles): every exchange is checked end to end. The
// producer XORs a 32-bit hash of each 16 B vector it sends (keyed by the
// vector's position) into a digest and stores it, tagged with the low 32 bits
// of the flag's sequence, in the 8 bytes behind the flag; the consumer hashes
// the values it actually used and compares. A mismatch is read again after
// growing delays (a late posted write heals): a combine is recomputed, a
// request is copied again; what does not heal aborts with phase
// HX_PH_DIGEST. Counters: dig_stat[] (HX_DS_*, hx_dig_stats).
#define HX_DIGEST 256
#define HX_DIG_OFF 8
#define HX_DIG_RETRIES 5
#define HX_DIG_DELAY_US 4

// dig_stat[] words: exchanges that matched, exchanges whose first read did
// not match, mismatches that healed, mismatches that did not (abort), digest
// words that were not yet visible behind a seen flag, details of the last
// mismatch (A: epoch << 32 | layer << 16 | peer << 8 | rescans used,
// B: expected << 32 | got), rescans run in total.
#define HX_DS_OK 0
#define HX_DS_BAD 1
#define HX_DS_HEALED 2
#define HX_DS_UNHEALED 3
#define HX_DS_LATE 4
#define HX_DS_LAST_A 5
#define HX_DS_LAST_B 6
#define HX_DS_RESCANS 7
#define HX_DS_WORDS 8

// Abort record phases (record bits 4-7).
#define HX_PH_DISPATCH 1
#define HX_PH_COMBINE 2
#define HX_PH_SRV_WAIT 3
#define HX_PH_SRV_BEGIN 4
#define HX_PH_HDR 5
#define HX_PH_PUSH 6
#define HX_PH_PLAN 7
#define HX_PH_HOST 8
#define HX_PH_DIGEST 9

// Header mismatch fields (record peer byte of phase 5).
#define HX_MM_MAGIC 1
#define HX_MM_PLAN 2
#define HX_MM_EPOCH 3
#define HX_MM_T 4
#define HX_MM_BUCKET 5
#define HX_MM_OPCODE 6
#define HX_MM_VERSION 7
#define HX_MM_SKIPPED 8

// Record bits.
#define HX_REC_VALID 1
#define HX_REC_LIVE 2

// status[] words (hx_status returns them in this order).
#define HX_ST_RECORD 0
#define HX_ST_EPOCH 1
#define HX_ST_LIVE 2
#define HX_ST_STEPS 3
#define HX_ST_DISPATCHES 4
#define HX_ST_COMBINES 5
#define HX_ST_PAIRS_HUB 6
#define HX_ST_PAIRS_REMOTE 7

// layer_stats[layer][HX_LSTAT_COLS] (hub, optional, cumulative pair counts).
#define HX_LSTAT_COLS 4
#define HX_LSTAT_HUB 0
#define HX_LSTAT_COLD 1
#define HX_LSTAT_SPILL 2
#define HX_LSTAT_LOST 3

// Header field offsets (bytes) inside the 64 B header of the inbox control
// page.
#define HX_HDR_OFF_MAGIC 0
#define HX_HDR_OFF_EPOCH 8
#define HX_HDR_OFF_T 16
#define HX_HDR_OFF_BUCKET 20
#define HX_HDR_OFF_OPCODE 24
#define HX_HDR_OFF_VERSION 26
#define HX_HDR_OFF_MODE 28
#define HX_HDR_OFF_PLAN 32
#define HX_HDR_OFF_HUB 40
#define HX_HDR_OFF_SEQ 56

struct HxHdr {
  uint64_t magic;
  uint64_t epoch;
  uint32_t T;
  uint32_t bucket;
  uint16_t opcode;
  uint16_t version;
  uint32_t mode;
  uint64_t plan_hash;
  uint32_t hub_index;
  uint32_t rsv0;
  uint64_t rsv1;
  uint64_t seq;  // written LAST with release semantics; == epoch
};
static_assert(sizeof(HxHdr) == HX_HDR_BYTES, "hdr is 64 bytes");

// ---------------------------------------------------------------------------
// Device context: one per process and role (hub or server), lives in device
// memory (a torch int64[HX_CTX_WORDS] tensor), filled by the host at init and
// updated by the kernels. Kernel launches never take step state as arguments
// (a replayed graph would freeze it): epoch, K, descriptors are read from here.
// ---------------------------------------------------------------------------
struct HxCtx {
  uint64_t magic;  // HX_MAGIC once initialised
  uint32_t role;   // 1 hub, 2 server
  uint32_t version;
  uint64_t plan_hash;
  uint32_t index;  // hub index, or server index inside its island
  uint32_t nsrv;   // hub: servers it serves
  uint32_t rows_cap;
  uint32_t topk;
  uint32_t hidden;
  uint32_t nslots;
  uint32_t num_experts;
  uint32_t nlayers;
  uint32_t flush_mode;  // HX_FLUSH_* bits, HX_RD_SHIFT read mode
  uint32_t pad0;
  int64_t wait_ticks;        // per-wait bound inside a step
  int64_t begin_wait_ticks;  // server: bound of the wait for the next step
  int64_t ticks_per_ms;
  // Inbox geometry (hx_proto.inbox_layout): slot_stride, ids/w offsets inside a
  // slot; reply geometry: y_stride.
  uint64_t slot_stride;
  uint64_t off_ids;
  uint64_t off_w;
  uint64_t y_stride;
  // Step state. Hub: hx_k_begin_step. Server: adopted by hx_k_srv_begin.
  uint64_t last_epoch;
  // Server: host descriptor of the next step, written by ONE 16 B H2D copy
  // (desc_epoch, desc_T, desc_bucket are contiguous) before each replay.
  uint64_t desc_epoch;
  uint32_t desc_T;
  uint32_t desc_bucket;
  uint32_t cur_T;
  uint32_t cur_bucket;
  uint32_t cur_K;
  uint32_t cur_mode;
  uint32_t arrive_a;    // arrival counter of the last-block flag publication
  uint32_t abort_word;  // sticky claim (atomicCAS 0 -> 1)
  uint64_t status[8];   // see HX_ST_*; read back as one 64 B block
  uint8_t* own_base;    // own exported allocation (hub: reply, server: inbox)
  uint8_t*
      peer_base[HX_MAX_SERVERS];  // hub: server inboxes; server: [0] hub reply
  const int16_t* hub_slot;        // [nlayers][num_experts] hub slot or -1
  const int8_t* cold_srv;         // [nlayers][num_experts][HX_MAX_COLD], -1 pad
  const int8_t* n_cold;           // [nlayers][num_experts]
  const uint8_t*
      owned;  // [nlayers][num_experts] 1 = hub's island owns it; null = all
  const int16_t* ktab;  // [rows_cap + 1]: spill K for T rows
  int32_t* ids32;       // server workspace [rows_cap * topk]
  // Server workspaces in normal (cached) memory: hx_k_srv_wait copies the x
  // rows and the weights of the request out of the uncached inbox so the MoE
  // kernels re-read them through L2. Null = not copied (the consumer reads the
  // inbox slot itself).
  uint8_t* x_ws;  // [rows_cap][hidden] fp16
  float* w_ws;    // [rows_cap][topk]
  // Hub: cumulative per-layer pair counts, [HX_MAX_LAYERS][HX_LSTAT_COLS] u64.
  // Null = off (zero cost, graph safe).
  uint64_t* layer_stats;
  // Digest mode (HX_DIGEST): arrival counter of the verifying kernel (combine,
  // srv_wait), the cross-block XOR accumulators of the exchange in flight
  // ([0] x rows / combined, [1 + s] ids and weights of server s; kernels
  // leave them zero), the counters, and four words a fault-injection build
  // (HX_FAULT_INJECT, tests only) reads. All zero in production.
  uint32_t arrive_b;
  uint32_t dig_pad;
  uint32_t dig_acc[HX_MAX_SERVERS + 2];
  uint64_t dig_stat[HX_DS_WORDS];
  uint64_t dbg[4];
};
static_assert(sizeof(HxCtx) <= HX_CTX_WORDS * 8, "ctx fits HX_CTX_WORDS");
static_assert(offsetof(HxCtx, status) % 8 == 0, "status is 8-byte aligned");

// ---------------------------------------------------------------------------
// Device primitives
// ---------------------------------------------------------------------------
#define HX_POLL_PAUSE() __builtin_amdgcn_s_sleep(2)

__device__ __forceinline__ uint64_t hx_ld_acq(const uint64_t* p) {
  return __hip_atomic_load(const_cast<uint64_t*>(p), __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ uint64_t hx_ld_rlx(const uint64_t* p) {
  return __hip_atomic_load(const_cast<uint64_t*>(p), __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ uint32_t hx_ld_rlx32(const uint32_t* p) {
  return __hip_atomic_load(const_cast<uint32_t*>(p), __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void hx_st_rel(uint64_t* p, uint64_t v) {
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void hx_st_rlx(uint64_t* p, uint64_t v) {
  __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void hx_st_rlx32(uint32_t* p, uint32_t v) {
  __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

// Serial-number order: true when flag value `a` is a sequence before `b`.
__device__ __forceinline__ bool hx_before(uint64_t a, uint64_t b) {
  return (int64_t)(a - b) < 0;
}
__device__ __forceinline__ uint64_t hx_seq(uint64_t epoch, int layer) {
  return epoch * 64ull + (uint64_t)layer + 1ull;
}

__device__ __forceinline__ uint64_t hx_record(uint32_t phase, uint32_t peer,
                                              uint64_t seq, int64_t ms,
                                              uint32_t bits) {
  const uint64_t u = ms < 0 ? 0ull : (ms > 0xFFFF ? 0xFFFFull : (uint64_t)ms);
  return (uint64_t)HX_REC_VALID | (uint64_t)bits |
         ((uint64_t)(phase & 0xFu) << 4) | ((uint64_t)(peer & 0xFFu) << 8) |
         (u << 16) | ((seq & 0xFFFFFFFFull) << 32);
}

__device__ __forceinline__ bool hx_aborted(const HxCtx* c) {
  return __hip_atomic_load(const_cast<uint32_t*>(&c->abort_word),
                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != 0u;
}

// The first claimer wins and writes the record; later claimers do nothing.
__device__ __forceinline__ bool hx_claim(HxCtx* c, uint64_t rec) {
  if (atomicCAS(&c->abort_word, 0u, 1u) == 0u) {
    __hip_atomic_store(&c->status[HX_ST_RECORD], rec, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_SYSTEM);
    return true;
  }
  return false;
}

__device__ __forceinline__ const uint64_t* hx_host_abort_ptr(const HxCtx* c) {
  return reinterpret_cast<const uint64_t*>(c->own_base + HX_CTRL_HOST_ABORT);
}

#define HX_WAIT_OK 0
#define HX_WAIT_TIMEOUT 1
#define HX_WAIT_ABORT 2

// Wait (one thread) until `*flag` has reached `seq`. A wait past `budget`
// ticks claims the abort with (phase, peer); an abort claimed elsewhere, or
// the host poking the control page (peer death seen by pidfd), ends the wait
// without a second claim (host pokes are recorded as phase HX_PH_HOST).
// `live_writer` refreshes status[HX_ST_LIVE] once per second of waiting so a
// hung wait is visible before it times out.
__device__ __forceinline__ int hx_wait_flag(HxCtx* c, const uint64_t* flag,
                                            uint64_t seq, int64_t budget,
                                            uint32_t phase, uint32_t peer,
                                            bool live_writer) {
  if (!hx_before(hx_ld_acq(flag), seq)) return HX_WAIT_OK;
  const int64_t tpm = c->ticks_per_ms;
  const int64_t t0 = wall_clock64();
  int64_t next_live = 1000 * tpm;
  unsigned it = 0;
  for (;;) {
    HX_POLL_PAUSE();
    if (!hx_before(hx_ld_acq(flag), seq)) return HX_WAIT_OK;
    const int64_t dt = wall_clock64() - t0;
    if ((++it & 7u) == 0u || dt > budget) {
      if (hx_ld_rlx(hx_host_abort_ptr(c)) != 0ull) {
        hx_claim(c, hx_record(HX_PH_HOST, peer, seq, dt / tpm, 0u));
        return HX_WAIT_ABORT;
      }
      if (hx_aborted(c)) return HX_WAIT_ABORT;
    }
    if (dt > budget) {
      // This wave may have been descheduled since its last poll.
      if (!hx_before(hx_ld_acq(flag), seq)) return HX_WAIT_OK;
      hx_claim(c, hx_record(phase, peer, seq, dt / tpm, 0u));
      return HX_WAIT_TIMEOUT;
    }
    if (live_writer && dt >= next_live) {
      hx_st_rlx(&c->status[HX_ST_LIVE],
                hx_record(phase, peer, seq, dt / tpm, HX_REC_LIVE));
      next_live = dt + 1000 * tpm;
    }
  }
}

// ---------------------------------------------------------------------------
// Addressing
// ---------------------------------------------------------------------------
// Inbox of server s as seen from the hub (hub only) / own inbox (server).
__device__ __forceinline__ uint8_t* hx_inbox_ctrl(const HxCtx* c, int s) {
  return (c->role == 1 ? c->peer_base[s] : c->own_base);
}
__device__ __forceinline__ uint8_t* hx_inbox_slot(const HxCtx* c, int s,
                                                  int slot) {
  return hx_inbox_ctrl(c, s) + HX_PAGE + (uint64_t)slot * c->slot_stride;
}
// Reply control page and y slot of server s (hub: own_base; server: peer 0).
__device__ __forceinline__ uint8_t* hx_reply_base(const HxCtx* c) {
  return c->role == 1 ? c->own_base : c->peer_base[0];
}
__device__ __forceinline__ uint64_t* hx_reply_flag(const HxCtx* c, int s,
                                                   int slot) {
  return reinterpret_cast<uint64_t*>(hx_reply_base(c) + HX_CTRL_REPLY_FLAG +
                                     (uint64_t)HX_FLAG_STRIDE *
                                         (uint64_t)(s * c->nslots + slot));
}
__device__ __forceinline__ uint8_t* hx_reply_y(const HxCtx* c, int s,
                                               int slot) {
  return hx_reply_base(c) + HX_PAGE +
         (uint64_t)(s * c->nslots + slot) * c->y_stride;
}
__device__ __forceinline__ uint64_t* hx_req_flag(const HxCtx* c, int s,
                                                 int slot) {
  return reinterpret_cast<uint64_t*>(hx_inbox_ctrl(c, s) + HX_CTRL_REQ_FLAG +
                                     (uint64_t)HX_FLAG_STRIDE * slot);
}

// Optional flush: a non-posted read of the peer cannot pass the posted
// payload writes ahead of it on the same path, so it drains them (used only
// when stores were ever seen out of order; ctx.flush_mode, T1/T5).
__device__ __forceinline__ void hx_flush_peer(const uint8_t* base) {
  if (base != nullptr)
    (void)__hip_atomic_load(
        reinterpret_cast<uint32_t*>(const_cast<uint8_t*>(base)) + 2,
        __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

// Read-back of one dword the calling wave just wrote to a peer: a non-posted
// read cannot pass the posted writes ahead of it and reads the same address
// back, so its data means the write reached the target. The wave's next fence
// (s_waitcnt vmcnt(0)) waits for it; hx_sink keeps the value alive.
__device__ __forceinline__ uint32_t hx_readback(const void* p) {
  return __hip_atomic_load(reinterpret_cast<const uint32_t*>(p),
                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void hx_sink(uint32_t v) {
  asm volatile("" : : "v"(v));
}

// Flag publication: a plain release store, or (HX_FLUSH_ATOMIC) a returning
// release exchange: the wave waits for the completion of a non-posted request.
__device__ __forceinline__ void hx_publish(const HxCtx* c, uint64_t* p,
                                           uint64_t v) {
  if (c->flush_mode & HX_FLUSH_ATOMIC) {
    const uint64_t old = __hip_atomic_exchange(p, v, __ATOMIC_RELEASE,
                                               __HIP_MEMORY_SCOPE_SYSTEM);
    hx_sink((uint32_t)old ^ (uint32_t)(old >> 32));
  } else {
    hx_st_rel(p, v);
  }
}

// HX_SETTLE: wait about 4 us (ticks_per_ms / 250) after the flags were seen.
__device__ __forceinline__ void hx_settle(const HxCtx* c) {
  if (c->flush_mode & HX_SETTLE) {
    const int64_t until = wall_clock64() + c->ticks_per_ms / 250;
    while (wall_clock64() < until) {
    }
  }
}

typedef unsigned int hx_u32x4 __attribute__((ext_vector_type(4)));

// Consumer-side read of reply payload (RD = ctx.flush_mode bits 4-7): 0 plain
// (production), 1 nontemporal, 2 cache-bypassing relaxed system-scope 8 B
// atomics; mode 3 is plain loads after an acquire fence in every wave.
template <int RD>
__device__ __forceinline__ uint4 hx_ld_reply(const uint4* p) {
  if constexpr (RD == 1) {
    const hx_u32x4 r =
        __builtin_nontemporal_load(reinterpret_cast<const hx_u32x4*>(p));
    return make_uint4(r.x, r.y, r.z, r.w);
  } else if constexpr (RD == 2) {
    const uint64_t* q = reinterpret_cast<const uint64_t*>(p);
    const uint64_t a =
        __hip_atomic_load(q, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    const uint64_t b =
        __hip_atomic_load(q + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    return make_uint4((uint32_t)a, (uint32_t)(a >> 32), (uint32_t)b,
                      (uint32_t)(b >> 32));
  } else {
    return *p;
  }
}

// ---------------------------------------------------------------------------
// Digest (ctx.flush_mode & HX_DIGEST)
// ---------------------------------------------------------------------------
// 32-bit hash of one 16 B vector at position k. Every step is a bijection of
// the running value, so changing any one word of one vector changes the hash.
// A digest is the XOR of the hashes of all vectors of a payload, so the order
// of accumulation does not matter. tests/distributed/hetero_moe/hx_digest.py
// holds the same function.
__device__ __forceinline__ uint32_t hx_dig_vec(uint32_t k, uint4 v) {
  uint32_t h = (k + 0x9E3779B9u) * 0x85EBCA6Bu;
  h = (h ^ v.x) * 0xC2B2AE35u;
  h ^= h >> 15;
  h = (h ^ v.y) * 0x27D4EB2Fu;
  h ^= h >> 13;
  h = (h ^ v.z) * 0x165667B1u;
  h ^= h >> 16;
  h = (h ^ v.w) * 0x85EBCA77u;
  h ^= h >> 14;
  return h;
}
// One pair of a request: id as stored in the inbox (int16 bits, 0xFFFF = none)
// and the weight's bits. Keys 0x80000000 | p never meet the x rows' keys.
__device__ __forceinline__ uint32_t hx_dig_idw(uint32_t p, uint32_t id16,
                                               uint32_t wbits) {
  return hx_dig_vec(0x80000000u | p, make_uint4(id16, wbits, 0u, 0u));
}

// Digest word of a flag: the 8 bytes behind it (same 64 B line), tagged with
// the low 32 bits of the flag's sequence so that a stale word is recognised.
static_assert(HX_DIG_OFF == 8, "the digest word follows the 8 B flag");
__device__ __forceinline__ uint64_t* hx_reply_dig(const HxCtx* c, int s,
                                                  int slot) {
  return hx_reply_flag(c, s, slot) + 1;
}
__device__ __forceinline__ uint64_t* hx_req_dig(const HxCtx* c, int s,
                                                int slot) {
  return hx_req_flag(c, s, slot) + 1;
}
__device__ __forceinline__ uint64_t hx_dig_word(uint64_t seq, uint32_t dig) {
  return (seq << 32) | (uint64_t)dig;
}

__device__ __forceinline__ void hx_dig_count(HxCtx* c, int word,
                                             unsigned long long n) {
  atomicAdd(reinterpret_cast<unsigned long long*>(&c->dig_stat[word]), n);
}

// Wait (one thread) for the digest word of the exchange `seq` behind a flag
// that was seen. False after an abort, or after the wait budget (which claims
// HX_PH_DIGEST with peer | 0x80).
__device__ __forceinline__ bool hx_wait_dig(HxCtx* c, const uint64_t* p,
                                            uint64_t seq, uint32_t peer,
                                            uint32_t* dig) {
  const uint32_t tag = (uint32_t)seq;
  uint64_t w = hx_ld_acq(p);
  if ((uint32_t)(w >> 32) != tag) {
    hx_dig_count(c, HX_DS_LATE, 1ull);
    const int64_t t0 = wall_clock64();
    for (;;) {
      HX_POLL_PAUSE();
      w = hx_ld_acq(p);
      if ((uint32_t)(w >> 32) == tag) break;
      if (hx_aborted(c)) return false;
      const int64_t dt = wall_clock64() - t0;
      if (dt > c->wait_ticks) {
        w = hx_ld_acq(p);
        if ((uint32_t)(w >> 32) == tag) break;
        hx_claim(c, hx_record(HX_PH_DIGEST, peer | 0x80u, seq,
                              dt / c->ticks_per_ms, 0u));
        return false;
      }
    }
  }
  *dig = (uint32_t)w;
  return true;
}

// Spin (one thread) for `us` microseconds.
__device__ __forceinline__ void hx_spin_us(const HxCtx* c, int64_t us) {
  const int64_t until = wall_clock64() + c->ticks_per_ms * us / 1000;
  while (wall_clock64() < until) {
  }
}
// Pause before rescan number `attempt`: 4, 16, 64, 256, 1024 us.
__device__ __forceinline__ void hx_dig_delay(const HxCtx* c, int attempt) {
  hx_spin_us(c, (int64_t)HX_DIG_DELAY_US << (2 * attempt));
}

// ---------------------------------------------------------------------------
// Hub: begin step
// ---------------------------------------------------------------------------
// One block, one thread per server: epoch++ (device counter, so a replayed
// graph advances it), header into every server inbox, then hdr_seq with
// release semantics so a server that sees it sees the whole header.
__global__ void __launch_bounds__(64)
    hx_k_begin_step(HxCtx* __restrict__ c, int T, int bucket) {
  __shared__ uint64_t s_epoch;
  __shared__ int s_go;
  const int t = threadIdx.x;
  if (t == 0) {
    const bool ab = hx_aborted(c);
    s_go = ab ? 0 : 1;
    if (!ab) {
      const uint64_t e = c->status[HX_ST_EPOCH] + 1ull;
      c->status[HX_ST_EPOCH] = e;
      c->status[HX_ST_STEPS] += 1ull;
      c->cur_T = (uint32_t)T;
      c->cur_bucket = (uint32_t)bucket;
      c->cur_mode = bucket > 0 ? 0u : 1u;
      const int tk = T < (int)c->rows_cap ? T : (int)c->rows_cap;
      c->cur_K = c->ktab != nullptr ? (uint32_t)c->ktab[tk] : 0u;
      s_epoch = e;
    }
  }
  __syncthreads();
  if (!s_go) return;
  const int nsrv = (int)c->nsrv;
  if (t < nsrv) {
    HxHdr* h = reinterpret_cast<HxHdr*>(c->peer_base[t]);
    h->magic = HX_MAGIC;
    h->epoch = s_epoch;
    h->T = (uint32_t)T;
    h->bucket = (uint32_t)bucket;
    h->opcode = HX_OPCODE_MOE;
    h->version = HX_VERSION;
    h->mode = c->cur_mode;
    h->plan_hash = c->plan_hash;
    h->hub_index = c->index;
    __threadfence_system();
    if (c->flush_mode & (HX_FLUSH_CTRL | HX_FLUSH_PAYLOAD))
      hx_flush_peer(c->peer_base[t]);
    hx_publish(c, &h->seq, s_epoch);
  }
}

// ---------------------------------------------------------------------------
// Hub: dispatch
// ---------------------------------------------------------------------------
// Grid: nblocks x 256. Block 0 additionally runs the spill rule and writes the
// per-server masked ids and the weights; every block pushes a slice of x to
// every server. No block waits for another: the LAST block to finish (ticket)
// publishes the request flags after every other block's payload was fenced.
//
// Spill rule (exact, == HubMethod's ref_hx and hx_model.dispatch_ref): pair
// valid iff 0 <= id < E and owned[layer][id] (no owned table: all owned; a
// non-owned pair is dropped, never a plan error); resident iff valid and
// hub_slot[layer][id] >= 0; rank(id) = # touched resident ids with a smaller
// slot; the pair stays on the hub iff resident and rank < K. Otherwise it goes
// to the cold copy cold_srv[layer][id][(row + layer) % n_cold[layer][id]].
#define HX_DISPATCH_THREADS 256

// x rows to every server: one load, nsrv posted stores. Returns the XOR of the
// hashes of the vectors this thread sent (DIG, else 0). skip_srv / skip_chunk
// are the fault-injection hook of test builds: that server's chunk is not
// written here (hx_k_dispatch writes it late).
template <bool DIG>
__device__ __forceinline__ uint32_t
hx_dispatch_x(const HxCtx* c, const uint4* __restrict__ x4, int nsrv, int slot,
              long long nvec, long long gid, long long gstride, int skip_srv,
              long long skip_chunk) {
  uint32_t dg = 0u;
  for (long long v = gid; v < nvec; v += gstride) {
    const uint4 val = x4[v];
    if constexpr (DIG) dg ^= hx_dig_vec((uint32_t)v, val);
    for (int s = 0; s < nsrv; s++) {
#ifdef HX_FAULT_INJECT
      if (s == skip_srv && (v >> 4) == skip_chunk) {
        // The chunk holds garbage until hx_k_dispatch writes the data late.
        uint4 bad = val;
        bad.x ^= 0xA5A5A5A5u;
        reinterpret_cast<uint4*>(hx_inbox_slot(c, s, slot))[v] = bad;
        continue;
      }
#endif
      reinterpret_cast<uint4*>(hx_inbox_slot(c, s, slot))[v] = val;
    }
  }
  (void)skip_srv;
  (void)skip_chunk;
  return dg;
}

__global__ void __launch_bounds__(HX_DISPATCH_THREADS)
    hx_k_dispatch(HxCtx* __restrict__ c, const uint4* __restrict__ x4,
                  const int32_t* __restrict__ ids, const float* __restrict__ w,
                  int32_t* __restrict__ hub_out, int T, int layer,
                  int nblocks) {
  __shared__ uint32_t s_touched[HX_MAX_EXPERTS / 32];
  __shared__ uint32_t s_keep[HX_MAX_EXPERTS / 32];
  __shared__ uint32_t s_prefix[HX_MAX_EXPERTS / 32 + 1];
  __shared__ int s_abort;
  __shared__ int s_plan_err;
  __shared__ uint32_t s_cnt_hub, s_cnt_cold, s_cnt_spill, s_cnt_lost;
  __shared__ uint64_t s_seq;
  __shared__ uint32_t s_dig[HX_MAX_SERVERS + 1];
  const int t = threadIdx.x, b = blockIdx.x;
  constexpr int NT = HX_DISPATCH_THREADS;
  const int nsrv = (int)c->nsrv, topk = (int)c->topk;
  const int E = (int)c->num_experts;
  const int slot = layer % (int)c->nslots;
  const int P = T * topk;
  const bool dig = (c->flush_mode & HX_DIGEST) != 0u;
  if (t == 0) {
    s_abort = hx_aborted(c) ? 1 : 0;
    s_seq = hx_seq(c->status[HX_ST_EPOCH], layer);
  }
  if (t <= HX_MAX_SERVERS) s_dig[t] = 0u;
  __syncthreads();
  if (s_abort) {
    // Sticky abort: nothing is pushed, nothing is published; the hub computes
    // no local pair and combine emits NaN.
    if (b == 0)
      for (int p = t; p < P; p += NT) hub_out[p] = -1;
    return;
  }

  if (b == 0) {
    if (t < HX_MAX_EXPERTS / 32) s_touched[t] = 0u;
    if (t == 0) {
      s_plan_err = 0;
      s_cnt_hub = 0u;
      s_cnt_cold = 0u;
      s_cnt_spill = 0u;
      s_cnt_lost = 0u;
    }
    __syncthreads();
    const int16_t* slot_of = c->hub_slot + (size_t)layer * E;
    const uint8_t* own_l =
        c->owned != nullptr ? c->owned + (size_t)layer * E : nullptr;
    for (int p = t; p < P; p += NT) {
      const int e = ids[p];
      if ((unsigned)e < (unsigned)E && (own_l == nullptr || own_l[e] != 0)) {
        const int sl = slot_of[e];
        if (sl >= 0) atomicOr(&s_touched[sl >> 5], 1u << (sl & 31));
      }
    }
    __syncthreads();
    // Rank = number of touched slots below. Keep the first K touched slots.
    if (t == 0) {
      uint32_t acc = 0u;
      for (int i = 0; i < HX_MAX_EXPERTS / 32; i++) {
        s_prefix[i] = acc;
        acc += (uint32_t)__popc(s_touched[i]);
      }
      s_prefix[HX_MAX_EXPERTS / 32] = acc;
    }
    __syncthreads();
    if (t < HX_MAX_EXPERTS / 32) {
      const uint32_t K = c->cur_K;
      const uint32_t base = s_prefix[t];
      uint32_t word = s_touched[t], keep = 0u;
      if (K > base) {
        uint32_t room = K - base;  // set bits of this word that may stay
        while (word != 0u && room > 0u) {
          const uint32_t low = word & (~word + 1u);
          keep |= low;
          word ^= low;
          room--;
        }
      }
      s_keep[t] = keep;
    }
    __syncthreads();
    const int8_t* ncold = c->n_cold + (size_t)layer * E;
    const int8_t* csrv = c->cold_srv + (size_t)layer * E * HX_MAX_COLD;
    uint32_t n_hub = 0u, n_cold = 0u, n_spill = 0u, n_lost = 0u;
    uint32_t idw[HX_MAX_SERVERS] = {};
    for (int p = t; p < P; p += NT) {
      const int e = ids[p];
      const int r = p / topk;
      int dest = -1, keep_here = 0;
      if ((unsigned)e < (unsigned)E && (own_l == nullptr || own_l[e] != 0)) {
        const int sl = slot_of[e];
        if (sl >= 0 && ((s_keep[sl >> 5] >> (sl & 31)) & 1u)) {
          keep_here = 1;
          n_hub++;
        } else {
          const int nc = ncold[e];
          if (nc <= 0) {
            s_plan_err = 1;
            n_lost++;
          } else {
            const int j = (r + layer) % nc;
            dest = csrv[(size_t)e * HX_MAX_COLD + j];
            if (dest < 0 || dest >= nsrv) {
              dest = -1;
              s_plan_err = 1;
              n_lost++;
            } else if (sl >= 0) {
              n_spill++;
            } else {
              n_cold++;
            }
          }
        }
      }
      hub_out[p] = keep_here ? e : -1;
      const float wv = w[p];
      for (int s = 0; s < nsrv; s++) {
        uint8_t* sb = hx_inbox_slot(c, s, slot);
        reinterpret_cast<int16_t*>(sb + c->off_ids)[p] =
            (s == dest) ? (int16_t)e : (int16_t)-1;
        reinterpret_cast<float*>(sb + c->off_w)[p] = wv;
      }
      if (dig) {
#pragma unroll
        for (int s = 0; s < HX_MAX_SERVERS; s++)
          if (s < nsrv)
            idw[s] ^= hx_dig_idw((uint32_t)p,
                                 (s == dest) ? (uint32_t)(uint16_t)e : 0xFFFFu,
                                 __float_as_uint(wv));
      }
    }
    if (dig) {
#pragma unroll
      for (int s = 0; s < HX_MAX_SERVERS; s++)
        if (s < nsrv && idw[s] != 0u) atomicXor(&s_dig[1 + s], idw[s]);
    }
    if (c->flush_mode & HX_FLUSH_PAYLOAD) {
      // Read back what this wave wrote: one non-posted read per server behind
      // its posted ids and weights stores (the thread's last pair).
      uint32_t sink = 0u;
      if (t < P) {
        const int pl = t + ((P - 1 - t) / NT) * NT;
        for (int s = 0; s < nsrv; s++) {
          const uint8_t* sb = hx_inbox_slot(c, s, slot);
          sink += hx_readback(sb + c->off_ids + (size_t)(pl & ~1) * 2);
          sink += hx_readback(sb + c->off_w + (size_t)pl * 4);
        }
      }
      hx_sink(sink);
    }
    atomicAdd(&s_cnt_hub, n_hub);
    atomicAdd(&s_cnt_cold, n_cold);
    atomicAdd(&s_cnt_spill, n_spill);
    atomicAdd(&s_cnt_lost, n_lost);
    __syncthreads();
    if (t == 0) {
      if (s_plan_err) hx_claim(c, hx_record(HX_PH_PLAN, 0u, s_seq, 0, 0u));
      c->status[HX_ST_PAIRS_HUB] += s_cnt_hub;
      c->status[HX_ST_PAIRS_REMOTE] += (uint64_t)s_cnt_cold + s_cnt_spill;
      if (c->layer_stats != nullptr) {
        uint64_t* ls = c->layer_stats + (size_t)layer * HX_LSTAT_COLS;
        ls[HX_LSTAT_HUB] += s_cnt_hub;
        ls[HX_LSTAT_COLD] += s_cnt_cold;
        ls[HX_LSTAT_SPILL] += s_cnt_spill;
        ls[HX_LSTAT_LOST] += s_cnt_lost;
      }
    }
  }

  // x rows to every server: one load, nsrv posted stores.
  const long long nvec = (long long)T * (long long)c->hidden / 8;
  const long long gid = (long long)b * NT + t;
  const long long gstride = (long long)nblocks * NT;
  int skip_srv = -1;
  long long skip_chunk = -1;
#ifdef HX_FAULT_INJECT
  const bool inj = c->dbg[0] != 0ull && (uint32_t)c->dbg[0] == (uint32_t)s_seq;
  if (inj) {
    skip_srv = (int)c->dbg[3];
    skip_chunk = (long long)c->dbg[1];
  }
#endif
  const uint32_t dgx =
      dig ? hx_dispatch_x<true>(c, x4, nsrv, slot, nvec, gid, gstride, skip_srv,
                                skip_chunk)
          : hx_dispatch_x<false>(c, x4, nsrv, slot, nvec, gid, gstride,
                                 skip_srv, skip_chunk);
  if (c->flush_mode & HX_FLUSH_PAYLOAD) {
    uint32_t sink = 0u;
    for (long long v = gid; v < nvec; v += gstride)
      if ((v & 15) == 15 || v == nvec - 1)
        for (int s = 0; s < nsrv; s++)
          sink += hx_readback(hx_inbox_slot(c, s, slot) + (uint64_t)v * 16);
    hx_sink(sink);
  }
  if (dig) {
    // Digest of what this block sent, into the cross-block accumulators.
    if (dgx != 0u) atomicXor(&s_dig[0], dgx);
    __syncthreads();
    if (t <= nsrv && s_dig[t] != 0u) atomicXor(&c->dig_acc[t], s_dig[t]);
  }
  // Payload before flag: every thread fences its own stores, the block
  // barrier collects the block, the last block to arrive publishes the flags.
  __threadfence_system();
  __syncthreads();
  if (t == 0) {
    const unsigned ticket = __hip_atomic_fetch_add(
        &c->arrive_a, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    if (ticket == (unsigned)nblocks - 1u) {
      c->arrive_a = 0u;
      if (dig) {
        // All blocks are in: read and clear the accumulators.
#pragma unroll
        for (int i = 0; i <= HX_MAX_SERVERS; i++)
          if (i <= nsrv)
            s_dig[i] = __hip_atomic_exchange(
                &c->dig_acc[i], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
      for (int s = 0; s < nsrv; s++) {
        if (dig)
          hx_st_rlx(hx_req_dig(c, s, slot),
                    hx_dig_word(s_seq, s_dig[0] ^ s_dig[1 + s]));
        if (c->flush_mode & HX_FLUSH_CTRL) hx_flush_peer(hx_inbox_ctrl(c, s));
        hx_publish(c, hx_req_flag(c, s, slot), s_seq);
      }
      c->status[HX_ST_DISPATCHES] += 1ull;
#ifdef HX_FAULT_INJECT
      if (inj) {
        // The skipped chunk reaches the server dbg[2] us after its flag.
        hx_spin_us(c, (int64_t)c->dbg[2]);
        for (int i = 0; i < 16; i++) {
          const long long v = skip_chunk * 16 + i;
          if (v < nvec)
            reinterpret_cast<uint4*>(hx_inbox_slot(c, skip_srv, slot))[v] =
                x4[v];
        }
        __threadfence_system();
      }
#endif
    }
  }
}

// ---------------------------------------------------------------------------
// Hub: combine
// ---------------------------------------------------------------------------
// out = fp16( ((f32(y_hub) + f32(y_0)) + f32(y_1)) + ... ), servers in order.
// Each block's thread 0 waits for the S reply flags in order (timeouts
// per wait), then every thread sums its vectors. An abort (this launch's or an
// earlier sticky one) fills the whole output with NaN.
#define HX_COMBINE_THREADS 256
#define HX_NAN_HALF2 0x7E007E00u

// DIG: also XOR the hashes of the reply vectors that were actually summed into
// s_dig[s] (LDS, one word per server).
template <int RD, bool DIG>
__device__ __forceinline__ void hx_combine_sum(
    const HxCtx* c, const uint4* y_hub, uint4* out, int nsrv, int slot,
    long long nvec, long long gid, long long gstride, uint32_t* s_dig) {
  const uint4* ys[HX_MAX_SERVERS];
  for (int s = 0; s < HX_MAX_SERVERS; s++)
    ys[s] = s < nsrv ? reinterpret_cast<const uint4*>(hx_reply_y(c, s, slot))
                     : nullptr;
  uint32_t dg[HX_MAX_SERVERS];
#pragma unroll
  for (int s = 0; s < HX_MAX_SERVERS; s++) dg[s] = 0u;
  for (long long v = gid; v < nvec; v += gstride) {
    uint4 raw[HX_MAX_SERVERS + 1];
    raw[0] = y_hub[v];
#pragma unroll
    for (int s = 0; s < HX_MAX_SERVERS; s++)
      if (s < nsrv) raw[s + 1] = hx_ld_reply<RD>(ys[s] + v);
    if constexpr (DIG) {
#pragma unroll
      for (int s = 0; s < HX_MAX_SERVERS; s++)
        if (s < nsrv) dg[s] ^= hx_dig_vec((uint32_t)v, raw[s + 1]);
    }
    float acc[8];
    {
      const __half* h = reinterpret_cast<const __half*>(&raw[0]);
#pragma unroll
      for (int q = 0; q < 8; q++) acc[q] = __half2float(h[q]);
    }
#pragma unroll
    for (int s = 0; s < HX_MAX_SERVERS; s++) {
      if (s < nsrv) {
        const __half* h = reinterpret_cast<const __half*>(&raw[s + 1]);
#pragma unroll
        for (int q = 0; q < 8; q++) acc[q] += __half2float(h[q]);
      }
    }
    uint4 res;
    __half* ro = reinterpret_cast<__half*>(&res);
#pragma unroll
    for (int q = 0; q < 8; q++) ro[q] = __float2half(acc[q]);
    out[v] = res;
  }
  if constexpr (DIG) {
#pragma unroll
    for (int s = 0; s < HX_MAX_SERVERS; s++)
      if (s < nsrv && dg[s] != 0u) atomicXor(&s_dig[s], dg[s]);
  }
}

__device__ __forceinline__ void hx_fill_nan(uint4* out, long long nvec,
                                            long long gid, long long gstride) {
  uint4 nan;
  nan.x = nan.y = nan.z = nan.w = HX_NAN_HALF2;
  for (long long v = gid; v < nvec; v += gstride) out[v] = nan;
}

// Digest mode of a combine (all threads of a block): sum while hashing what was
// read, add the block's hashes to the cross-block accumulators, and let the
// LAST block to arrive compare them with the servers' digest words. A mismatch
// is read again after growing delays; when the replies became right the last
// block recomputes the whole output (and checks the digests of that pass);
// what does not heal aborts with phase HX_PH_DIGEST and a NaN output.
template <int RD>
__device__ __forceinline__ void hx_combine_digest(
    HxCtx* c, const uint4* y_hub, uint4* out, int layer, int nblocks, int slot,
    int nsrv, long long nvec, long long gid, long long gstride) {
  constexpr int NT = HX_COMBINE_THREADS;
  __shared__ uint32_t s_dig[HX_MAX_SERVERS];
  __shared__ uint32_t s_exp[HX_MAX_SERVERS];
  __shared__ uint32_t s_got[HX_MAX_SERVERS];
  __shared__ uint32_t s_re[HX_MAX_SERVERS];
  __shared__ uint32_t s_badmask;
  __shared__ int s_last, s_same;
  __shared__ uint64_t s_seq;
  const int t = threadIdx.x;
  if (t < HX_MAX_SERVERS) s_dig[t] = 0u;
  if (t == 0) {
    s_badmask = 0u;
    s_seq = hx_seq(c->status[HX_ST_EPOCH], layer);
  }
  __syncthreads();
  hx_combine_sum<RD, true>(c, y_hub, out, nsrv, slot, nvec, gid, gstride,
                           s_dig);
  __syncthreads();
  if (t < nsrv && s_dig[t] != 0u) atomicXor(&c->dig_acc[t], s_dig[t]);
  __threadfence();
  __syncthreads();
  if (t == 0) {
    const unsigned ticket = __hip_atomic_fetch_add(
        &c->arrive_b, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    s_last = ticket == (unsigned)nblocks - 1u;
  }
  __syncthreads();
  if (!s_last) return;

  // The last block: expected digests, then the verdict.
  const uint64_t seq = s_seq;
  if (t == 0) c->arrive_b = 0u;
  if (t < nsrv) {
    uint32_t d = 0u;
    const bool got =
        hx_wait_dig(c, hx_reply_dig(c, t, slot), seq, (uint32_t)t, &d);
    s_exp[t] = d;
    s_got[t] = __hip_atomic_exchange(&c->dig_acc[t], 0u, __ATOMIC_RELAXED,
                                     __HIP_MEMORY_SCOPE_AGENT);
    if (!got)
      atomicOr(&s_badmask, 0x100u);
    else if (d != s_got[t])
      atomicOr(&s_badmask, 1u << t);
  }
  __syncthreads();
  const uint32_t first = s_badmask;
  if (first == 0u) {
    if (t == 0) hx_dig_count(c, HX_DS_OK, 1ull);
    return;
  }
  if (first & 0x100u) {  // the digest word never came: abort is claimed
    hx_fill_nan(out, nvec, t, NT);
    return;
  }
  const int first_srv = __ffs((int)first) - 1;
  if (t == 0) hx_dig_count(c, HX_DS_BAD, 1ull);
  int attempts = 0;
  for (; attempts < HX_DIG_RETRIES && s_badmask != 0u; attempts++) {
    if (t == 0) hx_dig_delay(c, attempts);
    if (t < HX_MAX_SERVERS) s_re[t] = 0u;
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    for (int s = 0; s < nsrv; s++) {
      if (((s_badmask >> s) & 1u) == 0u) continue;
      const uint4* ys = reinterpret_cast<const uint4*>(hx_reply_y(c, s, slot));
      uint32_t d = 0u;
      for (long long v = t; v < nvec; v += NT)
        d ^= hx_dig_vec((uint32_t)v, hx_ld_reply<RD>(ys + v));
      if (d != 0u) atomicXor(&s_re[s], d);
    }
    __syncthreads();
    if (t == 0) {
      hx_dig_count(c, HX_DS_RESCANS, 1ull);
      for (int s = 0; s < nsrv; s++)
        if (((s_badmask >> s) & 1u) != 0u && s_re[s] == s_exp[s])
          s_badmask &= ~(1u << s);
    }
    __syncthreads();
  }
  if (t == 0) {
    c->dig_stat[HX_DS_LAST_A] = (c->status[HX_ST_EPOCH] << 32) |
                                ((uint64_t)layer << 16) |
                                ((uint64_t)first_srv << 8) | (uint64_t)attempts;
    c->dig_stat[HX_DS_LAST_B] =
        ((uint64_t)s_exp[first_srv] << 32) | (uint64_t)s_got[first_srv];
  }
  if (s_badmask != 0u) {  // never became right
    if (t == 0) {
      hx_dig_count(c, HX_DS_UNHEALED, 1ull);
      hx_claim(c, hx_record(HX_PH_DIGEST, (uint32_t)(__ffs((int)s_badmask) - 1),
                            seq, attempts, 0u));
    }
    hx_fill_nan(out, nvec, t, NT);
    return;
  }
  // The replies are right now: recompute the whole output, check that pass.
  // In place (out == y_hub) the first pass overwrote the hub's own part and
  // nothing is left to recompute from: that is a failure to heal (the host
  // wrapper never combines in place in digest mode).
  if (reinterpret_cast<const void*>(out) ==
      reinterpret_cast<const void*>(y_hub)) {
    if (t == 0) {
      hx_dig_count(c, HX_DS_UNHEALED, 1ull);
      hx_claim(c,
               hx_record(HX_PH_DIGEST, (uint32_t)first_srv, seq, attempts, 0u));
    }
    hx_fill_nan(out, nvec, t, NT);
    return;
  }
  if (t < HX_MAX_SERVERS) s_dig[t] = 0u;
  __syncthreads();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  hx_combine_sum<RD, true>(c, y_hub, out, nsrv, slot, nvec, t, NT, s_dig);
  __syncthreads();
  if (t == 0) {
    int same = 1;
    for (int s = 0; s < nsrv; s++) same &= s_dig[s] == s_exp[s];
    s_same = same;
    if (same) {
      hx_dig_count(c, HX_DS_HEALED, 1ull);
    } else {
      hx_dig_count(c, HX_DS_UNHEALED, 1ull);
      hx_claim(c,
               hx_record(HX_PH_DIGEST, (uint32_t)first_srv, seq, attempts, 0u));
    }
  }
  __syncthreads();
  if (!s_same) hx_fill_nan(out, nvec, t, NT);
}

__global__ void __launch_bounds__(HX_COMBINE_THREADS)
    hx_k_combine(HxCtx* __restrict__ c, const uint4* y_hub, uint4* out, int T,
                 int layer, int nblocks) {
  __shared__ int s_bad;
  constexpr int NT = HX_COMBINE_THREADS;
  const int t = threadIdx.x, b = blockIdx.x;
  const int nsrv = (int)c->nsrv;
  const int slot = layer % (int)c->nslots;
  if (t == 0) {
    int bad = hx_aborted(c) ? 1 : 0;
    if (!bad) {
      const uint64_t seq = hx_seq(c->status[HX_ST_EPOCH], layer);
      for (int s = 0; s < nsrv && !bad; s++) {
        const int rc =
            hx_wait_flag(c, hx_reply_flag(c, s, slot), seq, c->wait_ticks,
                         HX_PH_COMBINE, (uint32_t)s, b == 0);
        bad = rc != HX_WAIT_OK;
      }
      if (!bad) hx_settle(c);
    }
    s_bad = bad;
  }
  __syncthreads();
  const long long nvec = (long long)T * (long long)c->hidden / 8;
  const long long gid = (long long)b * NT + t;
  const long long gstride = (long long)nblocks * NT;
  if (s_bad) {
    hx_fill_nan(out, nvec, gid, gstride);
    return;
  }
  const uint32_t rd = (c->flush_mode >> HX_RD_SHIFT) & HX_RD_MASK;
  if (rd == 3u) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  if (c->flush_mode & HX_DIGEST) {
    if (rd == 1u)
      hx_combine_digest<1>(c, y_hub, out, layer, nblocks, slot, nsrv, nvec, gid,
                           gstride);
    else if (rd == 2u)
      hx_combine_digest<2>(c, y_hub, out, layer, nblocks, slot, nsrv, nvec, gid,
                           gstride);
    else
      hx_combine_digest<0>(c, y_hub, out, layer, nblocks, slot, nsrv, nvec, gid,
                           gstride);
  } else if (rd == 1u) {
    hx_combine_sum<1, false>(c, y_hub, out, nsrv, slot, nvec, gid, gstride,
                             nullptr);
  } else if (rd == 2u) {
    hx_combine_sum<2, false>(c, y_hub, out, nsrv, slot, nvec, gid, gstride,
                             nullptr);
  } else {
    hx_combine_sum<0, false>(c, y_hub, out, nsrv, slot, nvec, gid, gstride,
                             nullptr);
  }
  if (t == 0 && b == 0) c->status[HX_ST_COMBINES] += 1ull;
}

// ---------------------------------------------------------------------------
// Server: begin step
// ---------------------------------------------------------------------------
// Waits for the hub's header of the next epoch (the host launches this only
// after the hub host posted the step descriptor, so the wait is short; the
// bound is begin_wait_ticks), checks it against the descriptor the host copied
// into ctx.desc_*, and adopts the epoch. A mismatch never skips silently.
__global__ void __launch_bounds__(64) hx_k_srv_begin(HxCtx* __restrict__ c) {
  if (threadIdx.x != 0) return;
  if (hx_aborted(c)) return;
  const HxHdr* h = reinterpret_cast<const HxHdr*>(c->own_base);
  const uint64_t expected = c->last_epoch + 1ull;
  const int rc = hx_wait_flag(c, &h->seq, expected, c->begin_wait_ticks,
                              HX_PH_SRV_BEGIN, 0u, true);
  if (rc != HX_WAIT_OK) return;
  const uint64_t magic = hx_ld_rlx(&h->magic);
  const uint64_t epoch = hx_ld_rlx(&h->epoch);
  const uint64_t seq = hx_ld_rlx(&h->seq);
  const uint32_t T = hx_ld_rlx32(&h->T);
  const uint32_t bucket = hx_ld_rlx32(&h->bucket);
  const uint32_t opv = hx_ld_rlx32(reinterpret_cast<const uint32_t*>(
      reinterpret_cast<const uint8_t*>(h) + HX_HDR_OFF_OPCODE));
  const uint64_t plan = hx_ld_rlx(&h->plan_hash);
  uint32_t bad = 0u;
  if (magic != HX_MAGIC)
    bad = HX_MM_MAGIC;
  else if ((opv & 0xFFFFu) != HX_OPCODE_MOE)
    bad = HX_MM_OPCODE;
  else if ((opv >> 16) != HX_VERSION)
    bad = HX_MM_VERSION;
  else if (plan != c->plan_hash)
    bad = HX_MM_PLAN;
  else if (epoch > expected || seq != epoch)
    bad = HX_MM_SKIPPED;
  else if (epoch != c->desc_epoch)
    bad = HX_MM_EPOCH;
  else if (T != c->desc_T)
    bad = HX_MM_T;
  else if (bucket != c->desc_bucket)
    bad = HX_MM_BUCKET;
  if (bad != 0u) {
    hx_claim(c, hx_record(HX_PH_HDR, bad, expected, 0, 0u));
    return;
  }
  c->status[HX_ST_EPOCH] = epoch;
  c->last_epoch = epoch;
  c->cur_T = T;
  c->cur_bucket = bucket;
  c->cur_mode = bucket > 0 ? 0u : 1u;
  c->status[HX_ST_STEPS] += 1ull;
}

// ---------------------------------------------------------------------------
// Server: wait for the layer request, widen ids
// ---------------------------------------------------------------------------
#define HX_SRV_WAIT_THREADS 512

// One pass over a request (digest mode): hash x (rows gid, gstride) and, when
// do_ids, the pairs (t, nt). COPY: also copy x into x_ws and widen the ids and
// copy the weights into the workspaces. Returns the XOR of the hashes of the
// values read, whether or not x_ws exists (the consumer's x is x_ws).
template <bool COPY>
__device__ __forceinline__ uint32_t hx_srv_pass(HxCtx* c, int slot, int T,
                                                long long gid,
                                                long long gstride, bool do_ids,
                                                int t, int nt) {
  uint32_t dg = 0u;
  const uint4* src = reinterpret_cast<const uint4*>(hx_inbox_slot(c, 0, slot));
  uint4* dst = reinterpret_cast<uint4*>(c->x_ws);
  const long long nvec = (long long)T * (long long)c->hidden / 8;
  for (long long v = gid; v < nvec; v += gstride) {
    const uint4 val = src[v];
    dg ^= hx_dig_vec((uint32_t)v, val);
    if constexpr (COPY) {
      if (dst != nullptr) dst[v] = val;
    }
  }
  if (do_ids) {
    const int n = T * (int)c->topk;
    const int E = (int)c->num_experts;
    const uint8_t* sb = hx_inbox_slot(c, 0, slot);
    const int16_t* ids16 = reinterpret_cast<const int16_t*>(sb + c->off_ids);
    const float* w16 = reinterpret_cast<const float*>(sb + c->off_w);
    for (int i = t; i < n; i += nt) {
      const int raw = (int)ids16[i];
      const float wv = w16[i];
      dg ^=
          hx_dig_idw((uint32_t)i, (uint32_t)(uint16_t)raw, __float_as_uint(wv));
      if constexpr (COPY) {
        c->ids32[i] = (unsigned)raw >= (unsigned)E ? -1 : raw;
        if (c->w_ws != nullptr) c->w_ws[i] = wv;
      }
    }
  }
  return dg;
}

// Digest mode of the request wait (all threads of a block, after the flag was
// seen or the wait failed: ok): every block copies and hashes its share, the
// LAST block to arrive compares with the hub's digest word. A mismatch is read
// again after growing delays; when the request became right it is copied
// again (and that pass is checked); what does not heal aborts with phase
// HX_PH_DIGEST and leaves no pair to compute.
__device__ __forceinline__ void hx_srv_wait_digest(HxCtx* c, int layer, int T,
                                                   int nblocks, int slot,
                                                   int ok) {
  constexpr int NT = HX_SRV_WAIT_THREADS;
  __shared__ uint32_t s_dig, s_exp, s_got, s_re;
  __shared__ int s_last, s_bad;
  __shared__ uint64_t s_seq;
  const int t = threadIdx.x, b = blockIdx.x;
  if (t == 0) {
    s_dig = 0u;
    s_re = 0u;
    s_bad = 0;
    s_seq = hx_seq(c->status[HX_ST_EPOCH], layer);
  }
  __syncthreads();
  const int n = T * (int)c->topk;
  if (ok) {
    const uint32_t dg =
        hx_srv_pass<true>(c, slot, T, (long long)b * NT + t,
                          (long long)nblocks * NT, b == 0, t, NT);
    if (dg != 0u) atomicXor(&s_dig, dg);
  } else if (b == 0) {
    for (int i = t; i < n; i += NT) c->ids32[i] = -1;
  }
  __syncthreads();
  if (t == 0 && s_dig != 0u) atomicXor(&c->dig_acc[0], s_dig);
  __threadfence();
  __syncthreads();
  if (t == 0) {
    const unsigned ticket = __hip_atomic_fetch_add(
        &c->arrive_b, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    s_last = ticket == (unsigned)nblocks - 1u;
  }
  __syncthreads();
  if (!s_last) return;

  // The last block: the verdict.
  const uint64_t seq = s_seq;
  if (t == 0) {
    c->arrive_b = 0u;
    s_got = __hip_atomic_exchange(&c->dig_acc[0], 0u, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_AGENT);
    uint32_t d = 0u;
    int bad;
    if (!ok || hx_aborted(c))
      bad = 3;  // a failed wait claimed the abort already
    else if (!hx_wait_dig(c, hx_req_dig(c, 0, slot), seq, 0u, &d))
      bad = 3;  // the digest word never came
    else
      bad = d != s_got ? 1 : 0;
    s_exp = d;
    s_bad = bad;
  }
  __syncthreads();
  if (s_bad == 0) {
    if (t == 0) hx_dig_count(c, HX_DS_OK, 1ull);
    return;
  }
  if (s_bad == 3) {
    for (int i = t; i < n; i += NT) c->ids32[i] = -1;
    return;
  }
  if (t == 0) hx_dig_count(c, HX_DS_BAD, 1ull);
  int attempts = 0;
  for (; attempts < HX_DIG_RETRIES && s_bad == 1; attempts++) {
    if (t == 0) {
      hx_dig_delay(c, attempts);
      s_re = 0u;
    }
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    const uint32_t d = hx_srv_pass<false>(c, slot, T, t, NT, true, t, NT);
    if (d != 0u) atomicXor(&s_re, d);
    __syncthreads();
    if (t == 0) {
      hx_dig_count(c, HX_DS_RESCANS, 1ull);
      if (s_re == s_exp) s_bad = 0;
    }
    __syncthreads();
  }
  if (t == 0) {
    c->dig_stat[HX_DS_LAST_A] = (c->status[HX_ST_EPOCH] << 32) |
                                ((uint64_t)layer << 16) | (uint64_t)attempts;
    c->dig_stat[HX_DS_LAST_B] = ((uint64_t)s_exp << 32) | (uint64_t)s_got;
  }
  if (s_bad == 0) {  // right now: copy the request again, check that pass
    if (t == 0) s_re = 0u;
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    const uint32_t d = hx_srv_pass<true>(c, slot, T, t, NT, true, t, NT);
    if (d != 0u) atomicXor(&s_re, d);
    __syncthreads();
    if (t == 0) {
      if (s_re == s_exp)
        hx_dig_count(c, HX_DS_HEALED, 1ull);
      else
        s_bad = 1;
    }
    __syncthreads();
  }
  if (s_bad != 0) {  // never became right
    if (t == 0) {
      hx_dig_count(c, HX_DS_UNHEALED, 1ull);
      hx_claim(c, hx_record(HX_PH_DIGEST, 0u, seq, attempts, 0u));
    }
    for (int i = t; i < n; i += NT) c->ids32[i] = -1;
  }
}

// Every block waits for the request flag itself (local memory, no PCIe reads).
// Block 0 widens the ids and copies the weights; all blocks share the copy of
// the x rows into the cached workspace (ctx.x_ws).
__global__ void __launch_bounds__(HX_SRV_WAIT_THREADS)
    hx_k_srv_wait(HxCtx* __restrict__ c, int layer, int T, int nblocks) {
  __shared__ int s_ok;
  const int t = threadIdx.x, b = blockIdx.x;
  const int slot = layer % (int)c->nslots;
  if (t == 0) {
    int ok = 0;
    if (!hx_aborted(c)) {
      const uint64_t seq = hx_seq(c->status[HX_ST_EPOCH], layer);
      ok = hx_wait_flag(c, hx_req_flag(c, 0, slot), seq, c->wait_ticks,
                        HX_PH_SRV_WAIT, 0u, b == 0) == HX_WAIT_OK;
      if (ok) hx_settle(c);
    }
    s_ok = ok;
  }
  __syncthreads();
  const int ok = s_ok;
  if (c->flush_mode & HX_DIGEST) {
    hx_srv_wait_digest(c, layer, T, nblocks, slot, ok);
    return;
  }
  if (ok && c->x_ws != nullptr) {
    const uint4* src =
        reinterpret_cast<const uint4*>(hx_inbox_slot(c, 0, slot));
    uint4* dst = reinterpret_cast<uint4*>(c->x_ws);
    const long long nvec = (long long)T * (long long)c->hidden / 8;
    const long long gid = (long long)b * HX_SRV_WAIT_THREADS + t;
    const long long gstride = (long long)nblocks * HX_SRV_WAIT_THREADS;
    for (long long v = gid; v < nvec; v += gstride) dst[v] = src[v];
  }
  if (b != 0) return;
  const int n = T * (int)c->topk;
  const int E = (int)c->num_experts;
  const uint8_t* sb = hx_inbox_slot(c, 0, slot);
  const int16_t* ids16 = reinterpret_cast<const int16_t*>(sb + c->off_ids);
  const float* w16 = reinterpret_cast<const float*>(sb + c->off_w);
  for (int i = t; i < n; i += HX_SRV_WAIT_THREADS) {
    int id = -1;
    if (ok) {
      id = (int)ids16[i];
      if ((unsigned)id >= (unsigned)E) id = -1;
      if (c->w_ws != nullptr) c->w_ws[i] = w16[i];
    }
    c->ids32[i] = id;
  }
}

// ---------------------------------------------------------------------------
// Server: push y into the hub's reply slot, then the flag
// ---------------------------------------------------------------------------
#define HX_PUSH_THREADS 256

// y to the hub's reply slot. Returns the XOR of the hashes of the vectors this
// thread sent (DIG, else 0); the hash is of y, the source. skip_chunk is the
// fault-injection hook of test builds (that 256 B chunk is written late by
// hx_k_srv_push).
template <bool DIG>
__device__ __forceinline__ uint32_t hx_push_y(uint4* dst,
                                              const uint4* __restrict__ y,
                                              long long nvec, long long gid,
                                              long long gstride,
                                              long long skip_chunk) {
  uint32_t dg = 0u;
  for (long long v = gid; v < nvec; v += gstride) {
    const uint4 val = y[v];
    if constexpr (DIG) dg ^= hx_dig_vec((uint32_t)v, val);
#ifdef HX_FAULT_INJECT
    if ((v >> 4) == skip_chunk) {
      // The chunk holds garbage until hx_k_srv_push writes the data late.
      uint4 bad = val;
      bad.x ^= 0xA5A5A5A5u;
      dst[v] = bad;
      continue;
    }
#endif
    dst[v] = val;
  }
  (void)skip_chunk;
  return dg;
}

__global__ void __launch_bounds__(HX_PUSH_THREADS)
    hx_k_srv_push(HxCtx* __restrict__ c, const uint4* __restrict__ y, int T,
                  int layer, int nblocks) {
  __shared__ int s_abort;
  __shared__ uint64_t s_seq;
  __shared__ uint32_t s_dig;
  constexpr int NT = HX_PUSH_THREADS;
  const int t = threadIdx.x, b = blockIdx.x;
  const int slot = layer % (int)c->nslots;
  const int s = (int)c->index;
  if (t == 0) {
    s_abort = hx_aborted(c) ? 1 : 0;
    s_seq = hx_seq(c->status[HX_ST_EPOCH], layer);
    s_dig = 0u;
  }
  __syncthreads();
  if (s_abort) return;
  uint4* dst = reinterpret_cast<uint4*>(hx_reply_y(c, s, slot));
  const long long nvec = (long long)T * (long long)c->hidden / 8;
  const long long gid = (long long)b * NT + t;
  const long long gstride = (long long)nblocks * NT;
  const bool dig = (c->flush_mode & HX_DIGEST) != 0u;
  long long skip_chunk = -1;
#ifdef HX_FAULT_INJECT
  const bool inj = c->dbg[0] != 0ull && (uint32_t)c->dbg[0] == (uint32_t)s_seq;
  if (inj) skip_chunk = (long long)c->dbg[1];
#endif
  const uint32_t dgy =
      dig ? hx_push_y<true>(dst, y, nvec, gid, gstride, skip_chunk)
          : hx_push_y<false>(dst, y, nvec, gid, gstride, skip_chunk);
  if (c->flush_mode & HX_FLUSH_PAYLOAD) {
    uint32_t sink = 0u;
    for (long long v = gid; v < nvec; v += gstride)
      if ((v & 15) == 15 || v == nvec - 1) sink += hx_readback(dst + v);
    hx_sink(sink);
  }
  if (dig) {
    if (dgy != 0u) atomicXor(&s_dig, dgy);
    __syncthreads();
    if (t == 0 && s_dig != 0u) atomicXor(&c->dig_acc[0], s_dig);
  }
  __threadfence_system();
  __syncthreads();
  if (t == 0) {
    const unsigned ticket = __hip_atomic_fetch_add(
        &c->arrive_a, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    if (ticket == (unsigned)nblocks - 1u) {
      c->arrive_a = 0u;
      if (dig)
        hx_st_rlx(hx_reply_dig(c, s, slot),
                  hx_dig_word(s_seq, __hip_atomic_exchange(
                                         &c->dig_acc[0], 0u, __ATOMIC_RELAXED,
                                         __HIP_MEMORY_SCOPE_AGENT)));
      if (c->flush_mode & HX_FLUSH_CTRL) hx_flush_peer(hx_reply_base(c));
      hx_publish(c, hx_reply_flag(c, s, slot), s_seq);
#ifdef HX_FAULT_INJECT
      if (inj) {
        // The skipped chunk reaches the hub dbg[2] us after the flag.
        hx_spin_us(c, (int64_t)c->dbg[2]);
        for (int i = 0; i < 16; i++) {
          const long long v = skip_chunk * 16 + i;
          if (v < nvec) dst[v] = y[v];
        }
        __threadfence_system();
      }
#endif
    }
  }
}

// ---------------------------------------------------------------------------
// Host launch helpers (grid sizes from payload bytes; measured in T5)
// ---------------------------------------------------------------------------
static inline int hx_push_blocks(uint64_t bytes) {
  if (bytes <= (32u << 10)) return 4;
  if (bytes <= (128u << 10)) return 8;
  if (bytes <= (512u << 10)) return 16;
  if (bytes <= (2u << 20)) return 32;
  return 64;
}
static inline int hx_combine_blocks(uint64_t out_bytes) {
  int nb = (int)((out_bytes / 16 + 1023) / 1024);
  if (nb < 1) nb = 1;
  return nb > 32 ? 32 : nb;
}

// The helpers only enqueue; the caller checks the launch (hipGetLastError, or
// C10_CUDA_KERNEL_LAUNCH_CHECK in the torch ops).
static inline void hx_launch_begin_step(HxCtx* c, int T, int bucket,
                                        hipStream_t st) {
  hipLaunchKernelGGL(hx_k_begin_step, dim3(1), dim3(64), 0, st, c, T, bucket);
}

// blocks <= 0: by size.
static inline void hx_launch_dispatch(HxCtx* c, const void* x,
                                      const int32_t* ids, const float* w,
                                      int32_t* hub_out, int T, int layer,
                                      int nsrv, int hidden, int blocks,
                                      hipStream_t st) {
  const int nb =
      blocks > 0 ? blocks
                 : hx_push_blocks((uint64_t)T * hidden * 2u * (uint64_t)nsrv);
  hipLaunchKernelGGL(hx_k_dispatch, dim3(nb), dim3(HX_DISPATCH_THREADS), 0, st,
                     c, reinterpret_cast<const uint4*>(x), ids, w, hub_out, T,
                     layer, nb);
}

static inline void hx_launch_combine(HxCtx* c, const void* y_hub, void* out,
                                     int T, int layer, int hidden, int blocks,
                                     hipStream_t st) {
  const int nb =
      blocks > 0 ? blocks : hx_combine_blocks((uint64_t)T * hidden * 2u);
  hipLaunchKernelGGL(hx_k_combine, dim3(nb), dim3(HX_COMBINE_THREADS), 0, st, c,
                     reinterpret_cast<const uint4*>(y_hub),
                     reinterpret_cast<uint4*>(out), T, layer, nb);
}

static inline void hx_launch_srv_begin(HxCtx* c, hipStream_t st) {
  hipLaunchKernelGGL(hx_k_srv_begin, dim3(1), dim3(64), 0, st, c);
}

static inline int hx_srv_wait_blocks(uint64_t x_bytes) {
  if (x_bytes <= (64u << 10)) return 1;
  if (x_bytes <= (512u << 10)) return 4;
  return 16;
}

// blocks <= 0: by size.
static inline void hx_launch_srv_wait(HxCtx* c, int layer, int T, int hidden,
                                      int blocks, hipStream_t st) {
  const int nb =
      blocks > 0 ? blocks : hx_srv_wait_blocks((uint64_t)T * hidden * 2u);
  hipLaunchKernelGGL(hx_k_srv_wait, dim3(nb), dim3(HX_SRV_WAIT_THREADS), 0, st,
                     c, layer, T, nb);
}

static inline void hx_launch_srv_push(HxCtx* c, const void* y, int T, int layer,
                                      int hidden, int blocks, hipStream_t st) {
  const int nb =
      blocks > 0 ? blocks : hx_push_blocks((uint64_t)T * hidden * 2u);
  hipLaunchKernelGGL(hx_k_srv_push, dim3(nb), dim3(HX_PUSH_THREADS), 0, st, c,
                     reinterpret_cast<const uint4*>(y), T, layer, nb);
}
