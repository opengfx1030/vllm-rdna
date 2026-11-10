// hx_loopback: standalone, torch-free loopback test of the hx push protocol.
//
// One hub process and --nsrv server processes run the REAL kernels of
// csrc/rocm/hetero_moe/hx_core.cuh across processes over PCIe: IPC-exported
// uncached VRAM, posted stores, sequence-numbered flags, device-side timeouts,
// sticky abort. The hub generates random MoE inputs and runs per layer
// dispatch -> synthetic compute -> combine, then compares the result byte for
// byte with a CPU model of the whole chain. A server runs per layer
// srv_wait -> synthetic compute -> srv_push. The processes meet in a
// rendezvous directory (--dir, under /dev/shm) like
// vllm/distributed/hetero_moe/rdv.py does it (hub<h>.json, srv<s>.json, hex IPC
// handles, atomic publish, pidfd peer watcher poking HX_CTRL_HOST_ABORT).
//
// Build (run_loopback.sh does it under the build lock):
//   hipcc --offload-arch=gfx1030 --offload-arch=gfx1100 -O3 -std=c++17 -x hip
//     -I csrc/rocm/hetero_moe -Wall -Wextra -o hx_loopback hx_loopback.cpp
//     -pthread
//
// Roles (pin every process with HIP_VISIBLE_DEVICES on its own command line):
//   hx_loopback hub --dir D --nsrv N [options]
//   hx_loopback srv --dir D --nsrv N --index S [options]
//   hx_loopback buses       HIP ordinal -> PCI bus id of the visible devices
//   hx_loopback selftest    CPU-only checks of the reference (no HIP call)
// --help lists the options and exits before any HIP call.
//
// Exit codes: 0 ok, 2 mismatch, 3 abort or peer death (clean), 4 setup error,
// rendezvous timeout or HIP error, 9 injected death, 128+N killed by signal N.
// Every hub/srv process ends with one line
//   HXLB <role> <index> iters=<n> mismatches=<m> abort=<0|0xrecord> exit=<c>
//   reason=<token> ... (key=value pairs without blanks)
// --bench adds HXBENCH lines (latency percentiles per T).
//
// The synthetic compute (identical on hub and servers) is exact dyadic
// arithmetic, so no FMA or summation-order ambiguity exists:
//   y[r][c] = fp16(sum_{k, id[r][k] >= 0} float(x[r][c]) * w[r][k] *
//                  float((id[r][k] & 7) + 1)),  k ascending, fp32.
// The reference (section "reference model") restates the dispatch rule of the
// contract and the combine order independently of the kernels.
#include <hip/hip_runtime.h>

#include "hx_core.cuh"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <poll.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <map>
#include <string>
#include <thread>
#include <utility>
#include <vector>

static_assert(offsetof(HxCtx, desc_T) == offsetof(HxCtx, desc_epoch) + 8 &&
                  offsetof(HxCtx, desc_bucket) ==
                      offsetof(HxCtx, desc_epoch) + 12,
              "desc_* must be contiguous: the descriptor is one 16 B copy");

static const uint64_t kDefaultPlanHash = 0x4C4F4F5042414B31ull;
static const int kMaxGraphT = 48;
static const int kDescRing = 64;
static const double kFinalWaitS = 60.0;
static const int kDvRecs = 256, kDvAbRecs = 32;

enum { kExitOk = 0, kExitMismatch = 2, kExitAbort = 3, kExitSetup = 4 };

// ---------------------------------------------------------------------------
// process state, logging, exit
// ---------------------------------------------------------------------------
struct Global {
  const char* role = "init";
  int index = 0;
  int64_t t0_ns = 0;
  std::atomic<uint64_t> iters{0}, mismatches{0}, steps{0}, abort_rec{0};
  std::atomic<int> finishing{0};
  std::atomic<int> peer_dead{-1};
  std::atomic<int64_t> t_detect_ns{0}, t_poke_ns{0};
  bool clean_dir = false;
  bool watcher_on = false;
  char arch[96] = "";
  char alloc[24] = "";
  long long tpm = 0;
};
static Global G;

static int64_t now_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000000ll + ts.tv_nsec;
}
static void sleep_us(long us) {
  timespec ts;
  ts.tv_sec = us / 1000000;
  ts.tv_nsec = (us % 1000000) * 1000;
  nanosleep(&ts, nullptr);
}
static void wr_all(int fd, const char* s, size_t n) {
  while (n > 0) {
    const ssize_t k = write(fd, s, n);
    if (k < 0) {
      if (errno == EINTR) continue;
      return;
    }
    s += k;
    n -= (size_t)k;
  }
}

static void say(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
static void say(const char* fmt, ...) {
  char buf[4096];
  int n = snprintf(buf, sizeof buf, "[%s%d +%7.3fs] ", G.role, G.index,
                   (double)(now_ns() - G.t0_ns) / 1e9);
  if (n < 0) n = 0;
  va_list ap;
  va_start(ap, fmt);
  int m = vsnprintf(buf + n, sizeof buf - (size_t)n - 2, fmt, ap);
  va_end(ap);
  if (m < 0) m = 0;
  size_t len = (size_t)n + (size_t)m;
  if (len > sizeof buf - 2) len = sizeof buf - 2;
  buf[len++] = '\n';
  wr_all(1, buf, len);
}

[[noreturn]] static void finish(int code, const char* reason);
[[noreturn]] static void die_setup(const char* fmt, ...)
    __attribute__((format(printf, 1, 2)));
[[noreturn]] static void die_setup(const char* fmt, ...) {
  char buf[2048];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  say("ERROR: %s", buf);
  finish(kExitSetup, "setup");
}

// ---------------------------------------------------------------------------
// reference model (host only): fp16 codec, RNG, tables, dispatch rule,
// synthetic compute, combine. Nothing here touches HIP.
// ---------------------------------------------------------------------------
// REF-BEGIN
static inline uint32_t f32_bits(float f) {
  uint32_t u;
  memcpy(&u, &f, sizeof u);
  return u;
}
static inline float bits_f32(uint32_t u) {
  float f;
  memcpy(&f, &u, sizeof f);
  return f;
}

// fp32 -> fp16, round to nearest even, bit operations only.
static uint16_t f2h(float f) {
  const uint32_t u = f32_bits(f);
  const uint32_t sign = (u >> 16) & 0x8000u;
  const uint32_t ex = (u >> 23) & 0xFFu;
  const uint32_t man = u & 0x7FFFFFu;
  if (ex == 0xFFu)
    return (uint16_t)(sign | 0x7C00u | (man ? (0x200u | (man >> 13)) : 0u));
  const int e = (int)ex - 127 + 15;
  if (e >= 31) return (uint16_t)(sign | 0x7C00u);
  if (e <= 0) {
    if (e < -10) return (uint16_t)sign;
    const uint32_t m = man | 0x800000u;
    const int sh = 14 - e;
    uint32_t h = m >> sh;
    const uint32_t rem = m & ((1u << sh) - 1u), half = 1u << (sh - 1);
    if (rem > half || (rem == half && (h & 1u))) h++;
    return (uint16_t)(sign | h);
  }
  uint32_t h = ((uint32_t)e << 10) | (man >> 13);
  const uint32_t rem = man & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h++;
  return (uint16_t)(sign | h);
}

static float h2f(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  const uint32_t ex = (h >> 10) & 0x1Fu;
  uint32_t man = h & 0x3FFu;
  if (ex == 0) {
    if (man == 0) return bits_f32(sign);
    int e = 0;
    while (!(man & 0x400u)) {
      man <<= 1;
      e++;
    }
    man &= 0x3FFu;
    return bits_f32(sign | ((uint32_t)(113 - e) << 23) | (man << 13));
  }
  if (ex == 31) return bits_f32(sign | 0x7F800000u | (man << 13));
  return bits_f32(sign | ((ex + 112u) << 23) | (man << 13));
}

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed) {}
  uint64_t next() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  uint32_t below(uint32_t n) {
    return (uint32_t)(((next() >> 32) * (uint64_t)n) >> 32);
  }
};

// x values are q/8 for q in [-32, 31] (exact in fp16); fp16 bits per q+32.
static uint16_t g_x16_of_q[64];
static void init_x16_table() {
  for (int q = 0; q < 64; q++) g_x16_of_q[q] = f2h((float)(q - 32) * 0.125f);
}

// One layer of inputs, deterministic in (seed, step, layer): x fp16 (and its
// q = 8x in xq), ids (distinct experts per row, ~2% -1, ~1% >= E), w = k/16.
static void gen_inputs(uint64_t seed, uint64_t step, int layer, int T, int H,
                       int topk, int E, uint16_t* x16, int8_t* xq, int32_t* ids,
                       float* w) {
  Rng rng(seed ^ (step * 0x9E3779B97F4A7C15ull) ^
          ((uint64_t)(layer + 1) * 0xD1B54A32D192ED03ull));
  rng.s = rng.next();
  const size_t nx = (size_t)T * H;
  for (size_t i = 0; i < nx;) {
    uint64_t v = rng.next();
    for (int j = 0; j < 10 && i < nx; j++, i++) {
      const unsigned q = (unsigned)(v & 63u);
      v >>= 6;
      xq[i] = (int8_t)((int)q - 32);
      x16[i] = g_x16_of_q[q];
    }
  }
  for (int r = 0; r < T; r++) {
    int pick[HX_MAX_TOPK];
    for (int k = 0; k < topk; k++) {
      for (;;) {
        const int e = (int)rng.below((uint32_t)E);
        bool dup = false;
        for (int j = 0; j < k; j++) dup |= pick[j] == e;
        if (!dup) {
          pick[k] = e;
          break;
        }
      }
      const uint32_t u = rng.below(100);
      int id = pick[k];
      if (u < 2)
        id = -1;
      else if (u == 2)
        id = E + (int)rng.below(1024);
      ids[(size_t)r * topk + k] = id;
      w[(size_t)r * topk + k] = (float)rng.below(17) / 16.0f;
    }
  }
}

struct Tables {
  int E = 0, L = 0, nsrv = 0, rows_cap = 0;
  std::vector<int16_t> hub_slot, ktab;
  std::vector<int8_t> cold_srv, n_cold;
  std::vector<uint8_t> owned;
};

// kmode: 0 lin (K = min(512, 2 + T)), 1 fixed (kfixed), 2 zero.
static void build_tables(Tables& t, int E, int L, int nsrv, int rows_cap,
                         int kmode, int kfixed) {
  t.E = E;
  t.L = L;
  t.nsrv = nsrv;
  t.rows_cap = rows_cap;
  const size_t n = (size_t)L * E;
  t.hub_slot.assign(n, -1);
  t.n_cold.assign(n, 0);
  t.owned.assign(n, 1);
  t.cold_srv.assign(n * HX_MAX_COLD, -1);
  for (int l = 0; l < L; l++) {
    int next = 0;
    for (int e = 0; e < E; e++) {
      const size_t i = (size_t)l * E + (size_t)e;
      if ((e + l) % 8 < 3) t.hub_slot[i] = (int16_t)next++;
      const int nc = std::min(1 + (e * 7 + l) % 3, nsrv);
      t.n_cold[i] = (int8_t)nc;
      for (int c = 0; c < nc; c++)
        t.cold_srv[i * HX_MAX_COLD + (size_t)c] = (int8_t)((e + c + l) % nsrv);
      if (e % 61 == 0 && (l & 1)) t.owned[i] = 0;
    }
  }
  t.ktab.assign((size_t)rows_cap + 1, 0);
  for (int T = 0; T <= rows_cap; T++)
    t.ktab[(size_t)T] = (int16_t)(kmode == 0   ? std::min(512, 2 + T)
                                  : kmode == 1 ? kfixed
                                               : 0);
}

// Where each pair of one layer goes: dest -1 dropped/invalid, 0 hub, 1 + s
// server s. hub_ids is hub_ids_out of the contract; n_cold counts remote pairs
// of experts without a hub slot, n_spill remote pairs of resident experts
// beyond K, n_lost plan errors (the layer_stats columns).
struct Plan {
  std::vector<int32_t> hub_ids;
  std::vector<int8_t> dest;
  uint64_t n_hub = 0, n_remote = 0, n_cold = 0, n_spill = 0, n_lost = 0;
  bool plan_error = false;
};

// The dispatch rule of the contract (hx-protocol.md, "Hub dispatch
// semantics"): valid = 0 <= id < E and owned; resident = valid and slot >= 0;
// touched = distinct resident ids; rank = touched ids with a smaller slot;
// keep iff resident and rank < K[T]; else remote to copy (row + layer) %
// n_cold of the id's cold list.
static void ref_plan(const Tables& t, int layer, int T, int topk,
                     const int32_t* ids, Plan& p) {
  const int E = t.E;
  const size_t base = (size_t)layer * E;
  const size_t np = (size_t)T * topk;
  auto valid = [&](int e) {
    return e >= 0 && e < E && t.owned[base + (size_t)e] == 1;
  };
  std::vector<int> rank((size_t)E + 1, 0);
  std::vector<uint8_t> touched((size_t)E, 0);
  for (size_t i = 0; i < np; i++) {
    const int e = ids[i];
    if (!valid(e)) continue;
    const int s = t.hub_slot[base + (size_t)e];
    if (s >= 0) touched[(size_t)s] = 1;
  }
  for (int s = 0; s < E; s++)
    rank[(size_t)s + 1] = rank[(size_t)s] + touched[(size_t)s];
  const int kmax = t.ktab[(size_t)std::min(T, t.rows_cap)];
  p.hub_ids.assign(np, -1);
  p.dest.assign(np, -1);
  p.n_hub = p.n_remote = p.n_cold = p.n_spill = p.n_lost = 0;
  p.plan_error = false;
  for (size_t i = 0; i < np; i++) {
    const int e = ids[i];
    if (!valid(e)) continue;
    const int row = (int)(i / (size_t)topk);
    const int s = t.hub_slot[base + (size_t)e];
    if (s >= 0 && rank[(size_t)s] < kmax) {
      p.hub_ids[i] = e;
      p.dest[i] = 0;
      p.n_hub++;
      continue;
    }
    const int nc = t.n_cold[base + (size_t)e];
    if (nc <= 0) {
      p.plan_error = true;
      p.n_lost++;
      continue;
    }
    const int srv = t.cold_srv[(base + (size_t)e) * HX_MAX_COLD +
                               (size_t)((row + layer) % nc)];
    if (srv < 0 || srv >= t.nsrv) {
      p.plan_error = true;
      p.n_lost++;
      continue;
    }
    p.dest[i] = (int8_t)(1 + srv);
    p.n_remote++;
    (s >= 0 ? p.n_spill : p.n_cold)++;
  }
}

// The synthetic compute is a function of (x value, row's ids and w), and x
// takes 64 values only: evaluate the formula of the header comment (per
// destination, fp32, k ascending, hand-written fp16 rounding) and the combine
// (fp32 left to right, hub first) once per (row, q); tabs[d][row][q] holds
// the fp16 partial of destination d, otab[row][q] the combined output.
struct RefTabs {
  int T = 0, nd = 0;
  std::vector<uint16_t> ytab, otab;
};

static void ref_tabs(const Plan& p, const int32_t* ids, const float* w, int T,
                     int topk, int nsrv, RefTabs& r) {
  const int nd = nsrv + 1;
  r.T = T;
  r.nd = nd;
  r.ytab.assign((size_t)nd * T * 64, 0);
  r.otab.assign((size_t)T * 64, 0);
  for (int row = 0; row < T; row++) {
    for (int q = 0; q < 64; q++) {
      const float xv = (float)(q - 32) * 0.125f;
      float acc[HX_MAX_SERVERS + 1];
      for (int d = 0; d < nd; d++) acc[d] = 0.0f;
      for (int k = 0; k < topk; k++) {
        const size_t i = (size_t)row * topk + (size_t)k;
        const int d = p.dest[i];
        if (d < 0) continue;
        const float sc = (float)((ids[i] & 7) + 1);
        acc[d] = acc[d] + (xv * w[i]) * sc;
      }
      uint16_t yb[HX_MAX_SERVERS + 1];
      for (int d = 0; d < nd; d++) {
        yb[d] = f2h(acc[d]);
        r.ytab[((size_t)d * T + (size_t)row) * 64 + (size_t)q] = yb[d];
      }
      float f = h2f(yb[0]);
      for (int d = 1; d < nd; d++) f = f + h2f(yb[d]);
      r.otab[(size_t)row * 64 + (size_t)q] = f2h(f);
    }
  }
}

static void ref_fill(const uint16_t* tab, const int8_t* xq, int T, int H,
                     uint16_t* out) {
  for (int r = 0; r < T; r++) {
    const uint16_t* t = tab + (size_t)r * 64 + 32;
    const int8_t* x = xq + (size_t)r * H;
    uint16_t* o = out + (size_t)r * H;
    for (int c = 0; c < H; c++) o[c] = t[x[c]];
  }
}

// The same chain column by column from the fp16 bits of x (slow path of
// --ref-slow and of the self-test).
static void ref_out_slow(const Plan& p, const int32_t* ids, const float* w,
                         const uint16_t* x16, int T, int H, int topk, int nsrv,
                         uint16_t* out) {
  const int nd = nsrv + 1;
  for (int r = 0; r < T; r++) {
    for (int c = 0; c < H; c++) {
      const float xv = h2f(x16[(size_t)r * H + (size_t)c]);
      float acc[HX_MAX_SERVERS + 1];
      for (int d = 0; d < nd; d++) acc[d] = 0.0f;
      for (int k = 0; k < topk; k++) {
        const size_t i = (size_t)r * topk + (size_t)k;
        const int d = p.dest[i];
        if (d < 0) continue;
        acc[d] = acc[d] + (xv * w[i]) * (float)((ids[i] & 7) + 1);
      }
      float f = h2f(f2h(acc[0]));
      for (int d = 1; d < nd; d++) f = f + h2f(f2h(acc[d]));
      out[(size_t)r * H + (size_t)c] = f2h(f);
    }
  }
}

static size_t count_diff(const uint16_t* a, const uint16_t* b, size_t n,
                         size_t* first) {
  size_t bad = 0;
  *first = (size_t)-1;
  for (size_t i = 0; i < n; i++)
    if (a[i] != b[i] && bad++ == 0) *first = i;
  return bad;
}
static size_t count_diff32(const int32_t* a, const int32_t* b, size_t n,
                           size_t* first) {
  size_t bad = 0;
  *first = (size_t)-1;
  for (size_t i = 0; i < n; i++)
    if (a[i] != b[i] && bad++ == 0) *first = i;
  return bad;
}
// REF-END

// CPU-only checks of the reference; returns the number of failures.
static int selftest_cpu(bool verbose) {
  int bad = 0;
  init_x16_table();
  for (uint32_t h = 0; h < 65536u; h++) {
    const bool nan = (h & 0x7C00u) == 0x7C00u && (h & 0x3FFu) != 0u;
    const float f = h2f((uint16_t)h);
    if (nan ? (f == f) : (f2h(f) != (uint16_t)h)) bad++;
  }
  if (verbose) say("selftest: fp16 round trip of all patterns: %d bad", bad);
  Rng rng(1234);
  int bad_rne = 0;
  for (int i = 0; i < 1000000; i++) {
    const float f = bits_f32((uint32_t)rng.next());
    if (!(fabsf(f) < 1e9f)) continue;
    const double a = fabs((double)f);
    const uint16_t h = f2h(fabsf(f));
    if (a >= 65520.0) {
      bad_rne += h != 0x7C00u;
      continue;
    }
    const double dh = fabs((double)h2f(h) - a);
    const double dl = h > 0 ? fabs((double)h2f((uint16_t)(h - 1)) - a) : 1e30;
    const double du =
        h < 0x7BFFu ? fabs((double)h2f((uint16_t)(h + 1)) - a) : 1e30;
    if (dh > dl || dh > du || (dh == dl && (h & 1u)) || (dh == du && (h & 1u)))
      bad_rne++;
  }
  bad += bad_rne;
  if (verbose) say("selftest: fp32 -> fp16 nearest-even: %d bad", bad_rne);
  // fast reference == per-column evaluation, on a random plan
  int bad_ref = 0;
  Tables t;
  const int E = 64, L = 4, ns = 3, T = 7, H = 40, K = 6;
  build_tables(t, E, L, ns, T, 0, 0);
  std::vector<uint16_t> x16((size_t)T * H), o1((size_t)T * H),
      o2((size_t)T * H);
  std::vector<int8_t> xq((size_t)T * H);
  std::vector<int32_t> ids((size_t)T * K);
  std::vector<float> w((size_t)T * K);
  Plan p;
  RefTabs rt;
  for (int it = 0; it < 200; it++) {
    const int layer = it % L;
    gen_inputs(77, (uint64_t)it, layer, T, H, K, E, x16.data(), xq.data(),
               ids.data(), w.data());
    ref_plan(t, layer, T, K, ids.data(), p);
    bad_ref += p.plan_error;
    ref_tabs(p, ids.data(), w.data(), T, K, ns, rt);
    ref_fill(rt.otab.data(), xq.data(), T, H, o1.data());
    ref_out_slow(p, ids.data(), w.data(), x16.data(), T, H, K, ns, o2.data());
    size_t first;
    bad_ref += count_diff(o1.data(), o2.data(), o1.size(), &first) != 0;
  }
  bad += bad_ref;
  if (verbose) say("selftest: table reference vs per-column: %d bad", bad_ref);
  return bad;
}

// ---------------------------------------------------------------------------
// text helpers: hex, JSON (flat objects), pid liveness
// ---------------------------------------------------------------------------
static std::string hex_encode(const void* p, size_t n) {
  static const char* d = "0123456789abcdef";
  std::string s(2 * n, '0');
  const unsigned char* b = (const unsigned char*)p;
  for (size_t i = 0; i < n; i++) {
    s[2 * i] = d[b[i] >> 4];
    s[2 * i + 1] = d[b[i] & 15];
  }
  return s;
}
static bool hex_decode(const std::string& s, void* out, size_t n) {
  if (s.size() != 2 * n) return false;
  unsigned char* b = (unsigned char*)out;
  for (size_t i = 0; i < n; i++) {
    int v[2];
    for (int k = 0; k < 2; k++) {
      const char ch = s[2 * i + (size_t)k];
      if (ch >= '0' && ch <= '9')
        v[k] = ch - '0';
      else if (ch >= 'a' && ch <= 'f')
        v[k] = ch - 'a' + 10;
      else
        return false;
    }
    b[i] = (unsigned char)(v[0] << 4 | v[1]);
  }
  return true;
}

struct Json {
  std::string s = "{";
  Json& kv(const char* k, long long v) {
    s += (s.size() > 1 ? ", \"" : "\"") + std::string(k) +
         "\": " + std::to_string(v);
    return *this;
  }
  Json& kv(const char* k, const std::string& v) {
    s += (s.size() > 1 ? ", \"" : "\"") + std::string(k) + "\": \"" + v + "\"";
    return *this;
  }
  std::string str() const { return s + "}\n"; }
};
static bool js_find(const std::string& t, const char* key, std::string* val) {
  const std::string pat = std::string("\"") + key + "\"";
  size_t p = t.find(pat);
  if (p == std::string::npos) return false;
  p = t.find(':', p + pat.size());
  if (p == std::string::npos) return false;
  p++;
  while (p < t.size() && t[p] == ' ') p++;
  if (p < t.size() && t[p] == '"') {
    const size_t q = t.find('"', p + 1);
    if (q == std::string::npos) return false;
    *val = t.substr(p + 1, q - p - 1);
    return true;
  }
  size_t q = p;
  while (q < t.size() && t[q] != ',' && t[q] != '}' && t[q] != '\n') q++;
  *val = t.substr(p, q - p);
  return true;
}
static long long js_int(const std::string& t, const char* key, long long def) {
  std::string v;
  return js_find(t, key, &v) ? strtoll(v.c_str(), nullptr, 0) : def;
}
static std::string js_str(const std::string& t, const char* key) {
  std::string v;
  return js_find(t, key, &v) ? v : std::string();
}

// True if pid exists and is not a zombie (a dead child nobody waited for).
static bool pid_alive(long pid) {
  if (pid <= 0) return false;
  if (kill((pid_t)pid, 0) != 0 && errno != EPERM) return false;
  if (access("/proc/self/stat", F_OK) != 0) return true;
  char path[64];
  snprintf(path, sizeof path, "/proc/%ld/stat", pid);
  FILE* f = fopen(path, "r");
  if (!f) return false;
  char buf[512];
  const size_t n = fread(buf, 1, sizeof buf - 1, f);
  fclose(f);
  buf[n] = 0;
  const char* rp = strrchr(buf, ')');
  if (!rp || rp[1] != ' ') return true;
  return rp[2] != 'Z' && rp[2] != 'X';
}

// ---------------------------------------------------------------------------
// options
// ---------------------------------------------------------------------------
struct Opt {
  std::string role, dir, alloc = "uncached";
  int index = 0, hub_index = 0, device = 0, nsrv = 1;
  int experts = 512, layers = 48, rows_cap = 2048, hidden = 2560, topk = 10;
  int nslots = 2, flush = 0, rd_mode = 0, kmode = 0, kfixed = 0;
  int verify_every = -1;
  int warmup_steps = 5, pipeline = 3, layer_stats = -1, srv_ws = 1;
  int blk_dispatch = 0, blk_combine = 0, blk_push = 0, blk_srv_wait = 0;
  uint64_t plan_hash = kDefaultPlanHash, seed = 1;
  long long wait_ms = 2000, begin_wait_ms = 600000, iters = 1000000;
  long long die_after = -1, hang_after = -1, hub_die_after = -1;
  long long log_every = 100000;
  double srv_work_us = 0, rdv_timeout_s = 120;
  std::vector<int> tlist = {3, 1, 6, 24, 48};
  bool eager = false, graph = false, bench = false, no_watcher = false;
  bool stop_on_mismatch = true, ref_slow = false;
  int check_every = 512, dv_lines = 20, blk_check = 0, dv_reread_us = 20;
  int inject_srv = 0, inject_chunk = 1, inject_rd_kind = 1;
  long long inject_stale = -1, inject_out = -1, inject_rd = -1;
  bool digest = false, inject_late_req = false;
  long long inject_late = -1;
  int inject_late_us = 100;
  bool dev_verify = false, tlist_set = false, layers_set = false;
};
static Opt O;

static const char* kUsage =
    R"(usage: hx_loopback hub|srv|buses|selftest [options]

  --dir D              rendezvous directory (hub, srv; under /dev/shm, not /tmp)
  --nsrv N             servers of the hub, 1..8 (default 1)
  --index S            server index (srv)
  --hub-index H        hub index (default 0)
  --device D           in-process HIP ordinal of this role's GPU (default 0)
  --alloc KIND         exported region: uncached (default, no fallback: a
                       failing allocation is an error) | coarse | finegrained
  --flush N            ctx.flush_mode bits 0..3 (0..15, default 0): 1 control
                       word read-back before a flag, 2 payload read-back
                       before the flag, 4 flag published by a returning
                       system-scope atomic exchange, 8 consumer settles about
                       4 us after the flags
  --rd-mode M          consumer payload read mode (ctx.flush_mode bits 4-7):
                       0 plain, 1 nontemporal, 2 relaxed system atomic 2x8B,
                       3 plain after a per-wave acquire fence (default 0);
                       4 (dev-verify only): loopback-local combine that reads
                       every vector twice back to back, plain then relaxed
                       system atomics, sums the plain read, classifies both
                       reads against the pattern (DEVAB); an exchange whose
                       two reads differ counts as a mismatch
  --wait-ms MS         device timeout of one wait (default 2000)
  --begin-wait-ms MS   server wait for the next step (default 600000)
  --plan-hash HEX      equal on every role
  --experts E --layers L --rows-cap R --hidden H --topk K --nslots N
                       geometry (512, 48, 2048, 2560, 10, 2)
  --iters N            layer exchanges in total (default 1e6)
  --T-list a,b,c       T of step i is the (i mod n)-th entry (3,1,6,24,48)
  --seed S             input generator seed
  --k-mode M           fixed:K | lin (K = min(512, 2 + T), default) | zero
  --eager              bucket 0 steps
  --graph              hipGraph per T (T <= 48), replayed per step
  --verify-every K     full CPU check of every K-th exchange (default 1 for
                       T <= 48, else 16; 0 = never); hub_ids always checked
  --stop-on-mismatch [0|1]  exit 2 at the first mismatch (default 1)
  --no-stop-on-mismatch     = --stop-on-mismatch 0: run to the end, then exit 2
  --ref-slow           per-column reference instead of the table reference
  --bench              latency mode: no verification, no host work in the loop
  --warmup-steps W     bench steps before measuring (default 5)
  --pipeline D         bench, dev-verify: steps in flight (default 3)
  --srv-work-us U      server busy wait after the synthetic compute
  --die-after N        srv: _exit(9) after N exchanges
  --hang-after N       srv: skip srv_push from exchange N on
  --hub-die-after N    hub: _exit(9) after N exchanges
  --no-watcher         no pidfd watcher (device timeouts only)
  --blocks-dispatch N --blocks-combine N --blocks-push N --blocks-srv-wait N
                       grids (0 = by size)
  --srv-ws 0|1         srv: read x and w from the cached workspace that
                       hx_k_srv_wait fills (ctx.x_ws/w_ws, default 1) or from
                       the uncached inbox slot (0)
  --layer-stats 0|1    hub: ctx.layer_stats pair counters, checked once per
                       step (default 1, 0 with --bench)
  --dev-verify         device-side verification stress: payloads are a
                       deterministic pattern, a device kernel checks every
                       combine output right after the combine (no CPU model,
                       no host sync per exchange, no hipGraph); defaults
                       --T-list 3 --layers 6; excludes --graph, --bench,
                       --verify-every, --ref-slow. Prints DEVVERIFY, DEVSTAT,
                       DEVSRV, DEVLAYER, DEVCHUNK, DEVAB, DEVDETAIL lines
  --check-every N      dev-verify: exchanges between reads of the device
                       counters (default 512); a mismatch is noticed, and the
                       run stopped (--stop-on-mismatch), at the next read
  --dev-verify-lines N dev-verify: detail records printed (0..256, default 20)
  --blocks-check N     dev-verify: grid of the check kernel (0 = by size)
  --dv-reread-us U     dev-verify: pause before the delayed re-read of a wrong
                       vector (default 20, 0 = none)
  --inject-stale N     dev-verify positive control: at exchange N server
                       --inject-srv (default 0) overwrites 256 B chunk
                       --inject-chunk (default 1) of its reply with the
                       previous occupant of the reply slot
  --inject-out N       dev-verify positive control: at exchange N the hub
                       corrupts chunk --inject-chunk of the combine output
  --inject-rd N        dev-verify --rd-mode 4 positive control: at exchange N
                       the paired read of server --inject-srv, chunk
                       --inject-chunk is faked: --inject-rd-kind 1 plain read
                       stale, 2 atomic read stale, 3 plain stale and atomic
                       read garbage (default 1)
  --digest             ctx.flush_mode bit 8 (HX_DIGEST) on every process: each
                       exchange is checked end to end by a payload digest; a
                       DIGSTAT line per process reports the counters. Excludes
                       --rd-mode 4
  --inject-late N      digest positive control (dev-verify, build with
                       -DHX_FAULT_INJECT): at exchange N server --inject-srv
                       writes 256 B chunk --inject-chunk of its reply
                       --inject-late-us (default 100) microseconds AFTER its
                       flag; --inject-late-req: the hub does the same to the
                       request x rows it sends to server --inject-srv
  --rdv-timeout-s S    rendezvous waits (default 120)
  --log-every N        progress line every N exchanges (default 100000)
)";

[[noreturn]] static void usage_exit(int code) {
  const char* s = kUsage;
  wr_all(code == 0 ? 1 : 2, s, strlen(s));
  _exit(code);
}
[[noreturn]] static void die_usage(const char* fmt, ...)
    __attribute__((format(printf, 1, 2)));
[[noreturn]] static void die_usage(const char* fmt, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  fprintf(stderr, "hx_loopback: %s (see --help)\n", buf);
  _exit(kExitSetup);
}

static long long to_ll(const std::string& v, const std::string& name) {
  char* e = nullptr;
  errno = 0;
  double d = strtod(v.c_str(), &e);
  if (errno || e == v.c_str() || *e)
    die_usage("bad number for %s: '%s'", name.c_str(), v.c_str());
  return (long long)llround(d);
}
static double to_dbl(const std::string& v, const std::string& name) {
  char* e = nullptr;
  const double d = strtod(v.c_str(), &e);
  if (e == v.c_str() || *e)
    die_usage("bad number for %s: '%s'", name.c_str(), v.c_str());
  return d;
}

static void parse_args(int argc, char** argv) {
  if (argc < 2) usage_exit(kExitSetup);
  const std::string first = argv[1];
  if (first == "-h" || first == "--help") usage_exit(0);
  O.role = first;
  if (first != "hub" && first != "srv" && first != "buses" &&
      first != "selftest")
    die_usage("unknown role '%s'", first.c_str());
  int i = 2;
  auto val = [&](const std::string& a) -> std::string {
    if (i + 1 >= argc) die_usage("%s needs a value", a.c_str());
    return argv[++i];
  };
  for (; i < argc; i++) {
    const std::string a = argv[i];
    if (a == "-h" || a == "--help")
      usage_exit(0);
    else if (a == "--dir")
      O.dir = val(a);
    else if (a == "--nsrv")
      O.nsrv = (int)to_ll(val(a), a);
    else if (a == "--index")
      O.index = (int)to_ll(val(a), a);
    else if (a == "--hub-index")
      O.hub_index = (int)to_ll(val(a), a);
    else if (a == "--device")
      O.device = (int)to_ll(val(a), a);
    else if (a == "--alloc")
      O.alloc = val(a);
    else if (a == "--flush")
      O.flush = (int)to_ll(val(a), a);
    else if (a == "--rd-mode")
      O.rd_mode = (int)to_ll(val(a), a);
    else if (a == "--wait-ms")
      O.wait_ms = to_ll(val(a), a);
    else if (a == "--begin-wait-ms")
      O.begin_wait_ms = to_ll(val(a), a);
    else if (a == "--plan-hash")
      O.plan_hash = strtoull(val(a).c_str(), nullptr, 16);
    else if (a == "--experts")
      O.experts = (int)to_ll(val(a), a);
    else if (a == "--layers") {
      O.layers = (int)to_ll(val(a), a);
      O.layers_set = true;
    } else if (a == "--rows-cap")
      O.rows_cap = (int)to_ll(val(a), a);
    else if (a == "--hidden")
      O.hidden = (int)to_ll(val(a), a);
    else if (a == "--topk")
      O.topk = (int)to_ll(val(a), a);
    else if (a == "--nslots")
      O.nslots = (int)to_ll(val(a), a);
    else if (a == "--iters")
      O.iters = to_ll(val(a), a);
    else if (a == "--seed")
      O.seed = (uint64_t)to_ll(val(a), a);
    else if (a == "--T-list" || a == "--t-list") {
      O.tlist.clear();
      O.tlist_set = true;
      const std::string v = val(a);
      size_t p = 0;
      while (p <= v.size()) {
        size_t q = v.find(',', p);
        if (q == std::string::npos) q = v.size();
        O.tlist.push_back((int)to_ll(v.substr(p, q - p), a));
        p = q + 1;
      }
    } else if (a == "--k-mode") {
      const std::string v = val(a);
      if (v == "lin")
        O.kmode = 0;
      else if (v == "zero")
        O.kmode = 2;
      else if (v.rfind("fixed:", 0) == 0) {
        O.kmode = 1;
        O.kfixed = (int)to_ll(v.substr(6), a);
      } else
        die_usage("--k-mode must be fixed:K, lin or zero");
    } else if (a == "--eager")
      O.eager = true;
    else if (a == "--graph")
      O.graph = true;
    else if (a == "--bench")
      O.bench = true;
    else if (a == "--verify-every")
      O.verify_every = (int)to_ll(val(a), a);
    else if (a == "--stop-on-mismatch") {
      const std::string v = i + 1 < argc ? argv[i + 1] : "";
      if (v == "0" || v == "1") {
        O.stop_on_mismatch = v == "1";
        i++;
      } else
        O.stop_on_mismatch = true;
    } else if (a == "--no-stop-on-mismatch")
      O.stop_on_mismatch = false;
    else if (a == "--ref-slow")
      O.ref_slow = true;
    else if (a == "--warmup-steps")
      O.warmup_steps = (int)to_ll(val(a), a);
    else if (a == "--pipeline")
      O.pipeline = (int)to_ll(val(a), a);
    else if (a == "--srv-work-us")
      O.srv_work_us = to_dbl(val(a), a);
    else if (a == "--die-after")
      O.die_after = to_ll(val(a), a);
    else if (a == "--hang-after")
      O.hang_after = to_ll(val(a), a);
    else if (a == "--hub-die-after")
      O.hub_die_after = to_ll(val(a), a);
    else if (a == "--no-watcher")
      O.no_watcher = true;
    else if (a == "--blocks-dispatch")
      O.blk_dispatch = (int)to_ll(val(a), a);
    else if (a == "--blocks-combine")
      O.blk_combine = (int)to_ll(val(a), a);
    else if (a == "--blocks-push")
      O.blk_push = (int)to_ll(val(a), a);
    else if (a == "--blocks-srv-wait")
      O.blk_srv_wait = (int)to_ll(val(a), a);
    else if (a == "--srv-ws")
      O.srv_ws = (int)to_ll(val(a), a);
    else if (a == "--layer-stats")
      O.layer_stats = (int)to_ll(val(a), a);
    else if (a == "--rdv-timeout-s")
      O.rdv_timeout_s = to_dbl(val(a), a);
    else if (a == "--log-every")
      O.log_every = to_ll(val(a), a);
    else if (a == "--dev-verify")
      O.dev_verify = true;
    else if (a == "--check-every")
      O.check_every = (int)to_ll(val(a), a);
    else if (a == "--dev-verify-lines")
      O.dv_lines = (int)to_ll(val(a), a);
    else if (a == "--blocks-check")
      O.blk_check = (int)to_ll(val(a), a);
    else if (a == "--dv-reread-us")
      O.dv_reread_us = (int)to_ll(val(a), a);
    else if (a == "--inject-stale")
      O.inject_stale = to_ll(val(a), a);
    else if (a == "--inject-out")
      O.inject_out = to_ll(val(a), a);
    else if (a == "--inject-rd")
      O.inject_rd = to_ll(val(a), a);
    else if (a == "--inject-rd-kind")
      O.inject_rd_kind = (int)to_ll(val(a), a);
    else if (a == "--inject-srv")
      O.inject_srv = (int)to_ll(val(a), a);
    else if (a == "--inject-chunk")
      O.inject_chunk = (int)to_ll(val(a), a);
    else if (a == "--digest")
      O.digest = true;
    else if (a == "--inject-late")
      O.inject_late = to_ll(val(a), a);
    else if (a == "--inject-late-us")
      O.inject_late_us = (int)to_ll(val(a), a);
    else if (a == "--inject-late-req")
      O.inject_late_req = true;
    else
      die_usage("unknown option '%s'", a.c_str());
  }
}

static void validate_args() {
  auto need = [](bool ok, const char* msg) {
    if (!ok) die_usage("%s", msg);
  };
  if (O.dev_verify) {
    if (!O.tlist_set) O.tlist = {3};
    if (!O.layers_set) O.layers = 6;
  }
  need(O.nsrv >= 1 && O.nsrv <= HX_MAX_SERVERS, "--nsrv must be in [1, 8]");
  need(O.layers >= 1 && O.layers <= HX_MAX_LAYERS,
       "--layers must be in [1, 64]");
  need(O.experts >= 1 && O.experts <= HX_MAX_EXPERTS,
       "--experts must be in [1, 512]");
  need(O.topk >= 1 && O.topk <= HX_MAX_TOPK, "--topk must be in [1, 16]");
  need(O.experts >= O.topk, "--experts must be >= --topk (distinct ids)");
  need(O.nslots >= 1 && O.nslots <= HX_MAX_SLOTS, "--nslots must be in [1, 4]");
  need(O.hidden >= 8 && O.hidden % 8 == 0, "--hidden must be a multiple of 8");
  need(O.rows_cap >= 1, "--rows-cap must be >= 1");
  need(O.hub_index >= 0 && O.hub_index < 256,
       "--hub-index must be in [0, 255]");
  need(O.index >= 0 && O.index < O.nsrv, "--index must be in [0, nsrv)");
  need(O.device >= 0, "--device must be >= 0");
  need(O.flush >= 0 && O.flush <= 15, "--flush must be in [0, 15]");
  need(O.rd_mode >= 0 && O.rd_mode <= 4, "--rd-mode must be in [0, 4]");
  need(O.wait_ms >= 1 && O.begin_wait_ms >= 1,
       "--wait-ms and --begin-wait-ms must be >= 1");
  need(O.alloc == "uncached" || O.alloc == "coarse" || O.alloc == "finegrained",
       "--alloc must be uncached, coarse or finegrained");
  need(O.iters >= 1, "--iters must be >= 1");
  need(!O.tlist.empty(), "--T-list is empty");
  for (int T : O.tlist) {
    need(T >= 1 && T <= O.rows_cap,
         "every --T-list entry must be in "
         "[1, rows-cap]");
    need(!O.graph || T <= kMaxGraphT, "--graph needs every T <= 48");
  }
  need(!(O.graph && O.eager), "--graph and --eager are exclusive");
  need(O.check_every >= 1, "--check-every must be >= 1");
  need(O.dv_lines >= 0 && O.dv_lines <= kDvRecs,
       "--dev-verify-lines must be in [0, 256]");
  need(O.blk_check >= 0, "--blocks-check must be >= 0");
  need(O.dv_reread_us >= 0 && O.dv_reread_us <= 1000000,
       "--dv-reread-us must be in [0, 1000000]");
  const bool inject = O.inject_stale >= 0 || O.inject_out >= 0 ||
                      O.inject_rd >= 0 || O.inject_late >= 0;
  need(O.rd_mode != 4 || O.dev_verify, "--rd-mode 4 needs --dev-verify");
  need(!inject || O.dev_verify, "--inject-* needs --dev-verify");
  need(O.inject_rd < 0 || O.rd_mode == 4, "--inject-rd needs --rd-mode 4");
  need(O.inject_rd_kind >= 1 && O.inject_rd_kind <= 3,
       "--inject-rd-kind must be in [1, 3]");
  need(O.inject_srv >= 0 && O.inject_srv < O.nsrv,
       "--inject-srv must be in [0, nsrv)");
  need(O.inject_chunk >= 0, "--inject-chunk must be >= 0");
  need(!O.digest || O.rd_mode != 4,
       "--digest excludes --rd-mode 4 (its combine is loopback-local)");
  need(O.inject_late < 0 || (O.dev_verify && O.digest),
       "--inject-late needs --dev-verify and --digest");
  need(O.inject_late_us >= 0 && O.inject_late_us <= 10000000,
       "--inject-late-us must be in [0, 10000000]");
  need(!O.inject_late_req || O.inject_late >= 0,
       "--inject-late-req needs --inject-late");
#ifndef HX_FAULT_INJECT
  need(O.inject_late < 0,
       "--inject-late needs a build with -DHX_FAULT_INJECT "
       "(HX_LB_CXXFLAGS of run_loopback.sh)");
#endif
  if (O.dev_verify) {
    need(!O.graph && !O.bench, "--dev-verify excludes --graph and --bench");
    need(O.verify_every < 0 && !O.ref_slow,
         "--dev-verify excludes --verify-every and --ref-slow");
    for (int T : O.tlist)
      need(!inject ||
               (long long)O.inject_chunk * 256 < (long long)T * O.hidden * 2,
           "--inject-chunk lies beyond the payload of the smallest T");
  }
  need(O.kmode != 1 || (O.kfixed >= 0 && O.kfixed <= 512),
       "--k-mode fixed:K needs K in [0, 512]");
  need(O.pipeline >= 1 && O.pipeline <= kDescRing - 2 && O.warmup_steps >= 0,
       "--pipeline must be in [1, 62] and --warmup-steps >= 0");
  need(O.srv_ws == 0 || O.srv_ws == 1, "--srv-ws must be 0 or 1");
  need(O.layer_stats >= -1 && O.layer_stats <= 1,
       "--layer-stats must be 0 or 1");
  if (O.layer_stats < 0) O.layer_stats = O.bench || O.dev_verify ? 0 : 1;
  need(O.srv_work_us >= 0, "--srv-work-us must be >= 0");
  need(O.rdv_timeout_s > 0, "--rdv-timeout-s must be > 0");
  if (O.role == "hub" || O.role == "srv")
    need(!O.dir.empty(), "--dir is required");
}

static int tmax_of_list() {
  int m = 1;
  for (int T : O.tlist) m = std::max(m, T);
  return m;
}
static int verify_every_for(int T) {
  if (O.verify_every >= 0) return O.verify_every;
  return T <= 48 ? 1 : 16;
}
static bool whole_steps() { return O.graph || O.bench || O.dev_verify; }
static long long steps_total() {
  const long long n = (O.iters + O.layers - 1) / O.layers;
  return O.bench ? n + O.warmup_steps : n;
}
static int layers_in_step(long long step) {
  if (whole_steps()) return O.layers;
  return (int)std::min<long long>(O.layers, O.iters - step * O.layers);
}

// ---------------------------------------------------------------------------
// rendezvous directory
// ---------------------------------------------------------------------------
static char g_eph[64][512];
static char g_pending[512];
static volatile sig_atomic_t g_neph = 0;

static std::string rpath(const std::string& n) { return O.dir + "/" + n; }
static std::string hub_name() { return "hub" + std::to_string(O.hub_index); }
static std::string srv_name(int s) { return "srv" + std::to_string(s); }
static std::string ack_name(int s) {
  return "ack_s" + std::to_string(s) + "_h" + std::to_string(O.hub_index);
}

static void eph_unlink_all() {
  if (g_pending[0]) unlink(g_pending);
  for (int i = 0; i < g_neph; i++) unlink(g_eph[i]);
}

static bool rdv_publish(const std::string& name, const std::string& text,
                        bool ephemeral) {
  const std::string fin = rpath(name);
  const std::string tmp =
      O.dir + "/." + name + ".tmp." + std::to_string((long)getpid());
  if (tmp.size() >= sizeof g_pending || fin.size() >= sizeof g_eph[0])
    return false;
  if (ephemeral && g_neph < 64) {
    memcpy(g_eph[g_neph], fin.c_str(), fin.size() + 1);
    g_neph = g_neph + 1;
  }
  memcpy(g_pending, tmp.c_str(), tmp.size() + 1);
  const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) return false;
  size_t off = 0;
  while (off < text.size()) {
    const ssize_t w = ::write(fd, text.data() + off, text.size() - off);
    if (w < 0 && errno == EINTR) continue;
    if (w <= 0) {
      ::close(fd);
      unlink(tmp.c_str());
      g_pending[0] = 0;
      return false;
    }
    off += (size_t)w;
  }
  ::close(fd);
  const bool ok = rename(tmp.c_str(), fin.c_str()) == 0;
  g_pending[0] = 0;
  return ok;
}

static bool rdv_read(const std::string& name, std::string* out) {
  FILE* f = fopen(rpath(name).c_str(), "rb");
  if (!f) return false;
  out->clear();
  char buf[4096];
  size_t n;
  while ((n = fread(buf, 1, sizeof buf, f)) > 0) out->append(buf, n);
  fclose(f);
  return !out->empty();
}

static void rdv_remove(const std::string& name) { unlink(rpath(name).c_str()); }

static void mkdir_p(const std::string& path) {
  for (size_t p = 1; p <= path.size(); p++)
    if (p == path.size() || path[p] == '/') {
      const std::string d = path.substr(0, p);
      if (mkdir(d.c_str(), 0755) != 0 && errno != EEXIST)
        die_setup("mkdir %s: %s", d.c_str(), strerror(errno));
    }
}

static bool under_tmp(const std::string& p) {
  for (const char* b : {"/tmp", "/var/tmp", "/private/tmp"}) {
    const size_t n = strlen(b);
    if (p.compare(0, n, b) == 0 && (p.size() == n || p[n] == '/')) return true;
  }
  return false;
}

// Removes the known rendezvous files (and the directory if it is empty) after
// a clean run; unknown files are left alone.
static void rdv_clean_dir() {
  DIR* d = opendir(O.dir.c_str());
  if (!d) return;
  while (const dirent* e = readdir(d)) {
    const std::string n = e->d_name;
    for (const char* pre :
         {"hub", "srv", "ack_", "ready_", "done_", "fault_", "."})
      if (n != "." && n != ".." && n.rfind(pre, 0) == 0) {
        unlink(rpath(n).c_str());
        break;
      }
  }
  closedir(d);
  rmdir(O.dir.c_str());
}

// Earliest fault stamp (die/hang injection) in the directory, or 0.
static int64_t earliest_fault_ns() {
  int64_t best = 0;
  DIR* d = opendir(O.dir.c_str());
  if (!d) return 0;
  while (const dirent* e = readdir(d)) {
    if (strncmp(e->d_name, "fault_", 6) != 0) continue;
    std::string t;
    if (!rdv_read(e->d_name, &t)) continue;
    const size_t sp = t.find(' ');
    const int64_t v =
        sp == std::string::npos ? 0 : strtoll(t.c_str() + sp + 1, nullptr, 10);
    if (v > 0 && (best == 0 || v < best)) best = v;
  }
  closedir(d);
  return best;
}

static void write_fault_stamp(const char* kind) {
  char buf[96];
  snprintf(buf, sizeof buf, "%s %lld\n", kind, (long long)now_ns());
  rdv_publish(std::string("fault_") + G.role + std::to_string(G.index), buf,
              false);
}

// Waits for <name>; fails fast if the peer pid (when known) is gone.
static bool rdv_wait_file(const std::string& name, long peer_pid,
                          double timeout_s, std::string* text) {
  const int64_t t0 = now_ns();
  for (;;) {
    if (rdv_read(name, text)) return true;
    if (peer_pid > 0 && !pid_alive(peer_pid)) return false;
    if ((double)(now_ns() - t0) / 1e9 > timeout_s) return false;
    sleep_us(2000);
  }
}

// Waits for the info file of a peer; a file whose pid is not running is stale.
static void wait_peer_info(const std::string& name, const char* who,
                           std::string* js, long* pid) {
  const int64_t t0 = now_ns();
  bool warned = false;
  int64_t last_note = t0;
  for (;;) {
    if (rdv_read(name, js)) {
      *pid = (long)js_int(*js, "pid", 0);
      if (*pid > 0 && pid_alive(*pid)) return;
      if (!warned)
        say("%s: %s is stale (pid %ld not running), waiting for a fresh one",
            who, name.c_str(), *pid);
      warned = true;
    }
    const int64_t now = now_ns();
    if ((double)(now - t0) / 1e9 > O.rdv_timeout_s)
      die_setup(
          "%s did not publish %s within %.0f s (no such process running? "
          "rendezvous dir %s)",
          who, name.c_str(), O.rdv_timeout_s, O.dir.c_str());
    if (now - last_note > 10000000000ll) {
      say("still waiting for %s (%s)", who, name.c_str());
      last_note = now;
    }
    sleep_us(2000);
  }
}

// ---------------------------------------------------------------------------
// abort record, exit
// ---------------------------------------------------------------------------
static const char* phase_name(unsigned p) {
  static const char* n[] = {"none",      "DISPATCH", "COMBINE", "SRV_WAIT",
                            "SRV_BEGIN", "HDR",      "PUSH",    "PLAN",
                            "HOST",      "DIGEST"};
  return p < 10 ? n[p] : "?";
}
static const char* mm_name(unsigned f) {
  static const char* n[] = {"?",      "MAGIC",  "PLAN",    "EPOCH",  "T",
                            "BUCKET", "OPCODE", "VERSION", "SKIPPED"};
  return f < 9 ? n[f] : "?";
}

struct Rec {
  unsigned phase = 0, peer = 0, ms = 0;
  uint32_t seq_lo = 0, epoch_lo = 0;
  int layer = -1;
};
static Rec decode_rec(uint64_t r) {
  Rec d;
  d.phase = (unsigned)((r >> 4) & 0xFu);
  d.peer = (unsigned)((r >> 8) & 0xFFu);
  d.ms = (unsigned)((r >> 16) & 0xFFFFu);
  d.seq_lo = (uint32_t)(r >> 32);
  if (d.phase == HX_PH_SRV_BEGIN || d.phase == HX_PH_HDR) {
    d.epoch_lo = d.seq_lo;
  } else {
    d.layer = (int)((d.seq_lo + 63u) & 63u);
    d.epoch_lo = (d.seq_lo - (uint32_t)(d.layer + 1)) >> 6;
  }
  return d;
}
static std::string rec_text(uint64_t r) {
  if (r == 0) return "no record";
  const Rec d = decode_rec(r);
  char buf[256];
  if (d.phase == HX_PH_HDR)
    snprintf(buf, sizeof buf, "phase=%u(%s) field=%s epoch=%u", d.phase,
             phase_name(d.phase), mm_name(d.peer), d.epoch_lo);
  else if (d.phase == HX_PH_SRV_BEGIN)
    snprintf(buf, sizeof buf, "phase=%u(%s) peer=%u waited_ms=%u epoch=%u",
             d.phase, phase_name(d.phase), d.peer, d.ms, d.epoch_lo);
  else
    snprintf(buf, sizeof buf,
             "phase=%u(%s) peer=%u waited_ms=%u layer=%d epoch=%u", d.phase,
             phase_name(d.phase), d.peer, d.ms, d.layer, d.epoch_lo);
  return buf;
}

static size_t put_u64(char* dst, uint64_t v) {
  char tmp[24];
  size_t n = 0;
  do {
    tmp[n++] = (char)('0' + v % 10);
    v /= 10;
  } while (v);
  for (size_t i = 0; i < n; i++) dst[i] = tmp[n - 1 - i];
  return n;
}
static size_t put_str(char* dst, const char* s) {
  const size_t n = strlen(s);
  memcpy(dst, s, n);
  return n;
}

// SIGTERM/SIGINT/SIGHUP: final line (async-signal-safe), unlink, exit.
static void on_signal(int sig) {
  if (G.finishing.exchange(1) != 0) _exit(128 + sig);
  char b[320];
  size_t n = 0;
  n += put_str(b + n, "HXLB ");
  n += put_str(b + n, G.role);
  b[n++] = ' ';
  n += put_u64(b + n, (uint64_t)G.index);
  n += put_str(b + n, " iters=");
  n += put_u64(b + n, G.iters.load());
  n += put_str(b + n, " mismatches=");
  n += put_u64(b + n, G.mismatches.load());
  n += put_str(b + n, " abort=0 exit=");
  n += put_u64(b + n, (uint64_t)(128 + sig));
  n += put_str(b + n, " reason=signal");
  n += put_u64(b + n, (uint64_t)sig);
  b[n++] = '\n';
  wr_all(1, b, n);
  eph_unlink_all();
  _exit(128 + sig);
}
static void install_signals() {
  struct sigaction sa;
  memset(&sa, 0, sizeof sa);
  sa.sa_handler = on_signal;
  sigemptyset(&sa.sa_mask);
  sigaction(SIGINT, &sa, nullptr);
  sigaction(SIGTERM, &sa, nullptr);
  sigaction(SIGHUP, &sa, nullptr);
}

[[noreturn]] static void finish(int code, const char* reason) {
  int expected = 0;
  if (!G.finishing.compare_exchange_strong(expected, 1))
    for (;;) sleep(1);
  const int64_t t_exit = now_ns();
  const uint64_t rec = G.abort_rec.load();
  char abort_s[32] = "0";
  if (rec)
    snprintf(abort_s, sizeof abort_s, "0x%016llx", (unsigned long long)rec);
  const double wall = (double)(t_exit - G.t0_ns) / 1e9;
  char line[1400];
  int n = snprintf(
      line, sizeof line,
      "HXLB %s %d iters=%llu mismatches=%llu abort=%s exit=%d reason=%s "
      "steps=%llu wall_s=%.3f rate_per_s=%.1f",
      G.role, G.index, (unsigned long long)G.iters.load(),
      (unsigned long long)G.mismatches.load(), abort_s, code, reason,
      (unsigned long long)G.steps.load(), wall,
      wall > 0 ? (double)G.iters.load() / wall : 0.0);
  if (rec) {
    const Rec d = decode_rec(rec);
    n += snprintf(line + n, sizeof line - (size_t)n,
                  " phase=%u phase_name=%s peer=%u waited_ms=%u layer=%d",
                  d.phase, phase_name(d.phase), d.peer, d.ms, d.layer);
  }
  const int64_t fault = O.dir.empty() ? 0 : earliest_fault_ns();
  const int64_t det = G.t_detect_ns.load();
  n += snprintf(line + n, sizeof line - (size_t)n,
                " arch=%s alloc=%s tpm=%lld watcher=%d peer_dead=%d "
                "kill_to_exit_ms=%.1f kill_to_detect_ms=%.1f flush=%d "
                "rd_mode=%d\n",
                G.arch[0] ? G.arch : "-", G.alloc[0] ? G.alloc : "-", G.tpm,
                G.watcher_on ? 1 : 0, G.peer_dead.load(),
                fault > 0 ? (double)(t_exit - fault) / 1e6 : -1.0,
                fault > 0 && det > 0 ? (double)(det - fault) / 1e6 : -1.0,
                O.flush, O.rd_mode);
  wr_all(1, line, (size_t)n);
  eph_unlink_all();
  if (G.clean_dir) rdv_clean_dir();
  _exit(code);
}

[[noreturn]] static void die_now(const char* what) {
  write_fault_stamp(what);
  say("fault injection: %s now (_exit(9), no cleanup)", what);
  _exit(9);
}

// ---------------------------------------------------------------------------
// device code
// ---------------------------------------------------------------------------
__global__ void k_wall(long long* out) {
  if (threadIdx.x == 0) *out = wall_clock64();
}

__global__ void k_spin(long long ticks) {
  if (threadIdx.x != 0) return;
  const long long t0 = wall_clock64();
  while (wall_clock64() - t0 < ticks) __builtin_amdgcn_s_sleep(1);
}

// The synthetic compute: grid (ceil(hidden / 256), T), one column per thread.
__global__ void __launch_bounds__(256)
    k_synth(const __half* __restrict__ x, const int32_t* __restrict__ ids,
            const float* __restrict__ w, __half* __restrict__ y, int topk,
            int hidden) {
  const int r = (int)blockIdx.y;
  const int c = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (c >= hidden) return;
  const float xv = __half2float(x[(size_t)r * hidden + c]);
  float acc = 0.0f;
  for (int k = 0; k < topk; k++) {
    const int id = ids[r * topk + k];
    if (id >= 0) acc = acc + (xv * w[r * topk + k]) * (float)((id & 7) + 1);
  }
  y[(size_t)r * hidden + c] = __float2half(acc);
}

__global__ void k_cvt(const float* in, uint16_t* out, int n) {
  const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) out[i] = __half_as_ushort(__float2half(in[i]));
}

// ---------------------------------------------------------------------------
// dev-verify: pattern, device checker and record layouts (host and device)
// ---------------------------------------------------------------------------
// Payload of participant `who` (kDvHub or a server index) for (epoch, layer):
// finite fp16 with |x| in [1, 64), a pure function of (seed, epoch, layer, who,
// element index). The hub's y and every server's reply are such streams, so a
// device kernel can recompute the combine output and every input of it.
#define DV_HD static inline __host__ __device__
#define DV_CHECK_THREADS 256
enum { kDvHub = 255 };
typedef unsigned long long dvu64;

DV_HD uint64_t dv_mix(uint64_t z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
DV_HD uint64_t dv_base(uint64_t seed, uint64_t epoch, uint32_t layer,
                       uint32_t who) {
  uint64_t k = dv_mix(seed ^ 0xD1B54A32D192ED03ull);
  k = dv_mix(k ^ epoch);
  return dv_mix(k ^ (((uint64_t)layer << 8) | who));
}
DV_HD uint32_t dv_h16(uint32_t r) {
  const uint32_t e = 15u + ((((r >> 10) & 7u) * 6u) >> 3);
  return (r & 0x8000u) | (e << 10) | (r & 0x3FFu);
}
// Vector v = elements 8v..8v+7 (the even element in the low half of a word).
DV_HD uint4 dv_vec(uint64_t base, uint64_t v) {
  const uint64_t a = dv_mix(base ^ ((2ull * v) * 0xA24BAED4963EE407ull));
  const uint64_t b = dv_mix(base ^ ((2ull * v + 1ull) * 0xA24BAED4963EE407ull));
  uint4 r;
  r.x = dv_h16((uint32_t)a) | (dv_h16((uint32_t)(a >> 16)) << 16);
  r.y = dv_h16((uint32_t)(a >> 32)) | (dv_h16((uint32_t)(a >> 48)) << 16);
  r.z = dv_h16((uint32_t)b) | (dv_h16((uint32_t)(b >> 16)) << 16);
  r.w = dv_h16((uint32_t)(b >> 32)) | (dv_h16((uint32_t)(b >> 48)) << 16);
  return r;
}
// The slot of (epoch, layer) was last written by the occupant before it:
// nslots layers earlier, in the previous step for the first layers.
DV_HD bool dv_prev_occ(uint64_t epoch, int layer, int nslots, int nlayers,
                       uint64_t* pe, int* pl) {
  if (layer >= nslots) {
    *pe = epoch;
    *pl = layer - nslots;
    return true;
  }
  if (epoch <= 1) return false;
  *pe = epoch - 1;
  *pl = layer + nslots * ((nlayers - 1 - layer) / nslots);
  return true;
}
DV_HD bool dv_eq(const uint4& a, const uint4& b) {
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

// One server's reply vector as the classifier of a wrong output saw it.
struct DvSrv {
  uint32_t plain[4], atomic[4], later[4], hash[4], prev[4];
};
struct DvRec {
  dvu64 valid, exch, epoch, v;
  uint32_t layer, T, slot, nsrv;
  uint32_t chunk, nvec_chunk, bad_mask, elems_bad;
  uint32_t hub_y_bad, now_bad, atomic_bad, later_bad;
  uint32_t now_prev, explains, prev_valid, prev_layer;
  dvu64 prev_epoch;
  uint32_t got[4], want[4], yh[4], yh_hash[4];
  DvSrv s[HX_MAX_SERVERS];
};
// --rd-mode 4: one wrong vector of a paired (plain, atomic) read.
struct DvAbRec {
  dvu64 valid, exch, epoch, v;
  uint32_t layer, T, slot, srv;
  uint32_t chunk, cat, a_prev, b_prev;
  uint32_t injected, pad[3];
  uint32_t a[4], b[4], want[4], prev[4], later[4];
};
struct DvBlk {
  dvu64 checks, elems, bad_exch, bad_vec, bad_elem, bad_chunk;
  dvu64 last_bad_exch, nrec, hub_y_bad, explain_none, ab_nrec;
  dvu64 rnow_ok[HX_MAX_SERVERS], rnow_bad[HX_MAX_SERVERS];
  dvu64 ratomic_bad[HX_MAX_SERVERS], rnow_prev[HX_MAX_SERVERS];
  dvu64 explain_prev[HX_MAX_SERVERS];
  dvu64 layer_elem[HX_MAX_LAYERS], layer_chunk[HX_MAX_LAYERS];
  dvu64 layer_exch[HX_MAX_LAYERS];
  dvu64 ab_cat[HX_MAX_SERVERS][4], ab_a_prev[HX_MAX_SERVERS];
  dvu64 ab_b_prev[HX_MAX_SERVERS];
};
static_assert(sizeof(DvRec) % 8 == 0 && sizeof(DvAbRec) % 8 == 0 &&
                  sizeof(DvBlk) % 8 == 0,
              "device records are 8-byte multiples");
struct DvArgs {
  HxCtx* c;
  DvBlk* d;
  DvRec* rec;
  DvAbRec* ab;
  dvu64* mark;
  dvu64* abmark;
  unsigned* hist;
  const uint4* yh;
  uint4* out;
  dvu64 seed, exch;
  int T, layer, nblocks, reread_us;
  int inj_srv, inj_chunk, inj_kind;
};

// n parts (hub first); the loop is unrolled to the maximum with a guard so the
// callers' arrays stay in registers.
static __device__ __forceinline__ uint4 dv_sum_dev(const uint4* p, int n) {
  float acc[8];
  {
    const __half* h = reinterpret_cast<const __half*>(&p[0]);
#pragma unroll
    for (int q = 0; q < 8; q++) acc[q] = __half2float(h[q]);
  }
#pragma unroll
  for (int s = 1; s <= HX_MAX_SERVERS; s++) {
    if (s < n) {
      const __half* h = reinterpret_cast<const __half*>(&p[s]);
#pragma unroll
      for (int q = 0; q < 8; q++) acc[q] += __half2float(h[q]);
    }
  }
  uint4 res;
  __half* ro = reinterpret_cast<__half*>(&res);
#pragma unroll
  for (int q = 0; q < 8; q++) ro[q] = __float2half(acc[q]);
  return res;
}
// Cache-bypassing read: two relaxed system-scope 8 B atomic loads.
static __device__ __forceinline__ uint4 dv_ld_atomic(const uint4* p) {
  const uint64_t* q = reinterpret_cast<const uint64_t*>(p);
  const uint64_t a =
      __hip_atomic_load(q, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  const uint64_t b =
      __hip_atomic_load(q + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  return make_uint4((uint32_t)a, (uint32_t)(a >> 32), (uint32_t)b,
                    (uint32_t)(b >> 32));
}
static __device__ __forceinline__ int dv_nbad(const uint4& a, const uint4& b) {
  const uint32_t d[4] = {a.x ^ b.x, a.y ^ b.y, a.z ^ b.z, a.w ^ b.w};
  int n = 0;
  for (int i = 0; i < 4; i++)
    n += (int)((d[i] & 0xFFFFu) != 0u) + (int)((d[i] >> 16) != 0u);
  return n;
}
static __device__ __forceinline__ void dv_put(uint32_t* d, const uint4& v) {
  d[0] = v.x;
  d[1] = v.y;
  d[2] = v.z;
  d[3] = v.w;
}
static __device__ __forceinline__ void dv_spin_us(const HxCtx* c, int us) {
  if (us <= 0) return;
  const long long until =
      wall_clock64() + (long long)us * c->ticks_per_ms / 1000;
  while (wall_clock64() < until) {
  }
}

// Fills layer blockIdx.y of dst (layer_stride_vec vectors apart) with the
// pattern of this process (hub or server) for the epoch the ctx holds.
__global__ void __launch_bounds__(256)
    k_dv_fill(const HxCtx* __restrict__ c, uint4* __restrict__ dst,
              long long layer_stride_vec, dvu64 seed, int T) {
  if (hx_aborted(c)) return;
  const int l = (int)blockIdx.y;
  const uint32_t who = c->role == 1u ? (uint32_t)kDvHub : c->index;
  const uint64_t base = dv_base(seed, c->status[HX_ST_EPOCH], (uint32_t)l, who);
  const long long nvec = (long long)T * (long long)c->hidden / 8;
  uint4* y = dst + (long long)l * layer_stride_vec;
  for (long long v = (long long)blockIdx.x * 256 + (long long)threadIdx.x;
       v < nvec; v += (long long)gridDim.x * 256)
    y[v] = dv_vec(base, (uint64_t)v);
}

// A wrong output vector: counters, classification of every server's reply by
// re-reads (plain, atomic), and, once per 256 B chunk and exchange, a record
// with a second atomic re-read after --dv-reread-us.
static __device__ __noinline__ void dv_bad_vec(DvArgs a, long long v, uint4 got,
                                               uint4 want) {
  const HxCtx* c = a.c;
  DvBlk* d = a.d;
  const int ns = (int)c->nsrv, nslots = (int)c->nslots, L = (int)c->nlayers;
  const int layer = a.layer, slot = layer % nslots;
  const dvu64 ex1 = a.exch + 1ull;
  const uint64_t epoch = c->status[HX_ST_EPOCH];
  uint64_t base[HX_MAX_SERVERS + 1];
  uint4 part[HX_MAX_SERVERS + 1];
  base[0] = dv_base(a.seed, epoch, (uint32_t)layer, (uint32_t)kDvHub);
  part[0] = dv_vec(base[0], (uint64_t)v);
  for (int s = 0; s < ns; s++) {
    base[s + 1] = dv_base(a.seed, epoch, (uint32_t)layer, (uint32_t)s);
    part[s + 1] = dv_vec(base[s + 1], (uint64_t)v);
  }
  const int nbe = dv_nbad(got, want);
  atomicAdd(&d->bad_vec, 1ull);
  atomicAdd(&d->bad_elem, (dvu64)nbe);
  atomicAdd(&d->layer_elem[layer], (dvu64)nbe);
  if (atomicExch(&d->last_bad_exch, ex1) != ex1) {
    atomicAdd(&d->bad_exch, 1ull);
    atomicAdd(&d->layer_exch[layer], 1ull);
  }
  const unsigned chunk = (unsigned)(v >> 4);
  const bool first = atomicExch(&a.mark[chunk], ex1) != ex1;
  if (first) {
    atomicAdd(&d->bad_chunk, 1ull);
    atomicAdd(&d->layer_chunk[layer], 1ull);
    atomicAdd(&a.hist[chunk], 1u);
  }
  uint64_t pe = 0;
  int pl = 0;
  const bool has_prev = dv_prev_occ(epoch, layer, nslots, L, &pe, &pl);
  uint4 nw[HX_MAX_SERVERS], at[HX_MAX_SERVERS], prv[HX_MAX_SERVERS];
  uint32_t now_bad = 0u, atomic_bad = 0u, now_prev = 0u, explains = 0u;
  for (int s = 0; s < ns; s++) {
    const uint4* ry = reinterpret_cast<const uint4*>(hx_reply_y(c, s, slot));
    nw[s] = ry[v];
    at[s] = dv_ld_atomic(ry + v);
    prv[s] = has_prev ? dv_vec(dv_base(a.seed, pe, (uint32_t)pl, (uint32_t)s),
                               (uint64_t)v)
                      : make_uint4(0u, 0u, 0u, 0u);
    if (dv_eq(nw[s], part[s + 1])) {
      atomicAdd(&d->rnow_ok[s], 1ull);
    } else {
      atomicAdd(&d->rnow_bad[s], 1ull);
      now_bad |= 1u << s;
      if (dv_eq(nw[s], prv[s])) {
        atomicAdd(&d->rnow_prev[s], 1ull);
        now_prev |= 1u << s;
      }
    }
    if (!dv_eq(at[s], part[s + 1])) {
      atomicAdd(&d->ratomic_bad[s], 1ull);
      atomic_bad |= 1u << s;
    }
  }
  const uint4 yhn = a.yh[v];
  const bool yb = !dv_eq(yhn, part[0]);
  if (yb) atomicAdd(&d->hub_y_bad, 1ull);
  for (int s = 0; s < ns; s++) {
    uint4 q[HX_MAX_SERVERS + 1];
    for (int j = 0; j <= ns; j++) q[j] = part[j];
    q[s + 1] = prv[s];
    if (dv_eq(dv_sum_dev(q, ns + 1), got)) {
      explains |= 1u << s;
      atomicAdd(&d->explain_prev[s], 1ull);
    }
  }
  if (explains == 0u) atomicAdd(&d->explain_none, 1ull);
  if (!first) return;
  const dvu64 ticket = atomicAdd(&d->nrec, 1ull);
  if (ticket >= (dvu64)kDvRecs) return;
  DvRec* r = a.rec + ticket;
  const long long nvec = (long long)a.T * (long long)c->hidden / 8;
  const long long v0 = (long long)chunk * 16;
  const int nvc = (int)(nvec - v0 < 16 ? nvec - v0 : 16);
  uint32_t mask = 0u;
  for (int j = 0; j < nvc; j++) {
    uint4 q[HX_MAX_SERVERS + 1];
    for (int s = 0; s <= ns; s++) q[s] = dv_vec(base[s], (uint64_t)(v0 + j));
    if (!dv_eq(dv_sum_dev(q, ns + 1), a.out[v0 + j])) mask |= 1u << j;
  }
  r->exch = a.exch;
  r->epoch = epoch;
  r->v = (dvu64)v;
  r->layer = (uint32_t)layer;
  r->T = (uint32_t)a.T;
  r->slot = (uint32_t)slot;
  r->nsrv = (uint32_t)ns;
  r->chunk = chunk;
  r->nvec_chunk = (uint32_t)nvc;
  r->bad_mask = mask;
  r->elems_bad = (uint32_t)nbe;
  r->hub_y_bad = yb ? 1u : 0u;
  r->now_bad = now_bad;
  r->atomic_bad = atomic_bad;
  r->now_prev = now_prev;
  r->explains = explains;
  r->prev_valid = has_prev ? 1u : 0u;
  r->prev_layer = (uint32_t)pl;
  r->prev_epoch = pe;
  dv_put(r->got, got);
  dv_put(r->want, want);
  dv_put(r->yh, yhn);
  dv_put(r->yh_hash, part[0]);
  for (int s = 0; s < ns; s++) {
    dv_put(r->s[s].plain, nw[s]);
    dv_put(r->s[s].atomic, at[s]);
    dv_put(r->s[s].hash, part[s + 1]);
    dv_put(r->s[s].prev, prv[s]);
  }
  dv_spin_us(c, a.reread_us);
  uint32_t later_bad = 0u;
  for (int s = 0; s < ns; s++) {
    const uint4* ry = reinterpret_cast<const uint4*>(hx_reply_y(c, s, slot));
    const uint4 lt = dv_ld_atomic(ry + v);
    dv_put(r->s[s].later, lt);
    if (!dv_eq(lt, part[s + 1])) later_bad |= 1u << s;
  }
  r->later_bad = later_bad;
  __threadfence();
  __hip_atomic_store(&r->valid, 1ull, __ATOMIC_RELEASE,
                     __HIP_MEMORY_SCOPE_SYSTEM);
}

// Checks the combine output of layer a.layer against the sum of the patterns
// (the combine's own arithmetic: hub y first, then the servers in order).
__global__ void __launch_bounds__(DV_CHECK_THREADS) k_dv_check(DvArgs a) {
  __shared__ int s_skip;
  __shared__ unsigned s_vecs;
  const int t = (int)threadIdx.x, b = (int)blockIdx.x;
  const HxCtx* c = a.c;
  if (t == 0) {
    s_skip = hx_aborted(c) ? 1 : 0;
    s_vecs = 0u;
  }
  __syncthreads();
  if (s_skip) return;
  const int ns = (int)c->nsrv;
  const uint64_t epoch = c->status[HX_ST_EPOCH];
  const long long nvec = (long long)a.T * (long long)c->hidden / 8;
  const long long gid = (long long)b * DV_CHECK_THREADS + t;
  const long long gstride = (long long)a.nblocks * DV_CHECK_THREADS;
  uint64_t base[HX_MAX_SERVERS + 1];
#pragma unroll
  for (int s = 0; s <= HX_MAX_SERVERS; s++)
    if (s <= ns)
      base[s] = dv_base(a.seed, epoch, (uint32_t)a.layer,
                        s == 0 ? (uint32_t)kDvHub : (uint32_t)(s - 1));
  unsigned mine = 0u;
  for (long long v = gid; v < nvec; v += gstride) {
    uint4 part[HX_MAX_SERVERS + 1];
#pragma unroll
    for (int s = 0; s <= HX_MAX_SERVERS; s++)
      if (s <= ns) part[s] = dv_vec(base[s], (uint64_t)v);
    const uint4 want = dv_sum_dev(part, ns + 1);
    const uint4 got = a.out[v];
    mine++;
    if (!dv_eq(got, want)) dv_bad_vec(a, v, got, want);
  }
  atomicAdd(&s_vecs, mine);
  __syncthreads();
  if (t == 0) {
    atomicAdd(&a.d->elems, (dvu64)s_vecs * 8ull);
    if (b == 0) atomicAdd(&a.d->checks, 1ull);
  }
}

// --rd-mode 4: one wrong vector of a paired read of server s (A plain, B
// atomic) against the pattern; once per (chunk, server, exchange) a record.
static __device__ __noinline__ void dv_ab_bad(DvArgs a, int s, long long v,
                                              uint4 A, uint4 Bv, uint4 hs,
                                              uint32_t injected) {
  const HxCtx* c = a.c;
  DvBlk* d = a.d;
  const uint64_t epoch = c->status[HX_ST_EPOCH];
  const bool aok = dv_eq(A, hs), bok = dv_eq(Bv, hs);
  const int cat = (!aok && bok) ? 0 : (aok && !bok) ? 1 : dv_eq(A, Bv) ? 2 : 3;
  uint64_t pe = 0;
  int pl = 0;
  const bool has_prev =
      dv_prev_occ(epoch, a.layer, (int)c->nslots, (int)c->nlayers, &pe, &pl);
  const uint4 pv =
      has_prev
          ? dv_vec(dv_base(a.seed, pe, (uint32_t)pl, (uint32_t)s), (uint64_t)v)
          : make_uint4(0u, 0u, 0u, 0u);
  const bool a_prev = !aok && dv_eq(A, pv), b_prev = !bok && dv_eq(Bv, pv);
  atomicAdd(&d->ab_cat[s][cat], 1ull);
  if (a_prev) atomicAdd(&d->ab_a_prev[s], 1ull);
  if (b_prev) atomicAdd(&d->ab_b_prev[s], 1ull);
  const dvu64 ex1 = a.exch + 1ull;
  if (atomicExch(&d->last_bad_exch, ex1) != ex1) {
    atomicAdd(&d->bad_exch, 1ull);
    atomicAdd(&d->layer_exch[a.layer], 1ull);
  }
  const unsigned chunk = (unsigned)(v >> 4);
  if (atomicExch(&a.abmark[(size_t)chunk * HX_MAX_SERVERS + (size_t)s], ex1) ==
      ex1)
    return;
  const dvu64 ticket = atomicAdd(&d->ab_nrec, 1ull);
  if (ticket >= (dvu64)kDvAbRecs) return;
  DvAbRec* r = a.ab + ticket;
  r->exch = a.exch;
  r->epoch = epoch;
  r->v = (dvu64)v;
  r->layer = (uint32_t)a.layer;
  r->T = (uint32_t)a.T;
  r->slot = (uint32_t)(a.layer % (int)c->nslots);
  r->srv = (uint32_t)s;
  r->chunk = chunk;
  r->cat = (uint32_t)cat;
  r->a_prev = a_prev ? 1u : 0u;
  r->b_prev = b_prev ? 1u : 0u;
  r->injected = injected;
  dv_put(r->a, A);
  dv_put(r->b, Bv);
  dv_put(r->want, hs);
  dv_put(r->prev, pv);
  dv_spin_us(c, a.reread_us);
  const uint4* ry = reinterpret_cast<const uint4*>(
      hx_reply_y(c, s, a.layer % (int)c->nslots));
  dv_put(r->later, dv_ld_atomic(ry + v));
  __threadfence();
  __hip_atomic_store(&r->valid, 1ull, __ATOMIC_RELEASE,
                     __HIP_MEMORY_SCOPE_SYSTEM);
}

// hx_k_combine (same waits, abort handling and status) with every payload
// vector read twice back to back: A = plain uint4 load as in production, B =
// two relaxed system-scope atomic 8 B loads. The output is summed from A, so
// the normal check still judges production-style reads; A and B are classified
// against the pattern.
__global__ void __launch_bounds__(HX_COMBINE_THREADS)
    k_dv_combine_ab(DvArgs a) {
  __shared__ int s_bad;
  constexpr int NT = HX_COMBINE_THREADS;
  HxCtx* c = a.c;
  const int t = (int)threadIdx.x, b = (int)blockIdx.x;
  const int nsrv = (int)c->nsrv, layer = a.layer;
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
  const long long nvec = (long long)a.T * (long long)c->hidden / 8;
  const long long gid = (long long)b * NT + t;
  const long long gstride = (long long)a.nblocks * NT;
  if (s_bad) {
    uint4 nan;
    nan.x = nan.y = nan.z = nan.w = HX_NAN_HALF2;
    for (long long v = gid; v < nvec; v += gstride) a.out[v] = nan;
    return;
  }
  const uint64_t epoch = c->status[HX_ST_EPOCH];
  const uint4* ys[HX_MAX_SERVERS];
  uint64_t base[HX_MAX_SERVERS];
#pragma unroll
  for (int s = 0; s < HX_MAX_SERVERS; s++)
    if (s < nsrv) {
      ys[s] = reinterpret_cast<const uint4*>(hx_reply_y(c, s, slot));
      base[s] = dv_base(a.seed, epoch, (uint32_t)layer, (uint32_t)s);
    }
  for (long long v = gid; v < nvec; v += gstride) {
    uint4 pa[HX_MAX_SERVERS + 1], pb[HX_MAX_SERVERS], hs[HX_MAX_SERVERS];
    pa[0] = a.yh[v];
#pragma unroll
    for (int s = 0; s < HX_MAX_SERVERS; s++)
      if (s < nsrv) {
        pa[s + 1] = ys[s][v];
        pb[s] = dv_ld_atomic(ys[s] + v);
      }
#pragma unroll
    for (int s = 0; s < HX_MAX_SERVERS; s++)
      if (s < nsrv) hs[s] = dv_vec(base[s], (uint64_t)v);
    const bool inj = a.inj_chunk >= 0 && (int)(v >> 4) == a.inj_chunk;
    if (inj) {
      uint64_t pe = 0;
      int pl = 0;
      const bool hp =
          dv_prev_occ(epoch, layer, (int)c->nslots, (int)c->nlayers, &pe, &pl);
#pragma unroll
      for (int s = 0; s < HX_MAX_SERVERS; s++)
        if (s < nsrv && s == a.inj_srv) {
          const uint4 pv =
              hp ? dv_vec(dv_base(a.seed, pe, (uint32_t)pl, (uint32_t)s),
                          (uint64_t)v)
                 : make_uint4(0u, 0u, 0u, 0u);
          if (a.inj_kind != 2) pa[s + 1] = pv;
          if (a.inj_kind == 2) pb[s] = pv;
          if (a.inj_kind == 3)
            pb[s] = make_uint4(hs[s].x ^ 0x00010001u, hs[s].y ^ 0x00010001u,
                               hs[s].z ^ 0x00010001u, hs[s].w ^ 0x00010001u);
        }
    }
    a.out[v] = dv_sum_dev(pa, nsrv + 1);
#pragma unroll
    for (int s = 0; s < HX_MAX_SERVERS; s++)
      if (s < nsrv && (!dv_eq(pa[s + 1], hs[s]) || !dv_eq(pb[s], hs[s])))
        dv_ab_bad(a, s, v, pa[s + 1], pb[s], hs[s],
                  inj && s == a.inj_srv ? 1u : 0u);
  }
  if (t == 0 && b == 0) c->status[HX_ST_COMBINES] += 1ull;
}

// Positive controls. Server: chunk of the staged reply := previous occupant of
// the reply slot (a persistent stale payload). Hub: chunk of the output
// corrupted after the combine (replies fine, output wrong).
__global__ void __launch_bounds__(16)
    k_dv_inject_stale(const HxCtx* __restrict__ c, uint4* __restrict__ y,
                      dvu64 seed, int T, int layer, int chunk) {
  const long long nvec = (long long)T * (long long)c->hidden / 8;
  const long long v = (long long)chunk * 16 + (long long)threadIdx.x;
  if (v >= nvec || hx_aborted(c)) return;
  uint64_t pe = 0;
  int pl = 0;
  y[v] = dv_prev_occ(c->status[HX_ST_EPOCH], layer, (int)c->nslots,
                     (int)c->nlayers, &pe, &pl)
             ? dv_vec(dv_base(seed, pe, (uint32_t)pl, c->index), (uint64_t)v)
             : make_uint4(0u, 0u, 0u, 0u);
}

__global__ void __launch_bounds__(16)
    k_dv_inject_out(const HxCtx* __restrict__ c, uint4* __restrict__ out, int T,
                    int chunk) {
  const long long nvec = (long long)T * (long long)c->hidden / 8;
  const long long v = (long long)chunk * 16 + (long long)threadIdx.x;
  if (v >= nvec || hx_aborted(c)) return;
  uint4 x = out[v];
  x.x ^= 0x00010001u;
  x.y ^= 0x00010001u;
  x.z ^= 0x00010001u;
  x.w ^= 0x00010001u;
  out[v] = x;
}

// Start-up check of the pattern and of the sum arithmetic against the host.
__global__ void k_dv_selftest(uint4* vec, uint4* sum, dvu64 seed, int n) {
  const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;
  const int ns = 1 + i % HX_MAX_SERVERS;
  const uint64_t e = 1ull + (uint64_t)(i / 7);
  const uint32_t l = (uint32_t)(i % HX_MAX_LAYERS);
  const uint64_t v = (uint64_t)i * 977ull + (uint64_t)(i % 13);
  uint4 p[HX_MAX_SERVERS + 1];
  p[0] = dv_vec(dv_base(seed, e, l, (uint32_t)kDvHub), v);
  for (int s = 0; s < ns; s++)
    p[s + 1] = dv_vec(dv_base(seed, e, l, (uint32_t)s), v);
  vec[i] = p[1 + i % ns];
  sum[i] = dv_sum_dev(p, ns + 1);
}

// ---------------------------------------------------------------------------
// HIP setup: device, clock, allocation, ctx
// ---------------------------------------------------------------------------
[[noreturn]] static void hip_fail(const char* what, hipError_t e,
                                  const char* file, int line) {
  say("HIP error %s (%d) in %s at %s:%d", hipGetErrorName(e), (int)e, what,
      file, line);
  finish(kExitSetup, "hip_error");
}
#define HCK(call)                                                        \
  do {                                                                   \
    const hipError_t e_hck = (call);                                     \
    if (e_hck != hipSuccess) hip_fail(#call, e_hck, __FILE__, __LINE__); \
  } while (0)

struct Proc {
  hipStream_t st = nullptr, side = nullptr;
  HxCtx* d_ctx = nullptr;
  HxCtx img;
  uint8_t* own = nullptr;
  size_t own_bytes = 0;
  hipIpcMemHandle_t own_handle;
  uint8_t* peer[HX_MAX_SERVERS] = {};
  void* pin = nullptr;
  std::string bus;
  int wall_khz = 0;
  unsigned desc_next = 0;
};
static Proc P;

struct Geo {
  uint64_t x_bytes, ids_bytes, w_bytes, off_ids, off_w, slot_stride;
  uint64_t inbox_bytes, y_stride, reply_bytes;
};
static uint64_t align_up(uint64_t n, uint64_t a) { return (n + a - 1) / a * a; }
static Geo make_geo() {
  Geo g;
  g.x_bytes = (uint64_t)O.rows_cap * (uint64_t)O.hidden * 2;
  g.ids_bytes = (uint64_t)O.rows_cap * (uint64_t)O.topk * 2;
  g.w_bytes = (uint64_t)O.rows_cap * (uint64_t)O.topk * 4;
  g.off_ids = g.x_bytes;
  g.off_w = g.x_bytes + align_up(g.ids_bytes, 16);
  g.slot_stride = align_up(g.off_w + g.w_bytes, HX_PAGE);
  g.inbox_bytes = HX_PAGE + (uint64_t)O.nslots * g.slot_stride;
  g.y_stride = align_up(g.x_bytes, HX_PAGE);
  g.reply_bytes = HX_PAGE + (uint64_t)O.nsrv * (uint64_t)O.nslots * g.y_stride;
  return g;
}
static Geo GEO;

template <class T>
static T* upload(const std::vector<T>& v) {
  T* d = nullptr;
  HCK(hipMalloc((void**)&d, v.size() * sizeof(T)));
  HCK(hipMemcpy(d, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
  return d;
}

static void device_init() {
  int n = 0;
  HCK(hipGetDeviceCount(&n));
  const char* vis = getenv("HIP_VISIBLE_DEVICES");
  if (O.device >= n)
    die_setup(
        "--device %d but this process sees %d device(s) "
        "(HIP_VISIBLE_DEVICES=%s)",
        O.device, n, vis ? vis : "unset");
  HCK(hipSetDevice(O.device));
  hipDeviceProp_t pr;
  HCK(hipGetDeviceProperties(&pr, O.device));
  snprintf(G.arch, sizeof G.arch, "%s", pr.gcnArchName);
  char bus[64] = {0};
  HCK(hipDeviceGetPCIBusId(bus, sizeof bus, O.device));
  P.bus = bus;
  if (hipDeviceGetAttribute(&P.wall_khz, hipDeviceAttributeWallClockRate,
                            O.device) != hipSuccess) {
    (void)hipGetLastError();
    P.wall_khz = 0;
  }
  HCK(hipStreamCreateWithFlags(&P.st, hipStreamNonBlocking));
  HCK(hipStreamCreateWithFlags(&P.side, hipStreamNonBlocking));
  HCK(hipHostMalloc(&P.pin, 4096, hipHostMallocDefault));
  memset(P.pin, 0, 4096);
  say("device %d of %d (HIP_VISIBLE_DEVICES=%s): %s bus %s arch %s", O.device,
      n, vis ? vis : "unset", pr.name, bus, G.arch);
}

// Wall-clock ticks per ms against the host clock: kernel stamp, 150 ms host
// sleep, kernel stamp; median of 3.
static void calibrate() {
  long long* d = nullptr;
  HCK(hipMalloc((void**)&d, 16));
  hipStream_t s;
  HCK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  hipLaunchKernelGGL(k_wall, dim3(1), dim3(64), 0, s, d);
  HCK(hipStreamSynchronize(s));
  double r[3];
  for (int i = 0; i < 3; i++) {
    hipLaunchKernelGGL(k_wall, dim3(1), dim3(64), 0, s, d);
    HCK(hipStreamSynchronize(s));
    const int64_t h0 = now_ns();
    sleep_us(150000);
    hipLaunchKernelGGL(k_wall, dim3(1), dim3(64), 0, s, d + 1);
    HCK(hipStreamSynchronize(s));
    const int64_t h1 = now_ns();
    long long v[2];
    HCK(hipMemcpy(v, d, 16, hipMemcpyDeviceToHost));
    r[i] = (double)(v[1] - v[0]) / ((double)(h1 - h0) / 1e6);
  }
  HCK(hipStreamDestroy(s));
  HCK(hipFree(d));
  std::sort(r, r + 3);
  G.tpm = llround(r[1]);
  say("wall clock: attribute %d kHz, measured %.3f ticks/ms (median of 3 x "
      "150 ms), using %lld",
      P.wall_khz, r[1], G.tpm);
  if (G.tpm <= 0) die_setup("wall clock calibration failed");
  if (P.wall_khz > 0 && fabs(r[1] - P.wall_khz) > 0.05 * P.wall_khz)
    say("WARNING: measured tick rate differs from the attribute by > 5%%");
  if (O.wait_ms > INT64_MAX / G.tpm || O.begin_wait_ms > INT64_MAX / G.tpm)
    die_setup("--wait-ms/--begin-wait-ms overflow the tick counter");
}

// fp16 conversion on the device against the hand-written host codec over the
// grid the test uses (multiples of 1/128, |v| <= 1093) plus tie cases.
static void device_cvt_selftest() {
  std::vector<float> in;
  for (int q = -140000; q <= 140000; q++) in.push_back((float)q / 128.0f);
  Rng rng(99);
  for (int i = 0; i < 20000; i++)
    in.push_back((float)((int)rng.below(2000001) - 1000000) / 1024.0f);
  const int n = (int)in.size();
  float* d_in = nullptr;
  uint16_t* d_out = nullptr;
  HCK(hipMalloc((void**)&d_in, (size_t)n * 4));
  HCK(hipMalloc((void**)&d_out, (size_t)n * 2));
  HCK(hipMemcpy(d_in, in.data(), (size_t)n * 4, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(k_cvt, dim3((unsigned)((n + 255) / 256)), dim3(256), 0,
                     P.st, d_in, d_out, n);
  HCK(hipGetLastError());
  std::vector<uint16_t> out((size_t)n);
  HCK(hipStreamSynchronize(P.st));
  HCK(hipMemcpy(out.data(), d_out, (size_t)n * 2, hipMemcpyDeviceToHost));
  HCK(hipFree(d_in));
  HCK(hipFree(d_out));
  int bad = 0;
  for (int i = 0; i < n; i++) {
    if (out[(size_t)i] != f2h(in[(size_t)i]) && bad++ < 5)
      say("fp16 selftest: %.9g device 0x%04x host 0x%04x",
          (double)in[(size_t)i], out[(size_t)i], f2h(in[(size_t)i]));
  }
  if (bad)
    die_setup(
        "fp16 conversion: %d of %d values differ between the "
        "device and the host reference",
        bad, n);
  say("fp16 selftest: %d values identical on the device", n);
}

// Exported zeroed region. No fallback: a failing allocation is an error; a
// coarse region is used only when --alloc coarse asks for it.
static void alloc_region(uint64_t bytes) {
  static const char* names[] = {"coarse", "uncached", "finegrained"};
  const int kind = O.alloc == "coarse" ? 0 : O.alloc == "finegrained" ? 2 : 1;
  void* p = nullptr;
  const hipError_t e =
      kind == 0 ? hipMalloc(&p, (size_t)bytes)
                : hipExtMallocWithFlags(&p, (size_t)bytes,
                                        kind == 1 ? hipDeviceMallocUncached
                                                  : hipDeviceMallocFinegrained);
  if (e != hipSuccess)
    die_setup("%s allocation of %llu bytes failed: %s", names[kind],
              (unsigned long long)bytes, hipGetErrorString(e));
  HCK(hipMemset(p, 0, (size_t)bytes));
  HCK(hipDeviceSynchronize());
  HCK(hipIpcGetMemHandle(&P.own_handle, p));
  P.own = (uint8_t*)p;
  P.own_bytes = (size_t)bytes;
  snprintf(G.alloc, sizeof G.alloc, "%s", names[kind]);
  if (kind == 0)
    say("NOTE: --alloc coarse: peer stores may be served stale from this "
        "GPU's L2, the coherence assumption of the protocol is not met "
        "(comparison runs only)");
  say("exported region: %llu bytes %s", (unsigned long long)bytes, names[kind]);
}

static uint8_t* open_peer(const std::string& js, const char* who) {
  hipIpcMemHandle_t h;
  if (!hex_decode(js_str(js, "handle"), &h, sizeof h))
    die_setup("%s published a malformed IPC handle", who);
  void* p = nullptr;
  const hipError_t e =
      hipIpcOpenMemHandle(&p, h, hipIpcMemLazyEnablePeerAccess);
  if (e != hipSuccess)
    die_setup(
        "hipIpcOpenMemHandle of %s failed: %s (peer arch %s bus %s; "
        "check HIP_VISIBLE_DEVICES, try --visible shared)",
        who, hipGetErrorString(e), js_str(js, "gcn").c_str(),
        js_str(js, "bus").c_str());
  if (js_str(js, "alloc") == "coarse")
    say("WARNING: %s exported a COARSE allocation", who);
  return (uint8_t*)p;
}

static std::string plan_hash_hex() {
  char b[24];
  snprintf(b, sizeof b, "%016llx", (unsigned long long)O.plan_hash);
  return b;
}

static void check_peer_info(const std::string& js, const char* who,
                            bool from_hub) {
  struct F {
    const char* k;
    long long mine;
  } f[] = {{"rows_cap", O.rows_cap},
           {"topk", O.topk},
           {"hidden", O.hidden},
           {"nslots", O.nslots},
           {"num_experts", O.experts},
           {"nlayers", O.layers},
           {"nsrv", O.nsrv},
           {"slot_stride", (long long)GEO.slot_stride},
           {"off_ids", (long long)GEO.off_ids},
           {"off_w", (long long)GEO.off_w},
           {"y_stride", (long long)GEO.y_stride}};
  for (const F& x : f) {
    const long long v = js_int(js, x.k, -1);
    if (v != x.mine)
      die_setup("%s published %s=%lld but this process has %lld", who, x.k, v,
                x.mine);
  }
  if (js_str(js, "plan_hash") != plan_hash_hex())
    die_setup("%s has a different plan_hash", who);
  const long long want =
      from_hub ? (long long)GEO.reply_bytes : (long long)GEO.inbox_bytes;
  if (js_int(js, "bytes", 0) < want)
    die_setup("%s exported %lld bytes, the layout needs %lld", who,
              js_int(js, "bytes", 0), want);
}

static std::string own_info(bool hub) {
  Json j;
  const std::string ph = plan_hash_hex();
  j.kv("role", hub ? "hub" : "srv")
      .kv("index", hub ? O.hub_index : O.index)
      .kv("pid", (long long)getpid())
      .kv("dev", O.device)
      .kv("bus", P.bus)
      .kv("gcn", G.arch)
      .kv("alloc", G.alloc)
      .kv("bytes", (long long)P.own_bytes)
      .kv("handle", hex_encode(&P.own_handle, sizeof P.own_handle))
      .kv("rows_cap", O.rows_cap)
      .kv("topk", O.topk)
      .kv("hidden", O.hidden)
      .kv("nslots", O.nslots)
      .kv("num_experts", O.experts)
      .kv("nlayers", O.layers)
      .kv("nsrv", O.nsrv)
      .kv("slot_stride", (long long)GEO.slot_stride)
      .kv("off_ids", (long long)GEO.off_ids)
      .kv("off_w", (long long)GEO.off_w)
      .kv("y_stride", (long long)GEO.y_stride)
      .kv("plan_hash", ph)
      .kv("mono_ns", (long long)now_ns());
  return j.str();
}

// rd-mode 4 is the loopback-local combine: the ctx keeps the plain read mode.
static uint32_t ctx_flush_mode() {
  const uint32_t rd = O.rd_mode == 4 ? 0u : (uint32_t)O.rd_mode;
  return ((uint32_t)O.flush & 15u) | (rd << HX_RD_SHIFT) |
         (O.digest ? (uint32_t)HX_DIGEST : 0u);
}

static std::string flush_names(int f) {
  static const struct {
    int bit;
    const char* name;
  } kBits[] = {{HX_FLUSH_CTRL, "ctrl"},
               {HX_FLUSH_PAYLOAD, "payload"},
               {HX_FLUSH_ATOMIC, "atomic"},
               {HX_SETTLE, "settle"}};
  std::string s;
  for (const auto& b : kBits) {
    if (!(f & b.bit)) continue;
    if (!s.empty()) s += '|';
    s += b.name;
  }
  return s.empty() ? "none" : s;
}

static void fill_ctx_common(uint32_t role, uint32_t index) {
  HxCtx& c = P.img;
  memset(&c, 0, sizeof c);
  c.magic = HX_MAGIC;
  c.role = role;
  c.version = HX_VERSION;
  c.plan_hash = O.plan_hash;
  c.index = index;
  c.nsrv = (uint32_t)O.nsrv;
  c.rows_cap = (uint32_t)O.rows_cap;
  c.topk = (uint32_t)O.topk;
  c.hidden = (uint32_t)O.hidden;
  c.nslots = (uint32_t)O.nslots;
  c.num_experts = (uint32_t)O.experts;
  c.nlayers = (uint32_t)O.layers;
  c.flush_mode = ctx_flush_mode();
  c.wait_ticks = O.wait_ms * G.tpm;
  c.begin_wait_ticks = O.begin_wait_ms * G.tpm;
  c.ticks_per_ms = G.tpm;
  c.slot_stride = GEO.slot_stride;
  c.off_ids = GEO.off_ids;
  c.off_w = GEO.off_w;
  c.y_stride = GEO.y_stride;
  c.own_base = P.own;
}

static void upload_ctx() {
  HCK(hipMalloc((void**)&P.d_ctx, HX_CTX_WORDS * 8));
  HCK(hipMemset(P.d_ctx, 0, HX_CTX_WORDS * 8));
  HCK(hipMemcpy(P.d_ctx, &P.img, sizeof(HxCtx), hipMemcpyHostToDevice));
}

// Status block + abort word of this process's ctx via the side stream.
struct Snap {
  uint64_t st[8];
  uint32_t abort_word;
};
static Snap read_snap() {
  Snap s;
  uint8_t* base = reinterpret_cast<uint8_t*>(P.d_ctx);
  uint8_t* pin = static_cast<uint8_t*>(P.pin);
  HCK(hipMemcpyAsync(pin, base + offsetof(HxCtx, status), 64,
                     hipMemcpyDeviceToHost, P.side));
  HCK(hipMemcpyAsync(pin + 64, base + offsetof(HxCtx, abort_word), 4,
                     hipMemcpyDeviceToHost, P.side));
  HCK(hipStreamSynchronize(P.side));
  memcpy(s.st, pin, 64);
  memcpy(&s.abort_word, pin + 64, 4);
  return s;
}
static void dump_status(const Snap& s) {
  say("status: record=0x%llx epoch=%llu live=0x%llx steps=%llu dispatches=%llu "
      "combines=%llu pairs_hub=%llu pairs_remote=%llu abort_word=%u",
      (unsigned long long)s.st[HX_ST_RECORD],
      (unsigned long long)s.st[HX_ST_EPOCH],
      (unsigned long long)s.st[HX_ST_LIVE],
      (unsigned long long)s.st[HX_ST_STEPS],
      (unsigned long long)s.st[HX_ST_DISPATCHES],
      (unsigned long long)s.st[HX_ST_COMBINES],
      (unsigned long long)s.st[HX_ST_PAIRS_HUB],
      (unsigned long long)s.st[HX_ST_PAIRS_REMOTE], s.abort_word);
}

// Digest mode: this process's counters (HX_DS_*), one DIGSTAT line.
static void print_digstat() {
  if (P.d_ctx == nullptr || !(ctx_flush_mode() & HX_DIGEST)) return;
  uint8_t* pin = static_cast<uint8_t*>(P.pin) + 2048;
  HCK(hipMemcpyAsync(
      pin, reinterpret_cast<uint8_t*>(P.d_ctx) + offsetof(HxCtx, dig_stat),
      HX_DS_WORDS * 8, hipMemcpyDeviceToHost, P.side));
  HCK(hipStreamSynchronize(P.side));
  uint64_t w[HX_DS_WORDS];
  memcpy(w, pin, sizeof w);
  typedef unsigned long long ull;
  say("DIGSTAT role=%s index=%d ok=%llu bad=%llu healed=%llu unhealed=%llu "
      "late=%llu rescans=%llu last_epoch=%llu last_layer=%llu last_peer=%llu "
      "last_rescans=%llu last_expected=0x%08llx last_got=0x%08llx",
      G.role, G.index, (ull)w[HX_DS_OK], (ull)w[HX_DS_BAD],
      (ull)w[HX_DS_HEALED], (ull)w[HX_DS_UNHEALED], (ull)w[HX_DS_LATE],
      (ull)w[HX_DS_RESCANS], (ull)(w[HX_DS_LAST_A] >> 32),
      (ull)((w[HX_DS_LAST_A] >> 16) & 0xFFFF),
      (ull)((w[HX_DS_LAST_A] >> 8) & 0xFF), (ull)(w[HX_DS_LAST_A] & 0xFF),
      (ull)(w[HX_DS_LAST_B] >> 32), (ull)(w[HX_DS_LAST_B] & 0xFFFFFFFFull));
}

// Fault-injection words of the next exchange (HX_FAULT_INJECT builds): the
// 256 B chunk `chunk` of the payload to server `srv` is written `us`
// microseconds after the flag of the exchange whose sequence is `seq32`.
static void inject_late_arm(uint32_t seq32, int chunk, int us, int srv) {
  static uint64_t w[4];
  w[0] = seq32;
  w[1] = (uint64_t)chunk;
  w[2] = (uint64_t)us;
  w[3] = (uint64_t)srv;
  HCK(hipMemcpyAsync(reinterpret_cast<uint8_t*>(P.d_ctx) + offsetof(HxCtx, dbg),
                     w, sizeof w, hipMemcpyHostToDevice, P.st));
}
[[noreturn]] static void handle_abort(const Snap& s0, const char* where) {
  Snap s = s0;
  if (s.st[HX_ST_RECORD] == 0 && s.abort_word != 0) {
    sleep_us(2000);
    s = read_snap();
  }
  const uint64_t rec = s.st[HX_ST_RECORD];
  G.abort_rec = rec;
  say("ABORT (%s): %s", where, rec_text(rec).c_str());
  dump_status(s);
  print_digstat();
  const int pd = G.peer_dead.load();
  if (pd >= 0) say("a peer death was seen by the pidfd watcher (peer %d)", pd);
  finish(kExitAbort, pd >= 0 ? "peer_death" : "abort");
}
static Snap check_abort(const char* where) {
  const Snap s = read_snap();
  if (s.abort_word != 0 || s.st[HX_ST_RECORD] != 0) handle_abort(s, where);
  return s;
}

template <class F>
static hipGraphExec_t capture_graph(F&& body) {
  HCK(hipStreamBeginCapture(P.st, hipStreamCaptureModeThreadLocal));
  body();
  hipGraph_t g = nullptr;
  HCK(hipStreamEndCapture(P.st, &g));
  hipGraphExec_t ge = nullptr;
  HCK(hipGraphInstantiateWithFlags(&ge, g, 0));
  HCK(hipGraphDestroy(g));
  return ge;
}

static void launch_synth(const void* x, const int32_t* ids, const float* w,
                         void* y, int T) {
  hipLaunchKernelGGL(k_synth,
                     dim3((unsigned)((O.hidden + 255) / 256), (unsigned)T),
                     dim3(256), 0, P.st, static_cast<const __half*>(x), ids, w,
                     static_cast<__half*>(y), O.topk, O.hidden);
}

// ---------------------------------------------------------------------------
// host watcher: pidfd on every peer; a dead peer pokes the own control page
// ---------------------------------------------------------------------------
struct Peer {
  int idx;
  long pid;
  int fd;
  std::string name;
};
static std::vector<Peer> g_peers;
static int g_wake[2] = {-1, -1};
static std::atomic<bool> g_stop_watch{false};
static std::thread g_wthread;

static int open_pidfd(long pid) {
#if defined(__linux__)
  #ifndef __NR_pidfd_open
    #define __NR_pidfd_open 434
  #endif
  return (int)syscall(__NR_pidfd_open, (pid_t)pid, 0);
#else
  (void)pid;
  errno = ENOSYS;
  return -1;
#endif
}

static void watcher_main() {
  HCK(hipSetDevice(O.device));
  hipStream_t ws;
  HCK(hipStreamCreateWithFlags(&ws, hipStreamNonBlocking));
  uint64_t* pin = nullptr;
  HCK(hipHostMalloc((void**)&pin, 64, hipHostMallocDefault));
  pin[0] = 1;
  uint64_t last_live = 0;
  bool fallback = false;
  std::vector<pollfd> pf;
  pf.push_back({g_wake[0], POLLIN, 0});
  for (const Peer& p : g_peers) {
    if (p.fd >= 0)
      pf.push_back({p.fd, POLLIN, 0});
    else
      fallback = true;
  }
  int dead = -1;
  while (dead < 0) {
    const int r = poll(pf.data(), (nfds_t)pf.size(), fallback ? 50 : 1000);
    if (g_stop_watch.load() || (r > 0 && pf[0].revents != 0)) return;
    if (r > 0)
      for (size_t i = 1; i < pf.size() && dead < 0; i++)
        if (pf[i].revents != 0)
          for (const Peer& p : g_peers)
            if (p.fd == pf[i].fd) dead = p.idx;
    for (const Peer& p : g_peers)
      if (dead < 0 && p.fd < 0 && !pid_alive(p.pid)) dead = p.idx;
    if (dead < 0 && r == 0) {
      // a hung wait shows up in status[LIVE] before it times out
      if (hipMemcpyAsync(pin + 2,
                         reinterpret_cast<uint8_t*>(P.d_ctx) +
                             offsetof(HxCtx, status) + 8 * HX_ST_LIVE,
                         8, hipMemcpyDeviceToHost, ws) == hipSuccess &&
          hipStreamSynchronize(ws) == hipSuccess && pin[2] != 0 &&
          pin[2] != last_live) {
        last_live = pin[2];
        say("live wait record: %s", rec_text(pin[2]).c_str());
      }
    }
  }
  G.t_detect_ns = now_ns();
  G.peer_dead = dead;
  say("peer %d died (pidfd): poking the host abort word", dead);
  const hipError_t e = hipMemcpyAsync(P.own + HX_CTRL_HOST_ABORT, pin, 8,
                                      hipMemcpyHostToDevice, ws);
  if (e != hipSuccess || hipStreamSynchronize(ws) != hipSuccess) {
    say("host abort poke failed: %s", hipGetErrorString(e));
    finish(kExitAbort, "poke_failed");
  }
  G.t_poke_ns = now_ns();
  for (int i = 0; i < 300 && G.finishing.load() == 0; i++) sleep_us(10000);
  if (G.finishing.load() == 0) {
    say("the main thread did not exit within 3 s of the poke: forcing exit");
    finish(kExitAbort, "watcher_forced");
  }
}

static void start_watcher() {
  if (O.no_watcher || g_peers.empty()) {
    say("host watcher: off");
    return;
  }
  if (pipe(g_wake) != 0) die_setup("pipe: %s", strerror(errno));
  for (Peer& p : g_peers) {
    p.fd = open_pidfd(p.pid);
    if (p.fd < 0 && errno == ESRCH) {
      G.peer_dead = p.idx;
      die_setup("peer %s (pid %ld) is already gone", p.name.c_str(), p.pid);
    }
    if (p.fd < 0)
      say("pidfd_open(%ld) failed (%s): polling /proc for %s", p.pid,
          strerror(errno), p.name.c_str());
  }
  G.watcher_on = true;
  g_wthread = std::thread(watcher_main);
  say("host watcher: on (%zu peer(s))", g_peers.size());
}

static void stop_watcher() {
  if (!g_wthread.joinable()) return;
  g_stop_watch = true;
  wr_all(g_wake[1], "x", 1);
  g_wthread.join();
}

// ---------------------------------------------------------------------------
// hub
// ---------------------------------------------------------------------------
struct LayerPtrs {
  const uint16_t* x;
  const int32_t* ids;
  const float* w;
  int32_t* hubids;
  uint16_t* y;
};

struct HubBufs {
  size_t nbuf = 1, tcap = 0;
  uint16_t *h_x = nullptr, *h_out = nullptr, *d_x = nullptr, *d_y = nullptr;
  int32_t *h_ids = nullptr, *h_hubids = nullptr, *d_ids = nullptr,
          *d_hubids = nullptr;
  float *h_w = nullptr, *d_w = nullptr;
  uint64_t *h_lstat = nullptr, *d_lstat = nullptr;
  std::vector<int8_t> xq;
};
static HubBufs B;
static Tables TAB;
static std::map<int, hipGraphExec_t> g_graphs;
static const size_t kLstatWords = (size_t)HX_MAX_LAYERS * HX_LSTAT_COLS;

static LayerPtrs hub_ptrs(size_t i) {
  const size_t H = (size_t)O.hidden, K = (size_t)O.topk;
  return {B.d_x + i * B.tcap * H, B.d_ids + i * B.tcap * K,
          B.d_w + i * B.tcap * K, B.d_hubids + i * B.tcap * K,
          B.d_y + i * B.tcap * H};
}

static void hub_enqueue_layer(const LayerPtrs& b, int layer, int T) {
  hx_launch_dispatch(P.d_ctx, b.x, b.ids, b.w, b.hubids, T, layer, O.nsrv,
                     O.hidden, O.blk_dispatch, P.st);
  launch_synth(b.x, b.hubids, b.w, b.y, T);
  hx_launch_combine(P.d_ctx, b.y, b.y, T, layer, O.hidden, O.blk_combine, P.st);
}
static int bucket_of(int T) { return O.eager ? 0 : T; }

// The per-layer pair counters of the dispatch kernel (ctx.layer_stats); the
// pointer must be in the ctx image before it is uploaded.
static void hub_alloc_lstat() {
  if (!O.layer_stats) return;
  HCK(hipMalloc((void**)&B.d_lstat, kLstatWords * 8));
  HCK(hipMemset(B.d_lstat, 0, kLstatWords * 8));
  HCK(hipHostMalloc((void**)&B.h_lstat, kLstatWords * 8, hipHostMallocDefault));
  P.img.layer_stats = B.d_lstat;
}

static void hub_alloc_buffers(size_t nbuf, size_t tcap) {
  B.nbuf = nbuf;
  B.tcap = tcap;
  const size_t nx = nbuf * tcap * (size_t)O.hidden;
  const size_t nk = nbuf * tcap * (size_t)O.topk;
  HCK(hipHostMalloc((void**)&B.h_x, nx * 2, hipHostMallocDefault));
  HCK(hipHostMalloc((void**)&B.h_out, nx * 2, hipHostMallocDefault));
  HCK(hipHostMalloc((void**)&B.h_ids, nk * 4, hipHostMallocDefault));
  HCK(hipHostMalloc((void**)&B.h_hubids, nk * 4, hipHostMallocDefault));
  HCK(hipHostMalloc((void**)&B.h_w, nk * 4, hipHostMallocDefault));
  HCK(hipMalloc((void**)&B.d_x, nx * 2));
  HCK(hipMalloc((void**)&B.d_y, nx * 2));
  HCK(hipMalloc((void**)&B.d_ids, nk * 4));
  HCK(hipMalloc((void**)&B.d_hubids, nk * 4));
  HCK(hipMalloc((void**)&B.d_w, nk * 4));
  HCK(hipMemset(B.d_y, 0, nx * 2));
  HCK(hipMemset(B.d_hubids, 0xFF, nk * 4));
  B.xq.assign(nx, 0);
}

// What the device must read after the exchanges so far.
struct Expect {
  uint64_t steps = 0, disp = 0, comb = 0, pairs_hub = 0, pairs_remote = 0;
  uint64_t ls[HX_MAX_LAYERS][HX_LSTAT_COLS] = {};
};

// 256 B chunks (byte offset in the buffer / 256) that hold a difference.
static std::vector<size_t> diff_chunks(const uint16_t* a, const uint16_t* b,
                                       size_t n) {
  std::vector<size_t> ch;
  for (size_t j = 0; j < n; j++)
    if (a[j] != b[j] && (ch.empty() || ch.back() != j / 128))
      ch.push_back(j / 128);
  return ch;
}

static std::string chunk_list(const std::vector<size_t>& ch) {
  std::string s = std::to_string(ch.size()) + " distinct 256 B chunk(s)";
  char buf[32];
  for (size_t k = 0; k < ch.size() && k < 8; k++) {
    snprintf(buf, sizeof buf, "%s0x%zx", k ? " " : ", first byte offsets ",
             ch[k] * 256);
    s += buf;
  }
  return ch.size() > 8 ? s + " ..." : s;
}

// Raw 256 B chunk (16 B per line), got | expected; * marks a differing line.
static void say_chunk_hex(const char* what, const uint16_t* got,
                          const uint16_t* want, size_t n, size_t chunk) {
  const size_t e0 = chunk * 128, e1 = std::min(n, e0 + 128);
  say("  %s: 256 B chunk at byte offset 0x%zx (elements %zu..%zu), got | "
      "expected, * = line differs",
      what, e0 * 2, e0, e1 - 1);
  for (size_t j = e0; j < e1; j += 8) {
    char g[48] = "", w[48] = "";
    bool d = false;
    for (size_t q = j; q < std::min(e1, j + 8); q++) {
      snprintf(g + strlen(g), sizeof g - strlen(g), "%s%04x", q > j ? " " : "",
               got[q]);
      snprintf(w + strlen(w), sizeof w - strlen(w), "%s%04x", q > j ? " " : "",
               want[q]);
      d |= got[q] != want[q];
    }
    say("    +0x%03zx %s | %s%s", (j - e0) * 2, g, w, d ? " *" : "");
  }
}

// The output of exchange i copied back again (synchronous hipMemcpy: pinned,
// pinned after ~1 ms, pageable): wrong in device memory, or only in the first
// host copy.
static void hub_reread(size_t i, int T, const uint16_t* first_copy,
                       const uint16_t* want) {
  const size_t n = (size_t)T * (size_t)O.hidden;
  static uint16_t* pin = nullptr;
  static std::vector<uint16_t> pageable;
  if (pin == nullptr)
    HCK(hipHostMalloc((void**)&pin, B.tcap * (size_t)O.hidden * 2,
                      hipHostMallocDefault));
  pageable.resize(n);
  const std::vector<size_t> ch0 = diff_chunks(first_copy, want, n);
  static const char* const kind[3] = {"pinned, immediate", "pinned, after 1 ms",
                                      "pageable"};
  int clean = 0, same = 0;
  for (int k = 0; k < 3; k++) {
    uint16_t* dst = k == 2 ? pageable.data() : pin;
    memset(dst, 0xFF, n * 2);
    if (k == 1) sleep_us(1000);
    HCK(hipMemcpy(dst, B.d_y + i * B.tcap * (size_t)O.hidden, n * 2,
                  hipMemcpyDeviceToHost));
    size_t first;
    const size_t bad = count_diff(dst, want, n, &first);
    const bool same_copy = memcmp(dst, first_copy, n * 2) == 0;
    const std::vector<size_t> ch = diff_chunks(dst, want, n);
    clean += bad == 0;
    same += same_copy;
    say("  re-read %d (%s): %zu of %zu elements differ from the reference, "
        "%zu chunk(s)%s; identical to the first host copy: %s",
        k + 1, kind[k], bad, n, ch.size(),
        bad && ch == ch0 ? " (the same chunks as the first copy)" : "",
        same_copy ? "yes" : "no");
  }
  if (clean == 3)
    say("  re-read verdict: the device buffer is correct now; the first host "
        "copy was wrong (D2H copy) or the buffer was wrong only transiently");
  else if (same == 3)
    say("  re-read verdict: PERSISTENT in device memory (every re-read equals "
        "the first copy): the combine output itself is wrong");
  else
    say("  re-read verdict: CHANGING (the re-reads differ from the first copy "
        "or from each other)");
}

static void hub_dump_servers(int layer, int T, const RefTabs& rt,
                             const int8_t* xq, size_t chunk) {
  const int slot = layer % O.nslots;
  const size_t n = (size_t)T * O.hidden;
  std::vector<uint16_t> got(n), want(n);
  for (int s = 0; s < O.nsrv; s++) {
    const uint8_t* src =
        P.own + HX_PAGE + (size_t)(s * O.nslots + slot) * GEO.y_stride;
    if (hipMemcpy(got.data(), src, n * 2, hipMemcpyDeviceToHost) != hipSuccess)
      continue;
    ref_fill(rt.ytab.data() + (size_t)(1 + s) * T * 64, xq, T, O.hidden,
             want.data());
    size_t first;
    const size_t bad = count_diff(got.data(), want.data(), n, &first);
    if (bad == 0) {
      say("  reply of server %d: identical to the reference", s);
      continue;
    }
    say("  reply of server %d: %zu of %zu differ, first (r=%zu,c=%zu) got "
        "0x%04x expected 0x%04x",
        s, bad, n, first / O.hidden, first % O.hidden, got[first], want[first]);
    say("  reply of server %d differs in %s", s,
        chunk_list(diff_chunks(got.data(), want.data(), n)).c_str());
    char what[32];
    snprintf(what, sizeof what, "reply of server %d", s);
    say_chunk_hex(what, got.data(), want.data(), n, chunk);
  }
}

// Cumulative per-layer pair counters against the reference, once per step.
static uint64_t hub_check_lstat(const Expect& ex, long long step) {
  static const char* const col[HX_LSTAT_COLS] = {"hub", "cold", "spill",
                                                 "lost"};
  HCK(hipMemcpyAsync(B.h_lstat, B.d_lstat, kLstatWords * 8,
                     hipMemcpyDeviceToHost, P.side));
  HCK(hipStreamSynchronize(P.side));
  uint64_t bad = 0;
  for (int l = 0; l < O.layers; l++)
    for (int c = 0; c < HX_LSTAT_COLS; c++) {
      const uint64_t got = B.h_lstat[(size_t)l * HX_LSTAT_COLS + (size_t)c];
      if (got == ex.ls[l][c]) continue;
      if (bad++ < 8)
        say("MISMATCH layer_stats[layer %d].%s: device %llu expected %llu "
            "(after step %lld)",
            l, col[c], (unsigned long long)got, (unsigned long long)ex.ls[l][c],
            step);
    }
  return bad;
}

// One batch = `cnt` consecutive layers from l0 of a step of nl layers: one
// layer in eager mode, the whole step in graph mode.
static void hub_batch(long long step, int T, int l0, int cnt, int nl,
                      uint64_t exch0, Expect& ex, uint64_t* verified) {
  const size_t H = (size_t)O.hidden, K = (size_t)O.topk, tc = B.tcap;
  static std::vector<Plan> plans;
  static std::vector<RefTabs> rts;
  static std::vector<std::vector<uint16_t>> refout;
  plans.resize((size_t)cnt);
  rts.resize((size_t)cnt);
  refout.resize((size_t)cnt);
  std::vector<char> full((size_t)cnt, 0);
  for (int i = 0; i < cnt; i++) {
    const size_t o = (size_t)i;
    gen_inputs(O.seed, (uint64_t)step, l0 + i, T, O.hidden, O.topk, O.experts,
               B.h_x + o * tc * H, B.xq.data() + o * tc * H,
               B.h_ids + o * tc * K, B.h_w + o * tc * K);
    const int ve = verify_every_for(T);
    full[o] = ve > 0 && (exch0 + o) % (uint64_t)ve == 0;
  }
  const size_t cn = (size_t)cnt;
  HCK(hipMemcpy2DAsync(B.d_x, tc * H * 2, B.h_x, tc * H * 2, (size_t)T * H * 2,
                       cn, hipMemcpyHostToDevice, P.st));
  HCK(hipMemcpy2DAsync(B.d_ids, tc * K * 4, B.h_ids, tc * K * 4,
                       (size_t)T * K * 4, cn, hipMemcpyHostToDevice, P.st));
  HCK(hipMemcpy2DAsync(B.d_w, tc * K * 4, B.h_w, tc * K * 4, (size_t)T * K * 4,
                       cn, hipMemcpyHostToDevice, P.st));
  if (O.graph) {
    HCK(hipGraphLaunch(g_graphs.at(T), P.st));
  } else {
    if (l0 == 0) hx_launch_begin_step(P.d_ctx, T, bucket_of(T), P.st);
    hub_enqueue_layer(hub_ptrs(0), l0, T);
  }
  HCK(hipGetLastError());
  HCK(hipMemcpy2DAsync(B.h_hubids, tc * K * 4, B.d_hubids, tc * K * 4,
                       (size_t)T * K * 4, cn, hipMemcpyDeviceToHost, P.st));
  for (size_t i = 0; i < cn; i++)
    if (full[i])
      HCK(hipMemcpyAsync(B.h_out + i * tc * H, B.d_y + i * tc * H,
                         (size_t)T * H * 2, hipMemcpyDeviceToHost, P.st));
  // CPU model of the batch while the GPU works
  uint64_t sum_hub = 0, sum_remote = 0;
  for (size_t i = 0; i < cn; i++) {
    const int layer = l0 + (int)i;
    const int32_t* ids = B.h_ids + i * tc * K;
    ref_plan(TAB, layer, T, O.topk, ids, plans[i]);
    if (plans[i].plan_error)
      die_setup(
          "the reference predicts a plan error (layer %d): cannot "
          "happen with the synthetic tables",
          layer);
    sum_hub += plans[i].n_hub;
    sum_remote += plans[i].n_remote;
    ex.ls[layer][HX_LSTAT_HUB] += plans[i].n_hub;
    ex.ls[layer][HX_LSTAT_COLD] += plans[i].n_cold;
    ex.ls[layer][HX_LSTAT_SPILL] += plans[i].n_spill;
    ex.ls[layer][HX_LSTAT_LOST] += plans[i].n_lost;
    if (!full[i]) continue;
    refout[i].resize((size_t)T * H);
    ref_tabs(plans[i], ids, B.h_w + i * tc * K, T, O.topk, O.nsrv, rts[i]);
    if (O.ref_slow)
      ref_out_slow(plans[i], ids, B.h_w + i * tc * K, B.h_x + i * tc * H, T,
                   O.hidden, O.topk, O.nsrv, refout[i].data());
    else
      ref_fill(rts[i].otab.data(), B.xq.data() + i * tc * H, T, O.hidden,
               refout[i].data());
  }
  HCK(hipStreamSynchronize(P.st));
  const Snap s = check_abort(O.graph ? "after the step" : "after the exchange");
  ex.steps += l0 == 0 ? 1 : 0;
  ex.disp += cn;
  ex.comb += cn;
  ex.pairs_hub += sum_hub;
  ex.pairs_remote += sum_remote;
  uint64_t bad_batch = 0;
  const struct {
    const char* name;
    uint64_t got, want;
  } cnt_chk[] = {{"epoch", s.st[HX_ST_EPOCH], (uint64_t)step + 1},
                 {"steps", s.st[HX_ST_STEPS], ex.steps},
                 {"dispatches", s.st[HX_ST_DISPATCHES], ex.disp},
                 {"combines", s.st[HX_ST_COMBINES], ex.comb},
                 {"pairs_hub", s.st[HX_ST_PAIRS_HUB], ex.pairs_hub},
                 {"pairs_remote", s.st[HX_ST_PAIRS_REMOTE], ex.pairs_remote}};
  for (const auto& c : cnt_chk)
    if (c.got != c.want) {
      say("MISMATCH status.%s: device %llu expected %llu (step %lld, layers "
          "%d..%d)",
          c.name, (unsigned long long)c.got, (unsigned long long)c.want, step,
          l0, l0 + cnt - 1);
      bad_batch++;
    }
  if (O.layer_stats && l0 + cnt == nl) bad_batch += hub_check_lstat(ex, step);
  for (size_t i = 0; i < cn; i++) {
    const int layer = l0 + (int)i;
    const uint64_t exch = exch0 + i;
    size_t first;
    const size_t nbad_ids =
        count_diff32(B.h_hubids + i * tc * K, plans[i].hub_ids.data(),
                     (size_t)T * K, &first);
    if (nbad_ids) {
      const size_t p = first;
      say("MISMATCH hub_ids_out: step %lld layer %d T %d exch %llu: %zu of %zu "
          "differ, first (r=%zu,k=%zu) device %d expected %d",
          step, layer, T, (unsigned long long)exch, nbad_ids, (size_t)T * K,
          p / K, p % K, B.h_hubids[i * tc * K + p], plans[i].hub_ids[p]);
      bad_batch++;
    }
    if (!full[i]) continue;
    (*verified)++;
    const size_t n = (size_t)T * H;
    const size_t nbad =
        count_diff(B.h_out + i * tc * H, refout[i].data(), n, &first);
    if (nbad == 0) continue;
    const uint16_t* got = B.h_out + i * tc * H;
    size_t rows_bad = 0, nan_bad = 0;
    for (int r = 0; r < T; r++) {
      bool rb = false;
      for (size_t c = 0; c < H; c++) {
        const size_t j = (size_t)r * H + c;
        if (got[j] != refout[i][j]) {
          rb = true;
          nan_bad += (got[j] & 0x7FFFu) > 0x7C00u;
        }
      }
      rows_bad += rb;
    }
    int cnt_dest[HX_MAX_SERVERS + 1] = {};
    for (int8_t d : plans[i].dest)
      if (d >= 0) cnt_dest[d]++;
    say("MISMATCH out: step %lld layer %d T %d exch %llu epoch %lld: %zu of "
        "%zu elements differ in %zu rows (%zu NaN), first (r=%zu,c=%zu) "
        "device 0x%04x expected 0x%04x",
        step, layer, T, (unsigned long long)exch, step + 1, nbad, n, rows_bad,
        nan_bad, first / H, first % H, got[first], refout[i][first]);
    say("  pairs: hub %d, servers %d %d %d %d %d %d %d %d", cnt_dest[0],
        cnt_dest[1], cnt_dest[2], cnt_dest[3], cnt_dest[4], cnt_dest[5],
        cnt_dest[6], cnt_dest[7], cnt_dest[8]);
    say("  differing outputs: %s",
        chunk_list(diff_chunks(got, refout[i].data(), n)).c_str());
    say_chunk_hex("output", got, refout[i].data(), n, first / 128);
    hub_reread(i, T, got, refout[i].data());
    if (!O.graph) hub_dump_servers(layer, T, rts[i], B.xq.data(), first / 128);
    bad_batch++;
  }
  if (bad_batch) {
    G.mismatches += bad_batch;
    dump_status(s);
    if (O.stop_on_mismatch) finish(kExitMismatch, "mismatch");
  }
}

static void hub_run_verify() {
  const long long total = O.iters;
  const long long nsteps = steps_total();
  uint64_t exch = 0, verified = 0;
  Expect ex;
  const int64_t t_start = now_ns();
  char ve[48];
  if (O.verify_every < 0)
    snprintf(ve, sizeof ve, "auto (1 for T <= 48, else 16)");
  else
    snprintf(ve, sizeof ve, "%d", O.verify_every);
  say("run: %lld exchanges in %lld steps, T list %zu entries, %s mode, "
      "verify-every %s",
      total, nsteps, O.tlist.size(), O.graph ? "graph" : "stream", ve);
  for (long long step = 0; step < nsteps; step++) {
    const int T = O.tlist[(size_t)step % O.tlist.size()];
    const int nl = layers_in_step(step);
    const int per = O.graph ? nl : 1;
    for (int l0 = 0; l0 < nl; l0 += per) {
      const uint64_t before = exch;
      hub_batch(step, T, l0, per, nl, exch, ex, &verified);
      exch += (uint64_t)per;
      G.iters = exch;
      G.steps = (uint64_t)step + 1;
      if (O.log_every > 0 &&
          exch / (uint64_t)O.log_every != before / (uint64_t)O.log_every)
        say("progress: %llu exchanges, %llu verified, %.0f/s",
            (unsigned long long)exch, (unsigned long long)verified,
            (double)exch / ((double)(now_ns() - t_start) / 1e9));
      if (O.hub_die_after >= 0 && exch >= (uint64_t)O.hub_die_after)
        die_now("die");
    }
  }
  say("run complete: %llu exchanges, %llu with a full output check, %llu "
      "mismatches",
      (unsigned long long)exch, (unsigned long long)verified,
      (unsigned long long)G.mismatches.load());
}

// ---------------------------------------------------------------------------
// dev-verify host side: buffers, start-up self-test, counters, records
// ---------------------------------------------------------------------------
struct DvState {
  DvBlk *d_blk = nullptr, *h_blk = nullptr;
  DvRec *d_rec = nullptr, *h_rec2 = nullptr;
  DvAbRec *d_ab = nullptr, *h_ab2 = nullptr;
  dvu64 *d_mark = nullptr, *d_abmark = nullptr;
  unsigned *d_hist = nullptr, *h_hist = nullptr;
  uint4 *d_yh = nullptr, *d_out = nullptr, *d_ys = nullptr;
  size_t stride_bytes = 0, nchunk = 0;
  int printed = 0, ab_printed = 0;
};
static DvState DV;

template <class T>
static T* dv_zalloc(size_t n) {
  T* p = nullptr;
  HCK(hipMalloc((void**)&p, n * sizeof(T)));
  HCK(hipMemset(p, 0, n * sizeof(T)));
  return p;
}
template <class T>
static T* dv_halloc(size_t n) {
  T* p = nullptr;
  HCK(hipHostMalloc((void**)&p, n * sizeof(T), hipHostMallocDefault));
  memset(p, 0, n * sizeof(T));
  return p;
}

// Per-layer buffers are 4096 B apart, so a layer's 256 B chunks are aligned
// like the chunks of the reply regions.
static void dv_geometry() {
  const size_t tmax = (size_t)tmax_of_list();
  DV.stride_bytes = (size_t)align_up(tmax * (size_t)O.hidden * 2, HX_PAGE);
  DV.nchunk = tmax * (size_t)O.hidden * 2 / 256 + 2;
}
static void dv_alloc_hub() {
  dv_geometry();
  const size_t nv = DV.stride_bytes * (size_t)O.layers / sizeof(uint4);
  DV.d_yh = dv_zalloc<uint4>(nv);
  DV.d_out = dv_zalloc<uint4>(nv);
  DV.d_blk = dv_zalloc<DvBlk>(1);
  DV.d_rec = dv_zalloc<DvRec>((size_t)kDvRecs);
  DV.d_ab = dv_zalloc<DvAbRec>((size_t)kDvAbRecs);
  DV.d_mark = dv_zalloc<dvu64>(DV.nchunk);
  DV.d_abmark = dv_zalloc<dvu64>(DV.nchunk * HX_MAX_SERVERS);
  DV.d_hist = dv_zalloc<unsigned>(DV.nchunk);
  DV.h_blk = dv_halloc<DvBlk>(1);
  DV.h_rec2 = dv_halloc<DvRec>(1);
  DV.h_ab2 = dv_halloc<DvAbRec>(1);
  DV.h_hist = dv_halloc<unsigned>(DV.nchunk);
}
static void dv_alloc_srv() {
  dv_geometry();
  DV.d_ys =
      dv_zalloc<uint4>(DV.stride_bytes * (size_t)O.layers / sizeof(uint4));
}

static uint16_t dv_half(const uint4& v, int q) {
  const uint32_t w = q < 2 ? v.x : q < 4 ? v.y : q < 6 ? v.z : v.w;
  return (uint16_t)((q & 1) ? (w >> 16) : (w & 0xFFFFu));
}
// The combine's arithmetic on the host codec: fp32 sum in order, one rounding.
static uint4 dv_sum_host(const uint4* p, int n) {
  uint32_t o[8];
  for (int q = 0; q < 8; q++) {
    float acc = h2f(dv_half(p[0], q));
    for (int s = 1; s < n; s++) acc += h2f(dv_half(p[s], q));
    o[q] = f2h(acc);
  }
  return make_uint4(o[0] | (o[1] << 16), o[2] | (o[3] << 16),
                    o[4] | (o[5] << 16), o[6] | (o[7] << 16));
}
static uint4 dv_u4(const uint32_t* w) {
  return make_uint4(w[0], w[1], w[2], w[3]);
}
static std::string dv_hex(const uint32_t* w) {
  char b[48];
  snprintf(b, sizeof b, "%04x %04x %04x %04x %04x %04x %04x %04x",
           w[0] & 0xFFFFu, w[0] >> 16, w[1] & 0xFFFFu, w[1] >> 16,
           w[2] & 0xFFFFu, w[2] >> 16, w[3] & 0xFFFFu, w[3] >> 16);
  return b;
}

// The pattern and the sum arithmetic on the device against the host model.
static void dv_selftest() {
  const int n = 4096;
  uint4 *d_vec = nullptr, *d_sum = nullptr;
  HCK(hipMalloc((void**)&d_vec, (size_t)n * sizeof(uint4)));
  HCK(hipMalloc((void**)&d_sum, (size_t)n * sizeof(uint4)));
  hipLaunchKernelGGL(k_dv_selftest, dim3((unsigned)((n + 255) / 256)),
                     dim3(256), 0, P.st, d_vec, d_sum, (dvu64)O.seed, n);
  HCK(hipGetLastError());
  HCK(hipStreamSynchronize(P.st));
  std::vector<uint4> hv((size_t)n), hs((size_t)n);
  HCK(hipMemcpy(hv.data(), d_vec, (size_t)n * sizeof(uint4),
                hipMemcpyDeviceToHost));
  HCK(hipMemcpy(hs.data(), d_sum, (size_t)n * sizeof(uint4),
                hipMemcpyDeviceToHost));
  HCK(hipFree(d_vec));
  HCK(hipFree(d_sum));
  int bad = 0;
  for (int i = 0; i < n; i++) {
    const int ns = 1 + i % HX_MAX_SERVERS;
    const uint64_t e = 1ull + (uint64_t)(i / 7);
    const uint32_t l = (uint32_t)(i % HX_MAX_LAYERS);
    const uint64_t v = (uint64_t)i * 977ull + (uint64_t)(i % 13);
    uint4 p[HX_MAX_SERVERS + 1];
    p[0] = dv_vec(dv_base(O.seed, e, l, (uint32_t)kDvHub), v);
    for (int s = 0; s < ns; s++)
      p[s + 1] = dv_vec(dv_base(O.seed, e, l, (uint32_t)s), v);
    const bool ok_v = dv_eq(hv[(size_t)i], p[1 + i % ns]);
    const bool ok_s = dv_eq(hs[(size_t)i], dv_sum_host(p, ns + 1));
    if (ok_v && ok_s) continue;
    if (bad++ < 3)
      say("dev-verify selftest: case %d: pattern %s, sum %s", i,
          ok_v ? "ok" : "DIFFERS", ok_s ? "ok" : "DIFFERS");
  }
  if (bad)
    die_setup(
        "dev-verify: %d of %d pattern/sum cases differ between the device "
        "and the host model",
        bad, n);
  say("dev-verify selftest: %d pattern vectors and sums identical on the "
      "device",
      n);
}

static void dv_read_blk() {
  HCK(hipMemcpyAsync(DV.h_blk, DV.d_blk, sizeof(DvBlk), hipMemcpyDeviceToHost,
                     P.side));
  HCK(hipStreamSynchronize(P.side));
}

static std::string dv_verdict(const DvRec& r) {
  const uint32_t pers = r.now_bad | r.atomic_bad;
  if (pers == 0u)
    return "TRANSIENT: every server reply reads correct now (plain and atomic, "
           "also after the delay): the combine consumed stale or "
           "not-yet-written data that has healed since, or its output was "
           "computed or stored wrong";
  std::string s;
  for (uint32_t k = 0; k < r.nsrv; k++) {
    if (!((pers >> k) & 1u)) continue;
    const bool pl = (r.now_bad >> k) & 1u, at = (r.atomic_bad >> k) & 1u,
               lt = (r.later_bad >> k) & 1u;
    const char* what =
        !at ? "STALE CACHE (plain re-read wrong, cache-bypassing re-read right)"
        : !pl ? "INCONSISTENT (plain re-read right, atomic re-read wrong)"
        : !lt ? "LATE WRITE (wrong at check time, right after the delay)"
              : "PERSISTENT (the reply memory holds wrong data)";
    char b[256];
    snprintf(b, sizeof b, "%sserver %u: %s%s", s.empty() ? "" : "; ", k, what,
             ((r.now_prev >> k) & 1u)
                 ? ", holds the previous occupant of the slot"
                 : "");
    s += b;
  }
  return s;
}

// Which single replacement (zeros, or the pattern of another recent
// (epoch, layer)) of one participant turns the right sum into the output.
static void dv_explain(int idx, const DvRec& r) {
  const int ns = (int)r.nsrv;
  uint4 part[HX_MAX_SERVERS + 1];
  part[0] = dv_u4(r.yh_hash);
  for (int s = 0; s < ns; s++) part[s + 1] = dv_u4(r.s[s].hash);
  const uint4 got = dv_u4(r.got);
  int found = 0;
  for (int j = 0; j <= ns && found < 6; j++) {
    const std::string who =
        j == 0 ? std::string("the hub y") : "server " + std::to_string(j - 1);
    uint4 q[HX_MAX_SERVERS + 1];
    for (int s = 0; s <= ns; s++) q[s] = part[s];
    q[j] = make_uint4(0u, 0u, 0u, 0u);
    if (dv_eq(dv_sum_host(q, ns + 1), got)) {
      say("DEVDETAIL #%d explain: output == sum with %s replaced by zeros", idx,
          who.c_str());
      found++;
    }
    const uint64_t e0 = r.epoch > 3 ? r.epoch - 3 : 1;
    for (uint64_t e = e0; e <= r.epoch && found < 6; e++)
      for (int l = 0; l < O.layers && found < 6; l++) {
        if (e == r.epoch && l == (int)r.layer) continue;
        q[j] = dv_vec(dv_base(O.seed, e, (uint32_t)l,
                              j == 0 ? (uint32_t)kDvHub : (uint32_t)(j - 1)),
                      r.v);
        if (!dv_eq(dv_sum_host(q, ns + 1), got)) continue;
        say("DEVDETAIL #%d explain: output == sum with %s replaced by the "
            "pattern of epoch %llu layer %d",
            idx, who.c_str(), (unsigned long long)e, l);
        found++;
      }
  }
  if (!found)
    say("DEVDETAIL #%d explain: no single replacement by zeros or by the "
        "pattern of another (epoch, layer) of the last 3 epochs reproduces "
        "the output",
        idx);
}

static void dv_print_rec(int idx, const DvRec& r) {
  std::string tag;
  if ((long long)r.exch == O.inject_stale)
    tag += " [--inject-stale: server " + std::to_string(O.inject_srv) +
           " chunk " + std::to_string(O.inject_chunk) + "]";
  if ((long long)r.exch == O.inject_out)
    tag += " [--inject-out: chunk " + std::to_string(O.inject_chunk) + "]";
  say("DEVDETAIL #%d exch=%llu step=%llu layer=%u epoch=%llu T=%u slot=%u "
      "chunk=%u chunk_byte_offset=0x%x v=%llu row=%zu col=%zu "
      "chunk_bad_vectors=%d/%u chunk_bad_mask=0x%04x elems_bad_in_vector=%u "
      "hub_y_bad=%u%s",
      idx, r.exch, r.exch / (dvu64)O.layers, r.layer, r.epoch, r.T, r.slot,
      r.chunk, r.chunk * 256u, r.v, (size_t)r.v * 8 / (size_t)O.hidden,
      (size_t)r.v * 8 % (size_t)O.hidden, __builtin_popcount(r.bad_mask),
      r.nvec_chunk, r.bad_mask, r.elems_bad, r.hub_y_bad, tag.c_str());
  say("DEVDETAIL #%d got  %s", idx, dv_hex(r.got).c_str());
  say("DEVDETAIL #%d want %s", idx, dv_hex(r.want).c_str());
  say("DEVDETAIL #%d hub y %s (pattern %s)%s", idx, dv_hex(r.yh).c_str(),
      dv_hex(r.yh_hash).c_str(), r.hub_y_bad ? " HUB Y WRONG" : "");
  for (uint32_t s = 0; s < r.nsrv; s++) {
    const DvSrv& q = r.s[s];
    const bool pl = (r.now_bad >> s) & 1u, at = (r.atomic_bad >> s) & 1u,
               lt = (r.later_bad >> s) & 1u;
    say("DEVDETAIL #%d server %u: plain %s [%s] | atomic %s [%s] | pattern %s",
        idx, s, dv_hex(q.plain).c_str(), pl ? "BAD" : "ok",
        dv_hex(q.atomic).c_str(), at ? "BAD" : "ok", dv_hex(q.hash).c_str());
    say("DEVDETAIL #%d server %u: after %d us %s [%s] | previous occupant %s%s",
        idx, s, O.dv_reread_us, dv_hex(q.later).c_str(), lt ? "BAD" : "ok",
        dv_hex(q.prev).c_str(),
        ((r.now_prev >> s) & 1u) ? " [plain re-read == previous occupant]"
                                 : "");
  }
  std::string exs;
  for (uint32_t s = 0; s < r.nsrv; s++)
    if ((r.explains >> s) & 1u)
      exs += (exs.empty() ? "" : ",") + std::to_string(s);
  const std::string prevs = r.prev_valid
                                ? "epoch " + std::to_string(r.prev_epoch) +
                                      " layer " + std::to_string(r.prev_layer)
                                : std::string("none (zeros)");
  say("DEVDETAIL #%d previous occupant of the slot: %s; output == sum with "
      "the previous occupant of server(s) [%s]",
      idx, prevs.c_str(), exs.c_str());
  say("DEVDETAIL #%d verdict: %s", idx, dv_verdict(r).c_str());
  dv_explain(idx, r);
}

static void dv_print_ab(int idx, const DvAbRec& r) {
  static const char* const kCat[4] = {
      "A wrong, B right: plain read stale, bypass read right",
      "A right, B wrong", "A and B wrong and equal: memory wrong at both reads",
      "A and B wrong and different"};
  say("DEVDETAIL AB#%d exch=%llu step=%llu layer=%u epoch=%llu T=%u slot=%u "
      "server=%u chunk=%u v=%llu: %s%s",
      idx, r.exch, r.exch / (dvu64)O.layers, r.layer, r.epoch, r.T, r.slot,
      r.srv, r.chunk, r.v, kCat[r.cat & 3u],
      r.injected ? " [--inject-rd: faked read]" : "");
  say("DEVDETAIL AB#%d A plain  %s%s", idx, dv_hex(r.a).c_str(),
      r.a_prev ? " == previous occupant of the slot" : "");
  say("DEVDETAIL AB#%d B atomic %s%s", idx, dv_hex(r.b).c_str(),
      r.b_prev ? " == previous occupant of the slot" : "");
  say("DEVDETAIL AB#%d pattern  %s", idx, dv_hex(r.want).c_str());
  say("DEVDETAIL AB#%d previous %s", idx, dv_hex(r.prev).c_str());
  say("DEVDETAIL AB#%d later    %s (atomic re-read after %d us)", idx,
      dv_hex(r.later).c_str(), O.dv_reread_us);
}

// Prints the new records (up to --dev-verify-lines). Mid-run a record is taken
// only when two copies agree and it is complete; after the stream sync it is
// final.
static void dv_poll_records(bool final_read) {
  const DvBlk& b = *DV.h_blk;
  const int n = (int)std::min<dvu64>(b.nrec, (dvu64)kDvRecs);
  while (DV.printed < n && DV.printed < O.dv_lines) {
    const size_t i = (size_t)DV.printed;
    HCK(hipMemcpyAsync(DV.h_rec2, DV.d_rec + i, sizeof(DvRec),
                       hipMemcpyDeviceToHost, P.side));
    HCK(hipStreamSynchronize(P.side));
    const DvRec r = *DV.h_rec2;
    if (r.valid == 0) break;
    if (!final_read) {
      HCK(hipMemcpyAsync(DV.h_rec2, DV.d_rec + i, sizeof(DvRec),
                         hipMemcpyDeviceToHost, P.side));
      HCK(hipStreamSynchronize(P.side));
      if (memcmp(&r, DV.h_rec2, sizeof r) != 0) break;
    }
    dv_print_rec(DV.printed, r);
    DV.printed++;
  }
  const int nab = (int)std::min<dvu64>(b.ab_nrec, (dvu64)kDvAbRecs);
  while (DV.ab_printed < nab && DV.ab_printed < O.dv_lines) {
    const size_t i = (size_t)DV.ab_printed;
    HCK(hipMemcpyAsync(DV.h_ab2, DV.d_ab + i, sizeof(DvAbRec),
                       hipMemcpyDeviceToHost, P.side));
    HCK(hipStreamSynchronize(P.side));
    const DvAbRec r = *DV.h_ab2;
    if (r.valid == 0) break;
    if (!final_read) {
      HCK(hipMemcpyAsync(DV.h_ab2, DV.d_ab + i, sizeof(DvAbRec),
                         hipMemcpyDeviceToHost, P.side));
      HCK(hipStreamSynchronize(P.side));
      if (memcmp(&r, DV.h_ab2, sizeof r) != 0) break;
    }
    dv_print_ab(DV.ab_printed, r);
    DV.ab_printed++;
  }
}

// The result of the device counters read last (dv_read_blk).
static void dv_report() {
  const DvBlk& b = *DV.h_blk;
  typedef unsigned long long ull;
  say("DEVVERIFY exchanges=%llu mismatches=%llu elems_checked=%llu",
      (ull)b.checks, (ull)b.bad_exch, (ull)b.elems);
  say("DEVSTAT bad_exchanges=%llu bad_vectors=%llu bad_elements=%llu "
      "bad_chunks=%llu hub_y_bad=%llu explained_by_nothing=%llu records=%llu "
      "ab_records=%llu",
      (ull)b.bad_exch, (ull)b.bad_vec, (ull)b.bad_elem, (ull)b.bad_chunk,
      (ull)b.hub_y_bad, (ull)b.explain_none, (ull)b.nrec, (ull)b.ab_nrec);
  if (b.bad_vec != 0) {
    for (int s = 0; s < O.nsrv; s++)
      say("DEVSRV s=%d reply_plain_ok=%llu reply_plain_bad=%llu "
          "reply_atomic_bad=%llu reply_plain_is_prev=%llu "
          "output_explained_by_prev=%llu",
          s, (ull)b.rnow_ok[s], (ull)b.rnow_bad[s], (ull)b.ratomic_bad[s],
          (ull)b.rnow_prev[s], (ull)b.explain_prev[s]);
    for (int l = 0; l < O.layers; l++)
      if (b.layer_exch[l] || b.layer_chunk[l] || b.layer_elem[l])
        say("DEVLAYER l=%d slot=%d bad_exchanges=%llu bad_chunks=%llu "
            "bad_elements=%llu",
            l, l % O.nslots, (ull)b.layer_exch[l], (ull)b.layer_chunk[l],
            (ull)b.layer_elem[l]);
    HCK(hipMemcpyAsync(DV.h_hist, DV.d_hist, DV.nchunk * sizeof(unsigned),
                       hipMemcpyDeviceToHost, P.side));
    HCK(hipStreamSynchronize(P.side));
    int lines = 0;
    for (size_t c = 0; c < DV.nchunk; c++)
      if (DV.h_hist[c] != 0 && lines++ < 32)
        say("DEVCHUNK chunk=%zu byte_offset=0x%zx wrong_exchanges=%u", c,
            c * 256, DV.h_hist[c]);
  }
  if (O.rd_mode == 4)
    for (int s = 0; s < O.nsrv; s++)
      say("DEVAB s=%d a_wrong_b_right=%llu a_right_b_wrong=%llu "
          "both_wrong_equal=%llu both_wrong_different=%llu a_is_prev=%llu "
          "b_is_prev=%llu",
          s, (ull)b.ab_cat[s][0], (ull)b.ab_cat[s][1], (ull)b.ab_cat[s][2],
          (ull)b.ab_cat[s][3], (ull)b.ab_a_prev[s], (ull)b.ab_b_prev[s]);
}

// Drains the stream, reads counters and records and prints the report; an
// abort ends the process after the report.
static Snap dv_drain(const char* where) {
  HCK(hipStreamSynchronize(P.st));
  const Snap s = read_snap();
  dv_read_blk();
  dv_poll_records(true);
  dv_report();
  if (s.abort_word != 0 || s.st[HX_ST_RECORD] != 0) handle_abort(s, where);
  return s;
}

static double pct(std::vector<double>& v, double p) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[(size_t)(p * (double)(v.size() - 1) + 0.5)];
}

static std::map<int, LayerPtrs> g_bench_in;
static std::vector<hipEvent_t> g_ev0, g_ev1;

// Everything that must exist before the ready barrier: buffers, bench inputs,
// graphs (a capture needs no peer).
static void hub_prepare() {
  const int L = O.layers, tmax = tmax_of_list();
  hub_alloc_buffers(O.graph && !O.bench ? (size_t)L : 1, (size_t)tmax);
  if (O.bench || O.dev_verify) {
    for (int T : O.tlist) {
      if (g_bench_in.count(T)) continue;
      std::vector<uint16_t> x((size_t)T * O.hidden);
      std::vector<int8_t> q(x.size());
      std::vector<int32_t> ids((size_t)T * O.topk);
      std::vector<float> w(ids.size());
      gen_inputs(O.seed, 0, 0, T, O.hidden, O.topk, O.experts, x.data(),
                 q.data(), ids.data(), w.data());
      g_bench_in[T] = {upload(x), upload(ids), upload(w), B.d_hubids, B.d_y};
    }
    if (O.bench && !O.graph)
      for (int l = 0; l < L; l++) {
        hipEvent_t a, b;
        HCK(hipEventCreate(&a));
        HCK(hipEventCreate(&b));
        g_ev0.push_back(a);
        g_ev1.push_back(b);
      }
  }
  if (O.dev_verify) dv_alloc_hub();
  if (O.graph)
    for (int T : O.tlist)
      if (!g_graphs.count(T))
        g_graphs[T] = capture_graph([&] {
          hx_launch_begin_step(P.d_ctx, T, T, P.st);
          for (int l = 0; l < L; l++)
            hub_enqueue_layer(O.bench ? g_bench_in.at(T) : hub_ptrs((size_t)l),
                              l, T);
        });
}

// Latency mode: inputs uploaded once per T, no verification, no host work
// between the steps except the status read.
static void hub_run_bench() {
  const int L = O.layers;
  const long long nsteps = steps_total();
  std::map<int, std::vector<double>> lat, stepms;
  uint64_t exch = 0;
  say("bench: %lld steps (%d warm-up) of %d layers, %s", nsteps, O.warmup_steps,
      L, O.graph ? "graph" : "stream");
  for (long long step = 0; step < nsteps; step++) {
    const int T = O.tlist[(size_t)step % O.tlist.size()];
    const int64_t t0 = now_ns();
    if (O.graph) {
      HCK(hipGraphLaunch(g_graphs.at(T), P.st));
    } else {
      hx_launch_begin_step(P.d_ctx, T, bucket_of(T), P.st);
      for (int l = 0; l < L; l++) {
        HCK(hipEventRecord(g_ev0[(size_t)l], P.st));
        hub_enqueue_layer(g_bench_in.at(T), l, T);
        HCK(hipEventRecord(g_ev1[(size_t)l], P.st));
      }
    }
    HCK(hipGetLastError());
    HCK(hipStreamSynchronize(P.st));
    const int64_t t1 = now_ns();
    check_abort("after the step");
    exch += (uint64_t)L;
    G.iters = exch;
    G.steps = (uint64_t)step + 1;
    if (step >= O.warmup_steps) {
      stepms[T].push_back((double)(t1 - t0) / 1e6);
      if (!O.graph)
        for (int l = 0; l < L; l++) {
          float ms = 0;
          HCK(hipEventElapsedTime(&ms, g_ev0[(size_t)l], g_ev1[(size_t)l]));
          lat[T].push_back((double)ms * 1000.0);
        }
    }
    if (O.hub_die_after >= 0 && exch >= (uint64_t)O.hub_die_after)
      die_now("die");
  }
  std::vector<int> seen;
  for (int T : O.tlist) {
    if (std::find(seen.begin(), seen.end(), T) != seen.end()) continue;
    seen.push_back(T);
    std::vector<double>& sm = stepms[T];
    std::vector<double>& ly = lat[T];
    const double s50 = pct(sm, 0.5), s99 = pct(sm, 0.99);
    char line[512];
    int n = snprintf(line, sizeof line,
                     "HXBENCH hub T=%d mode=%s eager=%d nsrv=%d layers=%d "
                     "steps=%zu step_ms_p50=%.3f step_ms_p99=%.3f",
                     T, O.graph ? "graph" : "stream", O.eager ? 1 : 0, O.nsrv,
                     L, sm.size(), s50, s99);
    if (!ly.empty()) {
      const double l50 = pct(ly, 0.5), l90 = pct(ly, 0.9), l99 = pct(ly, 0.99);
      n += snprintf(line + n, sizeof line - (size_t)n,
                    " layer_n=%zu layer_us_p50=%.1f layer_us_p90=%.1f "
                    "layer_us_p99=%.1f layer_us_max=%.1f",
                    ly.size(), l50, l90, l99, ly.back());
    }
    line[n++] = '\n';
    wr_all(1, line, (size_t)n);
  }
}

static DvArgs dv_args(uint4* yh, uint4* out, uint64_t exch, int T, int layer,
                      int nblocks) {
  DvArgs a;
  memset(&a, 0, sizeof a);
  a.c = P.d_ctx;
  a.d = DV.d_blk;
  a.rec = DV.d_rec;
  a.ab = DV.d_ab;
  a.mark = DV.d_mark;
  a.abmark = DV.d_abmark;
  a.hist = DV.d_hist;
  a.yh = yh;
  a.out = out;
  a.seed = (dvu64)O.seed;
  a.exch = (dvu64)exch;
  a.T = T;
  a.layer = layer;
  a.nblocks = nblocks;
  a.reread_us = O.dv_reread_us;
  a.inj_srv = O.inject_srv;
  a.inj_chunk =
      O.inject_rd >= 0 && (long long)exch == O.inject_rd ? O.inject_chunk : -1;
  a.inj_kind = O.inject_rd_kind;
  return a;
}

// Device-side verification. The hub fills its y and every server its reply
// from the pattern on the device; after each combine a check kernel recomputes
// the sum and compares the whole output. The host reads the counters every
// --check-every exchanges only; up to --pipeline steps are in flight.
static void hub_run_devverify() {
  const int L = O.layers, D = O.pipeline;
  const long long nsteps = steps_total();
  const size_t stride_vec = DV.stride_bytes / sizeof(uint4);
  std::vector<hipEvent_t> ev((size_t)D);
  for (hipEvent_t& e : ev)
    HCK(hipEventCreateWithFlags(&e, hipEventDisableTiming));
  uint64_t enq = 0, retired = 0, enq_elems = 0, seen_bad = 0, log_bucket = 0;
  uint64_t next_read = (uint64_t)O.check_every;
  const int64_t t_start = now_ns();
  say("dev-verify: %lld steps of %d layers, %d in flight, counters read every "
      "%d exchanges, combine %s (read mode %d)",
      nsteps, L, D, O.check_every,
      O.rd_mode == 4 ? "k_dv_combine_ab" : "hx_k_combine", O.rd_mode);
  auto periodic = [&]() {
    const Snap s = read_snap();
    if (s.abort_word != 0 || s.st[HX_ST_RECORD] != 0) dv_drain("dev-verify");
    dv_read_blk();
    const DvBlk& b = *DV.h_blk;
    G.mismatches = b.bad_exch;
    dv_poll_records(false);
    if (b.bad_exch > seen_bad) {
      seen_bad = b.bad_exch;
      say("MISMATCH dev-verify: %llu exchange(s) with a wrong combine output%s "
          "after %llu exchanges checked on the device",
          (unsigned long long)b.bad_exch,
          O.rd_mode == 4 ? " or different paired reads" : "",
          (unsigned long long)b.checks);
      if (O.stop_on_mismatch) {
        dump_status(dv_drain("after the mismatch"));
        finish(kExitMismatch, "mismatch");
      }
    }
    if (O.log_every > 0 && b.checks / (dvu64)O.log_every != log_bucket) {
      log_bucket = b.checks / (dvu64)O.log_every;
      say("progress: %llu exchanges checked, %llu wrong, %.0f/s",
          (unsigned long long)b.checks, (unsigned long long)b.bad_exch,
          (double)b.checks / ((double)(now_ns() - t_start) / 1e9));
    }
  };
  for (long long step = 0; step < nsteps; step++) {
    if (step >= D) {
      HCK(hipEventSynchronize(ev[(size_t)(step % D)]));
      retired += (uint64_t)L;
      G.iters = retired;
      if (O.hub_die_after >= 0 && retired >= (uint64_t)O.hub_die_after)
        die_now("die");
    }
    const int T = O.tlist[(size_t)step % O.tlist.size()];
    const LayerPtrs& in = g_bench_in.at(T);
    const long long nvec = (long long)T * O.hidden / 8;
    const unsigned nb_fill = (unsigned)std::min<long long>(
        32, std::max<long long>(1, (nvec + 255) / 256));
    const int nb_chk =
        O.blk_check > 0
            ? O.blk_check
            : (int)std::min<long long>(
                  32, std::max<long long>(
                          1, (nvec + DV_CHECK_THREADS - 1) / DV_CHECK_THREADS));
    const int nb_comb = O.blk_combine > 0
                            ? O.blk_combine
                            : hx_combine_blocks((uint64_t)T * O.hidden * 2u);
    hx_launch_begin_step(P.d_ctx, T, bucket_of(T), P.st);
    hipLaunchKernelGGL(k_dv_fill, dim3(nb_fill, (unsigned)L), dim3(256), 0,
                       P.st, P.d_ctx, DV.d_yh, (long long)stride_vec,
                       (dvu64)O.seed, T);
    for (int l = 0; l < L; l++) {
      const uint64_t exch = enq + (uint64_t)l;
      uint4* yh = DV.d_yh + (size_t)l * stride_vec;
      uint4* out = DV.d_out + (size_t)l * stride_vec;
      if ((long long)exch == O.inject_late && O.inject_late_req)
        inject_late_arm(
            (uint32_t)((uint64_t)(step + 1) * 64u + (uint64_t)l + 1u),
            O.inject_chunk, O.inject_late_us, O.inject_srv);
      hx_launch_dispatch(P.d_ctx, in.x, in.ids, in.w, in.hubids, T, l, O.nsrv,
                         O.hidden, O.blk_dispatch, P.st);
      if (O.rd_mode == 4)
        hipLaunchKernelGGL(k_dv_combine_ab, dim3((unsigned)nb_comb),
                           dim3(HX_COMBINE_THREADS), 0, P.st,
                           dv_args(yh, out, exch, T, l, nb_comb));
      else
        hx_launch_combine(P.d_ctx, yh, out, T, l, O.hidden, O.blk_combine,
                          P.st);
      if ((long long)exch == O.inject_out)
        hipLaunchKernelGGL(k_dv_inject_out, dim3(1), dim3(16), 0, P.st, P.d_ctx,
                           out, T, O.inject_chunk);
      hipLaunchKernelGGL(k_dv_check, dim3((unsigned)nb_chk),
                         dim3(DV_CHECK_THREADS), 0, P.st,
                         dv_args(yh, out, exch, T, l, nb_chk));
    }
    HCK(hipGetLastError());
    HCK(hipEventRecord(ev[(size_t)(step % D)], P.st));
    enq += (uint64_t)L;
    enq_elems += (uint64_t)L * (uint64_t)T * (uint64_t)O.hidden;
    G.steps = (uint64_t)step + 1;
    if (enq >= next_read) {
      periodic();
      next_read = enq + (uint64_t)O.check_every;
    }
  }
  const Snap s = dv_drain("after the run");
  const DvBlk& b = *DV.h_blk;
  G.iters = enq;
  G.mismatches = b.bad_exch;
  const struct {
    const char* name;
    uint64_t got, want;
  } cnt_chk[] = {{"epoch", s.st[HX_ST_EPOCH], (uint64_t)nsteps},
                 {"steps", s.st[HX_ST_STEPS], (uint64_t)nsteps},
                 {"dispatches", s.st[HX_ST_DISPATCHES], enq},
                 {"combines", s.st[HX_ST_COMBINES], enq}};
  uint64_t bad_status = 0;
  for (const auto& c : cnt_chk)
    if (c.got != c.want) {
      say("MISMATCH status.%s: device %llu expected %llu", c.name,
          (unsigned long long)c.got, (unsigned long long)c.want);
      bad_status++;
    }
  if (b.checks != enq || b.elems != enq_elems)
    die_setup(
        "dev-verify: the device checked %llu exchanges / %llu elements, "
        "expected %llu / %llu",
        (unsigned long long)b.checks, (unsigned long long)b.elems,
        (unsigned long long)enq, (unsigned long long)enq_elems);
  G.mismatches += bad_status;
  say("run complete: %llu exchanges checked on the device (%llu elements), "
      "%llu bad exchanges, %.0f exchanges/s",
      (unsigned long long)b.checks, (unsigned long long)b.elems,
      (unsigned long long)b.bad_exch,
      (double)enq / ((double)(now_ns() - t_start) / 1e9));
}

[[noreturn]] static void hub_end(const std::vector<long>& pids) {
  stop_watcher();
  print_digstat();
  rdv_publish("done_hub" + std::to_string(O.hub_index), "ok\n", true);
  for (int s = 0; s < O.nsrv; s++) {
    std::string t;
    if (!rdv_wait_file("done_srv" + std::to_string(s), pids[(size_t)s],
                       kFinalWaitS, &t) &&
        pid_alive(pids[(size_t)s]))
      say("WARNING: server %d did not acknowledge the end of the run", s);
  }
  G.clean_dir = true;
  const uint64_t mm = G.mismatches.load();
  finish(mm ? kExitMismatch : kExitOk, mm ? "mismatch" : "ok");
}

static void hub_main() {
  const std::string h = std::to_string(O.hub_index);
  for (const std::string& n :
       {hub_name() + ".json", "ready_h" + h, "done_hub" + h, "fault_hub" + h})
    rdv_remove(n);
  device_cvt_selftest();
  if (O.dev_verify) dv_selftest();
  alloc_region(GEO.reply_bytes);
  if (!rdv_publish(hub_name() + ".json", own_info(true), true))
    die_setup("cannot publish %s", rpath(hub_name() + ".json").c_str());
  std::vector<long> pids((size_t)O.nsrv, 0);
  for (int s = 0; s < O.nsrv; s++) {
    std::string js;
    const std::string who = "server " + std::to_string(s);
    wait_peer_info(srv_name(s) + ".json", who.c_str(), &js, &pids[(size_t)s]);
    check_peer_info(js, who.c_str(), false);
    if (js_int(js, "index", -1) != s)
      die_setup("%s: published index %lld", who.c_str(),
                js_int(js, "index", -1));
    P.peer[s] = open_peer(js, who.c_str());
    say("%s: pid %ld arch %s bus %s alloc %s", who.c_str(), pids[(size_t)s],
        js_str(js, "gcn").c_str(), js_str(js, "bus").c_str(),
        js_str(js, "alloc").c_str());
  }
  build_tables(TAB, O.experts, O.layers, O.nsrv, O.rows_cap, O.kmode, O.kfixed);
  fill_ctx_common(1, (uint32_t)O.hub_index);
  hub_alloc_lstat();
  P.img.hub_slot = upload(TAB.hub_slot);
  P.img.cold_srv = upload(TAB.cold_srv);
  P.img.n_cold = upload(TAB.n_cold);
  P.img.owned = upload(TAB.owned);
  P.img.ktab = upload(TAB.ktab);
  for (int s = 0; s < O.nsrv; s++) P.img.peer_base[s] = P.peer[s];
  upload_ctx();
  hub_prepare();
  rdv_publish("ready_h" + h, "1\n", true);
  for (int s = 0; s < O.nsrv; s++) {
    std::string t;
    if (!rdv_wait_file(ack_name(s), pids[(size_t)s], O.rdv_timeout_s, &t))
      die_setup("server %d did not become ready (%s)", s,
                pid_alive(pids[(size_t)s]) ? "timeout" : "process gone");
  }
  for (int s = 0; s < O.nsrv; s++)
    g_peers.push_back({s, pids[(size_t)s], -1, "server " + std::to_string(s)});
  start_watcher();
  if (O.bench)
    hub_run_bench();
  else if (O.dev_verify)
    hub_run_devverify();
  else
    hub_run_verify();
  hub_end(pids);
}

// ---------------------------------------------------------------------------
// server
// ---------------------------------------------------------------------------
static uint16_t* g_srv_y = nullptr;
static int32_t* g_srv_ids32 = nullptr;
static uint8_t* g_srv_xws = nullptr;
static float* g_srv_wws = nullptr;
static std::map<int, hipGraphExec_t> g_sgraph[2];

// The host descriptor of the next step: one 16 B copy ordered before the
// srv_begin that reads it.
static void srv_post_desc(uint64_t epoch, int T, int bucket) {
  uint8_t* slot = static_cast<uint8_t*>(P.pin) + 128 +
                  16 * (P.desc_next++ % (unsigned)kDescRing);
  const uint32_t t = (uint32_t)T, b = (uint32_t)bucket;
  memcpy(slot, &epoch, 8);
  memcpy(slot + 8, &t, 4);
  memcpy(slot + 12, &b, 4);
  HCK(hipMemcpyAsync(
      reinterpret_cast<uint8_t*>(P.d_ctx) + offsetof(HxCtx, desc_epoch), slot,
      16, hipMemcpyHostToDevice, P.st));
}

static void srv_enqueue_layer(int layer, int T, bool skip_push) {
  const uint8_t* sb =
      P.own + HX_PAGE + (size_t)(layer % O.nslots) * GEO.slot_stride;
  hx_launch_srv_wait(P.d_ctx, layer, T, O.hidden, O.blk_srv_wait, P.st);
  // the cached workspace copy of hx_k_srv_wait, or the uncached inbox slot
  const void* x = O.srv_ws ? static_cast<const void*>(g_srv_xws) : sb;
  const float* w =
      O.srv_ws ? g_srv_wws : reinterpret_cast<const float*>(sb + GEO.off_w);
  launch_synth(x, P.img.ids32, w, g_srv_y, T);
  if (O.srv_work_us > 0)
    hipLaunchKernelGGL(k_spin, dim3(1), dim3(64), 0, P.st,
                       (long long)(O.srv_work_us * (double)G.tpm / 1000.0));
  if (!skip_push)
    hx_launch_srv_push(P.d_ctx, g_srv_y, T, layer, O.hidden, O.blk_push, P.st);
}

static bool srv_hanging(uint64_t exch) {
  static bool noted = false;
  const bool h = O.hang_after >= 0 && exch >= (uint64_t)O.hang_after;
  if (h && !noted) {
    noted = true;
    write_fault_stamp("hang");
    say("fault injection: hang from exchange %llu on (srv_push skipped)",
        (unsigned long long)exch);
  }
  return h;
}

static void srv_prepare() {
  const size_t rows = (size_t)O.rows_cap;
  HCK(hipMalloc((void**)&g_srv_y,
                (size_t)tmax_of_list() * (size_t)O.hidden * 2));
  HCK(hipMalloc((void**)&g_srv_ids32, rows * (size_t)O.topk * 4));
  HCK(hipMemset(g_srv_ids32, 0xFF, rows * (size_t)O.topk * 4));
  if (O.srv_ws) {
    HCK(hipMalloc((void**)&g_srv_xws, rows * (size_t)O.hidden * 2));
    HCK(hipMalloc((void**)&g_srv_wws, rows * (size_t)O.topk * 4));
    HCK(hipMemset(g_srv_xws, 0, rows * (size_t)O.hidden * 2));
    HCK(hipMemset(g_srv_wws, 0, rows * (size_t)O.topk * 4));
  }
  if (O.dev_verify) {
    dv_alloc_srv();
    dv_selftest();
  }
}

static void srv_capture_graphs() {
  if (!O.graph) return;
  for (int T : O.tlist)
    for (int hang = 0; hang < 2; hang++) {
      if ((hang == 1 && O.hang_after < 0) || g_sgraph[hang].count(T)) continue;
      g_sgraph[hang][T] = capture_graph([&] {
        hx_launch_srv_begin(P.d_ctx, P.st);
        for (int l = 0; l < O.layers; l++) srv_enqueue_layer(l, T, hang == 1);
      });
    }
}

static void srv_run() {
  const long long nsteps = steps_total();
  uint64_t exch = 0;
  const int64_t t_start = now_ns();
  say("run: %lld steps of up to %d layers, %s mode", nsteps, O.layers,
      O.graph ? "graph" : "stream");
  auto after = [&]() {
    G.iters = exch;
    if (O.die_after >= 0 && exch >= (uint64_t)O.die_after) die_now("die");
  };
  for (long long step = 0; step < nsteps; step++) {
    const int T = O.tlist[(size_t)step % O.tlist.size()];
    const int nl = layers_in_step(step);
    srv_post_desc((uint64_t)step + 1, T, bucket_of(T));
    if (O.graph) {
      HCK(hipGraphLaunch(g_sgraph[srv_hanging(exch) ? 1 : 0].at(T), P.st));
      HCK(hipStreamSynchronize(P.st));
      check_abort("after the step");
      exch += (uint64_t)nl;
      after();
    } else {
      hx_launch_srv_begin(P.d_ctx, P.st);
      for (int l = 0; l < nl; l++) {
        srv_enqueue_layer(l, T, srv_hanging(exch));
        HCK(hipGetLastError());
        HCK(hipStreamSynchronize(P.st));
        check_abort("after the exchange");
        exch++;
        after();
      }
    }
    G.steps = (uint64_t)step + 1;
    if (O.log_every > 0 &&
        step % std::max<long long>(1, O.log_every / O.layers) == 0)
      say("progress: %llu exchanges, %.0f/s", (unsigned long long)exch,
          (double)exch / ((double)(now_ns() - t_start) / 1e9));
  }
}

// Bench: nothing waits on the host between layers; up to --pipeline steps are
// in flight, so the server is ready before the hub's next request.
static void srv_run_bench() {
  const long long nsteps = steps_total();
  const int D = O.pipeline;
  std::vector<hipEvent_t> ev((size_t)D);
  for (hipEvent_t& e : ev)
    HCK(hipEventCreateWithFlags(&e, hipEventDisableTiming));
  uint64_t enq = 0, done = 0;
  auto retire = [&]() {
    check_abort("after a step");
    done += (uint64_t)O.layers;
    G.iters = done;
    if (O.die_after >= 0 && done >= (uint64_t)O.die_after) die_now("die");
  };
  say("bench: %lld steps, %d in flight", nsteps, D);
  for (long long step = 0; step < nsteps; step++) {
    if (step >= D) {
      HCK(hipEventSynchronize(ev[(size_t)(step % D)]));
      retire();
    }
    const int T = O.tlist[(size_t)step % O.tlist.size()];
    srv_post_desc((uint64_t)step + 1, T, bucket_of(T));
    if (O.graph) {
      HCK(hipGraphLaunch(g_sgraph[srv_hanging(enq) ? 1 : 0].at(T), P.st));
    } else {
      hx_launch_srv_begin(P.d_ctx, P.st);
      for (int l = 0; l < O.layers; l++)
        srv_enqueue_layer(l, T, srv_hanging(enq + (uint64_t)l));
    }
    enq += (uint64_t)O.layers;
    HCK(hipGetLastError());
    HCK(hipEventRecord(ev[(size_t)(step % D)], P.st));
    G.steps = (uint64_t)step + 1;
  }
  HCK(hipStreamSynchronize(P.st));
  while (done < enq) retire();
}

// Device-side verification: the staged replies come from the pattern kernel
// (one 4096-aligned buffer per layer) instead of the synth; the waits and the
// pushes are the production kernels.
static void srv_run_devverify() {
  const int L = O.layers, D = O.pipeline;
  const long long nsteps = steps_total();
  const size_t stride_vec = DV.stride_bytes / sizeof(uint4);
  std::vector<hipEvent_t> ev((size_t)D);
  for (hipEvent_t& e : ev)
    HCK(hipEventCreateWithFlags(&e, hipEventDisableTiming));
  uint64_t enq = 0, done = 0;
  auto retire = [&]() {
    check_abort("after a step");
    done += (uint64_t)L;
    G.iters = done;
    if (O.die_after >= 0 && done >= (uint64_t)O.die_after) die_now("die");
  };
  say("dev-verify: %lld steps of %d layers, %d in flight", nsteps, L, D);
  for (long long step = 0; step < nsteps; step++) {
    if (step >= D) {
      HCK(hipEventSynchronize(ev[(size_t)(step % D)]));
      retire();
    }
    const int T = O.tlist[(size_t)step % O.tlist.size()];
    const long long nvec = (long long)T * O.hidden / 8;
    const unsigned nb_fill = (unsigned)std::min<long long>(
        32, std::max<long long>(1, (nvec + 255) / 256));
    srv_post_desc((uint64_t)step + 1, T, bucket_of(T));
    hx_launch_srv_begin(P.d_ctx, P.st);
    hipLaunchKernelGGL(k_dv_fill, dim3(nb_fill, (unsigned)L), dim3(256), 0,
                       P.st, P.d_ctx, DV.d_ys, (long long)stride_vec,
                       (dvu64)O.seed, T);
    for (int l = 0; l < L; l++) {
      const uint64_t exch = enq + (uint64_t)l;
      uint4* y = DV.d_ys + (size_t)l * stride_vec;
      if ((long long)exch == O.inject_stale && O.index == O.inject_srv)
        hipLaunchKernelGGL(k_dv_inject_stale, dim3(1), dim3(16), 0, P.st,
                           P.d_ctx, y, (dvu64)O.seed, T, l, O.inject_chunk);
      hx_launch_srv_wait(P.d_ctx, l, T, O.hidden, O.blk_srv_wait, P.st);
      if (O.srv_work_us > 0)
        hipLaunchKernelGGL(k_spin, dim3(1), dim3(64), 0, P.st,
                           (long long)(O.srv_work_us * (double)G.tpm / 1000.0));
      if ((long long)exch == O.inject_late && !O.inject_late_req &&
          O.index == O.inject_srv)
        inject_late_arm(
            (uint32_t)((uint64_t)(step + 1) * 64u + (uint64_t)l + 1u),
            O.inject_chunk, O.inject_late_us, 0);
      if (!srv_hanging(exch))
        hx_launch_srv_push(P.d_ctx, y, T, l, O.hidden, O.blk_push, P.st);
    }
    enq += (uint64_t)L;
    HCK(hipGetLastError());
    HCK(hipEventRecord(ev[(size_t)(step % D)], P.st));
    G.steps = (uint64_t)step + 1;
  }
  HCK(hipStreamSynchronize(P.st));
  while (done < enq) retire();
  say("run complete: %llu exchanges", (unsigned long long)done);
}

[[noreturn]] static void srv_end(long hub_pid) {
  stop_watcher();
  print_digstat();
  rdv_publish("done_srv" + std::to_string(O.index), "ok\n", false);
  std::string t;
  if (!rdv_wait_file("done_hub" + std::to_string(O.hub_index), hub_pid,
                     kFinalWaitS, &t) &&
      pid_alive(hub_pid))
    say("WARNING: the hub did not acknowledge the end of the run");
  finish(kExitOk, "ok");
}

static void srv_main() {
  const std::string s = std::to_string(O.index),
                    h = std::to_string(O.hub_index);
  for (const std::string& n : {srv_name(O.index) + ".json", ack_name(O.index),
                               "done_srv" + s, "fault_srv" + s})
    rdv_remove(n);
  alloc_region(GEO.inbox_bytes);
  if (!rdv_publish(srv_name(O.index) + ".json", own_info(false), true))
    die_setup("cannot publish %s", rpath(srv_name(O.index) + ".json").c_str());
  std::string js;
  long hub_pid = 0;
  wait_peer_info(hub_name() + ".json", "the hub", &js, &hub_pid);
  check_peer_info(js, "the hub", true);
  P.peer[0] = open_peer(js, "the hub");
  say("hub: pid %ld arch %s bus %s alloc %s", hub_pid,
      js_str(js, "gcn").c_str(), js_str(js, "bus").c_str(),
      js_str(js, "alloc").c_str());
  srv_prepare();
  fill_ctx_common(2, (uint32_t)O.index);
  P.img.peer_base[0] = P.peer[0];
  P.img.ids32 = g_srv_ids32;
  if (O.srv_ws) {
    P.img.x_ws = g_srv_xws;
    P.img.w_ws = g_srv_wws;
  }
  upload_ctx();
  srv_capture_graphs();
  rdv_publish(ack_name(O.index), "1\n", true);
  std::string t;
  if (!rdv_wait_file("ready_h" + h, hub_pid, O.rdv_timeout_s, &t))
    die_setup("the hub did not become ready (%s)",
              pid_alive(hub_pid) ? "timeout" : "process gone");
  g_peers.push_back({0, hub_pid, -1, "the hub"});
  start_watcher();
  if (O.bench)
    srv_run_bench();
  else if (O.dev_verify)
    srv_run_devverify();
  else
    srv_run();
  srv_end(hub_pid);
}

// ---------------------------------------------------------------------------
// auxiliary roles and main
// ---------------------------------------------------------------------------
static int buses_main() {
  int n = 0;
  const hipError_t e = hipGetDeviceCount(&n);
  if (e != hipSuccess) {
    fprintf(stderr, "hipGetDeviceCount: %s\n", hipGetErrorString(e));
    return kExitSetup;
  }
  for (int i = 0; i < n; i++) {
    char bus[64] = {0};
    hipDeviceProp_t pr;
    if (hipDeviceGetPCIBusId(bus, sizeof bus, i) != hipSuccess ||
        hipGetDeviceProperties(&pr, i) != hipSuccess) {
      fprintf(stderr, "cannot query device %d\n", i);
      return kExitSetup;
    }
    std::string name = pr.name;
    std::replace(name.begin(), name.end(), ' ', '_');
    printf("HIPBUS idx=%d bus=%s arch=%s name=%s\n", i, bus, pr.gcnArchName,
           name.c_str());
  }
  fflush(stdout);
  return kExitOk;
}

static int selftest_main() {
  G.role = "selftest";
  const int bad = selftest_cpu(true);
  if (bad)
    say("selftest: FAIL, %d failure(s)", bad);
  else
    say("selftest: PASS");
  return bad ? kExitSetup : kExitOk;
}

static void common_start() {
  G.role = O.role == "hub" ? "hub" : "srv";
  G.index = O.role == "hub" ? O.hub_index : O.index;
  install_signals();
  if (under_tmp(O.dir))
    die_setup("refusing a rendezvous directory under /tmp: %s", O.dir.c_str());
  mkdir_p(O.dir);
  init_x16_table();
  if (selftest_cpu(false) != 0)
    die_setup(
        "the CPU reference failed its self-test (run: hx_loopback "
        "selftest)");
  device_init();
  calibrate();
  GEO = make_geo();
  say("geometry: rows_cap %d hidden %d topk %d nslots %d layers %d experts %d "
      "nsrv %d; slot_stride %llu, y_stride %llu, wait %lld ms, begin wait "
      "%lld ms; flush %d (%s) rd-mode %d (ctx.flush_mode 0x%02x)",
      O.rows_cap, O.hidden, O.topk, O.nslots, O.layers, O.experts, O.nsrv,
      (unsigned long long)GEO.slot_stride, (unsigned long long)GEO.y_stride,
      O.wait_ms, O.begin_wait_ms, O.flush, flush_names(O.flush).c_str(),
      O.rd_mode, (unsigned)ctx_flush_mode());
}

int main(int argc, char** argv) {
  G.t0_ns = now_ns();
  parse_args(argc, argv);
  validate_args();
  if (O.role == "selftest") return selftest_main();
  if (O.role == "buses") return buses_main();
  common_start();
  if (O.role == "hub")
    hub_main();
  else
    srv_main();
  return kExitOk;
}
