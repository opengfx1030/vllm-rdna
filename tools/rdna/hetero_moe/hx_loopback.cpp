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
  int nslots = 2, flush = 0, kmode = 0, kfixed = 0, verify_every = -1;
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
  --flush 0|1          ctx.flush_mode
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
  --pipeline D         bench: server steps in flight (default 3)
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
    else if (a == "--wait-ms")
      O.wait_ms = to_ll(val(a), a);
    else if (a == "--begin-wait-ms")
      O.begin_wait_ms = to_ll(val(a), a);
    else if (a == "--plan-hash")
      O.plan_hash = strtoull(val(a).c_str(), nullptr, 16);
    else if (a == "--experts")
      O.experts = (int)to_ll(val(a), a);
    else if (a == "--layers")
      O.layers = (int)to_ll(val(a), a);
    else if (a == "--rows-cap")
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
    else
      die_usage("unknown option '%s'", a.c_str());
  }
}

static void validate_args() {
  auto need = [](bool ok, const char* msg) {
    if (!ok) die_usage("%s", msg);
  };
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
  need(O.flush == 0 || O.flush == 1, "--flush must be 0 or 1");
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
  need(O.kmode != 1 || (O.kfixed >= 0 && O.kfixed <= 512),
       "--k-mode fixed:K needs K in [0, 512]");
  need(O.pipeline >= 1 && O.pipeline <= kDescRing - 2 && O.warmup_steps >= 0,
       "--pipeline must be in [1, 62] and --warmup-steps >= 0");
  need(O.srv_ws == 0 || O.srv_ws == 1, "--srv-ws must be 0 or 1");
  need(O.layer_stats >= -1 && O.layer_stats <= 1,
       "--layer-stats must be 0 or 1");
  if (O.layer_stats < 0) O.layer_stats = O.bench ? 0 : 1;
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
static bool whole_steps() { return O.graph || O.bench; }
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
  static const char* n[] = {"none",     "DISPATCH",  "COMBINE",
                            "SRV_WAIT", "SRV_BEGIN", "HDR",
                            "PUSH",     "PLAN",      "HOST"};
  return p < 9 ? n[p] : "?";
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
                "kill_to_exit_ms=%.1f kill_to_detect_ms=%.1f\n",
                G.arch[0] ? G.arch : "-", G.alloc[0] ? G.alloc : "-", G.tpm,
                G.watcher_on ? 1 : 0, G.peer_dead.load(),
                fault > 0 ? (double)(t_exit - fault) / 1e6 : -1.0,
                fault > 0 && det > 0 ? (double)(det - fault) / 1e6 : -1.0);
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
  c.flush_mode = (uint32_t)O.flush;
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

static void hub_dump_servers(int layer, int T, const RefTabs& rt,
                             const int8_t* xq) {
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
    if (bad == 0)
      say("  reply of server %d: identical to the reference", s);
    else
      say("  reply of server %d: %zu of %zu differ, first (r=%zu,c=%zu) got "
          "0x%04x expected 0x%04x",
          s, bad, n, first / O.hidden, first % O.hidden, got[first],
          want[first]);
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
    if (!O.graph) hub_dump_servers(layer, T, rts[i], B.xq.data());
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
  if (O.bench) {
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
    if (!O.graph)
      for (int l = 0; l < L; l++) {
        hipEvent_t a, b;
        HCK(hipEventCreate(&a));
        HCK(hipEventCreate(&b));
        g_ev0.push_back(a);
        g_ev1.push_back(b);
      }
  }
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

[[noreturn]] static void hub_end(const std::vector<long>& pids) {
  stop_watcher();
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

[[noreturn]] static void srv_end(long hub_pid) {
  stop_watcher();
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
      "%lld ms",
      O.rows_cap, O.hidden, O.topk, O.nslots, O.layers, O.experts, O.nsrv,
      (unsigned long long)GEO.slot_stride, (unsigned long long)GEO.y_stride,
      O.wait_ms, O.begin_wait_ms);
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
