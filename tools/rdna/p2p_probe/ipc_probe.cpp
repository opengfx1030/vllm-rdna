// T1 probe for the hx protocol: cross-arch IPC and P2P (no torch, plain HIP).
//
// Answers, on BOTH arches (V620 gfx1030, W7800 gfx1100) and for every
// visibility configuration run_t1.sh drives, the questions of the hetero-MoE
// design (section 11 row T1, section 6.4): does hipIpcOpenMemHandle work
// across arches and disjoint visible sets, are kernel stores into an
// IPC-mapped buffer byte-exact, what are the one-way latency / fan-in /
// bandwidth numbers, is a polled flag ever stale, what is the wall_clock64
// tick rate, and which allocation kinds (coarse, uncached, finegrained) are
// peer-coherent.
//
// Build (fat binary, wave32 on both arches; run_t1.sh does this under the
// build lock):
//   hipcc -x hip -O3 -std=c++17 --offload-arch=gfx1030 --offload-arch=gfx1100
//     -o ipc_probe ipc_probe.cpp -pthread
//
// Single-process commands (devices are the caller's HIP_VISIBLE_DEVICES set):
//   ipc_probe buses                 HIP ordinal -> PCI bus + arch (no alloc)
//   ipc_probe info                  arch, bus, wall-clock attribute, uncached
//                                   and finegrained allocation, P2P attributes
//   ipc_probe tick [--tick-ms 1000 --tick-reps 5]
//                                   wall_clock64 vs the host steady clock
//   ipc_probe alloc                 32 MiB per allocation kind: memset + IPC
//                                   handle export
// Two-process roles (one hub process, --n-srv server processes; every process
// is pinned to its devices only by the caller's HIP_VISIBLE_DEVICES):
//   ipc_probe hub --dir D --n-srv N [--dev ORD]
//   ipc_probe srv --dir D --n-srv N --id I [--dev ORD]
//     common: --tests ipc_open,store,lat,poll,fanin,alloc_kinds
//             --kind auto|coarse|uncached|fine   (buffer kind; auto = uncached
//                                                 else coarse)
//             --poll relaxed|acquire|volfence    (flag load for store/fanin)
//             --sleep 0|1 (s_sleep in poll loops)  --fence none|agent|system
//             --dload plain|atomic  (payload loads of the verifying side)
//             --expect-arch gfx1100  --label TEXT  --quick
//             --wait-ms 10000        (device wait budget per flag wait)
//             --host-timeout-ms 120000  (rendezvous waits)
//             --kernel-timeout-s 900
//             --iters-16k --iters-1m --lat-iters --poll-iters --fanin-iters
//             --fanin-bw-iters --ak-iters --ak-poll-iters
//   --dev is the ordinal inside the process's visible set of the device the
//   role runs on (default 0). --dir must not be under /tmp.
//
// Output: one line per measurement, written with a single write(2):
//   T1 <test> role=<hub|srv<i>|-> [label=..] key=value ... PASS|FAIL|INFO
// Values never contain blanks. The exit code is non-zero if any line is FAIL
// (2 = runtime error, 3 = rendezvous/kernel hang, 64 = usage).
// The hub line of a test is the combined verdict (it also reports the
// server's counters); the server line carries the server's own counters.
// Targets (spec T1): lat p50 <= 9 us and p99 <= 20 us for 16 KiB + flag, fan-in
// 4 x 15,540 B p50 <= 15 us, 1 MiB fan-in >= 24 GB/s; alloc_kinds is INFO
// (which kinds are peer-coherent), everything else PASS/FAIL.
//
// Memory layout (both roles; every region is 4 KiB control page + 1 MiB):
//   server inbox  (size 1 region, written by the hub):  +0x000 flag word,
//                 +0x080 probe words (written by peer), +0x100 probe words
//                 (read by peer), +0x1000 payload
//   hub buffer    (size nact regions, region s written by server s): same
//                 layout, region s at s * (4096 + 1 MiB)
// Every flag is a u64 in the receiver's own buffer, polled locally; payload is
// pushed with posted stores, then __threadfence_system(), then a release
// store at system scope (the hx / rdna_ar producer order).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ipc_rdv.h"

using Clock = std::chrono::steady_clock;
using rdv::Kv;

// ---------------------------------------------------------------------------
// constants and shared (host + device) helpers
// ---------------------------------------------------------------------------
constexpr uint32_t kRegion = 4096 + (1u << 20);
constexpr size_t kOffPayload = 4096;
constexpr size_t kOffProbeW = 0x80;
constexpr size_t kOffProbeR = 0x100;
constexpr int kMaxSrv = 8;
constexpr int kMaxBlk = 64;
constexpr int kProbeWords = 8;

__host__ __device__ inline uint64_t mix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}
// 64-bit word i of iteration it: different for every (it, i).
__host__ __device__ inline uint64_t pat64(uint64_t it, uint64_t i) {
  return mix64(mix64(it) ^ (i * 0xD1B54A32D192ED03ull));
}
// 32-bit word w = half of pat64 (low half first), for payload tails.
__host__ __device__ inline uint32_t pat32(uint64_t it, uint64_t w) {
  uint64_t v = pat64(it, w >> 1);
  return (w & 1) ? (uint32_t)(v >> 32) : (uint32_t)v;
}

struct Stat {
  uint64_t mism, timeouts, ooo, polls, aux0, aux1, pad[2];
};
struct FirstBad {
  uint64_t claimed, it, word, got, exp, pad[3];
};

// ---------------------------------------------------------------------------
// device code (arch-neutral: wall_clock64, __hip_atomic_*, fences, s_sleep)
// ---------------------------------------------------------------------------
enum PollVariant { kRelaxed = 0, kAcquire = 1, kVolFence = 2 };

template <int V>
__device__ __forceinline__ uint64_t poll_load(const uint64_t* p) {
  if (V == kRelaxed)
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  if (V == kAcquire)
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
  uint64_t x = *(const volatile uint64_t*)p;
  __threadfence_system();
  return x;
}

__device__ __forceinline__ void store_release_sys(uint64_t* p, uint64_t v) {
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ uint64_t add_relaxed(uint64_t* p, uint64_t v) {
  return __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED,
                                __HIP_MEMORY_SCOPE_AGENT);
}

// Wait until *p >= want (u64, no wrap in this probe). The clock is read only
// every 256 spins, so a wait that is satisfied quickly costs no clock read.
// Returns false when `budget` ticks have passed without success.
template <int V>
__device__ bool wait_ge(const uint64_t* p, uint64_t want, uint64_t budget,
                        int do_sleep) {
  uint64_t t0 = 0;
  uint32_t spins = 0;
  for (;;) {
    if (poll_load<V>(p) >= want) return true;
    if (do_sleep) __builtin_amdgcn_s_sleep(1);
    if ((++spins & 255u) == 0) {
      uint64_t now = (uint64_t)wall_clock64();
      if (t0 == 0)
        t0 = now;
      else if (now - t0 > budget)
        return false;
    }
  }
}

// Write nbytes of pattern for iteration `it` (16 B vectors, then <= 3 u32
// tail words). gtid/gthreads: global thread index and count of the grid.
__device__ __forceinline__ void push_payload(uint8_t* dst, uint32_t nbytes,
                                             uint64_t it, uint32_t gtid,
                                             uint32_t gthreads) {
  const uint32_t nvec = nbytes / 16;
  uint4* d = (uint4*)dst;
  for (uint32_t c = gtid; c < nvec; c += gthreads) {
    uint64_t a = pat64(it, 2ull * c), b = pat64(it, 2ull * c + 1);
    d[c] = make_uint4((uint32_t)a, (uint32_t)(a >> 32), (uint32_t)b,
                      (uint32_t)(b >> 32));
  }
  const uint32_t tail = (nbytes - nvec * 16) / 4;
  if (gtid < tail)
    ((uint32_t*)dst)[nvec * 4 + gtid] = pat32(it, nvec * 4 + gtid);
}

__global__ void k_nop() {}

__global__ void k_wall(uint64_t* out) { *out = (uint64_t)wall_clock64(); }

// Push-and-flag loop. Initiator: push, flag, then wait for the peer's flag
// (wait_after). Responder: wait for the peer's flag, push, flag (wait_first).
// With several blocks the last arriving block releases the flag. samples
// (optional) gets the per-iteration round trip in device ticks (block 0).
struct PP {
  uint8_t* dst;
  uint64_t* push_flag;
  const uint64_t* my_flag;
  uint64_t* ctr;
  uint32_t* samples;
  Stat* st;
  uint64_t budget;
  uint32_t nbytes, iters;
  int wait_first, wait_after, do_sleep;
};

template <int V>
__global__ void k_pp(PP a) {
  __shared__ int s_ok;
  const uint32_t nblk = gridDim.x;
  const uint32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t gthreads = nblk * blockDim.x;
  for (uint32_t it = 1; it <= a.iters; it++) {
    if (a.wait_first) {
      if (threadIdx.x == 0)
        s_ok = wait_ge<V>(a.my_flag, it, a.budget, a.do_sleep);
      __syncthreads();
      if (!s_ok) {
        if (threadIdx.x == 0) add_relaxed(&a.st->timeouts, 1);
        return;
      }
    }
    uint64_t t0 = 0;
    if (a.samples && threadIdx.x == 0) t0 = (uint64_t)wall_clock64();
    push_payload(a.dst, a.nbytes, it, gtid, gthreads);
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
      uint64_t old = __hip_atomic_fetch_add(
          a.ctr, (uint64_t)1, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
      if (old + 1 == (uint64_t)nblk * it) store_release_sys(a.push_flag, it);
    }
    if (a.wait_after) {
      if (threadIdx.x == 0)
        s_ok = wait_ge<V>(a.my_flag, it, a.budget, a.do_sleep);
      __syncthreads();
      if (!s_ok) {
        if (threadIdx.x == 0) add_relaxed(&a.st->timeouts, 1);
        return;
      }
      if (a.samples && threadIdx.x == 0 && blockIdx.x == 0)
        a.samples[it - 1] = (uint32_t)((uint64_t)wall_clock64() - t0);
    }
  }
}

// Server side of the store test: wait for the flag, verify every word of the
// payload against the pattern of this iteration, count mismatches, ack by a
// release store into the hub's buffer.
struct VA {
  const uint8_t* src;
  uint64_t* ack;
  const uint64_t* flag;
  uint64_t* ctr;
  Stat* st;
  FirstBad* fb;
  uint64_t budget;
  uint32_t nbytes, iters;
  int do_sleep, fence, dload;
};

__device__ __forceinline__ void note_bad(const VA& a, uint64_t it,
                                         uint64_t word, uint64_t got,
                                         uint64_t exp) {
  FirstBad* f = &a.fb[blockIdx.x];
  uint64_t expected = 0;
  if (__hip_atomic_compare_exchange_strong(&f->claimed, &expected, (uint64_t)1,
                                           __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                           __HIP_MEMORY_SCOPE_AGENT)) {
    f->it = it;
    f->word = word;
    f->got = got;
    f->exp = exp;
  }
}

template <int V>
__global__ void k_verify(VA a) {
  __shared__ int s_ok;
  const uint32_t nvec = a.nbytes / 16;
  const uint32_t chunk = (nvec + gridDim.x - 1) / gridDim.x;
  const uint32_t lo = blockIdx.x * chunk;
  const uint32_t hi = (lo + chunk < nvec) ? lo + chunk : nvec;
  for (uint32_t it = 1; it <= a.iters; it++) {
    if (threadIdx.x == 0) s_ok = wait_ge<V>(a.flag, it, a.budget, a.do_sleep);
    __syncthreads();
    if (!s_ok) {
      if (threadIdx.x == 0) add_relaxed(&a.st->timeouts, 1);
      return;
    }
    // The flag was read by thread 0 only: every thread issues a fence at the
    // chosen scope (--fence) before reading the payload.
    if (a.fence == 2)
      __threadfence_system();
    else if (a.fence == 1)
      __threadfence();
    uint64_t bad = 0;
    for (uint32_t c = lo + threadIdx.x; c < hi; c += blockDim.x) {
      uint64_t g0, g1;
      const uint8_t* p = a.src + 16ull * c;
      if (a.dload == 0) {
        uint4 v = *(const uint4*)p;
        g0 = ((uint64_t)v.y << 32) | v.x;
        g1 = ((uint64_t)v.w << 32) | v.z;
      } else {
        g0 = __hip_atomic_load((const uint64_t*)p, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_SYSTEM);
        g1 = __hip_atomic_load((const uint64_t*)(p + 8), __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_SYSTEM);
      }
      uint64_t e0 = pat64(it, 2ull * c), e1 = pat64(it, 2ull * c + 1);
      if (g0 != e0) {
        bad++;
        note_bad(a, it, 2ull * c, g0, e0);
      }
      if (g1 != e1) {
        bad++;
        note_bad(a, it, 2ull * c + 1, g1, e1);
      }
    }
    if (blockIdx.x == 0) {  // tail words (nbytes % 16)
      const uint32_t tail = (a.nbytes - nvec * 16) / 4;
      if (threadIdx.x < tail) {
        uint32_t w = nvec * 4 + threadIdx.x;
        uint32_t g = ((const uint32_t*)a.src)[w], e = pat32(it, w);
        if (g != e) {
          bad++;
          note_bad(a, it, w, g, e);
        }
      }
    }
    if (bad) add_relaxed(&a.st->mism, bad);
    __syncthreads();
    if (threadIdx.x == 0) {
      uint64_t old = __hip_atomic_fetch_add(
          a.ctr, (uint64_t)1, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
      if (old + 1 == (uint64_t)gridDim.x * it) store_release_sys(a.ack, it);
    }
  }
}

// Hub side of the fan-in: lane s stores go=it into server s and waits for the
// server's flag in the hub buffer; per-iteration go -> all flags seen time.
struct FH {
  uint64_t* go[kMaxSrv];
  const uint64_t* done[kMaxSrv];
  uint32_t* samples;
  Stat* st;
  uint64_t budget;
  uint32_t iters;
  int n, do_sleep;
};

template <int V>
__global__ void k_fanin_hub(FH a) {
  __shared__ uint64_t s_t0;
  __shared__ int s_bad;
  if (threadIdx.x == 0) s_bad = 0;
  __syncthreads();
  for (uint32_t it = 1; it <= a.iters; it++) {
    if (threadIdx.x == 0) s_t0 = (uint64_t)wall_clock64();
    __syncthreads();
    if ((int)threadIdx.x < a.n) {
      store_release_sys(a.go[threadIdx.x], it);
      if (!wait_ge<V>(a.done[threadIdx.x], it, a.budget, a.do_sleep)) s_bad = 1;
    }
    __syncthreads();
    if (s_bad) {
      if (threadIdx.x == 0) add_relaxed(&a.st->timeouts, 1);
      return;
    }
    if (threadIdx.x == 0)
      a.samples[it - 1] = (uint32_t)((uint64_t)wall_clock64() - s_t0);
  }
}

// Poll test. The writer waits a random gap, stamps the device clock, release
// stores the next value into the poller's memory and waits for the ack in its
// own memory; the sample is store -> ack in device ticks (one device clock).
struct PW {
  uint64_t* flag;
  const uint64_t* ack;
  uint32_t* samples;
  Stat* st;
  uint64_t budget, seed;
  uint32_t iters, ticks_per_us;
  int do_sleep;
};

template <int V>
__global__ void k_poll_writer(PW a) {
  if (threadIdx.x != 0) return;
  uint64_t r = a.seed | 1;
  for (uint32_t it = 1; it <= a.iters; it++) {
    r ^= r << 13;
    r ^= r >> 7;
    r ^= r << 17;
    // 1 in 16 gaps is long (up to 300 us), the rest are short (up to 6 us).
    uint64_t ns = ((r & 15) == 0) ? (r >> 8) % 300000u : (r >> 8) % 6000u;
    uint64_t gap = ns * a.ticks_per_us / 1000;
    uint64_t g0 = (uint64_t)wall_clock64();
    while ((uint64_t)wall_clock64() - g0 < gap) {
    }
    uint64_t t0 = (uint64_t)wall_clock64();
    store_release_sys(a.flag, it);
    if (!wait_ge<V>(a.ack, it, a.budget, a.do_sleep)) {
      add_relaxed(&a.st->timeouts, 1);
      return;
    }
    a.samples[it - 1] = (uint32_t)((uint64_t)wall_clock64() - t0);
  }
}

struct PR {
  const uint64_t* flag;
  uint64_t* ack;
  Stat* st;
  uint64_t budget;
  uint32_t iters;
  int do_sleep;
};

// Every observed value must be >= the previous observation and <= the
// iteration being waited for (the writer never runs ahead of the ack).
template <int V>
__global__ void k_poll_poller(PR a) {
  if (threadIdx.x != 0) return;
  uint64_t prev = 0, ooo = 0, polls = 0, tmo = 0;
  for (uint32_t it = 1; it <= a.iters; it++) {
    uint64_t t0 = 0;
    uint32_t spins = 0;
    bool got = false;
    for (;;) {
      uint64_t v = poll_load<V>(a.flag);
      polls++;
      if (v < prev || v > it) ooo++;
      if (v > prev) prev = v;
      if (v >= it) {
        got = true;
        break;
      }
      if (a.do_sleep) __builtin_amdgcn_s_sleep(1);
      if ((++spins & 255u) == 0) {
        uint64_t now = (uint64_t)wall_clock64();
        if (t0 == 0)
          t0 = now;
        else if (now - t0 > a.budget)
          break;
      }
    }
    if (!got) {
      tmo = 1;
      break;
    }
    store_release_sys(a.ack, it);
  }
  add_relaxed(&a.st->polls, polls);
  add_relaxed(&a.st->ooo, ooo);
  add_relaxed(&a.st->timeouts, tmo);
}

// ipc_open data probes: write 8 words to / read 8 words from a remote buffer.
__global__ void k_write_words(uint64_t* dst, uint64_t tag) {
  if (threadIdx.x < kProbeWords) {
    dst[threadIdx.x] = pat64(tag, threadIdx.x);
  }
  __threadfence_system();
}
__global__ void k_read_words(const uint64_t* src, uint64_t* out) {
  if (threadIdx.x < kProbeWords)
    out[threadIdx.x] = __hip_atomic_load(src + threadIdx.x, __ATOMIC_RELAXED,
                                         __HIP_MEMORY_SCOPE_SYSTEM);
}

// ---------------------------------------------------------------------------
// host: output, errors, options
// ---------------------------------------------------------------------------
static std::string g_role = "-", g_label;
static bool g_fail = false;

static std::string tok(const std::string& s) {
  std::string r = s;
  for (char& ch : r)
    if (isspace((unsigned char)ch) || ch == '=') ch = '_';
  return r.empty() ? "-" : r;
}

__attribute__((format(printf, 3, 4))) static void emit(const char* test,
                                                       const char* verdict,
                                                       const char* fmt, ...) {
  char body[2048];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(body, sizeof body, fmt, ap);
  va_end(ap);
  std::string line = std::string("T1 ") + test + " role=" + g_role;
  if (!g_label.empty()) line += " label=" + g_label;
  line += std::string(" ") + body + " " + verdict + "\n";
  if (!strcmp(verdict, "FAIL")) g_fail = true;
  size_t off = 0;
  while (off < line.size()) {
    ssize_t n = ::write(1, line.data() + off, line.size() - off);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) break;
    off += (size_t)n;
  }
}

// Leave without running HIP teardown (a hung kernel must not block exit).
[[noreturn]] static void bail(int code) {
  rdv::cleanup_all();
  _exit(code);
}

static void hck(const char* test, hipError_t e, const char* expr,
                const char* file, int line) {
  if (e == hipSuccess) return;
  emit(test, "FAIL", "call=%s error=%s src=%s:%d", tok(expr).c_str(),
       hipGetErrorName(e), file, line);
  bail(2);
}
#define HCK(test, x) hck(test, (x), #x, __FILE__, __LINE__)

struct Opts {
  std::string cmd, dir, tests = "ipc_open,store,lat,poll,fanin,alloc_kinds";
  std::string kind = "auto", expect_arch;
  int dev = 0, id = 0, nsrv = 1;
  int poll = kAcquire, sleep = 0, fence = 2, dload = 0;
  double tick_ms = 1000;
  int tick_reps = 5;
  long iters16k = 1000000, iters1m = 10000, lat_iters = 100000,
       poll_iters = 200000, fanin_iters = 20000, fanin_bw_iters = 2000,
       ak_iters = 100000, ak_poll_iters = 50000;
  int big_blocks = 16;
  long wait_ms = 10000, host_timeout_ms = 120000, kernel_timeout_s = 900;
  bool quick = false;
};

[[noreturn]] static void usage(const char* why) {
  fprintf(stderr,
          "ipc_probe: %s\nusage: ipc_probe buses|info|tick|alloc | "
          "hub|srv --dir D --n-srv N [--id I] [--dev ORD] [--tests ...] "
          "(see the header comment of ipc_probe.cpp)\n",
          why);
  exit(64);
}

static int parse_poll(const std::string& s) {
  if (s == "relaxed") return kRelaxed;
  if (s == "acquire") return kAcquire;
  if (s == "volfence") return kVolFence;
  usage("--poll must be relaxed|acquire|volfence");
}
static const char* poll_name(int v) {
  return v == kRelaxed ? "relaxed" : v == kAcquire ? "acquire" : "volfence";
}
static const char* fence_name(int f) {
  return f == 2 ? "system" : f == 1 ? "agent" : "none";
}

static Opts parse_args(int argc, char** argv) {
  Opts o;
  if (argc < 2) usage("missing command");
  o.cmd = argv[1];
  for (int i = 2; i < argc; i++) {
    std::string a = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) usage(("missing value for " + a).c_str());
      return argv[++i];
    };
    auto num = [&]() -> long { return strtol(val().c_str(), nullptr, 0); };
    if (a == "--dir")
      o.dir = val();
    else if (a == "--tests")
      o.tests = val();
    else if (a == "--kind")
      o.kind = val();
    else if (a == "--expect-arch")
      o.expect_arch = val();
    else if (a == "--label")
      g_label = tok(val());
    else if (a == "--dev")
      o.dev = (int)num();
    else if (a == "--id")
      o.id = (int)num();
    else if (a == "--n-srv")
      o.nsrv = (int)num();
    else if (a == "--poll")
      o.poll = parse_poll(val());
    else if (a == "--sleep")
      o.sleep = (int)num();
    else if (a == "--fence") {
      std::string v = val();
      o.fence = v == "system" ? 2 : v == "agent" ? 1 : 0;
      if (v != "system" && v != "agent" && v != "none")
        usage("--fence must be none|agent|system");
    } else if (a == "--dload") {
      std::string v = val();
      if (v != "plain" && v != "atomic") usage("--dload must be plain|atomic");
      o.dload = v == "atomic";
    } else if (a == "--tick-ms")
      o.tick_ms = atof(val().c_str());
    else if (a == "--tick-reps")
      o.tick_reps = (int)num();
    else if (a == "--iters-16k")
      o.iters16k = num();
    else if (a == "--iters-1m")
      o.iters1m = num();
    else if (a == "--lat-iters")
      o.lat_iters = num();
    else if (a == "--poll-iters")
      o.poll_iters = num();
    else if (a == "--fanin-iters")
      o.fanin_iters = num();
    else if (a == "--fanin-bw-iters")
      o.fanin_bw_iters = num();
    else if (a == "--ak-iters")
      o.ak_iters = num();
    else if (a == "--ak-poll-iters")
      o.ak_poll_iters = num();
    else if (a == "--blocks")
      o.big_blocks = (int)num();
    else if (a == "--wait-ms")
      o.wait_ms = num();
    else if (a == "--host-timeout-ms")
      o.host_timeout_ms = num();
    else if (a == "--kernel-timeout-s")
      o.kernel_timeout_s = num();
    else if (a == "--quick")
      o.quick = true;
    else
      usage(("unknown option " + a).c_str());
  }
  if (o.quick) {
    o.iters16k = 20000;
    o.iters1m = 500;
    o.lat_iters = 10000;
    o.poll_iters = 20000;
    o.fanin_iters = 2000;
    o.fanin_bw_iters = 200;
    o.ak_iters = 10000;
    o.ak_poll_iters = 5000;
  }
  if (o.nsrv < 1 || o.nsrv > kMaxSrv) usage("--n-srv must be 1..8");
  if (o.big_blocks < 1 || o.big_blocks > kMaxBlk) usage("--blocks 1..64");
  return o;
}

// ---------------------------------------------------------------------------
// host: HIP helpers
// ---------------------------------------------------------------------------
enum Kind { kCoarse = 0, kUncached = 1, kFine = 2, kAuto = 3 };
static const char* kind_name(Kind k) {
  return k == kCoarse     ? "coarse"
         : k == kUncached ? "uncached"
         : k == kFine     ? "finegrained"
                          : "auto";
}
static Kind parse_kind(const std::string& s) {
  if (s == "coarse") return kCoarse;
  if (s == "uncached") return kUncached;
  if (s == "fine" || s == "finegrained") return kFine;
  if (s == "auto") return kAuto;
  usage("--kind must be auto|coarse|uncached|fine");
}
static hipError_t alloc_kind(void** p, size_t n, Kind k) {
  switch (k) {
    case kUncached:
      return hipExtMallocWithFlags(p, n, hipDeviceMallocUncached);
    case kFine:
      return hipExtMallocWithFlags(p, n, hipDeviceMallocFinegrained);
    default:
      return hipMalloc(p, n);
  }
}
static Kind resolve_kind(Kind k) {
  if (k != kAuto) return k;
  void* p = nullptr;
  if (alloc_kind(&p, 1 << 20, kUncached) == hipSuccess) {
    (void)hipFree(p);
    return kUncached;
  }
  (void)hipGetLastError();
  return kCoarse;
}

struct DevInfo {
  std::string name, arch, bus;
  int wall_khz = 0, cus = 0, warp = 0;
};
static DevInfo dev_info(int d) {
  DevInfo r;
  hipDeviceProp_t p;
  HCK("info", hipGetDeviceProperties(&p, d));
  r.name = tok(p.name);
  r.arch = tok(p.gcnArchName);
  char bus[64] = {0};
  HCK("info", hipDeviceGetPCIBusId(bus, sizeof bus, d));
  r.bus = bus;
  (void)hipDeviceGetAttribute(&r.wall_khz, hipDeviceAttributeWallClockRate, d);
  (void)hipDeviceGetAttribute(&r.cus, hipDeviceAttributeMultiprocessorCount, d);
  (void)hipDeviceGetAttribute(&r.warp, hipDeviceAttributeWarpSize, d);
  return r;
}

// Device ticks per host millisecond: kernel stamp, host busy-wait, kernel
// stamp. The host clock is read right after each synchronise, so the launch
// and wake-up latency cancels. Median of `reps`; never compares two devices.
static std::vector<double> calibrate_samples(int dev, double ms, int reps) {
  HCK("tick", hipSetDevice(dev));
  uint64_t* d = nullptr;
  HCK("tick", hipMalloc((void**)&d, 16));
  hipStream_t s;
  HCK("tick", hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  k_wall<<<1, 1, 0, s>>>(d);  // warm up (code object load)
  HCK("tick", hipStreamSynchronize(s));
  std::vector<double> r;
  for (int i = 0; i < reps; i++) {
    k_wall<<<1, 1, 0, s>>>(d);
    HCK("tick", hipStreamSynchronize(s));
    auto h0 = Clock::now();
    auto end = h0 + std::chrono::microseconds((long)(ms * 1000));
    while (Clock::now() < end) {
    }
    k_wall<<<1, 1, 0, s>>>(d + 1);
    HCK("tick", hipStreamSynchronize(s));
    auto h1 = Clock::now();
    uint64_t v[2];
    HCK("tick", hipMemcpy(v, d, 16, hipMemcpyDeviceToHost));
    double dt_ms = std::chrono::duration<double, std::milli>(h1 - h0).count();
    r.push_back((double)(v[1] - v[0]) / dt_ms);
  }
  HCK("tick", hipStreamDestroy(s));
  HCK("tick", hipFree(d));
  return r;
}
static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// ---------------------------------------------------------------------------
// single-process commands
// ---------------------------------------------------------------------------
static int cmd_buses() {
  int n = 0;
  HCK("buses", hipGetDeviceCount(&n));
  for (int d = 0; d < n; d++) {
    DevInfo i = dev_info(d);
    emit("buses", "INFO", "dev=%d bus=%s arch=%s name=%s", d, i.bus.c_str(),
         i.arch.c_str(), i.name.c_str());
  }
  return g_fail;
}

static int cmd_info() {
  int n = 0;
  HCK("info", hipGetDeviceCount(&n));
  for (int d = 0; d < n; d++) {
    DevInfo i = dev_info(d);
    HCK("info", hipSetDevice(d));
    void* p = nullptr;
    hipError_t eu = hipExtMallocWithFlags(&p, 1 << 20, hipDeviceMallocUncached);
    if (eu == hipSuccess) (void)hipFree(p);
    (void)hipGetLastError();
    p = nullptr;
    hipError_t ef =
        hipExtMallocWithFlags(&p, 1 << 20, hipDeviceMallocFinegrained);
    if (ef == hipSuccess) (void)hipFree(p);
    (void)hipGetLastError();
    emit("info", "INFO",
         "dev=%d arch=%s name=%s bus=%s wall_clock_khz=%d cus=%d warp=%d "
         "uncached=%s finegrained=%s",
         d, i.arch.c_str(), i.name.c_str(), i.bus.c_str(), i.wall_khz, i.cus,
         i.warp, hipGetErrorName(eu), hipGetErrorName(ef));
  }
  for (int a = 0; a < n; a++)
    for (int b = 0; b < n; b++) {
      if (a == b) continue;
      int can = -1, acc = -1, rank = -1, atom = -1;
      hipError_t e1 = hipDeviceCanAccessPeer(&can, a, b);
      hipError_t e2 =
          hipDeviceGetP2PAttribute(&acc, hipDevP2PAttrAccessSupported, a, b);
      hipError_t e3 =
          hipDeviceGetP2PAttribute(&rank, hipDevP2PAttrPerformanceRank, a, b);
      hipError_t e4 = hipDeviceGetP2PAttribute(
          &atom, hipDevP2PAttrNativeAtomicSupported, a, b);
      emit("info_p2p", "INFO",
           "pair=%d->%d can_access=%d(%s) access=%d(%s) perf_rank=%d(%s) "
           "native_atomics=%d(%s)",
           a, b, can, hipGetErrorName(e1), acc, hipGetErrorName(e2), rank,
           hipGetErrorName(e3), atom, hipGetErrorName(e4));
    }
  return g_fail;
}

static int cmd_tick(const Opts& o) {
  int n = 0;
  HCK("tick", hipGetDeviceCount(&n));
  for (int d = 0; d < n; d++) {
    DevInfo i = dev_info(d);
    std::vector<double> r = calibrate_samples(d, o.tick_ms, o.tick_reps);
    double med = median(r);
    double mn = *std::min_element(r.begin(), r.end());
    double mx = *std::max_element(r.begin(), r.end());
    double err = i.wall_khz > 0 ? (med - i.wall_khz) / i.wall_khz * 100.0 : 0.0;
    bool ok = i.wall_khz > 0 && std::abs(err) <= 5.0;
    emit(
        "tick", ok ? "PASS" : "INFO",
        "dev=%d arch=%s attr_ticks_per_ms=%d measured_ticks_per_ms=%.1f "
        "min=%.1f max=%.1f spread_pct=%.4f err_pct=%.3f reps=%d window_ms=%.0f "
        "use=%s",
        d, i.arch.c_str(), i.wall_khz, med, mn, mx, (mx - mn) / med * 100.0,
        err, o.tick_reps, o.tick_ms, ok ? "attribute" : "measured");
  }
  return g_fail;
}

static int cmd_alloc() {
  int n = 0;
  HCK("alloc", hipGetDeviceCount(&n));
  const size_t bytes = 32u << 20;
  for (int d = 0; d < n; d++) {
    HCK("alloc", hipSetDevice(d));
    DevInfo i = dev_info(d);
    for (Kind k : {kCoarse, kUncached, kFine}) {
      void* p = nullptr;
      hipError_t ea = alloc_kind(&p, bytes, k), em = hipErrorNotReady,
                 eh = hipErrorNotReady;
      int ptype = -1;
      if (ea == hipSuccess) {
        em = hipMemset(p, 0xA5, bytes);
        if (em == hipSuccess) em = hipDeviceSynchronize();
        hipIpcMemHandle_t h;
        memset(&h, 0, sizeof h);
        eh = hipIpcGetMemHandle(&h, p);
        hipPointerAttribute_t at;
        if (hipPointerGetAttributes(&at, p) == hipSuccess) ptype = (int)at.type;
        (void)hipFree(p);
      }
      (void)hipGetLastError();
      bool ok = ea == hipSuccess && em == hipSuccess && eh == hipSuccess;
      emit("alloc", ok ? "PASS" : "INFO",
           "dev=%d arch=%s kind=%s bytes=%zu alloc=%s memset=%s ipc_export=%s "
           "ptr_type=%d",
           d, i.arch.c_str(), kind_name(k), bytes, hipGetErrorName(ea),
           ea == hipSuccess ? hipGetErrorName(em) : "skipped",
           ea == hipSuccess ? hipGetErrorName(eh) : "skipped", ptype);
    }
  }
  return g_fail;
}

// ---------------------------------------------------------------------------
// two-process roles
// ---------------------------------------------------------------------------
struct Ctx {
  Opts o;
  rdv::Rdv rv;
  bool hub = false;
  int id = 0;
  std::string me;
  DevInfo di;
  Kind kind = kAuto;
  double tpms = 0, tpus = 0;  // measured device ticks per ms / us
  hipStream_t st = nullptr;
  Stat* d_st = nullptr;
  FirstBad* d_fb = nullptr;
  uint64_t* d_ctr = nullptr;
  uint32_t* d_samples = nullptr;
  size_t samples_cap = 0;
  uint64_t budget() const { return (uint64_t)((double)o.wait_ms * tpms); }
  std::vector<std::string> active(int nact) const {
    std::vector<std::string> v = {"hub"};
    for (int i = 0; i < nact; i++) v.push_back("srv" + std::to_string(i));
    return v;
  }
  std::vector<std::string> everyone() const { return active(o.nsrv); }
  static std::string srv_name(int i) { return "srv" + std::to_string(i); }
};

struct Bufs {
  std::string tag;
  Kind kind = kCoarse;
  int nact = 1;
  void* own = nullptr;
  size_t own_bytes = 0;
  void* peer[kMaxSrv] = {};  // hub: server inboxes; server: peer[0] = hub buf
  Kv peer_kv[kMaxSrv];
  hipError_t open_err[kMaxSrv] = {};
  bool ok = false;
  std::string why;
};

static char* region(void* base, int r) {
  return (char*)base + (size_t)r * kRegion;
}

static void must(Ctx& c, bool ok, const char* test, const char* what) {
  if (ok) return;
  emit(test, "FAIL", "rendezvous=%s err=%s", tok(what).c_str(),
       tok(c.rv.error()).c_str());
  bail(3);
}
static void sync(Ctx& c, const char* test, const std::string& name, int nact) {
  must(c, c.rv.barrier(name, c.active(nact), c.o.host_timeout_ms), test,
       name.c_str());
}

static bool wait_kernel(Ctx& c, const char* test) {
  int64_t t0 = rdv::now_ms();
  for (;;) {
    hipError_t e = hipStreamQuery(c.st);
    if (e == hipSuccess) return true;
    if (e != hipErrorNotReady) {
      emit(test, "FAIL", "call=hipStreamQuery error=%s", hipGetErrorName(e));
      bail(2);
    }
    if (rdv::now_ms() - t0 > c.o.kernel_timeout_s * 1000) {
      emit(test, "FAIL", "kernel_hang_s=%ld", c.o.kernel_timeout_s);
      bail(3);
    }
    rdv::sleep_us(200);
  }
}

#define LAUNCH_V(K, V, G, B, S, ...)               \
  do {                                             \
    switch (V) {                                   \
      case kRelaxed:                               \
        K<kRelaxed><<<G, B, 0, S>>>(__VA_ARGS__);  \
        break;                                     \
      case kAcquire:                               \
        K<kAcquire><<<G, B, 0, S>>>(__VA_ARGS__);  \
        break;                                     \
      default:                                     \
        K<kVolFence><<<G, B, 0, S>>>(__VA_ARGS__); \
        break;                                     \
    }                                              \
    HCK("launch", hipGetLastError());              \
  } while (0)

static void reset_state(Ctx& c, Bufs& b) {
  const int nreg = c.hub ? b.nact : 1;
  for (int r = 0; r < nreg; r++)
    HCK("reset", hipMemset(region(b.own, r), 0, 4096));
  HCK("reset", hipMemset(c.d_st, 0, sizeof(Stat)));
  HCK("reset", hipMemset(c.d_fb, 0, kMaxBlk * sizeof(FirstBad)));
  HCK("reset", hipMemset(c.d_ctr, 0, sizeof(uint64_t)));
  HCK("reset", hipMemset(c.d_samples, 0, c.samples_cap * sizeof(uint32_t)));
  HCK("reset", hipDeviceSynchronize());
}

static Stat read_stat(Ctx& c) {
  Stat s;
  HCK("stat", hipMemcpy(&s, c.d_st, sizeof s, hipMemcpyDeviceToHost));
  return s;
}

static std::string first_bad(Ctx& c) {
  FirstBad f[kMaxBlk];
  HCK("stat", hipMemcpy(f, c.d_fb, sizeof f, hipMemcpyDeviceToHost));
  int best = -1;
  for (int i = 0; i < kMaxBlk; i++)
    if (f[i].claimed && (best < 0 || f[i].it < f[best].it ||
                         (f[i].it == f[best].it && f[i].word < f[best].word)))
      best = i;
  if (best < 0) return "none";
  char buf[160];
  snprintf(buf, sizeof buf, "it%llu:word%llu:got0x%llx:exp0x%llx",
           (unsigned long long)f[best].it, (unsigned long long)f[best].word,
           (unsigned long long)f[best].got, (unsigned long long)f[best].exp);
  return buf;
}

static const char* mem_type_name(int t) {
  switch (t) {
    case hipMemoryTypeHost:
      return "host";
    case hipMemoryTypeDevice:
      return "device";
    case hipMemoryTypeManaged:
      return "managed";
    default:
      return "other";
  }
}

// Allocate this role's buffer (kind `want`), export it, import the peers'
// buffers with hipIpcMemLazyEnablePeerAccess. The decision go/nogo is made by
// the hub (it sees every server's status) so all processes agree. Returns
// b.ok; b.why says what failed. Every active process must call it with the
// same tag/nact.
static bool setup_bufs(Ctx& c, Bufs& b, Kind want, int nact,
                       const std::string& tag, const char* test) {
  b = Bufs();
  b.tag = tag;
  b.nact = nact;
  b.kind = want;
  b.own_bytes = c.hub ? (size_t)nact * kRegion : (size_t)kRegion;
  const std::string t = tag;
  hipIpcMemHandle_t h;
  memset(&h, 0, sizeof h);
  std::string mine_err;
  hipError_t e = alloc_kind(&b.own, b.own_bytes, want);
  if (e != hipSuccess) {
    mine_err = std::string("alloc:") + hipGetErrorName(e);
    b.own = nullptr;
  } else if ((e = hipMemset(b.own, 0, b.own_bytes)) != hipSuccess ||
             (e = hipDeviceSynchronize()) != hipSuccess) {
    mine_err = std::string("memset:") + hipGetErrorName(e);
  } else if ((e = hipIpcGetMemHandle(&h, b.own)) != hipSuccess) {
    mine_err = std::string("ipc_export:") + hipGetErrorName(e);
  }
  (void)hipGetLastError();
  char dev[16];
  snprintf(dev, sizeof dev, "%d", c.o.dev);
  must(c,
       c.rv.publish(c.me + "." + t + ".buf",
                    rdv::kv_text({{"ok", mine_err.empty() ? "1" : "0"},
                                  {"err", mine_err.empty() ? "none" : mine_err},
                                  {"handle", rdv::hex_encode(&h, sizeof h)},
                                  {"bytes", std::to_string(b.own_bytes)},
                                  {"kind", kind_name(want)},
                                  {"dev", dev},
                                  {"arch", c.di.arch},
                                  {"bus", c.di.bus}}),
                    true),
       test, "publish buffer");
  std::vector<std::string> peers;
  if (c.hub)
    for (int i = 0; i < nact; i++) peers.push_back(Ctx::srv_name(i));
  else
    peers.push_back("hub");
  bool all_ok = mine_err.empty();
  b.why = mine_err;
  for (size_t i = 0; i < peers.size(); i++) {
    std::string txt;
    must(c,
         c.rv.wait_read(peers[i] + "." + t + ".buf", c.o.host_timeout_ms, &txt),
         test, "wait peer buffer");
    b.peer_kv[i] = rdv::kv_parse(txt);
    b.open_err[i] = hipSuccess;
    if (rdv::kv_get(b.peer_kv[i], "ok") != "1") {
      b.open_err[i] = hipErrorNotInitialized;
      if (all_ok)
        b.why = "peer_" + peers[i] + ":" + rdv::kv_get(b.peer_kv[i], "err");
      all_ok = false;
      continue;
    }
    hipIpcMemHandle_t ph;
    if (!rdv::hex_decode(rdv::kv_get(b.peer_kv[i], "handle"), &ph, sizeof ph)) {
      b.open_err[i] = hipErrorInvalidValue;
      all_ok = false;
      continue;
    }
    void* p = nullptr;
    hipError_t eo = hipIpcOpenMemHandle(&p, ph, hipIpcMemLazyEnablePeerAccess);
    b.open_err[i] = eo;
    if (eo == hipSuccess) {
      b.peer[i] = p;
    } else {
      if (all_ok) b.why = std::string("open:") + hipGetErrorName(eo);
      all_ok = false;
      (void)hipGetLastError();
    }
  }
  must(c,
       c.rv.publish(c.me + "." + t + ".opened",
                    rdv::kv_text({{"ok", all_ok ? "1" : "0"},
                                  {"why", b.why.empty() ? "none" : b.why}}),
                    true),
       test, "publish opened");
  if (c.hub) {
    for (int i = 0; i < nact; i++) {
      std::string txt;
      must(c,
           c.rv.wait_read(Ctx::srv_name(i) + "." + t + ".opened",
                          c.o.host_timeout_ms, &txt),
           test, "wait peer opened");
      Kv kv = rdv::kv_parse(txt);
      if (rdv::kv_get(kv, "ok") != "1" && all_ok) {
        all_ok = false;
        b.why = Ctx::srv_name(i) + ":" + rdv::kv_get(kv, "why");
      }
    }
    must(
        c,
        c.rv.publish("hub." + t + ".verdict", all_ok ? "go\n" : "nogo\n", true),
        test, "publish verdict");
  } else {
    std::string v;
    must(c, c.rv.wait_read("hub." + t + ".verdict", c.o.host_timeout_ms, &v),
         test, "wait verdict");
    bool go = v.rfind("go", 0) == 0;
    if (!go && all_ok) b.why = "peer_failed";
    all_ok = go;
  }
  b.ok = all_ok;
  return b.ok;
}

static void teardown_bufs(Ctx& c, Bufs& b, const char* test) {
  for (int i = 0; i < kMaxSrv; i++)
    if (b.peer[i]) {
      (void)hipIpcCloseMemHandle(b.peer[i]);
      b.peer[i] = nullptr;
    }
  sync(c, test, b.tag + ".closed", b.nact);
  if (b.own) (void)hipFree(b.own);
  b.own = nullptr;
}

static void setup_failed(Ctx& c, const char* test, const Bufs& b) {
  emit(test, "FAIL", "stage=setup kind=%s why=%s", kind_name(b.kind),
       tok(b.why).c_str());
  (void)c;
}

// ---- percentile helper ----------------------------------------------------
struct Pct {
  double p50 = 0, p99 = 0, max = 0, min = 0, mean = 0;
  size_t n = 0;
};
static Pct pct(std::vector<uint32_t> v, size_t skip, double div) {
  Pct r;
  if (v.size() <= skip) return r;
  v.erase(v.begin(), v.begin() + skip);
  std::sort(v.begin(), v.end());
  r.n = v.size();
  r.p50 = v[v.size() / 2] / div;
  r.p99 = v[(v.size() * 99) / 100] / div;
  r.max = v.back() / div;
  r.min = v.front() / div;
  double s = 0;
  for (uint32_t x : v) s += x;
  r.mean = s / v.size() / div;
  return r;
}
static std::vector<uint32_t> read_samples(Ctx& c, size_t n) {
  std::vector<uint32_t> v(n);
  HCK("samples", hipMemcpy(v.data(), c.d_samples, n * sizeof(uint32_t),
                           hipMemcpyDeviceToHost));
  return v;
}

// ---- ipc_open -------------------------------------------------------------
static void test_ipc_open(Ctx& c) {
  const char* T = "ipc_open";
  Bufs b;
  const int nact = c.o.nsrv;
  bool ok = setup_bufs(c, b, c.kind, nact, "io", T);
  const int npeer = c.hub ? nact : 1;
  // Per-peer open report (exact error and pointer attributes).
  int ndev = 0;
  (void)hipGetDeviceCount(&ndev);
  if (!ok) {
    for (int i = 0; i < npeer; i++) {
      const std::string pname = c.hub ? Ctx::srv_name(i) : "hub";
      const Kv& kv = b.peer_kv[i];
      emit(T, "FAIL",
           "peer=%s open=%s stage=%s peer_arch=%s peer_bus=%s peer_dev=%s "
           "peer_kind=%s own_kind=%s own_arch=%s",
           pname.c_str(), hipGetErrorName(b.open_err[i]), tok(b.why).c_str(),
           rdv::kv_get(kv, "arch", "na").c_str(),
           rdv::kv_get(kv, "bus", "na").c_str(),
           rdv::kv_get(kv, "dev", "na").c_str(),
           rdv::kv_get(kv, "kind", "na").c_str(), kind_name(b.kind),
           c.di.arch.c_str());
    }
    teardown_bufs(c, b, T);
    return;
  }
  // Data probes: each side fills 8 words at +0x100 of its own regions, the
  // peer reads them with a kernel (remote read) and writes 8 words at +0x80
  // of the peer's regions (remote write); the owner then checks via D2H.
  auto fill_tag = [&](bool hubside, int i) -> uint64_t {
    return (hubside ? 0x1000u : 0x2000u) + i;
  };
  auto write_tag = [&](bool hubside, int i) -> uint64_t {
    return (hubside ? 0x3000u : 0x4000u) + i;
  };
  uint64_t* d_out = nullptr;
  HCK(T, hipMalloc((void**)&d_out, 8 * kProbeWords * kMaxSrv));
  for (int i = 0; i < npeer; i++) {
    int r = c.hub ? i : 0;  // region index inside the owner's buffer
    uint64_t w[kProbeWords];
    int me_i = c.hub ? i : c.id;  // hub region index == server index
    for (int k = 0; k < kProbeWords; k++)
      w[k] = pat64(fill_tag(c.hub, me_i), k);
    HCK(T, hipMemcpy(region(b.own, r) + kOffProbeR, w, sizeof w,
                     hipMemcpyHostToDevice));
  }
  sync(c, T, "io.filled", nact);
  int srv_i = c.id;
  for (int i = 0; i < npeer; i++) {
    char* pbase = c.hub ? (char*)b.peer[i] : region(b.peer[0], srv_i);
    k_read_words<<<1, 32, 0, c.st>>>((const uint64_t*)(pbase + kOffProbeR),
                                     d_out + i * kProbeWords);
    HCK(T, hipGetLastError());
    k_write_words<<<1, 32, 0, c.st>>>((uint64_t*)(pbase + kOffProbeW),
                                      write_tag(c.hub, c.hub ? i : srv_i));
    HCK(T, hipGetLastError());
  }
  wait_kernel(c, T);
  sync(c, T, "io.done", nact);
  for (int i = 0; i < npeer; i++) {
    const std::string pname = c.hub ? Ctx::srv_name(i) : "hub";
    int me_i = c.hub ? i : srv_i;  // server index of this link
    int r = c.hub ? i : 0;
    uint64_t got[kProbeWords], rd[kProbeWords];
    HCK(T, hipMemcpy(got, region(b.own, r) + kOffProbeW, sizeof got,
                     hipMemcpyDeviceToHost));
    HCK(T, hipMemcpy(rd, d_out + i * kProbeWords, sizeof rd,
                     hipMemcpyDeviceToHost));
    int wbad = 0, rbad = 0;
    for (int k = 0; k < kProbeWords; k++) {
      if (got[k] != pat64(write_tag(!c.hub, me_i), k)) wbad++;
      if (rd[k] != pat64(fill_tag(!c.hub, me_i), k)) rbad++;
    }
    hipPointerAttribute_t at;
    memset(&at, 0, sizeof at);
    hipError_t ep = hipPointerGetAttributes(&at, b.peer[i]);
    (void)hipGetLastError();
    std::string pdev_arch = "na", pdev_bus = "na";
    if (ep == hipSuccess && at.device >= 0 && at.device < ndev) {
      hipDeviceProp_t pr;
      char bus[64] = {0};
      if (hipGetDeviceProperties(&pr, at.device) == hipSuccess)
        pdev_arch = tok(pr.gcnArchName);
      if (hipDeviceGetPCIBusId(bus, sizeof bus, at.device) == hipSuccess)
        pdev_bus = bus;
    }
    const Kv& kv = b.peer_kv[i];
    std::string peer_bus = rdv::kv_get(kv, "bus", "na");
    bool maps_peer = pdev_bus != "na" && pdev_bus == peer_bus;
    int can = -1;
    if (ep == hipSuccess && at.device >= 0 && at.device < ndev)
      (void)hipDeviceCanAccessPeer(&can, c.o.dev, at.device);
    bool pass = !wbad && !rbad;
    emit(T, pass ? "PASS" : "FAIL",
         "peer=%s open=hipSuccess ptr_attr=%s ptr_type=%s ptr_device=%d "
         "ptr_device_arch=%s ptr_device_bus=%s ptr_maps_peer_bus=%s "
         "can_access_ptr_device=%d own_dev=%d own_arch=%s peer_dev_in_peer=%s "
         "peer_arch=%s peer_bus=%s own_kind=%s peer_kind=%s visible=%d "
         "remote_write_bad_words=%d remote_read_bad_words=%d",
         pname.c_str(), hipGetErrorName(ep), mem_type_name((int)at.type),
         at.device, pdev_arch.c_str(), pdev_bus.c_str(),
         maps_peer ? "yes" : "no", can, c.o.dev, c.di.arch.c_str(),
         rdv::kv_get(kv, "dev", "na").c_str(),
         rdv::kv_get(kv, "arch", "na").c_str(), peer_bus.c_str(),
         kind_name(b.kind), rdv::kv_get(kv, "kind", "na").c_str(), ndev, wbad,
         rbad);
  }
  (void)hipFree(d_out);
  teardown_bufs(c, b, T);
}

// ---- store ----------------------------------------------------------------
struct StoreOut {
  uint64_t mism = 0, srv_timeouts = 0, hub_timeouts = 0;
  double secs = 0;
  std::string first = "none";
  bool ok() const { return !mism && !srv_timeouts && !hub_timeouts; }
};

static StoreOut run_store(Ctx& c, Bufs& b, const char* test,
                          const std::string& sub, uint32_t nbytes,
                          uint32_t iters, uint32_t blocks, bool print) {
  StoreOut out;
  const std::string pre = std::string(test) + "." + b.tag + "." + sub;
  reset_state(c, b);
  sync(c, test, pre + ".go", b.nact);
  const char* fnc = fence_name(c.o.fence);
  if (c.hub) {
    PP a{};
    a.dst = (uint8_t*)b.peer[0] + kOffPayload;
    a.push_flag = (uint64_t*)b.peer[0];
    a.my_flag = (const uint64_t*)b.own;
    a.ctr = c.d_ctr;
    a.samples = nullptr;
    a.st = c.d_st;
    a.budget = c.budget();
    a.nbytes = nbytes;
    a.iters = iters;
    a.wait_first = 0;
    a.wait_after = 1;
    a.do_sleep = c.o.sleep;
    auto t0 = Clock::now();
    LAUNCH_V(k_pp, c.o.poll, blocks, 256, c.st, a);
    wait_kernel(c, test);
    out.secs = std::chrono::duration<double>(Clock::now() - t0).count();
    out.hub_timeouts = read_stat(c).timeouts;
    sync(c, test, pre + ".fin", b.nact);
    std::string txt;
    must(c, c.rv.wait_read("res." + pre + ".srv0", c.o.host_timeout_ms, &txt),
         test, "wait server result");
    Kv kv = rdv::kv_parse(txt);
    out.mism = (uint64_t)rdv::kv_int(kv, "mism");
    out.srv_timeouts = (uint64_t)rdv::kv_int(kv, "timeouts");
    out.first = rdv::kv_get(kv, "first", "none");
    if (print)
      emit(test, out.ok() ? "PASS" : "FAIL",
           "size=%u iters=%u blocks=%u poll=%s fence=%s dload=%s sleep=%d "
           "kind=%s peer_kind=%s secs=%.3f iters_per_s=%.0f mismatches=%llu "
           "first_bad=%s srv_timeouts=%llu hub_timeouts=%llu",
           nbytes, iters, blocks, poll_name(c.o.poll), fnc,
           c.o.dload ? "atomic" : "plain", c.o.sleep, kind_name(b.kind),
           rdv::kv_get(b.peer_kv[0], "kind", "na").c_str(), out.secs,
           out.secs > 0 ? iters / out.secs : 0.0, (unsigned long long)out.mism,
           out.first.c_str(), (unsigned long long)out.srv_timeouts,
           (unsigned long long)out.hub_timeouts);
  } else {
    VA a{};
    a.src = (const uint8_t*)b.own + kOffPayload;
    a.ack = (uint64_t*)region(b.peer[0], c.id);
    a.flag = (const uint64_t*)b.own;
    a.ctr = c.d_ctr;
    a.st = c.d_st;
    a.fb = c.d_fb;
    a.budget = c.budget();
    a.nbytes = nbytes;
    a.iters = iters;
    a.do_sleep = c.o.sleep;
    a.fence = c.o.fence;
    a.dload = c.o.dload;
    auto t0 = Clock::now();
    LAUNCH_V(k_verify, c.o.poll, blocks, 256, c.st, a);
    wait_kernel(c, test);
    out.secs = std::chrono::duration<double>(Clock::now() - t0).count();
    Stat s = read_stat(c);
    out.mism = s.mism;
    out.srv_timeouts = s.timeouts;
    out.first = first_bad(c);
    must(c,
         c.rv.publish("res." + pre + "." + c.me,
                      rdv::kv_text({{"mism", std::to_string(out.mism)},
                                    {"timeouts", std::to_string(s.timeouts)},
                                    {"first", out.first}})),
         test, "publish result");
    sync(c, test, pre + ".fin", b.nact);
    if (print)
      emit(test, out.ok() ? "PASS" : "FAIL",
           "size=%u iters=%u blocks=%u poll=%s fence=%s dload=%s sleep=%d "
           "kind=%s secs=%.3f mismatches=%llu first_bad=%s timeouts=%llu",
           nbytes, iters, blocks, poll_name(c.o.poll), fnc,
           c.o.dload ? "atomic" : "plain", c.o.sleep, kind_name(b.kind),
           out.secs, (unsigned long long)out.mism, out.first.c_str(),
           (unsigned long long)out.srv_timeouts);
  }
  return out;
}

static void test_store(Ctx& c) {
  const char* T = "store";
  Bufs b;
  if (!setup_bufs(c, b, c.kind, 1, "st", T)) {
    setup_failed(c, T, b);
    teardown_bufs(c, b, T);
    return;
  }
  run_store(c, b, T, "16k", 16384, (uint32_t)c.o.iters16k, 1, true);
  run_store(c, b, T, "1m", 1u << 20, (uint32_t)c.o.iters1m,
            (uint32_t)c.o.big_blocks, true);
  teardown_bufs(c, b, T);
}

// ---- lat ------------------------------------------------------------------
static void run_lat(Ctx& c, Bufs& b, const char* test, uint32_t nbytes,
                    uint32_t iters, int V) {
  const uint32_t warm = std::min<uint32_t>(1000, iters / 10);
  const uint32_t total = iters + warm;
  const std::string pre = std::string(test) + "." + b.tag + "." +
                          std::to_string(nbytes) + "." + poll_name(V);
  reset_state(c, b);
  sync(c, test, pre + ".go", b.nact);
  PP a{};
  a.st = c.d_st;
  a.ctr = c.d_ctr;
  a.budget = c.budget();
  a.nbytes = nbytes;
  a.iters = total;
  a.do_sleep = c.o.sleep;
  if (c.hub) {
    a.dst = (uint8_t*)b.peer[0] + kOffPayload;
    a.push_flag = (uint64_t*)b.peer[0];
    a.my_flag = (const uint64_t*)b.own;
    a.samples = c.d_samples;
    a.wait_first = 0;
    a.wait_after = 1;
  } else {
    char* hr = region(b.peer[0], c.id);
    a.dst = (uint8_t*)hr + kOffPayload;
    a.push_flag = (uint64_t*)hr;
    a.my_flag = (const uint64_t*)b.own;
    a.samples = nullptr;
    a.wait_first = 1;
    a.wait_after = 0;
  }
  LAUNCH_V(k_pp, V, 1, 256, c.st, a);
  wait_kernel(c, test);
  Stat s = read_stat(c);
  must(c,
       c.rv.publish("res." + pre + "." + c.me,
                    rdv::kv_text({{"timeouts", std::to_string(s.timeouts)}})),
       test, "publish result");
  sync(c, test, pre + ".fin", b.nact);
  std::string txt;
  must(c,
       c.rv.wait_read("res." + pre + "." + (c.hub ? "srv0" : "hub"),
                      c.o.host_timeout_ms, &txt),
       test, "wait peer result");
  uint64_t peer_t = (uint64_t)rdv::kv_int(rdv::kv_parse(txt), "timeouts");
  if (!c.hub) {
    emit(test, s.timeouts || peer_t ? "FAIL" : "INFO",
         "size=%u variant=%s sleep=%d iters=%u timeouts=%llu hub_timeouts=%llu",
         nbytes, poll_name(V), c.o.sleep, total, (unsigned long long)s.timeouts,
         (unsigned long long)peer_t);
    return;
  }
  Pct p = pct(read_samples(c, total), warm, c.tpus * 2.0);  // one way, us
  const bool target = nbytes == 16384;
  bool pass =
      !s.timeouts && !peer_t && (!target || (p.p50 <= 9.0 && p.p99 <= 20.0));
  emit(test,
       (s.timeouts || peer_t) ? "FAIL"
       : target               ? (pass ? "PASS" : "FAIL")
                              : "INFO",
       "size=%u variant=%s sleep=%d iters=%zu p50_us=%.2f p99_us=%.2f "
       "max_us=%.2f min_us=%.2f mean_us=%.2f target_p50_us=%s "
       "target_p99_us=%s ticks_per_us=%.3f timeouts=%llu srv_timeouts=%llu "
       "kind=%s peer_kind=%s",
       nbytes, poll_name(V), c.o.sleep, p.n, p.p50, p.p99, p.max, p.min, p.mean,
       target ? "9" : "na", target ? "20" : "na", c.tpus,
       (unsigned long long)s.timeouts, (unsigned long long)peer_t,
       kind_name(b.kind), rdv::kv_get(b.peer_kv[0], "kind", "na").c_str());
}

static void test_lat(Ctx& c) {
  const char* T = "lat";
  Bufs b;
  if (!setup_bufs(c, b, c.kind, 1, "lat", T)) {
    setup_failed(c, T, b);
    teardown_bufs(c, b, T);
    return;
  }
  for (uint32_t size : {16384u, 0u})
    for (int v : {kRelaxed, kAcquire, kVolFence})
      run_lat(c, b, T, size, (uint32_t)c.o.lat_iters, v);
  teardown_bufs(c, b, T);
}

// ---- poll -----------------------------------------------------------------
struct PollOut {
  bool ok = true;
  uint64_t slow = 0, ooo = 0, timeouts = 0, poller_timeouts = 0;
};

// dir 0: hub writes, server polls; dir 1: server writes, hub polls.
static PollOut run_poll(Ctx& c, Bufs& b, const char* test, int dir, int V,
                        uint32_t iters, bool print) {
  PollOut out;
  const bool writer = (dir == 0) == c.hub;
  const char* dname = dir == 0 ? "h2s" : "s2h";
  const std::string pre =
      std::string(test) + "." + b.tag + "." + dname + "." + poll_name(V);
  reset_state(c, b);
  sync(c, test, pre + ".go", b.nact);
  char* peer_ctl = c.hub ? (char*)b.peer[0] : region(b.peer[0], c.id);
  if (writer) {
    PW a{};
    a.flag = (uint64_t*)peer_ctl;
    a.ack = (const uint64_t*)b.own;
    a.samples = c.d_samples;
    a.st = c.d_st;
    a.budget = c.budget();
    a.seed = 0x9E3779B97F4A7C15ull ^ ((uint64_t)V << 8) ^ (uint64_t)dir;
    a.iters = iters;
    a.ticks_per_us = (uint32_t)(c.tpus + 0.5);
    a.do_sleep = c.o.sleep;
    LAUNCH_V(k_poll_writer, V, 1, 32, c.st, a);
  } else {
    PR a{};
    a.flag = (const uint64_t*)b.own;
    a.ack = (uint64_t*)peer_ctl;
    a.st = c.d_st;
    a.budget = c.budget();
    a.iters = iters;
    a.do_sleep = c.o.sleep;
    LAUNCH_V(k_poll_poller, V, 1, 32, c.st, a);
  }
  wait_kernel(c, test);
  Stat s = read_stat(c);
  must(c,
       c.rv.publish("res." + pre + "." + c.me,
                    rdv::kv_text({{"timeouts", std::to_string(s.timeouts)},
                                  {"ooo", std::to_string(s.ooo)},
                                  {"polls", std::to_string(s.polls)}})),
       test, "publish result");
  sync(c, test, pre + ".fin", b.nact);
  std::string txt;
  must(c,
       c.rv.wait_read("res." + pre + "." + (c.hub ? "srv0" : "hub"),
                      c.o.host_timeout_ms, &txt),
       test, "wait peer result");
  Kv peer = rdv::kv_parse(txt);
  const uint64_t peer_t = (uint64_t)rdv::kv_int(peer, "timeouts");
  if (!writer) {
    out.ooo = s.ooo;
    out.timeouts = s.timeouts;
    out.poller_timeouts = s.timeouts;
    out.ok = !s.ooo && !s.timeouts && !peer_t;
    if (print)
      emit(test, out.ok ? "PASS" : "FAIL",
           "dir=%s variant=%s sleep=%d iters=%u polls=%llu out_of_order=%llu "
           "timeouts=%llu writer_timeouts=%llu kind=%s",
           dname, poll_name(V), c.o.sleep, iters, (unsigned long long)s.polls,
           (unsigned long long)s.ooo, (unsigned long long)s.timeouts,
           (unsigned long long)peer_t, kind_name(b.kind));
    return out;
  }
  std::vector<uint32_t> v = read_samples(c, iters);
  if (s.timeouts) {  // aborted early: keep the filled prefix only
    size_t k = 0;
    while (k < v.size() && v[k]) k++;
    v.resize(k);
  }
  Pct p = pct(v, 0, c.tpus);
  uint64_t slow = 0;
  if (!v.empty()) {
    std::vector<uint32_t> sorted = v;
    std::sort(sorted.begin(), sorted.end());
    double med = sorted[sorted.size() / 2];
    for (uint32_t x : v)
      if (x > 100.0 * med) slow++;
  }
  out.slow = slow;
  out.ooo = (uint64_t)rdv::kv_int(peer, "ooo");
  out.timeouts = s.timeouts;
  out.poller_timeouts = peer_t;
  out.ok = !slow && !out.ooo && !s.timeouts && !peer_t;
  if (print)
    emit(test, out.ok ? "PASS" : "FAIL",
         "dir=%s variant=%s sleep=%d iters=%u rtt_p50_us=%.2f rtt_p99_us=%.2f "
         "rtt_max_us=%.2f slow_gt100x_median=%llu out_of_order=%llu "
         "timeouts=%llu poller_timeouts=%llu polls=%lld kind=%s peer_kind=%s",
         dname, poll_name(V), c.o.sleep, iters, p.p50, p.p99, p.max,
         (unsigned long long)slow, (unsigned long long)out.ooo,
         (unsigned long long)s.timeouts, (unsigned long long)peer_t,
         rdv::kv_int(peer, "polls"), kind_name(b.kind),
         rdv::kv_get(b.peer_kv[0], "kind", "na").c_str());
  return out;
}

static void test_poll(Ctx& c) {
  const char* T = "poll";
  Bufs b;
  if (!setup_bufs(c, b, c.kind, 1, "poll", T)) {
    setup_failed(c, T, b);
    teardown_bufs(c, b, T);
    return;
  }
  for (int dir : {0, 1})
    for (int v : {kRelaxed, kAcquire, kVolFence})
      run_poll(c, b, T, dir, v, (uint32_t)c.o.poll_iters, true);
  teardown_bufs(c, b, T);
}

// ---- fanin ----------------------------------------------------------------
static void run_fanin(Ctx& c, Bufs& b, const char* test, const char* what,
                      uint32_t nbytes, uint32_t iters, uint32_t blocks) {
  const int n = b.nact;
  const std::string pre = std::string(test) + "." + b.tag + "." + what;
  reset_state(c, b);
  sync(c, test, pre + ".go", n);
  if (c.hub) {
    FH a{};
    for (int s = 0; s < n; s++) {
      a.go[s] = (uint64_t*)b.peer[s];
      a.done[s] = (const uint64_t*)region(b.own, s);
    }
    a.samples = c.d_samples;
    a.st = c.d_st;
    a.budget = c.budget();
    a.iters = iters;
    a.n = n;
    a.do_sleep = c.o.sleep;
    auto t0 = Clock::now();
    LAUNCH_V(k_fanin_hub, c.o.poll, 1, 32, c.st, a);
    wait_kernel(c, test);
    double secs = std::chrono::duration<double>(Clock::now() - t0).count();
    Stat s = read_stat(c);
    sync(c, test, pre + ".fin", n);
    // Last iteration's payload of every server must be exact.
    uint64_t bad = 0;
    if (!s.timeouts) {
      std::vector<uint32_t> h(nbytes / 4);
      for (int sr = 0; sr < n; sr++) {
        HCK(test, hipMemcpy(h.data(), region(b.own, sr) + kOffPayload, nbytes,
                            hipMemcpyDeviceToHost));
        for (uint32_t w = 0; w < nbytes / 4; w++)
          if (h[w] != pat32(iters, w)) bad++;
      }
    }
    Pct p = pct(read_samples(c, iters), std::min<uint32_t>(100, iters / 10),
                c.tpus);
    const bool is_lat = nbytes < (1u << 20);
    // Aggregate bandwidth: bytes of all servers per iteration / iteration time.
    double gbps_mean = p.mean > 0 ? (double)n * nbytes / (p.mean * 1e3) : 0;
    double gbps_p50 = p.p50 > 0 ? (double)n * nbytes / (p.p50 * 1e3) : 0;
    bool pass = !s.timeouts && !bad;
    if (is_lat)
      pass = pass && p.p50 <= 15.0;
    else
      pass = pass && gbps_mean >= 24.0;
    emit(test, pass ? "PASS" : "FAIL",
         "what=%s n=%d bytes_per_srv=%u iters=%u poll=%s p50_us=%.2f "
         "p99_us=%.2f max_us=%.2f mean_us=%.2f agg_gbps_mean=%.2f "
         "agg_gbps_p50=%.2f secs=%.3f target=%s verify_bad_words=%llu "
         "timeouts=%llu kind=%s peer_kind=%s",
         what, n, nbytes, iters, poll_name(c.o.poll), p.p50, p.p99, p.max,
         p.mean, gbps_mean, gbps_p50, secs,
         is_lat ? "p50_le_15us" : "agg_ge_24GBps", (unsigned long long)bad,
         (unsigned long long)s.timeouts, kind_name(b.kind),
         rdv::kv_get(b.peer_kv[0], "kind", "na").c_str());
  } else {
    PP a{};
    char* hr = region(b.peer[0], c.id);
    a.dst = (uint8_t*)hr + kOffPayload;
    a.push_flag = (uint64_t*)hr;
    a.my_flag = (const uint64_t*)b.own;
    a.ctr = c.d_ctr;
    a.samples = nullptr;
    a.st = c.d_st;
    a.budget = c.budget();
    a.nbytes = nbytes;
    a.iters = iters;
    a.wait_first = 1;
    a.wait_after = 0;
    a.do_sleep = c.o.sleep;
    LAUNCH_V(k_pp, c.o.poll, blocks, 256, c.st, a);
    wait_kernel(c, test);
    Stat s = read_stat(c);
    sync(c, test, pre + ".fin", n);
    emit(test, s.timeouts ? "FAIL" : "INFO",
         "what=%s n=%d bytes_per_srv=%u iters=%u timeouts=%llu", what, n,
         nbytes, iters, (unsigned long long)s.timeouts);
  }
}

static void test_fanin(Ctx& c) {
  const char* T = "fanin";
  Bufs b;
  const int n = c.o.nsrv;
  if (!setup_bufs(c, b, c.kind, n, "fan", T)) {
    setup_failed(c, T, b);
    teardown_bufs(c, b, T);
    return;
  }
  run_fanin(c, b, T, "lat15540", 15540, (uint32_t)c.o.fanin_iters, 1);
  run_fanin(c, b, T, "bw1m", 1u << 20, (uint32_t)c.o.fanin_bw_iters,
            (uint32_t)c.o.big_blocks);
  teardown_bufs(c, b, T);
}

// ---- alloc_kinds ------------------------------------------------------------
static void test_alloc_kinds(Ctx& c) {
  const char* T = "alloc_kinds";
  for (Kind k : {kCoarse, kUncached, kFine}) {
    Bufs b;
    std::string tag = std::string("ak.") + kind_name(k);
    if (!setup_bufs(c, b, k, 1, tag, T)) {
      emit(T, "INFO", "kind=%s setup=failed why=%s coherent=na", kind_name(k),
           tok(b.why).c_str());
      teardown_bufs(c, b, T);
      continue;
    }
    StoreOut so =
        run_store(c, b, T, "16k", 16384, (uint32_t)c.o.ak_iters, 1, false);
    PollOut p1 =
        run_poll(c, b, T, 0, kRelaxed, (uint32_t)c.o.ak_poll_iters, false);
    PollOut p2 =
        run_poll(c, b, T, 1, kRelaxed, (uint32_t)c.o.ak_poll_iters, false);
    PollOut p3 =
        run_poll(c, b, T, 0, kAcquire, (uint32_t)c.o.ak_poll_iters, false);
    PollOut p4 =
        run_poll(c, b, T, 1, kAcquire, (uint32_t)c.o.ak_poll_iters, false);
    bool coherent = so.ok() && p1.ok && p2.ok && p3.ok && p4.ok;
    emit(T, coherent ? "PASS" : "INFO",
         "kind=%s peer_kind=%s coherent=%s store_mismatches=%llu "
         "store_timeouts=%llu poll_relaxed_h2s=%s poll_relaxed_s2h=%s "
         "poll_acquire_h2s=%s poll_acquire_s2h=%s slow_total=%llu "
         "ooo_total=%llu "
         "timeouts_total=%llu",
         kind_name(k), rdv::kv_get(b.peer_kv[0], "kind", "na").c_str(),
         coherent ? "yes" : "no", (unsigned long long)so.mism,
         (unsigned long long)(so.srv_timeouts + so.hub_timeouts),
         p1.ok ? "ok" : "bad", p2.ok ? "ok" : "bad", p3.ok ? "ok" : "bad",
         p4.ok ? "ok" : "bad",
         (unsigned long long)(p1.slow + p2.slow + p3.slow + p4.slow),
         (unsigned long long)(p1.ooo + p2.ooo + p3.ooo + p4.ooo),
         (unsigned long long)(p1.timeouts + p1.poller_timeouts + p2.timeouts +
                              p2.poller_timeouts + p3.timeouts +
                              p3.poller_timeouts + p4.timeouts +
                              p4.poller_timeouts));
    teardown_bufs(c, b, T);
  }
}

// ---- role driver
// --------------------------------------------------------------
static int run_role(Opts& o, bool hub) {
  Ctx c;
  c.o = o;
  c.hub = hub;
  c.id = hub ? 0 : o.id;
  c.me = hub ? "hub" : Ctx::srv_name(o.id);
  g_role = c.me;
  if (!hub && (o.id < 0 || o.id >= o.nsrv)) usage("--id must be < --n-srv");
  if (o.dir.empty()) usage("--dir is required");
  if (!c.rv.open(o.dir, c.me)) {
    emit("rdv", "FAIL", "err=%s", tok(c.rv.error()).c_str());
    return 3;
  }
  for (const std::string& p : c.everyone()) c.rv.watch(p);
  int ndev = 0;
  HCK("init", hipGetDeviceCount(&ndev));
  if (o.dev < 0 || o.dev >= ndev) {
    emit("init", "FAIL", "dev=%d visible=%d", o.dev, ndev);
    return 2;
  }
  HCK("init", hipSetDevice(o.dev));
  c.di = dev_info(o.dev);
  if (!o.expect_arch.empty() && c.di.arch.rfind(o.expect_arch, 0) != 0) {
    emit("init", "FAIL", "dev=%d arch=%s expected=%s bus=%s", o.dev,
         c.di.arch.c_str(), o.expect_arch.c_str(), c.di.bus.c_str());
    return 2;
  }
  k_nop<<<1, 1>>>();
  HCK("init", hipGetLastError());
  HCK("init", hipDeviceSynchronize());
  c.tpms = median(calibrate_samples(o.dev, 100, 5));
  c.tpus = c.tpms / 1000.0;
  c.kind = resolve_kind(parse_kind(o.kind));
  HCK("init", hipStreamCreateWithFlags(&c.st, hipStreamNonBlocking));
  c.samples_cap =
      (size_t)std::max<long>({o.lat_iters + 1000, o.poll_iters, o.fanin_iters,
                              o.fanin_bw_iters, o.ak_poll_iters}) +
      16;
  HCK("init", hipMalloc((void**)&c.d_st, sizeof(Stat)));
  HCK("init", hipMalloc((void**)&c.d_fb, kMaxBlk * sizeof(FirstBad)));
  HCK("init", hipMalloc((void**)&c.d_ctr, sizeof(uint64_t)));
  HCK("init",
      hipMalloc((void**)&c.d_samples, c.samples_cap * sizeof(uint32_t)));
  emit("hello", "INFO",
       "dev=%d visible=%d arch=%s name=%s bus=%s wall_clock_khz=%d "
       "measured_ticks_per_ms=%.1f buffer_kind=%s n_srv=%d",
       o.dev, ndev, c.di.arch.c_str(), c.di.name.c_str(), c.di.bus.c_str(),
       c.di.wall_khz, c.tpms, kind_name(c.kind), o.nsrv);
  must(c, c.rv.barrier("hello", c.everyone(), o.host_timeout_ms), "hello",
       "barrier hello");

  std::vector<std::string> tests;
  for (size_t pos = 0; pos <= o.tests.size();) {
    size_t e = o.tests.find(',', pos);
    if (e == std::string::npos) e = o.tests.size();
    if (e > pos) tests.push_back(o.tests.substr(pos, e - pos));
    pos = e + 1;
  }
  for (const std::string& t : tests) {
    int nact = (t == "ipc_open" || t == "fanin") ? o.nsrv : 1;
    if (t != "ipc_open" && t != "store" && t != "lat" && t != "poll" &&
        t != "fanin" && t != "alloc_kinds")
      usage(("unknown test " + t).c_str());
    must(c, c.rv.barrier("t." + t + ".begin", c.everyone(), o.host_timeout_ms),
         t.c_str(), "begin");
    if (hub || o.id < nact) {
      if (t == "ipc_open")
        test_ipc_open(c);
      else if (t == "store")
        test_store(c);
      else if (t == "lat")
        test_lat(c);
      else if (t == "poll")
        test_poll(c);
      else if (t == "fanin")
        test_fanin(c);
      else
        test_alloc_kinds(c);
    }
    // Servers that sit a test out wait here for the whole test, so the bound
    // is the kernel bound, not the rendezvous bound.
    must(c,
         c.rv.barrier("t." + t + ".end", c.everyone(),
                      std::max(o.host_timeout_ms, o.kernel_timeout_s * 1000)),
         t.c_str(), "end");
  }
  c.rv.cleanup();
  return g_fail ? 1 : 0;
}

int main(int argc, char** argv) {
  Opts o = parse_args(argc, argv);
  if (o.cmd == "buses") return cmd_buses();
  if (o.cmd == "info") return cmd_info();
  if (o.cmd == "tick") return cmd_tick(o);
  if (o.cmd == "alloc") return cmd_alloc();
  if (o.cmd == "hub") return run_role(o, true);
  if (o.cmd == "srv") return run_role(o, false);
  usage("unknown command");
}
