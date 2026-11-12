// P2P microbenchmark for the par1-cs25 PLX topology (2026-11-06).
//
// Measures what rdna_ar uses: kernel-issued posted stores into a peer GPU's
// VRAM (push), kernel loads from a peer (pull), and hipMemcpyPeerAsync (sdma),
// plus flag ping-pong latency with an optional payload push before the flag.
//
// Device ids on the command line are logical (index into HIP_VISIBLE_DEVICES).
//
//   p2p_probe info
//   p2p_probe lat   <iters> <payload_bytes> <a:b> [<a:b> ...]
//   p2p_probe bw    <seconds> <bytes> <blocks> <xfer> [<xfer> ...]
//       xfer = [bg:]<p|l|s>:<src>><dst>   p=push (stores), l=pull (loads),
//       s=hipMemcpyPeerAsync. All xfers run concurrently for <seconds>;
//       bg: xfers are reported in a separate aggregate (interference runs).
//   p2p_probe version
//   p2p_probe buses
//   p2p_probe hbw   <seconds> <bytes> <blocks> <hxfer> [<hxfer> ...]
//       hxfer = [bg:]<mode>:<dev>, mode = h2d d2h h2d_pg d2h_pg h2d_k d2h_k
//       (pinned copy, pageable copy, zero-copy kernel). Prints HBW and HAGG.
//   p2p_probe hlat  <iters> <devs_csv> <size> [<size> ...]
//   p2p_probe hflag <iters> <dev> [<dev> ...]
//   p2p_probe curve <push|pull|sdma> <iters> <blocks> <sizes_csv> <a:b> [...]
//       latency percentiles per size; blocks 0 = automatic.
//   p2p_probe batch   one command per stdin line; prints READY once, then per
//       command a ### echo, its output and an END line.
//   Option --no-peer (before the command): no peer access and no per-GPU
//   buffers; then only version, buses, hbw, hlat, hflag and batch work.
//   A failing HIP call exits the process with status 1 (a batch ends
//   without END).
//
// Build: hipcc -x hip -O3 --offload-arch=gfx1030 --offload-arch=gfx1100
//   -o p2p_probe p2p_probe.cpp -pthread
// (fat binary: the same file runs on V620 gfx1030 and W7800 gfx1100)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#define CK(x)                                                     \
  do {                                                            \
    hipError_t e_ = (x);                                          \
    if (e_ != hipSuccess) {                                       \
      fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, \
              hipGetErrorString(e_));                             \
      exit(1);                                                    \
    }                                                             \
  } while (0)

using Clock = std::chrono::steady_clock;
constexpr size_t kBufBytes = 512u << 20;
constexpr int kSlices = 8;  // per-device slices, one per concurrent xfer
constexpr size_t kPerLaunchBytes = 8u << 20;

// Each repetition copies into its own window (dst + r * nvec) so posted stores
// are not coalesced in L2 between passes; src likewise varies so a pull does
// not hit the local L2 after the first pass. Needs reps * nvec * 16 <= slice.
__global__ void k_copy(uint4* __restrict__ dst, const uint4* __restrict__ src,
                       size_t nvec, int reps) {
  size_t gid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t gs = (size_t)gridDim.x * blockDim.x;
  for (int r = 0; r < reps; r++) {
    const uint4* s = src + (size_t)r * nvec;
    uint4* d = dst + (size_t)r * nvec;
    for (size_t i = gid; i < nvec; i += gs) d[i] = s[i];
  }
}

__device__ __forceinline__ int ld_sys(const int* p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

// One block per GPU. Initiator: push payload, flag peer, wait for flag back.
// Responder: wait, push payload back, flag. Mirrors the rdna_ar protocol
// (payload stores, system fence, release flag store, local poll).
__global__ void k_pingpong(uint4* peer_buf, const uint4* local_src,
                           int* peer_flag, int* my_flag, size_t nvec, int iters,
                           int initiator) {
  __shared__ int s_dummy;
  for (int i = 1; i <= iters; i++) {
    if (!initiator) {
      if (threadIdx.x == 0)
        while (ld_sys(my_flag) < i) {
        }
      __syncthreads();
    }
    for (size_t j = threadIdx.x; j < nvec; j += blockDim.x)
      peer_buf[j] = local_src[j];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0)
      __hip_atomic_store(peer_flag, i, __ATOMIC_RELEASE,
                         __HIP_MEMORY_SCOPE_SYSTEM);
    if (initiator) {
      if (threadIdx.x == 0)
        while (ld_sys(my_flag) < i) {
        }
      __syncthreads();
    }
  }
  if (threadIdx.x == 0 && iters < 0) s_dummy = 1;
}

constexpr int kMaxG = 8;
struct A2aArgs {
  uint4* peer_dst[kMaxG];  // peer j's landing zone (our slot is rank * slot)
  int* peer_flag[kMaxG];   // peer j's flag array
  int rank, world;
};

// One-shot all-to-all step as in rdna_ar: push our payload to every peer's
// slot[rank], system fence, release-store a flag into every peer, then wait
// until every peer's flag for this iteration is in our own flag array.
__global__ void k_a2a(A2aArgs a, const uint4* local_src, int* my_flag,
                      size_t nvec, size_t slot_vec, int iters) {
  for (int it = 1; it <= iters; it++) {
    for (int k = 1; k < a.world; k++) {
      int j = (a.rank + k) % a.world;
      uint4* d = a.peer_dst[j] + (size_t)a.rank * slot_vec;
      for (size_t i = threadIdx.x; i < nvec; i += blockDim.x)
        d[i] = local_src[i];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
      for (int k = 1; k < a.world; k++) {
        int j = (a.rank + k) % a.world;
        __hip_atomic_store(a.peer_flag[j] + a.rank, it, __ATOMIC_RELEASE,
                           __HIP_MEMORY_SCOPE_SYSTEM);
      }
      for (int k = 1; k < a.world; k++) {
        int j = (a.rank + k) % a.world;
        while (ld_sys(my_flag + j) < it) {
        }
      }
    }
    __syncthreads();
  }
}

struct Dev {
  uint4* src = nullptr;   // pattern source
  uint4* usrc = nullptr;  // uncached pattern source (pull reads this)
  uint4* dst = nullptr;   // uncached landing zone (like the rdna_ar stage)
  int* flag = nullptr;
};

static std::vector<Dev> g_dev;
static int g_ndev = 0;

static void enable_all_peers() {
  for (int a = 0; a < g_ndev; a++) {
    CK(hipSetDevice(a));
    for (int b = 0; b < g_ndev; b++) {
      if (a == b) continue;
      int can = 0;
      CK(hipDeviceCanAccessPeer(&can, a, b));
      if (!can) {
        fprintf(stderr, "no peer access %d->%d\n", a, b);
        exit(2);
      }
      hipError_t e = hipDeviceEnablePeerAccess(b, 0);
      if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) CK(e);
    }
  }
}

__global__ void k_fill(uint4* p, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t gs = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += gs)
    p[i] = make_uint4((unsigned)i, 0x11111111u, 0x22222222u, 0x33333333u);
}

static void alloc_all() {
  g_dev.resize(g_ndev);
  for (int d = 0; d < g_ndev; d++) {
    CK(hipSetDevice(d));
    CK(hipMalloc(&g_dev[d].src, kBufBytes));
    CK(hipExtMallocWithFlags((void**)&g_dev[d].usrc, kBufBytes,
                             hipDeviceMallocUncached));
    CK(hipExtMallocWithFlags((void**)&g_dev[d].dst, kBufBytes,
                             hipDeviceMallocUncached));
    CK(hipExtMallocWithFlags((void**)&g_dev[d].flag, 4096,
                             hipDeviceMallocUncached));
    k_fill<<<256, 256>>>(g_dev[d].src, kBufBytes / 16);
    k_fill<<<256, 256>>>(g_dev[d].usrc, kBufBytes / 16);
    CK(hipMemset(g_dev[d].dst, 0, kBufBytes));
    CK(hipMemset(g_dev[d].flag, 0, 4096));
    CK(hipDeviceSynchronize());
  }
}

static int cmd_info() {
  for (int d = 0; d < g_ndev; d++) {
    hipDeviceProp_t p;
    CK(hipGetDeviceProperties(&p, d));
    char bus[32];
    CK(hipDeviceGetPCIBusId(bus, sizeof bus, d));
    printf("DEV %d %s %s bus=%s\n", d, p.name, p.gcnArchName, bus);
  }
  for (int a = 0; a < g_ndev; a++)
    for (int b = 0; b < g_ndev; b++) {
      if (a == b) continue;
      int acc = 0, rank = 0, atom = 0;
      CK(hipDeviceGetP2PAttribute(&acc, hipDevP2PAttrAccessSupported, a, b));
      CK(hipDeviceGetP2PAttribute(&rank, hipDevP2PAttrPerformanceRank, a, b));
      CK(hipDeviceGetP2PAttribute(&atom, hipDevP2PAttrNativeAtomicSupported, a,
                                  b));
      printf("P2P %d->%d access=%d perf_rank=%d native_atomics=%d\n", a, b, acc,
             rank, atom);
    }
  return 0;
}

static int cmd_lat(int argc, char** argv) {
  int iters = atoi(argv[0]);
  size_t bytes = strtoull(argv[1], nullptr, 10);
  size_t nvec = bytes / 16;
  for (int i = 2; i < argc; i++) {
    int a, b;
    if (sscanf(argv[i], "%d:%d", &a, &b) != 2) return 1;
    double best = 1e30;
    for (int rep = 0; rep < 3; rep++) {
      CK(hipSetDevice(a));
      CK(hipMemset(g_dev[a].flag, 0, 4096));
      CK(hipDeviceSynchronize());
      CK(hipSetDevice(b));
      CK(hipMemset(g_dev[b].flag, 0, 4096));
      CK(hipDeviceSynchronize());
      hipStream_t sa, sb;
      hipEvent_t e0, e1;
      CK(hipSetDevice(b));
      CK(hipStreamCreate(&sb));
      CK(hipSetDevice(a));
      CK(hipStreamCreate(&sa));
      CK(hipEventCreate(&e0));
      CK(hipEventCreate(&e1));
      CK(hipSetDevice(b));
      k_pingpong<<<1, 256, 0, sb>>>(g_dev[a].dst, g_dev[b].src, g_dev[a].flag,
                                    g_dev[b].flag, nvec, iters, 0);
      CK(hipSetDevice(a));
      CK(hipEventRecord(e0, sa));
      k_pingpong<<<1, 256, 0, sa>>>(g_dev[b].dst, g_dev[a].src, g_dev[b].flag,
                                    g_dev[a].flag, nvec, iters, 1);
      CK(hipEventRecord(e1, sa));
      CK(hipStreamSynchronize(sa));
      CK(hipSetDevice(b));
      CK(hipStreamSynchronize(sb));
      float ms;
      CK(hipEventElapsedTime(&ms, e0, e1));
      double one_way_us = ms * 1000.0 / iters / 2.0;
      best = std::min(best, one_way_us);
      CK(hipSetDevice(a));
      CK(hipStreamDestroy(sa));
      CK(hipSetDevice(b));
      CK(hipStreamDestroy(sb));
    }
    printf("LAT %d<->%d payload=%zu one_way_us=%.2f\n", a, b, bytes, best);
    fflush(stdout);
  }
  return 0;
}

// a2a <iters> <payload_bytes> <g0,g1,..> [<g..> ...]: every group runs one-shot
// all-to-all steps back to back, all groups at once. Prints the per-step time
// (max over ranks) per group.
static int cmd_a2a(int argc, char** argv) {
  int iters = atoi(argv[0]);
  size_t bytes = strtoull(argv[1], nullptr, 10);
  size_t nvec = bytes / 16, slot_vec = std::max<size_t>(nvec, 1);
  std::vector<std::vector<int>> groups;
  for (int i = 2; i < argc; i++) {
    std::vector<int> g;
    for (char* t = strtok(argv[i], ","); t; t = strtok(nullptr, ","))
      g.push_back(atoi(t));
    if (g.size() < 2 || g.size() > kMaxG) return 1;
    groups.push_back(g);
  }
  for (int d = 0; d < g_ndev; d++) {
    CK(hipSetDevice(d));
    CK(hipMemset(g_dev[d].flag, 0, 4096));
    CK(hipDeviceSynchronize());
  }
  struct Run {
    int dev;
    size_t gi;
    hipStream_t st;
    hipEvent_t e0, e1;
  };
  std::vector<Run> runs;
  for (size_t gi = 0; gi < groups.size(); gi++)
    for (size_t r = 0; r < groups[gi].size(); r++) {
      Run u;
      u.dev = groups[gi][r];
      u.gi = gi;
      CK(hipSetDevice(u.dev));
      CK(hipStreamCreate(&u.st));
      CK(hipEventCreate(&u.e0));
      CK(hipEventCreate(&u.e1));
      runs.push_back(u);
    }
  size_t ri = 0;
  for (size_t gi = 0; gi < groups.size(); gi++)
    for (size_t r = 0; r < groups[gi].size(); r++, ri++) {
      A2aArgs a{};
      a.rank = (int)r;
      a.world = (int)groups[gi].size();
      for (size_t j = 0; j < groups[gi].size(); j++) {
        a.peer_dst[j] = g_dev[groups[gi][j]].dst;
        a.peer_flag[j] = g_dev[groups[gi][j]].flag;
      }
      Run& u = runs[ri];
      CK(hipSetDevice(u.dev));
      CK(hipEventRecord(u.e0, u.st));
      k_a2a<<<1, 256, 0, u.st>>>(a, g_dev[u.dev].src, g_dev[u.dev].flag, nvec,
                                 slot_vec, iters);
      CK(hipEventRecord(u.e1, u.st));
    }
  std::vector<double> worst(groups.size(), 0.0);
  for (auto& u : runs) {
    CK(hipSetDevice(u.dev));
    CK(hipStreamSynchronize(u.st));
    float ms;
    CK(hipEventElapsedTime(&ms, u.e0, u.e1));
    worst[u.gi] = std::max(worst[u.gi], ms * 1000.0 / iters);
    CK(hipStreamDestroy(u.st));
  }
  for (size_t gi = 0; gi < groups.size(); gi++) {
    printf("A2A payload=%zu group=", bytes);
    for (size_t j = 0; j < groups[gi].size(); j++)
      printf("%s%d", j ? "," : "", groups[gi][j]);
    printf(" step_us=%.2f\n", worst[gi]);
  }
  return 0;
}

// latc <iters> <payload_bytes> <a:b> [<a:b> ...]: the ping-pongs of lat, but
// all pairs (disjoint GPUs) at once. Prints one-way us per pair.
static int cmd_latc(int argc, char** argv) {
  int iters = atoi(argv[0]);
  size_t nvec = strtoull(argv[1], nullptr, 10) / 16;
  struct P {
    int a, b;
    hipStream_t sa, sb;
    hipEvent_t e0, e1;
  };
  std::vector<P> ps;
  for (int i = 2; i < argc; i++) {
    P p;
    if (sscanf(argv[i], "%d:%d", &p.a, &p.b) != 2) return 1;
    ps.push_back(p);
  }
  for (int d = 0; d < g_ndev; d++) {
    CK(hipSetDevice(d));
    CK(hipMemset(g_dev[d].flag, 0, 4096));
    CK(hipDeviceSynchronize());
  }
  for (auto& p : ps) {
    CK(hipSetDevice(p.b));
    CK(hipStreamCreate(&p.sb));
    CK(hipSetDevice(p.a));
    CK(hipStreamCreate(&p.sa));
    CK(hipEventCreate(&p.e0));
    CK(hipEventCreate(&p.e1));
  }
  for (auto& p : ps) {
    CK(hipSetDevice(p.b));
    k_pingpong<<<1, 256, 0, p.sb>>>(g_dev[p.a].dst, g_dev[p.b].src,
                                    g_dev[p.a].flag, g_dev[p.b].flag, nvec,
                                    iters, 0);
  }
  for (auto& p : ps) {
    CK(hipSetDevice(p.a));
    CK(hipEventRecord(p.e0, p.sa));
    k_pingpong<<<1, 256, 0, p.sa>>>(g_dev[p.b].dst, g_dev[p.a].src,
                                    g_dev[p.b].flag, g_dev[p.a].flag, nvec,
                                    iters, 1);
    CK(hipEventRecord(p.e1, p.sa));
  }
  for (auto& p : ps) {
    CK(hipSetDevice(p.a));
    CK(hipStreamSynchronize(p.sa));
    CK(hipSetDevice(p.b));
    CK(hipStreamSynchronize(p.sb));
    float ms;
    CK(hipEventElapsedTime(&ms, p.e0, p.e1));
    printf("LATC %d<->%d payload=%zu one_way_us=%.2f\n", p.a, p.b, nvec * 16,
           ms * 1000.0 / iters / 2.0);
  }
  return 0;
}

struct Xfer {
  std::string label;
  char mode = 'p';
  int src = 0, dst = 0;
  bool bg = false;
  uint4* sbuf = nullptr;
  uint4* dbuf = nullptr;
  double bytes_done = 0;
  double secs = 0;
  size_t soff_vec = 0;
};

static int cmd_bw(int argc, char** argv) {
  double dur = atof(argv[0]);
  size_t bytes = strtoull(argv[1], nullptr, 10);
  int blocks = atoi(argv[2]);
  size_t nvec = bytes / 16;
  int reps = (int)std::max<size_t>(1, kPerLaunchBytes / bytes);
  std::vector<Xfer> xs;
  for (int i = 3; i < argc; i++) {
    Xfer x;
    x.label = argv[i];
    std::string s = argv[i];
    if (s.rfind("bg:", 0) == 0) {
      x.bg = true;
      s = s.substr(3);
    }
    x.mode = s[0];
    if (sscanf(s.c_str() + 2, "%d>%d", &x.src, &x.dst) != 2) return 1;
    xs.push_back(x);
  }
  // Private src/dst region per transfer so concurrent flows do not overlap.
  std::vector<int> used_s(g_ndev, 0), used_d(g_ndev, 0);
  const size_t slice = kBufBytes / kSlices;
  if (bytes * reps > slice) {
    fprintf(stderr, "bytes * reps > slice (%zu)\n", slice);
    return 1;
  }
  for (auto& x : xs) {
    int ns = used_s[x.src]++, nd = used_d[x.dst]++;
    if (ns >= kSlices || nd >= kSlices) {
      fprintf(stderr, "too many xfers per device\n");
      return 1;
    }
    x.soff_vec = ns * slice / 16;
    x.sbuf =
        (uint4*)((char*)(x.mode == 'l' ? g_dev[x.src].usrc : g_dev[x.src].src) +
                 ns * slice);
    x.dbuf = (uint4*)((char*)g_dev[x.dst].dst + nd * slice);
  }
  for (int d = 0; d < g_ndev; d++) {
    CK(hipSetDevice(d));
    CK(hipMemset(g_dev[d].dst, 0, kBufBytes));
    CK(hipDeviceSynchronize());
  }
  std::atomic<int> ready{0};
  std::atomic<bool> go{false};
  std::atomic<int64_t> stop_ns{0};
  std::vector<std::thread> th;
  for (size_t xi = 0; xi < xs.size(); xi++) {
    th.emplace_back([&, xi]() {
      Xfer& x = xs[xi];
      int dev = x.mode == 'l' ? x.dst : x.src;
      CK(hipSetDevice(dev));
      hipStream_t st;
      CK(hipStreamCreate(&st));
      const int batch = x.mode == 's' ? 16 : 4;
      auto launch = [&]() {
        if (x.mode == 's')
          CK(hipMemcpyPeerAsync(x.dbuf, x.dst, x.sbuf, x.src, bytes, st));
        else
          k_copy<<<blocks, 256, 0, st>>>(x.dbuf, x.sbuf, nvec, reps);
      };
      const double per_launch =
          x.mode == 's' ? (double)bytes : (double)bytes * reps;
      for (int i = 0; i < 3; i++) launch();
      CK(hipStreamSynchronize(st));
      ready++;
      while (!go.load()) {
      }
      auto t0 = Clock::now();
      int64_t stop = stop_ns.load();
      double done = 0;
      while (Clock::now().time_since_epoch().count() < stop) {
        for (int i = 0; i < batch; i++) launch();
        CK(hipStreamSynchronize(st));
        done += batch * per_launch;
      }
      auto t1 = Clock::now();
      x.bytes_done = done;
      x.secs = std::chrono::duration<double>(t1 - t0).count();
      CK(hipStreamDestroy(st));
    });
  }
  while (ready.load() < (int)xs.size()) {
  }
  stop_ns = (Clock::now() + std::chrono::duration_cast<Clock::duration>(
                                std::chrono::duration<double>(dur)))
                .time_since_epoch()
                .count();
  go = true;
  for (auto& t : th) t.join();
  // Verify the landing zones (dst.x == i + reps - 1 for kernels, i for sdma).
  double fg = 0, bgs = 0;
  for (auto& x : xs) {
    CK(hipSetDevice(x.dst));
    const size_t windows = x.mode == 's' ? 1 : reps;
    std::vector<uint4> h(nvec * windows);
    CK(hipMemcpy(h.data(), x.dbuf, bytes * windows, hipMemcpyDeviceToHost));
    size_t bad = 0;
    for (size_t i = 0; i < nvec * windows; i++)
      if (h[i].x != (unsigned)(x.soff_vec + i) || h[i].y != 0x11111111u) bad++;
    double gbps = x.bytes_done / x.secs / 1e9;
    (x.bg ? bgs : fg) += x.bytes_done / x.secs / 1e9;
    printf("X %s size=%zu blocks=%d gbps=%.2f verify=%s\n", x.label.c_str(),
           bytes, blocks, gbps, bad ? "FAIL" : "ok");
  }
  printf("AGG size=%zu fg_gbps=%.2f bg_gbps=%.2f n=%zu\n", bytes, fg, bgs,
         xs.size());
  return 0;
}

// ===========================================================================
// pciescope extensions (2026-11-07). Everything above this block is the legacy
// probe: its modes, kernels and output lines are unchanged.
//
// New output lines, one record per line, key=value tokens in this order:
//   VERSION <string> modes=<csv>
//   READY devices=<n> tick_mhz=<f,f,..> version=<string>    (batch, once)
//   END rc=<n> ms=<t>              (batch, last line of every command)
//   HBW <label> size= blocks= gbps= verify=<ok|FAIL> secs=
//   HAGG size= fg_gbps= bg_gbps= n=
//   HLAT dev= dir= mem= size= n= min_us= p50_us= p99_us= max_us= mean_us=
//   HFLAG dev= n= rtt_min_us= rtt_p50_us= rtt_p99_us= rtt_max_us= rtt_mean_us=
//   CURVE mode= src= dst= size= blocks= n= min_us= p50_us= p99_us= max_us=
//         mean_us= [ev_us=]
//   ERR usage <command> | ERR size <command> ... | ERR timeout <command> ...
// Percentiles are nearest-rank on the sorted samples. Every GPU spin loop has
// a deadline (P2P_KERNEL_TIMEOUT_S, default 10 s, counted in wall_clock64
// ticks) and reports a timeout through a status word; the host spin loops use
// the same limit. Device ids are logical (index into HIP_VISIBLE_DEVICES).
// ===========================================================================

constexpr const char* kVersion = "pciescope-probe/1";
constexpr const char* kModes =
    "version,buses,info,lat,latc,bw,a2a,hbw,hlat,hflag,curve,batch";
constexpr double kDefaultTimeoutS = 10.0;
constexpr double kCellBudgetS = 0.25;  // wall time one curve/hlat cell may use
constexpr int kMaxIters = 1 << 22;
constexpr int kMaxBlocks = 256;  // curve: every block must be resident at once
constexpr int kMaxHostBlocks = 4096;
constexpr size_t kMaxCurveBytes = kBufBytes / kSlices;  // one 64 MiB slice
constexpr size_t kMaxHostBytes = 1u << 30;
constexpr int kRcUsage = 1;
constexpr int kRcTimeout = 3;

// Runs a cleanup action at scope exit: batch mode runs many commands in one
// process, so HIP buffers and streams must not leak.
template <class F>
struct Defer {
  F fn;
  ~Defer() { fn(); }
};
template <class F>
static Defer<F> defer(F fn) {
  return Defer<F>{std::move(fn)};
}

// ---- statistics -----------------------------------------------------------

struct Summary {
  size_t n = 0;
  double min = 0, p50 = 0, p99 = 0, max = 0, mean = 0;
};

// Nearest-rank percentile of ascending samples: the value at rank
// ceil(p * n / 100), computed in integers.
static double percentile(const std::vector<double>& sorted, int p) {
  size_t rank = (sorted.size() * (size_t)p + 99) / 100;
  rank = std::min(std::max<size_t>(rank, 1), sorted.size());
  return sorted[rank - 1];
}

// Sorts `v` in place and reduces it; an empty input gives all zeros.
static Summary summarize(std::vector<double>& v) {
  Summary s;
  s.n = v.size();
  if (v.empty()) return s;
  std::sort(v.begin(), v.end());
  double sum = 0;
  for (double x : v) sum += x;
  s.min = v.front();
  s.max = v.back();
  s.mean = sum / (double)v.size();
  s.p50 = percentile(v, 50);
  s.p99 = percentile(v, 99);
  return s;
}

// " <prefix>min_us=.. <prefix>p50_us=.." tail of HLAT, HFLAG and CURVE lines.
static void print_stats(const char* prefix, const Summary& s) {
  printf(" %smin_us=%.3f %sp50_us=%.3f %sp99_us=%.3f %smax_us=%.3f", prefix,
         s.min, prefix, s.p50, prefix, s.p99, prefix, s.max);
  printf(" %smean_us=%.3f", prefix, s.mean);
}

// Iterations that fit the per-cell time budget: at most `iters`, at least
// `lo` unless `iters` is smaller.
static int adaptive_n(double iter_us, int iters, int lo) {
  if (!(iter_us > 0)) return iters;
  const double fit = kCellBudgetS * 1e6 / iter_us;
  const int n = fit >= (double)iters ? iters : (int)fit;
  return std::min(iters, std::max(lo, n));
}

// ---- argument parsing -----------------------------------------------------

// Bad arguments: ERR usage on stdout (the parsed record), syntax on stderr.
static int usage_error(const char* cmd, const char* hint) {
  printf("ERR usage %s\n", cmd);
  fprintf(stderr, "p2p_probe %s: %s\n", cmd, hint);
  return kRcUsage;
}

static bool parse_int(const char* s, long long lo, long long hi,
                      long long* out) {
  if (!s || !*s) return false;
  char* end = nullptr;
  errno = 0;
  const long long v = strtoll(s, &end, 10);
  if (errno || *end || v < lo || v > hi) return false;
  *out = v;
  return true;
}

static bool parse_csv(const char* s, long long lo, long long hi,
                      std::vector<long long>* out) {
  out->clear();
  const std::string str = s ? s : "";
  size_t pos = 0;
  while (true) {
    const size_t comma = str.find(',', pos);
    long long v = 0;
    if (!parse_int(str.substr(pos, comma - pos).c_str(), lo, hi, &v))
      return false;
    out->push_back(v);
    if (comma == std::string::npos) return true;
    pos = comma + 1;
  }
}

// "<a>:<b>": two distinct logical device ids.
static bool parse_pair(const char* s, int* a, int* b) {
  const char* colon = strchr(s, ':');
  if (!colon) return false;
  long long x = 0, y = 0;
  if (!parse_int(std::string(s, colon).c_str(), 0, g_ndev - 1, &x) ||
      !parse_int(colon + 1, 0, g_ndev - 1, &y) || x == y)
    return false;
  *a = (int)x;
  *b = (int)y;
  return true;
}

// ---- time and host spin loops ---------------------------------------------

static double elapsed_us(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::micro>(b - a).count();
}

static Clock::duration seconds_dur(double s) {
  return std::chrono::duration_cast<Clock::duration>(
      std::chrono::duration<double>(s));
}

// Deadline of every GPU spin loop and host spin loop (P2P_KERNEL_TIMEOUT_S).
static double kernel_timeout_s() {
  static const double v = []() {
    const char* e = getenv("P2P_KERNEL_TIMEOUT_S");
    if (!e || !*e) return kDefaultTimeoutS;
    char* end = nullptr;
    const double x = strtod(e, &end);
    if (*end || !(x > 0.0) || x > 3600.0) {
      fprintf(stderr, "ignoring P2P_KERNEL_TIMEOUT_S=%s (want 0 < s <= 3600)\n",
              e);
      return kDefaultTimeoutS;
    }
    return x;
  }();
  return v;
}

// Words shared with the GPU live as plain ints in coherent pinned memory; the
// host reads and writes them as lock-free atomics of the same layout.
static_assert(std::atomic<int>::is_always_lock_free, "needs lock-free int");
static_assert(sizeof(std::atomic<int>) == sizeof(int), "unexpected layout");
static std::atomic<int>& host_atomic(int& word) {
  return *reinterpret_cast<std::atomic<int>*>(&word);
}

// Spins until `word >= want`. False when the timeout passes or `abort_word`
// (when given) turns non-zero. The clock is read every 256th poll.
static bool host_wait_ge(int& word, int want, int* abort_word = nullptr) {
  std::atomic<int>& w = host_atomic(word);
  const auto limit = Clock::now() + seconds_dur(kernel_timeout_s());
  for (unsigned n = 1; w.load(std::memory_order_acquire) < want; n++) {
    if ((n & 255) != 0) continue;
    if (Clock::now() > limit) return false;
    if (abort_word &&
        host_atomic(*abort_word).load(std::memory_order_relaxed) != 0)
      return false;
  }
  return true;
}

// ---- GPU spin-loop building blocks ----------------------------------------

constexpr int kWaitFlag = 1;  // status word: low 2 bits = what was waited for
constexpr int kWaitBarrier = 2;
// Per-launch control block of a GPU, one 128-byte line per word (ints).
constexpr int kCtlStatus = 0;     // 0 = ok, else (iteration << 2) | kWait*
constexpr int kCtlArrive = 32;    // k_pp2 arrival counter
constexpr int kCtlBarCount = 64;  // grid barrier: blocks arrived
constexpr int kCtlBarGen = 96;    // grid barrier: generation
constexpr int kCtlWords = 128;

// Keeps the compiler from moving memory accesses across a timestamp read or
// from reusing a load of an earlier iteration.
__device__ __forceinline__ void compiler_fence() {
  __asm__ volatile("" ::: "memory");
}

// Polls until *p >= want and returns the value seen; INT_MIN once `timeout`
// ticks of wall_clock64() passed (wanted values are positive, so the two cannot
// be confused). The clock is read when the wait starts and then every 64th
// poll, so the poll loop stays as tight as the loops of the legacy kernels.
__device__ __forceinline__ int wait_ge(const int* p, int want,
                                       long long timeout) {
  const long long limit = wall_clock64() + timeout;
  for (unsigned n = 1;; n++) {
    const int v = ld_sys(p);
    if (v >= want) return v;
    if ((n & 63) == 0 && wall_clock64() > limit) return INT_MIN;
  }
}

// Same for "*p != v" on a device-local word (grid barrier generation).
__device__ __forceinline__ bool wait_ne(const unsigned* p, unsigned v,
                                        long long timeout) {
  const long long limit = wall_clock64() + timeout;
  for (unsigned n = 1;
       __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) == v;
       n++)
    if ((n & 63) == 0 && wall_clock64() > limit) return false;
  return true;
}

__device__ __forceinline__ void report_timeout(int* status, int iter,
                                               int kind) {
  if (threadIdx.x == 0)
    __hip_atomic_store(status, (iter << 2) | kind, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_SYSTEM);
}

// Block-wide wait for flag >= want. Thread 0 polls with the deadline; the
// verdict reaches the whole block through *s_stop (a __shared__ int that is 0
// at entry and only ever goes 0 -> 1, so no thread can read a stale value after
// the barrier). Returns false in every thread of the block on timeout.
__device__ __forceinline__ bool block_wait_flag(const int* flag, int want,
                                                long long timeout,
                                                int* s_stop) {
  if (threadIdx.x == 0 && wait_ge(flag, want, timeout) == INT_MIN) *s_stop = 1;
  __syncthreads();
  return *s_stop == 0;
}

// Software grid barrier; every block of the grid must be resident and call it.
// The words live in the ctl block (zero before the launch). A block reads the
// generation BEFORE it arrives: the generation cannot change until this block
// has arrived, so the value read is the one this barrier ends. The last block
// to arrive resets the count, then publishes generation + 1 with a release
// store; the others spin on the generation (relaxed agent-scope loads) and take
// one acquire fence when it changed. Returns false in every thread of the
// block on timeout (*s_stop as in block_wait_flag).
__device__ bool grid_barrier(int* ctl, long long timeout, int* s_stop) {
  __syncthreads();  // the block's own work is complete and block-visible
  if (threadIdx.x == 0) {
    unsigned* count = reinterpret_cast<unsigned*>(ctl + kCtlBarCount);
    unsigned* gen = reinterpret_cast<unsigned*>(ctl + kCtlBarGen);
    const unsigned g =
        __hip_atomic_load(gen, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    const unsigned old = __hip_atomic_fetch_add(count, 1u, __ATOMIC_ACQ_REL,
                                                __HIP_MEMORY_SCOPE_AGENT);
    if (old + 1 == gridDim.x) {
      __hip_atomic_store(count, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_store(gen, g + 1, __ATOMIC_RELEASE,
                         __HIP_MEMORY_SCOPE_AGENT);
    } else if (wait_ne(gen, g, timeout)) {
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    } else {
      *s_stop = 1;
    }
  }
  __syncthreads();
  return *s_stop == 0;
}

// ---- wall_clock64() calibration -------------------------------------------

struct CalMem {  // coherent pinned memory shared by k_tick_cal and the host
  alignas(128) int started;
  alignas(128) int done;
  alignas(128) long long t_begin;
  alignas(128) long long t_end;
};

// One thread spins for `spin` ticks of wall_clock64() and brackets the spin
// with a start and an end marker in host memory; the host timestamps both
// markers. Each marker is "sample, plain store of the sample, release store of
// the marker", so both markers see the same latency and it cancels in the
// difference. The iteration cap only matters if the clock does not advance.
__global__ void k_tick_cal(CalMem* m, long long spin) {
  const long long t0 = wall_clock64();
  m->t_begin = t0;
  __hip_atomic_store(&m->started, 1, __ATOMIC_RELEASE,
                     __HIP_MEMORY_SCOPE_SYSTEM);
  long long t = t0;
  for (unsigned n = 0; t - t0 < spin && n < (1u << 28); n++) t = wall_clock64();
  m->t_end = t;
  __hip_atomic_store(&m->done, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// Ticks per microsecond of wall_clock64() on `dev`: median of three ~20 ms
// spins, each timed by the host between the two markers. The spin length is
// seeded from the nominal rate (hipDeviceAttributeWallClockRate, kHz).
static double calibrate_dev(int dev) {
  static CalMem* cal = nullptr;
  CK(hipSetDevice(dev));
  if (!cal)
    CK(hipHostMalloc((void**)&cal, sizeof(CalMem), hipHostMallocCoherent));
  CalMem* dcal = nullptr;
  CK(hipHostGetDevicePointer((void**)&dcal, cal, 0));
  int khz = 0;
  if (hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, dev) !=
          hipSuccess ||
      khz <= 0)
    khz = 100000;
  const long long spin = (long long)khz * 20;  // 20 ms
  double mhz[3];
  for (double& m : mhz) {
    host_atomic(cal->started).store(0);
    host_atomic(cal->done).store(0);
    k_tick_cal<<<1, 1>>>(dcal, spin);
    CK(hipGetLastError());
    if (!host_wait_ge(cal->started, 1)) {
      fprintf(stderr, "tick calibration: GPU %d never started\n", dev);
      exit(1);
    }
    const auto h0 = Clock::now();
    if (!host_wait_ge(cal->done, 1)) {
      fprintf(stderr, "tick calibration: GPU %d never finished\n", dev);
      exit(1);
    }
    const auto h1 = Clock::now();
    CK(hipDeviceSynchronize());
    m = (double)(cal->t_end - cal->t_begin) / elapsed_us(h0, h1);
  }
  std::sort(mhz, mhz + 3);
  // A host thread paused between the two markers skews a sample by orders of
  // magnitude and would stretch or cut every deadline: a median far from the
  // nominal rate is not trusted.
  const double nominal = khz / 1000.0;
  if (!(mhz[1] > 0.5 * nominal && mhz[1] < 2.0 * nominal)) {
    fprintf(stderr,
            "tick calibration: GPU %d measured %.3f MHz, nominal %.3f, "
            "using nominal\n",
            dev, mhz[1], nominal);
    return nominal;
  }
  return mhz[1];
}

static std::vector<double> g_tick_mhz;  // measured ticks per us, 0 = not yet

static double tick_mhz(int dev) {
  if ((int)g_tick_mhz.size() < g_ndev) g_tick_mhz.resize(g_ndev, 0.0);
  if (g_tick_mhz[dev] <= 0) g_tick_mhz[dev] = calibrate_dev(dev);
  return g_tick_mhz[dev];
}

// The deadline of the GPU spin loops of `dev`, in wall_clock64 ticks.
static long long deadline_ticks(int dev) {
  return (long long)(kernel_timeout_s() * 1e6 * tick_mhz(dev));
}

// ---- version, buses -------------------------------------------------------

static int cmd_version(int, char**) {
  printf("VERSION %s modes=%s\n", kVersion, kModes);
  return 0;
}

// DEV lines for what HIP exposes; no peers, no allocation. Nothing visible
// (HIP_VISIBLE_DEVICES empty) prints nothing and succeeds.
static int cmd_buses(int, char**) {
  int n = 0;
  const hipError_t e = hipGetDeviceCount(&n);
  if (e == hipErrorNoDevice)
    n = 0;
  else
    CK(e);
  for (int d = 0; d < n; d++) {
    hipDeviceProp_t p;
    CK(hipGetDeviceProperties(&p, d));
    char bus[32];
    CK(hipDeviceGetPCIBusId(bus, sizeof bus, d));
    const std::string arch = p.gcnArchName;
    printf("DEV %d %s %s bus=%s\n", d, p.name,
           arch.substr(0, arch.find(':')).c_str(), bus);
  }
  return 0;
}

// ---- hbw: host <-> GPU bandwidth ------------------------------------------

enum class HDir { H2D, D2H };
enum class HMem { Pinned, Pageable, Kernel };
struct HMode {
  const char* name;
  HDir dir;
  HMem mem;
};
static const HMode kHModes[] = {
    {"h2d", HDir::H2D, HMem::Pinned},
    {"d2h", HDir::D2H, HMem::Pinned},
    {"h2d_pg", HDir::H2D, HMem::Pageable},
    {"d2h_pg", HDir::D2H, HMem::Pageable},
    {"h2d_k", HDir::H2D, HMem::Kernel},
    {"d2h_k", HDir::D2H, HMem::Kernel},
};

struct HXfer {
  std::string label;  // as given, including a bg: prefix
  const HMode* mode = nullptr;
  int dev = 0;
  bool bg = false;
  size_t nvec = 0;            // 16-byte vectors per buffer
  uint4* host = nullptr;      // pinned (hipHostMalloc) or pageable memory
  uint4* host_dev = nullptr;  // device view of the pinned buffer (kernel modes)
  uint4* dbuf = nullptr;      // GPU buffer
  double bytes_done = 0;
  double secs = 0;
};

// "[bg:]<mode>:<dev>"
static bool parse_hxfer(const char* arg, HXfer* x) {
  std::string s = arg;
  x->label = s;
  if (s.rfind("bg:", 0) == 0) {
    x->bg = true;
    s.erase(0, 3);
  }
  const size_t colon = s.rfind(':');
  if (colon == std::string::npos) return false;
  const std::string name = s.substr(0, colon);
  for (const HMode& m : kHModes)
    if (name == m.name) x->mode = &m;
  long long dev = 0;
  if (!x->mode || !parse_int(s.c_str() + colon + 1, 0, g_ndev - 1, &dev))
    return false;
  x->dev = (int)dev;
  return true;
}

// Same pattern as k_fill: vector i = (i, 0x11111111, 0x22222222, 0x33333333).
static void fill_pattern(uint4* p, size_t nvec) {
  for (size_t i = 0; i < nvec; i++)
    p[i] = make_uint4((unsigned)i, 0x11111111u, 0x22222222u, 0x33333333u);
}

static bool pattern_ok(const uint4* p, size_t nvec) {
  for (size_t i = 0; i < nvec; i++)
    if (p[i].x != (unsigned)i || p[i].y != 0x11111111u ||
        p[i].z != 0x22222222u || p[i].w != 0x33333333u)
      return false;
  return true;
}

static void hxfer_alloc(HXfer& x) {
  const size_t span = x.nvec * sizeof(uint4);
  CK(hipSetDevice(x.dev));
  if (x.mode->mem == HMem::Pageable) {
    x.host = (uint4*)aligned_alloc(4096, (span + 4095) & ~(size_t)4095);
    if (!x.host) {
      fprintf(stderr, "out of host memory (%zu bytes)\n", span);
      exit(1);
    }
  } else {
    CK(hipHostMalloc((void**)&x.host, span, hipHostMallocDefault));
    if (x.mode->mem == HMem::Kernel)
      CK(hipHostGetDevicePointer((void**)&x.host_dev, x.host, 0));
  }
  CK(hipMalloc((void**)&x.dbuf, span));
  if (x.mode->dir == HDir::H2D) {  // host holds the pattern, GPU starts empty
    fill_pattern(x.host, x.nvec);
    CK(hipMemset(x.dbuf, 0, span));
  } else {  // GPU holds the pattern, host starts empty (and pre-touched)
    memset(x.host, 0, span);
    k_fill<<<256, 256>>>(x.dbuf, x.nvec);
    CK(hipGetLastError());
  }
  CK(hipDeviceSynchronize());
}

static void hxfer_free(HXfer& x) {
  (void)hipSetDevice(x.dev);
  if (x.dbuf) (void)hipFree(x.dbuf);
  if (x.mode->mem == HMem::Pageable)
    free(x.host);
  else if (x.host)
    (void)hipHostFree(x.host);
}

// What landed must be the pattern: h2d reads the GPU buffer back into
// pageable scratch, d2h checks the host buffer in place.
static bool hxfer_verify(const HXfer& x) {
  if (x.mode->dir == HDir::D2H) return pattern_ok(x.host, x.nvec);
  std::vector<uint4> back(x.nvec);
  CK(hipSetDevice(x.dev));
  CK(hipMemcpy(back.data(), x.dbuf, x.nvec * sizeof(uint4),
               hipMemcpyDeviceToHost));
  return pattern_ok(back.data(), x.nvec);
}

// hbw <seconds> <bytes> <blocks> <xfer>...: the thread-per-xfer structure of
// cmd_bw (ready barrier, shared stop time, per-xfer bytes and seconds), with
// host memory instead of peer GPUs. Pinned buffers use hipHostMalloc defaults,
// pageable ones are page-aligned malloc memory written once (pre-touched); each
// xfer owns a GPU buffer and a stream. Kernel modes run k_copy over the
// device view of the pinned buffer with `reps` windows per launch as cmd_bw
// does, so their buffers hold bytes * reps.
static int cmd_hbw(int argc, char** argv) {
  static const char* kSyntax =
      "hbw <seconds> <bytes> <blocks> <[bg:]h2d|d2h|h2d_pg|d2h_pg|h2d_k|d2h_k"
      ":dev>...  (bytes: multiple of 16)";
  if (argc < 4) return usage_error("hbw", kSyntax);
  char* end = nullptr;
  const double dur = strtod(argv[0], &end);
  long long bytes = 0, blocks = 0;
  if (end == argv[0] || *end || !(dur > 0 && dur <= 3600) ||
      !parse_int(argv[1], 16, kMaxHostBytes, &bytes) || bytes % 16 ||
      !parse_int(argv[2], 1, kMaxHostBlocks, &blocks))
    return usage_error("hbw", kSyntax);
  std::vector<HXfer> xs;
  std::vector<int> per_dev(g_ndev, 0);
  for (int i = 3; i < argc; i++) {
    HXfer x;
    if (!parse_hxfer(argv[i], &x) || ++per_dev[x.dev] > kSlices)
      return usage_error("hbw", kSyntax);
    xs.push_back(x);
  }
  const size_t nvec = (size_t)bytes / 16;
  const int reps = (int)std::max<size_t>(1, kPerLaunchBytes / (size_t)bytes);
  auto release = defer([&] {
    for (auto& x : xs) hxfer_free(x);
  });
  for (auto& x : xs) {
    x.nvec = nvec * (x.mode->mem == HMem::Kernel ? (size_t)reps : 1);
    hxfer_alloc(x);
  }
  std::atomic<int> ready{0};
  std::atomic<int> state{0};  // 0 wait, 1 go, 2 abort
  std::atomic<int64_t> stop_ns{0};
  std::vector<std::thread> th;
  for (size_t xi = 0; xi < xs.size(); xi++) {
    th.emplace_back([&, xi]() {
      HXfer& x = xs[xi];
      CK(hipSetDevice(x.dev));
      hipStream_t st;
      CK(hipStreamCreate(&st));
      const bool kern = x.mode->mem == HMem::Kernel;
      const bool h2d = x.mode->dir == HDir::H2D;
      const int batch = kern ? 4 : 16;
      auto launch = [&]() {
        if (kern && h2d)
          k_copy<<<(int)blocks, 256, 0, st>>>(x.dbuf, x.host_dev, nvec, reps);
        else if (kern)
          k_copy<<<(int)blocks, 256, 0, st>>>(x.host_dev, x.dbuf, nvec, reps);
        else if (h2d)
          CK(hipMemcpyAsync(x.dbuf, x.host, (size_t)bytes,
                            hipMemcpyHostToDevice, st));
        else
          CK(hipMemcpyAsync(x.host, x.dbuf, (size_t)bytes,
                            hipMemcpyDeviceToHost, st));
      };
      const double per_launch = (double)bytes * (kern ? reps : 1);
      for (int i = 0; i < 3; i++) launch();
      CK(hipStreamSynchronize(st));
      ready++;
      while (state.load() == 0) {
      }
      if (state.load() == 1) {
        const auto t0 = Clock::now();
        const int64_t stop = stop_ns.load();
        double done = 0;
        while (Clock::now().time_since_epoch().count() < stop) {
          for (int i = 0; i < batch; i++) launch();
          CK(hipStreamSynchronize(st));
          done += batch * per_launch;
        }
        x.bytes_done = done;
        x.secs = elapsed_us(t0, Clock::now()) / 1e6;
      }
      CK(hipStreamDestroy(st));
    });
  }
  const auto ready_limit = Clock::now() + seconds_dur(kernel_timeout_s());
  bool stalled = false;
  while (ready.load() < (int)xs.size() && !stalled)
    stalled = Clock::now() > ready_limit;
  if (!stalled)
    stop_ns = (Clock::now() + seconds_dur(dur)).time_since_epoch().count();
  state = stalled ? 2 : 1;
  for (auto& t : th) t.join();
  if (stalled) {
    printf("ERR timeout hbw ready size=%lld xfers=%zu\n", bytes, xs.size());
    return kRcTimeout;
  }
  double fg = 0, bgs = 0;
  for (const auto& x : xs) {
    const bool ok = hxfer_verify(x);
    const double gbps = x.secs > 0 ? x.bytes_done / x.secs / 1e9 : 0.0;
    (x.bg ? bgs : fg) += gbps;
    printf("HBW %s size=%lld blocks=%lld gbps=%.2f verify=%s secs=%.3f\n",
           x.label.c_str(), bytes, blocks, gbps, ok ? "ok" : "FAIL", x.secs);
  }
  printf("HAGG size=%lld fg_gbps=%.2f bg_gbps=%.2f n=%zu\n", bytes, fg, bgs,
         xs.size());
  fflush(stdout);
  return 0;
}

// ---- hlat: host-observed copy latency -------------------------------------

constexpr int kHlatWarm = 5;

struct HCombo {
  const char* dir;
  const char* mem;
  hipMemcpyKind kind;
  bool pinned;
};
static const HCombo kHCombos[] = {
    {"h2d", "pinned", hipMemcpyHostToDevice, true},
    {"h2d", "pageable", hipMemcpyHostToDevice, false},
    {"d2h", "pinned", hipMemcpyDeviceToHost, true},
    {"d2h", "pageable", hipMemcpyDeviceToHost, false},
};

// One hipMemcpyAsync plus hipStreamSynchronize as the host sees it (us).
static double timed_copy(void* dst, const void* src, size_t bytes,
                         hipMemcpyKind kind, hipStream_t st) {
  const auto t0 = Clock::now();
  CK(hipMemcpyAsync(dst, src, bytes, kind, st));
  CK(hipStreamSynchronize(st));
  return elapsed_us(t0, Clock::now());
}

// 5 warm-up copies (their median sizes the cell: at most `iters`, 0.25 s
// budget, at least 5), then one HLAT line over the measured copies.
static void hlat_cell(int dev, size_t size, int iters, const HCombo& c,
                      void* host, void* dbuf, hipStream_t st) {
  const bool up = c.kind == hipMemcpyHostToDevice;
  void* dst = up ? dbuf : host;
  const void* src = up ? host : dbuf;
  double warm[kHlatWarm];
  for (double& w : warm) w = timed_copy(dst, src, size, c.kind, st);
  std::sort(warm, warm + kHlatWarm);
  const int n = adaptive_n(warm[kHlatWarm / 2], iters, 5);
  std::vector<double> us(n);
  for (double& u : us) u = timed_copy(dst, src, size, c.kind, st);
  const Summary s = summarize(us);
  printf("HLAT dev=%d dir=%s mem=%s size=%zu n=%d", dev, c.dir, c.mem, size, n);
  print_stats("", s);
  printf("\n");
  fflush(stdout);
}

// Pinned, pageable and GPU buffer of one (dev, size), the four (dir, mem)
// combinations on one stream.
static void hlat_size(int dev, size_t size, int iters) {
  CK(hipSetDevice(dev));
  void* pinned = nullptr;
  void* pageable = nullptr;
  void* dbuf = nullptr;
  hipStream_t st = nullptr;
  auto release = defer([&] {
    (void)hipSetDevice(dev);
    if (st) (void)hipStreamDestroy(st);
    if (dbuf) (void)hipFree(dbuf);
    if (pinned) (void)hipHostFree(pinned);
    free(pageable);
  });
  CK(hipStreamCreate(&st));
  CK(hipHostMalloc(&pinned, size, hipHostMallocDefault));
  pageable = aligned_alloc(4096, (size + 4095) & ~(size_t)4095);
  if (!pageable) {
    fprintf(stderr, "out of host memory (%zu bytes)\n", size);
    exit(1);
  }
  CK(hipMalloc(&dbuf, size));
  memset(pinned, 0xA5, size);
  memset(pageable, 0xA5, size);  // pre-touch
  CK(hipMemset(dbuf, 0, size));
  CK(hipDeviceSynchronize());
  for (const HCombo& c : kHCombos)
    hlat_cell(dev, size, iters, c, c.pinned ? pinned : pageable, dbuf, st);
}

static int cmd_hlat(int argc, char** argv) {
  static const char* kSyntax = "hlat <iters> <devs_csv> <size>...";
  long long iters = 0;
  std::vector<long long> devs, sizes;
  if (argc < 3 || !parse_int(argv[0], 1, kMaxIters, &iters) ||
      !parse_csv(argv[1], 0, g_ndev - 1, &devs))
    return usage_error("hlat", kSyntax);
  for (int i = 2; i < argc; i++) {
    long long s = 0;
    if (!parse_int(argv[i], 1, kMaxHostBytes, &s))
      return usage_error("hlat", kSyntax);
    sizes.push_back(s);
  }
  for (long long dev : devs)
    for (long long size : sizes) hlat_size((int)dev, (size_t)size, (int)iters);
  return 0;
}

// ---- hflag: GPU <-> CPU doorbell round trip -------------------------------

constexpr int kHflagWarm = 20;

struct HFlagMem {        // coherent pinned memory shared by the CPU and k_hflag
  alignas(128) int req;  // CPU -> GPU doorbell: round trip number
  alignas(128) int ack;  // GPU -> CPU answer: the same number
  alignas(128) int status;  // GPU deadline report (report_timeout)
};

// Protocol: round trip i is "CPU stores req = i, GPU answers ack = i". One
// GPU thread polls req (system scope, deadline-guarded) and answers with a
// system-scope release store. The CPU times each round trip with steady_clock.
// A request of INT_MAX is the host's abort: the kernel returns unanswered.
__global__ void k_hflag(HFlagMem* m, int iters, long long timeout) {
  for (int i = 1; i <= iters; i++) {
    const int seen = wait_ge(&m->req, i, timeout);
    if (seen == INT_MIN) {
      report_timeout(&m->status, i, kWaitFlag);
      return;
    }
    if (seen == INT_MAX) return;
    __hip_atomic_store(&m->ack, i, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

static int hflag_one(HFlagMem* m, int dev, int iters) {
  const int total = iters + kHflagWarm;
  const long long limit_ticks = deadline_ticks(dev);
  CK(hipSetDevice(dev));
  HFlagMem* dm = nullptr;
  CK(hipHostGetDevicePointer((void**)&dm, m, 0));
  host_atomic(m->req).store(0);
  host_atomic(m->ack).store(0);
  host_atomic(m->status).store(0);
  hipStream_t st;
  CK(hipStreamCreate(&st));
  auto drop_stream = defer([&] { (void)hipStreamDestroy(st); });
  k_hflag<<<1, 1, 0, st>>>(dm, total, limit_ticks);
  CK(hipGetLastError());
  std::vector<double> rtt;
  rtt.reserve(iters);
  int failed = 0;  // the round trip that did not complete
  for (int i = 1; i <= total && !failed; i++) {
    const auto t0 = Clock::now();
    host_atomic(m->req).store(i, std::memory_order_release);
    const bool ok = host_wait_ge(m->ack, i, &m->status);
    const auto t1 = Clock::now();
    if (!ok)
      failed = i;
    else if (i > kHflagWarm)
      rtt.push_back(elapsed_us(t0, t1));
  }
  // The host gave up: INT_MAX makes a kernel that still waits return at once.
  if (failed) host_atomic(m->req).store(INT_MAX, std::memory_order_release);
  CK(hipStreamSynchronize(st));
  const int code = host_atomic(m->status).load();
  if (failed || code) {
    printf("ERR timeout hflag dev=%d iter=%d wait=%s side=%s\n", dev,
           code ? code >> 2 : failed, code ? "doorbell" : "answer",
           code ? "gpu" : "host");
    return kRcTimeout;
  }
  const Summary s = summarize(rtt);
  printf("HFLAG dev=%d n=%d", dev, iters);
  print_stats("rtt_", s);
  printf("\n");
  fflush(stdout);
  return 0;
}

static int cmd_hflag(int argc, char** argv) {
  static const char* kSyntax = "hflag <iters> <dev>...";
  long long iters = 0;
  if (argc < 2 || !parse_int(argv[0], 1, kMaxIters, &iters))
    return usage_error("hflag", kSyntax);
  std::vector<int> devs;
  for (int i = 1; i < argc; i++) {
    long long d = 0;
    if (!parse_int(argv[i], 0, g_ndev - 1, &d))
      return usage_error("hflag", kSyntax);
    devs.push_back((int)d);
  }
  HFlagMem* m = nullptr;
  CK(hipHostMalloc((void**)&m, sizeof(HFlagMem), hipHostMallocCoherent));
  auto release = defer([&] { (void)hipHostFree(m); });
  for (int dev : devs) {
    const int rc = hflag_one(m, dev, (int)iters);
    if (rc) return rc;
  }
  return 0;
}

// ---- curve: latency versus size, per iteration ----------------------------

// Multi-block generalisation of k_pingpong (the lat protocol). Per iteration i
// the responder blocks wait for flag >= i; then every block pushes its part of
// the payload into the peer's uncached landing zone, every thread fences at
// system scope and the block synchronises. Thread 0 then adds 1 to the local
// arrival counter (device-scope acq_rel; zeroed before the launch); the block
// that saw old == i * gridDim.x - 1 is the last to arrive in iteration i and
// alone release-stores flag i (system scope) into the peer. No block starts
// iteration i + 1 before the peer answered iteration i, so the counter never
// mixes two iterations. The initiator pushes first and then waits for the
// answer in every block; block 0 thread 0 stamps wall_clock64() before its push
// (t0) and after its own wait and barrier (t1) and stores t1 - t0 (RTT ticks).
// Waits stop at the deadline and set the status word.
struct Pp2Args {
  uint4* peer_buf;   // peer's uncached landing zone
  const uint4* src;  // local pattern source
  int* peer_flag;    // peer's flag array: this side stores flag i here
  int* my_flag;      // local flag array: the peer stores here
  int* ctl;          // local control block (status, arrival counter)
  long long* ticks;  // initiator: RTT ticks of iteration i at [i - 1]
  size_t bytes;      // payload, multiple of 4
  int iters;
  int initiator;
  long long timeout_ticks;
};

// This block's share of a payload push. Below 16 bytes block 0 stores 4-byte
// words; otherwise 16-byte vectors are strided over all threads of the grid
// and block 0 stores the remaining (up to 3) words.
__device__ __forceinline__ void push_part(uint4* dst, const uint4* src,
                                          size_t bytes) {
  const size_t nvec = bytes / 16;
  const size_t words = (bytes - nvec * 16) / 4;
  if (nvec) {
    const size_t gid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t gs = (size_t)gridDim.x * blockDim.x;
    for (size_t j = gid; j < nvec; j += gs) dst[j] = src[j];
  }
  if (blockIdx.x == 0 && threadIdx.x < words)
    reinterpret_cast<int*>(dst + nvec)[threadIdx.x] =
        reinterpret_cast<const int*>(src + nvec)[threadIdx.x];
}

__global__ void __launch_bounds__(256) k_pp2(Pp2Args a) {
  __shared__ int s_stop;
  if (threadIdx.x == 0) s_stop = 0;
  __syncthreads();
  const bool lead = blockIdx.x == 0 && threadIdx.x == 0;
  for (int i = 1; i <= a.iters; i++) {
    long long t0 = 0;
    if (!a.initiator) {
      if (!block_wait_flag(a.my_flag, i, a.timeout_ticks, &s_stop)) {
        report_timeout(a.ctl + kCtlStatus, i, kWaitFlag);
        return;
      }
    } else if (lead) {
      t0 = wall_clock64();
    }
    compiler_fence();
    push_part(a.peer_buf, a.src, a.bytes);
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0 &&
        __hip_atomic_fetch_add(a.ctl + kCtlArrive, 1, __ATOMIC_ACQ_REL,
                               __HIP_MEMORY_SCOPE_AGENT) ==
            i * (int)gridDim.x - 1)
      __hip_atomic_store(a.peer_flag, i, __ATOMIC_RELEASE,
                         __HIP_MEMORY_SCOPE_SYSTEM);
    if (a.initiator) {
      if (!block_wait_flag(a.my_flag, i, a.timeout_ticks, &s_stop)) {
        report_timeout(a.ctl + kCtlStatus, i, kWaitFlag);
        return;
      }
      if (lead) a.ticks[i - 1] = wall_clock64() - t0;
    }
  }
}

// Pull latency on the reading GPU. Per iteration all threads load their
// strided part of `bytes` from the peer's uncached source (window `win`, a
// different window each iteration), XOR-reduce into a register and store it to
// a local sink word, which forces the loads to complete and keeps them alive;
// __syncthreads() then covers the block. With more than one block a grid
// barrier before the first load and another after the last store line the
// blocks up; block 0 thread 0 stamps wall_clock64() after the first and after
// the last barrier (without barriers: around the loads) and stores the
// difference. Waits stop at the deadline and set the status word.
struct PullArgs {
  const uint4* src;  // base of the peer's uncached source slice
  unsigned* sink;    // one word per thread of the grid
  long long* ticks;  // ticks of iteration `it` at [it]
  int* ctl;          // status word and grid barrier words
  size_t bytes;      // multiple of 4
  size_t stride;     // window stride in bytes (multiple of 256)
  size_t windows;    // windows that fit the slice
  int iters;
  long long timeout_ticks;
};

// XOR of this thread's loads of the `bytes` at `base` (same split as
// push_part: below 16 bytes 4-byte words, else 16-byte vectors plus a tail).
__device__ __forceinline__ unsigned pull_part(const char* base, size_t bytes,
                                              size_t gid, size_t gs) {
  const uint4* p = reinterpret_cast<const uint4*>(base);
  const size_t nvec = bytes / 16;
  const size_t words = (bytes - nvec * 16) / 4;
  unsigned acc = 0;
  for (size_t j = gid; j < nvec; j += gs) {
    const uint4 v = p[j];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (gid < words) acc ^= reinterpret_cast<const unsigned*>(p + nvec)[gid];
  return acc;
}

__global__ void __launch_bounds__(256) k_pull_lat(PullArgs a) {
  __shared__ int s_stop;
  if (threadIdx.x == 0) s_stop = 0;
  __syncthreads();
  const size_t gid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  const size_t gs = (size_t)gridDim.x * blockDim.x;
  const bool lead = blockIdx.x == 0 && threadIdx.x == 0;
  size_t win = 0;
  for (int it = 0; it < a.iters; it++) {
    if (gridDim.x > 1 && !grid_barrier(a.ctl, a.timeout_ticks, &s_stop)) {
      report_timeout(a.ctl + kCtlStatus, it + 1, kWaitBarrier);
      return;
    }
    long long t0 = 0;
    if (lead) t0 = wall_clock64();
    compiler_fence();
    a.sink[gid] =
        pull_part(reinterpret_cast<const char*>(a.src) + win * a.stride,
                  a.bytes, gid, gs);
    __syncthreads();
    if (gridDim.x > 1 && !grid_barrier(a.ctl, a.timeout_ticks, &s_stop)) {
      report_timeout(a.ctl + kCtlStatus, it + 1, kWaitBarrier);
      return;
    }
    if (lead) a.ticks[it] = wall_clock64() - t0;
    if (++win == a.windows) win = 0;
  }
}

// Stream, events and device words of one GPU taking part in a curve pair.
struct GpuRig {
  int dev = -1;
  hipStream_t st = nullptr;
  hipEvent_t e0 = nullptr, e1 = nullptr;
  int* ctl = nullptr;          // kCtlWords ints: status, counter, barrier
  long long* ticks = nullptr;  // one entry per iteration
  unsigned* sink = nullptr;    // pull: one word per thread of the grid

  GpuRig() = default;
  GpuRig(const GpuRig&) = delete;
  GpuRig& operator=(const GpuRig&) = delete;
  ~GpuRig() {
    if (dev < 0) return;
    (void)hipSetDevice(dev);
    if (sink) (void)hipFree(sink);
    if (ticks) (void)hipFree(ticks);
    if (ctl) (void)hipFree(ctl);
    if (e1) (void)hipEventDestroy(e1);
    if (e0) (void)hipEventDestroy(e0);
    if (st) (void)hipStreamDestroy(st);
  }
  // tick_iters > 0 allocates the per-iteration tick array.
  void open(int d, int tick_iters, bool pull) {
    dev = d;
    CK(hipSetDevice(d));
    CK(hipStreamCreate(&st));
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    CK(hipMalloc((void**)&ctl, kCtlWords * sizeof(int)));
    if (tick_iters > 0)
      CK(hipMalloc((void**)&ticks, (size_t)tick_iters * sizeof(long long)));
    if (pull)
      CK(hipMalloc((void**)&sink, (size_t)kMaxBlocks * 256 * sizeof(unsigned)));
  }
};

// One measured cell: samples in the unit of the CURVE line and what sizes the
// next cell.
struct Cell {
  std::vector<double> us;
  double ev_us = -1;   // event-timed cross-check, < 0 when absent
  double iter_us = 0;  // mean wall time of one iteration
  int fail_dev = -1;   // GPU whose status word reported a timeout
  int fail_code = 0;
};

static void zero_dev(int dev, void* p, size_t bytes) {
  CK(hipSetDevice(dev));
  CK(hipMemset(p, 0, bytes));
  CK(hipDeviceSynchronize());
}

static int read_status(int dev, const int* ctl) {
  int s = 0;
  CK(hipSetDevice(dev));
  CK(hipMemcpy(&s, ctl + kCtlStatus, sizeof s, hipMemcpyDeviceToHost));
  return s;
}

// The first n tick samples of g in microseconds times `scale`.
static std::vector<double> read_ticks(const GpuRig& g, int n, double scale) {
  std::vector<long long> t(n);
  CK(hipSetDevice(g.dev));
  CK(hipMemcpy(t.data(), g.ticks, (size_t)n * sizeof(long long),
               hipMemcpyDeviceToHost));
  std::vector<double> us(n);
  const double mhz = tick_mhz(g.dev);
  for (int i = 0; i < n; i++) us[i] = (double)t[i] / mhz * scale;
  return us;
}

// CURVE push: responder kernel first, then the initiator between two events,
// as cmd_lat launches its pair. Samples are RTT / 2 per iteration.
static bool curve_push(GpuRig& ini, GpuRig& rsp, size_t bytes, int blocks,
                       int n, Cell* out) {
  const int a = ini.dev, b = rsp.dev;
  const Pp2Args ia{g_dev[b].dst,
                   g_dev[a].src,
                   g_dev[b].flag,
                   g_dev[a].flag,
                   ini.ctl,
                   ini.ticks,
                   bytes,
                   n,
                   1,
                   deadline_ticks(a)};
  const Pp2Args ra{g_dev[a].dst,
                   g_dev[b].src,
                   g_dev[a].flag,
                   g_dev[b].flag,
                   rsp.ctl,
                   nullptr,
                   bytes,
                   n,
                   0,
                   deadline_ticks(b)};
  zero_dev(a, g_dev[a].flag, 4096);
  zero_dev(b, g_dev[b].flag, 4096);
  zero_dev(a, ini.ctl, kCtlWords * sizeof(int));
  zero_dev(b, rsp.ctl, kCtlWords * sizeof(int));
  CK(hipSetDevice(b));
  k_pp2<<<blocks, 256, 0, rsp.st>>>(ra);
  CK(hipGetLastError());
  CK(hipSetDevice(a));
  CK(hipEventRecord(ini.e0, ini.st));
  k_pp2<<<blocks, 256, 0, ini.st>>>(ia);
  CK(hipGetLastError());
  CK(hipEventRecord(ini.e1, ini.st));
  CK(hipStreamSynchronize(ini.st));
  CK(hipSetDevice(b));
  CK(hipStreamSynchronize(rsp.st));
  const int sa = read_status(a, ini.ctl), sb = read_status(b, rsp.ctl);
  if (sa || sb) {
    out->fail_dev = sa ? a : b;
    out->fail_code = sa ? sa : sb;
    return false;
  }
  float ms = 0;
  CK(hipSetDevice(a));
  CK(hipEventElapsedTime(&ms, ini.e0, ini.e1));
  out->iter_us = ms * 1000.0 / n;
  out->ev_us = out->iter_us / 2.0;
  out->us = read_ticks(ini, n, 0.5);
  return true;
}

// CURVE pull: the kernel runs on the destination GPU `g` and reads the source
// GPU's uncached buffer; window stride is the payload rounded up to 256 bytes
// so even 4-byte payloads of consecutive iterations sit in different lines.
static bool curve_pull(GpuRig& g, int src, size_t bytes, int blocks, int n,
                       Cell* out) {
  const size_t stride = (bytes + 255) & ~(size_t)255;
  const PullArgs pa{g_dev[src].usrc,
                    g.sink,
                    g.ticks,
                    g.ctl,
                    bytes,
                    stride,
                    kMaxCurveBytes / stride,
                    n,
                    deadline_ticks(g.dev)};
  zero_dev(g.dev, g.ctl, kCtlWords * sizeof(int));
  CK(hipSetDevice(g.dev));
  CK(hipEventRecord(g.e0, g.st));
  k_pull_lat<<<blocks, 256, 0, g.st>>>(pa);
  CK(hipGetLastError());
  CK(hipEventRecord(g.e1, g.st));
  CK(hipStreamSynchronize(g.st));
  const int s = read_status(g.dev, g.ctl);
  if (s) {
    out->fail_dev = g.dev;
    out->fail_code = s;
    return false;
  }
  float ms = 0;
  CK(hipEventElapsedTime(&ms, g.e0, g.e1));
  out->iter_us = out->ev_us = ms * 1000.0 / n;
  out->us = read_ticks(g, n, 1.0);
  return true;
}

// CURVE sdma: host-timed hipMemcpyPeerAsync src -> dst plus stream sync; the
// stream lives on the source GPU as in cmd_bw.
static bool curve_sdma(GpuRig& g, int src, int dst, size_t bytes, int n,
                       Cell* out) {
  std::vector<double> us(n);
  double sum = 0;
  CK(hipSetDevice(src));
  for (double& u : us) {
    const auto t0 = Clock::now();
    CK(hipMemcpyPeerAsync(g_dev[dst].dst, dst, g_dev[src].src, src, bytes,
                          g.st));
    CK(hipStreamSynchronize(g.st));
    u = elapsed_us(t0, Clock::now());
    sum += u;
  }
  out->us = std::move(us);
  out->iter_us = sum / n;
  return true;
}

enum class CMode { Push, Pull, Sdma };

// Automatic block count: one block up to 64 KiB, then 16 KiB per block, 64 max.
static int auto_blocks(size_t bytes) {
  return bytes <= 65536 ? 1 : (int)std::min<size_t>(64, bytes / 16384);
}

static int curve_timeout(const char* mode, int a, int b, size_t size,
                         int blocks, const Cell& c) {
  printf(
      "ERR timeout curve mode=%s src=%d dst=%d size=%zu blocks=%d gpu=%d "
      "iter=%d wait=%s\n",
      mode, a, b, size, blocks, c.fail_dev, c.fail_code >> 2,
      (c.fail_code & 3) == kWaitBarrier ? "barrier" : "flag");
  return kRcTimeout;
}

// curve <mode> <iters> <blocks> <sizes_csv> <a:b>...: one CURVE line per pair
// and size. <iters> is the maximum: 10 pilot iterations size the cell to a
// 0.25 s budget (at least 20 iterations), the used n is printed. The first
// max(2, n / 20) samples are dropped as warm-up.
static int cmd_curve(int argc, char** argv) {
  static const char* kSyntax =
      "curve <push|pull|sdma> <iters> <blocks> <sizes_csv> <a:b>...  "
      "(sizes: multiples of 4, 4..67108864; blocks 0 = auto, <= 256)";
  if (argc < 5) return usage_error("curve", kSyntax);
  const std::string m = argv[0];
  const CMode mode = m == "push"   ? CMode::Push
                     : m == "pull" ? CMode::Pull
                                   : CMode::Sdma;
  long long iters = 0, req_blocks = 0;
  std::vector<long long> sizes;
  if ((m != "push" && m != "pull" && m != "sdma") ||
      !parse_int(argv[1], 1, kMaxIters, &iters) ||
      !parse_int(argv[2], 0, kMaxBlocks, &req_blocks) ||
      !parse_csv(argv[3], 4, 1LL << 40, &sizes))
    return usage_error("curve", kSyntax);
  for (long long s : sizes) {
    if (s % 4) return usage_error("curve", kSyntax);
    if ((size_t)s > kMaxCurveBytes) {
      printf("ERR size curve size=%lld max=%zu\n", s, kMaxCurveBytes);
      return kRcUsage;
    }
  }
  std::vector<std::pair<int, int>> pairs;
  for (int i = 4; i < argc; i++) {
    int a = 0, b = 0;
    if (!parse_pair(argv[i], &a, &b)) return usage_error("curve", kSyntax);
    pairs.emplace_back(a, b);
  }
  for (const auto& pr : pairs) {
    const int a = pr.first, b = pr.second;
    GpuRig g0, g1;  // push: g0 initiator (a), g1 responder (b); pull: g0 on b
    if (mode == CMode::Push) {
      g0.open(a, (int)iters, false);
      g1.open(b, 0, false);
    } else if (mode == CMode::Pull) {
      g0.open(b, (int)iters, true);
    } else {
      g0.open(a, 0, false);
    }
    for (long long sz : sizes) {
      const size_t size = (size_t)sz;
      const int blocks = mode == CMode::Sdma ? 0
                         : req_blocks        ? (int)req_blocks
                                             : auto_blocks(size);
      auto run = [&](int n, Cell* c) {
        if (mode == CMode::Push) return curve_push(g0, g1, size, blocks, n, c);
        if (mode == CMode::Pull) return curve_pull(g0, a, size, blocks, n, c);
        return curve_sdma(g0, a, b, size, n, c);
      };
      int n = (int)iters;
      Cell cell;
      if (n > 20) {  // pilot decides how many iterations fit the budget
        if (!run(10, &cell))
          return curve_timeout(m.c_str(), a, b, size, blocks, cell);
        n = adaptive_n(cell.iter_us, n, 20);
        cell = Cell();
      }
      if (!run(n, &cell))
        return curve_timeout(m.c_str(), a, b, size, blocks, cell);
      const size_t drop =
          std::min<size_t>(std::max(2, n / 20), cell.us.size() - 1);
      std::vector<double> kept(cell.us.begin() + drop, cell.us.end());
      const Summary s = summarize(kept);
      printf("CURVE mode=%s src=%d dst=%d size=%zu blocks=%d n=%d", m.c_str(),
             a, b, size, blocks, n);
      print_stats("", s);
      if (cell.ev_us >= 0) printf(" ev_us=%.3f", cell.ev_us);
      printf("\n");
      fflush(stdout);
    }
  }
  return 0;
}

// ---- dispatch -------------------------------------------------------------

// Needs: no HIP init at all / devices only / peer access and per-GPU buffers.
enum class Needs { None, Host, Peers };
enum class Where { Both, DirectOnly, BatchOnly };

struct Command {
  const char* name;
  int (*run)(int argc, char** argv);
  Needs needs;  // Peers: not valid with --no-peer
  Where where;  // the legacy probe ran info only directly, latc only in batch
};

static int run_info(int, char**) { return cmd_info(); }

static const Command kCommands[] = {
    {"version", cmd_version, Needs::None, Where::Both},
    {"buses", cmd_buses, Needs::None, Where::Both},
    {"info", run_info, Needs::Peers, Where::DirectOnly},
    {"lat", cmd_lat, Needs::Peers, Where::Both},
    {"latc", cmd_latc, Needs::Peers, Where::BatchOnly},
    {"bw", cmd_bw, Needs::Peers, Where::Both},
    {"a2a", cmd_a2a, Needs::Peers, Where::Both},
    {"hbw", cmd_hbw, Needs::Host, Where::Both},
    {"hlat", cmd_hlat, Needs::Host, Where::Both},
    {"hflag", cmd_hflag, Needs::Host, Where::Both},
    {"curve", cmd_curve, Needs::Peers, Where::Both},
};

static const Command* find_command(const std::string& name) {
  for (const Command& c : kCommands)
    if (name == c.name) return &c;
  return nullptr;
}

// Unknown commands, and commands the legacy probe did not offer in this mode,
// return 1 without output (the batch loop then prints ERR rc=1).
static int dispatch(const std::string& name, int argc, char** argv, bool batch,
                    bool no_peer) {
  const Command* c = find_command(name);
  if (!c || c->where == (batch ? Where::DirectOnly : Where::BatchOnly))
    return 1;
  if (c->needs == Needs::Peers && no_peer)
    return usage_error(c->name, "needs peer access, not valid with --no-peer");
  return c->run(argc, argv);
}

// One command per stdin line, one process, one allocation. READY once after
// the init, then per command: the ### echo, its output, ERR rc=<n> when it
// failed, and END rc=<n> ms=<t>.
static int run_batch(bool no_peer) {
  std::vector<double> mhz(g_ndev);  // calibrate first: exit() must not cut
  for (int d = 0; d < g_ndev; d++) mhz[d] = tick_mhz(d);  // the READY record
  printf("READY devices=%d tick_mhz=", g_ndev);
  for (int d = 0; d < g_ndev; d++) printf("%s%.3f", d ? "," : "", mhz[d]);
  printf(" version=%s\n", kVersion);
  fflush(stdout);
  char line[4096];
  while (fgets(line, sizeof line, stdin)) {
    std::vector<std::string> tok;
    for (char* t = strtok(line, " \t\r\n"); t; t = strtok(nullptr, " \t\r\n"))
      tok.push_back(t);
    if (tok.empty() || tok[0][0] == '#') continue;
    std::vector<char*> av;
    for (auto& t : tok) av.push_back(t.data());
    printf("###");
    for (auto& t : tok) printf(" %s", t.c_str());
    printf("\n");
    const auto t0 = Clock::now();
    const int rc =
        dispatch(tok[0], (int)av.size() - 1, av.data() + 1, true, no_peer);
    if (rc) printf("ERR rc=%d\n", rc);
    printf("END rc=%d ms=%.1f\n", rc, elapsed_us(t0, Clock::now()) / 1e3);
    fflush(stdout);
  }
  return 0;
}

int main(int argc, char** argv) {
  int ai = 1;
  bool no_peer = false;  // --no-peer: no peer access, no per-GPU buffers
  if (ai < argc && strcmp(argv[ai], "--no-peer") == 0) {
    no_peer = true;
    ai++;
  }
  if (ai >= argc) return 1;
  const std::string c = argv[ai];
  const bool batch = c == "batch";
  const Command* cmd = find_command(c);
  if (!batch && (!cmd || cmd->where == Where::BatchOnly)) return 1;
  if (cmd && cmd->needs == Needs::None)  // version, buses: no HIP init at all
    return cmd->run(argc - ai - 1, argv + ai + 1);
  CK(hipGetDeviceCount(&g_ndev));
  if (!no_peer) {
    enable_all_peers();
    alloc_all();
  }
  if (batch) return run_batch(no_peer);
  return dispatch(c, argc - ai - 1, argv + ai + 1, false, no_peer);
}
