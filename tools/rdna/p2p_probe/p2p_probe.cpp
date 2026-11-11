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
//
// Build: hipcc -x hip -O3 --offload-arch=gfx1030 -o p2p_probe p2p_probe.cpp
// -pthread
#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
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

int main(int argc, char** argv) {
  if (argc < 2) return 1;
  CK(hipGetDeviceCount(&g_ndev));
  enable_all_peers();
  alloc_all();
  std::string c = argv[1];
  if (c == "info") return cmd_info();
  if (c == "lat") return cmd_lat(argc - 2, argv + 2);
  if (c == "bw") return cmd_bw(argc - 2, argv + 2);
  if (c == "a2a") return cmd_a2a(argc - 2, argv + 2);
  if (c == "batch") {  // one command per stdin line, one process, one alloc
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
      int rc = 1;
      if (tok[0] == "lat") rc = cmd_lat((int)av.size() - 1, av.data() + 1);
      if (tok[0] == "bw") rc = cmd_bw((int)av.size() - 1, av.data() + 1);
      if (tok[0] == "latc") rc = cmd_latc((int)av.size() - 1, av.data() + 1);
      if (tok[0] == "a2a") rc = cmd_a2a((int)av.size() - 1, av.data() + 1);
      if (rc) printf("ERR rc=%d\n", rc);
      fflush(stdout);
    }
    return 0;
  }
  return 1;
}
