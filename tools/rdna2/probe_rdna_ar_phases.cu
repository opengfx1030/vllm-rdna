// Phase dissection probe for rdna_ar_oneshot (gfx1030, 4x V620), take 3.
//
// Event-based timing around a launch measures the host's launch latency, not
// the kernel, so this probe timestamps the kernel from the inside with
// wall_clock64: entry / after-push / after-sync / exit, per block. The host
// harness only has to keep the ranks roughly in lockstep (one AR per rank per
// iteration, all ranks released together by a thread barrier).
//
//   hipcc -O3 -std=c++20 --offload-arch=gfx1030 -o /tmp/ar_probe3
//   /tmp/ar_probe3.cu /tmp/ar_probe3 <world> <bytes> <nblocks> <pace> <mode>
//   <iters> <warmup> [variant]
//
// modes: full push sync empty nosync noreduce nopush
// variant: 0 = system fence (served code)
//          1 = agent fence (__threadfence)
//          2 = no fence
//          3 = last-arriver announces, system fence
//          4 = last-arriver announces, agent fence
//          5 = last-arriver, agent fence, vector flag load
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#define MAXW 8
#define FLAG_PAGE 4096
#define POLL_PAUSE() __builtin_amdgcn_s_sleep(8)
#define TS_SLOTS 256

struct Peers {
  void* stage[MAXW];
  int* flags[MAXW];
};

template <typename T>
__device__ __forceinline__ void ar_fence(int variant) {
  if (variant == 0 || variant == 3) {
    __threadfence_system();
  } else if (variant == 1 || variant == 4 || variant == 5) {
    __threadfence();
  }
}

template <typename T>
__global__ void ar_kernel(const T* __restrict__ in, T* __restrict__ out,
                          Peers peers, unsigned* arrive, int* seqbuf, int rank,
                          int world, int n, long long max_elems, int nblocks,
                          int pace, int do_push, int do_sync, int do_reduce,
                          int variant, int* abortdev, long long max_polls,
                          unsigned long long* ts, volatile int* go_dev, int gen,
                          int do_bar, int do_ann) {
  __shared__ int s_seq;
  const int t = threadIdx.x, nt = blockDim.x, b = blockIdx.x;
  // Start gate: the host opens it only after every rank's kernel is queued, so
  // the measured span is not host launch skew.
  if (t == 0) {
    while (*go_dev < gen) POLL_PAUSE();
  }
  __syncthreads();
  unsigned long long TS[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  if (t == 0) TS[0] = wall_clock64();
  if (t == 0)
    s_seq =
        __hip_atomic_load(seqbuf, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) +
        1;
  __syncthreads();
  const int seq = s_seq;
  const int p = seq & 1;
  const int gid = b * nt + t, gstride = nblocks * nt;

  if (do_push) {
    for (int k = 1; k < world; k++) {
      const int j = (rank + k) % world;
      T* dst = reinterpret_cast<T*>(peers.stage[j]) +
               ((long long)p * world + rank) * max_elems;
      if (variant == 8 && pace == 0) {
        const int nvec = n / 8;
        uint4* d4 = reinterpret_cast<uint4*>(dst);
        const uint4* s4 = reinterpret_cast<const uint4*>(in);
        for (int i = gid; i < nvec; i += gstride) d4[i] = s4[i];
        for (int i = nvec * 8 + gid; i < n; i += gstride) dst[i] = in[i];
        continue;
      }
      for (int i = gid; i < n; i += gstride) {
        dst[i] = in[i];
        for (int q = 0; q < pace; q++) __builtin_amdgcn_s_sleep(1);
      }
    }
  }
  __syncthreads();
  if (t == 0) TS[1] = wall_clock64();

  if (do_sync) {
    if (t == 0) {
      if (!do_ann) {
        TS[2] = TS[3] = TS[4] = TS[5] = wall_clock64();
        goto sync_done;
      }
      if (variant == 7) {
        // Per-block announce: no grid barrier at all. Block b announces its own
        // slice and waits only for the peers' block-b flags.
        __threadfence_system();
        for (int j = 0; j < world; j++)
          __hip_atomic_store(&peers.flags[j][rank * nblocks + b], seq,
                             __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        TS[2] = TS[3] = TS[4] = wall_clock64();
        for (int j = 0; j < world; j++) {
          if (j == rank) continue;
          long long sp = 0;
          while (__hip_atomic_load(&peers.flags[rank][j * nblocks + b],
                                   __ATOMIC_ACQUIRE,
                                   __HIP_MEMORY_SCOPE_SYSTEM) < seq) {
            POLL_PAUSE();
            if (++sp > max_polls) {
              atomicAdd(abortdev, 1);
              break;
            }
          }
        }
        TS[5] = wall_clock64();
        if (b == 0)
          __hip_atomic_store(seqbuf, seq, __ATOMIC_RELEASE,
                             __HIP_MEMORY_SCOPE_AGENT);
        goto sync_done;
      }
      const bool last_arriver = (variant >= 3) && do_bar;
      ar_fence<T>(variant);
      unsigned old = 0;
      if (do_bar) old = atomicAdd(&arrive[p], 1u);
      TS[2] = wall_clock64();
      if (!last_arriver) {
        if (b == 0) {
          long long spins = 0;
          if (do_bar)
            while (__hip_atomic_load(&arrive[p], __ATOMIC_ACQUIRE,
                                     __HIP_MEMORY_SCOPE_AGENT) <
                   (unsigned)nblocks) {
              POLL_PAUSE();
              if (++spins > max_polls) {
                atomicAdd(abortdev, 1);
                break;
              }
            }
          arrive[p] = 0u;
          arrive[1 - p] = 0u;
          __hip_atomic_store(seqbuf, seq, __ATOMIC_RELEASE,
                             __HIP_MEMORY_SCOPE_AGENT);
          TS[3] = wall_clock64();
          for (int j = 0; j < world; j++)
            __hip_atomic_store(&peers.flags[j][rank], seq, __ATOMIC_RELEASE,
                               __HIP_MEMORY_SCOPE_SYSTEM);
          TS[4] = wall_clock64();
        }
      } else if (old == (unsigned)(nblocks - 1)) {
        arrive[p] = 0u;
        arrive[1 - p] = 0u;
        __hip_atomic_store(seqbuf, seq, __ATOMIC_RELEASE,
                           __HIP_MEMORY_SCOPE_AGENT);
        TS[3] = wall_clock64();
        for (int j = 0; j < world; j++)
          __hip_atomic_store(&peers.flags[j][rank], seq, __ATOMIC_RELEASE,
                             __HIP_MEMORY_SCOPE_SYSTEM);
        TS[4] = wall_clock64();
      }
      int* myflags = peers.flags[rank];
      long long spins = 0;
      if (variant == 5) {
        volatile unsigned long long* fv =
            reinterpret_cast<volatile unsigned long long*>(myflags);
        for (;;) {
          const unsigned long long a = fv[0];
          const unsigned long long bq = fv[1];
          const int f[4] = {(int)(a & 0xffffffffull), (int)(a >> 32),
                            (int)(bq & 0xffffffffull), (int)(bq >> 32)};
          bool miss = false;
          for (int j = 0; j < world; j++)
            if (j != rank && f[j] < seq) miss = true;
          if (!miss) break;
          POLL_PAUSE();
          if (++spins > max_polls) {
            atomicAdd(abortdev, 1);
            break;
          }
        }
      } else {
        for (int j = 0; j < world; j++) {
          if (j == rank) continue;
          while (__hip_atomic_load(&myflags[j], __ATOMIC_ACQUIRE,
                                   __HIP_MEMORY_SCOPE_SYSTEM) < seq) {
            POLL_PAUSE();
            if (++spins > max_polls) {
              atomicAdd(abortdev, 1);
              break;
            }
          }
        }
      }
      TS[5] = wall_clock64();
    }
  }
sync_done:
  __syncthreads();
  if (t == 0) TS[6] = wall_clock64();

  if (do_reduce) {
    const T* mine = reinterpret_cast<const T*>(peers.stage[rank]) +
                    ((long long)p * world) * max_elems;
    if (variant == 8) {
      const int nvec = n / 8;
      const uint4* m4 = reinterpret_cast<const uint4*>(mine);
      uint4* o4 = reinterpret_cast<uint4*>(out);
      const uint4* i4 = reinterpret_cast<const uint4*>(in);
      const long long slot_v = max_elems / 8;
      for (int i = gid; i < nvec; i += gstride) {
        uint4 acc{};
        float f[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int j = 0; j < world; j++) {
          uint4 v = (j == rank) ? i4[i] : m4[(long long)j * slot_v + i];
          const __half* hp = reinterpret_cast<const __half*>(&v);
          for (int q = 0; q < 8; q++) f[q] += __half2float(hp[q]);
        }
        __half* op = reinterpret_cast<__half*>(&acc);
        for (int q = 0; q < 8; q++) op[q] = __float2half(f[q]);
        o4[i] = acc;
      }
      for (int i = nvec * 8 + gid; i < n; i += gstride) {
        float v = 0.f;
        for (int j = 0; j < world; j++)
          v += (j == rank) ? __half2float(in[i])
                           : __half2float(mine[(long long)j * max_elems + i]);
        out[i] = __float2half(v);
      }
      goto reduce_done;
    }
    for (int i = gid; i < n; i += gstride) {
      float v = 0.f;
      for (int j = 0; j < world; j++) {
        float x = (j == rank)
                      ? __half2float(in[i])
                      : __half2float(mine[(long long)j * max_elems + i]);
        v += x;
      }
      out[i] = __float2half(v);
    }
  }
reduce_done:;
  __syncthreads();
  if (t == 0) {
    TS[7] = wall_clock64();
    const int slot = (seq & (TS_SLOTS - 1)) * nblocks + b;
    for (int q = 0; q < 8; q++) ts[slot * 8 + q] = TS[q];
  }
}

static void chk(hipError_t e, const char* what) {
  if (e != hipSuccess) {
    fprintf(stderr, "FATAL %s: %s\n", what, hipGetErrorString(e));
    exit(1);
  }
}

struct Mode {
  int push, sync, reduce;
};

int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc < 8) {
    fprintf(stderr,
            "usage: %s world bytes nblocks pace mode iters warmup [variant]\n",
            argv[0]);
    return 2;
  }
  const int world = atoi(argv[1]);
  const long long bytes = atoll(argv[2]);
  const int nblocks = atoi(argv[3]);
  const int pace = atoi(argv[4]);
  const std::string mode = argv[5];
  const int iters = atoi(argv[6]);
  const int warmup = atoi(argv[7]);
  const int variant = argc > 8 ? atoi(argv[8]) : 0;

  Mode m{1, 1, 1};
  int do_bar = 1;
  int do_ann = 1;
  if (mode == "bwfence") {
    m = {1, 1, 0};
    do_bar = 0;
    do_ann = 0;
  } else if (mode == "ann") {
    m = {0, 1, 0};
    do_bar = 0;
  } else if (mode == "annpush") {
    m = {1, 1, 0};
    do_bar = 0;
  } else if (mode == "full")
    m = {1, 1, 1};
  else if (mode == "nosync")
    m = {1, 0, 1};
  else if (mode == "nopush")
    m = {0, 1, 1};
  else if (mode == "noreduce")
    m = {1, 1, 0};
  else if (mode == "push")
    m = {1, 0, 0};
  else if (mode == "sync")
    m = {0, 1, 0};
  else if (mode == "empty")
    m = {0, 0, 0};
  else {
    fprintf(stderr, "bad mode %s\n", mode.c_str());
    return 2;
  }

  int ndev = 0;
  chk(hipGetDeviceCount(&ndev), "device count");
  if (world > ndev) {
    fprintf(stderr, "world %d > devices %d\n", world, ndev);
    return 2;
  }

  const long long max_bytes = 65536;
  const long long max_elems = max_bytes / 2;
  const int n = (int)(bytes / 2);
  if (n > max_elems) {
    fprintf(stderr, "n %d > max_elems %lld\n", n, max_elems);
    return 2;
  }
  const size_t stage_bytes = 2ull * world * (size_t)max_bytes;
  const size_t ts_bytes =
      (size_t)TS_SLOTS * nblocks * 8 * sizeof(unsigned long long);

  std::vector<void*> stage(world, nullptr);
  std::vector<unsigned*> arrive(world, nullptr);
  std::vector<int*> seqbuf(world, nullptr);
  std::vector<int*> abortbuf(world, nullptr);
  std::vector<__half*> in(world, nullptr), out(world, nullptr);
  std::vector<unsigned long long*> ts(world, nullptr);
  std::vector<int*> cflags(world, nullptr);
  std::vector<hipStream_t> stream(world, nullptr);
  std::vector<hipGraph_t> graph(world, nullptr), graph2(world, nullptr);
  std::vector<hipGraphExec_t> exec(world, nullptr), exec2(world, nullptr);
  int* go_host = nullptr;
  chk(hipHostMalloc(&go_host, 64, hipHostMallocMapped | hipHostMallocCoherent),
      "go host malloc");
  *go_host = 0;
  void* go_dev_any = nullptr;
  chk(hipHostGetDevicePointer(&go_dev_any, go_host, 0), "go dev ptr");
  volatile int* go_dev = reinterpret_cast<volatile int*>(go_dev_any);

  for (int d = 0; d < world; d++) {
    chk(hipSetDevice(d), "set device");
    chk(hipExtMallocWithFlags(&stage[d], stage_bytes + FLAG_PAGE,
                              hipDeviceMallocUncached),
        "stage malloc");
    chk(hipMemset(stage[d], 0, stage_bytes + FLAG_PAGE), "stage memset");
    chk(hipMalloc(&arrive[d], 2 * sizeof(unsigned)), "arrive");
    chk(hipMemset(arrive[d], 0, 2 * sizeof(unsigned)), "arrive memset");
    chk(hipMalloc(&seqbuf[d], sizeof(int)), "seqbuf");
    chk(hipMemset(seqbuf[d], 0, sizeof(int)), "seqbuf memset");
    chk(hipMalloc(&abortbuf[d], sizeof(int)), "abortbuf");
    chk(hipMemset(abortbuf[d], 0, sizeof(int)), "abort memset");
    chk(hipMalloc(&in[d], max_bytes), "in");
    chk(hipMalloc(&out[d], max_bytes), "out");
    chk(hipMalloc(&cflags[d], FLAG_PAGE), "cflags");
    chk(hipMemset(cflags[d], 0, FLAG_PAGE), "cflags memset");
    chk(hipMalloc(&ts[d], ts_bytes), "ts");
    chk(hipMemset(ts[d], 0, ts_bytes), "ts memset");
    std::vector<__half> h(n);
    for (int i = 0; i < n; i++)
      h[i] = __float2half((float)((i % 7) - 3) * 0.5f);
    chk(hipMemcpy(in[d], h.data(), (size_t)n * 2, hipMemcpyHostToDevice),
        "in cp");
    chk(hipMemset(out[d], 0, max_bytes), "out memset");
    chk(hipStreamCreate(&stream[d]), "stream");
  }
  for (int d = 0; d < world; d++) {
    chk(hipSetDevice(d), "set device 2");
    for (int j = 0; j < world; j++) {
      if (j == d) continue;
      hipError_t pe = hipDeviceEnablePeerAccess(j, 0);
      if (pe != hipSuccess && pe != hipErrorPeerAccessAlreadyEnabled) {
        fprintf(stderr, "peer access %d->%d: %s\n", d, j,
                hipGetErrorString(pe));
        return 1;
      }
      (void)hipGetLastError();
    }
  }
  std::vector<Peers*> dpeers(world, nullptr);
  for (int d = 0; d < world; d++) {
    Peers h{};
    for (int j = 0; j < world; j++) {
      h.stage[j] = stage[j];
      h.flags[j] = (variant == 6)
                       ? cflags[j]
                       : reinterpret_cast<int*>(static_cast<char*>(stage[j]) +
                                                stage_bytes);
    }
    chk(hipSetDevice(d), "set device 3");
    chk(hipMalloc(&dpeers[d], sizeof(Peers)), "peers");
    chk(hipMemcpy(dpeers[d], &h, sizeof(Peers), hipMemcpyHostToDevice),
        "peers cp");
  }

  const long long max_polls = 40000000LL;
  auto launch = [&](int d, int gen) {
    ar_kernel<__half><<<nblocks, 256, 0, stream[d]>>>(
        in[d], out[d], *dpeers[d], arrive[d], seqbuf[d], d, world, n, max_elems,
        nblocks, pace, m.push, m.sync, m.reduce, variant, abortbuf[d],
        max_polls, ts[d], go_dev, gen, do_bar, do_ann);
    hipError_t le = hipGetLastError();
    if (le != hipSuccess) {
      fprintf(stderr, "LAUNCH FAIL d=%d: %s\n", d, hipGetErrorString(le));
      exit(1);
    }
  };

  std::barrier<> bqueued(world);
  std::vector<double> host_wall(world, 0.0);
  const int reps = (argc > 9) ? atoi(argv[9]) : 10;

  auto worker = [&](int d) {
    chk(hipSetDevice(d), "worker set device");
    auto t0h = std::chrono::steady_clock::now();
    // Capture the burst: a graph replay costs a few us of host time for the
    // whole burst, so the GPU stays saturated and the clock stays ramped.
    chk(hipStreamBeginCapture(stream[d], hipStreamCaptureModeThreadLocal),
        "begin capture");
    for (int it = 0; it < warmup; it++) launch(d, 0);
    chk(hipStreamEndCapture(stream[d], &graph[d]), "end capture");
    chk(hipGraphInstantiateWithFlags(&exec[d], graph[d], 0), "instantiate");
    for (int r = 0; r < 2; r++)
      chk(hipGraphLaunch(exec[d], stream[d]), "warm graph");
    chk(hipStreamSynchronize(stream[d]), "warm sync");

    chk(hipStreamBeginCapture(stream[d], hipStreamCaptureModeThreadLocal),
        "begin capture 2");
    for (int it = 0; it < iters; it++) launch(d, it + 1);
    chk(hipStreamEndCapture(stream[d], &graph2[d]), "end capture 2");
    chk(hipGraphInstantiateWithFlags(&exec2[d], graph2[d], 0), "instantiate 2");

    bqueued.arrive_and_wait();
    if (d == 0) *go_host = iters;
    for (int r = 0; r < reps; r++)
      chk(hipGraphLaunch(exec2[d], stream[d]), "graph launch");
    chk(hipStreamSynchronize(stream[d]), "sync");
    host_wall[d] =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0h)
            .count() *
        1e6 / (double)(iters * reps);
  };
  std::vector<std::thread> th;
  for (int d = 0; d < world; d++) th.emplace_back(worker, d);
  for (auto& t : th) t.join();

  // output check: every rank's `in` is identical, so out = world * in
  {
    std::vector<__half> h(n);
    chk(hipSetDevice(0), "set 0 check");
    chk(hipMemcpy(h.data(), out[0], (size_t)n * 2, hipMemcpyDeviceToHost),
        "out read");
    int bad = 0;
    for (int i = 0; i < n; i++) {
      const float want =
          world * __half2float(__float2half((float)((i % 7) - 3) * 0.5f));
      if (fabsf(__half2float(h[i]) - want) > 1e-3f) bad++;
    }
    printf("  check: %d/%d wrong\n", bad, n);
  }

  // Aggregate in-kernel phase spans per rank, then the collective.
  const int used = std::min(iters, TS_SLOTS);  // ts holds the LAST 256 ARs
  std::vector<std::vector<double>> span(world), ph_push(world), ph_sync(world),
      ph_red(world), ph_bar(world), ph_ann(world), ph_wait(world);
  for (int d = 0; d < world; d++) {
    chk(hipSetDevice(d), "set dev final");
    std::vector<unsigned long long> h((size_t)TS_SLOTS * nblocks * 8, 0);
    chk(hipMemcpy(h.data(), ts[d], ts_bytes, hipMemcpyDeviceToHost), "ts read");
    const double k = 1000.0 / 100000.0;  // ticks -> us (wall_clock 100 MHz)
    for (int s = 0; s < used; s++) {
      unsigned long long mn0 = ~0ull, mx1 = 0, mx6 = 0, mx7 = 0;
      unsigned long long b2 = 0, b3 = 0, b4 = 0, b5 = 0;
      bool have = false;
      for (int b = 0; b < nblocks; b++) {
        const size_t o = ((size_t)s * nblocks + b) * 8;
        if (h[o] == 0) continue;
        mn0 = std::min(mn0, h[o + 0]);
        mx1 = std::max(mx1, h[o + 1]);
        mx6 = std::max(mx6, h[o + 6]);
        mx7 = std::max(mx7, h[o + 7]);
        if (b == 0) {
          b2 = h[o + 2];
          b3 = h[o + 3];
          b4 = h[o + 4];
          b5 = h[o + 5];
          have = true;
        }
      }
      if (mn0 == ~0ull || mx7 == 0) continue;
      span[d].push_back((mx7 - mn0) * k);
      ph_push[d].push_back((mx1 - mn0) * k);
      ph_sync[d].push_back((mx6 - mx1) * k);
      ph_red[d].push_back((mx7 - mx6) * k);
      if (have && b3 && b4 && b5) {
        ph_bar[d].push_back((b3 - b2) * k);
        ph_ann[d].push_back((b4 - b3) * k);
        ph_wait[d].push_back((b5 - b4) * k);
      }
    }
  }
  std::vector<double> coll;
  {
    size_t m = ~0ull;
    for (int d = 0; d < world; d++) m = std::min(m, span[d].size());
    for (size_t i = 0; i < m; i++) {
      double worst = 0;
      for (int d = 0; d < world; d++) worst = std::max(worst, span[d][i]);
      coll.push_back(worst);
    }
  }
  auto st = [](std::vector<double> v, double* mn, double* med) {
    if (v.empty()) {
      *mn = *med = 0;
      return;
    }
    std::sort(v.begin(), v.end());
    *mn = v.front();
    *med = v[v.size() / 2];
  };
  printf("mode=%-8s world=%d bytes=%lld nblocks=%d pace=%d var=%d n=%d\n",
         mode.c_str(), world, bytes, nblocks, pace, variant, n);
  for (int d = 0; d < world; d++) {
    double mn, med;
    st(span[d], &mn, &med);
    double a, b2, c, d2, e, f, g, h2;
    st(ph_push[d], &a, &b2);
    st(ph_sync[d], &c, &d2);
    st(ph_red[d], &e, &f);
    st(ph_bar[d], &g, &h2);
    printf(
        "  rank %d: total %6.2f/%6.2f | push %5.2f/%5.2f | sync %6.2f/%6.2f "
        "| reduce %5.2f/%5.2f\n",
        d, mn, med, a, b2, c, d2, e, f);
    double g2, h3, i2, j2, k2, l2;
    st(ph_bar[d], &g2, &h3);
    st(ph_ann[d], &i2, &j2);
    st(ph_wait[d], &k2, &l2);
    printf(
        "          [block0] barrier %5.2f/%5.2f | announce %5.2f/%5.2f | "
        "peer-wait %6.2f/%6.2f us\n",
        g2, h3, i2, j2, k2, l2);
  }
  {
    auto pct = [](std::vector<double> v, double p) {
      if (v.empty()) return 0.0;
      std::sort(v.begin(), v.end());
      return v[(size_t)(p * (v.size() - 1))];
    };
    printf(
        "  COLLECTIVE: p05 %7.2f p25 %7.2f p50 %7.2f p75 %7.2f p95 %7.2f max "
        "%7.2f\n",
        pct(coll, 0.05), pct(coll, 0.25), pct(coll, 0.50), pct(coll, 0.75),
        pct(coll, 0.95), pct(coll, 1.0));
    printf("  rank0 spans: ");
    for (int i = 0; i < (int)span[0].size() && i < 256; i++)
      printf("%.0f ", span[0][i]);
    printf("\n");
  }
  for (int d = 0; d < world; d++) {
    chk(hipSetDevice(d), "set dev final2");
    int ab = 0;
    chk(hipMemcpy(&ab, abortbuf[d], sizeof(int), hipMemcpyDeviceToHost),
        "abort read");
    if (ab) printf("  !! rank %d ABORTED (spin cap) %d times\n", d, ab);
  }
  return 0;
}
