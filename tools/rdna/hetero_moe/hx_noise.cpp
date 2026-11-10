// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// hx_noise: standalone PCIe background-traffic generator (no torch, plain
// HIP). Run it next to the hx loopback soak to amplify rare ordering races:
// it floods the fabric between two GPUs with posted writes, SDMA copies or
// non-posted reads while the soak runs on other GPUs behind the same switch.
//
// Usage (the process must see exactly two GPUs):
//   run as HIP_VISIBLE_DEVICES=S,D hx_noise [options]
//   ordinal 0 = source S, ordinal 1 = destination D. The caller must hold the
//   leases of BOTH GPUs (S and D) before starting it; the program refuses to
//   start unless HIP_VISIBLE_DEVICES is set and exactly two devices are
//   visible.
//
//   --mode kernel|copy|read|bidir-kernel   (default kernel)
//       kernel        grid-stride kernel on S stores uint4 vectors (iteration
//                     counter mixed with the index) to D's buffer: a flood of
//                     posted writes S->D
//       copy          hipMemcpyPeerAsync loops S->D (SDMA/blit engines)
//       read          kernel on S loads D's buffer (non-posted reads) and
//                     accumulates into a sink buffer on S
//       bidir-kernel  kernel mode plus a second kernel on D (second stream)
//                     storing to a buffer on S concurrently
//   --bytes N         buffer size on D, also the source buffer on S (default
//                     64 MiB; rounded down to a multiple of --chunk-bytes)
//   --seconds T       run time (default 60; 0 = until SIGTERM/SIGINT)
//   --blocks B        grid blocks (default 64)   --threads 256
//   --chunk-bytes C   bytes stored/loaded per thread-iteration, a multiple of
//                     16 (default 16)
//   --idle-us U       sleep between batches of passes (default 0)
//   --report-s R      report interval in seconds (default 5)
//
// Output (stdout, line buffered):
//   NOISE start mode=.. src=<bus> dst=<bus> bytes=.. blocks=.. ...
//   NOISE mode=.. GB/s=..                  every --report-s seconds
//   NOISE done iters=.. bytes=.. GB/s_avg=..
// GB/s counts payload bytes of all directions (1 GB = 1e9 B). One iteration is
// one pass over the buffer (bidir-kernel counts two passes per iteration).
// SIGTERM/SIGINT: finish the running batch, synchronise, free, exit 0.
// Exit codes: 0 ok, 2 HIP runtime error, 4 no peer access, 64 usage.
//
// Build (no GPU is touched):
//   flock ~/w4a8_runs/audit/build.lock nice -n 19 /opt/rocm/core-7.14/bin/hipcc
//     --offload-arch=gfx1030 -O3 -std=c++17 -x hip -Wall -Wextra
//     -o ~/w4a8_runs/hxnoise/hx_noise tools/rdna/hetero_moe/hx_noise.cpp
#include <hip/hip_runtime.h>

#include <signal.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

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

volatile sig_atomic_t g_stop = 0;
void on_signal(int) { g_stop = 1; }

constexpr int kBatch = 4;  // passes queued between synchronisations

__host__ __device__ inline uint64_t mix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

// Each thread handles whole chunks of `vec` uint4 (vec * 16 bytes); chunk c
// covers uint4 [c * vec, (c + 1) * vec). Data varies with the iteration and
// the index so stores are never idempotent.
__global__ void k_store(uint4* dst, size_t nchunks, uint32_t vec,
                        uint64_t iter) {
  const size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t gs = (size_t)gridDim.x * blockDim.x;
  for (size_t c = gid; c < nchunks; c += gs) {
    uint4* p = dst + c * vec;
    for (uint32_t v = 0; v < vec; v++) {
      uint64_t a = mix64((iter << 40) ^ (c * vec + v));
      p[v] = make_uint4((uint32_t)a, (uint32_t)(a >> 32), (uint32_t)iter,
                        (uint32_t)(iter >> 32));
    }
  }
}

// Loads start at a rotating offset so consecutive passes do not hit the same
// cache lines in the same order; the sum goes to `sink` so loads stay live.
__global__ void k_read(const uint4* src, size_t nchunks, uint32_t vec,
                       size_t off, uint64_t* sink) {
  const size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t gs = (size_t)gridDim.x * blockDim.x;
  uint64_t acc = 0;
  for (size_t c = gid; c < nchunks; c += gs) {
    size_t cc = c + off;
    if (cc >= nchunks) cc -= nchunks;
    const uint4* p = src + cc * vec;
    for (uint32_t v = 0; v < vec; v++) {
      uint4 x = p[v];
      acc += ((uint64_t)x.y << 32 | x.x) ^ ((uint64_t)x.w << 32 | x.z);
    }
  }
  sink[gid] = acc;
}

[[noreturn]] void usage(const char* why) {
  fprintf(stderr,
          "hx_noise: %s\n"
          "run as HIP_VISIBLE_DEVICES=S,D hx_noise [--mode "
          "kernel|copy|read|bidir-kernel] [--bytes N] [--seconds T] [--blocks "
          "B] [--threads 256] [--chunk-bytes C] [--idle-us U] [--report-s R]\n"
          "ordinal 0 = source S, ordinal 1 = destination D; the caller must "
          "hold the leases of both GPUs (see the header of hx_noise.cpp)\n",
          why);
  exit(64);
}

enum Mode { kKernel, kCopy, kRead, kBidir };

struct Opts {
  Mode mode = kKernel;
  std::string mode_name = "kernel";
  size_t bytes = 64u << 20;
  double seconds = 60;
  int blocks = 64, threads = 256;
  size_t chunk = 16;
  long idle_us = 0;
  double report_s = 5;
};

Opts parse(int argc, char** argv) {
  Opts o;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) usage(("missing value for " + a).c_str());
      return argv[++i];
    };
    if (a == "--mode") {
      o.mode_name = val();
      if (o.mode_name == "kernel")
        o.mode = kKernel;
      else if (o.mode_name == "copy")
        o.mode = kCopy;
      else if (o.mode_name == "read")
        o.mode = kRead;
      else if (o.mode_name == "bidir-kernel")
        o.mode = kBidir;
      else
        usage("--mode must be kernel|copy|read|bidir-kernel");
    } else if (a == "--bytes") {
      o.bytes = strtoull(val().c_str(), nullptr, 0);
    } else if (a == "--seconds") {
      o.seconds = atof(val().c_str());
    } else if (a == "--blocks") {
      o.blocks = atoi(val().c_str());
    } else if (a == "--threads") {
      o.threads = atoi(val().c_str());
    } else if (a == "--chunk-bytes") {
      o.chunk = strtoull(val().c_str(), nullptr, 0);
    } else if (a == "--idle-us") {
      o.idle_us = atol(val().c_str());
    } else if (a == "--report-s") {
      o.report_s = atof(val().c_str());
    } else {
      usage(("unknown option " + a).c_str());
    }
  }
  if (o.blocks < 1 || o.threads < 1 || o.threads > 1024)
    usage("--blocks >= 1 and --threads in 1..1024");
  if (o.chunk < 16 || o.chunk % 16)
    usage("--chunk-bytes must be a multiple of 16");
  if (o.bytes < o.chunk) usage("--bytes must be at least --chunk-bytes");
  o.bytes -= o.bytes % o.chunk;
  if (o.seconds < 0 || o.report_s <= 0 || o.idle_us < 0)
    usage("--seconds >= 0, --report-s > 0, --idle-us >= 0");
  return o;
}

std::string bus_of(int d) {
  char b[64] = {0};
  CK(hipDeviceGetPCIBusId(b, sizeof b, d));
  return b;
}

void enable_peer(int from, int to) {
  int can = 0;
  CK(hipDeviceCanAccessPeer(&can, from, to));
  if (!can) {
    fprintf(stderr,
            "hx_noise: no peer access from visible ordinal %d to %d "
            "(hipDeviceCanAccessPeer = 0)\n",
            from, to);
    exit(4);
  }
  CK(hipSetDevice(from));
  hipError_t e = hipDeviceEnablePeerAccess(to, 0);
  if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
    fprintf(stderr,
            "hx_noise: hipDeviceEnablePeerAccess(%d -> %d) failed: %s\n", from,
            to, hipGetErrorString(e));
    exit(4);
  }
  (void)hipGetLastError();
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
  if (ndev != 2) {
    fprintf(stderr,
            "hx_noise: %d devices visible (HIP_VISIBLE_DEVICES=%s), exactly 2 "
            "are required\n",
            ndev, vis);
    return 64;
  }
  struct sigaction sa;
  memset(&sa, 0, sizeof sa);
  sa.sa_handler = on_signal;
  sigemptyset(&sa.sa_mask);
  sigaction(SIGINT, &sa, nullptr);
  sigaction(SIGTERM, &sa, nullptr);

  const int S = 0, D = 1;
  const size_t nchunks = o.bytes / o.chunk;
  const uint32_t vec = (uint32_t)(o.chunk / 16);
  const size_t nthreads = (size_t)o.blocks * o.threads;
  const bool bidir = o.mode == kBidir;

  // Peer access first: fail with exit 4 before allocating anything.
  enable_peer(S, D);
  if (bidir) enable_peer(D, S);

  void *d_buf = nullptr, *s_src = nullptr, *s_rev = nullptr;
  uint64_t* sink = nullptr;
  CK(hipSetDevice(D));
  CK(hipMalloc(&d_buf, o.bytes));
  CK(hipMemset(d_buf, 0x5A, o.bytes));
  CK(hipSetDevice(S));
  CK(hipMalloc(&s_src, o.bytes));
  CK(hipMemset(s_src, 0xA5, o.bytes));
  if (bidir) {
    CK(hipMalloc(&s_rev, o.bytes));
    CK(hipMemset(s_rev, 0, o.bytes));
  }
  if (o.mode == kRead)
    CK(hipMalloc((void**)&sink, nthreads * sizeof(uint64_t)));
  CK(hipDeviceSynchronize());
  hipStream_t st_s, st_d = nullptr;
  CK(hipStreamCreateWithFlags(&st_s, hipStreamNonBlocking));
  if (bidir) {
    CK(hipSetDevice(D));
    CK(hipStreamCreateWithFlags(&st_d, hipStreamNonBlocking));
    CK(hipSetDevice(S));
  }

  printf(
      "NOISE start mode=%s src=%s dst=%s bytes=%zu blocks=%d threads=%d "
      "chunk_bytes=%zu idle_us=%ld seconds=%.1f report_s=%.1f\n",
      o.mode_name.c_str(), bus_of(S).c_str(), bus_of(D).c_str(), o.bytes,
      o.blocks, o.threads, o.chunk, o.idle_us, o.seconds, o.report_s);

  uint64_t iter = 0;       // passes completed per direction
  double total_bytes = 0;  // all directions
  double window_bytes = 0;
  const size_t passes_dirs = bidir ? 2 : 1;
  const auto t0 = Clock::now();
  auto t_rep = t0;
  while (!g_stop) {
    const double el = std::chrono::duration<double>(Clock::now() - t0).count();
    if (o.seconds > 0 && el >= o.seconds) break;
    CK(hipSetDevice(S));
    for (int b = 0; b < kBatch; b++) {
      const uint64_t it = iter + 1 + b;
      switch (o.mode) {
        case kKernel:
        case kBidir:
          k_store<<<o.blocks, o.threads, 0, st_s>>>((uint4*)d_buf, nchunks, vec,
                                                    it);
          break;
        case kCopy:
          CK(hipMemcpyPeerAsync(d_buf, D, s_src, S, o.bytes, st_s));
          break;
        case kRead: {
          size_t off = (size_t)((it * nthreads) % nchunks);
          k_read<<<o.blocks, o.threads, 0, st_s>>>((const uint4*)d_buf, nchunks,
                                                   vec, off, sink);
          break;
        }
      }
      CK(hipGetLastError());
    }
    if (bidir) {
      CK(hipSetDevice(D));
      for (int b = 0; b < kBatch; b++) {
        k_store<<<o.blocks, o.threads, 0, st_d>>>((uint4*)s_rev, nchunks, vec,
                                                  iter + 1 + b);
        CK(hipGetLastError());
      }
      CK(hipStreamSynchronize(st_d));
      CK(hipSetDevice(S));
    }
    CK(hipStreamSynchronize(st_s));
    iter += kBatch;
    const double b = (double)kBatch * (double)o.bytes * (double)passes_dirs;
    total_bytes += b;
    window_bytes += b;
    auto now = Clock::now();
    double dt = std::chrono::duration<double>(now - t_rep).count();
    if (dt >= o.report_s) {
      printf("NOISE mode=%s GB/s=%.2f\n", o.mode_name.c_str(),
             window_bytes / dt / 1e9);
      window_bytes = 0;
      t_rep = now;
    }
    if (o.idle_us > 0) usleep((useconds_t)o.idle_us);
  }
  const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
  CK(hipDeviceSynchronize());
  if (bidir) {
    CK(hipSetDevice(D));
    CK(hipDeviceSynchronize());
    CK(hipStreamDestroy(st_d));
    CK(hipSetDevice(S));
  }
  CK(hipStreamDestroy(st_s));
  CK(hipFree(s_src));
  if (s_rev) CK(hipFree(s_rev));
  if (sink) CK(hipFree(sink));
  CK(hipSetDevice(D));
  CK(hipFree(d_buf));
  printf("NOISE done iters=%llu bytes=%.0f GB/s_avg=%.2f\n",
         (unsigned long long)iter, total_bytes,
         secs > 0 ? total_bytes / secs / 1e9 : 0.0);
  return 0;
}
