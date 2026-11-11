// MonoKernel P0 probe for V620 (gfx1030), ROCm 7.14.0.
// Standalone: no vLLM / fork dependencies.
//
// Tests:
//   [residency] occupancy API x CU count vs. the 72-WG target
//   [capture]   hipGraph stream capture + N replays, plain launch and
//               hipLaunchCooperativeKernel (cooperative_groups grid.sync)
//   [progress]  flag-wait kernel while a busy kernel runs on another stream
//
// Every inter-workgroup wait is bounded by wall_clock64() (s_memrealtime,
// a constant-rate clock, unlike clock64()/s_memtime which follows shader clock).
// On timeout a workgroup records {phase, wg, observed tag} into a sticky error
// record in host-visible (pinned, coherent) memory and every workgroup bails.
// The host polls that record and the stream with its own timeout.
//
// Caveat: grid.sync() itself is NOT bounded (it's the ROCm implementation).
// The cooperative variant therefore also does bounded flag waits around it,
// and the host timeout is the backstop if grid.sync hangs.

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unistd.h>  // _exit

namespace cg = cooperative_groups;

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  printf("  HIP error %s (%d) at %s:%d: %s\n", hipGetErrorName(e_), (int)e_, \
         __FILE__, __LINE__, #x); return e_; } } while (0)

// ---------------------------------------------------------------- config
struct Cfg {
  int      grid      = 72;        // one WG per CU on V620
  int      block     = 256;
  size_t   lds_bytes = 0;         // dynamic LDS per WG; 0 = auto (65536 - static, 16 B aligned)
  bool     lds_user  = false;     // set when --lds is passed
  size_t   lds_static = 0;        // max static LDS of the two flag kernels (hipFuncGetAttributes)
  int      phases    = 16;
  int      replays   = 100;
  double   wait_ms   = 2000.0;    // device-side bound per flag wait
  double   host_ms   = 20000.0;   // host-side backstop per test
  double   busy_ms   = 500.0;     // busy-kernel duration for progress test
  int      busy_blocks_per_cu = 4;
  bool     busy_first = true;     // --order busy-first|flag-first
  double   gap_ms    = 20.0;      // host delay between the two launches
  int      inject_phase = -1;     // --inject-stall P:W: WG W stops publishing at phase P
  int      inject_wg    = -1;
};

// Sticky error record. Lives in pinned host memory (hipHostMallocCoherent),
// so it's readable by the host while kernels run.
struct StickyErr {
  int code;      // 0 = ok, 1 = flag-wait timeout
  int phase;     // stalled phase id
  int wg;        // waiting workgroup
  int peer;      // workgroup it was waiting on
  unsigned observed; // tag observed at timeout
  unsigned expected;
};

// wall_clock64 on gfx10 ticks at a fixed rate (queried from
// hipDeviceAttributeWallClockRate, in kHz).
__device__ inline unsigned long long now_ticks() { return wall_clock64(); }

__device__ inline bool sticky_set(StickyErr* e) {
  return __hip_atomic_load(&e->code, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) != 0;
}

// Bounded wait: spin until flags[peer] >= expected (acquire, agent scope).
// Returns false on timeout or if another WG already flagged an error.
__device__ bool bounded_wait(unsigned* flags, int peer, unsigned expected,
                             unsigned long long timeout_ticks,
                             StickyErr* err, int phase, int wg) {
  unsigned long long t0 = now_ticks();
  while (true) {
    unsigned v = __hip_atomic_load(&flags[peer], __ATOMIC_ACQUIRE,
                                   __HIP_MEMORY_SCOPE_AGENT);
    if (v >= expected) return true;
    if (sticky_set(err)) return false;
    if (now_ticks() - t0 > timeout_ticks) {
      int zero = 0;
      // first reporter wins; system scope so the host sees it
      if (__hip_atomic_compare_exchange_strong(&err->code, &zero, 1,
            __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)) {
        err->phase = phase; err->wg = wg; err->peer = peer;
        err->observed = v; err->expected = expected;
        __atomic_thread_fence(__ATOMIC_RELEASE); // flush fields
      }
      return false;
    }
    __builtin_amdgcn_s_sleep(2);
  }
}

// One phase of "work": touch the full dynamic LDS allocation so the
// allocation is real, then reduce into a checksum.
__device__ unsigned lds_work(unsigned* lds, size_t n_words, int phase) {
  for (size_t i = threadIdx.x; i < n_words; i += blockDim.x)
    lds[i] = (unsigned)(i * 2654435761u) ^ (unsigned)phase;
  __syncthreads();
  unsigned acc = 0;
  for (size_t i = threadIdx.x; i < n_words; i += blockDim.x) acc += lds[i];
  __syncthreads();
  return acc;
}

// Per-phase protocol (ring): WG w publishes tag=phase+1 on flags[w]
// (release), then waits for flags[(w+1)%G] >= phase+1 (acquire).
// Thread 0 does the flag ops; __syncthreads orders the rest of the WG.
// Flags are zeroed by a memset before each launch (captured into the graph),
// so tags restart at 1 every replay.
template <bool COOP>
__global__ void __launch_bounds__(256)
flag_kernel(unsigned* flags, unsigned* sink, StickyErr* err, int phases,
            unsigned lds_words, unsigned long long timeout_ticks,
            int inject_phase, int inject_wg,
            unsigned long long* trace, unsigned* done) {
  extern __shared__ unsigned lds[];
  __shared__ int bail;
  const int wg = blockIdx.x, G = gridDim.x;
  if (threadIdx.x == 0) { bail = 0; trace[2 * wg] = wall_clock64(); }
  __syncthreads();
  unsigned acc = 0;
  bool completed = true;

  for (int p = 0; p < phases; ++p) {
    acc += lds_work(lds, lds_words, p);
    // Fault injection: this WG silently stops publishing (no sticky record of
    // its own), so its predecessor's bounded wait must fire.
    if (wg == inject_wg && p == inject_phase) { completed = false; break; }
    if (threadIdx.x == 0) {
      __hip_atomic_store(&flags[wg], (unsigned)(p + 1), __ATOMIC_RELEASE,
                         __HIP_MEMORY_SCOPE_AGENT);
      if (!bounded_wait(flags, (wg + 1) % G, (unsigned)(p + 1),
                        timeout_ticks, err, p, wg))
        bail = 1;
    }
    __syncthreads();
    if (bail) { completed = false; break; }  // whole WG exits; no grid.sync after a bail (would hang)
    if constexpr (COOP) {
      // grid.sync is unbounded; every WG only reaches it if its own bounded
      // wait passed, but a WG that bailed skips it -> others may hang here.
      // Host timeout is the backstop; the sticky record still names the phase.
      cg::this_grid().sync();
    }
  }
  if (threadIdx.x == 0) {
    sink[wg] = acc;
    trace[2 * wg + 1] = wall_clock64();
    if (completed) atomicAdd(done, 1u);
  }
}

// Busy kernel: spins for `ticks` of wall clock, occupying CUs.
__global__ void busy_kernel(unsigned long long ticks, unsigned* sink) {
  unsigned long long t0 = wall_clock64();
  unsigned x = threadIdx.x;
  while (wall_clock64() - t0 < ticks) x = x * 1664525u + 1013904223u;
  if (x == 0xdeadbeef) sink[0] = x;
}

// ---------------------------------------------------------------- host
struct Ctx {
  Cfg c;
  unsigned *flags = nullptr, *sink = nullptr;
  StickyErr* err = nullptr;
  unsigned long long* trace = nullptr;  // per WG {start, end} wall_clock64
  unsigned* done = nullptr;             // WGs that finished every phase
  unsigned long long timeout_ticks = 0, busy_ticks = 0;
  long long wall_khz = 0;
};

static void reset_err(Ctx& x) { memset((void*)x.err, 0, sizeof(StickyErr)); }

static void print_err(const Ctx& x) {
  volatile StickyErr* e = x.err;
  if (e->code)
    printf("  STICKY ERROR: flag-wait timeout phase=%d wg=%d waiting_on=%d "
           "observed=%u expected=%u\n", e->phase, e->wg, e->peer,
           e->observed, e->expected);
}

// Poll stream completion with a host timeout. Never blocks forever.
// Returns 0 done, 1 timed out (kernel possibly hung), <0 HIP error.
static int host_wait(hipStream_t s, const Ctx& x, double ms) {
  auto t0 = std::chrono::steady_clock::now();
  while (true) {
    hipError_t q = hipStreamQuery(s);
    if (q == hipSuccess) return 0;
    if (q != hipErrorNotReady) {
      printf("  stream error: %s\n", hipGetErrorName(q)); return -1;
    }
    double el = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - t0).count();
    if (el > ms) {
      printf("  HOST TIMEOUT after %.0f ms (stream still busy)\n", el);
      print_err(x);
      return 1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

static hipError_t enqueue(Ctx& x, hipStream_t s, bool coop) {
  CHECK(hipMemsetAsync(x.flags, 0, x.c.grid * sizeof(unsigned), s));
  CHECK(hipMemsetAsync(x.done, 0, sizeof(unsigned), s));
  unsigned words = (unsigned)(x.c.lds_bytes / 4);
  int phases = x.c.phases, ip = x.c.inject_phase, iw = x.c.inject_wg;
  if (!coop) {
    (void)hipGetLastError();  // drop a stale error (e.g. an earlier refused coop launch)
    hipLaunchKernelGGL(flag_kernel<false>, dim3(x.c.grid), dim3(x.c.block),
                       x.c.lds_bytes, s, x.flags, x.sink, x.err, phases, words,
                       x.timeout_ticks, ip, iw, x.trace, x.done);
    CHECK(hipGetLastError());
  } else {
    void* args[] = {&x.flags, &x.sink, &x.err, &phases, &words, &x.timeout_ticks,
                    &ip, &iw, &x.trace, &x.done};
    CHECK(hipLaunchCooperativeKernel((const void*)flag_kernel<true>,
          dim3(x.c.grid), dim3(x.c.block), args, (unsigned)x.c.lds_bytes, s));
  }
  return hipSuccess;
}

static void set_lds_attr(size_t dyn) {
  // Allow >48 KiB dynamic LDS if the runtime enforces a default cap.
  (void)hipFuncSetAttribute((const void*)flag_kernel<false>,
        hipFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);
  (void)hipFuncSetAttribute((const void*)flag_kernel<true>,
        hipFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);
  (void)hipGetLastError();
}

static void test_residency(Ctx& x, int cus) {
  printf("\n[residency] grid=%d block=%d dyn_lds=%zu\n", x.c.grid, x.c.block,
         x.c.lds_bytes);
  const void* fns[2] = {(const void*)flag_kernel<false>, (const void*)flag_kernel<true>};
  const char* names[2] = {"plain", "coop"};
  for (int i = 0; i < 2; ++i) {
    int per_cu = -1;
    hipError_t e = hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &per_cu, fns[i], x.c.block, x.c.lds_bytes);
    hipFuncAttributes fa{};
    hipError_t e2 = hipFuncGetAttributes(&fa, fns[i]);
    printf("  %-5s occupancy: %d blocks/CU x %d CUs = %d resident (need %d) -> %s  [%s]\n",
           names[i], per_cu, cus, per_cu * cus, x.c.grid,
           (e == hipSuccess && per_cu * cus >= x.c.grid) ? "OK" : "INSUFFICIENT",
           hipGetErrorName(e));
    if (e2 == hipSuccess)
      printf("  %-5s funcattr: numRegs=%d static_lds=%zu maxThreads=%d localSize=%zu\n",
             names[i], fa.numRegs, fa.sharedSizeBytes, fa.maxThreadsPerBlock,
             fa.localSizeBytes);
  }
}

// Co-residency from the device timestamps: the peak number of WGs whose
// [start, end] intervals overlap, plus how many finished every phase.
static void report_trace(const Ctx& x, const char* n) {
  const int G = x.c.grid;
  unsigned long long* t = (unsigned long long*)malloc(sizeof(unsigned long long) * 2 * G);
  unsigned done = 0;
  if (hipMemcpy(t, x.trace, sizeof(unsigned long long) * 2 * G, hipMemcpyDeviceToHost) != hipSuccess ||
      hipMemcpy(&done, x.done, sizeof(unsigned), hipMemcpyDeviceToHost) != hipSuccess) {
    printf("  trace %-5s: copy failed\n", n); free(t); return;
  }
  unsigned long long t_min = ~0ull, t_max = 0, last_start = 0, first_end = ~0ull;
  int peak = 0;
  for (int i = 0; i < G; ++i) {
    t_min = t[2*i] < t_min ? t[2*i] : t_min; t_max = t[2*i+1] > t_max ? t[2*i+1] : t_max;
    last_start = t[2*i] > last_start ? t[2*i] : last_start;
    first_end = t[2*i+1] < first_end ? t[2*i+1] : first_end;
    int c = 0;
    for (int j = 0; j < G; ++j) c += (t[2*j] <= t[2*i] && t[2*i] <= t[2*j+1]);
    peak = c > peak ? c : peak;
  }
  double k = (double)x.wall_khz;  // ticks per ms
  printf("  trace %-5s: done=%u/%d peak_coresident=%d span=%.3f ms last_start-first_start=%.3f ms "
         "all_overlap=%s\n", n, done, G, peak, (t_max - t_min) / k, (last_start - t_min) / k,
         last_start <= first_end ? "yes" : "no");
  free(t);
}

// Returns false if a host timeout happened (GPU work still outstanding).
static bool test_direct(Ctx& x, hipStream_t s, bool coop) {
  const char* n = coop ? "coop" : "plain";
  reset_err(x);
  hipError_t e = enqueue(x, s, coop);
  if (e != hipSuccess) { printf("  direct %-5s: LAUNCH FAILED %s\n", n, hipGetErrorName(e)); return true; }
  auto t0 = std::chrono::steady_clock::now();
  int r = host_wait(s, x, x.c.host_ms);
  double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  printf("  direct %-5s: %s (%.2f ms)\n", n, r == 0 && !x.err->code ? "PASS" : "FAIL", ms);
  print_err(x);
  if (r == 0) report_trace(x, n);
  return r != 1;
}

// Returns false if a host timeout happened (GPU may be wedged; stop testing).
static bool test_graph(Ctx& x, hipStream_t s, bool coop) {
  const char* n = coop ? "coop" : "plain";
  hipGraph_t g = nullptr; hipGraphExec_t ge = nullptr;
  reset_err(x);
  hipError_t e = hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal);
  if (e != hipSuccess) { printf("  graph %-5s: BeginCapture FAILED %s\n", n, hipGetErrorName(e)); return true; }
  hipError_t le = enqueue(x, s, coop);
  e = hipStreamEndCapture(s, &g);
  if (le != hipSuccess || e != hipSuccess) {
    printf("  graph %-5s: CAPTURE FAILED launch=%s end=%s\n", n,
           hipGetErrorName(le), hipGetErrorName(e));
    if (g) hipGraphDestroy(g);
    (void)hipGetLastError();
    return true;
  }
  size_t nodes = 0; hipGraphGetNodes(g, nullptr, &nodes);
  e = hipGraphInstantiate(&ge, g, nullptr, nullptr, 0);
  if (e != hipSuccess) {
    printf("  graph %-5s: INSTANTIATE FAILED %s (nodes=%zu)\n", n, hipGetErrorName(e), nodes);
    hipGraphDestroy(g); return true;
  }
  auto t0 = std::chrono::steady_clock::now();
  int ok = 0; bool alive = true;
  for (int i = 0; i < x.c.replays; ++i) {
    e = hipGraphLaunch(ge, s);
    if (e != hipSuccess) { printf("  graph %-5s: replay %d LAUNCH FAILED %s\n", n, i, hipGetErrorName(e)); break; }
    int r = host_wait(s, x, x.c.host_ms);  // per-replay sync -> pinpoints failing replay
    if (r != 0) { printf("  graph %-5s: replay %d did not complete\n", n, i); alive = r != 1; break; }
    if (x.err->code) { printf("  graph %-5s: replay %d sticky error\n", n, i); print_err(x); break; }
    ++ok;
  }
  double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  printf("  graph %-5s: %s  %d/%d replays ok, nodes=%zu, %.3f ms/replay (incl. host sync)\n",
         n, ok == x.c.replays ? "PASS" : "FAIL", ok, x.c.replays, nodes,
         ok ? ms / ok : 0.0);
  if (alive) { hipGraphExecDestroy(ge); hipGraphDestroy(g); }
  return alive;
}

// Forward progress: start a long busy kernel on stream B that fills every CU,
// then launch the flag kernel on stream A. If the flag kernel's 72 WGs are not
// all co-resident, the bounded waits time out and the sticky record names the
// stalled phase/WG instead of deadlocking.
static double now_ms(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// Forward progress: flag kernel on stream A, busy kernel (fills every CU) on
// stream B. --order picks which is launched first; a host sleep of --gap-ms
// between the launches makes the order hold on the GPU. Bounded waits turn a
// non-resident WG into a STALL REPORTED with phase/WG instead of a deadlock.
static bool test_progress(Ctx& x, hipStream_t a, hipStream_t b, int cus, bool coop) {
  const char* n = coop ? "coop" : "plain";
  const char* order = x.c.busy_first ? "busy-first" : "flag-first";
  reset_err(x);
  int busy_grid = cus * x.c.busy_blocks_per_cu;
  auto T0 = std::chrono::steady_clock::now();
  double t_busy = -1, t_flag = -1;
  auto launch_busy = [&]() -> bool {
    t_busy = now_ms(T0);
    (void)hipGetLastError();  // drop a stale error from an earlier refused launch
    hipLaunchKernelGGL(busy_kernel, dim3(busy_grid), dim3(256), 0, b, x.busy_ticks, x.sink);
    return hipGetLastError() == hipSuccess;
  };
  auto launch_flag = [&]() -> hipError_t { t_flag = now_ms(T0); return enqueue(x, a, coop); };
  auto gap = [&]() { std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(x.c.gap_ms)); };

  hipError_t e = hipSuccess; bool bok = true;
  if (x.c.busy_first) { bok = launch_busy(); gap(); e = launch_flag(); }
  else                { e = launch_flag(); gap(); bok = launch_busy(); }
  printf("  progress %-5s order=%s gap=%.1fms launch t(ms): busy=%.3f flag=%.3f\n",
         n, order, x.c.gap_ms, t_busy, t_flag);
  if (!bok) printf("  progress %-5s: busy launch failed\n", n);
  if (e != hipSuccess) {
    printf("  progress %-5s: LAUNCH FAILED %s\n", n, hipGetErrorName(e));
    return host_wait(b, x, x.c.host_ms + x.c.busy_ms) != 1;
  }
  int r = host_wait(a, x, x.c.host_ms + x.c.busy_ms);
  double t_fdone = now_ms(T0);
  int rb = host_wait(b, x, x.c.host_ms + x.c.busy_ms);
  double t_bdone = now_ms(T0);
  const char* verdict = (r == 0 && !x.err->code) ? "PASS (no stall)"
                       : (x.err->code ? "STALL REPORTED (bounded wait fired)" : "FAIL");
  printf("  progress %-5s order=%s: %s; done t(ms): flag=%.1f busy=%.1f (busy %.0f ms on %d WGs)\n",
         n, order, verdict, t_fdone, t_bdone, x.c.busy_ms, busy_grid);
  print_err(x);
  return r != 1 && rb != 1;
}

static void usage() {
  printf("probe [--lds BYTES] [--grid N] [--phases N] [--replays N] [--wait-ms MS]\n"
         "      [--host-ms MS] [--busy-ms MS] [--busy-per-cu N]\n      [--order busy-first|flag-first] [--gap-ms MS] [--only residency|graph|progress]\n"
         "      [--inject-stall PHASE:WG]\n");
}

int main(int argc, char** argv) {
  Ctx x; std::string only;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto nx = [&]() { if (i + 1 >= argc) { usage(); exit(2); } return argv[++i]; };
    if (a == "--lds") { x.c.lds_bytes = strtoull(nx(), 0, 0); x.c.lds_user = true; }
    else if (a == "--grid") x.c.grid = atoi(nx());
    else if (a == "--phases") x.c.phases = atoi(nx());
    else if (a == "--replays") x.c.replays = atoi(nx());
    else if (a == "--wait-ms") x.c.wait_ms = atof(nx());
    else if (a == "--host-ms") x.c.host_ms = atof(nx());
    else if (a == "--busy-ms") x.c.busy_ms = atof(nx());
    else if (a == "--busy-per-cu") x.c.busy_blocks_per_cu = atoi(nx());
    else if (a == "--order") { std::string o = nx();
      if (o == "busy-first") x.c.busy_first = true;
      else if (o == "flag-first") x.c.busy_first = false;
      else { usage(); return 2; } }
    else if (a == "--gap-ms") x.c.gap_ms = atof(nx());
    else if (a == "--only") only = nx();
    else if (a == "--inject-stall") {
      if (sscanf(nx(), "%d:%d", &x.c.inject_phase, &x.c.inject_wg) != 2) { usage(); return 2; } }
    else { usage(); return 2; }
  }

  int dev = 0; hipDeviceProp_t p{};
  CHECK(hipGetDeviceProperties(&p, dev));
  int cus = p.multiProcessorCount, coop_ok = 0, khz = 0;
  hipDeviceGetAttribute(&coop_ok, hipDeviceAttributeCooperativeLaunch, dev);
  CHECK(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, dev));
  x.wall_khz = khz;
  x.timeout_ticks = (unsigned long long)(x.c.wait_ms * khz);
  x.busy_ticks = (unsigned long long)(x.c.busy_ms * khz);
  int rt = 0, drv = 0; hipRuntimeGetVersion(&rt); hipDriverGetVersion(&drv);

  printf("device: %s arch=%s CUs=%d lds/blk max=%zu coopLaunch=%d wallclock=%d kHz\n",
         p.name, p.gcnArchName, cus, p.sharedMemPerBlock, coop_ok, khz);
  int host_atomics = -1;
  (void)hipDeviceGetAttribute(&host_atomics, hipDeviceAttributeHostNativeAtomicSupported, dev);
  printf("hip runtime=%d driver=%d hostNativeAtomics=%d\n", rt, drv, host_atomics);
  // ---- LDS budget: static (__shared__ bail etc.) + dynamic <= 64 KiB on gfx1030.
  const size_t kLdsLimit = 65536;
  {
    const void* fns[2] = {(const void*)flag_kernel<false>, (const void*)flag_kernel<true>};
    for (int i = 0; i < 2; ++i) {
      hipFuncAttributes fa{};
      hipError_t e = hipFuncGetAttributes(&fa, fns[i]);
      if (e != hipSuccess) {
        printf("ERROR: hipFuncGetAttributes(flag_kernel<%s>) failed: %s\n",
               i ? "coop" : "plain", hipGetErrorName(e));
        return 2;
      }
      if (fa.sharedSizeBytes > x.c.lds_static) x.c.lds_static = fa.sharedSizeBytes;
    }
  }
  if (!x.c.lds_user) {
    x.c.lds_bytes = (kLdsLimit - x.c.lds_static) & ~(size_t)15;
  } else if (x.c.lds_static + x.c.lds_bytes > kLdsLimit) {
    printf("ERROR: --lds %zu + static LDS %zu = %zu bytes exceeds the gfx1030 "
           "per-workgroup limit of %zu. Use --lds <= %zu.\n",
           x.c.lds_bytes, x.c.lds_static, x.c.lds_static + x.c.lds_bytes,
           kLdsLimit, kLdsLimit - x.c.lds_static);
    return 2;
  }

  printf("cfg: grid=%d block=%d lds static=%zu dynamic=%zu total=%zu (limit %zu) phases=%d replays=%d wait=%.0fms host=%.0fms busy=%.0fms order=%s gap=%.1fms\n",
         x.c.grid, x.c.block, x.c.lds_static, x.c.lds_bytes,
         x.c.lds_static + x.c.lds_bytes, kLdsLimit, x.c.phases, x.c.replays,
         x.c.wait_ms, x.c.host_ms, x.c.busy_ms,
         x.c.busy_first ? "busy-first" : "flag-first", x.c.gap_ms);
  if (x.c.inject_wg >= 0)
    printf("inject-stall: wg=%d stops publishing at phase=%d (expect sticky phase=%d wg=%d peer=%d)\n",
           x.c.inject_wg, x.c.inject_phase, x.c.inject_phase,
           (x.c.inject_wg + x.c.grid - 1) % x.c.grid, x.c.inject_wg);

  CHECK(hipMalloc(&x.flags, x.c.grid * sizeof(unsigned)));
  CHECK(hipMalloc(&x.sink, x.c.grid * sizeof(unsigned)));
  CHECK(hipMalloc(&x.trace, 2 * x.c.grid * sizeof(unsigned long long)));
  CHECK(hipMalloc(&x.done, sizeof(unsigned)));
  CHECK(hipHostMalloc((void**)&x.err, sizeof(StickyErr), hipHostMallocCoherent));
  set_lds_attr(x.c.lds_bytes);

  hipStream_t a, b;
  CHECK(hipStreamCreateWithFlags(&a, hipStreamNonBlocking));
  CHECK(hipStreamCreateWithFlags(&b, hipStreamNonBlocking));

  bool alive = true;
  if (only.empty() || only == "residency") test_residency(x, cus);
  if (only.empty() || only == "graph") {
    printf("\n[direct launch sanity]\n");
    alive = test_direct(x, a, false);
    if (alive) alive = test_direct(x, a, true);
    if (alive) printf("\n[graph capture + replay]\n");
    if (alive) alive = test_graph(x, a, false);
    if (alive) alive = test_graph(x, a, true);
  }
  if (alive && (only.empty() || only == "progress")) {
    printf("\n[forward progress, busy kernel on second stream]\n");
    alive = test_progress(x, a, b, cus, false);
    if (alive) test_progress(x, a, b, cus, true);
  }
  if (!alive) {
    printf("\nGPU work still outstanding after host timeout; exiting without "
           "teardown (process exit lets the driver reclaim the queue).\n");
    fflush(stdout);
    _exit(3);
  }
  printf("\ndone\n");
  hipStreamDestroy(a); hipStreamDestroy(b);
  hipFree(x.flags); hipFree(x.sink); hipHostFree(x.err);
  return 0;
}
