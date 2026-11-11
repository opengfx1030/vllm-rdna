// T44 — torch ops around rdna_ar_oneshot: a push-based one-shot all-reduce for
// small tensor-parallel messages on gfx1030 (2..8 ranks, one process per rank).
//
// Ported from leapdragon/vllm-rdna2-qwen T44/T44b (Aron Hsiao).
//
//   rdna_ar_init(rank, world, device_ids, max_bytes, shm_name, gate_bytes) ->
//   uint8[64] IPC handle (max_bytes = staging per slot, gate_bytes = flat
//   gate, 0 = max_bytes) rdna_ar_connect(handles uint8[world,64]) (opens
//   every peer's staging) rdna_ar_can(t) -> bool, rdna_ar_all_reduce(t) ->
//   Tensor, rdna_ar_timed_out() -> bool, rdna_ar_timeout_info() -> int64 abort
//   code (T44b; 0 = none). rdna_ar_hier_init / rdna_ar_all_reduce_algo: the
//   hierarchical all-reduce, see rdna_allreduce_hier.cuh.
//
// Why not vLLM's own custom all-reduce: its barrier spins on coarse-grained
// device memory, which a peer's write never makes visible on this platform
// (T18); and it is gated to XGMI-connected gfx9. See rdna_allreduce.cuh for the
// protocol.
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <string>

#include "rdna_allreduce.cuh"
#include "rdna_allreduce_hier.cuh"

#define RDNA_AR_CHK(x)                                                     \
  do {                                                                     \
    hipError_t e_ = (x);                                                   \
    TORCH_CHECK(e_ == hipSuccess, "rdna_ar HIP: ", hipGetErrorString(e_)); \
  } while (0)

namespace {
struct RdnaArState {
  bool ready = false;
  int rank = -1, world = 0;
  int64_t max_bytes = 0;   // staging capacity per slot (the stride)
  int64_t gate_bytes = 0;  // flat gate: largest payload of the flat kernels
  void* stage =
      nullptr;  // ours: 2 * world * max_bytes uncached, then the flag page
  RdnaArPeers peers{};  // [j] = peer j's staging / flags (IPC), [rank] = ours
  bool opened[RDNA_AR_MAX_WORLD] = {};
  unsigned int* arrive = nullptr;
  int* seqbuf = nullptr;
  unsigned* timeout =
      nullptr;  // device: sticky abort claim (atomicCAS in the kernel)
  unsigned long long* report =
      nullptr;  // device pointer of the host-mapped abort record
  unsigned long long* report_host =
      nullptr;  // host side of the same 64 bytes (plain loads)
  long long wait_ms = RDNA_AR_WAIT_MS;  // VLLM_RDNA_AR_WAIT_MS (one-shot)
  long long ticks_per_ms = 100000;      // wall_clock64 rate (kHz = ticks/ms)
  int64_t fast_calls = 0;
  int blocks_cap =
      0;         // VLLM_RDNA_AR_BLOCKS: cap on blocks per launch (0 = auto)
  int pace = 0;  // VLLM_RDNA_AR_PACE: s_sleep units between strided pushes
  int64_t oneshot_max = RDNA_AR_ONESHOT_MAX;  // VLLM_RDNA_AR_ONESHOT_KB
  // VLLM_RDNA_AR_VEC=0: scalar all-reduce finisher and the old launch width
  // (A/B and rollback of rdna_ar_sum_vec).
  bool vec = true;
  // Hierarchical all-reduce (rdna_ar_hier_init): two islands of G ranks.
  bool hier = false;
  RdnaArTopo topo{};
  int algo_force = -1;             // -1: by size, else an RDNA_AR_ALGO_*
  int64_t hier_min_bytes = 20480;  // below this: the legacy flat one-shot
  int64_t hier_gate = 0;  // largest all-reduce payload while hier is on
  int hier_owners = 0;    // owners per island (0 = by size)
  int hier_blocks = 0;    // exact blocks of a hier launch (0 = auto)
};
// One instance per process group (vLLM builds several GroupCoordinators over
// the same ranks: world, TP, EP ...). Addressed by the handle rdna_ar_init
// returns.
constexpr int kMaxInstances = 8;
RdnaArState g_inst[kMaxInstances];
int g_ninst = 0;
RdnaArState& inst(int64_t h) {
  TORCH_CHECK(h >= 0 && h < g_ninst, "rdna_ar: bad handle ", h);
  return g_inst[h];
}
}  // namespace

bool rdna_ar_device(long long handle, RdnaArDev* out) {
  if (handle < 0 || handle >= g_ninst) return false;
  const RdnaArState& g = g_inst[handle];
  if (!g.ready || g.world < 2) return false;
  out->peers = g.peers;
  out->seqbuf = g.seqbuf;
  out->max_bytes = g.max_bytes;
  out->rank = g.rank;
  out->world = g.world;
  out->ready = 1;
  return true;
}

// Returns [handle (int64), 64 bytes of IPC handle] packed in a uint8 tensor of
// 72 bytes.
at::Tensor rdna_ar_init(int64_t rank, int64_t world,
                        const at::Tensor& device_ids, int64_t max_bytes,
                        const std::string& shm_name, int64_t gate_bytes) {
  TORCH_CHECK(g_ninst < kMaxInstances, "rdna_ar: too many instances");
  RdnaArState& g = g_inst[g_ninst];
  const int64_t handle_id = g_ninst;
  TORCH_CHECK(world >= 2 && world <= RDNA_AR_MAX_WORLD,
              "rdna_ar: world must be 2..8");
  TORCH_CHECK(rank >= 0 && rank < world, "rdna_ar: bad rank");
  // Fabric-friendliness knobs (2026-09-01). Fewer blocks = fewer concurrent
  // PCIe push streams (T44: 20 KB at 16/4/1 blocks = 33/36/76 us -- the first
  // step down is almost free); pace idles each wave between strided stores. See
  // rdna_allreduce.cuh step 1.
  if (const char* e = std::getenv("VLLM_RDNA_AR_BLOCKS"))
    g.blocks_cap = std::max(0, std::atoi(e));
  if (const char* e = std::getenv("VLLM_RDNA_AR_PACE"))
    g.pace = std::max(0, std::min(127, std::atoi(e)));
  if (const char* e = std::getenv("VLLM_RDNA_AR_VEC"))
    g.vec = std::atoi(e) != 0;
  if (const char* e = std::getenv("VLLM_RDNA_AR_ONESHOT_KB")) {
    const long long kb = atoll(e);
    if (kb > 0) g.oneshot_max = kb * 1024;
  }
  // VLLM_RDNA_AR_ALGO is no longer read here: one-shot is the only kernel.
  // The Python side logs when a two-shot setting is ignored.
  TORCH_CHECK(
      device_ids.numel() == world && device_ids.scalar_type() == at::kLong,
      "rdna_ar: device_ids must be int64[world]");
  g.rank = (int)rank;
  g.world = (int)world;
  g.max_bytes = max_bytes;
  g.gate_bytes = gate_bytes > 0 ? std::min(gate_bytes, max_bytes) : max_bytes;
  const int64_t* dev = device_ids.data_ptr<int64_t>();
  int mydev = -1;
  RDNA_AR_CHK(hipGetDevice(&mydev));
  for (int j = 0; j < world; j++) {
    if (j == rank || dev[j] == mydev) continue;
    hipError_t pe = hipDeviceEnablePeerAccess((int)dev[j], 0);
    // Consume the sticky error before TORCH_CHECK can throw so the RCCL
    // fallback does not inherit a failed peer-access probe.
    (void)hipGetLastError();
    TORCH_CHECK(pe == hipSuccess || pe == hipErrorPeerAccessAlreadyEnabled,
                "rdna_ar: peer access to device ", dev[j], ": ",
                hipGetErrorString(pe));
  }
  // staging slots followed by one 4 KB flag page: peers write flags[rank] = seq
  // into it through the same IPC mapping (uncached, so a P2P write is visible
  // to our polls)
  const size_t stage_bytes = 2ull * world * (size_t)max_bytes;
  const size_t alloc_bytes = stage_bytes + RDNA_AR_FLAG_PAGE;
  RDNA_AR_CHK(
      hipExtMallocWithFlags(&g.stage, alloc_bytes, hipDeviceMallocUncached));
  RDNA_AR_CHK(hipMemset(g.stage, 0, alloc_bytes));
  RDNA_AR_CHK(hipMalloc((void**)&g.arrive, 8));
  RDNA_AR_CHK(hipMemset(g.arrive, 0, 8));
  RDNA_AR_CHK(hipMalloc((void**)&g.seqbuf, 4));
  RDNA_AR_CHK(hipMemset(g.seqbuf, 0, 4));
  RDNA_AR_CHK(hipMalloc((void**)&g.timeout, 4));
  RDNA_AR_CHK(hipMemset(g.timeout, 0, 4));
  // T44b: the abort record lives in fine-grained host memory. The kernel writes
  // it with one posted store (no PCIe atomics, which not every board supports);
  // the Python side reads it with a plain load once per engine step, so a wedge
  // is caught without a device sync.
  RDNA_AR_CHK(hipHostMalloc((void**)&g.report_host, 64,
                            hipHostMallocMapped | hipHostMallocCoherent));
  memset(g.report_host, 0, 64);
  RDNA_AR_CHK(hipHostGetDevicePointer((void**)&g.report, g.report_host, 0));
  if (const char* e = std::getenv("VLLM_RDNA_AR_WAIT_MS")) {
    const long long v = atoll(e);
    if (v > 0) g.wait_ms = v;
  }
  {
    int dev = 0, khz = 0;
    if (hipGetDevice(&dev) == hipSuccess &&
        hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, dev) ==
            hipSuccess &&
        khz > 0)
      g.ticks_per_ms = khz;
  }
  // shm_name is kept in the signature; the host-coherent flag page it named is
  // no longer used -- flags live in device memory, see rdna_allreduce.cuh.
  (void)shm_name;
  for (int j = 0; j < RDNA_AR_MAX_WORLD; j++) {
    g.peers.stage[j] = nullptr;
    g.peers.flags[j] = nullptr;
  }
  g.peers.stage[rank] = g.stage;
  g.peers.flags[rank] =
      reinterpret_cast<int*>(static_cast<uint8_t*>(g.stage) + stage_bytes);
  hipIpcMemHandle_t h;
  RDNA_AR_CHK(hipIpcGetMemHandle(&h, g.stage));
  g_ninst++;
  auto out = at::empty({(int64_t)(8 + sizeof(h))},
                       at::TensorOptions().dtype(at::kByte));
  std::memcpy(out.mutable_data_ptr(), &handle_id, 8);
  std::memcpy(static_cast<uint8_t*>(out.mutable_data_ptr()) + 8, &h, sizeof(h));
  return out;
}

void rdna_ar_connect(int64_t handle, const at::Tensor& handles) {
  RdnaArState& g = inst(handle);
  TORCH_CHECK(g.stage != nullptr && !g.ready,
              "rdna_ar: init first / already connected");
  TORCH_CHECK(handles.dim() == 2 && handles.size(0) == g.world &&
                  handles.size(1) == (int64_t)sizeof(hipIpcMemHandle_t) &&
                  handles.scalar_type() == at::kByte && handles.is_contiguous(),
              "rdna_ar: handles must be contiguous uint8[world, 64]");
  const uint8_t* p = handles.data_ptr<uint8_t>();
  for (int j = 0; j < g.world; j++) {
    if (j == g.rank) continue;
    hipIpcMemHandle_t h;
    std::memcpy(&h, p + (size_t)j * sizeof(h), sizeof(h));
    RDNA_AR_CHK(hipIpcOpenMemHandle(&g.peers.stage[j], h,
                                    hipIpcMemLazyEnablePeerAccess));
    g.peers.flags[j] =
        reinterpret_cast<int*>(static_cast<uint8_t*>(g.peers.stage[j]) +
                               2ull * g.world * (size_t)g.max_bytes);
    g.opened[j] = true;
  }
  g.ready = true;
}

namespace {
template <typename T>
void rdna_ar_launch(RdnaArState& g, const T* inp, T* out, int n, int nblocks,
                    cudaStream_t stream) {
  const long long max_elems = g.max_bytes / (long long)sizeof(T);
  rdna_ar_oneshot<T><<<nblocks, 256, 0, stream>>>(
      inp, out, g.peers, g.arrive, g.seqbuf, g.timeout, g.report, g.rank,
      g.world, n, max_elems, nblocks, g.pace, g.wait_ms * g.ticks_per_ms,
      g.ticks_per_ms, g.vec ? 1 : 0);
}

bool rdna_ar_flat(const at::Tensor& t) {
  if (t.is_contiguous()) return true;
  // Same span rule as custom all-reduce: a dense storage prefix is reduced
  // in storage order, which matches across ranks that share the layout.
  if (!t.is_non_overlapping_and_dense() || t.numel() <= 0) return false;
  const int64_t span = static_cast<int64_t>(t.storage().nbytes()) -
                       t.storage_offset() * t.element_size();
  return span == t.numel() * static_cast<int64_t>(t.element_size());
}

// Launch width by payload: 20 KB 1/4/16 blocks = 76/36/33 us; 5 KB 1/4/8 =
// 27/15/18 us; 80 KB 4/16/32 = 115/91/85 us (4x V620, T44). More blocks = more
// concurrent PCIe pushes.
int rdna_ar_blocks(const RdnaArState& g, int64_t staged_bytes) {
  int nblocks = staged_bytes <= 8192 ? 4 : (staged_bytes <= 32768 ? 16 : 32);
  if (g.blocks_cap > 0 && nblocks > g.blocks_cap) nblocks = g.blocks_cap;
  return nblocks;
}

// All-reduce launch width with the vector finisher (rdna_ar_sum_vec). The
// finisher needs far fewer threads than the old scalar one, and every extra
// block is one more arrival for the grid barrier. 4x V620, graph replay behind
// a 200 us busy kernel, median of 3, us (blocks: us):
//    15 KB  4: 18.1/19.5  6: 18.7/19.1  8: 20.1/20.4  16: 21.1  32: 20.1
//    30 KB  4: 30.2/29.5  6: 25.4/27.3  8: 27.6/29.1  16: 29.4  32: 32.0
//    60 KB  8: 42.4  16: 39.4  24: 39.9/40.4  28: 37.4  32: 38.8/39.5  48: 43.5
//   120 KB 16: 46.8  24: 46.0  32: 43.6/44.3/45.6  36: 42.1  48: 50.9  64: 50.5
// (noise about 1.5 us). Gather and argmax keep rdna_ar_blocks; so does
// VLLM_RDNA_AR_VEC=0.
int rdna_ar_reduce_blocks(const RdnaArState& g, int64_t staged_bytes) {
  if (!g.vec) return rdna_ar_blocks(g, staged_bytes);
  int nblocks = staged_bytes <= 8192 ? 4 : (staged_bytes <= 32768 ? 6 : 32);
  if (g.blocks_cap > 0 && nblocks > g.blocks_cap) nblocks = g.blocks_cap;
  return nblocks;
}
}  // namespace

// Largest all-reduce payload of the group: the flat gate, or the hier gate
// while the hierarchical mode is on.
static int64_t rdna_ar_gate(const RdnaArState& g) {
  return g.hier ? g.hier_gate : std::min(g.gate_bytes, g.oneshot_max);
}

bool rdna_ar_can(int64_t handle, const at::Tensor& t) {
  const RdnaArState& g = inst(handle);
  const auto dt = t.scalar_type();
  return g.ready && t.is_cuda() && rdna_ar_flat(t) &&
         (dt == at::kHalf || dt == at::kFloat || dt == at::kBFloat16) &&
         t.numel() * t.element_size() <= rdna_ar_gate(g) && t.numel() > 0;
}

// The algorithm of an all-reduce of `bytes`: a function of the size and the
// group's configuration only, so every rank picks the same one. Below
// hier_min_bytes the legacy flat one-shot (rank-order sum) runs unchanged; from
// there the 3-phase algorithm (island-grouped sum).
static int rdna_ar_pick_algo(const RdnaArState& g, int64_t bytes) {
  if (!g.hier) return RDNA_AR_ALGO_FLAT;
  if (g.algo_force >= 0) return g.algo_force;
  return bytes < g.hier_min_bytes ? RDNA_AR_ALGO_FLAT : RDNA_AR_ALGO_HIER;
}

// Blocks of a hier launch: each block is one independent pipeline over its
// chunk, so the width only has to keep the pushes in flight. 4 + 4 V620, graph
// replay, median of 3, us (docs/rdna2/rdna-ar-hier-2026-11-06.md):
//   x4 64 KiB 2/4/8/16/32 blocks: 42.0/39.6/38.9/37.2/37.9
//   x4 128 KiB 16/32: 39.7/35.8; x4 512 KiB 16/32: 80.3/77.9
static int rdna_ar_hier_blocks(const RdnaArState& g, int64_t bytes) {
  int nblocks = bytes <= 16384 ? 4 : (bytes <= 98304 ? 16 : 32);
  if (g.hier_blocks > 0) nblocks = g.hier_blocks;
  nblocks = std::min(nblocks, RDNA_AR_HFLAG_BLOCKS);
  return std::max(1, (int)std::min<int64_t>(nblocks, (bytes + 15) / 16));
}

// Owners per island of a hier launch. Below 56 KiB two ranks per island carry
// the cross-root flows (48 KiB, x2 vs x4: 36.0 vs 38.2 us), above that all of
// them (128 KiB: 35.8 vs 48.3 us).
static int rdna_ar_hier_owners(const RdnaArState& g, int64_t bytes) {
  const int x =
      g.hier_owners > 0 ? g.hier_owners : (bytes < 57344 ? 2 : g.topo.G);
  return std::min(x, g.topo.G);
}

template <typename T>
static void rdna_ar_launch_algo(RdnaArState& g, const T* inp, T* out, int n,
                                int algo, int nblocks, cudaStream_t stream) {
  const long long max_elems = g.max_bytes / (long long)sizeof(T);
  const long long wait = g.wait_ms * g.ticks_per_ms;
  RdnaArTopo tp = g.topo;
  tp.X = rdna_ar_hier_owners(g, (int64_t)n * (int64_t)sizeof(T));
  if (algo == RDNA_AR_ALGO_HIER) {
#define RDNA_AR_HIER_LAUNCH(GG)                                           \
  rdna_ar_hier<T, GG><<<nblocks, 256, 0, stream>>>(                       \
      inp, out, g.peers, g.arrive, g.seqbuf, g.timeout, g.report, g.rank, \
      g.world, n, (long long)g.max_bytes, nblocks, wait, g.ticks_per_ms, tp)
    switch (tp.G) {
      case 2:
        RDNA_AR_HIER_LAUNCH(2);
        break;
      case 3:
        RDNA_AR_HIER_LAUNCH(3);
        break;
      default:
        RDNA_AR_HIER_LAUNCH(4);
        break;
    }
#undef RDNA_AR_HIER_LAUNCH
  } else if (algo == RDNA_AR_ALGO_FLAT_ISL) {
    rdna_ar_oneshot_isl<T><<<nblocks, 256, 0, stream>>>(
        inp, out, g.peers, g.arrive, g.seqbuf, g.timeout, g.report, g.rank,
        g.world, n, max_elems, nblocks, g.pace, wait, g.ticks_per_ms, tp);
  } else {
    rdna_ar_launch<T>(g, inp, out, n, nblocks, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// algo < 0: by size. blocks <= 0: by size (a forced algo with a forced width
// is for the benchmarks and tests).
at::Tensor rdna_ar_all_reduce_algo(int64_t handle, const at::Tensor& in,
                                   int64_t algo_req, int64_t blocks_req) {
  TORCH_CHECK(rdna_ar_can(handle, in), "rdna_ar: tensor not eligible");
  RdnaArState& g = inst(handle);
  const at::cuda::OptionalCUDAGuard guard(in.device());
  // A fresh output per call: callers keep the result past the next
  // collective (e.g. the embedding all-reduce output becomes the decoder
  // residual), so a shared persistent buffer gets clobbered by the next
  // all-reduce. The kernel writes every element (dense layout, storage order).
  auto out = at::empty_like(in);
  const int n = (int)in.numel();
  const int64_t bytes = (int64_t)n * in.element_size();
  int algo = algo_req >= 0 ? (int)algo_req : rdna_ar_pick_algo(g, bytes);
  TORCH_CHECK(algo == RDNA_AR_ALGO_FLAT || g.hier,
              "rdna_ar: hier algorithm without rdna_ar_hier_init");
  // Measured on 4x V620: see rdna_ar_reduce_blocks.
  int nblocks = algo == RDNA_AR_ALGO_HIER ? rdna_ar_hier_blocks(g, bytes)
                                          : rdna_ar_reduce_blocks(g, bytes);
  if (blocks_req > 0)
    nblocks = (int)std::min<int64_t>(
        blocks_req, algo == RDNA_AR_ALGO_HIER ? RDNA_AR_HFLAG_BLOCKS : 64);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (in.scalar_type() == at::kHalf) {
    rdna_ar_launch_algo<__half>(
        g, reinterpret_cast<const __half*>(in.const_data_ptr()),
        reinterpret_cast<__half*>(out.mutable_data_ptr()), n, algo, nblocks,
        stream);
  } else if (in.scalar_type() == at::kBFloat16) {
    rdna_ar_launch_algo<__hip_bfloat16>(
        g, reinterpret_cast<const __hip_bfloat16*>(in.const_data_ptr()),
        reinterpret_cast<__hip_bfloat16*>(out.mutable_data_ptr()), n, algo,
        nblocks, stream);
  } else {
    rdna_ar_launch_algo<float>(
        g, reinterpret_cast<const float*>(in.const_data_ptr()),
        reinterpret_cast<float*>(out.mutable_data_ptr()), n, algo, nblocks,
        stream);
  }
  g.fast_calls++;
  return out;
}

at::Tensor rdna_ar_all_reduce(int64_t handle, const at::Tensor& in) {
  return rdna_ar_all_reduce_algo(handle, in, -1, 0);
}

// Two islands of G ranks (island_of_rank[r] in {0, 1}, equal sizes, G = 2..4):
// enables the hierarchical all-reduce for this group. `owners` (0 = by size)
// is the number of ranks per island that exchange across the root complex,
// `min_kb` the size from which the 3-phase algorithm replaces the flat
// exchange, `algo` forces one algorithm (-1 = by size), `blocks` forces the
// hier launch width (0 = by size), `gate_kb` is the largest all-reduce while
// the mode is on (0 = the flat gate; at most the staging capacity). Every rank
// of the group must call it with the same arguments between two collectives
// (the arguments pick the algorithm). An empty island_of_rank turns the mode
// off again.
void rdna_ar_hier_init(int64_t handle, const at::Tensor& island_of_rank,
                       int64_t owners, int64_t min_kb, int64_t algo,
                       int64_t blocks, int64_t gate_kb) {
  RdnaArState& g = inst(handle);
  TORCH_CHECK(g.ready, "rdna_ar: hier init before connect");
  if (island_of_rank.numel() == 0) {
    g.hier = false;
    g.algo_force = -1;
    return;
  }
  TORCH_CHECK(island_of_rank.device().is_cpu() &&
                  island_of_rank.scalar_type() == at::kLong &&
                  island_of_rank.numel() == g.world,
              "rdna_ar: island_of_rank must be a CPU int64[world]");
  TORCH_CHECK(algo >= -1 && algo <= RDNA_AR_ALGO_HIER,
              "rdna_ar: bad algorithm ", algo);
  const int64_t* kof = island_of_rank.data_ptr<int64_t>();
  int cnt[2] = {0, 0};
  for (int r = 0; r < g.world; r++) {
    TORCH_CHECK(kof[r] == 0 || kof[r] == 1, "rdna_ar: island ids are 0/1");
    cnt[kof[r]]++;
  }
  TORCH_CHECK(cnt[0] == cnt[1] && cnt[0] >= 2 && cnt[0] <= RDNA_AR_MAX_G,
              "rdna_ar: hier needs two islands of 2..", RDNA_AR_MAX_G,
              " ranks, got ", cnt[0], " + ", cnt[1]);
  // Island 0 is the one that holds rank 0, so the labels do not matter.
  const int flip = kof[0] == 1 ? 1 : 0;
  RdnaArTopo tp{};
  tp.G = cnt[0];
  tp.X = tp.G;  // per launch: rdna_ar_hier_owners
  int fill[2] = {0, 0};
  for (int r = 0; r < g.world; r++) {
    const int k = (int)kof[r] ^ flip;
    tp.kof[r] = k;
    tp.isl[k][fill[k]++] = r;  // ascending: local index = position
  }
  tp.k = tp.kof[g.rank];
  tp.l = 0;
  while (tp.isl[tp.k][tp.l] != g.rank) tp.l++;
  g.topo = tp;
  g.algo_force = (int)algo;
  g.hier_min_bytes = std::max<int64_t>(min_kb, 0) * 1024;
  g.hier_owners = (int)std::max<int64_t>(owners, 0);
  g.hier_blocks = (int)std::max<int64_t>(blocks, 0);
  g.hier_gate = gate_kb > 0 ? std::min(gate_kb * 1024, g.max_bytes)
                            : std::min(g.gate_bytes, g.oneshot_max);
  g.hier = true;
}

namespace {
bool rdna_ar_float_dtype(at::ScalarType dt) {
  return dt == at::kHalf || dt == at::kFloat || dt == at::kBFloat16;
}

// Bytes one rank pushes per peer: the flat gate (one-shot only), whatever the
// hierarchical mode does to the all-reduce gate.
bool rdna_ar_fits(const RdnaArState& g, int64_t staged_bytes) {
  return staged_bytes <= std::min(g.gate_bytes, g.oneshot_max);
}

template <typename T>
void rdna_ag_launch(RdnaArState& g, const T* inp, T* out, int n, int cols,
                    int nblocks, cudaStream_t stream) {
  const long long max_elems = g.max_bytes / (long long)sizeof(T);
  rdna_ag_oneshot<T><<<nblocks, 256, 0, stream>>>(
      inp, out, g.peers, g.arrive, g.seqbuf, g.timeout, g.report, g.rank,
      g.world, n, cols, max_elems, nblocks, g.pace, g.wait_ms * g.ticks_per_ms,
      g.ticks_per_ms);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename V>
void rdna_argmax_launch(RdnaArState& g, const V* vals, const int64_t* idx,
                        int64_t* out, int n, int nblocks, cudaStream_t stream) {
  const long long max_elems = g.max_bytes / (long long)sizeof(RdnaArPair);
  rdna_ar_argmax_oneshot<V><<<nblocks, 256, 0, stream>>>(
      vals, idx, out, g.peers, g.arrive, g.seqbuf, g.timeout, g.report, g.rank,
      g.world, n, max_elems, nblocks, g.pace, g.wait_ms * g.ticks_per_ms,
      g.ticks_per_ms);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

// Last-dim all-gather of a small contiguous tensor: [..., c] -> [..., world*c],
// rank j's block at [j*c, (j+1)*c). Bit-equal to all_gather_into_tensor (a
// copy, no arithmetic).
bool rdna_ar_can_all_gather(int64_t handle, const at::Tensor& t) {
  const RdnaArState& g = inst(handle);
  return g.ready && t.is_cuda() && t.is_contiguous() && t.dim() >= 1 &&
         t.numel() > 0 && rdna_ar_float_dtype(t.scalar_type()) &&
         rdna_ar_fits(g, t.numel() * t.element_size());
}

at::Tensor rdna_ar_all_gather(int64_t handle, const at::Tensor& in) {
  TORCH_CHECK(rdna_ar_can_all_gather(handle, in),
              "rdna_ar: tensor not eligible for all_gather");
  RdnaArState& g = inst(handle);
  const at::cuda::OptionalCUDAGuard guard(in.device());
  std::vector<int64_t> sizes(in.sizes().begin(), in.sizes().end());
  const int cols = (int)sizes.back();
  sizes.back() *= g.world;
  auto out = at::empty(sizes, in.options());
  const int n = (int)in.numel();
  const int nblocks = rdna_ar_blocks(g, (int64_t)n * in.element_size());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (in.scalar_type() == at::kHalf) {
    rdna_ag_launch<__half>(g,
                           reinterpret_cast<const __half*>(in.const_data_ptr()),
                           reinterpret_cast<__half*>(out.mutable_data_ptr()), n,
                           cols, nblocks, stream);
  } else if (in.scalar_type() == at::kBFloat16) {
    rdna_ag_launch<__hip_bfloat16>(
        g, reinterpret_cast<const __hip_bfloat16*>(in.const_data_ptr()),
        reinterpret_cast<__hip_bfloat16*>(out.mutable_data_ptr()), n, cols,
        nblocks, stream);
  } else {
    rdna_ag_launch<float>(g,
                          reinterpret_cast<const float*>(in.const_data_ptr()),
                          reinterpret_cast<float*>(out.mutable_data_ptr()), n,
                          cols, nblocks, stream);
  }
  g.fast_calls++;
  return out;
}

// Vocab-parallel argmax: per row, the global index of the largest value across
// ranks (ties to the lowest rank). One launch replaces the (value, index) pair
// all_gather plus the argmax/gather/cast kernels after it.
bool rdna_ar_can_argmax(int64_t handle, const at::Tensor& vals,
                        const at::Tensor& idx) {
  const RdnaArState& g = inst(handle);
  return g.ready && vals.is_cuda() && idx.is_cuda() &&
         vals.device() == idx.device() && vals.dim() == 1 && idx.dim() == 1 &&
         vals.numel() > 0 && vals.numel() == idx.numel() &&
         vals.is_contiguous() && idx.is_contiguous() &&
         idx.scalar_type() == at::kLong &&
         rdna_ar_float_dtype(vals.scalar_type()) &&
         rdna_ar_fits(g, vals.numel() * (int64_t)sizeof(RdnaArPair));
}

at::Tensor rdna_ar_argmax(int64_t handle, const at::Tensor& vals,
                          const at::Tensor& idx) {
  TORCH_CHECK(rdna_ar_can_argmax(handle, vals, idx),
              "rdna_ar: tensors not eligible for argmax");
  RdnaArState& g = inst(handle);
  const at::cuda::OptionalCUDAGuard guard(vals.device());
  auto out = at::empty({vals.numel()}, idx.options());
  const int n = (int)vals.numel();
  const int nblocks =
      rdna_ar_blocks(g, (int64_t)n * (int64_t)sizeof(RdnaArPair));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t* ip = idx.const_data_ptr<int64_t>();
  int64_t* op = out.mutable_data_ptr<int64_t>();
  if (vals.scalar_type() == at::kHalf) {
    rdna_argmax_launch<__half>(
        g, reinterpret_cast<const __half*>(vals.const_data_ptr()), ip, op, n,
        nblocks, stream);
  } else if (vals.scalar_type() == at::kBFloat16) {
    rdna_argmax_launch<__hip_bfloat16>(
        g, reinterpret_cast<const __hip_bfloat16*>(vals.const_data_ptr()), ip,
        op, n, nblocks, stream);
  } else {
    rdna_argmax_launch<float>(
        g, reinterpret_cast<const float*>(vals.const_data_ptr()), ip, op, n,
        nblocks, stream);
  }
  g.fast_calls++;
  return out;
}

// T44b: both read the host-mapped record -- no device copy, no stream sync,
// safe to call every step from the model thread. The code's layout is
// documented in rdna_allreduce.cuh.
static inline unsigned long long rdna_ar_report(const RdnaArState& g) {
  if (!g.ready || g.report_host == nullptr) return 0ull;
  return *reinterpret_cast<volatile unsigned long long*>(g.report_host);
}

bool rdna_ar_timed_out(int64_t handle) {
  return rdna_ar_report(inst(handle)) != 0ull;
}

int64_t rdna_ar_timeout_info(int64_t handle) {
  return (int64_t)rdna_ar_report(inst(handle));
}

// Last one-shot wait longer than RDNA_AR_SLOW_MS that completed (0 = none);
// same layout as the abort record.
int64_t rdna_ar_slow_info(int64_t handle) {
  const RdnaArState& g = inst(handle);
  if (!g.ready || g.report_host == nullptr) return 0;
  return (int64_t)reinterpret_cast<volatile unsigned long long*>(
      g.report_host)[1];
}

int64_t rdna_ar_fast_calls(int64_t handle) { return inst(handle).fast_calls; }
