// hx torch ops (docs/rdna2/hetero-hx-2026-11-07.md): the hub dispatch/combine
// and expert-server begin/wait/push kernels of hx_core.cuh as torch ops, plus
// the host-side setup the Python layer needs (IPC allocations, clock
// calibration, ctx image, status/abort side channel).
//
// Compute ops (CUDA key, enqueue one kernel on the current stream, nothing
// else, so they capture into graphs; all state lives in the device ctx):
//   hx_begin_step(ctx, T, bucket)            hub: epoch++, header to servers
//   hx_dispatch(x, topk_ids, topk_w, hub_ids_out, ctx, layer)
//   hx_combine(y_hub, out, ctx, layer)       out may alias y_hub
//   hx_srv_begin(ctx)  hx_srv_wait(ctx, layer, T)  hx_srv_push(y, ctx, layer,
//   T)
// Setup/host ops (CPU key; never inside a capture):
//   hx_alloc / hx_free / hx_open / hx_close  exported VRAM and IPC handles
//   hx_calibrate                             wall-clock ticks per ms
//   hx_ctx_new / hx_ctx_set / hx_ctx_get / hx_ctx_register / hx_ctx_unregister
//   hx_status / hx_host_abort / hx_srv_post_desc / hx_view
//
// The ctx is a torch int64[HX_CTX_WORDS] CUDA tensor (declared Tensor! so the
// compiler keeps the ops ordered); the host keeps the launch geometry in a
// registry keyed by its data pointer.
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "hx_core.cuh"

#define HX_CHK(x)                                                    \
  do {                                                               \
    hipError_t e_ = (x);                                             \
    TORCH_CHECK(e_ == hipSuccess, "hx HIP: ", hipGetErrorString(e_), \
                " in " #x);                                          \
  } while (0)

namespace {

// Host-side companion of a device ctx.
struct HxHost {
  int role = 0;
  int nsrv = 0, topk = 0, hidden = 0, rows_cap = 0, nslots = 0, nlayers = 0;
  uint8_t* own_base = nullptr;
  // 0 = by size
  int blocks_disp = 0, blocks_comb = 0, blocks_push = 0, blocks_wait = 0;
  hipStream_t side = nullptr;  // status read-back and abort poke
  uint8_t* stage = nullptr;    // pinned: [0,64) status, [64,72) poke, [128, ..)
  unsigned desc_next = 0;      //   descriptor ring of 64 x 16 B
  std::mutex side_mu;
};

constexpr int kDescRing = 64;
constexpr size_t kStageBytes = 128 + 16 * kDescRing;

std::mutex g_mu;
// node-based: references stay valid while other contexts are added
std::unordered_map<const void*, HxHost> g_hosts;

int env_int(const char* name, int dflt) {
  const char* e = std::getenv(name);
  return e != nullptr ? std::atoi(e) : dflt;
}

// role 0: either role.
HxHost& host_checked(const at::Tensor& ctx, int role) {
  TORCH_CHECK(ctx.is_cuda() && ctx.scalar_type() == at::kLong &&
                  ctx.numel() == HX_CTX_WORDS && ctx.is_contiguous(),
              "hx: ctx must be a contiguous int64[", HX_CTX_WORDS,
              "] CUDA tensor");
  std::lock_guard<std::mutex> lk(g_mu);
  auto it = g_hosts.find(ctx.data_ptr());
  TORCH_CHECK(it != g_hosts.end(), "hx: ctx is not registered");
  TORCH_CHECK(role == 0 || it->second.role == role, "hx: this op needs a ",
              role == 1 ? "hub" : "server", " ctx");
  return it->second;
}

HxCtx* dev_ctx(const at::Tensor& ctx) {
  return reinterpret_cast<HxCtx*>(ctx.data_ptr());
}

hipStream_t cur_stream() {
  const cudaStream_t s = at::cuda::getCurrentCUDAStream();
  return reinterpret_cast<hipStream_t>(s);
}

void check_cpu_image(const at::Tensor& t, const char* who) {
  TORCH_CHECK(t.device().is_cpu() && t.scalar_type() == at::kLong &&
                  t.numel() == HX_CTX_WORDS && t.is_contiguous(),
              who, ": ctx image must be a contiguous CPU int64[", HX_CTX_WORDS,
              "]");
}

struct Field {
  size_t off;
  size_t size;
};

const std::unordered_map<std::string, Field>& fields() {
  static const std::unordered_map<std::string, Field> f = [] {
    std::unordered_map<std::string, Field> m;
#define HXF(n) m[#n] = {offsetof(HxCtx, n), sizeof(((HxCtx*)nullptr)->n)}
    HXF(role);
    HXF(version);
    HXF(plan_hash);
    HXF(index);
    HXF(nsrv);
    HXF(rows_cap);
    HXF(topk);
    HXF(hidden);
    HXF(nslots);
    HXF(num_experts);
    HXF(nlayers);
    HXF(flush_mode);
    HXF(wait_ticks);
    HXF(begin_wait_ticks);
    HXF(ticks_per_ms);
    HXF(slot_stride);
    HXF(off_ids);
    HXF(off_w);
    HXF(y_stride);
    HXF(last_epoch);
    HXF(own_base);
    HXF(hub_slot);
    HXF(cold_srv);
    HXF(n_cold);
    HXF(owned);
    HXF(ktab);
    HXF(ids32);
    HXF(x_ws);
    HXF(w_ws);
    HXF(layer_stats);
#undef HXF
    for (int i = 0; i < HX_MAX_SERVERS; i++)
      m["peer_base" + std::to_string(i)] = {
          offsetof(HxCtx, peer_base) + (size_t)i * sizeof(uint8_t*),
          sizeof(uint8_t*)};
    return m;
  }();
  return f;
}

__global__ void hx_k_clock(long long* out) {
  if (threadIdx.x == 0) out[0] = wall_clock64();
}

}  // namespace

// ---------------------------------------------------------------------------
// Allocation and IPC
// ---------------------------------------------------------------------------
// kind: 0 coarse hipMalloc, 1 uncached (hipDeviceMallocUncached), 2
// fine-grained. Receive buffers must be uncached: a peer store over PCIe never
// invalidates the owner's L2, and rdna_allreduce.cu allocates its stage buffers
// the same way; the other kinds exist to measure alternatives (T1) and never
// substitute silently. Returns uint8[88]: u64 ptr | u64 nbytes | u64 kind |
// hipIpcMemHandle_t (64 B). The memory is zeroed.
static at::Tensor hx_alloc(int64_t nbytes, int64_t kind) {
  TORCH_CHECK(nbytes > 0 && nbytes % HX_PAGE == 0,
              "hx_alloc: nbytes must be a positive multiple of ", HX_PAGE);
  void* p = nullptr;
  if (kind == 1) {
    const hipError_t e =
        hipExtMallocWithFlags(&p, (size_t)nbytes, hipDeviceMallocUncached);
    TORCH_CHECK(e == hipSuccess,
                "hx_alloc: uncached allocation failed: ", hipGetErrorString(e),
                " (VLLM_HX_ALLOC=coarse|finegrained selects another kind)");
  } else if (kind == 2) {
    const hipError_t e =
        hipExtMallocWithFlags(&p, (size_t)nbytes, hipDeviceMallocFinegrained);
    TORCH_CHECK(e == hipSuccess, "hx_alloc: fine-grained allocation failed: ",
                hipGetErrorString(e));
  } else {
    TORCH_CHECK(kind == 0, "hx_alloc: bad kind ", kind);
    HX_CHK(hipMalloc(&p, (size_t)nbytes));
  }
  HX_CHK(hipMemset(p, 0, (size_t)nbytes));
  hipIpcMemHandle_t h;
  const hipError_t e = hipIpcGetMemHandle(&h, p);
  if (e != hipSuccess) {
    (void)hipFree(p);
    TORCH_CHECK(false, "hx_alloc: hipIpcGetMemHandle: ", hipGetErrorString(e));
  }
  auto out = at::empty({(int64_t)(24 + sizeof(h))},
                       at::TensorOptions().dtype(at::kByte));
  uint8_t* o = out.mutable_data_ptr<uint8_t>();
  const uint64_t u_ptr = reinterpret_cast<uint64_t>(p);
  const uint64_t u_n = (uint64_t)nbytes;
  const uint64_t u_k = (uint64_t)kind;
  std::memcpy(o, &u_ptr, 8);
  std::memcpy(o + 8, &u_n, 8);
  std::memcpy(o + 16, &u_k, 8);
  std::memcpy(o + 24, &h, sizeof(h));
  return out;
}

static void hx_free(int64_t ptr) {
  HX_CHK(hipFree(reinterpret_cast<void*>(ptr)));
}

// Maps a peer's exported allocation (hipIpcMemLazyEnablePeerAccess); handle is
// the 64 B hipIpcMemHandle_t.
static int64_t hx_open(const at::Tensor& handle) {
  TORCH_CHECK(handle.device().is_cpu() && handle.scalar_type() == at::kByte &&
                  handle.numel() == (int64_t)sizeof(hipIpcMemHandle_t) &&
                  handle.is_contiguous(),
              "hx_open: handle must be a contiguous uint8[64] CPU tensor");
  hipIpcMemHandle_t h;
  std::memcpy(&h, handle.data_ptr(), sizeof(h));
  void* p = nullptr;
  HX_CHK(hipIpcOpenMemHandle(&p, h, hipIpcMemLazyEnablePeerAccess));
  return reinterpret_cast<int64_t>(p);
}

static void hx_close(int64_t ptr) {
  HX_CHK(hipIpcCloseMemHandle(reinterpret_cast<void*>(ptr)));
}

// Zero-copy tensor over raw device memory (inbox slots of the server); never
// owns it.
static at::Tensor hx_view(int64_t ptr, at::IntArrayRef sizes,
                          at::ScalarType dtype, int64_t device) {
  return at::from_blob(
      reinterpret_cast<void*>(ptr), sizes, [](void*) {},
      at::TensorOptions().dtype(dtype).device(at::Device(at::kCUDA, device)));
}

// ---------------------------------------------------------------------------
// Clock calibration
// ---------------------------------------------------------------------------
// Returns float64[2]: measured wall-clock ticks per millisecond (median of 3
// rounds of interval_ms each, device clock against the host steady clock, two
// clock-read kernels per round) and the hipDeviceAttributeWallClockRate
// (kHz = ticks/ms) for comparison.
static at::Tensor hx_calibrate(int64_t interval_ms) {
  TORCH_CHECK(interval_ms >= 10 && interval_ms <= 5000,
              "hx_calibrate: interval_ms in [10, 5000]");
  long long* d = nullptr;
  HX_CHK(hipMalloc(reinterpret_cast<void**>(&d), sizeof(long long)));
  hipStream_t st = nullptr;
  HX_CHK(hipStreamCreateWithFlags(&st, hipStreamNonBlocking));
  auto sample = [&](long long* ticks) -> double {
    hipLaunchKernelGGL(hx_k_clock, dim3(1), dim3(1), 0, st, d);
    HX_CHK(hipStreamSynchronize(st));
    const auto t = std::chrono::steady_clock::now().time_since_epoch();
    HX_CHK(hipMemcpy(ticks, d, sizeof(long long), hipMemcpyDeviceToHost));
    return std::chrono::duration<double, std::milli>(t).count();
  };
  long long warm = 0;
  (void)sample(&warm);
  std::vector<double> rates;
  for (int r = 0; r < 3; r++) {
    long long k0 = 0, k1 = 0;
    const double h0 = sample(&k0);
    std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    const double h1 = sample(&k1);
    rates.push_back((double)(k1 - k0) / (h1 - h0));
  }
  (void)hipStreamDestroy(st);
  (void)hipFree(d);
  std::sort(rates.begin(), rates.end());
  int dev = 0, khz = 0;
  HX_CHK(hipGetDevice(&dev));
  if (hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, dev) !=
      hipSuccess) {
    (void)hipGetLastError();
    khz = 0;
  }
  auto out = at::empty({2}, at::TensorOptions().dtype(at::kDouble));
  out[0] = rates[1];
  out[1] = (double)khz;
  return out;
}

// ---------------------------------------------------------------------------
// ctx image
// ---------------------------------------------------------------------------
static at::Tensor hx_ctx_new() {
  auto t = at::zeros({HX_CTX_WORDS}, at::TensorOptions().dtype(at::kLong));
  HxCtx img;
  std::memset(&img, 0, sizeof(img));
  img.magic = HX_MAGIC;
  img.version = HX_VERSION;
  std::memcpy(t.mutable_data_ptr(), &img, sizeof(img));
  return t;
}

static void hx_ctx_set(at::Tensor& image, const std::string& name,
                       int64_t value) {
  check_cpu_image(image, "hx_ctx_set");
  const auto& f = fields();
  auto it = f.find(name);
  TORCH_CHECK(it != f.end(), "hx_ctx_set: unknown field '", name, "'");
  std::memcpy(static_cast<uint8_t*>(image.mutable_data_ptr()) + it->second.off,
              &value, it->second.size);  // little endian: low bytes first
}

static int64_t hx_ctx_get(const at::Tensor& image, const std::string& name) {
  check_cpu_image(image, "hx_ctx_get");
  const auto& f = fields();
  auto it = f.find(name);
  TORCH_CHECK(it != f.end(), "hx_ctx_get: unknown field '", name, "'");
  int64_t v = 0;
  std::memcpy(&v,
              static_cast<const uint8_t*>(image.data_ptr()) + it->second.off,
              it->second.size);
  return v;
}

// Uploads the image into the device ctx (synchronously) and records the launch
// geometry host-side. Call once per ctx, outside any capture.
static void hx_ctx_register(const at::Tensor& ctx, const at::Tensor& image) {
  check_cpu_image(image, "hx_ctx_register");
  TORCH_CHECK(ctx.is_cuda() && ctx.scalar_type() == at::kLong &&
                  ctx.numel() == HX_CTX_WORDS && ctx.is_contiguous(),
              "hx_ctx_register: ctx must be a contiguous int64[", HX_CTX_WORDS,
              "] CUDA tensor");
  HxCtx img;
  std::memcpy(&img, image.data_ptr(), sizeof(img));
  TORCH_CHECK(img.magic == HX_MAGIC && img.version == HX_VERSION,
              "hx_ctx_register: image was not made by hx_ctx_new");
  TORCH_CHECK(img.role == 1 || img.role == 2, "hx_ctx_register: bad role");
  TORCH_CHECK(img.rows_cap >= 1 && img.topk >= 1 && img.topk <= HX_MAX_TOPK &&
                  img.hidden >= 8 && img.hidden % 8 == 0 && img.nslots >= 1 &&
                  img.nslots <= HX_MAX_SLOTS && img.num_experts >= 1 &&
                  img.num_experts <= HX_MAX_EXPERTS && img.nlayers >= 1 &&
                  img.nlayers <= HX_MAX_LAYERS,
              "hx_ctx_register: geometry out of range");
  TORCH_CHECK(img.own_base != nullptr && img.ticks_per_ms > 0 &&
                  img.wait_ticks > 0 && img.slot_stride > 0 && img.y_stride > 0,
              "hx_ctx_register: own_base/ticks/wait/strides not set");
  if (img.role == 1) {
    TORCH_CHECK(img.nsrv >= 1 && img.nsrv <= HX_MAX_SERVERS &&
                    img.hub_slot != nullptr && img.cold_srv != nullptr &&
                    img.n_cold != nullptr && img.ktab != nullptr,
                "hx_ctx_register: hub needs nsrv and the tables");
    for (uint32_t s = 0; s < img.nsrv; s++)
      TORCH_CHECK(img.peer_base[s] != nullptr, "hx_ctx_register: peer_base", s,
                  " not set");
  } else {
    TORCH_CHECK(img.index < HX_MAX_SERVERS && img.ids32 != nullptr &&
                    img.peer_base[0] != nullptr && img.begin_wait_ticks > 0,
                "hx_ctx_register: server needs index, ids32, peer_base0, "
                "begin_wait_ticks");
  }
  const at::cuda::OptionalCUDAGuard guard(ctx.device());
  HX_CHK(hipMemcpy(ctx.data_ptr(), image.data_ptr(), sizeof(HxCtx),
                   hipMemcpyHostToDevice));
  std::lock_guard<std::mutex> lk(g_mu);
  HxHost& h = g_hosts[ctx.data_ptr()];
  if (h.side == nullptr) {
    HX_CHK(hipStreamCreateWithFlags(&h.side, hipStreamNonBlocking));
    void* pin = nullptr;
    HX_CHK(hipHostMalloc(&pin, kStageBytes, hipHostMallocDefault));
    h.stage = static_cast<uint8_t*>(pin);
    std::memset(h.stage, 0, kStageBytes);
  }
  h.role = (int)img.role;
  h.nsrv = (int)img.nsrv;
  h.topk = (int)img.topk;
  h.hidden = (int)img.hidden;
  h.rows_cap = (int)img.rows_cap;
  h.nslots = (int)img.nslots;
  h.nlayers = (int)img.nlayers;
  h.own_base = img.own_base;
  h.blocks_disp = env_int("VLLM_HX_DISPATCH_BLOCKS", 0);
  h.blocks_comb = env_int("VLLM_HX_COMBINE_BLOCKS", 0);
  h.blocks_push = env_int("VLLM_HX_PUSH_BLOCKS", 0);
  h.blocks_wait = env_int("VLLM_HX_WAIT_BLOCKS", 0);
}

static void hx_ctx_unregister(const at::Tensor& ctx) {
  std::lock_guard<std::mutex> lk(g_mu);
  auto it = g_hosts.find(ctx.data_ptr());
  if (it == g_hosts.end()) return;
  HxHost& h = it->second;
  if (h.side != nullptr) {
    (void)hipStreamSynchronize(h.side);
    (void)hipStreamDestroy(h.side);
  }
  if (h.stage != nullptr) (void)hipHostFree(h.stage);
  g_hosts.erase(it);
}

// ---------------------------------------------------------------------------
// Host side channel: status, abort poke, server step descriptor
// ---------------------------------------------------------------------------
// int64[8] = ctx.status: [abort record, epoch, live record, steps, dispatches,
// combines, pairs on hub, pairs remote]. A 64 B device-to-host copy on a side
// stream: it neither waits for nor blocks the compute stream.
static at::Tensor hx_status(const at::Tensor& ctx) {
  HxHost& h = host_checked(ctx, 0);
  const at::cuda::OptionalCUDAGuard guard(ctx.device());
  auto out = at::empty({8}, at::TensorOptions().dtype(at::kLong));
  std::lock_guard<std::mutex> lk(h.side_mu);
  HX_CHK(hipMemcpyAsync(
      h.stage,
      reinterpret_cast<uint8_t*>(ctx.data_ptr()) + offsetof(HxCtx, status),
      8 * sizeof(uint64_t), hipMemcpyDeviceToHost, h.side));
  HX_CHK(hipStreamSynchronize(h.side));
  std::memcpy(out.mutable_data_ptr(), h.stage, 8 * sizeof(uint64_t));
  return out;
}

// Wake every kernel of this process that waits in an hx poll (peer death seen
// by the host): a non-zero word in the own control page, written by the copy
// engine, so it works while the compute stream is stuck in a spin.
static void hx_host_abort(const at::Tensor& ctx) {
  HxHost& h = host_checked(ctx, 0);
  const at::cuda::OptionalCUDAGuard guard(ctx.device());
  std::lock_guard<std::mutex> lk(h.side_mu);
  const uint64_t one = 1;
  std::memcpy(h.stage + 64, &one, 8);
  HX_CHK(hipMemcpyAsync(h.own_base + HX_CTRL_HOST_ABORT, h.stage + 64, 8,
                        hipMemcpyHostToDevice, h.side));
  HX_CHK(hipStreamSynchronize(h.side));
}

// Server: the host descriptor of the next step into the ctx (one 16 B copy on
// the current stream, ordered before the replay that follows).
static void hx_srv_post_desc(at::Tensor& ctx, int64_t epoch, int64_t T,
                             int64_t bucket) {
  HxHost& h = host_checked(ctx, 2);
  const at::cuda::OptionalCUDAGuard guard(ctx.device());
  uint8_t* slot = h.stage + 128 + 16 * (h.desc_next++ % kDescRing);
  const uint64_t e = (uint64_t)epoch;
  const uint32_t t = (uint32_t)T, b = (uint32_t)bucket;
  std::memcpy(slot, &e, 8);
  std::memcpy(slot + 8, &t, 4);
  std::memcpy(slot + 12, &b, 4);
  HX_CHK(hipMemcpyAsync(
      reinterpret_cast<uint8_t*>(ctx.data_ptr()) + offsetof(HxCtx, desc_epoch),
      slot, 16, hipMemcpyHostToDevice, cur_stream()));
}

// ---------------------------------------------------------------------------
// Compute ops
// ---------------------------------------------------------------------------
static void hx_begin_step(at::Tensor& ctx, int64_t T, int64_t bucket) {
  HxHost& h = host_checked(ctx, 1);
  TORCH_CHECK(T >= 1 && T <= h.rows_cap, "hx_begin_step: T ", T,
              " outside [1, ", h.rows_cap, "]");
  TORCH_CHECK(bucket >= 0 && bucket <= h.rows_cap, "hx_begin_step: bad bucket ",
              bucket);
  const at::cuda::OptionalCUDAGuard guard(ctx.device());
  hx_launch_begin_step(dev_ctx(ctx), (int)T, (int)bucket, cur_stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void hx_dispatch(const at::Tensor& x, const at::Tensor& topk_ids,
                        const at::Tensor& topk_w, at::Tensor& hub_ids_out,
                        at::Tensor& ctx, int64_t layer) {
  HxHost& h = host_checked(ctx, 1);
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.dim() == 2 &&
                  x.size(1) == h.hidden && x.is_contiguous(),
              "hx_dispatch: x must be contiguous fp16 [T, ", h.hidden, "]");
  const int64_t T = x.size(0);
  TORCH_CHECK(T >= 1 && T <= h.rows_cap, "hx_dispatch: T ", T, " outside [1, ",
              h.rows_cap, "]");
  TORCH_CHECK(topk_ids.scalar_type() == at::kInt && topk_ids.dim() == 2 &&
                  topk_ids.size(0) == T && topk_ids.size(1) == h.topk &&
                  topk_ids.is_contiguous(),
              "hx_dispatch: topk_ids must be contiguous int32 [T, ", h.topk,
              "]");
  TORCH_CHECK(topk_w.scalar_type() == at::kFloat && topk_w.dim() == 2 &&
                  topk_w.size(0) == T && topk_w.size(1) == h.topk &&
                  topk_w.is_contiguous(),
              "hx_dispatch: topk_w must be contiguous float32 [T, ", h.topk,
              "]");
  TORCH_CHECK(hub_ids_out.scalar_type() == at::kInt && hub_ids_out.dim() == 2 &&
                  hub_ids_out.size(0) == T && hub_ids_out.size(1) == h.topk &&
                  hub_ids_out.is_contiguous(),
              "hx_dispatch: hub_ids_out must be contiguous int32 [T, ", h.topk,
              "]");
  TORCH_CHECK(layer >= 0 && layer < h.nlayers, "hx_dispatch: layer ", layer,
              " outside [0, ", h.nlayers, ")");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "hx_dispatch: x must be 16-byte aligned");
  const at::cuda::OptionalCUDAGuard guard(x.device());
  hx_launch_dispatch(dev_ctx(ctx), x.data_ptr(), topk_ids.data_ptr<int32_t>(),
                     topk_w.data_ptr<float>(), hub_ids_out.data_ptr<int32_t>(),
                     (int)T, (int)layer, h.nsrv, h.hidden, h.blocks_disp,
                     cur_stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void hx_combine(const at::Tensor& y_hub, at::Tensor& out,
                       at::Tensor& ctx, int64_t layer) {
  HxHost& h = host_checked(ctx, 1);
  TORCH_CHECK(y_hub.is_cuda() && y_hub.scalar_type() == at::kHalf &&
                  y_hub.dim() == 2 && y_hub.size(1) == h.hidden &&
                  y_hub.is_contiguous(),
              "hx_combine: y_hub must be contiguous fp16 [T, ", h.hidden, "]");
  const int64_t T = y_hub.size(0);
  TORCH_CHECK(T >= 1 && T <= h.rows_cap, "hx_combine: T ", T, " outside [1, ",
              h.rows_cap, "]");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kHalf &&
                  out.sizes() == y_hub.sizes() && out.is_contiguous(),
              "hx_combine: out must be contiguous fp16 like y_hub");
  TORCH_CHECK(layer >= 0 && layer < h.nlayers, "hx_combine: layer ", layer,
              " outside [0, ", h.nlayers, ")");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(y_hub.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
              "hx_combine: tensors must be 16-byte aligned");
  const at::cuda::OptionalCUDAGuard guard(y_hub.device());
  hx_launch_combine(dev_ctx(ctx), y_hub.data_ptr(), out.data_ptr(), (int)T,
                    (int)layer, h.hidden, h.blocks_comb, cur_stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void hx_srv_begin(at::Tensor& ctx) {
  (void)host_checked(ctx, 2);
  const at::cuda::OptionalCUDAGuard guard(ctx.device());
  hx_launch_srv_begin(dev_ctx(ctx), cur_stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void hx_srv_wait(at::Tensor& ctx, int64_t layer, int64_t T) {
  HxHost& h = host_checked(ctx, 2);
  TORCH_CHECK(T >= 1 && T <= h.rows_cap, "hx_srv_wait: T ", T, " outside [1, ",
              h.rows_cap, "]");
  TORCH_CHECK(layer >= 0 && layer < h.nlayers, "hx_srv_wait: layer ", layer,
              " outside [0, ", h.nlayers, ")");
  const at::cuda::OptionalCUDAGuard guard(ctx.device());
  hx_launch_srv_wait(dev_ctx(ctx), (int)layer, (int)T, h.hidden, h.blocks_wait,
                     cur_stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void hx_srv_push(const at::Tensor& y, at::Tensor& ctx, int64_t layer,
                        int64_t T) {
  HxHost& h = host_checked(ctx, 2);
  TORCH_CHECK(T >= 1 && T <= h.rows_cap, "hx_srv_push: T ", T, " outside [1, ",
              h.rows_cap, "]");
  TORCH_CHECK(layer >= 0 && layer < h.nlayers, "hx_srv_push: layer ", layer,
              " outside [0, ", h.nlayers, ")");
  TORCH_CHECK(y.is_cuda() && y.scalar_type() == at::kHalf && y.dim() == 2 &&
                  y.size(0) == T && y.size(1) == h.hidden && y.is_contiguous(),
              "hx_srv_push: y must be contiguous fp16 [T, ", h.hidden, "]");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(y.data_ptr()) % 16 == 0,
              "hx_srv_push: y must be 16-byte aligned");
  const at::cuda::OptionalCUDAGuard guard(y.device());
  hx_launch_srv_push(dev_ctx(ctx), y.data_ptr(), (int)T, (int)layer, h.hidden,
                     h.blocks_push, cur_stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// Registration (called from csrc/rocm/torch_bindings.cpp inside the _rocm_C
// library; tools/rdna/hetero_moe/hx_dev_register.cpp registers the same ops
// into a library fragment for fast development builds).
// ---------------------------------------------------------------------------
void hx_register_ops(torch::Library& m) {
  m.def("hx_alloc(int nbytes, int kind) -> Tensor");
  m.impl("hx_alloc", torch::kCPU, &hx_alloc);
  m.def("hx_free(int ptr) -> ()", &hx_free);
  m.def("hx_open(Tensor handle) -> int");
  m.impl("hx_open", torch::kCPU, &hx_open);
  m.def("hx_close(int ptr) -> ()", &hx_close);
  m.def("hx_view(int ptr, int[] sizes, ScalarType dtype, int device) -> Tensor",
        &hx_view);
  m.def("hx_calibrate(int interval_ms) -> Tensor", &hx_calibrate);
  m.def("hx_ctx_new() -> Tensor", &hx_ctx_new);
  m.def("hx_ctx_set(Tensor(a!) image, str name, int value) -> ()");
  m.impl("hx_ctx_set", torch::kCPU, &hx_ctx_set);
  m.def("hx_ctx_get(Tensor image, str name) -> int");
  m.impl("hx_ctx_get", torch::kCPU, &hx_ctx_get);
  m.def("hx_ctx_register(Tensor ctx, Tensor image) -> ()");
  m.impl("hx_ctx_register", torch::kCUDA, &hx_ctx_register);
  m.def("hx_ctx_unregister(Tensor ctx) -> ()");
  m.impl("hx_ctx_unregister", torch::kCUDA, &hx_ctx_unregister);
  m.def("hx_status(Tensor ctx) -> Tensor");
  m.impl("hx_status", torch::kCUDA, &hx_status);
  m.def("hx_host_abort(Tensor ctx) -> ()");
  m.impl("hx_host_abort", torch::kCUDA, &hx_host_abort);
  m.def("hx_srv_post_desc(Tensor! ctx, int epoch, int T, int bucket) -> ()");
  m.impl("hx_srv_post_desc", torch::kCUDA, &hx_srv_post_desc);
  m.def("hx_begin_step(Tensor! ctx, int T, int bucket) -> ()");
  m.impl("hx_begin_step", torch::kCUDA, &hx_begin_step);
  m.def(
      "hx_dispatch(Tensor x, Tensor topk_ids, Tensor topk_w, "
      "Tensor! hub_ids_out, Tensor! ctx, int layer) -> ()");
  m.impl("hx_dispatch", torch::kCUDA, &hx_dispatch);
  m.def("hx_combine(Tensor y_hub, Tensor! out, Tensor! ctx, int layer) -> ()");
  m.impl("hx_combine", torch::kCUDA, &hx_combine);
  m.def("hx_srv_begin(Tensor! ctx) -> ()");
  m.impl("hx_srv_begin", torch::kCUDA, &hx_srv_begin);
  m.def("hx_srv_wait(Tensor! ctx, int layer, int T) -> ()");
  m.impl("hx_srv_wait", torch::kCUDA, &hx_srv_wait);
  m.def("hx_srv_push(Tensor y, Tensor! ctx, int layer, int T) -> ()");
  m.impl("hx_srv_push", torch::kCUDA, &hx_srv_push);
}
