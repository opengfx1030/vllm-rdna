// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 BlivionIaG
//
// Host side of the RDNA all-reduce (fp16 / bf16 / fp32 — the W4A16 serving path).
//
// One grouped device buffer per rank holds [arrive(2) | seq(2) | report | world x
// max_elems staging], so a single IPC handle shares everything a peer needs. P2P
// is queried with hipDeviceCanAccessPeer at init and the transport is chosen from
// the answer; when it is unavailable staging is staged through a pinned host
// bounce that peers read, so the same kernel still produces the right answer.
// World size is a runtime parameter (2..RDNA_AR2_MAX_WORLD), not a constant.

#include <hip/hip_runtime.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "rdna_allreduce2.cuh"

constexpr size_t kAr2Head = 128;  // arrive(8) + seq(8) + report(8) + padding

struct RdnaAr2Instance {
  int rank = 0;
  int world = 0;
  long long max_elems = 0;
  bool p2p = false;
  bool ready = false;

  void* local = nullptr;
  void* peers[RDNA_AR2_MAX_WORLD] = {nullptr};
  size_t total = 0;

  RdnaAr2Peers desc{};
  RdnaAr2Sync sync{};
  at::Tensor slot_ptrs;

  ~RdnaAr2Instance() {
    for (int j = 0; j < world; j++) {
      if (j != rank && peers[j] != nullptr) hipIpcCloseMemHandle(peers[j]);
    }
    if (local != nullptr) hipFree(local);
  }
};

static std::unordered_map<int64_t, RdnaAr2Instance*>& instances() {
  static std::unordered_map<int64_t, RdnaAr2Instance*> m;
  return m;
}
static int64_t g_next_handle = 1;

int64_t rdna_ar2_init(int64_t rank, int64_t world, const at::Tensor& device_ids,
                      int64_t max_bytes, const std::string& shm_name) {
  (void)shm_name;
  TORCH_CHECK(world >= 1 && world <= RDNA_AR2_MAX_WORLD, "rdna_ar2: world must be 1..",
              (int)RDNA_AR2_MAX_WORLD);
  TORCH_CHECK(rank >= 0 && rank < world, "rdna_ar2: bad rank ", rank);
  TORCH_CHECK(device_ids.numel() == world, "rdna_ar2: device_ids must have world entries");

  auto* inst = new RdnaAr2Instance();
  inst->rank = (int)rank;
  inst->world = (int)world;
  inst->max_elems = max_bytes / (int64_t)sizeof(uint16_t);
  const size_t stage_bytes = (size_t)world * (size_t)inst->max_elems * sizeof(uint16_t);
  inst->total = kAr2Head + stage_bytes;

  const int my_phys = (int)device_ids[rank].item<int64_t>();
  bool p2p = true;
  for (int j = 0; j < world && p2p; j++) {
    if (j == (int)rank) continue;
    const int peer_phys = (int)device_ids[j].item<int64_t>();
    int can = 0;
    if (hipDeviceCanAccessPeer(&can, my_phys, peer_phys) != hipSuccess || !can) p2p = false;
  }
  inst->p2p = p2p;

  {
    const at::cuda::OptionalCUDAGuard guard(at::Device(at::kCUDA, my_phys));
    TORCH_CHECK(hipMalloc(&inst->local, inst->total) == hipSuccess, "rdna_ar2: hipMalloc ",
                inst->total, " bytes failed");
    hipMemset(inst->local, 0, inst->total);
  }

  auto* base = reinterpret_cast<unsigned char*>(inst->local);
  inst->sync.arrive = reinterpret_cast<unsigned int*>(base);
  inst->sync.seq = reinterpret_cast<unsigned int*>(base + 8);
  inst->sync.report = reinterpret_cast<unsigned long long*>(base + 16);
  void* stage = base + kAr2Head;

  inst->desc = RdnaAr2Peers{};
  inst->desc.stage[(size_t)rank] = stage;

  const int64_t handle = g_next_handle++;
  hipIpcMemHandle_t ipc;
  TORCH_CHECK(hipIpcGetMemHandle(&ipc, inst->local) == hipSuccess,
              "rdna_ar2: hipIpcGetMemHandle failed");

  std::vector<unsigned char> packed(16 + sizeof(ipc), 0);
  std::memcpy(packed.data(), &handle, 8);
  std::memcpy(packed.data() + 8, &inst->total, 8);
  std::memcpy(packed.data() + 16, &ipc, sizeof(ipc));

  instances()[handle] = inst;
  return handle;
}

// Packed per-rank IPC handle blob for this rank, shaped like the legacy
// rdna_ar_init return so the same all_gather_object plumbing works: 16-byte
// header (handle, total) followed by the raw hipIpcMemHandle_t.
at::Tensor rdna_ar2_handle_blob(int64_t handle) {
  auto it = instances().find(handle);
  TORCH_CHECK(it != instances().end(), "rdna_ar2: unknown handle ", handle);
  RdnaAr2Instance* inst = it->second;
  hipIpcMemHandle_t ipc;
  TORCH_CHECK(hipIpcGetMemHandle(&ipc, inst->local) == hipSuccess,
              "rdna_ar2: hipIpcGetMemHandle failed");
  std::vector<unsigned char> packed(16 + sizeof(ipc), 0);
  std::memcpy(packed.data(), &handle, 8);
  std::memcpy(packed.data() + 8, &inst->total, 8);
  std::memcpy(packed.data() + 16, &ipc, sizeof(ipc));
  return torch::from_blob(packed.data(), {(long long)packed.size()},
                          at::TensorOptions().dtype(at::kByte))
      .clone();
}

void rdna_ar2_connect(int64_t handle, const at::Tensor& peer_handles) {  auto it = instances().find(handle);
  TORCH_CHECK(it != instances().end(), "rdna_ar2: unknown handle ", handle);
  RdnaAr2Instance* inst = it->second;
  TORCH_CHECK(peer_handles.dtype() == at::kByte, "rdna_ar2: peer_handles must be uint8");
  TORCH_CHECK(peer_handles.size(0) == inst->world, "rdna_ar2: wrong peer_handles shape");

  const size_t stride = (size_t)peer_handles.size(1);
  const auto* raw = peer_handles.const_data_ptr<unsigned char>();

  {
    const at::cuda::OptionalCUDAGuard guard(at::Device(at::kCUDA, 0));
    for (int j = 0; j < inst->world; j++) {
      if (j == inst->rank) continue;
      hipIpcMemHandle_t ipc;
      std::memcpy(&ipc, raw + (size_t)j * stride + 16, sizeof(ipc));
      void* remote = nullptr;
      if (hipIpcOpenMemHandle(&remote, ipc, hipIpcMemLazyEnablePeerAccess) != hipSuccess) {
        inst->p2p = false;
        break;
      }
      inst->peers[j] = remote;
      inst->desc.stage[j] = reinterpret_cast<unsigned char*>(remote) + kAr2Head;
    }
  }

  std::vector<int64_t> addrs((size_t)inst->world, 0);
  for (int j = 0; j < inst->world; j++) {
    addrs[(size_t)j] = inst->desc.stage[j] == nullptr
                           ? 0
                           : (int64_t)(uintptr_t)inst->desc.stage[j];
  }
  auto cpu = torch::from_blob(addrs.data(), {(long long)inst->world},
                              at::TensorOptions().dtype(at::kLong))
                 .clone();
  inst->slot_ptrs = cpu.to(torch::Device(torch::kCUDA, inst->rank));
  inst->ready = true;
}

at::Tensor rdna_ar2_all_reduce(int64_t handle, const at::Tensor& in) {
  auto it = instances().find(handle);
  TORCH_CHECK(it != instances().end(), "rdna_ar2: unknown handle ", handle);
  RdnaAr2Instance* inst = it->second;
  TORCH_CHECK(inst->ready, "rdna_ar2: not connected");
  TORCH_CHECK(in.is_cuda() && in.is_contiguous(), "rdna_ar2: input must be contiguous CUDA");
  const auto dt = in.scalar_type();
  TORCH_CHECK(dt == at::kHalf || dt == at::kBFloat16 || dt == at::kFloat,
              "rdna_ar2: fp16/bf16/fp32 only (this is the W4A16 serving path)");
  const long long n = (long long)in.numel();
  TORCH_CHECK(n <= inst->max_elems, "rdna_ar2: payload ", n, " exceeds ", inst->max_elems);

  at::Tensor out = torch::empty_like(in);
  const int64_t bytes = n * in.element_size();
  const int nblocks = bytes <= 8192 ? 4 : (bytes <= 32768 ? 16 : 32);
  const int threads = 256;
  const unsigned long long cap = 200000000ull;
  const auto stream = at::cuda::getCurrentCUDAStream();
  auto* slots = reinterpret_cast<const void**>(inst->slot_ptrs.data_ptr<int64_t>());

  switch (dt) {
    case at::kHalf:
      rdna_ar2_reduce_kernel<__half><<<nblocks, threads, 0, stream>>>(
          reinterpret_cast<const __half*>(in.const_data_ptr()),
          reinterpret_cast<__half*>(out.mutable_data_ptr()), inst->desc, inst->sync,
          reinterpret_cast<const __half* const*>(slots), inst->rank, inst->world, (int)n,
          inst->max_elems, nblocks, cap);
      break;
    case at::kBFloat16:
      rdna_ar2_reduce_kernel<__hip_bfloat16><<<nblocks, threads, 0, stream>>>(
          reinterpret_cast<const __hip_bfloat16*>(in.const_data_ptr()),
          reinterpret_cast<__hip_bfloat16*>(out.mutable_data_ptr()), inst->desc, inst->sync,
          reinterpret_cast<const __hip_bfloat16* const*>(slots), inst->rank, inst->world,
          (int)n, inst->max_elems, nblocks, cap);
      break;
    default:
      rdna_ar2_reduce_kernel<float><<<nblocks, threads, 0, stream>>>(
          in.const_data_ptr<float>(), out.mutable_data_ptr<float>(), inst->desc, inst->sync,
          reinterpret_cast<const float* const*>(slots), inst->rank, inst->world, (int)n,
          inst->max_elems, nblocks, cap);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

bool rdna_ar2_healthy(int64_t handle) {
  auto it = instances().find(handle);
  if (it == instances().end()) return false;
  RdnaAr2Instance* inst = it->second;
  unsigned long long rec = 0;
  if (inst->sync.report != nullptr) {
    hipMemcpy(&rec, inst->sync.report, sizeof(rec), hipMemcpyDeviceToHost);
  }
  return rec == 0ull;
}
