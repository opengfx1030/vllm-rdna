// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 BlivionIaG
//
// RDNA all-reduce v2 — reduce-scatter + allgather over PCIe P2P.
//
// The push-all shape moves (W-1)*N bytes per rank; reduce-scatter+allgather moves
// 2(W-1)/W*N, i.e. 1.5*N at W=4 and 1.75*N at W=8 versus 3*N and 7*N. RCCL
// reached 169 GB/s effective at 10 MB on this box, so bytes alone did not explain
// the old kernel's 5x loss; it also stopped the whole grid on a device-wide
// barrier between its push and reduce phases. This file removes that barrier:
// chunk c is reduced by its owner as soon as that owner's inputs land, and
// re-broadcast as soon as it is finished, so no rank waits on the whole grid.
//
// Protocol (validated in benchmarks/kernels/ra2_protocol_model.py before coding):
//   phase A  every rank stages its slice of EVERY chunk to the chunk's owner
//   reduce   owner (c % W) waits for W-1 flags, sums W contributions
//   phase B  owner publishes the finished chunk into every peer, then flags them
// Chunk count must satisfy CH % W == 0 and CH >= W, so every rank owns CH/W
// chunks; CH=2 with W=4 leaves two ranks owning nothing and idling each call.

#include <hip/hip_runtime.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "rdna_ar2_rs.cuh"

namespace {

constexpr int kMaxWorld = RDNA_ARS_MAX_WORLD;
constexpr int kMaxChunks = RDNA_ARS_MAX_CHUNKS;

// Head of the grouped buffer: flags for every (row, peer, chunk).
// Rows are phase A then phase B, so only two rows are needed.
constexpr size_t kFlagWords = (size_t)2 * (size_t)kMaxWorld * (size_t)kMaxChunks;

// Wedge flag, written by device code on spin-cap expiry so the host can report a
// dead peer instead of silently returning garbage.
constexpr size_t kFailWords = 16;

// Staging per rank: contrib[world senders][chunks] + result[chunks].
inline size_t kStageElems(int world, long long max_elems) {
  return (size_t)world * (size_t)world * (size_t)max_elems +
         (size_t)world * (size_t)max_elems;
}
constexpr size_t kFlagBytes = (kFlagWords + kFailWords) * sizeof(int) + 64;

struct RsInstance {
  int rank = 0;
  int world = 0;
  int chunks = 0;
  long long max_elems = 0;
  long long chunk_elems = 0;
  bool p2p = false;
  bool ready = false;
  int64_t gen_counter = 0;
  void* local = nullptr;
  void* peers[kMaxWorld] = {nullptr};
  size_t total = 0;

  RsPeers desc{};
  at::Tensor self_ptrs;
  int* fail = nullptr;

  ~RsInstance() {
    for (int j = 0; j < world; j++) {
      if (j != rank && peers[j] != nullptr) hipIpcCloseMemHandle(peers[j]);
    }
    if (local != nullptr) hipFree(local);
  }
};

std::unordered_map<int64_t, RsInstance*>& instances() {
  static std::unordered_map<int64_t, RsInstance*> m;
  return m;
}
int64_t g_next = 1;

constexpr int kThreads = 256;
constexpr int kBlocks = 32;
constexpr unsigned long long kSpinCap = 200000000ull;

int pick_chunks(int world, int n) {
  if (world <= 1) return 1;
  int ch = world;
  while (ch * 2 <= kMaxChunks && n % (ch * 2) == 0) ch *= 2;
  return ch;
}

}  // namespace

int64_t rdna_ars_init(int64_t rank, int64_t world, const at::Tensor& device_ids,
                      int64_t max_bytes) {
  TORCH_CHECK(world >= 1 && world <= kMaxWorld, "rdna_ars: world must be 1..", kMaxWorld);
  TORCH_CHECK(rank >= 0 && rank < world, "rdna_ars: bad rank ", rank);
  TORCH_CHECK(device_ids.numel() == world, "rdna_ars: device_ids must have world entries");

  auto* inst = new RsInstance();
  inst->rank = (int)rank;
  inst->world = (int)world;
  inst->max_elems = max_bytes / (int64_t)sizeof(uint16_t);

  // staging: contrib[world senders][world owners][chunk] + result[world][chunk]
  const size_t stage_elems = kStageElems((int)world, inst->max_elems);
  inst->total = kFlagBytes + stage_elems * sizeof(uint16_t);

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
    hipError_t e = hipMalloc(&inst->local, inst->total);
    TORCH_CHECK(e == hipSuccess, "rdna_ars: hipMalloc ", inst->total, " failed: ",
                hipGetErrorString(e));
    TORCH_CHECK(hipMemset(inst->local, 0, inst->total) == hipSuccess, "rdna_ars: memset");
  }

  auto* base = reinterpret_cast<unsigned char*>(inst->local);
  int* flags = reinterpret_cast<int*>(base);
  inst->fail = flags + kFlagWords;
  void* stage = base + kFlagBytes;

  inst->desc = RsPeers{};
  inst->desc.stage[(size_t)rank] = stage;
  inst->desc.flags[(size_t)rank] = flags;
  inst->desc.row_a = 0;
  inst->desc.row_b = 1;
  inst->desc.rank = (int)rank;

  const int64_t handle = g_next++;
  hipIpcMemHandle_t ipc;
  TORCH_CHECK(hipIpcGetMemHandle(&ipc, inst->local) == hipSuccess, "rdna_ars: ipc handle");

  std::vector<unsigned char> packed(16 + sizeof(ipc), 0);
  std::memcpy(packed.data(), &handle, 8);
  std::memcpy(packed.data() + 8, &inst->total, 8);
  std::memcpy(packed.data() + 16, &ipc, sizeof(ipc));

  instances()[handle] = inst;
  return handle;
}

at::Tensor rdna_ars_handle_blob(int64_t handle) {
  auto it = instances().find(handle);
  TORCH_CHECK(it != instances().end(), "rdna_ars: bad handle ", handle);
  auto* inst = it->second;
  hipIpcMemHandle_t ipc;
  TORCH_CHECK(hipIpcGetMemHandle(&ipc, inst->local) == hipSuccess, "rdna_ars: ipc handle");
  std::vector<unsigned char> packed(16 + sizeof(ipc), 0);
  std::memcpy(packed.data(), &handle, 8);
  std::memcpy(packed.data() + 8, &inst->total, 8);
  std::memcpy(packed.data() + 16, &ipc, sizeof(ipc));
  auto t = at::empty({(int64_t)packed.size()}, at::TensorOptions().dtype(at::kByte));
  std::memcpy(t.data_ptr(), packed.data(), packed.size());
  return t;
}

void rdna_ars_connect(int64_t handle, const at::Tensor& peer_handles) {
  auto it = instances().find(handle);
  TORCH_CHECK(it != instances().end(), "rdna_ars: bad handle ", handle);
  auto* inst = it->second;

  if (inst->world == 1) {
    inst->ready = true;
    return;
  }
  TORCH_CHECK(peer_handles.dim() == 2 && peer_handles.size(0) == inst->world,
              "rdna_ars: peer_handles must be [world, blob]");

  const int64_t blob = peer_handles.size(1);
  TORCH_CHECK(blob >= (int64_t)(16 + sizeof(hipIpcMemHandle_t)), "rdna_ars: blob too small");

  std::vector<unsigned char> row((size_t)blob);
  for (int j = 0; j < inst->world; j++) {
    if (j == inst->rank) continue;
    std::memcpy(row.data(), peer_handles[j].data_ptr(), (size_t)blob);
    hipIpcMemHandle_t ipc;
    std::memcpy(&ipc, row.data() + 16, sizeof(ipc));
    void* p = nullptr;
    hipError_t e = hipIpcOpenMemHandle(&p, ipc, hipIpcMemLazyEnablePeerAccess);
    TORCH_CHECK(e == hipSuccess, "rdna_ars: hipIpcOpenMemHandle peer ", j, " failed: ",
                hipGetErrorString(e));
    inst->peers[j] = p;
    auto* pbase = reinterpret_cast<unsigned char*>(p);
    inst->desc.stage[(size_t)j] = pbase + kFlagBytes;
    inst->desc.flags[(size_t)j] = reinterpret_cast<int*>(pbase);
  }
  inst->ready = true;
}

at::Tensor rdna_ars_all_reduce(int64_t handle, const at::Tensor& in) {
  auto it = instances().find(handle);
  TORCH_CHECK(it != instances().end(), "rdna_ars: bad handle ", handle);
  auto* inst = it->second;
  TORCH_CHECK(inst->ready, "rdna_ars: not connected");

  const int world = inst->world;
  if (world == 1) return in.clone();

  auto x = in.contiguous();
  const int64_t n = x.numel();
  TORCH_CHECK(n > 0, "rdna_ars: empty input");
  TORCH_CHECK(n <= inst->max_elems, "rdna_ars: input ", n, " exceeds max_elems ",
              inst->max_elems);

  const int chunks = pick_chunks(world, (int)n);
  const long long chunk_elems = ((long long)n + chunks - 1) / chunks;

  auto out = at::empty_like(x);
  auto stream = at::cuda::getCurrentCUDAStream();

  static thread_local int64_t gen_counter = 0;
  const int gen = (int)(++gen_counter);

  const dim3 grid(kBlocks);
  const dim3 block(kThreads);

  if (x.scalar_type() == at::kHalf) {
    rdna_ars_scatter_kernel<__half><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr()), inst->desc, world, (int)chunks,
        (int)chunk_elems, (int)n, gen, kSpinCap, inst->fail);
    rdna_ars_gather_kernel<__half><<<grid, block, 0, stream>>>(
        reinterpret_cast<__half*>(out.data_ptr()), inst->desc, world, (int)chunks,
        (int)chunk_elems, (int)n, gen, kSpinCap, inst->fail);
  } else if (x.scalar_type() == at::kBFloat16) {
    rdna_ars_scatter_kernel<__hip_bfloat16><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __hip_bfloat16*>(x.data_ptr()), inst->desc, world, (int)chunks,
        (int)chunk_elems, (int)n, gen, kSpinCap, inst->fail);
    rdna_ars_gather_kernel<__hip_bfloat16><<<grid, block, 0, stream>>>(
        reinterpret_cast<__hip_bfloat16*>(out.data_ptr()), inst->desc, world, (int)chunks,
        (int)chunk_elems, (int)n, gen, kSpinCap, inst->fail);
  } else if (x.scalar_type() == at::kFloat) {
    rdna_ars_scatter_kernel<float><<<grid, block, 0, stream>>>(
        reinterpret_cast<const float*>(x.data_ptr()), inst->desc, world, (int)chunks,
        (int)chunk_elems, (int)n, gen, kSpinCap, inst->fail);
    rdna_ars_gather_kernel<float><<<grid, block, 0, stream>>>(
        reinterpret_cast<float*>(out.data_ptr()), inst->desc, world, (int)chunks,
        (int)chunk_elems, (int)n, gen, kSpinCap, inst->fail);
  } else {
    TORCH_CHECK(false, "rdna_ars: unsupported dtype");
  }
  return out;
}

bool rdna_ars_healthy(int64_t handle) { return instances().find(handle) != instances().end(); }
