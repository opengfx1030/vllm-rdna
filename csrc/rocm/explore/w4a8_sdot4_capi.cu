// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// C ABI over the W4A8 explore kernels, for the V620 harness only. Not part
// of the vLLM build (no CMake entry, no torch op, no torch headers):
//
//   hipcc -x hip -O3 -std=c++17 --offload-arch=gfx1030 -fPIC -shared \
//       -o libw4a8_sdot4_explore.so csrc/rocm/explore/w4a8_sdot4_capi.cu
//
// benchmarks/kernels/w4a8_sdot4_explore/lib.py builds and loads it with
// ctypes; pointers are raw device pointers (tensor.data_ptr()) and layouts
// are the ones documented in w4a8_sdot4.cuh.
//
// Every entry point returns 0 on success, a positive hipError_t, or one of
// the negative Err codes below (w4a8_error_str names them).

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "w4a8_sdot4.cuh"

namespace ex = vllm::explore_w4a8;

namespace {

enum Err : int {
  kBadShape = -1,
  kBadConfig = -2,
  kNotGfx1030 = -3,
  kLdsTooBig = -4,
  kBadSplit = -5,
  kBadGroup = -6,
};

constexpr int kMaxSplit = 16;
constexpr int kProbeChains = 8;

bool on_gfx1030() {
  thread_local int cached_dev = -1;
  thread_local bool cached_ok = false;
  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess) {
    return false;
  }
  if (dev != cached_dev) {
    hipDeviceProp_t prop;
    cached_ok = hipGetDeviceProperties(&prop, dev) == hipSuccess &&
                std::strncmp(prop.gcnArchName, "gfx1030", 7) == 0;
    cached_dev = dev;
  }
  return cached_ok;
}

int group_index(int group_size) {
  switch (group_size) {
    case 32:
      return 0;
    case 64:
      return 1;
    case 128:
      return 2;
    default:
      return -1;
  }
}

struct GemmArgs {
  const int8_t* a;
  const uint32_t* w;
  const uint32_t* qzeros;
  const ex::f16_t* scales;
  const float* a_scale;
  const int32_t* asum;
  void* out;
  int m, n, k, zero_offset, split_k, out_f32;
};

// Same rule as reference.pick_split_k: ConfigA's compute_split_k (LDS
// budget 16/64/32 KiB by grid size, then grow while the grid is small or the
// K range long), restricted to group-aligned splits.
template <class C>
int pick_split_k(int m, int n, int k) {
  const int blocks =
      ((m + C::M_TILE - 1) / C::M_TILE) * ((n + C::N_TILE - 1) / C::N_TILE);
  const int budget = blocks > 1024  ? 16 * 1024
                     : blocks > 256 ? 64 * 1024
                                    : 32 * 1024;
  const int groups = k / C::GROUP;
  int splits[kMaxSplit];
  int count = 0;
  for (int s = 1; s <= kMaxSplit; ++s) {
    if (groups % s == 0) {
      splits[count++] = s;
    }
  }
  auto lds = [&](int s) {
    const int kps = k / s;
    return C::M_TILE * kps +
           C::M_TILE * (kps / C::GROUP) * (C::A_GROUP ? 8 : 4);
  };
  int i = 0;
  while (i + 1 < count && lds(splits[i]) > budget) {
    ++i;
  }
  while (i + 1 < count && (blocks * splits[i] < 2048 || k / splits[i] > 2048)) {
    if (lds(splits[i + 1]) > budget) {
      break;
    }
    ++i;
  }
  // launch_gemm() rejects above the hard cap with kLdsTooBig, so never pick a
  // split that exceeds it (larger splits shrink k_per_split and thus LDS).
  while (i + 1 < count && ex::lds_bytes<C>(k / splits[i]) > 64 * 1024) {
    ++i;
  }
  return splits[i];
}

template <class C>
int launch_gemm(const GemmArgs& p, hipStream_t stream) {
  if (p.m <= 0 || p.n <= 0 || p.n % 8 || p.k % 32 || p.k % C::GROUP) {
    return kBadShape;
  }
  const int groups = p.k / C::GROUP;
  const int split = p.split_k > 0 ? p.split_k : pick_split_k<C>(p.m, p.n, p.k);
  if (split > kMaxSplit || groups % split || (p.out_f32 && split != 1)) {
    return kBadSplit;
  }
  const int k_per_split = p.k / split;
  const int lds = ex::lds_bytes<C>(k_per_split);
  if (lds > 64 * 1024) {
    return kLdsTooBig;
  }
  if (!p.out_f32 && split > 1) {
    const hipError_t e = hipMemsetAsync(
        p.out, 0, static_cast<size_t>(p.m) * p.n * sizeof(ex::f16_t), stream);
    if (e != hipSuccess) {
      return static_cast<int>(e);
    }
  }
  const dim3 grid((p.n + C::N_TILE - 1) / C::N_TILE,
                  (p.m + C::M_TILE - 1) / C::M_TILE, split);
  ex::w4a8_gemm_kernel<C><<<grid, dim3(C::THREADS), lds, stream>>>(
      p.a, p.w, p.qzeros, p.scales, p.a_scale, p.asum, p.out, p.m, p.n, p.k,
      p.zero_offset, k_per_split, split, p.out_f32);
  return static_cast<int>(hipGetLastError());
}

using LaunchFn = int (*)(const GemmArgs&, hipStream_t);
using SplitFn = int (*)(int, int, int);

struct ConfigEntry {
  int id;
  const char* name;
  int m_tile;
  int n_tile;
  int a_group;         // per-(token, group) activation scales
  LaunchFn launch[3];  // group 32, 64, 128
  SplitFn split[3];
};

#define W4A8_CAPI_CFG(th, npt, ks, mt, g, src, ag) \
  ex::Cfg<th, npt, ks, mt, g, ex::ASrc::src, (ag) != 0>
#define W4A8_CAPI_ENTRY(id, name, th, npt, ks, mt, src, ag)      \
  {id,                                                           \
   name,                                                         \
   mt,                                                           \
   (th) * (npt),                                                 \
   ag,                                                           \
   {&launch_gemm<W4A8_CAPI_CFG(th, npt, ks, mt, 32, src, ag)>,   \
    &launch_gemm<W4A8_CAPI_CFG(th, npt, ks, mt, 64, src, ag)>,   \
    &launch_gemm<W4A8_CAPI_CFG(th, npt, ks, mt, 128, src, ag)>}, \
   {&pick_split_k<W4A8_CAPI_CFG(th, npt, ks, mt, 32, src, ag)>,  \
    &pick_split_k<W4A8_CAPI_CFG(th, npt, ks, mt, 64, src, ag)>,  \
    &pick_split_k<W4A8_CAPI_CFG(th, npt, ks, mt, 128, src, ag)>}},

const ConfigEntry kConfigs[] = {W4A8_EXPLORE_CONFIGS(W4A8_CAPI_ENTRY)};
constexpr int kNumConfigs = sizeof(kConfigs) / sizeof(kConfigs[0]);

const ConfigEntry* find_config(int id) {
  for (const ConfigEntry& c : kConfigs) {
    if (c.id == id) {
      return &c;
    }
  }
  return nullptr;
}

template <int MT, bool PER_GROUP>
int launch_act_quant(const void* x, long long x_row_stride, void* a,
                     void* a_scale, void* asum, int m, int k, int group_size,
                     hipStream_t stream) {
  ex::w4a8_act_quant_kernel<256, MT, PER_GROUP>
      <<<dim3((m + MT - 1) / MT), dim3(256), 0, stream>>>(
          static_cast<const ex::f16_t*>(x), x_row_stride,
          static_cast<int8_t*>(a), static_cast<float*>(a_scale),
          static_cast<int32_t*>(asum), m, k, group_size);
  return static_cast<int>(hipGetLastError());
}

}  // namespace

extern "C" {

int w4a8_abi_version() { return 2; }

int w4a8_num_configs() { return kNumConfigs; }

const char* w4a8_config_name(int id) {
  const ConfigEntry* c = find_config(id);
  return c ? c->name : nullptr;
}

int w4a8_config_m_tile(int id) {
  const ConfigEntry* c = find_config(id);
  return c ? c->m_tile : kBadConfig;
}

int w4a8_config_n_tile(int id) {
  const ConfigEntry* c = find_config(id);
  return c ? c->n_tile : kBadConfig;
}

int w4a8_config_a_group(int id) {
  const ConfigEntry* c = find_config(id);
  return c ? c->a_group : kBadConfig;
}

int w4a8_probe_chains() { return kProbeChains; }

const char* w4a8_error_str(int code) {
  switch (code) {
    case 0:
      return "ok";
    case kBadShape:
      return "bad shape (need N % 8 == 0, K % 32 == 0, K % group == 0)";
    case kBadConfig:
      return "unknown config id";
    case kNotGfx1030:
      return "current device is not gfx1030";
    case kLdsTooBig:
      return "K split does not fit 64 KiB of LDS";
    case kBadSplit:
      return "split_k must divide K/group, be <= 16, and be 1 for f32 out";
    case kBadGroup:
      return "group size must be 32, 64 or 128";
    default:
      return code > 0 ? hipGetErrorString(static_cast<hipError_t>(code))
                      : "unknown error";
  }
}

int w4a8_pick_split_k(int m, int n, int k, int group_size, int config_id) {
  const ConfigEntry* c = find_config(config_id);
  const int gi = group_index(group_size);
  if (!c) {
    return kBadConfig;
  }
  if (gi < 0) {
    return kBadGroup;
  }
  if (m <= 0 || n <= 0 || k % group_size) {
    return kBadShape;
  }
  return c->split[gi](m, n, k);
}

// x [M, K] fp16 with row stride x_row_stride (elements, multiple of 8).
// Writes a [T][K/8][m_tile][8] int8, asum [T][K/G][m_tile] int32, and
// a_scale as [M] f32, or [T][K/G][m_tile] f32 when per_group_scale != 0.
int w4a8_act_quant(const void* x, long long x_row_stride, void* a,
                   void* a_scale, void* asum, int m, int k, int group_size,
                   int m_tile, int per_group_scale, void* stream) {
  if (!on_gfx1030()) {
    return kNotGfx1030;
  }
  if (group_index(group_size) < 0) {
    return kBadGroup;
  }
  if (m <= 0 || k % 32 || k % group_size || x_row_stride % 8 ||
      x_row_stride < k) {
    return kBadShape;
  }
  const hipStream_t s = static_cast<hipStream_t>(stream);
  const bool pg = per_group_scale != 0;
  switch (m_tile) {
    case 8:
      return pg ? launch_act_quant<8, true>(x, x_row_stride, a, a_scale, asum,
                                            m, k, group_size, s)
                : launch_act_quant<8, false>(x, x_row_stride, a, a_scale, asum,
                                             m, k, group_size, s);
    case 16:
      return pg ? launch_act_quant<16, true>(x, x_row_stride, a, a_scale, asum,
                                             m, k, group_size, s)
                : launch_act_quant<16, false>(x, x_row_stride, a, a_scale, asum,
                                              m, k, group_size, s);
    case 32:
      return pg ? launch_act_quant<32, true>(x, x_row_stride, a, a_scale, asum,
                                             m, k, group_size, s)
                : launch_act_quant<32, false>(x, x_row_stride, a, a_scale, asum,
                                              m, k, group_size, s);
    default:
      return kBadShape;
  }
}

// split_k <= 0 picks the split with w4a8_pick_split_k. fp16 output with
// split_k > 1 is zero-filled here and accumulated with pk CAS atomics.
// a_scale is per token, or tiled per group when w4a8_config_a_group(id).
int w4a8_gemm(const void* a, const void* w, const void* qzeros,
              const void* scales, const void* a_scale, const void* asum,
              void* out, int m, int n, int k, int group_size, int zero_offset,
              int config_id, int split_k, int out_f32, void* stream) {
  if (!on_gfx1030()) {
    return kNotGfx1030;
  }
  const ConfigEntry* c = find_config(config_id);
  const int gi = group_index(group_size);
  if (!c) {
    return kBadConfig;
  }
  if (gi < 0) {
    return kBadGroup;
  }
  const GemmArgs p{static_cast<const int8_t*>(a),
                   static_cast<const uint32_t*>(w),
                   static_cast<const uint32_t*>(qzeros),
                   static_cast<const ex::f16_t*>(scales),
                   static_cast<const float*>(a_scale),
                   static_cast<const int32_t*>(asum),
                   out,
                   m,
                   n,
                   k,
                   zero_offset,
                   split_k,
                   out_f32};
  return c->launch[gi](p, static_cast<hipStream_t>(stream));
}

// G0 peak probe: kind 0 = v_dot4_i32_i8, 1 = v_dot2_f32_f16, 2 = v_fma_f32.
// Executes blocks * 256 * iters * w4a8_probe_chains() instructions.
int w4a8_probe(int kind, int blocks, int iters, void* out, void* stream) {
  if (!on_gfx1030()) {
    return kNotGfx1030;
  }
  const hipStream_t s = static_cast<hipStream_t>(stream);
  uint32_t* o = static_cast<uint32_t*>(out);
  const dim3 grid(blocks);
  switch (kind) {
    case 0:
      ex::w4a8_probe_kernel<ex::Probe::kSdot4, kProbeChains>
          <<<grid, dim3(256), 0, s>>>(iters, 0x12345678u, o);
      break;
    case 1:
      ex::w4a8_probe_kernel<ex::Probe::kFdot2, kProbeChains>
          <<<grid, dim3(256), 0, s>>>(iters, 0x12345678u, o);
      break;
    case 2:
      ex::w4a8_probe_kernel<ex::Probe::kFmaF32, kProbeChains>
          <<<grid, dim3(256), 0, s>>>(iters, 0x12345678u, o);
      break;
    default:
      return kBadConfig;
  }
  return static_cast<int>(hipGetLastError());
}

}  // extern "C"
