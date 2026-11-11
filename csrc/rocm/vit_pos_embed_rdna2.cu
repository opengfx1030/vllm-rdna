// HIP replacement for the vision-tower bilinear position-embedding
// interpolation Triton kernel on gfx1030 (RDNA2):
//
//   vllm/model_executor/models/qwen3_vl.py::_bilinear_pos_embed_kernel
//   (the same kernel is duplicated in models/cohere_compass.py)
//
// One launch per (t, h, w) grid, multimodal requests only. The spatial-merge
// reorder is folded into the index math, so the result is a drop-in
// replacement for the Triton helper: output row `pid` corresponds to spatial
// index `pid % (h*w)`.
//
// The index/weight arithmetic and the accumulation precision both mirror the
// Triton kernel expression for expression (fp32 index math, weights cast to
// the output dtype, MACs in the output dtype), so the two agree *bit for bit*
// at fp16 -- see tests/kernels/core/test_vit_pos_embed_rdna2.py.
//
// Geometry: one wave (64 lanes) per output row and kRowsPerBlock rows per
// block; each lane strides the row's 16-byte chunks by 64, so a load
// instruction covers a contiguous 1KB window and the row's offsets/weights are
// computed once per lane instead of once per chunk. The previous layout (a
// flat grid over chunks, 2 contiguous chunks per thread) touched a 2KB span
// per instruction with half of every 128B sector left for the sibling
// instruction, and recomputed the per-row index math in every thread: measured
// in-session at 6400x1152 fp16 it was 0.78x Triton (i.e. slower), this one is
// 1.32x. Every served grid with >=1200 rows measures 1.02-1.35x Triton; the
// 4- and 256-row startup-warmup grids stay ~1.5-2us/call behind (they are
// launch-cost dominated, ~8us per server lifetime in total).
//
// Not on the per-decode-step path (one launch per vision item), so a
// per-call allocation is harmless.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <type_traits>

namespace vllm {
namespace rdna2_vit_pos_embed {

constexpr int kThreads = 512;
// 128-bit global accesses; VEC is per-dtype so a pack is always 16 bytes.
constexpr int kVecBytes = 16;
// One wave per output row: 64 lanes x 16B = 1KB contiguous per instruction.
constexpr int kThreadsPerRow = 64;
constexpr int kRowsPerBlock = kThreads / kThreadsPerRow;

template <typename T>
struct alignas(kVecBytes) VecPack {
  T v[kVecBytes / sizeof(T)];
};

// Per-output-row source row offsets and bilinear weights.
struct RowWeights {
  int off00;
  int off01;
  int off10;
  int off11;
  float w00;
  float w01;
  float w10;
  float w11;
};

// RT is the spatial-merge size when the caller knows it at compile time (0 =
// take the runtime argument); a constexpr m_size turns three of the four
// integer divisions below into shifts/masks.
template <int RT>
__device__ __forceinline__ RowWeights row_weights(int spatial_idx, int num_grid,
                                                  int m_size, int num_blocks_w,
                                                  float h_scale,
                                                  float w_scale) {
  const int ms = RT ? RT : m_size;
  const int block_idx = spatial_idx / (ms * ms);
  const int local_idx = spatial_idx % (ms * ms);
  const int br = block_idx / num_blocks_w;
  const int bc = block_idx - br * num_blocks_w;
  const int lr = local_idx / ms;
  const int lc = local_idx - lr * ms;
  const int row = br * ms + lr;
  const int col = bc * ms + lc;

  // row, col >= 0 and both scales are >= 0, so the float->int conversion is
  // floor() (and cheaper).
  const float h_frac = (float)row * h_scale;
  const float w_frac = (float)col * w_scale;
  const int hf = (int)h_frac;
  const int wf = (int)w_frac;
  const int hc = min(hf + 1, num_grid - 1);
  const int wc = min(wf + 1, num_grid - 1);

  const float dh = h_frac - (float)hf;
  const float dw = w_frac - (float)wf;

  RowWeights rw;
  rw.w11 = dh * dw;
  rw.w10 = dh - rw.w11;
  rw.w01 = dw - rw.w11;
  rw.w00 = 1.0f - dh - rw.w01;
  rw.off00 = hf * num_grid + wf;
  rw.off01 = hf * num_grid + wc;
  rw.off10 = hc * num_grid + wf;
  rw.off11 = hc * num_grid + wc;
  return rw;
}

// `c00 * a + c01 * b + c10 * c + c11 * d` in the element type, with the
// weights cast to it first -- the same expression and the same precision as
// the Triton kernel. fp16 goes through packed half2 FMA; the other dtypes use
// the element-type operators (the compiler is free to pack them).
template <typename T>
__device__ __forceinline__ void interp_pack(const VecPack<T>& a,
                                            const VecPack<T>& b,
                                            const VecPack<T>& c,
                                            const VecPack<T>& d,
                                            const RowWeights& rw,
                                            VecPack<T>& r) {
  constexpr int VEC = kVecBytes / (int)sizeof(T);
  if constexpr (std::is_same<T, at::Half>::value) {
    const __half2 c00 = __half2half2((__half)rw.w00);
    const __half2 c01 = __half2half2((__half)rw.w01);
    const __half2 c10 = __half2half2((__half)rw.w10);
    const __half2 c11 = __half2half2((__half)rw.w11);
    const __half2* pa = reinterpret_cast<const __half2*>(a.v);
    const __half2* pb = reinterpret_cast<const __half2*>(b.v);
    const __half2* pc = reinterpret_cast<const __half2*>(c.v);
    const __half2* pd = reinterpret_cast<const __half2*>(d.v);
    __half2* pr = reinterpret_cast<__half2*>(r.v);
#pragma unroll
    for (int i = 0; i < VEC / 2; ++i) {
      __half2 t = __hmul2(c00, pa[i]);
      t = __hfma2(c01, pb[i], t);
      t = __hfma2(c10, pc[i], t);
      t = __hfma2(c11, pd[i], t);
      pr[i] = t;
    }
  } else {
    const T c00 = (T)rw.w00;
    const T c01 = (T)rw.w01;
    const T c10 = (T)rw.w10;
    const T c11 = (T)rw.w11;
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      r.v[i] = c00 * a.v[i] + c01 * b.v[i] + c10 * c.v[i] + c11 * d.v[i];
    }
  }
}

// Requires hidden_dim % VEC == 0 and 16B-aligned pointers. Lane `l` of a row
// covers chunks l, l + kThreadsPerRow, ... of that row.
template <typename T, int RT>
__global__ void bilinear_pos_embed_row_kernel(const T* __restrict__ embed,
                                              T* __restrict__ out, int num_grid,
                                              int m_size, int num_blocks_w,
                                              int hidden_dim, int num_rows,
                                              int total_spatial,
                                              float h_scale, float w_scale) {
  constexpr int VEC = kVecBytes / (int)sizeof(T);
  const int nvec = hidden_dim / VEC;
  const int pid = blockIdx.x * kRowsPerBlock + threadIdx.x / kThreadsPerRow;
  if (pid >= num_rows) {
    return;
  }
  const int lane = threadIdx.x % kThreadsPerRow;
  // t > 1 repeats the same spatial pattern, so only the first h*w rows need
  // the modulo.
  const int sidx = (num_rows != total_spatial) ? pid % total_spatial : pid;
  const RowWeights rw = row_weights<RT>(sidx, num_grid, m_size, num_blocks_w,
                                        h_scale, w_scale);

  const T* __restrict__ e00 = embed + (int64_t)rw.off00 * hidden_dim;
  const T* __restrict__ e01 = embed + (int64_t)rw.off01 * hidden_dim;
  const T* __restrict__ e10 = embed + (int64_t)rw.off10 * hidden_dim;
  const T* __restrict__ e11 = embed + (int64_t)rw.off11 * hidden_dim;
  T* __restrict__ o = out + (int64_t)pid * hidden_dim;

  for (int c = lane; c < nvec; c += kThreadsPerRow) {
    const int64_t off = (int64_t)c * VEC;
    VecPack<T> a, b, cc, d, r;
    __builtin_memcpy(&a, e00 + off, sizeof(VecPack<T>));
    __builtin_memcpy(&b, e01 + off, sizeof(VecPack<T>));
    __builtin_memcpy(&cc, e10 + off, sizeof(VecPack<T>));
    __builtin_memcpy(&d, e11 + off, sizeof(VecPack<T>));
    interp_pack<T>(a, b, cc, d, rw, r);
    __builtin_memcpy(o + off, &r, sizeof(VecPack<T>));
  }
}

// Scalar fallback for hidden_dim not a multiple of VEC or misaligned pointers.
template <typename T>
__global__ void bilinear_pos_embed_kernel(const T* __restrict__ embed,
                                          T* __restrict__ out, int num_grid,
                                          int m_size, int num_blocks_w,
                                          int hidden_dim, int total_spatial,
                                          float h_scale, float w_scale) {
  const int pid = blockIdx.x;
  const int sidx = (gridDim.x != (unsigned)total_spatial) ? pid % total_spatial
                                                          : pid;
  const RowWeights rw = row_weights<0>(sidx, num_grid, m_size, num_blocks_w,
                                       h_scale, w_scale);

  const T* __restrict__ e00 = embed + (int64_t)rw.off00 * hidden_dim;
  const T* __restrict__ e01 = embed + (int64_t)rw.off01 * hidden_dim;
  const T* __restrict__ e10 = embed + (int64_t)rw.off10 * hidden_dim;
  const T* __restrict__ e11 = embed + (int64_t)rw.off11 * hidden_dim;
  T* __restrict__ o = out + (int64_t)pid * hidden_dim;

  const T c00 = (T)rw.w00;
  const T c01 = (T)rw.w01;
  const T c10 = (T)rw.w10;
  const T c11 = (T)rw.w11;
  for (int d = threadIdx.x; d < hidden_dim; d += blockDim.x) {
    o[d] = c00 * e00[d] + c01 * e01[d] + c10 * e10[d] + c11 * e11[d];
  }
}

template <typename T, int RT>
void launch_rt(const T* embed, T* out, int64_t num_rows, int H, int W,
               float h_scale, float w_scale, int num_grid, int m_size,
               int hidden_dim, hipStream_t stream) {
  const unsigned int blocks =
      (unsigned int)((num_rows + kRowsPerBlock - 1) / kRowsPerBlock);
  bilinear_pos_embed_row_kernel<T, RT>
      <<<blocks, kThreads, 0, stream>>>(embed, out, num_grid, m_size,
                                        W / m_size, hidden_dim, (int)num_rows,
                                        H * W, h_scale, w_scale);
}

template <typename T>
void launch(const T* embed, T* out, int64_t num_rows, int H, int W,
            float h_scale, float w_scale, int num_grid, int m_size,
            int hidden_dim, hipStream_t stream) {
  constexpr int VEC = kVecBytes / (int)sizeof(T);
  const bool aligned = (reinterpret_cast<uintptr_t>(embed) % kVecBytes == 0) &&
                       (reinterpret_cast<uintptr_t>(out) % kVecBytes == 0);
  if (!aligned || hidden_dim % VEC != 0) {
    bilinear_pos_embed_kernel<T><<<(unsigned int)num_rows, kThreads, 0,
                                   stream>>>(embed, out, num_grid, m_size,
                                             W / m_size, hidden_dim, H * W,
                                             h_scale, w_scale);
    return;
  }
  // Spatial merge is 2 on the served vision towers; specializing it makes the
  // block/local index split inside row_weights a shift and a mask.
  if (m_size == 1) {
    launch_rt<T, 1>(embed, out, num_rows, H, W, h_scale, w_scale, num_grid,
                    m_size, hidden_dim, stream);
  } else if (m_size == 2) {
    launch_rt<T, 2>(embed, out, num_rows, H, W, h_scale, w_scale, num_grid,
                    m_size, hidden_dim, stream);
  } else if (m_size == 4) {
    launch_rt<T, 4>(embed, out, num_rows, H, W, h_scale, w_scale, num_grid,
                    m_size, hidden_dim, stream);
  } else {
    launch_rt<T, 0>(embed, out, num_rows, H, W, h_scale, w_scale, num_grid,
                    m_size, hidden_dim, stream);
  }
}

}  // namespace rdna2_vit_pos_embed
}  // namespace vllm

torch::Tensor bilinear_pos_embed_rdna2(torch::Tensor embed_weight,
                                       torch::Tensor out, int64_t H, int64_t W,
                                       double h_scale, double w_scale,
                                       int64_t num_grid, int64_t m_size) {
  using namespace vllm::rdna2_vit_pos_embed;
  TORCH_CHECK(embed_weight.dim() == 2, "embed_weight must be 2-D");
  TORCH_CHECK(embed_weight.is_contiguous(), "embed_weight must be contiguous");
  TORCH_CHECK(out.dim() == 2 && out.is_contiguous(),
              "out must be 2-D and contiguous");
  TORCH_CHECK(embed_weight.scalar_type() == out.scalar_type(),
              "embed_weight and out must share a dtype");
  TORCH_CHECK(out.size(1) == embed_weight.size(1), "hidden_dim mismatch");
  TORCH_CHECK(out.size(0) % (H * W) == 0,
              "out rows must be a multiple of H*W");
  TORCH_CHECK(H > 0 && W > 0 && m_size > 0 && H % m_size == 0 &&
                  W % m_size == 0,
              "H and W must be positive multiples of m_size");
  TORCH_CHECK(num_grid >= 1, "num_grid must be >= 1");

  const int64_t num_rows = out.size(0);
  if (num_rows == 0) {
    return out;
  TORCH_CHECK(num_rows <= INT32_MAX && out.size(1) <= INT32_MAX,
              "bilinear_pos_embed_rdna2: rows/hidden_dim must fit an int32");
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(out));
  auto stream = at::cuda::getCurrentCUDAStream();

  const int hidden_dim = (int)out.size(1);
  const int grid = (int)num_grid;
  const int ms = (int)m_size;
  const float hs = (float)h_scale;
  const float ws = (float)w_scale;
  const int hh = (int)H;
  const int ww = (int)W;
  auto* p = stream.stream();

  switch (out.scalar_type()) {
    case at::kHalf:
      launch<at::Half>(
          reinterpret_cast<const at::Half*>(embed_weight.data_ptr()),
          reinterpret_cast<at::Half*>(out.data_ptr()), num_rows, hh, ww, hs, ws,
          grid, ms, hidden_dim, p);
      break;
    case at::kBFloat16:
      launch<at::BFloat16>(
          reinterpret_cast<const at::BFloat16*>(embed_weight.data_ptr()),
          reinterpret_cast<at::BFloat16*>(out.data_ptr()), num_rows, hh, ww, hs,
          ws, grid, ms, hidden_dim, p);
      break;
    case at::kFloat:
      launch<float>(reinterpret_cast<const float*>(embed_weight.data_ptr()),
                    reinterpret_cast<float*>(out.data_ptr()), num_rows, hh, ww,
                    hs, ws, grid, ms, hidden_dim, p);
      break;
    default:
      TORCH_CHECK(false, "bilinear_pos_embed_rdna2: unsupported dtype ",
                  out.scalar_type());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
