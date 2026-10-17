// T47 — HyperConnection (HC) glue for Qwen3.8-Flash-Next on gfx1030.
//
// Five elementwise/affine kernels used by the Qwen4Exp HC path. All are
// bandwidth-bound, so each row is spread across the whole thread block and the
// inner loop moves 8 fp16 at a time (one 16-byte uint4). The two RMSNorm
// sum-of-squares reductions use a warp shuffle tree plus one shared round.
// The maths are taken verbatim from the matching Triton kernels in
// vllm/models/qwen4_exp/amd/ops/hc.py.
//
//   grouped_gemma_rmsnorm_rdna2
//       y = x * rsqrt(sum(x*x) / GROUP_DIM + eps) * (1 + w)
//       shared weight when weight.numel() == GROUP_DIM (W_SHARED); per-stream
//       weight when weight.numel() == DIM (mirrors the Triton ``W_SHARED``).
//
//   hc_silu_rdna2
//       y = (x / HC) * sigmoid(x / HC)           [HC residual scaler]
//
//   hc_gate_mix_rdna2
//       out[h] = (1/HC) * sum_c sigmoid(g[c*H+h]) * x[c*H+h]
//
//   hc_combine_rdna2
//       res[c, h] += block[h] * 2 * sigmoid(inj[c] / HC)
//
//   hc_combine_norm_rdna2
//       fused combine + grouped Gemma RMSNorm: writes ``out`` (combined
//       residual) AND ``y`` (post-norm) in a single pass; matches the
//       unfused combine -> RMSNorm boundary that the model relies on
//       when the next HC mix is fused into a linear projection.
//
// Opt-in: the Python dispatcher in vllm/models/qwen4_exp/amd/ops/hc.py
// gates these on VLLM_RDNA_HC_PREFILL_HIP=1 AND on_gfx10x(). Default is
// off (Triton path).

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {

constexpr int kThreads = 256;

// ---------------------------------------------------------------------------
// Vector helpers: 8 fp16 = 16 bytes = one uint4
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint4 ld_u4(const void* p) {
  return *reinterpret_cast<const uint4*>(p);
}
__device__ __forceinline__ void st_u4(void* p, uint4 v) {
  *reinterpret_cast<uint4*>(p) = v;
}

// Block-wide float sum, broadcast to every thread.
__device__ __forceinline__ float block_sum_bcast(float v) {
  __shared__ float smem[32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down(v, o);
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  const int nwarps = (blockDim.x + 31) >> 5;
  if (warp == 0) {
    v = (lane < nwarps) ? smem[lane] : 0.f;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down(v, o);
    if (lane == 0) smem[0] = v;
  }
  __syncthreads();
  return smem[0];
}

// vec8 is only legal when the base pointers and the row/group strides are all
// 16-byte aligned; the host wrappers fall back to the scalar instantiation.
static inline bool aligned16(const void* p) {
  return (reinterpret_cast<unsigned long long>(p) & 15ULL) == 0ULL;
}

// ---------------------------------------------------------------------------
// grouped_gemma_rmsnorm_rdna2
//   x: [N, DIM] fp16, w: [GROUP_DIM] or [DIM] fp16, y: [N, DIM] fp16
// ---------------------------------------------------------------------------

template <bool VEC>
__global__ void grouped_gemma_rmsnorm_kernel(const half* __restrict__ x,
                                             const half* __restrict__ w,
                                             half* __restrict__ y, int DIM,
                                             int NUM_GROUPS, int GROUP_DIM,
                                             int W_SHARED, float EPS) {
  const int pid = blockIdx.x;
  const int group_id = pid % NUM_GROUPS;
  const int row = pid / NUM_GROUPS;

  const half* x_row = x + (size_t)row * DIM + (size_t)group_id * GROUP_DIM;
  half* y_row = y + (size_t)row * DIM + (size_t)group_id * GROUP_DIM;
  const half* w_row = W_SHARED ? w : (w + (size_t)group_id * GROUP_DIM);

  const int step = blockDim.x * 8;
  float ss = 0.f;
  for (int b = threadIdx.x * 8; b < GROUP_DIM; b += step) {
    if (VEC && b + 8 <= GROUP_DIM) {
      uint4 v = ld_u4(x_row + b);
      const half2* h = reinterpret_cast<const half2*>(&v);
#pragma unroll
      for (int u = 0; u < 4; u++) {
        float2 f = __half22float2(h[u]);
        ss += f.x * f.x + f.y * f.y;
      }
    } else {
      for (int i = b; i < GROUP_DIM && i < b + 8; i++) {
        float f = __half2float(x_row[i]);
        ss += f * f;
      }
    }
  }
  const float rrms = rsqrtf(block_sum_bcast(ss) / float(GROUP_DIM) + EPS);

  // y = x * rrms * (1 + w) == x * rrms + (x * rrms) * w
  for (int b = threadIdx.x * 8; b < GROUP_DIM; b += step) {
    if (VEC && b + 8 <= GROUP_DIM) {
      uint4 xv = ld_u4(x_row + b);
      uint4 wv = ld_u4(w_row + b);
      const half2* xh = reinterpret_cast<const half2*>(&xv);
      const half2* wh = reinterpret_cast<const half2*>(&wv);
      uint4 yv;
      half2* yh = reinterpret_cast<half2*>(&yv);
#pragma unroll
      for (int u = 0; u < 4; u++) {
        float2 xf = __half22float2(xh[u]);
        float2 wf = __half22float2(wh[u]);
        float lo = xf.x * rrms;
        float hi = xf.y * rrms;
        yh[u] = __floats2half2_rn(lo + lo * wf.x, hi + hi * wf.y);
      }
      st_u4(y_row + b, yv);
    } else {
      for (int i = b; i < GROUP_DIM && i < b + 8; i++) {
        float v = __half2float(x_row[i]) * rrms;
        y_row[i] = __float2half(v + v * __half2float(w_row[i]));
      }
    }
  }
}

// Single-pass variant: the host only picks it when one vec8 per thread covers
// the whole group (GROUP_DIM / 8 threads, a multiple of 32), so the row stays
// in registers across the block reduction and there is no second read.
// Single-pass variant: the host picks it when one vec8 per thread (K = 1) or
// two (K = 2) cover the whole group with a block size that is a multiple of 32,
// so the row stays in registers across the block reduction and there is no
// second read.  K = 2 halves the number of warps in the reduction, which is
// what the wider rows are actually limited by.
template <int K>
__global__ void __launch_bounds__(1024, 1) grouped_gemma_rmsnorm_onepass_kernel(
    const half* __restrict__ x, const half* __restrict__ w,
    half* __restrict__ y, int DIM, int NUM_GROUPS, int GROUP_DIM, int W_SHARED,
    float EPS) {
  const int pid = blockIdx.x;
  const int group_id = pid % NUM_GROUPS;
  const int row = pid / NUM_GROUPS;

  const half* x_row = x + (size_t)row * DIM + (size_t)group_id * GROUP_DIM;
  half* y_row = y + (size_t)row * DIM + (size_t)group_id * GROUP_DIM;
  const half* w_row = W_SHARED ? w : (w + (size_t)group_id * GROUP_DIM);

  uint4 xv[K];
  uint4 wv[K];
  float ss = 0.f;
#pragma unroll
  for (int t = 0; t < K; t++) {
    const int b = (threadIdx.x + t * (int)blockDim.x) * 8;
    xv[t] = ld_u4(x_row + b);
    wv[t] = ld_u4(w_row + b);
    const half2* xh = reinterpret_cast<const half2*>(&xv[t]);
#pragma unroll
    for (int u = 0; u < 4; u++) {
      const float2 f = __half22float2(xh[u]);
      ss += f.x * f.x + f.y * f.y;
    }
  }
  const float rrms = rsqrtf(block_sum_bcast(ss) / float(GROUP_DIM) + EPS);

#pragma unroll
  for (int t = 0; t < K; t++) {
    const int b = (threadIdx.x + t * (int)blockDim.x) * 8;
    const half2* xh = reinterpret_cast<const half2*>(&xv[t]);
    const half2* wh = reinterpret_cast<const half2*>(&wv[t]);
    uint4 yv;
    half2* yh = reinterpret_cast<half2*>(&yv);
#pragma unroll
    for (int u = 0; u < 4; u++) {
      const float2 xf = __half22float2(xh[u]);
      const float2 wf = __half22float2(wh[u]);
      const float lo = xf.x * rrms;
      const float hi = xf.y * rrms;
      yh[u] = __floats2half2_rn(lo + lo * wf.x, hi + hi * wf.y);
    }
    st_u4(y_row + b, yv);
  }
}

void grouped_gemma_rmsnorm(const at::Tensor& x, const at::Tensor& weight,
                           at::Tensor& y, int64_t num_groups, double eps) {
  const int N = x.size(0);
  const int DIM = x.size(1);
  TORCH_CHECK(x.scalar_type() == at::kHalf && x.is_contiguous(),
              "grouped_gemma_rmsnorm_rdna2: x must be contiguous fp16");
  TORCH_CHECK(y.scalar_type() == at::kHalf && y.is_contiguous(),
              "grouped_gemma_rmsnorm_rdna2: y must be contiguous fp16");
  TORCH_CHECK(weight.scalar_type() == at::kHalf && weight.is_contiguous(),
              "grouped_gemma_rmsnorm_rdna2: weight must be contiguous fp16");
  TORCH_CHECK(DIM % num_groups == 0, "DIM must be divisible by num_groups");
  const int GROUP_DIM = DIM / num_groups;
  TORCH_CHECK(weight.numel() == GROUP_DIM || weight.numel() == DIM,
              "weight.numel() must equal GROUP_DIM or DIM");
  const int W_SHARED = (weight.numel() == GROUP_DIM) ? 1 : 0;
  const at::cuda::OptionalCUDAGuard guard(x.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const half* xp = reinterpret_cast<const half*>(x.const_data_ptr());
  const half* wp = reinterpret_cast<const half*>(weight.const_data_ptr());
  half* yp = reinterpret_cast<half*>(y.mutable_data_ptr());
  const bool vec = aligned16(xp) && aligned16(wp) && aligned16(yp) &&
                   (DIM % 8 == 0) && (GROUP_DIM % 8 == 0);
  const dim3 grid((unsigned)(N * num_groups));
  // One vec8 per thread needs GROUP_DIM / 8 slots; K = 2 halves the block (and
  // therefore the number of warps in the block reduction) while keeping the
  // block a multiple of 32, which the shuffle reduction needs.
  const int slots = GROUP_DIM / 8;
  const int k = (slots % 64 == 0) ? 2 : 1;
  const int threads = slots / k;
  if (vec && slots % 32 == 0 && threads >= 32 && threads <= 1024) {
    if (k == 2) {
      grouped_gemma_rmsnorm_onepass_kernel<2><<<grid, (unsigned)threads, 0,
                                                stream>>>(
          xp, wp, yp, DIM, (int)num_groups, GROUP_DIM, W_SHARED, (float)eps);
    } else {
      grouped_gemma_rmsnorm_onepass_kernel<1><<<grid, (unsigned)threads, 0,
                                                stream>>>(
          xp, wp, yp, DIM, (int)num_groups, GROUP_DIM, W_SHARED, (float)eps);
    }
  } else if (vec) {
    grouped_gemma_rmsnorm_kernel<true><<<grid, kThreads, 0, stream>>>(
        xp, wp, yp, DIM, (int)num_groups, GROUP_DIM, W_SHARED, (float)eps);
  } else {
    grouped_gemma_rmsnorm_kernel<false><<<grid, kThreads, 0, stream>>>(
        xp, wp, yp, DIM, (int)num_groups, GROUP_DIM, W_SHARED, (float)eps);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// hc_silu_rdna2: y = (x / HC) * sigmoid(x / HC), flat over N * DIM
// ---------------------------------------------------------------------------

template <bool VEC>
__global__ void hc_silu_kernel(const half* __restrict__ x, half* __restrict__ y,
                               long total, float inv_hc) {
  const long b = ((long)blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (VEC && b + 8 <= total) {
    uint4 xv = ld_u4(x + b);
    const half2* xh = reinterpret_cast<const half2*>(&xv);
    uint4 yv;
    half2* yh = reinterpret_cast<half2*>(&yv);
#pragma unroll
    for (int u = 0; u < 4; u++) {
      float2 f = __half22float2(xh[u]);
      f.x *= inv_hc;
      f.y *= inv_hc;
      float sx = 1.f / (1.f + __expf(-f.x));
      float sy = 1.f / (1.f + __expf(-f.y));
      yh[u] = __floats2half2_rn(f.x * sx, f.y * sy);
    }
    st_u4(y + b, yv);
  } else if (b < total) {
    for (long i = b; i < total && i < b + 8; i++) {
      float v = __half2float(x[i]) * inv_hc;
      float s = 1.f / (1.f + __expf(-v));
      y[i] = __float2half(v * s);
    }
  }
}

void hc_silu(const at::Tensor& x, at::Tensor& y, int64_t hc_count) {
  const long total = (long)x.numel();
  TORCH_CHECK(x.scalar_type() == at::kHalf && x.is_contiguous(),
              "hc_silu_rdna2: x must be contiguous fp16");
  TORCH_CHECK(y.scalar_type() == at::kHalf && y.is_contiguous(),
              "hc_silu_rdna2: y must be contiguous fp16");
  const at::cuda::OptionalCUDAGuard guard(x.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const half* xp = reinterpret_cast<const half*>(x.const_data_ptr());
  half* yp = reinterpret_cast<half*>(y.mutable_data_ptr());
  const long groups = (total + 7) / 8;
  const dim3 grid((unsigned)((groups + kThreads - 1) / kThreads));
  const float inv_hc = 1.f / float(hc_count);
  if (aligned16(xp) && aligned16(yp)) {
    hc_silu_kernel<true><<<grid, kThreads, 0, stream>>>(xp, yp, total, inv_hc);
  } else {
    hc_silu_kernel<false><<<grid, kThreads, 0, stream>>>(xp, yp, total, inv_hc);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// hc_gate_mix_rdna2: out[h] = (1/HC) * sum_c sigmoid(g[c*H+h]) * x[c*H+h]
//   x, g: [N, DIM] fp16, out: [N, DIM/HC] fp16
// ---------------------------------------------------------------------------

template <bool VEC>
__global__ void hc_gate_mix_kernel(const half* __restrict__ x,
                                   const half* __restrict__ g,
                                   half* __restrict__ y, int HC_DIM, int HC,
                                   float inv_hc) {
  const int base = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (base >= HC_DIM) return;
  const long row = blockIdx.y;
  const half* xr = x + row * (long)HC_DIM * HC;
  const half* gr = g + row * (long)HC_DIM * HC;
  if (VEC && base + 8 <= HC_DIM) {
    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = 0.f;
    for (int c = 0; c < HC; c++) {
      const long off = (long)c * HC_DIM + base;
      uint4 xv = ld_u4(xr + off);
      uint4 gv = ld_u4(gr + off);
      const half2* xh = reinterpret_cast<const half2*>(&xv);
      const half2* gh = reinterpret_cast<const half2*>(&gv);
#pragma unroll
      for (int u = 0; u < 4; u++) {
        float2 xf = __half22float2(xh[u]);
        float2 gf = __half22float2(gh[u]);
        acc[2 * u] += (1.f / (1.f + __expf(-gf.x))) * xf.x;
        acc[2 * u + 1] += (1.f / (1.f + __expf(-gf.y))) * xf.y;
      }
    }
    uint4 ov;
    half2* oh = reinterpret_cast<half2*>(&ov);
#pragma unroll
    for (int u = 0; u < 4; u++) {
      oh[u] = __floats2half2_rn(acc[2 * u] * inv_hc, acc[2 * u + 1] * inv_hc);
    }
    st_u4(y + row * (long)HC_DIM + base, ov);
  } else {
    for (int i = base; i < HC_DIM && i < base + 8; i++) {
      float acc = 0.f;
      for (int c = 0; c < HC; c++) {
        const long off = (long)c * HC_DIM + i;
        acc += (1.f / (1.f + __expf(-__half2float(gr[off])))) *
               __half2float(xr[off]);
      }
      y[row * (long)HC_DIM + i] = __float2half(acc * inv_hc);
    }
  }
}

void hc_gate_mix(const at::Tensor& x, const at::Tensor& gate, at::Tensor& y,
                 int64_t hc_count) {
  const int N = x.size(0);
  const int DIM = x.size(1);
  TORCH_CHECK(x.scalar_type() == at::kHalf && x.is_contiguous(), "x");
  TORCH_CHECK(gate.scalar_type() == at::kHalf && gate.is_contiguous(), "gate");
  TORCH_CHECK(y.scalar_type() == at::kHalf && y.is_contiguous(), "y");
  TORCH_CHECK(DIM % hc_count == 0, "DIM must be divisible by hc_count");
  const int HC_DIM = DIM / hc_count;
  const at::cuda::OptionalCUDAGuard guard(x.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const half* xp = reinterpret_cast<const half*>(x.const_data_ptr());
  const half* gp = reinterpret_cast<const half*>(gate.const_data_ptr());
  half* yp = reinterpret_cast<half*>(y.mutable_data_ptr());
  const bool vec = aligned16(xp) && aligned16(gp) && aligned16(yp) &&
                   (HC_DIM % 8 == 0);
  const dim3 grid((unsigned)((HC_DIM + kThreads * 8 - 1) / (kThreads * 8)),
                  (unsigned)N);
  const float inv_hc = 1.f / float(hc_count);
  if (vec) {
    hc_gate_mix_kernel<true><<<grid, kThreads, 0, stream>>>(
        xp, gp, yp, HC_DIM, (int)hc_count, inv_hc);
  } else {
    hc_gate_mix_kernel<false><<<grid, kThreads, 0, stream>>>(
        xp, gp, yp, HC_DIM, (int)hc_count, inv_hc);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
// ---------------------------------------------------------------------------
// hc_combine_rdna2:
//   res[c, h] = res[c, h] + block[h] * 2 * sigmoid(inj[c] / HC)
//   res: [N, DIM] fp16, block: [N, DIM/HC] fp16, inj: [N, HC] fp16
// ---------------------------------------------------------------------------

template <bool VEC>
__global__ void hc_combine_kernel(const half* __restrict__ block_in,
                                  const half* __restrict__ res_in,
                                  const half* __restrict__ inj_in,
                                  half* __restrict__ out, int HC_DIM, int HC,
                                  float inv_hc) {
  const int base = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (base >= HC_DIM) return;
  const long row = blockIdx.y;
  const half* jr = inj_in + row * HC;
  const half* br = block_in + row * (long)HC_DIM;
  const half* rrow = res_in + row * (long)HC_DIM * HC;
  half* orow = out + row * (long)HC_DIM * HC;
  if (VEC && base + 8 <= HC_DIM) {
    float bf[8];
    uint4 bvv = ld_u4(br + base);
    const half2* bh = reinterpret_cast<const half2*>(&bvv);
#pragma unroll
    for (int u = 0; u < 4; u++) {
      float2 t = __half22float2(bh[u]);
      bf[2 * u] = t.x;
      bf[2 * u + 1] = t.y;
    }
    for (int c = 0; c < HC; c++) {
      const float v = __half2float(jr[c]) * inv_hc;
      const float s = 2.f / (1.f + __expf(-v));
      const long off = (long)c * HC_DIM + base;
      uint4 rv = ld_u4(rrow + off);
      const half2* rh = reinterpret_cast<const half2*>(&rv);
      uint4 ov;
      half2* oh = reinterpret_cast<half2*>(&ov);
#pragma unroll
      for (int u = 0; u < 4; u++) {
        float2 rf = __half22float2(rh[u]);
        oh[u] = __floats2half2_rn(rf.x + bf[2 * u] * s,
                                  rf.y + bf[2 * u + 1] * s);
      }
      st_u4(orow + off, ov);
    }
  } else {
    for (int i = base; i < HC_DIM && i < base + 8; i++) {
      const float bv = __half2float(br[i]);
      for (int c = 0; c < HC; c++) {
        const float v = __half2float(jr[c]) * inv_hc;
        const float s = 2.f / (1.f + __expf(-v));
        const long off = (long)c * HC_DIM + i;
        orow[off] = __float2half(__half2float(rrow[off]) + bv * s);
      }
    }
  }
}

void hc_combine(const at::Tensor& residual, const at::Tensor& block_output,
                const at::Tensor& inj, at::Tensor& out, int64_t hc_count) {
  const int N = residual.size(0);
  const int DIM = residual.size(1);
  TORCH_CHECK(residual.scalar_type() == at::kHalf && residual.is_contiguous(),
              "residual");
  TORCH_CHECK(
      block_output.scalar_type() == at::kHalf && block_output.is_contiguous(),
      "block_output");
  TORCH_CHECK(inj.scalar_type() == at::kHalf && inj.is_contiguous(),
              "injection_logits");
  TORCH_CHECK(out.scalar_type() == at::kHalf && out.is_contiguous(), "out");
  TORCH_CHECK(DIM % hc_count == 0, "DIM must be divisible by hc_count");
  const int HC_DIM = DIM / hc_count;
  const at::cuda::OptionalCUDAGuard guard(residual.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const half* bp = reinterpret_cast<const half*>(block_output.const_data_ptr());
  const half* rp = reinterpret_cast<const half*>(residual.const_data_ptr());
  const half* jp = reinterpret_cast<const half*>(inj.const_data_ptr());
  half* op = reinterpret_cast<half*>(out.mutable_data_ptr());
  const bool vec = aligned16(bp) && aligned16(rp) && aligned16(jp) &&
                   aligned16(op) && (HC_DIM % 8 == 0);
  const dim3 grid((unsigned)((HC_DIM + kThreads * 8 - 1) / (kThreads * 8)),
                  (unsigned)N);
  const float inv_hc = 1.f / float(hc_count);
  if (vec) {
    hc_combine_kernel<true><<<grid, kThreads, 0, stream>>>(
        bp, rp, jp, op, HC_DIM, (int)hc_count, inv_hc);
  } else {
    hc_combine_kernel<false><<<grid, kThreads, 0, stream>>>(
        bp, rp, jp, op, HC_DIM, (int)hc_count, inv_hc);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// hc_combine_norm_rdna2:
//   For each stream c and row:
//     inj_scale[c] = 2 * sigmoid(inj[c] / HC)
//     out[row, c, h] = round_to_dtype(res[row, c, h] + block[h] * inj_scale[c])
//     y[row, c, h]   = norm(out[row, c, h])  with shared/per-stream weight
// ---------------------------------------------------------------------------

template <bool VEC>
__global__ void hc_combine_norm_kernel(const half* __restrict__ block_in,
                                       const half* __restrict__ res_in,
                                       const half* __restrict__ inj_in,
                                       const half* __restrict__ w_in,
                                       half* __restrict__ out,
                                       half* __restrict__ y, int HC_DIM, int HC,
                                       int W_SHARED, float EPS) {
  const long row = blockIdx.x;
  const int stream = blockIdx.y;
  const long hc_dim = HC_DIM;

  const half* br = block_in + row * hc_dim;
  const size_t base = (size_t)row * hc_dim * HC + (size_t)stream * hc_dim;
  const half* rr = res_in + base;
  half* orow = out + base;
  half* yrow = y + base;
  const half* w_row = W_SHARED ? w_in : (w_in + (size_t)stream * hc_dim);

  const float iv = __half2float(inj_in[row * HC + stream]) / float(HC);
  const float scale = 2.f / (1.f + __expf(-iv));

  const int step = blockDim.x * 8;
  float ss = 0.f;
  for (int b = threadIdx.x * 8; b < HC_DIM; b += step) {
    if (VEC && b + 8 <= HC_DIM) {
      uint4 bv = ld_u4(br + b);
      uint4 rv = ld_u4(rr + b);
      const half2* bh = reinterpret_cast<const half2*>(&bv);
      const half2* rh = reinterpret_cast<const half2*>(&rv);
      uint4 ov;
      half2* oh = reinterpret_cast<half2*>(&ov);
#pragma unroll
      for (int u = 0; u < 4; u++) {
        float2 bf = __half22float2(bh[u]);
        float2 rf = __half22float2(rh[u]);
        // Round to fp16 to match the unfused combine -> RMSNorm boundary.
        half2 o = __floats2half2_rn(rf.x + bf.x * scale, rf.y + bf.y * scale);
        oh[u] = o;
        float2 of = __half22float2(o);
        ss += of.x * of.x + of.y * of.y;
      }
      st_u4(orow + b, ov);
    } else {
      for (int i = b; i < HC_DIM && i < b + 8; i++) {
        half o = __float2half(__half2float(rr[i]) +
                              __half2float(br[i]) * scale);
        orow[i] = o;
        float of = __half2float(o);
        ss += of * of;
      }
    }
  }
  const float rrms = rsqrtf(block_sum_bcast(ss) / float(HC_DIM) + EPS);

  // y = out * rrms * (1 + w)
  for (int b = threadIdx.x * 8; b < HC_DIM; b += step) {
    if (VEC && b + 8 <= HC_DIM) {
      uint4 ov = ld_u4(orow + b);
      uint4 wv = ld_u4(w_row + b);
      const half2* oh = reinterpret_cast<const half2*>(&ov);
      const half2* wh = reinterpret_cast<const half2*>(&wv);
      uint4 yv;
      half2* yh = reinterpret_cast<half2*>(&yv);
#pragma unroll
      for (int u = 0; u < 4; u++) {
        float2 of = __half22float2(oh[u]);
        float2 wf = __half22float2(wh[u]);
        float lo = of.x * rrms;
        float hi = of.y * rrms;
        yh[u] = __floats2half2_rn(lo + lo * wf.x, hi + hi * wf.y);
      }
      st_u4(yrow + b, yv);
    } else {
      for (int i = b; i < HC_DIM && i < b + 8; i++) {
        float v = __half2float(orow[i]) * rrms;
        yrow[i] = __float2half(v + v * __half2float(w_row[i]));
      }
    }
  }
}

// Single-pass variant of hc_combine_norm_kernel: one vec8 per thread covers the
// whole stream, so the rounded `out` values stay in registers across the block
// reduction instead of being re-read from memory.
// Single-pass variant of hc_combine_norm_kernel: K vec8 per thread cover the
// whole stream, so the rounded `out` values stay in registers across the block
// reduction instead of being re-read from memory.  K = 2 also halves the number
// of warps in the reduction (80 threads is not a multiple of 32, so K = 4 is
// not used).
template <int K>
__global__ void __launch_bounds__(1024, 1) hc_combine_norm_onepass_kernel(
    const half* __restrict__ block_in, const half* __restrict__ res_in,
    const half* __restrict__ inj_in, const half* __restrict__ w_in,
    half* __restrict__ out, half* __restrict__ y, int HC_DIM, int HC,
    int W_SHARED, float EPS) {
  const long row = blockIdx.x;
  const int stream = blockIdx.y;

  const half* br = block_in + row * (long)HC_DIM;
  const size_t base = (size_t)row * HC_DIM * HC + (size_t)stream * HC_DIM;
  const half* rr = res_in + base;
  half* orow = out + base;
  half* yrow = y + base;
  const half* w_row = W_SHARED ? w_in : (w_in + (size_t)stream * HC_DIM);

  const float iv = __half2float(inj_in[row * HC + stream]) / float(HC);
  const float scale = 2.f / (1.f + __expf(-iv));

  uint4 ov[K];
  uint4 wv[K];
  float ss = 0.f;
#pragma unroll
  for (int t = 0; t < K; t++) {
    const int b = (threadIdx.x + t * (int)blockDim.x) * 8;
    const uint4 bv = ld_u4(br + b);
    const uint4 rv = ld_u4(rr + b);
    const half2* bh = reinterpret_cast<const half2*>(&bv);
    const half2* rh = reinterpret_cast<const half2*>(&rv);
    half2* oh = reinterpret_cast<half2*>(&ov[t]);
#pragma unroll
    for (int u = 0; u < 4; u++) {
      const float2 bf = __half22float2(bh[u]);
      const float2 rf = __half22float2(rh[u]);
      // Round to fp16 to match the unfused combine -> RMSNorm boundary.
      const half2 o =
          __floats2half2_rn(rf.x + bf.x * scale, rf.y + bf.y * scale);
      oh[u] = o;
      const float2 of = __half22float2(o);
      ss += of.x * of.x + of.y * of.y;
    }
    wv[t] = ld_u4(w_row + b);
    st_u4(orow + b, ov[t]);
  }
  const float rrms = rsqrtf(block_sum_bcast(ss) / float(HC_DIM) + EPS);

#pragma unroll
  for (int t = 0; t < K; t++) {
    const int b = (threadIdx.x + t * (int)blockDim.x) * 8;
    const half2* wh = reinterpret_cast<const half2*>(&wv[t]);
    const half2* oh = reinterpret_cast<const half2*>(&ov[t]);
    uint4 yv;
    half2* yh = reinterpret_cast<half2*>(&yv);
#pragma unroll
    for (int u = 0; u < 4; u++) {
      const float2 of = __half22float2(oh[u]);
      const float2 wf = __half22float2(wh[u]);
      const float lo = of.x * rrms;
      const float hi = of.y * rrms;
      yh[u] = __floats2half2_rn(lo + lo * wf.x, hi + hi * wf.y);
    }
    st_u4(yrow + b, yv);
  }
}

void hc_combine_norm(const at::Tensor& residual, const at::Tensor& block_output,
                     const at::Tensor& inj, const at::Tensor& norm_weight,
                     at::Tensor& out, at::Tensor& y, int64_t hc_count,
                     double eps) {
  const int N = residual.size(0);
  const int DIM = residual.size(1);
  TORCH_CHECK(residual.scalar_type() == at::kHalf && residual.is_contiguous(),
              "residual");
  TORCH_CHECK(
      block_output.scalar_type() == at::kHalf && block_output.is_contiguous(),
      "block_output");
  TORCH_CHECK(inj.scalar_type() == at::kHalf && inj.is_contiguous(), "inj");
  TORCH_CHECK(
      norm_weight.scalar_type() == at::kHalf && norm_weight.is_contiguous(),
      "norm_weight");
  TORCH_CHECK(out.scalar_type() == at::kHalf && out.is_contiguous(), "out");
  TORCH_CHECK(y.scalar_type() == at::kHalf && y.is_contiguous(), "y");
  TORCH_CHECK(DIM % hc_count == 0, "DIM must be divisible by hc_count");
  const int HC_DIM = DIM / hc_count;
  TORCH_CHECK(norm_weight.numel() == HC_DIM || norm_weight.numel() == DIM,
              "norm_weight.numel() must equal HC_DIM or DIM");
  const int W_SHARED = (norm_weight.numel() == HC_DIM) ? 1 : 0;
  const at::cuda::OptionalCUDAGuard guard(residual.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const half* bp = reinterpret_cast<const half*>(block_output.const_data_ptr());
  const half* rp = reinterpret_cast<const half*>(residual.const_data_ptr());
  const half* jp = reinterpret_cast<const half*>(inj.const_data_ptr());
  const half* wp = reinterpret_cast<const half*>(norm_weight.const_data_ptr());
  half* op = reinterpret_cast<half*>(out.mutable_data_ptr());
  half* yp = reinterpret_cast<half*>(y.mutable_data_ptr());
  const bool vec = aligned16(bp) && aligned16(rp) && aligned16(jp) &&
                   aligned16(wp) && aligned16(op) && aligned16(yp) &&
                   (HC_DIM % 8 == 0) && (DIM % 8 == 0);

  const dim3 grid((unsigned)N, (unsigned)hc_count);
  const int slots = HC_DIM / 8;
  const int k = (slots % 64 == 0) ? 2 : 1;
  const int threads = slots / k;
  if (vec && slots % 32 == 0 && threads >= 32 && threads <= 1024) {
    if (k == 2) {
      hc_combine_norm_onepass_kernel<2><<<grid, (unsigned)threads, 0, stream>>>(
          bp, rp, jp, wp, op, yp, HC_DIM, (int)hc_count, W_SHARED, (float)eps);
    } else {
      hc_combine_norm_onepass_kernel<1><<<grid, (unsigned)threads, 0, stream>>>(
          bp, rp, jp, wp, op, yp, HC_DIM, (int)hc_count, W_SHARED, (float)eps);
    }
  } else if (vec) {
    hc_combine_norm_kernel<true><<<grid, kThreads, 0, stream>>>(
        bp, rp, jp, wp, op, yp, HC_DIM, (int)hc_count, W_SHARED, (float)eps);
  } else {
    hc_combine_norm_kernel<false><<<grid, kThreads, 0, stream>>>(
        bp, rp, jp, wp, op, yp, HC_DIM, (int)hc_count, W_SHARED, (float)eps);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// ---------------------------------------------------------------------------
// Public host wrappers (torch::Tensors). Names are kept under the same
// _rocm_C:: namespace as the rest of the bindings.
// ---------------------------------------------------------------------------

void hc_grouped_gemma_rmsnorm_rdna2(
    torch::Tensor x,       // [N, DIM] fp16
    torch::Tensor weight,  // [GROUP_DIM] or [DIM] fp16
    torch::Tensor y,       // [N, DIM] fp16
    int64_t num_groups, double eps) {
  grouped_gemma_rmsnorm(x, weight, y, num_groups, eps);
}

void hc_silu_rdna2(torch::Tensor x,  // [N, DIM] fp16
                   torch::Tensor y,  // [N, DIM] fp16
                   int64_t hc_count) {
  hc_silu(x, y, hc_count);
}

void hc_gate_mix_rdna2(torch::Tensor x,     // [N, DIM] fp16
                       torch::Tensor gate,  // [N, DIM] fp16
                       torch::Tensor y,     // [N, DIM/HC] fp16
                       int64_t hc_count) {
  hc_gate_mix(x, gate, y, hc_count);
}

void hc_combine_rdna2(torch::Tensor residual,          // [N, DIM] fp16
                      torch::Tensor block_output,      // [N, DIM/HC] fp16
                      torch::Tensor injection_logits,  // [N, HC] fp16
                      torch::Tensor out,               // [N, DIM] fp16
                      int64_t hc_count) {
  hc_combine(residual, block_output, injection_logits, out, hc_count);
}

void hc_combine_norm_rdna2(torch::Tensor residual,          // [N, DIM] fp16
                           torch::Tensor block_output,      // [N, DIM/HC] fp16
                           torch::Tensor injection_logits,  // [N, HC] fp16
                           torch::Tensor norm_weight,  // [DIM/HC] or [DIM] fp16
                           torch::Tensor out,          // [N, DIM] fp16
                           torch::Tensor y,            // [N, DIM] fp16
                           int64_t hc_count, double eps) {
  hc_combine_norm(residual, block_output, injection_logits, norm_weight, out, y,
                  hc_count, eps);
}
