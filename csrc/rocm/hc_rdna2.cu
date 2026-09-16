// T47 — HyperConnection (HC) prefill glue for Qwen3.8-Flash-Next on gfx1030.
//
// Five elementwise/affine kernels that the Qwen4Exp HC prefill path uses.
// All are M-parallel (1 program per row of the output) and wave-per-row
// stream over the inner dim with vec8 fp16 loads. PDL/GDC (an
// NVIDIA-only Triton feature) is not used here; gfx1030 has no
// programmatic dependent launch. The maths are taken verbatim from the
// matching Triton kernels in vllm/models/qwen4_exp/amd/ops/hc.py.
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
// off (Triton path) so the existing prefill stays untouched until the
// HIP path is verified end-to-end.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {

// ---------------------------------------------------------------------------
// Vector helpers: load/store 8 fp16 = 16 bytes = one uint4
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint4 ld_u4(const void* p) {
  return *reinterpret_cast<const uint4*>(p);
}
__device__ __forceinline__ void st_u4(void* p, uint4 v) {
  *reinterpret_cast<uint4*>(p) = v;
}

// Load BLOCK fp16 from base + offsets (offsets in fp16 elements), masked.
template <int BLOCK>
__device__ __forceinline__ void load_fp16(
    const half* base, int row_stride, int row,
    const int offs[BLOCK], const bool mask[BLOCK], half out[BLOCK]) {
#pragma unroll
  for (int i = 0; i < BLOCK; i++) {
    out[i] = mask[i] ? base[row * row_stride + offs[i]] : __float2half(0.f);
  }
}

// ---------------------------------------------------------------------------
// grouped_gemma_rmsnorm_rdna2
//   x: [N, DIM] fp16, last-dim contiguous
//   w: [GROUP_DIM] or [DIM] fp16 (mirrors Triton W_SHARED)
//   y: [N, DIM] fp16
// ---------------------------------------------------------------------------

template <int BLOCK>
__global__ void grouped_gemma_rmsnorm_kernel(
    const half* __restrict__ x, const half* __restrict__ w, half* __restrict__ y,
    int N, int DIM, int NUM_GROUPS, int GROUP_DIM, int W_SHARED, float EPS) {
  const int pid = blockIdx.x;
  const int group_id = pid % NUM_GROUPS;
  const int row = pid / NUM_GROUPS;
  if (row >= N) return;

  const half* x_row = x + row * DIM;
  half* y_row = y + row * DIM;

  // Sum-of-squares over [0, GROUP_DIM). Bounded by the runtime GROUP_DIM
  // rather than the BLOCK template param: GROUP_DIM is 2560 for Flash-Next
  // (hidden 2560, hc_count 4), and baking it in would need one instantiation
  // per width. The loop steps 8 elements at a time, so register use is flat.
  float ss = 0.f;
#pragma unroll 8
  for (int b = 0; b < GROUP_DIM; b += 8) {
    int off = group_id * GROUP_DIM + b;
    uint4 v = (off + 7 < DIM || (b + 8 <= GROUP_DIM))
                  ? ld_u4(x_row + off)
                  : make_uint4(0, 0, 0, 0);
    const half2* h = reinterpret_cast<const half2*>(&v);
#pragma unroll
    for (int u = 0; u < 4; u++) {
      float2 f = __half22float2(h[u]);
      ss += f.x * f.x + f.y * f.y;
    }
  }
  // Trim tail if GROUP_DIM is not a multiple of 8
  for (int i = (GROUP_DIM / 8) * 8; i < GROUP_DIM; i++) {
    int off = group_id * GROUP_DIM + i;
    if (off < DIM) {
      float f = __half2float(x_row[off]);
      ss += f * f;
    }
  }
  float rrms = rsqrtf(ss / float(GROUP_DIM) + EPS);

  // y = x * rrms + x * rrms * w (fused into one fma per element)
  const half* w_base = (W_SHARED != 0) ? w : (w + group_id * GROUP_DIM);
#pragma unroll 8
  for (int b = 0; b < GROUP_DIM; b += 8) {
    int off = group_id * GROUP_DIM + b;
    if (off + 7 < DIM) {
      uint4 xv = ld_u4(x_row + off);
      uint4 wv = ld_u4(w_base + b);
      const half2* xh = reinterpret_cast<const half2*>(&xv);
      const half2* wh = reinterpret_cast<const half2*>(&wv);
      uint4 yv;
      half2* yh = reinterpret_cast<half2*>(&yv);
#pragma unroll
      for (int u = 0; u < 4; u++) {
        float xf = __half2float(__low2half(xh[u])) * rrms;
        float yf = __half2float(__high2half(xh[u])) * rrms;
        float wf_lo = __half2float(__low2half(wh[u]));
        float wf_hi = __half2float(__high2half(wh[u]));
        // y = x * rrms * (1 + w)  ==  x * rrms + (x * rrms) * w
        yh[u] = __floats2half2_rn(xf + xf * wf_lo, yf + yf * wf_hi);
      }
      st_u4(y_row + off, yv);
    } else {
      // Tail
      for (int i = b; i < b + 8 && (group_id * GROUP_DIM + i) < DIM; i++) {
        int off = group_id * GROUP_DIM + i;
        float xf = __half2float(x_row[off]) * rrms;
        float wf = __half2float(w_base[i]);
        y_row[off] = __float2half(xf + xf * wf);
      }
    }
  }
}

void grouped_gemma_rmsnorm(
    const at::Tensor& x, const at::Tensor& weight, at::Tensor& y,
    int64_t num_groups, double eps) {
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

  // Pick BLOCK: round up GROUP_DIM to a multiple of 8 (vec8 loads), clamp
  // to a sane upper bound so register pressure stays low.
  const int BLOCK = ((GROUP_DIM + 7) / 8) * 8;
  if (BLOCK <= 128) {
    grouped_gemma_rmsnorm_kernel<128><<<N * num_groups, 128>>>(
        reinterpret_cast<const half*>(x.const_data_ptr()),
        reinterpret_cast<const half*>(weight.const_data_ptr()),
        reinterpret_cast<half*>(y.mutable_data_ptr()),
        N, DIM, (int)num_groups, GROUP_DIM, W_SHARED, (float)eps);
  } else if (BLOCK <= 512) {
    grouped_gemma_rmsnorm_kernel<512><<<N * num_groups, 128>>>(
        reinterpret_cast<const half*>(x.const_data_ptr()),
        reinterpret_cast<const half*>(weight.const_data_ptr()),
        reinterpret_cast<half*>(y.mutable_data_ptr()),
        N, DIM, (int)num_groups, GROUP_DIM, W_SHARED, (float)eps);
  } else {
    // Wide groups: the loops are GROUP_DIM-bounded, so the same kernel handles
    // any width and the template arg is only an unroll hint.
    grouped_gemma_rmsnorm_kernel<512><<<N * num_groups, 128>>>(
        reinterpret_cast<const half*>(x.const_data_ptr()),
        reinterpret_cast<const half*>(weight.const_data_ptr()),
        reinterpret_cast<half*>(y.mutable_data_ptr()),
        N, DIM, (int)num_groups, GROUP_DIM, W_SHARED, (float)eps);
  }
}

// ---------------------------------------------------------------------------
// hc_silu_rdna2: y = (x / HC) * sigmoid(x / HC)
//   x: [N, DIM] fp16
// ---------------------------------------------------------------------------

template <int BLOCK>
__global__ void hc_silu_kernel(
    const half* __restrict__ x, half* __restrict__ y, int N, int DIM, float inv_hc) {
  const int row = blockIdx.x;
  if (row >= N) return;
  const half* xr = x + row * DIM;
  half* yr = y + row * DIM;
#pragma unroll 8
  for (int b = 0; b < DIM; b += 8) {
    if (b + 7 < DIM) {
      uint4 xv = ld_u4(xr + b);
      uint4 yv;
      const half2* xh = reinterpret_cast<const half2*>(&xv);
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
      st_u4(yr + b, yv);
    } else {
      for (int i = b; i < DIM; i++) {
        float v = __half2float(xr[i]) * inv_hc;
        float s = 1.f / (1.f + __expf(-v));
        yr[i] = __float2half(v * s);
      }
    }
  }
}

void hc_silu(const at::Tensor& x, at::Tensor& y, int64_t hc_count) {
  const int N = x.size(0);
  const int DIM = x.size(1);
  TORCH_CHECK(x.scalar_type() == at::kHalf && x.is_contiguous(),
              "hc_silu_rdna2: x must be contiguous fp16");
  TORCH_CHECK(y.scalar_type() == at::kHalf && y.is_contiguous(),
              "hc_silu_rdna2: y must be contiguous fp16");
  const at::cuda::OptionalCUDAGuard guard(x.device());

  const int BLOCK = ((DIM + 7) / 8) * 8;
  if (BLOCK <= 128) {
    hc_silu_kernel<128><<<N, 128>>>(
        reinterpret_cast<const half*>(x.const_data_ptr()),
        reinterpret_cast<half*>(y.mutable_data_ptr()),
        N, DIM, 1.f / float(hc_count));
  } else if (BLOCK <= 512) {
    hc_silu_kernel<512><<<N, 128>>>(
        reinterpret_cast<const half*>(x.const_data_ptr()),
        reinterpret_cast<half*>(y.mutable_data_ptr()),
        N, DIM, 1.f / float(hc_count));
  } else if (BLOCK <= 2048) {
    hc_silu_kernel<2048><<<N, 256>>>(
        reinterpret_cast<const half*>(x.const_data_ptr()),
        reinterpret_cast<half*>(y.mutable_data_ptr()),
        N, DIM, 1.f / float(hc_count));
  } else {
    // DIM-bounded loop, so any width works; BLOCK is only an unroll hint.
    hc_silu_kernel<512><<<N, 128>>>(
        reinterpret_cast<const half*>(x.const_data_ptr()),
        reinterpret_cast<half*>(y.mutable_data_ptr()),
        N, DIM, 1.f / float(hc_count));
  }
}

// ---------------------------------------------------------------------------
// hc_gate_mix_rdna2: out[h] = (1/HC) * sum_c sigmoid(g[c*H+h]) * x[c*H+h]
//   x: [N, DIM] fp16
//   g: [N, DIM] fp16
//   out: [N, DIM/HC] fp16
// ---------------------------------------------------------------------------

template <int BLOCK, int HC>
__global__ void hc_gate_mix_kernel(
    const half* __restrict__ x, const half* __restrict__ g,
    half* __restrict__ y, int N, int DIM, int HC_DIM) {
  const int row = blockIdx.x;
  const int tile = blockIdx.y;
  if (row >= N) return;
  const int base = tile * BLOCK;
  if (base >= HC_DIM) return;

  const half* xr = x + row * DIM;
  const half* gr = g + row * DIM;
  half* yr = y + row * HC_DIM;

  // HC unrolled loop: HC constexpr so the compiler folds it.
  float acc[BLOCK];
#pragma unroll
  for (int b = 0; b < BLOCK; b++) acc[b] = 0.f;
#pragma unroll
  for (int c = 0; c < HC; c++) {
    const int off = c * HC_DIM + base;
#pragma unroll
    for (int b = 0; b < BLOCK; b++) {
      if (base + b < HC_DIM) {
        float gv = __half2float(gr[off + b]);
        float xv = __half2float(xr[off + b]);
        acc[b] += (1.f / (1.f + __expf(-gv))) * xv;
      }
    }
  }
#pragma unroll
  for (int b = 0; b < BLOCK; b++) {
    if (base + b < HC_DIM) {
      yr[base + b] = __float2half(acc[b] / float(HC));
    }
  }
}

void hc_gate_mix(
    const at::Tensor& x, const at::Tensor& gate, at::Tensor& y,
    int64_t hc_count) {
  const int N = x.size(0);
  const int DIM = x.size(1);
  TORCH_CHECK(x.scalar_type() == at::kHalf && x.is_contiguous(), "x");
  TORCH_CHECK(gate.scalar_type() == at::kHalf && gate.is_contiguous(), "gate");
  TORCH_CHECK(y.scalar_type() == at::kHalf && y.is_contiguous(), "y");
  TORCH_CHECK(DIM % hc_count == 0, "DIM must be divisible by hc_count");
  const int HC_DIM = DIM / hc_count;
  const at::cuda::OptionalCUDAGuard guard(x.device());

  // Compile-time HC dispatch (matches the Triton static_range)
  const int BLOCK = 128;
  dim3 grid(N, (HC_DIM + BLOCK - 1) / BLOCK);
  auto launch = [&](auto hc_const) {
    constexpr int HC_V = decltype(hc_const)::value;
    hc_gate_mix_kernel<BLOCK, HC_V><<<grid, 128>>>(
        reinterpret_cast<const half*>(x.const_data_ptr()),
        reinterpret_cast<const half*>(gate.const_data_ptr()),
        reinterpret_cast<half*>(y.mutable_data_ptr()),
        N, DIM, HC_DIM);
  };
  if (hc_count == 1) { launch(std::integral_constant<int, 1>{}); }
  else if (hc_count == 2) { launch(std::integral_constant<int, 2>{}); }
  else if (hc_count == 4) { launch(std::integral_constant<int, 4>{}); }
  else if (hc_count == 8) { launch(std::integral_constant<int, 8>{}); }
  else {
    TORCH_CHECK(false, "hc_gate_mix_rdna2: hc_count in {1,2,4,8} only");
  }
}

// ---------------------------------------------------------------------------
// hc_combine_rdna2:
//   res[c, h] = res[c, h] + block[h] * 2 * sigmoid(inj[c] / HC)
//   res: [N, DIM] fp16 in-place (we write to out, callers may alias)
//   block: [N, DIM/HC] fp16
//   inj:  [N, HC]    fp16
// ---------------------------------------------------------------------------

template <int BLOCK, int HC>
__global__ void hc_combine_kernel(
    const half* __restrict__ block_in, const half* __restrict__ res_in,
    const half* __restrict__ inj_in, half* __restrict__ out,
    int N, int HC_DIM, int DIM) {
  const int row = blockIdx.x;
  const int tile = blockIdx.y;
  if (row >= N) return;
  const int base = tile * BLOCK;
  if (base >= HC_DIM) return;

  const half* br = block_in + row * HC_DIM;
  const half* rr = res_in + row * DIM;
  const half* jr = inj_in + row * HC;
  half* or_ = out + row * DIM;

  // Load inj once (HC <= 8 in the Qwen4Exp checkpoint).
  float inj_scale[HC];
#pragma unroll
  for (int c = 0; c < HC; c++) {
    float v = __half2float(jr[c]) / float(HC);
    inj_scale[c] = 2.f / (1.f + __expf(-v));
  }

#pragma unroll
  for (int b = 0; b < BLOCK; b++) {
    const int idx = base + b;
    if (idx < HC_DIM) {
      float bv = __half2float(br[idx]);
#pragma unroll
      for (int c = 0; c < HC; c++) {
        int off = c * HC_DIM + idx;
        float rv = __half2float(rr[off]);
        or_[off] = __float2half(rv + bv * inj_scale[c]);
      }
    }
  }
}

void hc_combine(
    const at::Tensor& residual, const at::Tensor& block_output,
    const at::Tensor& inj, at::Tensor& out, int64_t hc_count) {
  const int N = residual.size(0);
  const int DIM = residual.size(1);
  TORCH_CHECK(residual.scalar_type() == at::kHalf && residual.is_contiguous(),
              "residual");
  TORCH_CHECK(block_output.scalar_type() == at::kHalf && block_output.is_contiguous(),
              "block_output");
  TORCH_CHECK(inj.scalar_type() == at::kHalf && inj.is_contiguous(), "injection_logits");
  TORCH_CHECK(out.scalar_type() == at::kHalf && out.is_contiguous(), "out");
  TORCH_CHECK(DIM % hc_count == 0, "DIM must be divisible by hc_count");
  const int HC_DIM = DIM / hc_count;
  const at::cuda::OptionalCUDAGuard guard(residual.device());

  const int BLOCK = 128;
  dim3 grid(N, (HC_DIM + BLOCK - 1) / BLOCK);
  auto launch = [&](auto hc_const) {
    constexpr int HC_V = decltype(hc_const)::value;
    hc_combine_kernel<BLOCK, HC_V><<<grid, 128>>>(
        reinterpret_cast<const half*>(block_output.const_data_ptr()),
        reinterpret_cast<const half*>(residual.const_data_ptr()),
        reinterpret_cast<const half*>(inj.const_data_ptr()),
        reinterpret_cast<half*>(out.mutable_data_ptr()),
        N, HC_DIM, DIM);
  };
  if (hc_count == 1) { launch(std::integral_constant<int, 1>{}); }
  else if (hc_count == 2) { launch(std::integral_constant<int, 2>{}); }
  else if (hc_count == 4) { launch(std::integral_constant<int, 4>{}); }
  else if (hc_count == 8) { launch(std::integral_constant<int, 8>{}); }
  else { TORCH_CHECK(false, "hc_combine_rdna2: hc_count in {1,2,4,8} only"); }
}

// ---------------------------------------------------------------------------
// hc_combine_norm_rdna2:
//   For each stream c in [0..HC) and row:
//     inj_scale[c] = 2 * sigmoid(inj[c] / HC)
//     out[row, c, h] = round_to_dtype(res[row, c, h] + block[h] * inj_scale[c])
//     y[row, c, h]   = norm(out[row, c, h])  with shared/per-stream weight
// ---------------------------------------------------------------------------

template <int BLOCK, int HC>
__global__ void hc_combine_norm_kernel(
    const half* __restrict__ block_in, const half* __restrict__ res_in,
    const half* __restrict__ inj_in, const half* __restrict__ w_in,
    half* __restrict__ out, half* __restrict__ y,
    int N, int HC_DIM, int DIM, int W_SHARED, float EPS) {
  const int row = blockIdx.x;
  const int stream = blockIdx.y;
  if (row >= N) return;

  const half* br = block_in + row * HC_DIM;
  const half* rr = res_in + row * DIM;
  const half* jr = inj_in + row * HC;
  half* or_ = out + row * DIM;
  half* yr = y + row * DIM;
  const int base = stream * HC_DIM;

  // inj_scale for this stream
  float inj_v = __half2float(jr[stream]) / float(HC);
  float inj_scale = 2.f / (1.f + __expf(-inj_v));

  // Materialize combine result, round to fp16, store, then norm.
  // We do the reduce across BLOCK once for the round and once for the norm.
  // The group is walked in BLOCK-wide chunks: HC_DIM is 2560 for Flash-Next
  // (hidden 2560, hc_count 4) and a full-width register array would spill.
  // Pass 2 re-reads the rounded values from `out` rather than keeping a
  // width-sized array live across both passes.
  float ss = 0.f;

  for (int c0 = 0; c0 < HC_DIM; c0 += BLOCK) {
#pragma unroll 8
    for (int i = 0; i < BLOCK; i++) {
      const int b = c0 + i;
      if (b >= HC_DIM) break;
      const int idx = base + b;
      float bv = __half2float(br[b]);
      float rv = __half2float(rr[idx]);
      // Round to fp16 to match the unfused combine -> RMSNorm boundary.
      half vh = __float2half(rv + bv * inj_scale);
      or_[idx] = vh;
      float vf = __half2float(vh);
      ss += vf * vf;
    }
  }
  float rrms = rsqrtf(ss / float(HC_DIM) + EPS);

  const half* w_base = (W_SHARED != 0) ? w_in : (w_in + base);
  for (int c0 = 0; c0 < HC_DIM; c0 += BLOCK) {
#pragma unroll 8
    for (int i = 0; i < BLOCK; i++) {
      const int b = c0 + i;
      if (b >= HC_DIM) break;
      const int idx = base + b;
      float wf = __half2float(w_base[b]);
      // y = out * rrms * (1 + w)
      float v = __half2float(or_[idx]) * rrms;
      yr[idx] = __float2half(v + v * wf);
    }
  }
}

void hc_combine_norm(
    const at::Tensor& residual, const at::Tensor& block_output,
    const at::Tensor& inj, const at::Tensor& norm_weight,
    at::Tensor& out, at::Tensor& y, int64_t hc_count, double eps) {
  const int N = residual.size(0);
  const int DIM = residual.size(1);
  TORCH_CHECK(residual.scalar_type() == at::kHalf && residual.is_contiguous(),
              "residual");
  TORCH_CHECK(block_output.scalar_type() == at::kHalf && block_output.is_contiguous(),
              "block_output");
  TORCH_CHECK(inj.scalar_type() == at::kHalf && inj.is_contiguous(), "inj");
  TORCH_CHECK(norm_weight.scalar_type() == at::kHalf && norm_weight.is_contiguous(),
              "norm_weight");
  TORCH_CHECK(out.scalar_type() == at::kHalf && out.is_contiguous(), "out");
  TORCH_CHECK(y.scalar_type() == at::kHalf && y.is_contiguous(), "y");
  TORCH_CHECK(DIM % hc_count == 0, "DIM must be divisible by hc_count");
  const int HC_DIM = DIM / hc_count;
  TORCH_CHECK(norm_weight.numel() == HC_DIM || norm_weight.numel() == DIM,
              "norm_weight.numel() must equal HC_DIM or DIM");
  const int W_SHARED = (norm_weight.numel() == HC_DIM) ? 1 : 0;
  const at::cuda::OptionalCUDAGuard guard(residual.device());

  const int BLOCK = 512;  // matches the Triton default
  dim3 grid(N, hc_count);
  auto launch = [&](auto hc_const) {
    constexpr int HC_V = decltype(hc_const)::value;
    hc_combine_norm_kernel<BLOCK, HC_V><<<grid, 128>>>(
        reinterpret_cast<const half*>(block_output.const_data_ptr()),
        reinterpret_cast<const half*>(residual.const_data_ptr()),
        reinterpret_cast<const half*>(inj.const_data_ptr()),
        reinterpret_cast<const half*>(norm_weight.const_data_ptr()),
        reinterpret_cast<half*>(out.mutable_data_ptr()),
        reinterpret_cast<half*>(y.mutable_data_ptr()),
        N, HC_DIM, DIM, W_SHARED, (float)eps);
  };
  if (hc_count == 1) { launch(std::integral_constant<int, 1>{}); }
  else if (hc_count == 2) { launch(std::integral_constant<int, 2>{}); }
  else if (hc_count == 4) { launch(std::integral_constant<int, 4>{}); }
  else if (hc_count == 8) { launch(std::integral_constant<int, 8>{}); }
  else { TORCH_CHECK(false, "hc_combine_norm_rdna2: hc_count in {1,2,4,8} only"); }
}

}  // namespace

// ---------------------------------------------------------------------------
// Public host wrappers (torch::Tensors). Names are kept under the same
// _rocm_C:: namespace as the rest of the bindings.
// ---------------------------------------------------------------------------

void hc_grouped_gemma_rmsnorm_rdna2(
    torch::Tensor x,        // [N, DIM] fp16
    torch::Tensor weight,   // [GROUP_DIM] or [DIM] fp16
    torch::Tensor y,        // [N, DIM] fp16
    int64_t num_groups,
    double eps) {
  grouped_gemma_rmsnorm(x, weight, y, num_groups, eps);
}

void hc_silu_rdna2(
    torch::Tensor x,        // [N, DIM] fp16
    torch::Tensor y,        // [N, DIM] fp16
    int64_t hc_count) {
  hc_silu(x, y, hc_count);
}

void hc_gate_mix_rdna2(
    torch::Tensor x,        // [N, DIM] fp16
    torch::Tensor gate,     // [N, DIM] fp16
    torch::Tensor y,        // [N, DIM/HC] fp16
    int64_t hc_count) {
  hc_gate_mix(x, gate, y, hc_count);
}

void hc_combine_rdna2(
    torch::Tensor residual,         // [N, DIM] fp16
    torch::Tensor block_output,     // [N, DIM/HC] fp16
    torch::Tensor injection_logits, // [N, HC] fp16
    torch::Tensor out,              // [N, DIM] fp16
    int64_t hc_count) {
  hc_combine(residual, block_output, injection_logits, out, hc_count);
}

void hc_combine_norm_rdna2(
    torch::Tensor residual,         // [N, DIM] fp16
    torch::Tensor block_output,     // [N, DIM/HC] fp16
    torch::Tensor injection_logits, // [N, HC] fp16
    torch::Tensor norm_weight,      // [DIM/HC] or [DIM] fp16
    torch::Tensor out,              // [N, DIM] fp16
    torch::Tensor y,                // [N, DIM] fp16
    int64_t hc_count,
    double eps) {
  hc_combine_norm(residual, block_output, injection_logits, norm_weight,
                  out, y, hc_count, eps);
}
