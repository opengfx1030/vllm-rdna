// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// RidgeFill map GEMM ops for AMD RDNA2 (gfx1030). The kernels and their
// design notes are in ridgefill_map_rdna2.cuh; the tile configurations were
// picked on the real map shape with docs/explore/ridgefill/kbench.
//
// * ridgefill_map_rdna2: fp16 x[F, K] against the stacked fp16 maps
//   w[L * DOUT, K] (v_dot2_f32_f16, fp32 accumulate) into out[L, F, DOUT].
// * ridgefill_map_i8_rdna2: the same product on v_dot4_i32_i8. The op
//   rotates every 128-wide K block of x by the orthonormal Hadamard matrix
//   and quantizes it per row to int8; wq must be the maps rotated the same
//   way and quantized per output row (scales sw). The rotation cancels in
//   the product and spreads the fitted maps' outlier columns, which is what
//   lets per-row x per-channel int8 hold the fit (TESTPLAN, int8 study).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "ops.h"
#include "ridgefill_map_rdna2.cuh"

namespace {

namespace rf = vllm::ridgefill_map_rdna2;

// Cfg<dot, TM, TN, LM, WM, WN, BKB, RW, PL, ROT>: 8 x 8 accumulators per
// lane, 4 x 2 wave32s (256 threads), 128 x 128 block tile, 64 B K slabs.
using F16Cfg = rf::Cfg<rf::kF16, 8, 8, 4, 4, 2, 64, 4, 0, 1>;
using I8Cfg = rf::Cfg<rf::kI8, 8, 8, 4, 4, 2, 64, 4, 0, 1>;

template <class C>
void check_shapes(const torch::Tensor& x, const torch::Tensor& w,
                  const c10::optional<torch::Tensor>& bias,
                  const torch::Tensor& out, int64_t dout, int64_t layers,
                  int64_t elem_bytes) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && out.is_cuda(),
              "ridgefill_map: CUDA tensors required");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && out.dim() == 3,
              "ridgefill_map: x[F,K] w[L*D,K] out[L,F,D]");
  TORCH_CHECK(out.scalar_type() == torch::kHalf, "ridgefill_map: fp16 out");
  TORCH_CHECK(w.size(0) == layers * dout && w.size(1) == x.size(1),
              "ridgefill_map: weight shape");
  TORCH_CHECK(
      out.size(0) == layers && out.size(1) == x.size(0) && out.size(2) == dout,
      "ridgefill_map: out shape");
  TORCH_CHECK(out.is_contiguous() && w.is_contiguous(),
              "ridgefill_map: contiguous w and out");
  // 128-bit staging reads: unit stride along K, 16 B aligned rows.
  TORCH_CHECK(x.stride(1) == 1 && (x.stride(0) * elem_bytes) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
              "ridgefill_map: x rows must be K-contiguous and 16 B aligned");
  TORCH_CHECK((x.size(1) * elem_bytes) % C::BKB == 0,
              "ridgefill_map: K bytes must be a multiple of the K slab");
  TORCH_CHECK(dout % C::BN == 0, "ridgefill_map: BLOCK_N must divide dout");
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda() && bias->scalar_type() == torch::kHalf &&
                    bias->is_contiguous() && bias->numel() == layers * dout,
                "ridgefill_map: bias shape");
  }
}

const __half* bias_ptr(const c10::optional<torch::Tensor>& bias) {
  return bias.has_value() ? reinterpret_cast<const __half*>(bias->data_ptr())
                          : nullptr;
}

}  // namespace

void ridgefill_map_rdna2(torch::Tensor x, torch::Tensor w,
                         c10::optional<torch::Tensor> bias, torch::Tensor out,
                         int64_t dout, int64_t layers) {
#if !defined(VLLM_ROCM_GFX1030)
  // The kernel body is gfx1030-only (dot instructions); a launch on another
  // arch must fail closed rather than write nothing.
  TORCH_CHECK(false, "ridgefill_map: gfx1030 only");
#endif
  TORCH_CHECK(
      x.scalar_type() == torch::kHalf && w.scalar_type() == torch::kHalf,
      "ridgefill_map: fp16 x and w");
  check_shapes<F16Cfg>(x, w, bias, out, dout, layers, 2);
  if (x.size(0) == 0) return;

  const at::cuda::OptionalCUDAGuard guard(x.device());
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  const hipError_t err = rf::launch_map_gemm<F16Cfg>(
      x.data_ptr(), x.stride(0) * 2, w.data_ptr(), w.stride(0) * 2,
      static_cast<int>(x.size(0)), static_cast<int>(layers * dout),
      static_cast<int>(x.size(1) * 2), static_cast<int>(dout), nullptr, nullptr,
      bias_ptr(bias), reinterpret_cast<__half*>(out.data_ptr()), stream);
  TORCH_CHECK(err == hipSuccess,
              "ridgefill_map: launch failed: ", hipGetErrorString(err));
}

void ridgefill_map_i8_rdna2(torch::Tensor x, torch::Tensor wq, torch::Tensor sw,
                            c10::optional<torch::Tensor> bias,
                            torch::Tensor out, int64_t dout, int64_t layers) {
#if !defined(VLLM_ROCM_GFX1030)
  TORCH_CHECK(false, "ridgefill_map_i8: gfx1030 only");
#endif
  TORCH_CHECK(x.scalar_type() == torch::kHalf &&
                  wq.scalar_type() == torch::kInt8 &&
                  sw.scalar_type() == torch::kFloat32,
              "ridgefill_map_i8: fp16 x, int8 wq, fp32 sw");
  check_shapes<I8Cfg>(x, wq, bias, out, dout, layers, 1);
  TORCH_CHECK(x.size(1) % 128 == 0 && x.size(1) <= rf::kQuantMaxK,
              "ridgefill_map_i8: K must be a multiple of 128, at most ",
              rf::kQuantMaxK);
  TORCH_CHECK(sw.is_cuda() && sw.is_contiguous() && sw.numel() == layers * dout,
              "ridgefill_map_i8: sw shape");
  const int64_t rows = x.size(0);
  const int64_t k = x.size(1);
  if (rows == 0) return;

  const at::cuda::OptionalCUDAGuard guard(x.device());
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  auto xq = torch::empty({rows, k}, x.options().dtype(torch::kInt8));
  auto sx = torch::empty({rows}, x.options().dtype(torch::kFloat32));
  hipError_t err = rf::launch_quant_rows(
      reinterpret_cast<const __half*>(x.data_ptr()), x.stride(0),
      static_cast<int>(rows), static_cast<int>(k),
      reinterpret_cast<int8_t*>(xq.data_ptr()), sx.data_ptr<float>(),
      /*had=*/true, stream);
  TORCH_CHECK(err == hipSuccess, "ridgefill_map_i8: quant launch failed: ",
              hipGetErrorString(err));
  err = rf::launch_map_gemm<I8Cfg>(
      xq.data_ptr(), k, wq.data_ptr(), wq.stride(0), static_cast<int>(rows),
      static_cast<int>(layers * dout), static_cast<int>(k),
      static_cast<int>(dout), sx.data_ptr<float>(), sw.data_ptr<float>(),
      bias_ptr(bias), reinterpret_cast<__half*>(out.data_ptr()), stream);
  TORCH_CHECK(err == hipSuccess,
              "ridgefill_map_i8: launch failed: ", hipGetErrorString(err));
}
