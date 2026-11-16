// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Dense fp16 prefill GEMM for gfx1030 (wave32, v_dot2_f32_f16):
//
//   C[M, N] (fp16) = A[M, K] (fp16, row-major) x W[N, K]^T (fp16, row-major)
//
// fp32 accumulation, one fp16 rounding at the store (what rocBLAS does for
// these operands). It is the GEMM after the rdna_hint_flex dequant pass of the
// HI2 prefill path (A = transformed activation, W = dequantized scratch).
// Designs of one kernel template (rdna_hint_flex_dense_gemm_kernel.cuh: the
// design axes and knobs are described there), all measured against rocBLAS
// (heuristic and TunableOp-tuned) on the Qwen3.8-27B HI2 TP2 shapes; see
// docs/rdna2/hi2-prefill-gemm-2026-11-15.md and
// docs/rdna2/hi2-prefill-gemm2-2026-11-15.md. The sweep sources they come from
// are tools/rdna/rdna_hint_flex/gemm_sweep/gemm_variant.hip (the design name
// grammar of that tool is quoted next to every design below); the benchmark
// harness tools/rdna/rdna_hint_flex/gemm_sweep/dense_dev.hip compiles the same
// kernel header.
//
// Tails: M and N may be anything (row loads clamp to the last row, stores are
// guarded); K must be a multiple of the design's BK; operands need 16-byte
// aligned base pointers and row strides.
//
// All launches go to the current torch stream and are followed by
// C10_CUDA_KERNEL_LAUNCH_CHECK.

#include <cstdint>
#include <cstring>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "rdna_hint_flex_dense_gemm_kernel.cuh"
#include "rdna_hint_flex_dense_gemm_rdna2.cuh"
#include "rdna_hint_flex_dense_gemm_table.cuh"

// ---------------------------------------------------------------------------
// Host side.
// ---------------------------------------------------------------------------
namespace vllm {
namespace rdna_hint_flex {
namespace gemm {

// The shipped designs (sweep names in kDesigns below; the sweep grammar has no
// letters for the knobs the kernel header adds: +brot +skip +buf, see there).
// 128x128 BK32, 8x16 per thread (128 accumulators), swizzled LDS, 8-byte
// fragments, B fragment dwords rotated in LDS (no same-bank dot2 sources),
// idle tail waves skipped, buffer-descriptor global loads (no 64-bit address
// adds). 232 VGPR, 4 waves/SIMD, 4 blocks per WGP.
using D1 = Design<128, 128, 32, 8, 16, 2, 0, 8, 8, true, 1, 1, 1, 0, 0, 1>;
// D1 with blocks grouped by 4.
using D2 = Design<128, 128, 32, 8, 16, 2, 0, 4, 8, true, 1, 1, 1, 0, 0, 1>;
// 128x64 BK64, 8x8 per thread, one stage, pad 8, skip + buffer loads.
using D3 = Design<128, 64, 64, 8, 8, 1, 8, 4, 8, false, 1, 0, 1, 0, 0, 1>;
// 256x128 BK32, 8x16 per thread, swizzled LDS (48 KiB), grouped by 4.
using D4 = Design<256, 128, 32, 8, 16, 2, 0, 4, 8, true, 1, 1, 1, 0, 0, 1>;
// Tail designs for the last M % 128 rows (<= 32 and <= 64 rows): a 32x128 tile
// with 4x8 and a 64x128 tile with 8x8 accumulators per thread, 128 threads, BK
// 64 (a whole 128-byte line of every weight row per stage: the standalone
// kernel is latency-bound at BK 32), the swizzled 128-row weight tile of D1.
// Not in the dispatch table: tail_design().
using D5 = Design<32, 128, 64, 4, 8, 2, 0, 0, 8, true, 1, 0, 1, 0, 0, 1>;
using D6 = Design<64, 128, 64, 8, 8, 2, 0, 0, 8, true, 1, 0, 1, 0, 0, 1>;

namespace {

struct DesignInfo {
  const char* name;  // sweep grammar (tools/rdna/rdna_hint_flex/gemm_sweep)
  int bm, bn, bk;
};

constexpr DesignInfo kDesigns[] = {
    {"128x128k32t8x16n2p0r8g16b0w0s1m0u1l8z+brot+skip+buf", 128, 128, 32},
    {"128x128k32t8x16n2p0r4g16b0w0s1m0u1l8z+brot+skip+buf", 128, 128, 32},
    {"128x64k64t8x8n1p8r4g16b0w0s1m0u1l8+skip+buf", 128, 64, 64},
    {"256x128k32t8x16n2p0r4g16b0w0s1m0u1l8z+brot+skip+buf", 256, 128, 32},
    {"32x128k64t4x8n2p0r0g16b0w0s1m0u1l8z+skip+buf", 32, 128, 64},
    {"64x128k64t8x8n2p0r0g16b0w0s1m0u1l8z+skip+buf", 64, 128, 64},
};
constexpr int kNumDesigns = sizeof(kDesigns) / sizeof(kDesigns[0]);

bool on_gfx1030() {
  const hipDeviceProp_t* p = at::cuda::getCurrentDeviceProperties();
  return std::strncmp(p->gcnArchName, "gfx1030", 7) == 0;
}

template <class P>
void launch_design(const void* A, const void* W, void* C, int64_t M, int64_t N,
                   int64_t K, int64_t lda, int64_t ldw, int64_t ldc,
                   hipStream_t stream) {
  const int64_t ntm = (M + P::BM - 1) / P::BM, ntn = (N + P::BN - 1) / P::BN;
  gemm_kernel<P>
      <<<dim3(static_cast<unsigned>(ntm * ntn)), dim3(P::T), 0, stream>>>(
          static_cast<const char*>(A), static_cast<const char*>(W),
          static_cast<_Float16*>(C), static_cast<int>(M), static_cast<int>(N),
          static_cast<int>(K), static_cast<int>(lda), static_cast<int>(ldw),
          static_cast<int>(ldc));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

bool aligned16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 15) == 0;
}

}  // namespace

int tail_design(int64_t rows) {
  if (rows <= 0 || rows > kTailMaxRows) return kRocblas;
  return rows <= 32 ? 4 : 5;
}

int pick(int64_t M, int64_t N, int64_t K) {
  for (const Slot& s : kSlots) {
    if (s.n != N || s.k != K) continue;
    if (M < s.m_lo || M > s.m_hi) return kRocblas;
    // nearest measured M (ties go to the smaller one)
    int best = s.anchors[0].design;
    int64_t dist = M > s.anchors[0].m ? M - s.anchors[0].m : s.anchors[0].m - M;
    for (int i = 1; i < s.count; ++i) {
      const int64_t d =
          M > s.anchors[i].m ? M - s.anchors[i].m : s.anchors[i].m - M;
      if (d < dist) {
        dist = d;
        best = s.anchors[i].design;
      }
    }
    return best;
  }
  return kRocblas;
}

bool launch(int design, const void* A, const void* W, void* C, int64_t M,
            int64_t N, int64_t K, int64_t lda, int64_t ldw, int64_t ldc,
            hipStream_t stream) {
  if (design < 0 || design >= kNumDesigns) return false;
  if (M == 0 || N == 0) return true;
  const DesignInfo& d = kDesigns[design];
  // 32-bit byte offsets in the kernel; 16-byte global chunks.
  constexpr int64_t kMaxElems = int64_t{1} << 30;
  if (M < 0 || N < 0 || K <= 0 || K % d.bk != 0 || lda % 8 != 0 ||
      ldw % 8 != 0 || lda < K || ldw < K || ldc < N || M * lda >= kMaxElems ||
      N * ldw >= kMaxElems || ldc >= (int64_t{1} << 31) || !aligned16(A) ||
      !aligned16(W) || !on_gfx1030())
    return false;
  switch (design) {
    case 0:
      launch_design<D1>(A, W, C, M, N, K, lda, ldw, ldc, stream);
      break;
    case 1:
      launch_design<D2>(A, W, C, M, N, K, lda, ldw, ldc, stream);
      break;
    case 2:
      launch_design<D3>(A, W, C, M, N, K, lda, ldw, ldc, stream);
      break;
    case 3:
      launch_design<D4>(A, W, C, M, N, K, lda, ldw, ldc, stream);
      break;
    case 4:
      launch_design<D5>(A, W, C, M, N, K, lda, ldw, ldc, stream);
      break;
    default:
      launch_design<D6>(A, W, C, M, N, K, lda, ldw, ldc, stream);
      break;
  }
  return true;
}

}  // namespace gemm
}  // namespace rdna_hint_flex
}  // namespace vllm

namespace {
namespace hg = vllm::rdna_hint_flex::gemm;

void check_dense_operands(const torch::Tensor& x, const torch::Tensor& w,
                          const torch::Tensor& out) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && out.is_cuda(),
              "rdna_hint_flex_dense_gemm: tensors must be CUDA/HIP tensors");
  TORCH_CHECK(x.scalar_type() == torch::kHalf &&
                  w.scalar_type() == torch::kHalf &&
                  out.scalar_type() == torch::kHalf,
              "rdna_hint_flex_dense_gemm: x, w and out must be fp16");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && out.dim() == 2,
              "rdna_hint_flex_dense_gemm: x [M, K], w [N, K], out [M, N]");
  TORCH_CHECK(x.size(1) == w.size(1) && out.size(0) == x.size(0) &&
                  out.size(1) == w.size(0),
              "rdna_hint_flex_dense_gemm: shape mismatch x ", x.sizes(), " w ",
              w.sizes(), " out ", out.sizes());
  TORCH_CHECK(
      x.stride(1) == 1 && w.stride(1) == 1 && out.stride(1) == 1,
      "rdna_hint_flex_dense_gemm: row-major operands (unit column stride)");
  TORCH_CHECK(x.device() == w.device() && x.device() == out.device(),
              "rdna_hint_flex_dense_gemm: operands on different devices");
}

}  // namespace

// out = x . w^T. design < 0: the dispatch table picks (rocBLAS when it says so
// or the shape is outside the validated set); design >= 0: that design, an
// error if it cannot run the problem (tests and benchmarks).
void rdna_hint_flex_dense_gemm_rdna2(torch::Tensor x, torch::Tensor w,
                                     torch::Tensor out, int64_t design) {
  check_dense_operands(x, w, out);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t M = x.size(0), N = w.size(0), K = x.size(1);
  const bool forced = design >= 0;
  const int d = forced ? static_cast<int>(design) : hg::pick(M, N, K);
  if (d >= 0 && hg::launch(d, x.data_ptr(), w.data_ptr(), out.data_ptr(), M, N,
                           K, x.stride(0), w.stride(0), out.stride(0), stream))
    return;
  TORCH_CHECK(!forced, "rdna_hint_flex_dense_gemm: design ", design,
              " cannot run M=", M, " N=", N, " K=", K,
              " (needs gfx1030, K % BK == 0, 16-byte aligned operands and row "
              "strides)");
  at::mm_out(out, x, w.t());
}

// The design the table picks for (M, N, K); -1 = rocBLAS.
int64_t rdna_hint_flex_dense_gemm_pick_rdna2(int64_t M, int64_t N, int64_t K) {
  return hg::pick(M, N, K);
}
