// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// One design of the shipped dense GEMM kernel (csrc/rocm/rdna_hint_flex_dense_
// gemm_kernel.cuh) as a shared library with a plain C entry point, for
// ``bench_prod.py --dev``: the design is chosen with -DDEV_* at compile time,
//
//   hipcc -O3 -std=c++17 --offload-arch=gfx1030 -shared -fPIC \
//     -DDEV_BM=128 -DDEV_BN=128 -DDEV_BK=32 -DDEV_TM=8 -DDEV_TN=16 -DDEV_NBUF=2 \
//     -DDEV_PAD=0 -DDEV_RAST=8 -DDEV_LW=8 -DDEV_SWZ=1 -DDEV_KU=1 -DDEV_BROT=0 \
//     -DDEV_SKIP=0 -DDEV_PRIO=0 -DDEV_DMA=0 -DDEV_BUF=0 dense_dev.hip -o variant.so
//
// and the kernel is the one the production TU instantiates, so what is measured
// here is what ships once the design is added to kDesigns.
#include "../../../../csrc/rocm/rdna_hint_flex_dense_gemm_kernel.cuh"

#ifndef DEV_BROT
  #define DEV_BROT 0
#endif
#ifndef DEV_SKIP
  #define DEV_SKIP 0
#endif
#ifndef DEV_PRIO
  #define DEV_PRIO 0
#endif
#ifndef DEV_DMA
  #define DEV_DMA 0
#endif
#ifndef DEV_BUF
  #define DEV_BUF 0
#endif

using namespace vllm::rdna_hint_flex::gemm;
using P = Design<DEV_BM, DEV_BN, DEV_BK, DEV_TM, DEV_TN, DEV_NBUF, DEV_PAD,
                 DEV_RAST, DEV_LW, DEV_SWZ != 0, DEV_KU, DEV_BROT, DEV_SKIP,
                 DEV_PRIO, DEV_DMA, DEV_BUF>;

extern "C" {

// out[0..7]: VGPRs, static LDS bytes, scratch bytes, threads, blocks per WGP
// (hip occupancy), tile rows, tile cols, 0.
int variant_info(int* out) {
  hipFuncAttributes at;
  if (hipFuncGetAttributes(&at, reinterpret_cast<const void*>(gemm_kernel<P>)) !=
      hipSuccess)
    return -1;
  int nb = 0;
  if (hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &nb, reinterpret_cast<const void*>(gemm_kernel<P>), P::T, 0) !=
      hipSuccess)
    nb = -1;
  out[0] = at.numRegs;
  out[1] = static_cast<int>(at.sharedSizeBytes);
  out[2] = static_cast<int>(at.localSizeBytes);
  out[3] = P::T;
  out[4] = nb;
  out[5] = P::BM;
  out[6] = P::BN;
  out[7] = 0;
  return 0;
}

int variant_launch(const void* A, const void* W, void* C, int M, int N, int K,
                   int lda, int ldw, int ldc, void* stream) {
  if (K % P::BK != 0) return -2;
  const int ntm = (M + P::BM - 1) / P::BM, ntn = (N + P::BN - 1) / P::BN;
  hipLaunchKernelGGL(gemm_kernel<P>, dim3(ntm * ntn), dim3(P::T), 0,
                     static_cast<hipStream_t>(stream),
                     static_cast<const char*>(A), static_cast<const char*>(W),
                     static_cast<_Float16*>(C), M, N, K, lda, ldw, ldc);
  return static_cast<int>(hipGetLastError());
}

}  // extern "C"
