// HIP replacement for the top-k / top-p logit masking Triton kernels on
// gfx10x (RDNA2): vllm/v1/sample/ops/topk_topp_triton.py
// (_topk_topp_kernel, and the _topp_sb_{stats,step,mask}_kernel split
// pipeline that only serves p-only rows).
//
// The masking semantics reproduced here are those of
// apply_top_k_top_p_pytorch, the reference the Triton kernels approximate:
//   * top-k keeps every logit >= the k-th largest logit (ties included);
//   * top-p then keeps the smallest set of the surviving logits whose
//     normalised probability mass first reaches p, ties included.
// The Triton path instead keeps an index-ordered subset of the tied copies at
// the boundary and truncates top-k to exactly k. The two references already
// disagree with each other on exact ties (pytorch keeps the highest indices of
// the tied group, Triton the lowest), and real fp32 serving traffic has no
// exact ties, so the pytorch rule is what is implemented here.
//
// No sorting: a float maps monotonically onto a uint32 key, so a 3-level radix
// histogram cascade (12 + 12 + 8 bits = all 32 key bits) locates the exact
// boundary key. Each level is one block-wide pass over the row and the whole
// pipeline runs in a single launch: one block per row, 1024 threads, all state
// in shared memory. Passes are skipped when a row has no top-k / no top-p.
//
// Exact ties are handled by construction: equal values share a key at every
// level, so the k-th largest *value* comes out bit-exact even when thousands of
// logits are identical (grammar masks, degenerate rows).
#include <hip/hip_runtime.h>
#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <optional>

namespace vllm {
namespace rdna2_topk_topp {

constexpr int kThreads = 1024;
constexpr int kNBuckets = 4096;  // 12-bit levels; level 3 only uses 256
constexpr int kGroups = kNBuckets / kThreads;  // buckets per thread in a scan
// Candidate capacity of the k-only fast path.  The candidates are the logits in
// the level-1 prefix bucket that holds the k-th largest plus everything above
// it, i.e. `above + hist[p1]`, which the caller checks before gathering; for the
// served shapes (k = 20, vocab = 151936) that is ~60 entries.
constexpr int kCandCap = 2048;

// ---------------------------------------------------------------------------
// IEEE-754 <-> monotone uint32 key
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t key_of(float f) {
  const uint32_t b = __float_as_uint(f);
  return b ^ ((uint32_t)((int32_t)b >> 31) | 0x80000000u);
}

__device__ __forceinline__ float float_of(uint32_t k) {
  const uint32_t b = (k & 0x80000000u) ? (k ^ 0x80000000u) : ~k;
  return __uint_as_float(b);
}

// Triton's `math.exp` lowering on gfx1030 (see v1_sample_rdna2.cu).
__device__ __forceinline__ float tl_exp(float x) {
  return exp2f(x * 1.4426950408889634f);
}

// ---------------------------------------------------------------------------
// Row traversal (float4, 4 unrolled loads in flight, scalar tail)
// ---------------------------------------------------------------------------
template <typename F>
__device__ __forceinline__ void row_each_vec(const float* __restrict__ row,
                                             int vocab, int tid, F&& f) {
  const float4* __restrict__ r4 = reinterpret_cast<const float4*>(row);
  const int n4 = vocab >> 2;
  int i = tid;
  for (; i + 3 * kThreads < n4; i += 4 * kThreads) {
    const float4 a = r4[i];
    const float4 b = r4[i + kThreads];
    const float4 c = r4[i + 2 * kThreads];
    const float4 d = r4[i + 3 * kThreads];
    const int j0 = 4 * i;
    const int j1 = 4 * (i + kThreads);
    const int j2 = 4 * (i + 2 * kThreads);
    const int j3 = 4 * (i + 3 * kThreads);
    f(a.x, j0); f(a.y, j0 + 1); f(a.z, j0 + 2); f(a.w, j0 + 3);
    f(b.x, j1); f(b.y, j1 + 1); f(b.z, j1 + 2); f(b.w, j1 + 3);
    f(c.x, j2); f(c.y, j2 + 1); f(c.z, j2 + 2); f(c.w, j2 + 3);
    f(d.x, j3); f(d.y, j3 + 1); f(d.z, j3 + 2); f(d.w, j3 + 3);
  }
  for (; i < n4; i += kThreads) {
    const float4 a = r4[i];
    const int j = 4 * i;
    f(a.x, j); f(a.y, j + 1); f(a.z, j + 2); f(a.w, j + 3);
  }
  for (int j = (vocab & ~3) + tid; j < vocab; j += kThreads) f(row[j], j);
}

template <typename F>
__device__ __forceinline__ void row_each_scalar(const float* __restrict__ row,
                                                int vocab, int tid, F&& f) {
  int i = tid;
  for (; i + 3 * kThreads < vocab; i += 4 * kThreads) {
    f(row[i], i);
    f(row[i + kThreads], i + kThreads);
    f(row[i + 2 * kThreads], i + 2 * kThreads);
    f(row[i + 3 * kThreads], i + 3 * kThreads);
  }
  for (; i < vocab; i += kThreads) f(row[i], i);
}

// ---------------------------------------------------------------------------
// Block reductions / histogram crossing search
// ---------------------------------------------------------------------------

template <bool IS_MAX>
__device__ __forceinline__ float block_reduce(float v, float* __restrict__ s,
                                              int tid) {
  s[tid] = v;
  __syncthreads();
#pragma unroll
  for (int d = kThreads / 2; d > 0; d >>= 1) {
    if (tid < d) {
      s[tid] = IS_MAX ? fmaxf(s[tid], s[tid + d]) : (s[tid] + s[tid + d]);
    }
    __syncthreads();
  }
  const float r = s[0];
  __syncthreads();
  return r;
}

struct Crossing {
  uint32_t idx;   // histogram index (0 = highest key prefix)
  float above;    // cumulative weight strictly above that index
  float total;    // total weight in the histogram
  int found;
};

// Accumulate the histogram in `hist` and report where the cumulative weight
// (walking histogram indices low -> high, i.e. keys high -> low) first reaches
// `target`.
template <typename H>
__device__ __forceinline__ Crossing crossing(const H* __restrict__ hist,
                                             float* __restrict__ scan, int tid,
                                             float target) {
  // The caller has just filled `hist`; other threads may still be adding to it.
  __syncthreads();
  // target == 0 (p == 0) must still resolve to the first non-empty bucket.
  const float t = fmaxf(target, FLT_MIN);
  float g = 0.f;
#pragma unroll
  for (int i = 0; i < kGroups; ++i) g += (float)hist[tid * kGroups + i];
  scan[tid] = g;
  __syncthreads();
#pragma unroll
  for (int d = 1; d < kThreads; d <<= 1) {
    const float prev = (tid >= d) ? scan[tid - d] : 0.f;
    __syncthreads();
    scan[tid] += prev;
    __syncthreads();
  }
  const float total = scan[kThreads - 1];
  const float above = (tid > 0) ? scan[tid - 1] : 0.f;
  uint32_t idx = 0;
  float above_at = 0.f;
  int found = 0;
  if (above < t && scan[tid] >= t) {
    float acc = above;
#pragma unroll
    for (int i = 0; i < kGroups; ++i) {
      const float h = (float)hist[tid * kGroups + i];
      if (acc < t && acc + h >= t) {
        idx = (uint32_t)(tid * kGroups + i);
        above_at = acc;
        found = 1;
        break;
      }
      acc += h;
    }
  }
  __syncthreads();  // every thread has read its scan neighbours
  if (tid == 0) scan[kThreads] = 0.f;
  __syncthreads();
  if (found) {
    scan[0] = (float)idx;
    scan[1] = above_at;
    scan[kThreads] = 1.f;
  }
  __syncthreads();
  Crossing c;
  c.idx = (uint32_t)scan[0];
  c.above = scan[1];
  c.total = total;
  c.found = scan[kThreads] != 0.f;
  __syncthreads();
  return c;
}

// ---------------------------------------------------------------------------
// k-only fast path: bounded candidate gather + in-smem selection
// ---------------------------------------------------------------------------

// One gathered logit: its monotone key (sort field) and where it lives in the
// row, so the mask fix-up is a handful of scattered stores instead of a pass.
struct Cand {
  unsigned key;
  unsigned idx;
};

// Descending bitonic sort of `n` (power of two) candidates, keyed on `key`.
__device__ __forceinline__ void bitonic_desc(Cand* __restrict__ a, int n,
                                             int tid) {
  for (int k = 2; k <= n; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = tid; i < n; i += kThreads) {
        const int l = i ^ j;
        if (l <= i) continue;
        const Cand x = a[i];
        const Cand y = a[l];
        // Blocks with (i & k) == 0 sort descending, the others ascending.
        const bool swap = ((i & k) == 0) ? (y.key > x.key) : (x.key > y.key);
        if (swap) {
          a[i] = y;
          a[l] = x;
        }
      }
      __syncthreads();
    }
  }
}

// Mask every logit below the level-1 key prefix `p1` in place and gather the
// rest (all of them are finite) into `cand`; the caller has already checked
// that they fit in kCandCap.
__device__ __forceinline__ void fast_gather(float* __restrict__ row, int vocab,
                                            int tid, int vec, unsigned p1,
                                            float mask_value, Cand* cand,
                                            unsigned* cand_n) {
  auto push = [&](unsigned key, int j) {
    const unsigned pos = atomicAdd(cand_n, 1u);
    if (pos < (unsigned)kCandCap) {
      cand[pos].key = key;
      cand[pos].idx = (unsigned)j;
    }
  };
  auto lane = [&](float x, int j) -> float {
    if (isfinite(x)) {
      const unsigned key = key_of(x);
      if ((key >> 20) >= p1) {
        push(key, j);
        return x;
      }
    } else if (x > 0.f) {
      return x;  // +inf survives the threshold test in every reference
    }
    return mask_value;
  };
  if (vec) {
    float4* __restrict__ r4 = reinterpret_cast<float4*>(row);
    const int n4 = vocab >> 2;
    int i = tid;
    for (; i + 3 * kThreads < n4; i += 4 * kThreads) {
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const int j = 4 * (i + t * kThreads);
        float4 a = r4[i + t * kThreads];
        a.x = lane(a.x, j);
        a.y = lane(a.y, j + 1);
        a.z = lane(a.z, j + 2);
        a.w = lane(a.w, j + 3);
        r4[i + t * kThreads] = a;
      }
    }
    for (; i < n4; i += kThreads) {
      const int j = 4 * i;
      float4 a = r4[i];
      a.x = lane(a.x, j);
      a.y = lane(a.y, j + 1);
      a.z = lane(a.z, j + 2);
      a.w = lane(a.w, j + 3);
      r4[i] = a;
    }
    for (int j = (vocab & ~3) + tid; j < vocab; j += kThreads)
      row[j] = lane(row[j], j);
  } else {
    int i = tid;
    for (; i + 3 * kThreads < vocab; i += 4 * kThreads) {
      row[i] = lane(row[i], i);
      row[i + kThreads] = lane(row[i + kThreads], i + kThreads);
      row[i + 2 * kThreads] = lane(row[i + 2 * kThreads], i + 2 * kThreads);
      row[i + 3 * kThreads] = lane(row[i + 3 * kThreads], i + 3 * kThreads);
    }
    for (; i < vocab; i += kThreads) row[i] = lane(row[i], i);
  }
}
// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(kThreads, 1)
    topk_topp_kernel(const float* __restrict__ logits, int64_t stride0,
                     const int32_t* __restrict__ k_ptr,
                     const float* __restrict__ p_ptr, float mask_value,
                     int vocab, int vec) {
  const int tid = threadIdx.x;
  const int64_t row_id = blockIdx.x;
  const float* __restrict__ rowc = logits + row_id * stride0;
  float* __restrict__ row = const_cast<float*>(rowc);
  const int kv = k_ptr ? (int)k_ptr[row_id] : vocab;
  const float pv = p_ptr ? p_ptr[row_id] : 1.0f;
  const bool do_k = kv < vocab;
  const bool do_p = pv < 1.0f;

  // gfx10x lowers a shared-memory float atomicAdd to a CAS loop (~10x the cost
  // of the native integer one), so the top-k cascade counts in `unsigned`; only
  // the weighted top-p mass cascade still needs a float histogram.
  extern __shared__ __align__(16) unsigned char smem[];
  unsigned* chist = reinterpret_cast<unsigned*>(smem);         // kNBuckets
  float* fhist = reinterpret_cast<float*>(chist + kNBuckets);  // kNBuckets
  float* scan = fhist + kNBuckets;                             // kThreads + 2
  Cand* cand = reinterpret_cast<Cand*>(scan + kThreads + 2);   // kCandCap
  unsigned* cand_n = reinterpret_cast<unsigned*>(cand + kCandCap);

  auto each = [&](auto&& f) {
    if (vec) {
      row_each_vec(rowc, vocab, tid, f);
    } else {
      row_each_scalar(rowc, vocab, tid, f);
    }
  };
  auto clear_u = [&]() {
    for (int i = tid; i < kNBuckets; i += kThreads) chist[i] = 0u;
    __syncthreads();
  };
  auto clear_f = [&]() {
    for (int i = tid; i < kNBuckets; i += kThreads) fhist[i] = 0.f;
    __syncthreads();
  };

  // ---- pass 1: row max, finite count, top-k level-1 counts --------------
  clear_u();
  float lmax = -INFINITY;
  float lcnt = 0.f;
  each([&](float x, int) {
    if (!isfinite(x)) return;
    lmax = fmaxf(lmax, x);
    lcnt += 1.f;
    if (do_k) atomicAdd(&chist[(kNBuckets - 1) - (key_of(x) >> 20)], 1u);
  });
  const float maxv = block_reduce<true>(lmax, scan, tid);
  const float cnt = block_reduce<false>(lcnt, scan, tid);

  // ---- top-k: 3-level count cascade -> exact k-th largest value ---------
  bool have_v = false;
  float v = -INFINITY;
  if (do_k && cnt >= (float)kv) {
    const Crossing c1 = crossing(chist, scan, tid, (float)kv);
    if (c1.found) {
      // `c1.idx` is a reversed 12-bit prefix (bits 20..31 of the key). Every
      // logit below that prefix is below the k-th largest, every logit above it
      // (plus the prefix bucket itself, which holds the k-th largest) is a
      // candidate whose count `above + chist[c1.idx]` is known here, so k-only
      // rows can gather them and select in shared memory instead of scanning
      // the row twice more.
      const unsigned cp1 = chist[c1.idx];
      // kv >= 1: the cascade below mirrors the reference for a degenerate
      // k = 0 (keep only the argmax) and this path would index cand[kv - 1].
      if (kv >= 1 && !do_p && c1.above + (float)cp1 <= (float)kCandCap) {
        const unsigned kp1 = (unsigned)(kNBuckets - 1) - c1.idx;
        if (tid == 0) *cand_n = 0u;
        __syncthreads();
        fast_gather(row, vocab, tid, vec, kp1, mask_value, cand, cand_n);
        __syncthreads();
        const int g = (int)*cand_n;
        int np = 1;
        while (np < g) np <<= 1;
        for (int i = g + tid; i < np; i += kThreads) cand[i].key = 0u;
        __syncthreads();
        bitonic_desc(cand, np, tid);
        const float vk = float_of(cand[kv - 1].key);
        for (int i = tid; i < g; i += kThreads) {
          const unsigned j = cand[i].idx;
          if (!(row[j] >= vk)) row[j] = mask_value;
        }
        return;
      }
      const uint32_t p1 = (uint32_t)(kNBuckets - 1) - c1.idx;
      clear_u();
      each([&](float x, int) {
        if (!isfinite(x)) return;
        const uint32_t key = key_of(x);
        if ((key >> 20) != p1) return;
        atomicAdd(&chist[(kNBuckets - 1) - ((key >> 8) & 0xFFFu)], 1u);
      });
      const Crossing c2 = crossing(chist, scan, tid, (float)kv - c1.above);
      if (c2.found) {
        const uint32_t p24 = (p1 << 12) | ((uint32_t)(kNBuckets - 1) - c2.idx);
        clear_u();
        each([&](float x, int) {
          if (!isfinite(x)) return;
          const uint32_t key = key_of(x);
          if ((key >> 8) != p24) return;
          atomicAdd(&chist[255u - (key & 0xFFu)], 1u);
        });
        const Crossing c3 =
            crossing(chist, scan, tid, (float)kv - c1.above - c2.above);
        if (c3.found) {
          v = float_of((p24 << 8) | (255u - c3.idx));
          have_v = true;
        }
      }
    }
  }
  const float surv_min = have_v ? v : -INFINITY;

  // ---- top-p: 3-level mass cascade over the survivors -------------------
  float u = -INFINITY;
  if (do_p) {
    clear_f();
    float lmass = 0.f;
    each([&](float x, int) {
      if (!isfinite(x)) return;
      if (x < surv_min) return;
      const float m = tl_exp(x - maxv);
      lmass += m;
      atomicAdd(&fhist[(kNBuckets - 1) - (key_of(x) >> 20)], m);
    });
    const float z = block_reduce<false>(lmass, scan, tid);
    const float target = pv * z;
    Crossing c1 = crossing(fhist, scan, tid, target);
    Crossing c2 = {0, 0.f, 0.f, 0};
    Crossing c3 = {0, 0.f, 0.f, 0};
    if (c1.found) {
      const uint32_t p1 = (uint32_t)(kNBuckets - 1) - c1.idx;
      clear_f();
      each([&](float x, int) {
        if (!isfinite(x)) return;
        if (x < surv_min) return;
        const uint32_t key = key_of(x);
        if ((key >> 20) != p1) return;
        atomicAdd(&fhist[(kNBuckets - 1) - ((key >> 8) & 0xFFFu)], tl_exp(x - maxv));
      });
      c2 = crossing(fhist, scan, tid, target - c1.above);
      if (c2.found) {
        const uint32_t p24 = (p1 << 12) | ((uint32_t)(kNBuckets - 1) - c2.idx);
        clear_f();
        each([&](float x, int) {
          if (!isfinite(x)) return;
          if (x < surv_min) return;
          const uint32_t key = key_of(x);
          if ((key >> 8) != p24) return;
          atomicAdd(&fhist[255u - (key & 0xFFu)], tl_exp(x - maxv));
        });
        c3 = crossing(fhist, scan, tid, target - c1.above - c2.above);
        if (c3.found) u = float_of((p24 << 8) | (255u - c3.idx));
      }
    }
    if (u == -INFINITY) u = surv_min;
  }

  // ---- final pass: mask everything below the threshold ------------------
  const float threshold = do_p ? u : surv_min;
  if (threshold == -INFINITY) return;
  if (vec) {
    float4* __restrict__ r4 = reinterpret_cast<float4*>(row);
    const int n4 = vocab >> 2;
    int i = tid;
    for (; i + 3 * kThreads < n4; i += 4 * kThreads) {
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        float4 a = r4[i + t * kThreads];
        if (!(a.x >= threshold)) a.x = mask_value;
        if (!(a.y >= threshold)) a.y = mask_value;
        if (!(a.z >= threshold)) a.z = mask_value;
        if (!(a.w >= threshold)) a.w = mask_value;
        r4[i + t * kThreads] = a;
      }
    }
    for (; i < n4; i += kThreads) {
      float4 a = r4[i];
      if (!(a.x >= threshold)) a.x = mask_value;
      if (!(a.y >= threshold)) a.y = mask_value;
      if (!(a.z >= threshold)) a.z = mask_value;
      if (!(a.w >= threshold)) a.w = mask_value;
      r4[i] = a;
    }
    for (int j = (vocab & ~3) + tid; j < vocab; j += kThreads) {
      if (!(row[j] >= threshold)) row[j] = mask_value;
    }
  } else {
    int i = tid;
    for (; i + 3 * kThreads < vocab; i += 4 * kThreads) {
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const int j = i + t * kThreads;
        if (!(row[j] >= threshold)) row[j] = mask_value;
      }
    }
    for (; i < vocab; i += kThreads) {
      if (!(row[i] >= threshold)) row[i] = mask_value;
    }
  }
}

}  // namespace rdna2_topk_topp
}  // namespace vllm

void topk_topp_rdna2(torch::Tensor logits, int64_t logits_stride,
                     const std::optional<torch::Tensor>& k,
                     const std::optional<torch::Tensor>& p, double mask_value) {
  using namespace vllm::rdna2_topk_topp;
  TORCH_CHECK(logits.dim() == 2, "topk_topp_rdna2: logits must be 2-D");
  TORCH_CHECK(logits.scalar_type() == at::kFloat,
              "topk_topp_rdna2: logits must be fp32");
  TORCH_CHECK(logits.stride(1) == 1,
              "topk_topp_rdna2: logits must be row-contiguous");
  const int64_t batch = logits.size(0);
  const int64_t vocab = logits.size(1);
  if (batch == 0 || vocab == 0) return;
  TORCH_CHECK(vocab <= INT32_MAX / 2, "topk_topp_rdna2: vocab too large");
  TORCH_CHECK(logits.stride(0) == logits_stride,
              "topk_topp_rdna2: logits_stride mismatch");

  const int32_t* kp = nullptr;
  const float* pp = nullptr;
  if (k.has_value()) {
    TORCH_CHECK(k->scalar_type() == at::kInt && k->numel() == batch,
                "topk_topp_rdna2: k must be int32 [batch]");
    kp = reinterpret_cast<const int32_t*>(k->data_ptr());
  }
  if (p.has_value()) {
    TORCH_CHECK(p->scalar_type() == at::kFloat && p->numel() == batch,
                "topk_topp_rdna2: p must be fp32 [batch]");
    pp = reinterpret_cast<const float*>(p->data_ptr());
  }

  const at::cuda::OptionalCUDAGuard guard(device_of(logits));
  auto stream = at::cuda::getCurrentCUDAStream();
  const bool vec = (vocab % 4 == 0) && (logits_stride % 4 == 0) &&
                   (reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 == 0);
  const int smem = 2 * kNBuckets * (int)sizeof(float) +
                   (kThreads + 2) * (int)sizeof(float) +
                   kCandCap * (int)sizeof(Cand) + (int)sizeof(unsigned);
  static bool configured = false;
  if (!configured) {
    hipFuncSetAttribute(
        reinterpret_cast<const void*>(topk_topp_kernel),
        hipFuncAttributeMaxDynamicSharedMemorySize, smem);
    configured = true;
  }
  topk_topp_kernel<<<(unsigned)batch, kThreads, smem, stream.stream()>>>(
      reinterpret_cast<const float*>(logits.data_ptr()), logits_stride, kp, pp,
      (float)mask_value, (int)vocab, vec ? 1 : 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
