// SPDX-License-Identifier: Apache-2.0
// Sparse MLA decode kernel for gfx1030 (RDNA2).
//
// V2 head-split design. V1 (1 CTA/query, 32 threads, 2 heads/thread
// holding full 512-dim accumulators) was correct but ~32% slower than
// the Triton baseline: ~1028 floats/thread of accumulator state spilled
// to local memory, Q was reloaded from global inside the serial K loop,
// and the grid had only num_queries CTAs.
//
// Parallelization:
//   - Grid:  (num_queries, HEADS / HEADS_PER_CTA)  — 16 CTAs per query
//   - Block: (32,)                                 — 1 wave32
//   - Each CTA owns HEADS_PER_CTA=4 heads of one query. Thread t owns
//     nope dims [14t, 14t+14) and rope dims [2t, 2t+2) of every head in
//     the CTA, so Q slices and accumulators are register-resident
//     (64 + 64 floats/thread) and hoisted out of the K loop.
//   - Per-token score: per-thread partial dot + warp butterfly reduce
//     (__shfl_xor, all lanes get the sum). m/l/alpha/p are then computed
//     redundantly per lane, keeping the online-softmax state in
//     registers with no cross-lane traffic.
//   - K staging is double-buffered in smem: token k+1 is prefetched
//     (8-byte vector loads) while token k is scored, with one
//     __syncthreads per token.
//
// K redundancy: the 16 CTAs of a query stream the same K data (MLA K is
// head-shared). At c=1 long-context the working set (~19MB at 32k
// tokens) fits in the V620's 128MB Infinity Cache, so the redundancy
// costs L2 bandwidth, not DRAM.
//
// The K_nope path is FP8 (e4m3 OCP) with per-64-channel E8M0 block
// scales. K_rope is bf16. Matches the cache layout written by
// `_sparse_attn_decode_ragged_kernel` upstream.
#include <hip/hip_runtime.h>
#include <torch/all.h>
#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>

namespace rdna2_mla {

constexpr int NOPE_DIM = 448;
constexpr int ROPE_DIM = 64;
constexpr int COMB_DIM = NOPE_DIM + ROPE_DIM;       // 512
constexpr int HEADS = 64;
constexpr int THREADS = 32;                          // 1 wave32
constexpr int HEADS_PER_CTA = 4;
constexpr int HEAD_GROUPS = HEADS / HEADS_PER_CTA;   // 16
constexpr int NOPE_PER_THREAD = NOPE_DIM / THREADS;  // 14
constexpr int ROPE_PER_THREAD = ROPE_DIM / THREADS;  // 2
constexpr int TOKEN_BYTES = 576;                     // 448 nope + 128 rope (data region)
constexpr int SLOT_BYTES = TOKEN_BYTES + 8;          // 584 allocated slot (+ fp8 scales)
constexpr float NEG_LARGE = -3.4028234663852886e38f;


// Iterate one cache range (main or extra), updating the online-softmax
// state in place. Mirrors the main/extra loops of the Triton
// _sparse_attn_decode_ragged_kernel: the same m/l/acc state flows
// through both ranges.
//
// Double-buffered: s_fp8/s_k_rope/s_scales_raw hold two stages. At
// iteration k, buffer k&1 is consumed while buffer (k&1)^1 is filled
// with token k+1. The single __syncthreads at the end of each iteration
// orders both (a) this iteration's prefetch writes against next
// iteration's reads and (b) this iteration's reads against next
// iteration's prefetch writes into the other buffer.
__device__ __forceinline__ void process_cache_range(
    int query_idx,
    const uint8_t* __restrict__ cache,
    int cache_stride0,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ indptr,
    int block_size,
    int num_rows,
    float scale,
    uint8_t (*s_fp8)[NOPE_DIM],
    __hip_bfloat16 (*s_k_rope)[ROPE_DIM],
    uint8_t (*s_scales_raw)[8],
    float (&q_nope)[HEADS_PER_CTA][NOPE_PER_THREAD],
    float (&q_rope)[HEADS_PER_CTA][ROPE_PER_THREAD],
    float (&m_i)[HEADS_PER_CTA],
    float (&l_i)[HEADS_PER_CTA],
    float (&acc_nope)[HEADS_PER_CTA][NOPE_PER_THREAD],
    float (&acc_rope)[HEADS_PER_CTA][ROPE_PER_THREAD]
) {
    const int tid = threadIdx.x;
    const int dn0 = tid * NOPE_PER_THREAD;
    const int dr0 = tid * ROPE_PER_THREAD;
    const int range_start = indptr[query_idx];
    const int range_len = indptr[query_idx + 1] - range_start;
    if (range_len <= 0) return;

    // Cooperative load of one token into smem buffer `buf`.
    // 8-byte alignment is guaranteed for every region: the cache base is
    // a torch allocation, cache_stride0 = block_size*584, 584 % 8 == 0,
    // and token offsets are multiples of 576 (576 % 8 == 0).
    auto load_token = [&](int k, int buf) {
        int slot = indices[range_start + k];
        if (slot < 0 || slot >= num_rows) return;
        const int64_t block_off = (int64_t)(slot / block_size) * cache_stride0;
        const uint8_t* tok = cache + block_off
            + (int64_t)(slot % block_size) * TOKEN_BYTES;

        const uint2* src_nope = reinterpret_cast<const uint2*>(tok);
        uint2* dst_nope = reinterpret_cast<uint2*>(s_fp8[buf]);
        #pragma unroll
        for (int i = tid; i < NOPE_DIM / 8; i += THREADS) {
            dst_nope[i] = src_nope[i];
        }

        const uint2* src_rope = reinterpret_cast<const uint2*>(tok + NOPE_DIM);
        uint2* dst_rope = reinterpret_cast<uint2*>(s_k_rope[buf]);
        if (tid < (ROPE_DIM * 2) / 8) {
            dst_rope[tid] = src_rope[tid];
        }

        if (tid == 0) {
            const uint8_t* sp = cache + block_off
                + (int64_t)block_size * TOKEN_BYTES
                + (int64_t)(slot % block_size) * 8;
            *reinterpret_cast<uint2*>(s_scales_raw[buf]) =
                *reinterpret_cast<const uint2*>(sp);
        }
    };

    load_token(0, 0);
    __syncthreads();

    for (int k = 0; k < range_len; k++) {
        const int buf = k & 1;
        const int slot = indices[range_start + k];  // uniform across CTA

        // Prefetch token k+1 into the other buffer while computing k.
        if (k + 1 < range_len) {
            load_token(k + 1, buf ^ 1);
        }

        if (slot >= 0 && slot < num_rows) {
            // Dequant this thread's nope slice into registers. The 14
            // dims span at most two 64-channel scale groups; decode the
            // E8M0 scale once per group.
            float k_nope[NOPE_PER_THREAD];
            float sc = 0.0f;
            int last_g = -1;
            #pragma unroll
            for (int j = 0; j < NOPE_PER_THREAD; j++) {
                const int ch = dn0 + j;
                const int g = ch >> 6;
                if (g != last_g) {
                    sc = exp2f((float)s_scales_raw[buf][g] - 127.0f);
                    last_g = g;
                }
                __half_raw r = __hip_cvt_fp8_to_halfraw(s_fp8[buf][ch], __HIP_E4M3);
                k_nope[j] = __half2float(r) * sc;
            }
            float k_rope[ROPE_PER_THREAD];
            #pragma unroll
            for (int j = 0; j < ROPE_PER_THREAD; j++) {
                k_rope[j] = __bfloat162float(s_k_rope[buf][dr0 + j]);
            }

            #pragma unroll
            for (int hh = 0; hh < HEADS_PER_CTA; hh++) {
                float partial = 0.0f;
                #pragma unroll
                for (int j = 0; j < NOPE_PER_THREAD; j++) {
                    partial += q_nope[hh][j] * k_nope[j];
                }
                #pragma unroll
                for (int j = 0; j < ROPE_PER_THREAD; j++) {
                    partial += q_rope[hh][j] * k_rope[j];
                }
                // Butterfly reduce so every lane has the full score.
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    partial += __shfl_xor(partial, off);
                }
                const float score = partial * scale;

                const float m_new = fmaxf(m_i[hh], score);
                const float alpha = expf(m_i[hh] - m_new);
                const float p = expf(score - m_new);
                l_i[hh] = l_i[hh] * alpha + p;
                #pragma unroll
                for (int j = 0; j < NOPE_PER_THREAD; j++) {
                    acc_nope[hh][j] = acc_nope[hh][j] * alpha + p * k_nope[j];
                }
                #pragma unroll
                for (int j = 0; j < ROPE_PER_THREAD; j++) {
                    acc_rope[hh][j] = acc_rope[hh][j] * alpha + p * k_rope[j];
                }
                m_i[hh] = m_new;
            }
        }
        __syncthreads();
    }
}


// Main kernel: one CTA per (query, head-group), 32 threads per CTA.
__global__ void __launch_bounds__(THREADS) sparse_mla_decode_kernel(
    const __hip_bfloat16* __restrict__ q,            // [num_queries, HEADS, COMB_DIM]
    int q_stride0,                                    // query stride
    int q_stride1,                                    // head stride
    const uint8_t* __restrict__ main_cache,           // [num_blocks, block_size, 584]
    int main_cache_stride0,                           // block stride
    const int32_t* __restrict__ main_indices,         // [nnz]
    const int32_t* __restrict__ main_indptr,          // [num_queries + 1]
    int main_block_size,
    int main_num_rows,
    const uint8_t* __restrict__ extra_cache,          // [num_blocks, block_size, 584]
    int extra_cache_stride0,                          // block stride
    const int32_t* __restrict__ extra_indices,        // [nnz_extra]
    const int32_t* __restrict__ extra_indptr,         // [num_queries + 1]
    int extra_block_size,
    int extra_num_rows,
    float scale,
    const float* __restrict__ attn_sink,              // [HEADS] or null
    __hip_bfloat16* __restrict__ out,                 // [num_queries, HEADS, COMB_DIM]
    int out_stride0,
    int out_stride1
) {
    const int query_idx = blockIdx.x;
    const int h0 = blockIdx.y * HEADS_PER_CTA;
    const int tid = threadIdx.x;
    const int dn0 = tid * NOPE_PER_THREAD;
    const int dr0 = tid * ROPE_PER_THREAD;

    // Hoisted Q slice: this thread's dims of every head in the group,
    // loaded once and held in registers for the whole kernel.
    float q_nope[HEADS_PER_CTA][NOPE_PER_THREAD];
    float q_rope[HEADS_PER_CTA][ROPE_PER_THREAD];
    #pragma unroll
    for (int hh = 0; hh < HEADS_PER_CTA; hh++) {
        const __hip_bfloat16* q_row = q + (int64_t)query_idx * q_stride0
            + (int64_t)(h0 + hh) * q_stride1;
        #pragma unroll
        for (int j = 0; j < NOPE_PER_THREAD; j++) {
            q_nope[hh][j] = __bfloat162float(q_row[dn0 + j]);
        }
        #pragma unroll
        for (int j = 0; j < ROPE_PER_THREAD; j++) {
            q_rope[hh][j] = __bfloat162float(q_row[NOPE_DIM + dr0 + j]);
        }
    }

    float m_i[HEADS_PER_CTA], l_i[HEADS_PER_CTA];
    float acc_nope[HEADS_PER_CTA][NOPE_PER_THREAD];
    float acc_rope[HEADS_PER_CTA][ROPE_PER_THREAD];
    #pragma unroll
    for (int hh = 0; hh < HEADS_PER_CTA; hh++) {
        m_i[hh] = NEG_LARGE;
        l_i[hh] = 0.0f;
        #pragma unroll
        for (int j = 0; j < NOPE_PER_THREAD; j++) acc_nope[hh][j] = 0.0f;
        #pragma unroll
        for (int j = 0; j < ROPE_PER_THREAD; j++) acc_rope[hh][j] = 0.0f;
    }

    // Double-buffered K staging. 16-byte aligned for the uint2 casts.
    __shared__ __align__(16) uint8_t s_fp8[2][NOPE_DIM];
    __shared__ __align__(16) __hip_bfloat16 s_k_rope[2][ROPE_DIM];
    __shared__ __align__(16) uint8_t s_scales_raw[2][8];

    // Main range (SWA cache), then extra range (sparse topk MLA cache).
    // The online-softmax state flows through both, matching the Triton
    // HAS_EXTRA behaviour. The host passes a zeroed extra_indptr when
    // there is no extra cache, so the extra loop body never executes.
    process_cache_range(
        query_idx, main_cache, main_cache_stride0, main_indices, main_indptr,
        main_block_size, main_num_rows, scale,
        s_fp8, s_k_rope, s_scales_raw,
        q_nope, q_rope, m_i, l_i, acc_nope, acc_rope);
    if (extra_indptr != nullptr) {
        process_cache_range(
            query_idx, extra_cache, extra_cache_stride0, extra_indices,
            extra_indptr, extra_block_size, extra_num_rows, scale,
            s_fp8, s_k_rope, s_scales_raw,
            q_nope, q_rope, m_i, l_i, acc_nope, acc_rope);
    }

    // Final normalize (with attn_sink if provided) and write out.
    #pragma unroll
    for (int hh = 0; hh < HEADS_PER_CTA; hh++) {
        const int h = h0 + hh;
        float m_final, l_final, alpha;
        if (attn_sink != nullptr) {
            const float sink = attn_sink[h];
            m_final = fmaxf(m_i[hh], sink);
            alpha = expf(m_i[hh] - m_final);
            l_final = l_i[hh] * alpha + expf(sink - m_final);
        } else {
            m_final = m_i[hh];
            alpha = 1.0f;
            l_final = l_i[hh];
        }
        const float denom = fmaxf(l_final, 1e-30f);
        const float inv_denom = (l_final > 0.0f) ? (1.0f / denom) : 0.0f;

        __hip_bfloat16* out_row = out + (int64_t)query_idx * out_stride0
            + (int64_t)h * out_stride1;
        #pragma unroll
        for (int j = 0; j < NOPE_PER_THREAD; j++) {
            out_row[dn0 + j] = __float2bfloat16(acc_nope[hh][j] * alpha * inv_denom);
        }
        #pragma unroll
        for (int j = 0; j < ROPE_PER_THREAD; j++) {
            out_row[NOPE_DIM + dr0 + j] =
                __float2bfloat16(acc_rope[hh][j] * alpha * inv_denom);
        }
    }
}


// C++ wrapper
void sparse_mla_decode_launcher(
    torch::Tensor q,                  // [B, H, D] bf16
    torch::Tensor main_cache,         // [num_blocks, block_size, 584] uint8
    torch::Tensor main_indices,       // [nnz] int32
    torch::Tensor main_indptr,        // [B+1] int32
    torch::Tensor extra_cache,        // [num_blocks, block_size, 584] uint8 (may be empty)
    torch::Tensor extra_indices,      // [nnz_extra] int32 (may be empty)
    torch::Tensor extra_indptr,       // [B+1] int32 (zeroed when no extra)
    int main_block_size,
    int main_num_rows,
    int extra_block_size,
    int extra_num_rows,
    double scale,
    torch::Tensor attn_sink,          // [H] float32 or empty
    torch::Tensor out                 // [B, H, D] bf16
) {
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(main_cache.is_cuda() && main_cache.scalar_type() == torch::kUInt8);
    TORCH_CHECK(main_indices.is_cuda() && main_indices.scalar_type() == torch::kInt32);
    TORCH_CHECK(main_indptr.is_cuda() && main_indptr.scalar_type() == torch::kInt32);
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kBFloat16);

    int B = q.size(0);
    int H = q.size(1);
    int D = q.size(2);
    TORCH_CHECK(H == HEADS, "expected H=64");
    TORCH_CHECK(D == COMB_DIM, "expected D=512");
    TORCH_CHECK(main_cache.size(2) == SLOT_BYTES, "expected cache dim 2 = 584");
    // Kernel does 8-byte vector loads from the cache; see load_token.
    TORCH_CHECK((reinterpret_cast<uintptr_t>(main_cache.data_ptr()) & 7) == 0,
                "main_cache base must be 8-byte aligned");

    const float* sink_ptr = nullptr;
    if (attn_sink.defined() && attn_sink.numel() > 0) {
        TORCH_CHECK(attn_sink.is_cuda() && attn_sink.scalar_type() == torch::kFloat32);
        sink_ptr = attn_sink.data_ptr<float>();
    }

    // Extra cache range is optional. extra_indptr must always be a valid
    // [B+1] int32 CUDA tensor (zeroed when there is no extra range) so the
    // kernel can read uniform start/end offsets; extra_cache/extra_indices
    // may be empty in that case and are never dereferenced.
    const uint8_t* extra_cache_ptr = nullptr;
    const int32_t* extra_indices_ptr = nullptr;
    const int32_t* extra_indptr_ptr = nullptr;
    int extra_cache_stride0 = 0;
    if (extra_indptr.defined() && extra_indptr.numel() > 0) {
        TORCH_CHECK(extra_indptr.is_cuda() && extra_indptr.scalar_type() == torch::kInt32);
        TORCH_CHECK(extra_indptr.numel() == (int64_t)B + 1,
                    "expected extra_indptr [B+1]");
        extra_indptr_ptr = extra_indptr.data_ptr<int32_t>();
    }
    if (extra_cache.defined() && extra_cache.numel() > 0) {
        TORCH_CHECK(extra_cache.is_cuda() && extra_cache.scalar_type() == torch::kUInt8);
        TORCH_CHECK(extra_cache.size(2) == SLOT_BYTES,
                    "expected extra cache dim 2 = 584");
        TORCH_CHECK((reinterpret_cast<uintptr_t>(extra_cache.data_ptr()) & 7) == 0,
                    "extra_cache base must be 8-byte aligned");
        extra_cache_ptr = extra_cache.data_ptr<uint8_t>();
        extra_cache_stride0 = (int)extra_cache.stride(0);
    }
    if (extra_indices.defined() && extra_indices.numel() > 0) {
        TORCH_CHECK(extra_indices.is_cuda() && extra_indices.scalar_type() == torch::kInt32);
        extra_indices_ptr = extra_indices.data_ptr<int32_t>();
    }

    dim3 grid(B, HEAD_GROUPS);
    dim3 block(THREADS);

    hipStream_t stream = at::hip::getCurrentHIPStream();
    sparse_mla_decode_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __hip_bfloat16*>(q.data_ptr()),
        (int)q.stride(0),
        (int)q.stride(1),
        main_cache.data_ptr<uint8_t>(),
        (int)main_cache.stride(0),
        main_indices.data_ptr<int32_t>(),
        main_indptr.data_ptr<int32_t>(),
        main_block_size,
        main_num_rows,
        extra_cache_ptr,
        extra_cache_stride0,
        extra_indices_ptr,
        extra_indptr_ptr,
        extra_block_size,
        extra_num_rows,
        (float)scale,
        sink_ptr,
        reinterpret_cast<__hip_bfloat16*>(out.data_ptr()),
        (int)out.stride(0),
        (int)out.stride(1)
    );
}

}  // namespace rdna2_mla

// Global wrapper matching the declaration in ops.h. torch_bindings.cpp
// registers this symbol under the _rocm_C library.
void sparse_mla_decode_rdna2(
    torch::Tensor q,                  // [B, H, D] bf16
    torch::Tensor main_cache,         // [num_blocks, block_size, 584] uint8
    torch::Tensor main_indices,       // [nnz] int32
    torch::Tensor main_indptr,        // [B+1] int32
    torch::Tensor extra_cache,        // [num_blocks, block_size, 584] uint8 (may be empty)
    torch::Tensor extra_indices,      // [nnz_extra] int32 (may be empty)
    torch::Tensor extra_indptr,       // [B+1] int32 (zeroed when no extra)
    int64_t main_block_size,
    int64_t main_num_rows,
    int64_t extra_block_size,
    int64_t extra_num_rows,
    double scale,
    torch::Tensor attn_sink,          // [H] fp32 or empty
    torch::Tensor out                 // [B, H, D] bf16
) {
    rdna2_mla::sparse_mla_decode_launcher(
        q, main_cache, main_indices, main_indptr,
        extra_cache, extra_indices, extra_indptr,
        static_cast<int>(main_block_size),
        static_cast<int>(main_num_rows),
        static_cast<int>(extra_block_size),
        static_cast<int>(extra_num_rows),
        scale, attn_sink, out);
}
