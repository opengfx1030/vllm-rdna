// SPDX-License-Identifier: Apache-2.0
// causal_conv1d_update kernel for AMD RDNA2 (gfx1030).
//
// Single-token decode path: per-batch depthwise FIR filter with state
// (conv_state) update. One program per (batch, dim_block).
//
// Layout:
//   x:            [batch, dim, seqlen=1] fp16 (channel-last)
//   conv_state:   [num_cache_lines, dim, state_len] fp16/fp32
//   weight:       [dim, width] fp16 (channel-last)
//   bias:         [dim] fp16 or None
//   out:          [batch, dim, seqlen=1] fp16
//
// Per-program work:
//   1. Load state_len prior tokens from conv_state[slot, :, :]
//   2. Shift-left and append x[b, :, 0] to form the new state
//   3. Store the updated state back to conv_state[slot, :, :]
//   4. Compute out[b, :, 0] = sum_j weight[:, j] * state_at_offset_j + bias
//   5. Apply SiLU activation if requested
//
// fp16 in/out, fp32 accumulation. One warp (32 threads) per dim_block of
// 32 channels. dim must be a multiple of 32; state_len is small (width-1,
// typically 3 or 4 for GDN).
//
// This kernel is cudagraph-safe: all scratch is in registers, no global
// memory allocation, no Triton JIT scratch pointers.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {

constexpr int CONV1D_WARP_SIZE = 32;

__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + __expf(-x));
}

// One program per (batch_idx, dim_block_idx).
// BLOCK_DIM channels per program, one warp.
__global__ void causal_conv1d_update_kernel(
    const __half* __restrict__ x,             // [batch, dim, 1]
    __half* __restrict__ conv_state,          // [num_cache_lines, dim, state_len]
    const __half* __restrict__ weight,        // [dim, width]
    const __half* __restrict__ bias,          // [dim] or nullptr
    __half* __restrict__ out,                 // [batch, dim, 1]
    const int32_t* __restrict__ conv_state_indices,  // [batch]
    const int batch,
    const int dim,
    const int state_len,
    const int width,
    const int num_cache_lines,
    const bool has_bias,
    const bool silu_activation) {

    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch) return;

    const int dim_block_idx = blockIdx.y;
    const int tid = threadIdx.x;  // 0..31
    const int c = dim_block_idx * CONV1D_WARP_SIZE + tid;

    // Load conv_state slot for this batch
    const int32_t slot = conv_state_indices[batch_idx];
    if (slot < 0 || slot >= num_cache_lines) return;
    if (c >= dim) return;

    // Pointers for this batch/channel
    // x is [batch, dim, 1], channel-last: stride(dim) = 1, stride(batch) = dim
    const __half* x_ptr = x + batch_idx * dim + c;
    __half* out_ptr = out + batch_idx * dim + c;

    // conv_state layout: [num_cache_lines, dim, state_len]
    // stride(slot) = dim * state_len, stride(c) = state_len, stride(t) = 1
    __half* state_ptr = conv_state + slot * dim * state_len + c * state_len;

    // weight layout: [dim, width], channel-last: stride(dim) = width, stride(w) = 1
    const __half* w_ptr = weight + c * width;

    // Load prior state and new input
    float new_x = __half2float(x_ptr[0]);

    // Shift state left by 1, write new_x at position (state_len - 1)
    // Read order: state[1], state[2], ..., state[state_len-1], new_x
    for (int t = 0; t < state_len - 1; ++t) {
        state_ptr[t] = state_ptr[t + 1];
    }
    state_ptr[state_len - 1] = __float2half(new_x);

// Compute output: out = sum_{j=0..width-1} weight[c, j] * x[t-j].
    //     conv_state stores state_len prior inputs in shift-left order so
    // state_ptr[k] = x[t-state_len+1+k] for k in [0..state_len-1], with
    // state_ptr[state_len-1] freshly appended as x[t]. Canonical causal-conv1d
    // pairs w[j] with x[t-j]: state_ptr[k] = x[t-state_len+1+k] so
    // state_ptr[k] pairs with w[k+1] = w[state_len - (state_len-1-k)].
    // In other words, state_ptr[k] pairs with w[state_len - 1 - k]
    // where state_ptr is the post-shift layout. Equivalently:
    //   j=0:               w[0] * x[t-3]     = w[0] * state_ptr[state_len-3]
    //   j=1:               w[1] * x[t-2]     = w[1] * state_ptr[state_len-2]
    //   ...
    //   j=state_len-1:     w[state_len-1] * x[t-1] = w[state_len-1] * state_ptr[state_len-1]
    //   j=state_len:       w[state_len] * x[t]  (the fresh token, NOT in state)
    // So state_ptr[k] pairs with w[k+1] (when state[k] holds x[t-k-1] post-shift),
    // which is equivalent to state_ptr[k] pairing with w[state_len-1-k] when
    // state_ptr is indexed differently. Iterating k in [0..state_len-1] (NOT
    // j in [0..width-1]) avoids the OOB read on state_ptr[state_len] that
    // the original code had. Note: this matches the fwd kernel's shift-after
    // pattern (compute first with pre-shift layout, shift for next iteration),
    // but here for single-token decode we always shift first.
    float acc = has_bias ? __half2float(bias[c]) : 0.0f;
    for (int k = 0; k < state_len; ++k) {
        float s = __half2float(state_ptr[k]);
        float w = __half2float(w_ptr[k + 1]);
        acc += s * w;
    }

    if (silu_activation) {
        acc = silu_f32(acc);
    }

    out_ptr[0] = __float2half(acc);
}

}  // namespace

void causal_conv1d_update_rdna2(
    torch::Tensor x,                  // [batch, dim, 1] fp16
    torch::Tensor conv_state,         // [num_cache_lines, dim, state_len] fp16
    torch::Tensor weight,            // [dim, width] fp16
    torch::Tensor bias,              // [dim] fp16 or None
    torch::Tensor out,               // [batch, dim, 1] fp16
    torch::Tensor conv_state_indices,  // [batch] int32
    bool silu_activation) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(conv_state.is_cuda(), "conv_state must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kHalf, "x must be fp16");
    TORCH_CHECK(conv_state.scalar_type() == at::kHalf, "conv_state must be fp16");
    TORCH_CHECK(weight.scalar_type() == at::kHalf, "weight must be fp16");
    TORCH_CHECK(out.scalar_type() == at::kHalf, "out must be fp16");

    const int batch = x.size(0);
    const int dim = x.size(1);
    const int state_len = conv_state.size(2);
    const int width = weight.size(1);
    const int num_cache_lines = conv_state.size(0);
    // Python may pass torch.empty(0, dtype=...) for None bias. Such tensors
    // have defined()==true but numel()==0; data_ptr() returns a pointer to
    // a 0-byte allocation that any read overflows. Gate the kernel pointer
    // on numel() > 0 to avoid OOB reads.
    const bool has_bias = bias.defined() && bias.numel() > 0;

    TORCH_CHECK(dim % CONV1D_WARP_SIZE == 0,
                "causal_conv1d_update_rdna2 requires dim % 32 == 0, got ", dim);
    TORCH_CHECK(width == state_len + 1,
                "causal_conv1d_update_rdna2 requires width == state_len + 1, got width=",
                width, " state_len=", state_len);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    auto stream = at::cuda::getCurrentCUDAStream();

    dim3 grid(batch, dim / CONV1D_WARP_SIZE);
    dim3 block(CONV1D_WARP_SIZE);

    causal_conv1d_update_kernel<<<grid, block, 0, stream.stream()>>>(
        reinterpret_cast<const __half*>(x.data_ptr()),
        reinterpret_cast<__half*>(conv_state.data_ptr()),
        reinterpret_cast<const __half*>(weight.data_ptr()),
        has_bias ? reinterpret_cast<const __half*>(bias.data_ptr()) : nullptr,
        reinterpret_cast<__half*>(out.data_ptr()),
        conv_state_indices.data_ptr<int32_t>(),
        batch, dim, state_len, width, num_cache_lines,
        has_bias, silu_activation);
}

// =============================================================================
// causal_conv1d_fwd kernel for AMD RDNA2 (gfx1030).
//
// Varlen prefill path. One program per (sequence, dim_block).
//   x:           [dim, cu_seqlen] fp16 (channel-last: stride_dim=1)
//   weight:      [dim, width] fp16 (channel-last: stride_dim=width)
//   bias:        [dim] fp16 or None
//   conv_state:  [num_cache_lines, dim, state_len] fp16
//   query_start_loc: [batch+1] int32
//   cache_indices:   [batch] int32
//   has_initial_state: [batch] bool or None
//   out:         [dim, cu_seqlen] fp16 (in-place)
//   silu_activation: bool
//
// Per-sequence work:
//   1. Load state_len prior tokens from conv_state[slot, :, :] if has_initial_state
//   2. For each token: shift state left, append x, compute out (FIR + bias + SiLU)
//   3. (Optional) write back the last state_len tokens to conv_state[slot, :, :]
//      Skipped when cache_indices == -1 (writeback disabled) — caller is
//      responsible for managing conv_state in that case (matches eager semantics
//      when conv_state is None). We always copy back when cache_indices is valid.
//
// fp16 in/out, fp32 accumulation. One warp (32 threads) per dim_block of
// 32 channels. dim must be a multiple of 32; width in [2..5] (GDN widths).
//
// Cudagraph-safe: scratch state lives in shared memory, no global allocations,
// no Triton JIT scratch pointers.

namespace {

constexpr int CONV1D_FWD_MAX_WIDTH = 5;

__global__ void causal_conv1d_fwd_kernel(
    const __half* __restrict__ x,             // [dim, cu_seqlen]
    const __half* __restrict__ weight,        // [dim, width]
    const __half* __restrict__ bias,          // [dim] or nullptr
    __half* __restrict__ conv_state,          // [num_cache_lines, dim, state_len] (transposed view)
    __half* __restrict__ out,                 // [dim, cu_seqlen]
    const int32_t* __restrict__ query_start_loc,   // [batch+1]
    const int32_t* __restrict__ cache_indices,     // [batch]
    const bool* __restrict__ has_initial_state,    // [batch] or nullptr
    const int batch,
    const int dim,
    const int width,
    const int state_len,
    const int num_cache_lines,
    const int64_t stride_istate_seq,
    const int64_t stride_istate_dim,
    const int64_t stride_istate_token,
    const int64_t stride_x_token,
    const bool has_bias,
    const bool silu_activation) {

    const int seq_idx = blockIdx.x;
    if (seq_idx >= batch) return;

    const int dim_block_idx = blockIdx.y;
    const int tid = threadIdx.x;  // 0..31
    const int c = dim_block_idx * CONV1D_WARP_SIZE + tid;
    if (c >= dim) return;

    extern __shared__ __half smem[];
    __half* state = smem + tid * state_len;  // [warp_size, state_len]

    // Sequence bounds in cu_seqlen
    const int32_t seq_start = query_start_loc[seq_idx];
    const int32_t seq_end = query_start_loc[seq_idx + 1];
    const int seqlen = seq_end - seq_start;
    if (seqlen <= 0) return;

    // Load conv_state slot. The conv_state layout is [num_cache_lines, dim, state_len]
    // but vLLM stores it as [num_cache_lines, state_len, dim] (state_len is the
    // middle axis) and transposes(-1, -2) before passing to the kernel. So the
    // actual memory layout has:
    //   stride_istate_seq = stride(0)   (e.g. 802816 for dim=5120, state_len=3)
    //   stride_istate_dim = stride(1)   (e.g. 1 — dim is contiguous)
    //   stride_istate_token = stride(2) (e.g. 5120 — state_len has stride dim)
    // Strides are passed at runtime because the layout is non-contiguous.
    const int32_t slot = cache_indices[seq_idx];
    __half* slot_base = nullptr;
    bool load_initial = false;
    if (slot >= 0 && slot < num_cache_lines) {
        slot_base = conv_state + slot * stride_istate_seq + c * stride_istate_dim;
        load_initial = (has_initial_state != nullptr) && has_initial_state[seq_idx];
    }

    // Load initial state (state_len prior tokens) or zero it.
    // The conv_state is stored as (num_cache_lines, state_len, dim) in memory
    // and passed to us as (num_cache_lines, dim, state_len) via transpose(-1,-2).
    // So slot_base[t * stride_istate_token] reads conv_state[slot, c, t] in the
    // transposed view (where c is already baked into slot_base's offset).
    if (load_initial) {
        for (int t = 0; t < state_len; ++t) {
            state[t] = slot_base[t * stride_istate_token];
        }
    } else {
        for (int t = 0; t < state_len; ++t) {
            state[t] = __float2half(0.0f);
        }
    }

    // Load bias (broadcast across threads — each thread handles one channel)
    const float bias_val = has_bias ? __half2float(bias[c]) : 0.0f;

    // Load weights into registers
    float w[CONV1D_FWD_MAX_WIDTH];
    for (int j = 0; j < width; ++j) {
        w[j] = __half2float(weight[c * width + j]);
    }

    // Process each token in the sequence
    // Channel-last: x[d, t] = x_ptr[d * 1 + t * stride_x_token]. The stride
    // is passed at runtime — it's typically the next power of 2 >= dim to
    // satisfy torch's alignment requirements (e.g. dim=5120 → stride=8192).
    // Note: x_ptr must also skip past the tokens of preceding sequences in
    // the cu_seqlen-flattened token buffer. seq_idx=0 reads from x[d, 0];
    // seq_idx=k reads from x[d, cu_seqlen[k]] = x[d, seq_start].
    const __half* x_ptr = x + c + seq_start * stride_x_token;
    __half* out_ptr = out + c + seq_start * stride_x_token;

    // State layout BEFORE the FIR: state[k] = init[k] for k in [0..state_len-1]
    // (newest at high index, NO shift yet). The canonical causal-conv1d formula
    // pairs w[j] with x[t-j] for j in [0..width-1]. With width=state_len+1:
    //   j=0:               w[0] * init[0]            (x[t-3])
    //   j=1:               w[1] * init[1]            (x[t-2])
    //   ...
    //   j=state_len-1:     w[state_len-1] * init[state_len-1]   (x[t-1])
    //   j=state_len:       w[state_len] * x[t]       (the fresh token, not in state)
    // So state[k] pairs with w[k] (NOT w[k+1]!), and w[state_len] reads x[t].
    // Then for the next iteration, shift state left by 1 and append x[t] at
    // state[state_len-1] so the layout is preserved.
    for (int t = 0; t < seqlen; ++t) {
        // Compute out[d, t] = w[0..state_len-1] * state[0..state_len-1] + w[state_len] * x[t]
        float acc = bias_val;
        for (int k = 0; k < state_len; ++k) {
            acc += w[k] * __half2float(state[k]);
        }
        acc += w[state_len] * __half2float(x_ptr[t * stride_x_token]);
        if (silu_activation) {
            acc = silu_f32(acc);
        }
        out_ptr[t * stride_x_token] = __float2half(acc);

        // Shift state left and append x[d, t] for the next iteration.
        // (For the last token, this produces the post-convolution conv_state.)
        for (int s = 0; s < state_len - 1; ++s) {
            state[s] = state[s + 1];
        }
        state[state_len - 1] = x_ptr[t * stride_x_token];
    }

    // Write back the last state_len tokens to conv_state[slot, :, :]
    // using the transposed-view stride_istate_token.
    if (slot_base != nullptr) {
        for (int t = 0; t < state_len; ++t) {
            slot_base[t * stride_istate_token] = state[t];
        }
    }
}

}  // namespace

void causal_conv1d_fwd_rdna2(
    torch::Tensor x,                  // [dim, cu_seqlen] fp16
    torch::Tensor weight,             // [dim, width] fp16
    torch::Tensor bias,               // [dim] fp16 or None
    torch::Tensor conv_state,         // [num_cache_lines, dim, state_len] fp16
    torch::Tensor query_start_loc,    // [batch+1] int32
    torch::Tensor cache_indices,      // [batch] int32
    torch::Tensor has_initial_state,  // [batch] bool or None
    torch::Tensor out,                // [dim, cu_seqlen] fp16
    bool silu_activation) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kHalf, "x must be fp16");
    TORCH_CHECK(weight.scalar_type() == at::kHalf, "weight must be fp16");
    TORCH_CHECK(out.scalar_type() == at::kHalf, "out must be fp16");
    TORCH_CHECK(conv_state.scalar_type() == at::kHalf, "conv_state must be fp16");

    TORCH_CHECK(x.dim() == 2, "causal_conv1d_fwd_rdna2: x must be [dim, cu_seqlen], got dim=", x.dim());
    TORCH_CHECK(x.stride(0) == 1,
                "causal_conv1d_fwd_rdna2: x must be channel-last (stride[0]==1), got stride[0]=",
                x.stride(0));
    TORCH_CHECK(weight.stride(1) == 1,
                "causal_conv1d_fwd_rdna2: weight must be channel-last (stride[1]==1)");

    const int dim = x.size(0);
    const int width = weight.size(1);
    const int state_len = conv_state.size(2);
    const int num_cache_lines = conv_state.size(0);
    const int batch = query_start_loc.size(0) - 1;
    // Python may pass torch.empty(0, dtype=...) for None bias/has_initial_state.
    // Such tensors have defined()==true but numel()==0; data_ptr() returns a
    // pointer to a 0-byte allocation that any read overflows. Gate the kernel
    // pointer on numel() > 0 to avoid OOB reads.
    const bool has_bias = bias.defined() && bias.numel() > 0;
    const bool has_init = has_initial_state.defined() && has_initial_state.numel() > 0;

    TORCH_CHECK(dim % CONV1D_WARP_SIZE == 0,
                "causal_conv1d_fwd_rdna2 requires dim % 32 == 0, got ", dim);
    TORCH_CHECK(width >= 2 && width <= CONV1D_FWD_MAX_WIDTH,
                "causal_conv1d_fwd_rdna2 requires 2 <= width <= ", CONV1D_FWD_MAX_WIDTH,
                ", got ", width);
    TORCH_CHECK(width == state_len + 1,
                "causal_conv1d_fwd_rdna2 requires width == state_len + 1, got width=",
                width, " state_len=", state_len);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    auto stream = at::cuda::getCurrentCUDAStream();

    dim3 grid(batch, dim / CONV1D_WARP_SIZE);
    dim3 block(CONV1D_WARP_SIZE);
    const size_t smem_bytes = CONV1D_WARP_SIZE * state_len * sizeof(__half);

    // stride_x_token: row stride of x (and out) along the token dim. The
    // vLLM weight loader allocates x with stride[1] = next_pow2(dim) for
    // alignment (e.g. dim=5120 → stride=8192). Hardcoding dim here would
    // index the wrong memory lane.
    const int64_t stride_x_token = (int64_t)x.stride(1);

    if (std::getenv("VLLM_CONV1D_DEBUG")) {
        printf("[CONV1D-CPP] x.ptr=%p weight.ptr=%p bias.ptr=%p conv_state.ptr=%p out.ptr=%p\n",
               x.data_ptr(), weight.data_ptr(),
               has_bias ? bias.data_ptr() : nullptr,
               conv_state.data_ptr(), out.data_ptr());
        printf("[CONV1D-CPP] x.stride=(%ld,%ld) weight.stride=(%ld,%ld) conv_state.stride=(%ld,%ld,%ld) out.stride=(%ld,%ld)\n",
               x.stride(0), x.stride(1),
               weight.stride(0), weight.stride(1),
               conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
               out.stride(0), out.stride(1));
        printf("[CONV1D-CPP] dim=%d width=%d state_len=%d batch=%d num_cache_lines=%d stride_x_token=%ld\n",
               dim, width, state_len, batch, num_cache_lines, stride_x_token);
        printf("[CONV1D-CPP] query_start_loc.ptr=%p cache_indices.ptr=%p has_init_state.ptr=%p has_bias=%d\n",
               query_start_loc.data_ptr<int32_t>(), cache_indices.data_ptr<int32_t>(),
               has_init ? has_initial_state.data_ptr<bool>() : nullptr, has_bias);
    }

    causal_conv1d_fwd_kernel<<<grid, block, smem_bytes, stream.stream()>>>(
        reinterpret_cast<const __half*>(x.data_ptr()),
        reinterpret_cast<const __half*>(weight.data_ptr()),
        has_bias ? reinterpret_cast<const __half*>(bias.data_ptr()) : nullptr,
        reinterpret_cast<__half*>(conv_state.data_ptr()),
        reinterpret_cast<__half*>(out.data_ptr()),
        query_start_loc.data_ptr<int32_t>(),
        cache_indices.data_ptr<int32_t>(),
        has_init ? has_initial_state.data_ptr<bool>() : nullptr,
        batch, dim, width, state_len, num_cache_lines,
        (int64_t)conv_state.stride(0),
        (int64_t)conv_state.stride(1),
        (int64_t)conv_state.stride(2),
        stride_x_token,
        has_bias, silu_activation);
}
