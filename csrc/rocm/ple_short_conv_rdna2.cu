// T49 — Dilated PLE short-conv (depthwise) for Qwen3.8-Flash-Next on gfx1030.
//
// The PLE block on a PLE-layer (1-indexed layer in ``ple_layer_ids``)
// inserts a depthwise dilated 1D conv between the multi-stream combine
// and the next HC mix. The Triton/``F.conv1d`` reference path lives in
// ``_short_conv_dilated_decode_batched`` and
// ``_short_conv_dilated_prefill_batched`` of
// ``vllm/models/qwen4_exp/amd/ple_layer.py``; both share the same
// underlying math:
//
//     history[b, d, :] = [initial_state[b, d, :], x[b, d, :]]   // state_len + L
//     out[b, d, t]     = silu( sum_{k=0}^{K-1} w[d, k] * history[b, d, k*D + t] )
//     new_state[b, d, i] = history[b, d, L + i]                 // last state_len
//
// where D is the dilation (``short_conv_dilation``) and K = D + state_len
// + 1 (the dilated kernel spans state_len * D + 1 taps).
//
// We collapse the per-channel work to one program per (request, channel
// block of CH_TILE channels); each thread within a CTA streams one
// channel. Channels are independent (depthwise conv1d).
//
// Opt-in: VLLM_RDNA_PLE_CONV_HIP=1 + on_gfx10x() in the Python dispatcher.
// Default off (torch F.conv1d fallback) until the HIP path is verified
// end-to-end.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {

__device__ __forceinline__ half silu_h(half x) {
  float f = __half2float(x);
  return __float2half(f / (1.f + __expf(-f)));
}

// ---------------------------------------------------------------------------
// ple_short_conv_decode_rdna2:
//   x:            [B, D] fp16 (current step activations)
//   conv_state:   [num_lines, D, state_len] fp16, in-place updated
//   weight:       [D, K] fp16 (the dilated conv weights)
//   bias:         [D]   fp16 (optional)
//   out:          [B, D] fp16
//   state_idx:    [B]   int32 (per-batch slot in conv_state)
//   has_init:     [B]   uint8/bool (per-batch use cached state)
//   silu:         bool
//   dilation:     int32 (>= 1)
//   state_len:    int32 (K - dilation)
//   null_block:   int32 (rows whose state_idx == null_block write 0, leave state)
// ---------------------------------------------------------------------------

__global__ void ple_short_conv_decode_kernel(
    const half* __restrict__ x, half* __restrict__ conv_state,
    const half* __restrict__ weight, const half* __restrict__ bias,
    half* __restrict__ out, const int32_t* __restrict__ state_idx,
    const uint8_t* __restrict__ has_init, int B, int D, int K, int state_len,
    int dilation, bool silu, int null_block) {
  const int b = blockIdx.x;
  const int d = blockIdx.y * blockDim.x + threadIdx.x;
  if (b >= B || d >= D) return;

  const int slot = state_idx[b];
  const bool valid = slot != null_block;
  const int safe_slot = max(slot, 0);
  const bool use_init = valid && (has_init == nullptr || has_init[b] != 0);

  // history[d, k] for k in [0..K): prefer cached state for first state_len,
  // then x for the last dilation taps.
  half history_buf[64];  // K up to 64 (state_len + dilation + 1)
  if (K > 64) {
    // Should not happen for Qwen4Exp PLE; bail by writing zero output.
    if (valid) out[b * D + d] = __float2half(0.f);
    return;
  }

  // Load state into history[0..state_len)
  if (state_len > 0 && use_init) {
    const half* st = conv_state + (size_t)safe_slot * D * state_len
                   + (size_t)d * state_len;
#pragma unroll
    for (int i = 0; i < state_len; i++) {
      history_buf[i] = st[i];
    }
  } else {
#pragma unroll
    for (int i = 0; i < 64; i++) history_buf[i] = __float2half(0.f);
  }
  // Load x[d] into history[state_len] (single new token)
  history_buf[state_len] = x[b * D + d];

  // Convolution: sum_{k=0..K-1} w[d, k] * history_buf[k * dilation]
  float acc = 0.f;
#pragma unroll
  for (int k = 0; k < 64; k++) {
    if (k < K) {
      acc += __half2float(weight[d * K + k])
           * __half2float(history_buf[k * dilation]);
    }
  }
  if (bias != nullptr) acc += __half2float(bias[d]);
  if (silu) acc = acc / (1.f + __expf(-acc));

  if (valid) out[b * D + d] = __float2half(acc);

  // Write next state: history_buf[state_len..K) (the most recent
  // state_len + 1 taps), shifted to keep only the last state_len.
  // Simpler: shift left by dilation (the model's expected behaviour
  // mirrors causal_conv1d_update: drop the oldest ``dilation`` taps,
  // append the new tap).
  if (valid && state_len > 0) {
    half* st = conv_state + (size_t)safe_slot * D * state_len
             + (size_t)d * state_len;
    // Left-shift by ``dilation`` (the PLE dilated conv emits
    // ``state_len / dilation`` new state rows per step in the
    // eager path, but the conv_state only has state_len slots, so
    // we keep the last state_len taps).
    for (int i = 0; i < state_len; i++) {
      int src = i + dilation;
      half v = (src <= state_len) ? history_buf[src] : history_buf[state_len];
      st[i] = v;
    }
  }
}

void ple_short_conv_decode(
    const at::Tensor& x, at::Tensor& conv_state,
    const at::Tensor& weight, const std::optional<at::Tensor>& bias,
    at::Tensor& out, const at::Tensor& state_idx,
    const std::optional<at::Tensor>& has_init,
    int64_t dilation, int64_t state_len, bool silu, int64_t null_block) {
  TORCH_CHECK(x.is_cuda() && conv_state.is_cuda() && weight.is_cuda(),
              "ple_short_conv_decode_rdna2: tensors on HIP/CUDA");
  TORCH_CHECK(x.scalar_type() == at::kHalf, "fp16 only");
  TORCH_CHECK(conv_state.scalar_type() == at::kHalf, "conv_state fp16 only");
  TORCH_CHECK(weight.scalar_type() == at::kHalf, "weight fp16 only");
  TORCH_CHECK(out.scalar_type() == at::kHalf, "out fp16 only");
  TORCH_CHECK(x.dim() == 2, "x must be [B, D]");
  TORCH_CHECK(weight.dim() == 2, "weight must be [D, K]");
  const int B = x.size(0);
  const int D = x.size(1);
  const int K = weight.size(1);
  TORCH_CHECK(weight.size(0) == D, "weight.size(0) must equal D");
  TORCH_CHECK(K == state_len + dilation + 1, "K must equal state_len + dilation + 1");
  TORCH_CHECK(K <= 64, "K must be <= 64");
  TORCH_CHECK(conv_state.size(2) == state_len, "conv_state.size(2) must equal state_len");
  TORCH_CHECK(state_idx.scalar_type() == at::kInt, "state_idx int32");
  const at::cuda::OptionalCUDAGuard guard(x.device());

  const int TPB = 64;
  dim3 grid(B, (D + TPB - 1) / TPB);
  ple_short_conv_decode_kernel<<<grid, TPB>>>(
      reinterpret_cast<const half*>(x.const_data_ptr()),
      reinterpret_cast<half*>(conv_state.mutable_data_ptr()),
      reinterpret_cast<const half*>(weight.const_data_ptr()),
      bias.has_value() ? reinterpret_cast<const half*>(bias->const_data_ptr())
                       : nullptr,
      reinterpret_cast<half*>(out.mutable_data_ptr()),
      state_idx.const_data_ptr<int32_t>(),
      has_init.has_value() ? has_init->const_data_ptr<uint8_t>() : nullptr,
      B, D, K, (int)state_len, (int)dilation, silu, (int)null_block);
}

// ---------------------------------------------------------------------------
// ple_short_conv_prefill_rdna2:
//   x_packed:     [B, D, max_len]   fp16 (input tokens, padded with 0)
//   initial_state:[B, D, state_len] fp16 (cached state, or 0)
//   weight:       [D, K]            fp16
//   bias:         [D]               fp16 (optional)
//   out:          [B, D, max_len]   fp16
//   lengths:      [B]               int32 (true length per request; padded
//                                  positions are dropped via valid_mask)
//   valid_state:  [B]               uint8/bool (use cached state vs zero)
//   silu, dilation, state_len, null_block as above
// ---------------------------------------------------------------------------

__global__ void ple_short_conv_prefill_kernel(
    const half* __restrict__ x_packed, const half* __restrict__ init_state,
    const half* __restrict__ weight, const half* __restrict__ bias,
    half* __restrict__ out, const int32_t* __restrict__ lengths,
    const uint8_t* __restrict__ valid_state,
    int B, int D, int K, int state_len, int max_len, int dilation,
    bool silu, int null_block_slot) {
  // One program per (request, channel); each thread does one channel.
  // For each output position t in [0, max_len):
  //   history[d, i] = (i < state_len) ? init_state[d, i] : x_packed[d, i - state_len]
  //   out[d, t]     = silu( sum_k w[d, k] * history[d, k * dilation + t] )
  //
  // We unroll K taps (K up to 64).
  const int b = blockIdx.x;
  const int d = blockIdx.y * blockDim.x + threadIdx.x;
  if (b >= B || d >= D) return;

  const bool vstate = valid_state == nullptr || valid_state[b] != 0;
  const int L = lengths[b];

  // History for one channel: in registers.
  // Index convention: history[i] where i in [0, K * dilation).
  // taps covered: dilation * k for k in [0..K-1].
  // For position t: tap value is history[k * dilation + t].

  float acc[128];  // max_len up to 128
#pragma unroll 1
  for (int t = 0; t < 128; t++) {
    if (t < max_len) acc[t] = 0.f;
  }

#pragma unroll
  for (int k = 0; k < 64; k++) {
    if (k >= K) break;
    // Index in history buffer for tap k of position t:
    //   idx = k * dilation + t
    // Equivalent: history[k * dilation + t] for each t in [0, max_len).
    float w_k = __half2float(weight[d * K + k]);

    // We'll contribute w_k * history[k * dilation + t] to acc[t].
    // Read history for k in [k * dilation, k * dilation + max_len).
    // History layout: init_state for [0..state_len), then x_packed
    // for [state_len..state_len + max_len) (== K * dilation for the
    // largest tap).
    const int base_idx = k * dilation;
#pragma unroll
    for (int t = 0; t < 128; t++) {
      if (t >= max_len) break;
      const int idx = base_idx + t;
      float h = 0.f;
      if (idx < state_len) {
        if (vstate) h = __half2float(init_state[b * D * state_len
                                              + d * state_len + idx]);
      } else {
        const int xt = idx - state_len;
        if (xt < max_len) {
          h = __half2float(x_packed[b * D * max_len + d * max_len + xt]);
        }
      }
      acc[t] += w_k * h;
    }
  }

#pragma unroll
  for (int t = 0; t < 128; t++) {
    if (t < max_len) {
      float v = acc[t];
      if (bias != nullptr) v += __half2float(bias[d]);
      if (silu) v = v / (1.f + __expf(-v));
      // Mask out positions >= L (pad tokens).
      if (t >= L) v = 0.f;
      out[b * D * max_len + d * max_len + t] = __float2half(v);
    }
  }
}

void ple_short_conv_prefill(
    const at::Tensor& x_packed, const at::Tensor& init_state,
    const at::Tensor& weight, const std::optional<at::Tensor>& bias,
    at::Tensor& out, const at::Tensor& lengths,
    const std::optional<at::Tensor>& valid_state,
    int64_t dilation, int64_t state_len, bool silu) {
  TORCH_CHECK(x_packed.is_cuda(), "x_packed on HIP/CUDA");
  TORCH_CHECK(x_packed.dim() == 3, "x_packed must be [B, D, max_len]");
  TORCH_CHECK(x_packed.scalar_type() == at::kHalf, "fp16 only");
  TORCH_CHECK(init_state.scalar_type() == at::kHalf, "init_state fp16");
  TORCH_CHECK(weight.scalar_type() == at::kHalf, "weight fp16");
  TORCH_CHECK(out.scalar_type() == at::kHalf, "out fp16");
  TORCH_CHECK(lengths.scalar_type() == at::kInt, "lengths int32 only");
  const int B = x_packed.size(0);
  const int D = x_packed.size(1);
  const int max_len = x_packed.size(2);
  const int K = weight.size(1);
  TORCH_CHECK(K == state_len + dilation + 1, "K must equal state_len + dilation + 1");
  TORCH_CHECK(K <= 64, "K must be <= 64");
  TORCH_CHECK(max_len <= 128, "max_len must be <= 128 (per-CTA channel tiles)");
  TORCH_CHECK(init_state.size(2) == state_len, "init_state.size(2) must equal state_len");
  const at::cuda::OptionalCUDAGuard guard(x_packed.device());

  const int TPB = 64;
  dim3 grid(B, (D + TPB - 1) / TPB);
  ple_short_conv_prefill_kernel<<<grid, TPB>>>(
      reinterpret_cast<const half*>(x_packed.const_data_ptr()),
      reinterpret_cast<const half*>(init_state.const_data_ptr()),
      reinterpret_cast<const half*>(weight.const_data_ptr()),
      bias.has_value() ? reinterpret_cast<const half*>(bias->const_data_ptr())
                       : nullptr,
      reinterpret_cast<half*>(out.mutable_data_ptr()),
      lengths.const_data_ptr<int32_t>(),
      valid_state.has_value() ? valid_state->const_data_ptr<uint8_t>() : nullptr,
      B, D, K, (int)state_len, max_len, (int)dilation, silu, /*null_block_slot=*/-1);
}

}  // namespace

// ---------------------------------------------------------------------------
// Public host wrappers.
// ---------------------------------------------------------------------------

void ple_short_conv_decode_rdna2(
    torch::Tensor x,            // [B, D] fp16
    torch::Tensor conv_state,   // [num_lines, D, state_len] fp16 (in-place)
    torch::Tensor weight,       // [D, K] fp16
    torch::Tensor bias,         // [D] fp16 or undefined
    torch::Tensor out,          // [B, D] fp16
    torch::Tensor state_idx,    // [B] int32
    torch::Tensor has_init,     // [B] uint8 or undefined
    int64_t dilation, int64_t state_len, bool silu, int64_t null_block) {
  std::optional<at::Tensor> bias_opt;
  if (bias.defined()) bias_opt = bias;
  std::optional<at::Tensor> init_opt;
  if (has_init.defined()) init_opt = has_init;
  ple_short_conv_decode(x, conv_state, weight, bias_opt, out, state_idx,
                        init_opt, dilation, state_len, silu, null_block);
}

void ple_short_conv_prefill_rdna2(
    torch::Tensor x_packed,     // [B, D, max_len] fp16
    torch::Tensor init_state,   // [B, D, state_len] fp16
    torch::Tensor weight,       // [D, K] fp16
    torch::Tensor bias,         // [D] fp16 or undefined
    torch::Tensor out,          // [B, D, max_len] fp16
    torch::Tensor lengths,      // [B] int32
    torch::Tensor valid_state,  // [B] uint8 or undefined
    int64_t dilation, int64_t state_len, bool silu) {
  std::optional<at::Tensor> bias_opt;
  if (bias.defined()) bias_opt = bias;
  std::optional<at::Tensor> vs_opt;
  if (valid_state.defined()) vs_opt = valid_state;
  ple_short_conv_prefill(x_packed, init_state, weight, bias_opt, out,
                         lengths, vs_opt, dilation, state_len, silu);
}
