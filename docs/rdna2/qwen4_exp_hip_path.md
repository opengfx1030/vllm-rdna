# Qwen4Exp / Qwen3.8-Flash-Next HIP path on gfx1030

**Status**: opt-in scaffolding shipped 2026-09-14. All new ops default to
**off** — the existing Triton paths are the source of truth until the HIP
ports are verified end-to-end on `.176`. The goal of this note is to map
what's currently HIP on gfx1030, what's still Triton-only, and which
env-var gates flip the new HIP paths on.

## Model scope

Two checkpoint families share the same vLLM code path:

- `wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16` — AWQ-int4 dense + MoE; PLE sidecar
  embeds n-gram positions as an extra input.
- `primitive-ai/Qwen3.8-Flash-Next-PLE-quant` — same backbone + an offline
  PLE-int4 quantization of the embedding table (sidecar reads this when
  `VLLM_PLE_CPU_OFFLOAD=1`).

Both load via `vllm/models/qwen4_exp/amd/model.py` (the
`qwen4_exp_text` config + the ROCm `Qwen4ExpForCausalLM` /
`Qwen4ExpForConditionalGeneration`).

## What's HIP today

| Component | HIP kernel(s) | Trigger |
|---|---|---|
| **PLE CPU offload** | ZMQ/doorbell IPC + HIP shim for `cuStreamWriteValue32`/`cuStreamWaitValue32`/`cuMemHostRegister` (`vllm/v1/ple_offload/hip_driver.py`) | `VLLM_PLE_CPU_OFFLOAD=1` |
| **HC decode** (M ≤ 8) | `rdna_gemv_act`, `rdna_hc_up_gate_mix`, `rdna_se_gate_up_silu`, `rdna_se_down_gated` (`csrc/rocm/rdna_fused_glue.cu`) | `VLLM_RDNA_FUSED_HC=1` |
| **GDN decode** | `gdn_decode_rdna2` | `on_gfx10x()` |
| **GDN prefill** (5-kernel chain) | `gdn_prefill_{prep,kkt,solve_wy,delta_h,o}_rdna2` | `_gdn_prefill_dispatch_available()` |
| **Causal conv1d** | `causal_conv1d_update_rdna2`, `causal_conv1d_fwd_rdna2` | `VLLM_CAUSAL_CONV1D_RDNA2=1` |
| **M-RoPE** | `mrope_forward_rdna2` | `on_gfx10x()` + Qwen3-VL/Qwen3.8 hybrid |
| **Flash-Attention (RDNA2)** | `fa_rdna2_decode_paged`, `fa_rdna2_prefill_paged_varlen{,_short,_splitk,_gqa}` | `VLLM_USE_RDNA2_FA=1` |
| **W4A16 GPTQ** | `gptq_gemm_rdna2`, `gptq_gemm_rdna2_prefill`, `moe_gptq_gemm_rdna2` | quant config + on_gfx10x() |
| **AWQ prefill (high M)** | `q_gemm_rdna2_awq_prefill` | `VLLM_RDNA_AWQ_PREFILL=1` (implicit default) |
| **FA KV cache writer** | `reshape_and_cache_flash_rdna2` | `VLLM_USE_RDNA2_FA=1` |
| **Sparse MLA** | `sparse_mla_decode_rdna2`, `sparse_mla_prefill_rdna2` | `VLLM_USE_RDNA2_MLA=1` |
| **Paged MQA indexer** | `paged_mqa_logits_decode_rdna2` | `on_gfx10x()` + QSA-aware path |
| **RMSNorm / FusedAddRmsNorm / GatedRMSNorm** | `rms_norm`, `fused_add_rms_norm`, `gated_rms_norm` (`csrc/rocm/layernorm.cu`) | AOT-compiled; cudagraph-safe |
| **Skinny GEMM / GEMV decode** | `LLMM1`, `wvSplitK`, `gemv_f16_rdna2`, `gemv_i8_rdna2` | RDNA2 auto |

## What was still Triton-only (gap before this PR)

| Component | Triton source | Notes |
|---|---|---|
| **HC prefill** (M > 8: chunked prefill + decode-merged prefill) | `vllm/models/qwen4_exp/amd/ops/hc.py`: `_grouped_gemma_rmsnorm_kernel`, `_hc_silu_kernel`, `_hc_gate_mix_kernel`, `_hc_combine_kernel`, `_hc_combine_norm_kernel` | All elementwise / affine. Runs every prefill token, every decoder layer → highest-frequency gap. |
| **QSA store-cache-rows** | `vllm/models/qwen4_exp/amd/ops/qsa.py`: `_store_qsa_rows_kernel` | Pure scatter (mirrors `reshape_and_cache_flash_rdna2` stride contract). Runs each indexer step. |
| **QSA compress-groups** | `_compress_qsa_groups_kernel` | Sum with raw/state switch; also writes first-position tail. |
| **QSA MQA paged indexer logits** | `_qsa_mqa_paged_kernel` | Single Q head, single KV head, paged layout — matches `paged_mqa_logits_decode_rdna2` shape contract. |
| **QSA sparse splitk attention** | `_qsa_sparse_paged_gqa_splitk_kernel` + `_qsa_merge_splitk_kernel` | Only fires at prefill (large M); left to Triton for now. |
| **PLE dilated short-conv (decode + prefill)** | `_short_conv_dilated_decode_batched` / `_short_conv_dilated_prefill_batched` (`vllm/models/qwen4_exp/amd/ple_layer.py`) | Depthwise conv1d with dilation; per-channel CTA. |

## What this PR adds (opt-in scaffolding)

Three new HIP files, three new env-var gates, three Python dispatcher
modules. The default everywhere is **Triton** — these are flags you turn
on once you've validated the HIP path matches Triton's output on the
target shapes.

### 1. HC prefill HIP

- `csrc/rocm/hc_rdna2.cu` — 5 kernels, all M-parallel, vec8 fp16:
  `hc_grouped_gemma_rmsnorm_rdna2`, `hc_silu_rdna2`, `hc_gate_mix_rdna2`,
  `hc_combine_rdna2`, `hc_combine_norm_rdna2`.
- Gate: `VLLM_RDNA_HC_PREFILL_HIP=1` + `on_gfx10x()`.
- Dispatcher: `vllm/models/qwen4_exp/amd/ops/hc_rdna2.py`.

### 2. QSA decode HIP (store + compress + MQA wrapper)

- `csrc/rocm/qsa_rdna2.cu` — 3 entry points:
  `qsa_store_cache_rows_rdna2`, `qsa_compress_groups_rdna2`,
  `qsa_mqa_paged_rdna2` (thin wrapper that delegates to
  `paged_mqa_logits_decode_rdna2`).
- Gate: `VLLM_RDNA_QSA_HIP=1` + `on_gfx10x()`.
- Dispatcher: `vllm/models/qwen4_exp/amd/ops/qsa_rdna2.py`.
- The splitk prefill attention stays on Triton (only fires at large M,
  and the prefill cudagraph story on gfx1030 is still in flux).

### 3. PLE dilated short-conv HIP

- `csrc/rocm/ple_short_conv_rdna2.cu` — 2 kernels:
  `ple_short_conv_decode_rdna2` (one new token per request + state shift),
  `ple_short_conv_prefill_rdna2` (variable-length packed prefill).
- Gate: `VLLM_RDNA_PLE_CONV_HIP=1` + `on_gfx10x()`.
- Dispatcher: `vllm/models/qwen4_exp/amd/ops/ple_conv_rdna2.py`.
- Early-return branches added to
  `Qwen4ExpPLELayer._short_conv_dilated_decode_batched` and
  `_short_conv_dilated_prefill_batched` so the new path is reachable
  without touching any other code path.

## How to enable (production checklist)

```bash
# 1. Rebuild _rocm_C.abi3.so (the .cu files are added to the gfx1030
#    EXT_SRC list in CMakeLists.txt and the new ops are registered in
#    torch_bindings.cpp).
pip install -e . --no-build-isolation --no-deps

# 2. Smoke-test the new ops load (registered + callable):
python - <<'PY'
import torch
schemas = torch._C._jit_get_all_schemas()
new_ops = [
  "_rocm_C::hc_grouped_gemma_rmsnorm_rdna2",
  "_rocm_C::hc_silu_rdna2",
  "_rocm_C::hc_gate_mix_rdna2",
  "_rocm_C::hc_combine_rdna2",
  "_rocm_C::hc_combine_norm_rdna2",
  "_rocm_C::qsa_store_cache_rows_rdna2",
  "_rocm_C::qsa_compress_groups_rdna2",
  "_rocm_C::qsa_mqa_paged_rdna2",
  "_rocm_C::ple_short_conv_decode_rdna2",
  "_rocm_C::ple_short_conv_prefill_rdna2",
]
missing = [op for op in new_ops if not any(str(s) == op for s in schemas)]
print("missing:", missing or "none")
PY

# 3. Enable one gate at a time on a serve run, log correctness probes,
#    then enable the next. The PLE / HC paths are pure elementwise; the
#    QSA path touches the indexer output, so test a smoke prompt before
#    a full bench.
export VLLM_RDNA_HC_PREFILL_HIP=1
export VLLM_RDNA_QSA_HIP=1
export VLLM_RDNA_PLE_CONV_HIP=1
```

## Follow-up work (not in this PR)

- **HC shared/per-stream weight selection**: the HIP path picks
  `W_SHARED` from `weight.numel()`. The Triton kernel has the same gate,
  but doesn't support mixed shapes (e.g. checkpoint has weight
  `[HC, GROUP_DIM]` vs `[GROUP_DIM]`). If the checkpoint stores a
  reshaped weight, add a `stride` parameter and an explicit mode.
- **HC unroll factor**: the constexpr `HC` template covers `{1,2,4,8}`.
  Qwen4Exp defaults `hc_count=4`; other checkpoints may need additional
  values.
- **QSA MQA paged signature alignment**: the HIP wrapper
  (`qsa_mqa_paged_rdna2`) currently re-uses
  `paged_mqa_logits_decode_rdna2` when the shape contract matches.
  The Triton `qsa_mqa_paged` signature includes
  `query_positions` / `sequence_lengths` / `compress_ratio` /
  `score_scale` that the indexer doesn't use; a future PR can either
  extend the wrapper to translate those or add a dedicated kernel.
- **QSA sparse splitk attention**: ports well to RDNA2 (splitk +
  wave-per-row works similarly to `sparse_mla_decode_rdna2`); deferred
  until the prefill cudagraph story is solid.
- **PLE conv smem size**: the per-channel register buffer `history_buf[64]`
  caps `K` at 64 (state_len + dilation + 1). Qwen4Exp PLE uses
  `state_len + 1` (typically 4 + 1 = 5) so we have headroom, but the
  constant should be lifted into a runtime `TORCH_CHECK` once a wider
  shape appears.
- **PLE conv state-write semantics**: the eager PLE prefill path packs
  prefill tokens + initial state into `history = [init, x_packed]` and
  then collects `history[..., -state_len:]` for the new state. The HIP
  path's state write-back is currently a simpler left-shift by `dilation`
  taps; we should swap to a `next_state = history[:, state_len:state_len + max_len][..., -state_len:]`
  gather when the prefill opt-in lands in production.
