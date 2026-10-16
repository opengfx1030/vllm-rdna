# W8A8 sdot4 wiring — opt-in int8 x int8 dense and MoE GEMMs on gfx1030

Branch: `claude/w8a8-sdot4-rdna2` (= `rdna_extras` + the int8-weight sibling of
the W4A8 sdot4 kernels). Default OFF — nothing changes unless
`VLLM_RDNA2_W8A8_SDOT4=1`.

**Status: written against the W4A8 kernels, host-syntax-checked and layout-
simulated only. Not yet built or run on a gfx1030.** The V620 steps below are
the gate before this leaves draft.

## What it serves

compressed-tensors **W8A8 int8** checkpoints (llm-compressor `W8A8`): int8
weights with a symmetric per-channel (or per-tensor) scale, int8 activations
quantized dynamically per token. This is the scheme
[curvedinf/int8-vllm](https://github.com/curvedinf/int8-vllm) runs its
Qwen3.8-27B W8A8 recipe with on MI100 (per-channel weights, round-to-nearest
per-token activation quant, every large dense GEMM in int8); there the GEMMs are
AITER/CK MFMA kernels, which gfx1030 does not have. Here they are `v_dot4_i32_i8`
(`sdot4`) kernels in the shape of the fork's W4A8 pair.

Attribution: the quantization scheme (and the choice of which layers run int8)
follows curvedinf/int8-vllm at commit `c7fe970` (author curvedinf), from its
`README.md`, `docs/recipes/README.md` and `INT8_AUDIT_RESULTS.md`. No code was
copied from that repository; its kernels are gfx908 CK/AITER and Triton.

Dense goes through the upstream int8 scaled-mm registry
(`RDNA2Int8ScaledMMLinearKernel`, ahead of `TritonInt8ScaledMMLinearKernel` on
ROCm); MoE through `CompressedTensorsW8A8Int8RDNA2MoEMethod`, selected in
`CompressedTensorsMoEMethod.get_moe_method` before the generic int8 MoE method.

## What changed

| File | Purpose |
|---|---|
| `csrc/rocm/w8a8_sdot4_rdna2.cuh` | Kernel bodies: `mac_window` (shared i32 sdot4 loop + per-group flush), `w8a8_gemm_kernel<AG>` (dense) and `moe_w8a8_gemm_kernel<BLOCK_M, AG, C_T>` (MoE). Builtin-only, so it compiles under plain clang with a shim. |
| `csrc/rocm/w8a8_sdot4_rdna2.cu` | `w8a8_gemm_rdna2`: act quant (the W4A8 kernel) + dense GEMM, all M, split-K into an fp32 scratch at small M. |
| `csrc/rocm/moe_w8a8_rdna2.cu` | `moe_w8a8_gemm_rdna2`: act quant once per batch + fused MoE GEMM, `moe_w4a8_gemm_rdna2`'s contract and accumulation modes. |
| `csrc/rocm/ops.h`, `csrc/rocm/torch_bindings.cpp`, `CMakeLists.txt` | Declarations, registration, gfx1030 source list. |
| `vllm/_custom_ops.py` | Wrappers + `register_fake`. |
| `vllm/model_executor/layers/quantization/rdna2_w8a8.py` | Env latch, gfx1030 probe, `pack_weight` / `pack_moe_weight`, `act_group`. |
| `vllm/model_executor/kernels/linear/scaled_mm/rdna2_w8a8_int8.py` | `RDNA2Int8ScaledMMLinearKernel`; registered first in `_POSSIBLE_INT8_KERNELS[ROCM]`. |
| `.../compressed_tensors_moe/compressed_tensors_moe_w8a8_int8_rdna2.py` | `CompressedTensorsW8A8Int8RDNA2MoEMethod`; selected via `rocm_moe_rdna.is_supported_w8a8_int8`. |
| `tests/kernels/quantization/test_rdna2_w8a8.py`, `test_rdna2_moe_w8a8.py` | Packing (CPU), op vs reference, wiring. |
| `tools/rdna/bench_moe_w8a8.py` | W8A8 vs W8A16 int8 MoE micro A/B. |
| `docs/rdna2/tuning.md` | Env row. |

The W4A8 kernels, the W8A16 int8 MoE kernel and every default path are
untouched.

## Op contract

```text
w8a8_gemm_rdna2(a, w_packed, w_scales, a_group) -> Tensor
  a         [M, K]      fp16, rows 16-byte aligned (any aligned stride)
  w_packed  [K/4, N]    int32  rdna2_w8a8.pack_weight layout (below)
  w_scales  [K/WG, N]   fp16   symmetric; WG = K for per-channel
  a_group   32 | 64 | 128, a_group | K, a_group | WG
  returns   [M, N]      fp16

moe_w8a8_gemm_rdna2(a, c, b_q_weight, b_scales, topk_weights, sorted_token_ids,
                    expert_ids, num_tokens_post_padded, top_k, block_size_m,
                    mul_topk_weight, output_topk, a_group, fp32_accum=True) -> ()
  b_q_weight [E, K/4, N] int32, b_scales [E, K/WG, N] fp16; otherwise
  moe_w4a8_gemm_rdna2's arguments and output contract (accumulates into the
  pre-zeroed c, output_topk fuses the row reduce).
```

Both ops `TORCH_CHECK` on ineligible input: there is no fallback kernel that
reads this weight layout, and the Python side guarantees the shape rules
(`K % 32 == 0`, `N % 4 == 0`).

### Weight layout

`pack_weight` turns int8 `[N, K]` into int32 `[K/4, N]`: chunk `c` (K positions
`8c..8c+7`) becomes two dword rows per column, row `2c` holding bytes
`w[8c + {0, 4, 1, 5}]` and row `2c + 1` holding `w[8c + {2, 6, 3, 7}]` (byte 0
lowest). That is `kAPerm`, the byte order `w4a8_act_quant_kernel` writes into
its `[T][K/8][8][8]` tile, so `a8.x` pairs with row `2c` and `a8.y` with row
`2c + 1` and one `u32x4` load brings 4 columns x 4 K values.

## Kernel design

The activation side is the W4A8 one, unchanged: `w4a8_act_quant_kernel`
(PER_GROUP, MT = 8) writes per-(token, `a_group`) int8 with f32 scales. The
GEMM loop (`mac_window`) is `w4a8_gemm_kernel`'s with the nibble split replaced
by the second dword row: per 32-K step a thread loads 8 dwords per column
(4 columns), issues `2 x 4 x ROWS` `sdot4`, and flushes the i32 accumulator to
f32 once per activation group with the activation and weight scales. Weight
scales are fetched per activation group at `(k / w_group)`, so per-channel
(`w_group = K`) and per-group weights share the loop. No zero points.

* **Dense.** Grid `(ceil(N/1024), ceil(M/8), split_k)`, 256 threads, 4 columns
  per thread. A block walks its K range in 256-wide windows staged through
  2 KiB of LDS (row-major `[8][256]`), so no shape is LDS-bound and there is no
  `pick_split_k` mirror. `split_k > 1` only when the M x N grid has fewer than
  256 blocks (decode and small prefill): partials go through fp32 atomics into a
  zeroed scratch and one cast; prefill-sized M keeps plain fp16 stores.
* **MoE.** Grid `(num_token_blocks, ceil(N/1024), ceil(K/256))`, `BLOCK_M` in
  {1, 2, 4, 8}: `moe_w4a8_gemm_kernel`'s staging (window of `BLOCK_M` sorted
  rows in LDS, padding rows clamped and skipped) and epilogue (router weight in
  fp32, fp32 atomics into the cached scratch or the packed fp16 CAS add).
* **All M.** Unlike W4A8 there is no int8 fallback for this layout, so decode
  runs here too (the act quant is one small launch; per-call workspaces go
  through the caching allocator, capture-safe). Whether that beats the T45
  `gemv_i8_rdna2` design at M <= 8 is a V620 question.

Instantiations: dense 3 (`AG` 32/64/128); MoE 24 (`BLOCK_M` x `AG` x
{fp32, fp16} accumulator).

## Selection and flag

`rdna2_w8a8.enabled()` = `VLLM_RDNA2_W8A8_SDOT4 == "1"` and
`gcnArchName == gfx1030*` (the kernel bodies are compiled for gfx1030 alone).
It is read when a kernel/method is constructed, never in a forward.

* Dense: `RDNA2Int8ScaledMMLinearKernel.is_supported` = enabled + op built;
  `can_implement` declines static input scales and asymmetric activations.
  `process_weights_after_loading` packs the weight, expands per-tensor scales
  to per-channel (fused modules) and stores them as fp16 `[1, N]`.
  `apply_weights` picks `a_group = act_group(K)` (largest of 128/64/32 dividing
  K) and calls the op; bias is added in Python.
* MoE: `rocm_moe_rdna.is_supported_w8a8_int8` = enabled + op built +
  symmetric channel/token dynamic. The method reuses the parent's
  `create_weights` (checkpoint layout) and packs in
  `process_weights_after_loading`; `apply` mirrors the W8A16 RDNA2 method
  (`moe_align_block_size`, w13, activation, w2 with the fused top-k reduce).
  `block_size_m` is 1 (<= 4 tokens), 4, or 8 from 512 tokens.

## Build and verify

```bash
export PYTORCH_ROCM_ARCH=gfx1030 VLLM_TARGET_DEVICE=rocm
pip install -e . --no-build-isolation --no-deps
python -c "
import torch
names = sorted({str(s) for s in torch._C._jit_get_all_schemas() if 'w8a8_gemm_rdna2' in str(s)})
print(*names, sep='\n')"
# expect _rocm_C::w8a8_gemm_rdna2 and _rocm_C::moe_w8a8_gemm_rdna2
```

## Validate (gfx1030, to do)

1. `pytest tests/kernels/quantization/test_rdna2_w8a8.py tests/kernels/quantization/test_rdna2_moe_w8a8.py -v`
   (the CPU packing tests run anywhere with torch; the op tests need the GPU).
2. ISA audit of `_rocm_C.so` as for W4A8: `v_dot4c_i32_i8` present in the
   `w8a8` symbols, `sgpr_spill_count` / `vgpr_spill_count` 0, no scratch.
3. `python tools/rdna/bench_moe_w8a8.py` — W8A8 vs the W8A16 int8 MoE kernel
   per token count, with the relative error.
4. In-model: a compressed-tensors W8A8 int8 checkpoint with
   `VLLM_RDNA2_W8A8_SDOT4=0/1`. Look for `RDNA2 W8A8 sdot4 dense path active` /
   `MoE path active` once per worker, greedy probes, PPL on/off, prefill and
   decode tok/s at 1k/512 and 16k/1k, c=1 and c=8. Report both arms.

## Env gates

| Env | Effect |
|---|---|
| `VLLM_RDNA2_W8A8_SDOT4=1` | Opt in (dense and MoE). Default OFF. |
| `VLLM_RDNA2_W8A8_DEBUG=1`, `VLLM_RDNA2_W8A8_MOE_DEBUG=1` | Per-shape `[W8A8-DEBUG]` / `[W8A8-MOE-DEBUG]` log, once per shape. |
| `VLLM_RDNA2_MOE_FP32_ACCUM` | Shared with the other RDNA2 MoE kernels: accumulator mode. |

## Known limits

* **Symmetric int8 weights only.** GPTQ/AWQ-style int8 with zero points (the
  `uint8 + qzeros` layout the W8A16 MoE kernel reads) is not served; the zero
  fold the W4A8 kernel does (`acc = -z * asum`) would carry over, but no loader
  produces that layout for this kernel yet.
* **fp16 activations.** bf16 inputs are cast to fp16 around the op (gfx1030 has
  no bf16 dot; the fork serves fp16).
* **No bias in the MoE epilogue** (gpt-oss style expert biases are declined).
* **Static input scales declined** on the dense path; the kernel always
  quantizes dynamically.
* **Not measured.** Occupancy at decode (`split_k` into fp32 atomics) and the
  256-K LDS window versus the W4A8 whole-split staging are design choices that
  still need the V620 numbers above.
