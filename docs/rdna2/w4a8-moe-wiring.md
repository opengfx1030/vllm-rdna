# MoE W4A8 (int4 x int8 sdot4) wiring for gfx1030

Branch: `w4a8-wiring`. The MoE half of the W4A8 sdot4 alternative to W4A16.
Default OFF — `VLLM_RDNA2_W4A8_SDOT4=1` is required for the fused MoE kernel to
be used. This is what makes W4A8 able to fire on Qwen3.8-Flash-Next-AWQ, whose
quant is entirely `mlp.experts.*` (the dense W4A8 op is unreachable there).

## What changed

| File | Purpose |
|---|---|
| `csrc/rocm/moe_w4a8_rdna2.cu` | The fused MoE kernel + host dispatch. `moe_w4a8_gemm_kernel<BLOCK_M, GROUP>` (12 instantiations) reuses the `vllm::explore_w4a8` primitives; `moe_w4a8_gemm_rdna2` owns the eligibility gate, the activation-quant + GEMM launch, and the internal W4A16 fallback. |
| `csrc/rocm/ops.h` | Declaration. |
| `csrc/rocm/torch_bindings.cpp` | Registration under `VLLM_ROCM_GFX1030`. |
| `CMakeLists.txt` | Source added to the gfx1030 list. |
| `vllm/_custom_ops.py` | Wrapper + `register_fake`. |
| `vllm/model_executor/layers/fused_moe/experts/rdna2_w4a16_moe.py` | `resolve_w4a8_moe()` latch + constant branch in `apply`. |
| `.../compressed_tensors_moe/compressed_tensors_moe_wna16_rdna2.py` | Same latch on `CompressedTensorsWNA16RDNA2MoEMethod`; `_rdna2_fused_moe` gains `w4a8`. |
| `.../quantization/rdna2_moe_resident.py` | Resident path always calls `_rdna2_fused_moe(..., w4a8=False)`. |
| `tests/kernels/quantization/test_rdna2_moe_w4a8.py` | 27 tests. |
| `tools/rdna2_028/probe_w4a8_moe.py` | Per-expert micro A/B. |

## Op contract

```
moe_w4a8_gemm_rdna2(a, c, b_q_weight, b_scales, b_qzeros, topk_weights,
                    sorted_token_ids, expert_ids, num_tokens_post_padded,
                    top_k, block_size_m, mul_topk_weight,
                    output_topk=0, use_v2_format=False) -> ()
```

Same argument list as `moe_gptq_gemm_rdna2` plus the trailing
`use_v2_format` (zero-offset selector: 0 for AWQ uint4, 1 for GPTQv1 uint4b8).
Does not return a tensor: it accumulates into the pre-zeroed `c` exactly like
`moe_gptq_gemm_rdna2`, so the Python forward never branches on an op result.

* `a` `[M, K]` fp16 activations (w1) or `[M*top_k, K_inter]` (w2).
* `b_q_weight` `[E, K/8, N]` uint32 shuffled; `b_qzeros` `[E, G, N/8]`;
  `b_scales` `[E, G, N]` fp16 — the exact layout the W4A16 MoE path produces.

Eligibility (C++): gfx1030, fp16, `K % 32 == 0`, supported group (32/64/128),
`N % 8 == 0`, `block_size_m in {1,2,4,8}`, int32 routing tensors, contiguous
rows, 32-bit offset safety. Any failure falls back **internally** to
`moe_gptq_gemm_rdna2`.

## Kernel design

Grid `(num_token_blocks, ceil(N/1024), ceil(K/256))`, 256 threads, 4 N
columns per thread, K_STEP 32. Block `(BLOCK_M, GROUP)`.

* **Activation quant, once.** The dense `w4a8_act_quant_kernel` (per-(token,
  group) scales, `a8_lds_k32_ag` layout) runs once over the whole batch,
  expert-agnostic. A group equals the weight group.
* **Stable workspace.** `a_i8 [T,K/8,8,8]`, `a_scale [T,G,8] f32`,
  `a_sum [T,G,8] i32` are cached C++-side per `(M, K, G, device)` for the life
  of the process, so their device pointers never move inside a captured graph
  (the dense path's per-call allocation is the suspected graph-aliasing
  source).
* **A staged through LDS.** The block's whole `BLOCK_M x 256` int8 window is
  copied into 2 KiB of LDS in one cooperative sweep. Direct-global per-row A
  addressing kept `BLOCK_M` live uniform pointers, pushed SGPRs to 107 and
  spilled on `BLOCK_M=8`; staging mirrors the dense `a8_lds` config (46 SGPR,
  0 spills) and needs no LDS cap.
* **Group loop.** Mirrors `w4a8_gemm_kernel`: zero fold as the accumulator
  init, `sdot4` over K_STEP, group flush with the per-(token, group) A scale
  and the weight scale. Group parameters are double-buffered one group ahead,
  exactly as the dense kernel.
* **Epilogue** byte-identical to `moe_gptq_gemm_rdna2`: router-weight multiply
  in fp32, fp16 round-to-nearest, packed 64-bit CAS atomic add into the
  pre-zeroed output, `output_topk` row reduce fused.

## Flag (resolved at load time)

`resolve_w4a8_moe()` returns `os.environ["VLLM_RDNA2_W4A8_SDOT4"] == "1"` and
the op is registered (probed with `_jit_get_all_schemas`, because
`dir(torch.ops._rocm_C)` only reports `name`). It is hard-off when
`VLLM_RDNA_MOE_RESIDENT` or `VLLM_RDNA_MOE_RESIDENT_SKINNY` is active — the
resident layouts own the same weight buffers with a different packing.

`process_weights_after_loading` stores the bool on the method/experts object;
the traced forward only reads that Python bool (no env, logger, or shape test
inside the branch).

## ISA audit (gfx1030, ROCm 7.14, clang 23)

`v_dot4c_i32_i8` (sdot4) present: 3360. `v_dot2_f32_f16`: 0.
Non-zero `.sgpr_spill_count` / `.vgpr_spill_count`: **0**. `scratch_*`:
0. `private_segment_fixed_size`: 0 on all 13 device functions.

All 12 GEMM instantiations: SGPR 46-65, VGPR 57-168.

## L1 tests

`pytest tests/kernels/quantization/test_rdna2_moe_w4a8.py` — 27 passed:

* w1 vs `moe_gptq_gemm_rdna2` on real expert shapes (E in {4,16}, K in
  {2048,2560}, N_inter in {512,640,768}, top_k in {8,10}, G in {32,64,128},
  M in {1,4,16,64,256,512,2048}, bsm in {1,4,8}); rel-L2 < 0.1.
* w1 vs an independent per-(token, group) fp32 quantized reference; rel-L2 <
  1e-2.
* `output_topk` fused reduce vs `moe_sum`; rel-L2 < 0.05.
* invalid-shape fallback byte-identical to `moe_gptq_gemm_rdna2`.
* full w1 + SwiGLU + w2 forward through `_rdna2_fused_moe` on/off; rel-L2 in
  (1e-6, 0.1).
* latch default-off / env-on.
* w2 (intermediate -> hidden) vs W4A16.

## L2 micro (E=128, G=128, top_k=10, bsm=8)

`tools/rdna2_028/probe_w4a8_moe.py`, one V620:

| pass | tokens | W4A16 ms | W4A8 ms | speedup |
|---|---:|---:|---:|---:|
| w13 K=2560 N=1280 | 256 | 1.636 | 1.035 | 1.58x |
| w13 | 1024 | 5.650 | 3.213 | 1.76x |
| w13 | 2048 | 10.883 | 6.068 | 1.79x |
| w2 K=640 N=2560 | 256 | 0.679 | 0.402 | 1.69x |
| w2 | 1024 | 2.691 | 1.543 | 1.74x |
| w2 | 2048 | 5.288 | 3.026 | 1.75x |

rel-L2 vs the W4A16 output ~0.0065 on every cell.

## L3 in-model A/B (Flash-Next, TP=4, 4x V620)

MTP=0, FULL_AND_PIECEWISE, prefix caching, FA-RDNA2, sequential one-engine.
Marker `RDNA2 W4A8 sdot4 MoE path active`: 4 on the W4A8=1 arm (one per worker),
0 on the W4A8=0 arm. Coherence 4 OK / 0 BAD both arms. PCI-SERR flat at 15.

| cell | W4A8=1 out | W4A8=0 out | Δ | Δ TTFT | Δ prefill |
|---|---:|---:|---:|---:|---:|
| 1k/512 c=1 | 41.01 | 41.78 | -1.8% | -7.5% | +8.2% |
| 1k/512 c=8 | 165.52 | 163.54 | +1.2% | -5.7% | +6.1% |
| 16k/1k c=1 | 31.64 | 31.29 | +1.1% | -9.8% | +10.8% |
| 16k/1k c=8 | 69.32 | 65.57 | +5.7% | -8.2% | +8.9% |

Verdict: real but modest. Prefill +6-11 % and TTFT -6-10 % every cell; aggregate
output +1.1-5.7 % (decode-dominated cells dilute it). Shipped opt-in; the W4A16
default is unchanged. Full logs: `bench_results/2026-09-29_w4a8-moe/`.

## Env gate

| Env | Effect |
|---|---|
| `VLLM_RDNA2_W4A8_SDOT4=1` | Opt in to the MoE W4A8 path. Default OFF. |
| `VLLM_RDNA2_W4A8_MOE_DEBUG=1` | Per-shape `[W4A8-MOE-DEBUG]` log. |
| (anything else) | No change vs `rdna_extras`. |

`VLLM_RDNA2_W4A8_SDOT4` already exists in `vllm/envs.py` for the dense path;
the MoE wiring reads it through `resolve_w4a8_moe()`.
