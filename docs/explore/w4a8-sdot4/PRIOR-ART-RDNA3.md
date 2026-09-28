# Prior art: JartX's RDNA3 (gfx1100) kernels

**Status**: read-only survey for the W4A8 sdot4 explore, 2026-09-28. No code
is copied from the fork. Each finding that applies to gfx1030 became a check
or a rule in this directory, listed under "Here" below. Scope and gates:
[README](README.md), design: [DESIGN](DESIGN.md), runs: [TESTPLAN](TESTPLAN.md).

## Sources

| Source | What |
| --- | --- |
| [JartX/vllm `perf/rdna3_full_stack`][branch] at [`dafcde3`][dafcde3] (2026-09-23) | The integrated gfx1100 stack; that commit makes the fp16 W4A16 dequant exact |
| [vllm-project/vllm#57925][pr57925] | "[ROCm][DO NOT MERGE] RDNA3 (gfx1100) full inference stack": the tracking PR for that branch |
| [vllm-project/vllm@1e48f8c][1e48f8c] | Upstream "Native W4A16 kernel for AMD RDNA3 (gfx1100) — fp16 + bf16" |
| [vllm-project/vllm#55522][pr55522] | RDNA3 W4A16 MoE moved onto the oracle/experts path |
| [vllm-project/vllm#54706][pr54706] | Deterministic split-K epilogue, as described in the kernel comments |
| Fork branches skimmed | `feat/rdna3_int8_int4_hip_kernels`, `refactor/int8_wwma`, `feature/int2_int4_per_token_head_wmma`, `int8`, `feature/int8test`, `test/int8k`, `test/int8k_m2`, `perf/rdna3-gfx1100-dot2-skinny-gptq`, `feat/rdna3_mxfp4_native` |

Files read at `dafcde3`: [`q_gemm_rdna3.cu`][q3], [`q_gemm_rdna3_wmma.cu`][q3w],
`qdq_4_rdna3.cuh`, the three `README_RDNA3*.md` next to the GPTQ sources
([`README_RDNA3.md`][readme3]), [`README_RDNA3_HIP_KERNELS.md`][attnreadme],
[`paged_prefill_attn_rdna3_v2_int8.cu`][attn8], `..._v2_int4.cu`,
[`docs/design/rdna3_full_stack.md`][stack] and `tools/rdna3/`.

## What does not carry over

gfx1100 has WMMA (`v_wmma_*_16x16x16` for f16, bf16, iu8, iu4) and dual
issue. gfx1030 has neither. Everything WMMA-shaped in the fork has no gfx1030
equivalent: the W4A16 prefill kernel and both products of the attention
kernels. That leaves numerics, epilogue design, dispatch and process, which
is most of what follows. Both chips have `v_dot2_f32_f16` and a 4-way int8
dot, and neither has a packed fp16 global atomic add.

## Kernel inventory

| Kernel | File | Design | Status |
| --- | --- | --- | --- |
| W4A16 decode, M ≤ 8 | `csrc/rocm/q_gemm_rdna3.cu` | Scalar, M_COUNT ∈ {1, 2, 4, 8}; fp16 through explicit `fdot2`, bf16 widened to f32; the WMMA/scalar choice is made in the C++ entry | upstream (1e48f8c) |
| W4A16 prefill | `csrc/rocm/q_gemm_rdna3_wmma.cu` | f16/bf16 WMMA, f32 accumulate, own TU, exact dequant; fp16 takes it from M ≥ 12 since `dafcde3` (was 64) | fork |
| W4A16 MoE | `csrc/rocm/moe_q_gemm_rdna3.cu` | Still the baked dequant (per the `dafcde3` message) | #55522 |
| INT8 per-token-head KV prefill attention | `csrc/attention/paged_prefill_attn_rdna3_v2_int8.cu` | Raw int8 K in LDS; Q quantized per row inside the kernel; QKᵀ on `v_wmma_i32_16x16x16_iu8`, score × q_scale · k_scale; PV in f16/bf16 WMMA | fork (#57925) |
| INT4 KV prefill attention | `..._v2_int4.cu` | Nibble → center → fp16 in the loaders, f16/bf16 WMMA for both products | fork (#57925) |
| Split-KV decode attention | `pth_decode_int{8,4}_rdna3.cu` | Not read in detail | fork (#57925) |
| Dense dequant + rocBLAS, large M | `tools/rdna3/rocblas_prefill/` | Pulled from the GEMM path (finding 5) | tried, removed |

## Findings

### 1. The exllama fp16 dequant bakes a rounded bias, on gfx1030 too

The bit trick computes `w = (1024 + q)·s + z1` in one fp16 FMA, with
`z1 = s·(−1024 − z)` precomputed **as fp16**: about 0.008 absolute at
s ≈ 0.02, the same sign for every weight of a (group, column), accumulated
along K. `dafcde3` measured, on the four linear shapes of a 27B dense W4A16
G32 model at TP4 and every M from 1 to 24, a max abs error of 0.34–1.26
baked against 0.004–0.013 exact (64–142× worse), "a ~3% typical
perturbation" at |ref| ≈ 7. Computing in f32 does not help: the error is in
the fp16 value of `z1`.

gfx1030's W4A16 kernels use the same constants: `prep_zero_scale_fp16` in
`csrc/rocm/qdq_4_rdna2.cuh`. `reference.w4a16_rdna2_weights` emulates it
bit for bit. The low formula applies to K offsets {0, 1, 4, 5}; the rest use
`(1024 + 16q)·(s/16) + s·(−64 − z)`. On the reference's random problems
(`make_problem(64, 512, 2560, G, fmt, seed ∈ {0, 1, 2})`: x ~ N(0, 0.25),
s ∈ [0.002, 0.022]; outliers are 8 channels × 20), the output rel-L2 against
exact dequant is:

| fmt | G | outliers | W4A16 baked | W4A8, A per token | W4A8, A per (token, G) |
| --- | ---: | ---: | --- | --- | --- |
| uint4 | 32 | 0 | 2.29–2.35 % | 0.83–0.84 % | 0.53–0.54 % |
| uint4 | 32 | 8 | 2.40–2.68 % | 5.37–5.74 % | 0.97–1.03 % |
| uint4 | 128 | 0 | 2.26–2.37 % | 0.83–0.86 % | 0.65–0.66 % |
| uint4 | 128 | 8 | 2.34–2.73 % | 5.28–5.95 % | 1.86–1.92 % |
| uint4b8 | 32 | 0 | 3.33–3.34 % | 0.83–0.84 % | 0.54–0.54 % |
| uint4b8 | 32 | 8 | 3.42–3.75 % | 5.43–5.80 % | 0.98–1.01 % |
| uint4b8 | 128 | 0 | 3.26–3.34 % | 0.83–0.84 % | 0.65–0.65 % |
| uint4b8 | 128 | 8 | 3.39–3.95 % | 5.40–5.82 % | 1.86–1.89 % |

What that changes:

- The W4A8 weight term is exact (`q − z` in integers), so on well-behaved
  rows W4A8 swaps a ~2.3–3.3 % deterministic weight error for a ~0.8 %
  activation error. With outliers, per-token A8 is worse than the baked
  bias; per-(token, G) A8 is not. That is what the `*_ag` configs are for.
- G1 fake-quantizes A on top of the baked W4A16, so it charges W4A8 for both
  errors: a G1 pass is conservative, a marginal G1 fail is not final.
- The G2 baseline row compares the production W4A16 op with both exact and
  baked results. Against the emulation it should agree to fp16-output
  level; against exact it should be off by about 1e-2.
- Porting the exact form to the gfx1030 W4A16 kernels is separate work and
  probably the larger accuracy win of the two.

Test: `test_rdna2_w4a16_bakes_a_rounded_bias`.

### 2. The exact form is the W4A8 zero fold

`dafcde3` computes `Σ a·w = y·Σ a·(1024 + q) + z·Σ a` with `y`, `z` group
constants applied once per group in f32. `Σ a` does not depend on the column,
so one extra `v_dot2` against `half2(1, 1)` serves all four columns of a
thread. This is the structure the W4A8 draft uses in integers: `acc = −z·Σa`,
then `Σ a·q`, flushed once per group. On gfx1100 the exact fp16 path was
1.14–1.35× faster than baked at M=1, neutral at M=2–4 and 0.79–1.05× at
M=8. It stayed at 124 VGPRs, occupancy 10, like the baked form. Moving the
correction out of the inner loop was worth 1.04–1.25× on its own.

### 3. Deterministic split-K

The fork's first split-K epilogue added fp16/bf16-narrowed partials through a
CAS loop, so results depended on completion order. #54706 writes f32
partials per split to scratch and reduces them in a fixed order with one
final rounding. Split 1 stays a direct store with no zero-init. The CAS
path is kept for A/B only.

The draft here uses ConfigA's pk4 CAS epilogue, with the same order
dependence and one fp16 rounding per split. The G2 bound already counts that
rounding. **Here**: G2 runs every fp16 split-K case twice and records whether
the bits repeat. Graduation needs a deterministic epilogue or split 1
(DESIGN §10).

### 4. Split only when the grid is small

`compute_wmma_k_split_mn` splits only while the XY grid is below about 2×
oversubscription (1,500 blocks of 4 waves on a 3,072-slot 7900 XTX). Above
that it uses split 1: no scratch, no reduce, direct store.

Ours is ConfigA's rule: grow the split while `blocks·split < 2048` or the K
range is over 2,048. At N_TILE = 1024 the grids are small in blocks but not
in waves. The V620 holds 864 waves at occupancy 6 (72 CUs × 2 SIMD × 6).
Every M ≥ 624 cell with N ≥ 6144, and M=2048 × N=2560, is already 2.2–10.7×
that, yet the rule picks split 4–10. **Here**: `bench --split-k 1` (G3)
times the no-split alternative for every config.

### 5. Dense dequant plus a vendor GEMM: fast, golden-clean, corrupt in production

Tried on gfx1100 in August 2026, per the comment in `q_gemm_rdna3.cu`, it
gave 1.34× on the GEMM at M=2048 and +6.6–7.4 % cold prefill end to end,
and matched the fused kernel 4/4 against golden. In production (TP4,
MTP k=2, int8 per-token-head KV) it produced streams of `!!!!` within hours.
156 requests against the failing configuration did not reproduce it, and it
was pulled "until someone has a harness that fails first".

Shape (4) here, unpacking W to an i8 workspace for a W8A8 tile, is the same
class: a transient dense copy of the weights consumed by another kernel.
**Here**: shape (4) needs a fails-first soak harness before it is considered
(README).

### 6. Check outputs in the same process as the timing

From `rdna3_full_stack.md`, on a custom all-reduce corruption: the corrupted
engine benchmarked *faster*. Every measurement needs an output sanity check
in the same engine start. The checker must first be shown to catch a
known-bad sample: a `!!!!` grep called other garbage clean. **Here**: G3
compares every timed W4A8 output with the W4A16 op's (rel-L2 < 0.1, about
3e-2 expected on random data). `test_bench_flags_garbage_output` is the
known-bad sample.

### 7. INT8 attention: in-kernel activation quant pays

The int8 QKᵀ path quantizes Q per row inside the attention kernel. Each lane
holds a full row, so there is no cross-lane reduce. It then accumulates i32
dot products and scales once. That measured about 1.5× over dequantizing K
to fp16 at long context. The ingredients are W4A8's: per-row int8, an
integer dot, scales after. On gfx1030 the analogue would be `sdot4` QKᵀ for
int8 KV, a separate explore that G0 also informs. The INT4 README still
describes a centered-int8 `iu8` path the code no longer uses, so port from
the code, not the README.

### 8. Engineering lessons

From [`README_RDNA3.md`][readme3] and the TU comments:

| Lesson | gfx1100 evidence | Here |
| --- | --- | --- |
| Dispatch on M in C++, not Python | `if x.size(0) >= 16` in `apply_weights` made Dynamo guard on every layer; decode got 7× slower | DESIGN §10 puts the M threshold in the C++ op. Today's `RDNA2W4A16LinearKernel.apply_weights` picks its kernel from `x.size(0)` in Python; check on the box how torch.compile treats that branch |
| New kernels in their own TU; do not grow shared headers | WMMA in the scalar kernel's TU miscompiled M=1 though never instantiated there; adding unused functions to the shared dequant header cost decode tok/s | W4A8 stays out of `qdq_4_rdna2.cuh` and the W4A16 TUs; diff the W4A16 ISA before and after wiring. The audit here saw the effect: a change dead for non-`_ag` configs moved an SGPR spill in `a8_smem_k32` until rewritten |
| Audit VGPRs and spills from compiled output | VGPR/spill audit per bundle | `isa_check.py` (A2, B0) |
| Call `__builtin_amdgcn_fdot2` explicitly | hipcc 7.2.1 emitted 0 `v_dot2_f32_f16` for `__hfma2` + cast + add | The W4A16 budget here assumes `v_dot2`; B2 checks the baseline's ISA |
| Zero-fill only when split > 1 | `torch::zeros` only for K_SPLIT > 1 | Already so in `w4a8_gemm` |

## Not taken

WMMA kernels, the MXFP4 and int2/int4 per-token-head WMMA branches (no
gfx1030 hardware for them), and the INT8 attention work (a different
explore; finding 7).

[branch]: https://github.com/JartX/vllm/tree/perf/rdna3_full_stack
[dafcde3]: https://github.com/JartX/vllm/commit/dafcde3f8bb9da96e5ca24adcf8f54c9ced413d7
[pr57925]: https://github.com/vllm-project/vllm/pull/57925
[1e48f8c]: https://github.com/vllm-project/vllm/commit/1e48f8c
[pr55522]: https://github.com/vllm-project/vllm/pull/55522
[pr54706]: https://github.com/vllm-project/vllm/pull/54706
[q3]: https://github.com/JartX/vllm/blob/dafcde3f8bb9da96e5ca24adcf8f54c9ced413d7/csrc/rocm/q_gemm_rdna3.cu
[q3w]: https://github.com/JartX/vllm/blob/dafcde3f8bb9da96e5ca24adcf8f54c9ced413d7/csrc/rocm/q_gemm_rdna3_wmma.cu
[readme3]: https://github.com/JartX/vllm/blob/dafcde3f8bb9da96e5ca24adcf8f54c9ced413d7/csrc/libtorch_stable/quantization/gptq/README_RDNA3.md
[attnreadme]: https://github.com/JartX/vllm/blob/dafcde3f8bb9da96e5ca24adcf8f54c9ced413d7/csrc/attention/README_RDNA3_HIP_KERNELS.md
[attn8]: https://github.com/JartX/vllm/blob/dafcde3f8bb9da96e5ca24adcf8f54c9ced413d7/csrc/attention/paged_prefill_attn_rdna3_v2_int8.cu
[stack]: https://github.com/JartX/vllm/blob/dafcde3f8bb9da96e5ca24adcf8f54c9ced413d7/docs/design/rdna3_full_stack.md
