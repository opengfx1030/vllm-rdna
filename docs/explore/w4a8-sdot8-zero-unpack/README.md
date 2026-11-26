# Explore: W4A8 `sdot8` zero-unpack inner loop (from goss)

**Status**: explore only. This note is **not dest**. It proposes no default,
no `can_implement` change, no serve-recipe change, and makes **no tok/s
claim**. No code lands with it. Every number below is either copied from a
cited line or labeled as arithmetic or unsourced.

Base: `rdna_extras` @ [`93c59de2b50ccea1b26f299e2ac5713861ffb2e3`](https://github.com/opengfx1030/vllm-rdna/commit/93c59de2b50ccea1b26f299e2ac5713861ffb2e3).
External reference: [AgentAtelier/goss](https://github.com/AgentAtelier/goss) @
[`0793d7cfd642e2fea6bf3324f99de3ac767c3316`](https://github.com/AgentAtelier/goss/tree/0793d7cfd642e2fea6bf3324f99de3ac767c3316)
(MIT).

> goss is a personal project that "relies extensively on AI coding agents"
> and "has not undergone extensive independent review"
> ([README.md L37-39](https://github.com/AgentAtelier/goss/blob/0793d7cfd642e2fea6bf3324f99de3ac767c3316/README.md#L37-L39)).
> Its speed numbers are the author's only and are not repeated here.

## Prior work in this repo

- [#10](https://github.com/opengfx1030/vllm-rdna/pull/10) — `[explore] W4A4 sdot8 ranked shapes + test plan` (closed, branch
  `explore/w4a4-sdot8` @ `cf4ab85f81abe443d8651fa83e9235f337578c5f`, base `rdna_extra/v0.28.0`). That was i4 × **i4**
  activations. This note keeps **i8** activation precision (split into two
  nibbles), so it is a different contract; #10's test-plan shapes and its
  K_STEP-coverage checks are reused below.
- [#9](https://github.com/opengfx1030/vllm-rdna/pull/9) — `[explore] W4A8 sdot4` (closed). It put `sdot8` / `V_DOT8_I32_I4`
  on its leave list so the two siblings would not mix. The W4A8 `sdot4` path
  has since shipped (`w4a8_sdot4_rdna2.cuh`), so this note revisits `sdot8`
  as an inner-loop change to that shipped path, not as a new format.

## What goss does

- `sdot8(a, b, c)` = `__builtin_amdgcn_sdot8(a, b, c, false)`
  ([fuedem/kernels_common.cuh L11-13](https://github.com/AgentAtelier/goss/blob/0793d7cfd642e2fea6bf3324f99de3ac767c3316/fuedem/kernels_common.cuh#L11-L13)).
- Activations: each int8 value `a` is split as `a = 16*ah + al`, both signed
  nibbles, with `a` clamped to ±119 so `ah` fits in [-8, 7]
  ([kernels_common.cuh L42-54](https://github.com/AgentAtelier/goss/blob/0793d7cfd642e2fea6bf3324f99de3ac767c3316/fuedem/kernels_common.cuh#L42-L54);
  [docs/QUANTIZATION.md L34-54](https://github.com/AgentAtelier/goss/blob/0793d7cfd642e2fea6bf3324f99de3ac767c3316/docs/QUANTIZATION.md#L34-L54)).
  The activation scale is **one fp32 per token** across the whole K
  (QUANTIZATION.md L54); the author lists the quality cost of that as not
  measured (QUANTIZATION.md L135).
- Weights: W4_G256, eight **signed** nibbles per int32, one fp16 scale per
  256 weights (QUANTIZATION.md L9, L25-27).
- Inner loop, decode n=1: each packed weight word goes straight into two
  `sdot8` (one with `A_hi`, one with `A_lo`), no unpack, then
  `(h << 4) + l` and one fp32 FMA with the group scale
  ([fuedem/kernels_linear.hip L274-333](https://github.com/AgentAtelier/goss/blob/0793d7cfd642e2fea6bf3324f99de3ac767c3316/fuedem/kernels_linear.hip#L274-L333)).
- goss's own audit lists `mmvq_w4_g256_n1` at 31 VGPR, 0 LDS, no spills
  ([mooss/hsaco_audit.json L2-9](https://github.com/AgentAtelier/goss/blob/0793d7cfd642e2fea6bf3324f99de3ac767c3316/mooss/hsaco_audit.json#L2-L9)).
  Author's file, not re-measured.

## What dest does today

Links are at `93c59de2`.

- **W4A8 `sdot4`** — weights are the W4A16 GPTQ-shuffled buffer, **unsigned**
  nibbles, zero `z = nibble + zero_offset`; activations int8, scale per token
  or per (token, group); zero folded through per-group activation sums
  ([csrc/rocm/w4a8_sdot4_rdna2.cuh L7-27](https://github.com/opengfx1030/vllm-rdna/blob/93c59de2b50ccea1b26f299e2ac5713861ffb2e3/csrc/rocm/w4a8_sdot4_rdna2.cuh#L7-L27),
  `kLoMask`/`kAPerm` L44-47, `A_GROUP` L61-66). Activation quant is
  `absmax/127`, clamp [-128, 127]
  ([w4a8_act_quant_kernel, L428 ff.](https://github.com/opengfx1030/vllm-rdna/blob/93c59de2b50ccea1b26f299e2ac5713861ffb2e3/csrc/rocm/w4a8_sdot4_rdna2.cuh#L428)).
- Inner loop per W dword: `lo = w & 0x0F0F0F0F`, `hi = (w >> 4) & 0x0F0F0F0F`
  (hoisted out of the row loop), then two `sdot4` per row
  ([w4a8_sdot4_rdna2.cuh L318-333](https://github.com/opengfx1030/vllm-rdna/blob/93c59de2b50ccea1b26f299e2ac5713861ffb2e3/csrc/rocm/w4a8_sdot4_rdna2.cuh#L318-L333);
  MoE: [moe_w4a8_rdna2.cu L212-229](https://github.com/opengfx1030/vllm-rdna/blob/93c59de2b50ccea1b26f299e2ac5713861ffb2e3/csrc/rocm/moe_w4a8_rdna2.cu#L212-L229)
  and the v2 kernel at [L416-433](https://github.com/opengfx1030/vllm-rdna/blob/93c59de2b50ccea1b26f299e2ac5713861ffb2e3/csrc/rocm/moe_w4a8_rdna2.cu#L416-L433)).
- **W4A16 `fdot2`** — per W dword, `dequant_4bit_8_fp16` does 1 shift +
  4 mask/or + 4 `hfma2` (shared across rows), then `dot22_8_f` does 4
  `fdot2` per row: about 13 VALU ops per word at one row
  ([q_gemm_rdna2.cu L198-211](https://github.com/opengfx1030/vllm-rdna/blob/93c59de2b50ccea1b26f299e2ac5713861ffb2e3/csrc/rocm/q_gemm_rdna2.cu#L198-L211),
  [qdq_4_rdna2.cuh L62-83](https://github.com/opengfx1030/vllm-rdna/blob/93c59de2b50ccea1b26f299e2ac5713861ffb2e3/csrc/rocm/qdq_4_rdna2.cuh#L62-L83),
  [q_gemm_rdna2_common.cuh L40-48](https://github.com/opengfx1030/vllm-rdna/blob/93c59de2b50ccea1b26f299e2ac5713861ffb2e3/csrc/rocm/q_gemm_rdna2_common.cuh#L40-L48)).
  Counted from source, not from ISA.
- `__builtin_amdgcn_sdot8` / `v_dot8_i32_i4`: **zero uses** under
  `csrc/rocm/` at `93c59de2` (grep; the `dot8` helpers in
  `inkling2_linear_rdna2.cu` and `inkling2_dspark_rdna2.cu` are four chained fp16 `fdot2`).

## Proposal

Keep dest's contract (GPTQ-shuffled unsigned W buffer, per-group or
per-(token, group) activation scale, zero via `asum`). Change only the inner
dot:

1. **Weights, 1 op per word instead of 2–3.** `w' = w ^ 0x88888888` turns
   each unsigned nibble `q ∈ [0, 15]` into the signed nibble `q − 8`. Then
   `Σ a·(q − z) = Σ a·(q − 8) − (z − 8)·Σ a`, so the existing zero fold just
   uses `z − 8`. Exact in i32. No change to the shared W4A16 weight buffer.
2. **Activations, same bytes, new pack.** `w4a8_act_quant_kernel` gains a mode
   that writes `A_hi`/`A_lo` signed nibbles in the weight word's nibble order
   instead of int8 in `kAPerm` order. Storage is unchanged (8 bytes per 8
   values). Range is [-128, 119] with goss's split; symmetric options are a
   ±119 clamp (goss) or `absmax/119` scaling. `asum` stays the sum of the
   recombined `a`.
3. **Dot.** Per word per row: `h = sdot8(w', A_hi, h)`, `l = sdot8(w', A_lo, l)`;
   flush `(h << 4) + l` at the group boundary.

Nibble order (derived from the `kLoMask` comment, L44-45, not tested): bit
nibble `p` of a shuffled word holds K offset `{0,2,4,6,1,3,5,7}[p]`, so the
A nibbles must be packed in that order.

### Target files

- `csrc/rocm/w4a8_sdot4_rdna2.cuh` / `.cu` — dense GEMM inner loop + act-quant
  pack mode.
- `csrc/rocm/moe_w4a8_rdna2.cu` — both MoE kernels (L212-229, L416-433).
- `csrc/rocm/moe_prefill_bm_rdna2.cu` — **probably not applicable.** Its
  W4A8 path is `nvfp4_i8_prefill_kernel` (L530): e2m1 codes become int8
  through `V_PERM` tables (L133 ff.). e2m1 values are not a linear int4 grid
  (×2 they reach ±12), so `sdot8` cannot consume them directly. Keep it on
  the list only to confirm and drop.

### Scope

Compute-bound paths only: speculative verify at 2–8 rows and prefill.
Single-row decode is bandwidth-bound; fewer VALU ops there should not move
time (unsourced expectation; measure it rather than assume).

Honest ceiling (arithmetic from the cited loops): per W word per row the dot
count stays **2** (`2 sdot4` → `2 sdot8`). What goes away is the 2–3-op
unpack, which dest already shares across all `MT` rows of a tile. So against
W4A8 `sdot4` the saving per row-word is about `(2..3 − 1) / MT` ops, which is
largest at small M and small at prefill M. Against W4A16 `fdot2` (4 dots per
row-word plus dequant) the comparison is 2 integer dots vs 4 `fdot2`, *if*
`sdot8` and `sdot4` issue at the same rate (unsourced). A cost on the other
side: two i32 accumulators (`h`, `l`) per (row, column) instead of one, which
raises VGPR use in `MT × NPT` tiles.

## Open questions

- `v_dot8_i32_i4` issue rate vs `v_dot4_i32_i8` and `v_dot2_f32_f16` on
  gfx1030 — **unsourced**; needs a microbench on V620.
- Quality of the ±119 / split pack vs dest's ±127 path, per-token and
  per-(token, group).
- VGPR / occupancy receipts for both accumulators at the shipped config
  (`a8_lds_k32_ag`) and the MoE tiles. Not measured here (no ROCm compiler on
  the drafting machine).
- Whether pre-XORing the weights at load time is worth a separate buffer (it
  would make the unpack zero ops, but W4A8 currently reads the W4A16 buffer).

## Test plan (reused from #10 where it applies)

1. **Integer oracle.** CPU model of `sdot8` on the split pack; assert
   `(h << 4) + l` equals the current `sdot4` accumulator **bit-for-bit**
   per group, for AWQ (`zero_offset = 0`) and GPTQv1 (`zero_offset = 1`),
   group sizes 32 and 128.
2. **K_STEP coverage** (#10 §3, ConfigH lesson): every unrolled step consumes
   exactly `K_STEP / 8` dwords; negative tests for `K % 8 != 0`.
3. **GEMM microbench** vs the shipped `sdot4` config, same W buffer: #10's
   dense shape table (M ∈ {32, 96, 128, 256, 624, 1856, 2048}, N/K from the
   Qwen3.8-27B classes) plus verify-sized M ∈ {2, 4, 8}, and the MoE
   per-expert shapes the W4A8 MoE wiring uses.
4. **ISA check**: the inner loop is `v_xor_b32` + `v_dot8_i32_i4` only, no
   `v_and`/`v_lshr` per word.
5. **Quality gate**: the same eval `docs/rdna2/w4a8-quality-2026-11-04.md`
   uses, against the `sdot4` path.

## Exit criteria

- Bit-exact vs the `sdot4` reference at the integer-accumulator level (or a
  documented, bounded difference caused only by the ±119 clamp).
- No regression in the GEMM microbench at any shape in the table; a gain at
  verify-sized M, or the note closes as "not worth it".
- Quality gate passes against the `sdot4` path.

## Leave

Silent requant of leftover-BF16 / embed / head / norms / GDN state / KV;
default flips; dest bumps; tok/s claims before cells are filled; mixing with
the NVFP4/e2m1 path.

AI assistance was used to draft this note. A human must review every line
before this leaves draft. **Do not merge. Keep draft.**
