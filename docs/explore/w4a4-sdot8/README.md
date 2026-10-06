# Explore: W4A4 sdot8 — ranked shapes (i4×i4)

**Status**: Explore-only research scaffold. **Not dest.** Not a default
serve path. Do not merge as a production kernel.

**Card**: Project 4 — *Explore: W4A4 sdot8 — ranked shapes (i4×i4)*
(V620 inference board). Sibling of
[PR #9](https://github.com/opengfx1030/vllm-rdna/pull/9)
(W4A8 / `sdot4` on `explore/w4a8-sdot4`). Dest remains
[UNC-26](https://linear.app/uncoolred/issue/UNC-26/finish-exl3-hip-3inst-fdot2)
(EXL3 HIP 3inst → fdot2).

W4A4 here means **W i4 × A i4 → `sdot8`**
(`__builtin_amdgcn_sdot8` / `v_dot8_i32_i4`) on **gfx1030**.
This is **integer** packed signed i4, not MXFP4 / NVFP4 / E2M1.
`sdot8` on E2M1 bit patterns is **wrong math**.

A4 is **GEMM activations only**: runtime quant of fp16 rows for
dense / routed FFN that already have INT4 weights. No gfx1030
integer-W4A4 checkpoint is assumed.

## Contract

| Item | Rule |
|---|---|
| ISA | gfx1030 `v_dot8_i32_i4` via `__builtin_amdgcn_sdot8(a, b, acc, clamp=false)`. Eight signed i4×i4 products into i32. |
| Packing | **Both sides** are packed signed i4. One `uint32` holds 8 consecutive K values (LSB-first, K-contiguous nibbles, sign-extend, not uint4−8). One `sdot8` per K-dword. Do **not** unpack to i8 and call `sdot4`. |
| Alignment | `K % 8 == 0`. Reject other K. |
| Accumulate | **i32 through K**. Do not promote to fp16 inside the K loop. |
| Scales | Weight group scales and activation scales apply in the **epilogue** only. |
| A quant | **Runtime**, symmetric signed i4 seed `{−8…7}` (see below). Applied to fp16 GEMM rows immediately before the DOT. Not a checkpoint format. |
| Scope | GEMM on existing W4 packs. Do **not** silently A4 leftover-BF16, GDN state, KV, indexer, shared-expert, MTP heads, embed, or norms. |
| Gate | Off by default. Stub is **not** in `VLLM_ROCM_EXT_SRC` / `torch_bindings.cpp`. Intended CMake/env flag is `VLLM_RDNA2_W4A4_SDOT8=0`. No `can_implement` / serve-script change. No TP≤2 dest gate. |

One K-dword on W and one K-dword on A → **one `sdot8`**. That is the
inner math of every ranked shape. It is not W4A8 / `sdot4` (PR #9)
and not `sudot8`.

### Runtime A quant seed (not a checkpoint)

No gfx1030 integer-W4A4 weight+activation checkpoint is assumed. W is
the existing INT4 pack. A is quantized at runtime from fp16:

- Range: signed i4 `{−8…7}` (two's complement nibble).
- Symmetric seed: `s = max(|x|) / 7` over the scale group (row or
  groupsize), `q = clamp(round(x / s), −8, 7)`.
- Pack LSB-first K-contiguous nibbles. Scales land in the epilogue.
- Quant is **outside** the DOT loop. The kernel consumes packed A.

This seed is a research starting point, not a dest recipe.

## Ranked shapes (research order)

Measure in this order. Do not skip ahead because a later shape is
prettier on paper.

### (1) ConfigA-class, LDS=0 — measure first

- Tile class of ConfigA: `THREADS=256`, `N_PER_THREAD=4` (`N_TILE=1024`),
  `M_TILE=16`, **LDS=0** (no A/W tile in LDS; packed i4 lives in VGPR).
- `K_STEP ∈ {16, 32}` **only**.
- One `sdot8` per K-dword.
- **Inner loop must cover full `K_STEP`.** ConfigH lesson: never
  advertise `K_STEP=64` (or any step) while the unrolled body only
  covers 32. Advance `k` by `K_STEP` iff the body consumed
  `K_STEP / 8` W (and A) dwords.

Sketch: `csrc/rocm/explore/w4a4_sdot8_shape1.cu` (not wired).

### (2) LDS 64×64×64 i4 — only if (1) loses large-M

- WG=256, tile `64×64×64` i4, ~8 KiB LDS double-buf
  (`2 × (A 64×64 i4 + W 64×64 i4)` ≈ 8 KiB).
- Stage packed A i4 (and optionally W i4) in LDS; keep i32 acc +
  epilogue scales.
- Open this only after (1) is compute-bound and loses the ConfigA
  large-M soak (see [TESTPLAN.md](TESTPLAN.md)).

### (3) Decode skinny — optional / likely Leave

- Default: **decode stays W4A16**. Do not invent a decode-side A4
  path to make a cell green.
- Eligible only if TESTPLAN marks a cell optional **and** measured
  A4 decode beats dest W4A16 skinny. Until then this shape is
  **Leave**.

### (4) Later: workspace

- One-shot unpack / workspace tiles after (1)–(2) have numbers.
- Not a dest unpack. Not “unpack to i8 then reuse PR #9 `sdot4`”
  sold as W4A4.

## Leave list

Do not land any of these as “the W4A4 path”:

- **E2M1 / MXFP4 / NVFP4 `sdot8`** — integer i4×i4 only. E2M1 bits
  in `v_dot8_i32_i4` are the wrong math.
- **Mixing with PR #9** — do not unpack A/W to i8 and call `sdot4`.
  Do not share `docs/explore/w4a8-sdot4/` or `csrc/rocm/explore/w4a8_*`.
- **Silent A4 of non-GEMM tensors** — leftover-BF16, GDN state, KV,
  indexer, shared-expert, MTP heads, embed, norms.
- **Dest bump** — no default `EXT_SRC` / `torch_bindings` /
  `can_implement` / serve-script change. No TP≤2 dest gate.
- **`sudot8`** — unsigned/mixed DOT; not the signed i4×i4 contract.
- **tok/s claims** without V620 cells in [TESTPLAN.md](TESTPLAN.md).

## Sibling / dependency

This directory is a **sibling** of W4A8 / `sdot4`
[PR #9](https://github.com/opengfx1030/vllm-rdna/pull/9). Same W pack
(K-contiguous signed nibbles); A is i4 here, i8 there. Compare A4 vs
A8 on that shared W (TESTPLAN §2). **UNC-26 EXL3 stays first on dest.**
This directory does not bump dest, does not change EXL3/FA, and does
not enable a default kernel.
