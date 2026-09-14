# Explore: W4A8 sdot4 — ranked shapes (nibble→i8)

**Status**: Explore-only research scaffold. **Not dest.** Not UNC-26.
Not a default serve path. Do not merge as a production kernel.

**Card**: Project 4 — *Explore: W4A8 sdot4 — ranked shapes (nibble→i8)*
(V620 inference board). Dest remains
[UNC-26](https://linear.app/uncoolred/issue/UNC-26/finish-exl3-hip-3inst-fdot2)
(EXL3 HIP 3inst → fdot2).

W4A8 here means **W i4 × A i8 → `sdot4`**
(`__builtin_amdgcn_sdot4` / `v_dot4c_i32_i8`) on **gfx1030**.
A8 is **GEMM activations only**: routed expert / dense FFN that already
sit on W4 packs.

## Contract

| Item | Rule |
|---|---|
| ISA | gfx1030 `v_dot4c_i32_i8` via `__builtin_amdgcn_sdot4(a, b, acc, clamp=false)`. Four signed i8×i8 products into i32. |
| Packing | W is **K-contiguous nibbles**: one `uint32` holds 8 consecutive K values for one N. Expand nibble → **signed i8 in VGPR** (sign-extend, not uint4−8). A is packed i8 along K (8 values = two dwords). |
| Alignment | `K % 8 == 0`. Reject other K. |
| Accumulate | **i32 through K**. Do not promote to fp16 inside the K loop. |
| Scales | Weight group scales and activation scales apply in the **epilogue** only. |
| Scope | GEMM on existing W4 packs. Do **not** silently requant leftovers / embed / head / norms / GDN state / KV / indexer to A8. |
| Gate | Off by default. Stub is **not** in `VLLM_ROCM_EXT_SRC`. Intended CMake/env flag is `VLLM_RDNA2_W4A8_SDOT4=0`. No `can_implement` / serve-script change. |

One W dword → eight signed i8 → **two `sdot4`**. That is the inner
math of every ranked shape. It is not `sdot8` / `V_DOT8_I32_I4` (W4A4).

## Ranked shapes (research order)

Measure in this order. Do not skip ahead because a later shape is
prettier on paper.

### (1) ConfigA-class, LDS=0 — measure first

- Tile class of ConfigA: `THREADS=256`, `N_PER_THREAD=4` (`N_TILE=1024`),
  `M_TILE=16`, **LDS=0** (no A/W tile in LDS; W nibble→i8 lives in VGPR).
- `K_STEP ∈ {16, 32}` **only**.
- Two `sdot4` per W dword.
- **Inner loop must cover full `K_STEP`.** ConfigH lesson: never
  advertise `K_STEP=64` (or any step) while the unrolled body only
  covers 32. Advance `k` by `K_STEP` iff the body consumed
  `K_STEP / 8` W dwords.

Sketch: `csrc/rocm/explore/w4a8_sdot4_shape1.cu` (not wired).

### (2) Recipe-9 LDS 64×64×64 i8 — only if (1) loses large-M

- WG=256, tile `64×64×64` i8, ~8 KiB LDS.
- Stage A i8 (and optionally expanded W i8) in LDS; keep i32 acc +
  epilogue scales.
- Open this only after (1) is compute-bound and loses the ConfigA
  large-M soak (see [TESTPLAN.md](TESTPLAN.md)).

### (3) Decode skinny M≤4 — only if act quant already paid

- A i8 in LDS, W streamed (nibble→i8 in VGPR).
- Eligible only when the activation is **already** i8 for the GEMM
  (no new decode-side act-quant invented for this shape).
- Compare against existing W4A16 skinny (`q_gemm_rdna2.cu` /
  `skinny_gemms_int4.cu`), not against a fictional A8 baseline.

### (4) Later: unpack W to i8 workspace, reuse W8A8 tile

- One-shot nibble→i8 workspace, then a W8A8 INT8 `sdot4` tile.
- Depends on a real W8A8 INT8 `sdot4` kernel existing first
  (see dependency). Not a dest unpack.

## Leave list

Do not land any of these as “the W4A8 path”:

- **`sdot8` / `V_DOT8_I32_I4`** — that is W4A4, not W4A8.
- **`sudot4`** — unsigned/mixed DOT; not the signed i4×i8 contract.
- **FP8-bit `sdot4`** — `v_dot4c_i32_i8` is integer. Do not pack FP8
  bit patterns into it.
- **Permanent TP≤2 gate** — a TP cap is not dest for this explore.
- **Unpack W→fp16 then `fdot2` and call it `sdot4`** — that is the
  existing W4A16 path with extra copies.

## Dependency

Explore **after** a W8A8 INT8 `sdot4` kernel exists (reuse / compare
tiles; this tree today has W8A8-**FP8** `fdot2`, not INT8 `sdot4`).
**UNC-26 EXL3 stays first on dest.** This directory does not bump dest,
does not change EXL3/FA, and does not enable a default kernel.
