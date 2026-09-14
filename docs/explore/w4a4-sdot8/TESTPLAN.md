# W4A4 sdot8 test plan (fill on V620)

No fabricated timings. Every result cell is blank until measured on
gfx1030 (V620). Kernel is Explore-only; do not flip default serve /
`can_implement`. Do not write tok/s claims into this file without a
filled cell.

Reference W4A16 ConfigA soak:
`csrc/rocm/q_gemm_rdna2_prefill.cu` (`ConfigA` = 256/4/32/16/0) and
`docs/profiling/2026-09-10-awq-vs-gptq-prefill-microbench.md`.
Qwen3.8-27B-AWQ per-rank notes in that kernel: high-N intermediate
~8704, down ~2560 (TP=2). TP=2 here is a **shape note**, not a dest
`can_implement` gate.

Sibling W4A8 / `sdot4` shape (1):
[PR #9](https://github.com/opengfx1030/vllm-rdna/pull/9),
`csrc/rocm/explore/w4a8_sdot4_shape1.cu`. Same W pack; A is i8 there.

Oracle: pack W and A as K-contiguous signed nibbles; acc i32; scales
only in epilogue. Do **not** use the W4A16 `fdot2` dequant, E2M1, or
“unpack to i8 + `sdot4`” as the W4A4 oracle.

## 1. Prefill large-M vs dest W4A16 ConfigA

Fill µs / correctness on V620. `K` must be `% 8 == 0`. A is
**runtime** symmetric signed i4 `{−8…7}` of the same fp16 rows dest
feeds ConfigA.

| M | N | K | notes | W4A16 ConfigA µs | shape (1) K_STEP=16 µs | shape (1) K_STEP=32 µs | max \|Δ\| vs i32 ref |
|---|---:|---:|---:|---|---|---|---|
| 32 | 2560 | 8704 | down-proj class | | | | |
| 96 | 2560 | 8704 | ConfigA “wins ≥96” band | | | | |
| 128 | 6144 | 2560 | microbench M=128 | | | | |
| 256 | 6144 | 2560 | M=256 boundary | | | | |
| 624 | 1024 | 2560 | small-N | | | | |
| 624 | 6144 | 2560 | microbench mid-M | | | | |
| 624 | 8704 | 2560 | TP=2 intermediate class | | | | |
| 624 | 12288 | 2560 | microbench high-N | | | | |
| 1856 | 6144 | 2560 | large-M profile band | | | | |
| 2048 | 2560 | 8704 | full-chunk down | | | | |
| 2048 | 6144 | 2560 | microbench M=2048 | | | | |
| 2048 | 8704 | 2560 | full-chunk intermediate | | | | |

Optional K sweep on one large-M cell (`M=624`, `N=6144`):
`K ∈ {1024, 2560, 4096, 5120, 8704}`.

## 2. Same W pack vs W4A8/sdot4 shape 1 (A4 vs A8)

Hold **W** fixed (K-contiguous signed nibbles). Compare runtime A4
(`sdot8`, this PR) vs runtime A8 (`sdot4`, PR #9 shape 1). Do not
unpack A4 to i8 and call `sdot4` for the A4 column.

| M | N | K | W pack identical? | A8 sdot4 K_STEP=32 µs | A4 sdot8 K_STEP=32 µs | max \|Δ\| vs own i32 ref | notes |
|---|---:|---:|---:|---|---|---|---|
| 96 | 2560 | 8704 | | | | | |
| 624 | 6144 | 2560 | | | | | |
| 624 | 8704 | 2560 | | | | | |
| 2048 | 6144 | 2560 | | | | | |
| 2048 | 8704 | 2560 | | | | | |

If PR #9 shape (1) is not built on the same tree, record
`sibling not present` — do not copy `w4a8_*` files into this PR.

## 3. K_STEP=16 vs 32 coverage correctness

Guard the ConfigH failure mode: advertised step must equal consumed K.
One K-dword = 8 values = one `sdot8`.

| check | method | pass? |
|---|---|---|
| `K_STEP=16` body consumes 2 dwords / col (`16/8`) then `k += 16` | static_assert + unit launch `K=128` | |
| `K_STEP=32` body consumes 4 dwords / col then `k += 32` | same, `K=128` | |
| Reject `K_STEP` not in `{16,32}` | compile-time / host check | |
| Reject `K % 8 != 0` | host `TORCH_CHECK` | |
| `K=40` (not %8) does not launch | negative test | |
| `K=64` with only 32 covered would mismatch i32 ref | compare shape (1) vs CPU i32 `sdot8` model for `M=16,N=64,K=64` and `K=256` | |
| Group-scale epilogue: groupsize 32 and 128 | refresh on group boundary, not only at `k_start` | |
| A and W both packed signed i4 (no i8 expand in the DOT) | static read of `sdot8_from_i4_dwords` | |

## 4. Shape (1) vs (2) when (1) is compute-bound

Do **not** open the 64×64×64 i4 LDS tile until (1) is compute-bound
**and** loses large-M vs W4A16 ConfigA (or vs an i4 roofline you
record here).

| cell | (1) bound? (roofline / counters) | (1) µs | (2) 64×64×64 i4 µs | winner |
|---|---|---|---|---|
| M=624 N=6144 K=2560 | | | | |
| M=2048 N=6144 K=2560 | | | | |
| M=2048 N=8704 K=2560 | | | | |

If (1) is still memory-bound on those cells, stop. Do not add ~8 KiB
LDS “because a later shape exists”.

## 5. Decode skinny — optional / likely Leave

**Default: leave. Decode stays W4A16.** Fill only if you explicitly
opt in and dest skinny is the baseline. If the live path is still
fp16 activations (no paid A4), record `ineligible` / `Leave`.

| M | N | K | opted in? | W4A16 skinny µs | shape (3) µs | max \|Δ\| | verdict |
|---|---:|---:|---:|---|---|---|---|---|
| 1 | 2560 | 8704 | no (Leave) | | | | |
| 1 | 6144 | 2560 | no (Leave) | | | | |
| 2 | 2560 | 8704 | no (Leave) | | | | |
| 2 | 6144 | 2560 | no (Leave) | | | | |
| 4 | 2560 | 8704 | no (Leave) | | | | |
| 4 | 6144 | 2560 | no (Leave) | | | | |

Baseline files: `csrc/rocm/q_gemm_rdna2.cu` (`M_COUNT ∈ {1,2,4,8}`),
`csrc/rocm/skinny_gemms_int4.cu`.

## 6. Graph hygiene checklist

Same class as GDN / EXL3 capture bugs. Check before any capture-mode
bench.

| item | pass? |
|---|---|
| Scratch / outputs allocated with **zeros**, not `empty` | |
| Out tensors marked mutating (`Tensor!`) if a binding is ever added | |
| No `.item()` / D2H under capture | |
| No host `printf` of device scalars on the capture stream | |
| Persist / immortal buffers if the launch is ever graph-captured (see `rdna2_graph_keepalive.cuh`) | |
| Default path still does **not** register this op | |

Do not add a torch binding just to test capture. A standalone HIP
launch is enough for correctness cells.

## 7. Act path (GEMM A4 only)

Runtime A quant is **GEMM activations only** (fp16 rows of dense /
routed FFN that already sit on INT4 weights).

| check | pass? |
|---|---|
| A consumed by the kernel is already packed i4 (no fp16→i4 inside the DOT loop) | |
| Quant seed is symmetric signed i4 `{−8…7}`, not E2M1 / uint4−8 | |
| No silent A4 of leftover-BF16 | |
| No A4 of GDN state | |
| No A4 of KV / indexer | |
| No A4 of shared-expert / MTP heads | |
| No silent A4 of embed / norms | |
| Only dense / routed FFN already on W4 packs | |

If a layer is not already on a W4 pack, skip it. W4A16 leftovers stay
W4A16. Decode stays W4A16 unless §5 is opted in and measured.
