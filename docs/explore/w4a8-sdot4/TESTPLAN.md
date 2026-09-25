# W4A8 sdot4 test plan (fill on V620)

No fabricated timings. Every result cell is blank until measured on
gfx1030 (V620). Kernel is Explore-only; do not flip default serve /
`can_implement`.

Reference W4A16 ConfigA soak:
`csrc/rocm/q_gemm_rdna2_prefill.cu` (`ConfigA` = 256/4/32/16/0) and
`docs/profiling/2026-09-10-awq-vs-gptq-prefill-microbench.md`.
Qwen3.8-27B-AWQ per-rank notes in that kernel: high-N intermediate
~8704, down ~2560 (TP=2).

## 1. Prefill large-M shapes (mirror ConfigA soak)

Fill µs / correctness on V620. `K` must be `% 8 == 0`.

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

## 2. K_STEP=16 vs 32 correctness (full K coverage)

Guard the ConfigH failure mode: advertised step must equal consumed K.

| check | method | pass? |
|---|---|---|
| `K_STEP=16` body consumes 2 W dwords / col (`16/8`) then `k += 16` | static_assert + unit launch `K=128` | |
| `K_STEP=32` body consumes 4 W dwords / col then `k += 32` | same, `K=128` | |
| Reject `K_STEP` not in `{16,32}` | compile-time / host check | |
| Reject `K % 8 != 0` | host `TORCH_CHECK` | |
| `K=40` (not %8) does not launch | negative test | |
| `K=64` with only 32 covered would mismatch i32 ref | compare shape (1) vs CPU i32 `sdot4` model for `M=16,N=64,K=64` and `K=256` | |
| Group-scale epilogue: groupsize 32 and 128 | refresh on group boundary, not only at `k_start` | |

Reference: pack W as K-contiguous signed nibbles; A as i8; acc i32;
scales only in epilogue. Do not use the W4A16 `fdot2` dequant as the
oracle.

## 3. Shape (1) vs (2) when (1) is compute-bound

Do **not** open Recipe-9 until (1) is compute-bound **and** loses
large-M vs W4A16 ConfigA (or vs an i8 roofline you record here).

| cell | (1) bound? (roofline / counters) | (1) µs | (2) 64×64×64 µs | winner |
|---|---|---|---|---|
| M=624 N=6144 K=2560 | | | | |
| M=2048 N=6144 K=2560 | | | | |
| M=2048 N=8704 K=2560 | | | | |

If (1) is still memory-bound on those cells, stop. Do not add ~8 KiB
LDS “because Recipe-9 exists”.

## 4. Decode skinny M=1/2/4 vs W4A16 skinny

Eligibility: **A is already i8** for this GEMM (act quant already
paid). If the live path is still fp16 activations, record
`ineligible` — do not invent a decode act-quant to make the cell
green.

| M | N | K | eligible (A already i8)? | W4A16 skinny µs | shape (3) µs | max \|Δ\| |
|---|---:|---:|---:|---|---|---|---|
| 1 | 2560 | 8704 | | | | |
| 1 | 6144 | 2560 | | | | |
| 2 | 2560 | 8704 | | | | |
| 2 | 6144 | 2560 | | | | |
| 4 | 2560 | 8704 | | | | |
| 4 | 6144 | 2560 | | | | |

Baseline files: `csrc/rocm/q_gemm_rdna2.cu` (`M_COUNT ∈ {1,2,4,8}`),
`csrc/rocm/skinny_gemms_int4.cu`.

## 5. Graph hygiene checklist

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

## 6. Act path (GEMM A8 only)

| check | pass? |
|---|---|
| A consumed by the kernel is already i8 (no fp16→i8 inside the DOT loop) | |
| No second act-quant for GDN state | |
| No requant of KV / indexer to A8 | |
| No silent requant of leftovers / embed / head / norms | |
| Only routed expert / dense FFN that already sit on W4 packs | |

If a layer is not already on a W4 pack with i8 activations, skip it.
W4A16 leftovers stay W4A16.
