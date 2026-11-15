# Dense fp16 prefill GEMM for HI2 on the V620, round 2: wired behind a switch, then optimized (2026-11-15)

Branch `feat/hi2-prefill-gemm2` (base `feat/hi2-integration` b2d569c402 merged with `feat/hi2-prefill-gemm`
3235d53ffa). Box `par1-cs25`, tree `~/w4a8_runs/leapquant/tree-hi2g`, **HIP 5 (V620, PCI bus e2) exclusively** for
every timing below unless a table says HIP 4 (shared, lower clocks: relative numbers inside one window only).
Round 1 (`docs/rdna2/hi2-prefill-gemm-2026-11-15.md`) built the kernel and measured it; its internal +10 % bar was
overruled by the user, so round 2 wires it and keeps optimizing.

Every table: weights cold and rotating (more than 640 MB of weight sets, five times the Infinity Cache), every
contender of a cell interleaved, medians, CUDA-graph timed, the card at its 174-175 W cap (clocks 1.45-1.80 GHz,
stated per table). "Better rocBLAS" = the faster of the heuristic and the TunableOp-tuned row of the cell
(`rocm7.14-rocblas5.5-hi2-27b-tp2` rows at M 1023/1024/2046/2047/2048 plus the round-1 rows at M 512/700/1024/1500/
2048); "heuristic" is what serving runs at every other M.

## What is wired

| piece | where |
| --- | --- |
| kernel template, all designs and knobs | `csrc/rocm/rdna_hint_flex_dense_gemm_kernel.cuh` (no torch headers; the production TU and the dev harness `tools/rdna/rdna_hint_flex/gemm_sweep/dense_dev.hip` compile the same code) |
| host launch, design list, tail designs | `csrc/rocm/rdna_hint_flex_dense_gemm_rdna2.cu/.cuh` |
| (N, K, M) dispatch table, generated from measured CSVs | `csrc/rocm/rdna_hint_flex_dense_gemm_table.cuh` (`gen_dispatch.py`) |
| ops | `rdna_hint_flex_dense_gemm_rdna2(x, w, out, design=-1)` and `rdna_hint_flex_dense_gemm_pick_rdna2(m, n, k)` |
| prefill branch of the composite op | `rdna_hint_flex_linear_rdna2(..., prefill_gemm)` in `csrc/rocm/rdna_hint_flex_rdna2.cu` |
| switch | `RDNA_HINT_FLEX_PREFILL_GEMM` in `vllm/model_executor/layers/quantization/rdna_hint_flex.py` |
| tests | `tests/kernels/quantization/test_rdna_dense_gemm_rdna2.py` (60 passed on the box: error table of every design at the five TP2 slots and M in {1, 27, 512, 700, 1023, 1024, 1031, 1500, 2046, 2048}, tails off the tile grid, strides, fallbacks, composite on/off, tail split, switch values) |

* **Switch.** `RDNA_HINT_FLEX_PREFILL_GEMM` = unset / `0` / `off` / `rocblas`: rocBLAS (default, the call is
  byte-identical to the pre-existing one: the trailing op argument is only passed when the switch is on). `hip`: the
  dense kernel wherever the table measured it faster than the better rocBLAS row, rocBLAS everywhere else, and the
  tail rows of `M = 128 q + r` run apart from the whole tile rows (mode 4 below). `hip-notail`: the same kernel with the
  tail padded into one more tile row (mode 1). Unknown values raise. gfx10x only; an `_rocm_C` without the op warns once
  and stays on rocBLAS.
* **Dispatch by M lives in C++.** M is dynamic under `torch.compile` and the prefill branch is inside the C++
  composite (`rdna_hint_flex_linear_rdna2`), so the Python layer only forwards the mode integer; `pick(M, N, K)`
  (nearest measured M of the slot, rocBLAS outside [0.75 x first anchor, 2 x last anchor], any unlisted (N, K) is
  rocBLAS) and the tail rule run per call in C++. Any launch the kernel cannot take (wrong arch, K not a multiple of
  the design's BK, unaligned operand) falls back to `at::mm_out` on the same dequantized scratch.
* **`prefill_gemm` modes** (`rdna_hint_flex_dense_gemm_rdna2.cuh`): 0 rocBLAS; 1 dense kernel, a short tail padded
  into one more tile row; 2 tail rows on the decode GEMV (packed weights); 3 tail rows on the 32x128 / 64x128 tail
  designs (fp16 scratch); 4 (what `hip` sends) GEMV up to 16 tail rows, tail designs for 17..64, padded beyond.

### Scoreboard arm (Main runs it on HIP 0,1 or 2,3)

```bash
bash tools/rdna/rdna_hint_flex/scoreboard.sh gemm RDNA_HINT_FLEX_PREFILL_GEMM=hip
```

Trailing `RDNA_*` KEY=value arguments are exported by `serve_rdna.sh`; the recipe `27b-hi2-tp2` already carries
`TUNABLEOP_PREFER=rocm7.14-rocblas5.5-hi2-27b-tp2` (tuned rows at M 1023/1024/2046/2047/2048, heuristic elsewhere), so
the arm compares the kernel with the better rocBLAS row at the 16k/32k c=1 chunks (M = 2048: 87-93 % of the FLOPs) and
with the heuristic at the c=8 M = 1025..1031 chunks, which are 3 of the 5 scoreboard cells (census:
`~/w4a8_runs/hi2-tunable/sched/census.txt`). Compare against the same recipe without the argument.

## Kernel level: the production grid (`prod_b1.csv`, HIP 5, a7 build)

The four shipped designs (D1 `128x128k32 8x16 r8`, D2 = D1 with groups of 4, D3 `128x64k64 8x8`, D4 `256x128k32
8x16 r4`; all with BUF + SKIP, D1 / D2 / D4 with BROT, D4 swizzled) were timed one by one against the heuristic and the
TunableOp rows on 37 M values x 5 slots = 185 cells (M 384..2048, every multiple of 128 so that the tail split has a main
anchor), one interleaved window per cell, 174-176 W at 1.45-1.74 GHz (per-row clocks in the CSV). The dispatch table is
generated from these rows: a design must be >= 1 % faster than the better rocBLAS row of its cell, D1 wins ties inside
1 %, the nearest measured M of the slot applies in between. The table-driven numbers on the measured M are therefore
**in-sample** (best of four noisy designs per cell); the out-of-sample check is the composite run further down.

* All 185 cells: table-driven **1.156 over the better of heuristic / tuned rocBLAS, 1.169 over the heuristic**;
  the dense kernel beats rocBLAS in every cell (lowest best-of-four ratio 1.033: out_proj M = 576; qkv_proj M = 700
  1.038); picks D1 101, D4 64, D3 11, D2 9, rocBLAS 0.
* Per slot (vs better / vs heuristic): gate_up 1.230 / 1.259, down 1.143 / 1.143, in_proj_qkvz 1.144 / 1.160,
  qkv_proj 1.143 / 1.158, out_proj 1.122 / 1.130 (heuristic and better row coincide where no tuned row exists).
* The 25 protocol cells (5 slots x M 512 / 700 / 1024 / 1500 / 2048): **1.094 vs better, 1.161 vs heuristic, worst 1.038**
  (round 1, D1 alone: 1.071 vs better, 1.135 vs heuristic, worst 1.012).

Protocol cells and the scheduler's M (1023, 1031, 1045, 2046); `old` = the table-driven time of the round-2 starting
kernels (`prod_a1.csv`, other window), `new/old` above 1 = faster now:

| shape | M | heur us (TF/s) | tuned us | ours us (TF/s) | design | vs better | vs heur | old us | new/old | MHz / W |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| gate_up | 512 | 3875 (23.6) | 3878 | 3622 (25.2) | D1 | 1.070 | 1.070 | 3587 | 0.990 | 1652 / 174 |
| gate_up | 700 | 6579 (19.0) | 5721 | 5339 (23.4) | D1 | 1.072 | 1.232 | 5243 | 0.982 | 1570 / 175 |
| gate_up | 1024 | 9317 (19.6) | 8021 | 7174 (25.4) | D4 | 1.118 | 1.299 | 7088 | 0.988 | 1667 / 174 |
| gate_up | 1500 | 14198 (18.8) | 12045 | 10878 (24.6) | D4 | 1.107 | 1.305 | 10850 | 0.997 | 1584 / 174 |
| gate_up | 2048 | 19239 (19.0) | 16864 | 14583 (25.0) | D4 | 1.156 | 1.319 | 14489 | 0.994 | 1574 / 174 |
| gate_up | 1023 | 9198 (19.8) | 8022 | 7166 (25.4) | D4 | 1.119 | 1.283 | 7054 | 0.984 | 1671 / 174 |
| gate_up | 1031 | 10247 (17.9) |  | 7857 (23.4) | D4 | 1.304 | 1.304 | 8882 | 1.130 | 1665 / 174 |
| gate_up | 1045 | 10386 (17.9) |  | 8561 (21.8) | D1 | 1.213 | 1.213 | 9018 | 1.053 | 1454 / 174 |
| gate_up | 2046 | 19128 (19.1) | 16454 | 14571 (25.0) | D4 | 1.129 | 1.313 | 14456 | 0.992 | 1572 / 174 |
| down | 512 | 2298 (19.9) | 2307 | 1946 (23.5) | D1 | 1.181 | 1.181 | 1938 | 0.996 | 1717 / 175 |
| down | 700 | 3149 (19.8) | 3156 | 2786 (22.4) | D1 | 1.130 | 1.130 | 2792 | 1.002 | 1608 / 175 |
| down | 1024 | 4066 (22.4) | 4088 | 3698 (24.7) | D1 | 1.100 | 1.100 | 3675 | 0.994 | 1522 / 175 |
| down | 1500 | 5899 (22.7) | 5940 | 5414 (24.7) | D4 | 1.090 | 1.090 | 5448 | 1.006 | 1666 / 174 |
| down | 2048 | 7849 (23.3) | 7958 | 7134 (25.6) | D4 | 1.100 | 1.100 | 7160 | 1.004 | 1599 / 174 |
| down | 1023 | 4079 (22.4) |  | 3696 (24.7) | D1 | 1.104 | 1.104 | 3678 | 0.995 | 1527 / 175 |
| down | 1031 | 4812 (19.1) |  | 3931 (23.4) | D1 | 1.224 | 1.224 | 4077 | 1.037 | 1562 / 175 |
| down | 1045 | 4829 (19.3) |  | 4085 (22.8) | D1 | 1.182 | 1.182 | 4079 | 0.998 | 1495 / 175 |
| down | 2046 | 7847 (23.2) |  | 7136 (25.6) | D4 | 1.100 | 1.100 | 7159 | 1.003 | 1598 / 174 |
| in_proj_qkvz | 512 | 2117 (20.3) | 2118 | 1844 (23.3) | D4 | 1.148 | 1.148 | 1893 | 1.027 | 1742 / 174 |
| in_proj_qkvz | 700 | 3048 (19.3) | 2816 | 2678 (21.9) | D1 | 1.052 | 1.139 | 2674 | 0.999 | 1635 / 175 |
| in_proj_qkvz | 1024 | 3983 (21.6) | 3736 | 3571 (24.1) | D1 | 1.046 | 1.115 | 3582 | 1.003 | 1604 / 175 |
| in_proj_qkvz | 1500 | 6171 (20.4) | 5440 | 5124 (24.6) | D4 | 1.062 | 1.204 | 5125 | 1.000 | 1634 / 174 |
| in_proj_qkvz | 2048 | 8043 (21.4) | 7346 | 6800 (25.3) | D4 | 1.080 | 1.183 | 6836 | 1.005 | 1651 / 174 |
| in_proj_qkvz | 1023 | 3988 (21.5) | 3757 | 3578 (24.0) | D1 | 1.050 | 1.115 | 3582 | 1.001 | 1605 / 175 |
| in_proj_qkvz | 1031 | 4548 (19.0) |  | 3770 (22.9) | D4 | 1.206 | 1.206 | 3942 | 1.046 | 1705 / 174 |
| in_proj_qkvz | 1045 | 4526 (19.4) |  | 3944 (22.2) | D1 | 1.148 | 1.148 | 3954 | 1.003 | 1536 / 175 |
| in_proj_qkvz | 2046 | 8047 (21.3) | 7362 | 6791 (25.3) | D4 | 1.084 | 1.185 | 6860 | 1.010 | 1654 / 174 |
| qkv_proj | 512 | 1875 (20.0) | 1872 | 1661 (22.6) | D1 | 1.127 | 1.129 | 1653 | 0.996 | 1717 / 174 |
| qkv_proj | 700 | 2661 (19.3) | 2427 | 2338 (22.0) | D4 | 1.038 | 1.138 | 2354 | 1.007 | 1712 / 174 |
| qkv_proj | 1024 | 3518 (21.4) | 3277 | 3119 (24.1) | D1 | 1.051 | 1.128 | 3123 | 1.001 | 1609 / 175 |
| qkv_proj | 1500 | 5429 (20.3) | 4850 | 4529 (24.3) | D4 | 1.071 | 1.199 | 4574 | 1.010 | 1678 / 174 |
| qkv_proj | 2048 | 7017 (21.4) | 6503 | 5975 (25.2) | D4 | 1.088 | 1.174 | 5988 | 1.002 | 1641 / 174 |
| qkv_proj | 1023 | 3516 (21.4) | 3303 | 3115 (24.1) | D1 | 1.060 | 1.129 | 3124 | 1.003 | 1600 / 175 |
| qkv_proj | 1031 | 4002 (18.9) |  | 3346 (22.6) | D1 | 1.196 | 1.196 | 3494 | 1.044 | 1621 / 175 |
| qkv_proj | 1045 | 4009 (19.1) |  | 3472 (22.1) | D1 | 1.155 | 1.155 | 3495 | 1.007 | 1554 / 175 |
| qkv_proj | 2046 | 7035 (21.3) | 6528 | 6001 (25.0) | D4 | 1.088 | 1.172 | 5992 | 0.998 | 1643 / 174 |
| out_proj | 512 | 851 (18.9) | 824 | 714 (22.6) | D3 | 1.155 | 1.193 | 719 | 1.008 | 1693 / 176 |
| out_proj | 700 | 1131 (19.5) | 1131 | 1059 (20.8) | D3 | 1.068 | 1.068 | 1064 | 1.005 | 1688 / 175 |
| out_proj | 1024 | 1544 (20.9) | 1432 | 1373 (23.5) | D1 | 1.043 | 1.125 | 1382 | 1.007 | 1699 / 175 |
| out_proj | 1500 | 2270 (20.8) | 2263 | 1983 (23.8) | D4 | 1.141 | 1.145 | 2008 | 1.013 | 1698 / 174 |
| out_proj | 2048 | 2957 (21.8) | 2740 | 2581 (25.0) | D4 | 1.062 | 1.146 | 2600 | 1.007 | 1649 / 174 |
| out_proj | 1023 | 1551 (20.7) |  | 1372 (23.5) | D1 | 1.131 | 1.131 | 1394 | 1.016 | 1689 / 175 |
| out_proj | 1031 | 1690 (19.2) |  | 1474 (22.0) | D1 | 1.146 | 1.146 | 1528 | 1.036 | 1698 / 175 |
| out_proj | 1045 | 1696 (19.4) |  | 1510 (21.8) | D1 | 1.123 | 1.123 | 1535 | 1.016 | 1672 / 175 |
| out_proj | 2046 | 2956 (21.8) | 2757 | 2583 (24.9) | D4 | 1.067 | 1.145 | 2599 | 1.006 | 1648 / 174 |

45 cells: geomean 1.114 vs better, 1.168 vs heuristic; the new table-driven times are 1.009 x the round-2-start times
(M = 1031 / 1045: +3.6..+13 % from SKIP; the other cells within +-2 %, the noise between two windows). The tuned rows
exist only at M 512 / 700 / 1024 / 1500 / 2048 (round 1) and 1023 / 1024 / 2046 / 2047 / 2048 (hi2-27b-tp2 profile);
everywhere else (the c=8 chunks 1031 / 1045 among them) the heuristic is what serving runs.

## Tail rows: M = 128 q + r

The scheduler's c=8 chunks are 1024 + a few decode tokens (M = 1025..1031: 47-50 % of the c=8 steps) and MTP2 adds
r up to 21; a padded last tile row costs 1/q of the GEMM (q = 8: +12.5 %; D4's 256-row tiles at M = 2049: +12.5 %).
The composite op can instead run the `r` rows apart. `bench_linear.py` times the whole composite (input transform +
dequant + GEMM [+ tail]) on synthetic HI2 weights; HIP 4 under `flock`, so the clocks are lower (1.42-1.56 GHz) and
only the ratios inside a row are meaningful. Speed-up over the padded mode 1, per r (M = 1024 + r):

| r | GEMV tail (mode 2): gate_up / down / qkv / out_proj | tail designs (mode 3): gate_up / down / qkv / out_proj |
| ---: | --- | --- |
| 1 | 1.113 / 1.093 / 1.096 / 1.094 | 1.035 / 1.052 / 1.057 / 1.037 |
| 7 | 1.093 / 1.071 / 1.074 / 1.078 | 1.037 / 1.053 / 1.058 / 1.041 |
| 14 | 1.065 / 1.041 / 1.040 / 1.043 | 1.045 / 1.055 / 1.055 / 1.038 |
| 21 | 1.036 / 1.009 / 1.013 / 1.010 | 1.041 / 1.054 / 1.056 / 1.037 |
| 32 | 1.002 / 0.968 / 0.976 / 0.972 | 1.049 / 1.052 / 1.059 / 1.037 |
| 48 | 0.948 / 0.914 / 0.919 / 0.915 | 1.033 / 1.031 / 1.035 / 1.017 |

(M = 2048 + r: r = 7: GEMV 1.104 / 1.092 / 1.089 / 1.091, tail designs 1.077 / 1.083 / 1.083 / 1.083; r = 32: GEMV
1.049 / 1.032 / 1.033 / 1.033, tail designs 1.081 / 1.081 / 1.083 / 1.083; `lin2_hip4.csv`.) The GEMV is
compute-bound in r (~45 us per row at gate_up) and wins below r ~ 16; the tail designs stream the whole fp16 scratch
whatever r, a flat +3.5..5.9 %. Mode 4 takes the better of the two (threshold 16, `kHybridGemvMaxRows`).
Against the heuristic that serving runs, M = 1031 (r = 7) gets 1.30 (gate_up) to 1.17 (down) with the GEMV tail
(`lin1_hip4.csv`). The tail designs are latency-bound (gate_up ~0.9 ms for 178 MB against a ~0.35 ms DRAM floor
[derived]); see "What remains".

## Levers tried (round 2)

All screens: dev harness variants against the shipped design of the same window (`bench_prod.py --dev`), cold
weights, HIP 5 unless noted; ratio = shipped time / variant time (above 1 = variant faster). Files in
`bench_results/2026-11-15_hi2-prefill-gemm2/screen*.csv`.

| lever | result | verdict |
| --- | --- | --- |
| BROT (B fragment dwords rotated in LDS: no same-bank dot2 sources, 0 / 512 instead of 256 / 512) on D1 / D4 | D1 0.96-1.00, D4 ~1.00 (HIP 4); the 16 extra `v_mov` per stage pair eat the gain | alone: no; with BUF on D4s: kept |
| BROT 2 (four volatile dword LDS stores instead of the `v_mov`) | D1 spills (192 B scratch); D4 0.90-1.00: a lane-strided `ds_write_b32` is a 4-way bank conflict | dropped |
| `s_setprio` around compute or around the load phase (PRIO 1 / 2) | D1 0.97-1.00, D4 +-0.5 % | dropped |
| KU = 2 / 4 (k-steps unrolled) | D1 spills, 0.75-0.83; D4 0.82-0.84 with 1024 / 1024 same-bank dot2 | dropped |
| SKIP (a wave whose rows or columns all lie past M / N skips its dot2s) | D1 +2.6 % at gate_up M = 1031 (old build), no change at full tiles; with BUF+BROT: D1 +3.1..5.6 % and D4 +13..15 % at M = 1031 | kept |
| LDS-DMA (`buffer_load_dword ... lds`, no staging registers, no `ds_write`; swizzle through the lanes' source offsets; 183 VGPR) | correct first time (rel_fp32 2.07e-4 like the others); D1 0.95-0.99, D4 swizzled 0.99-1.01; with BROT (free: one more index in the source offset, 0 / 512 same-bank, VALU / dot2 1.023) D4s 1.007-1.013 | neutral: the knob stays in the header, not shipped |
| DMA + KU = 2 / 4 (the freed registers allow unrolling without spills) | 0.78-0.92 | dropped |
| BUF (global loads through a buffer descriptor: scalar K advance, no 64-bit VGPR address adds; VALU / dot2 1.062 -> 1.031, VGPR 255 -> 231) | D1 1.001-1.011, D4 swizzled +0.3..+1.1 % | kept |
| D4 with swizzled LDS instead of pad 8 (48 KiB instead of 60) | +0.4..+1.1 % at M = 2048 | kept |
| BUF + BROT on the swizzled D4 | **+1.7 % / +1.9 % / +2.2 % at M = 2048 (down / gate_up / in_proj_qkvz)**; D1: +0.4..+2.2 % | kept |
| 128x256 tile (weights tile twice the activation tile) | 0.91-0.94, one cell +8 % at M = 1031 | dropped |
| RAST 2 / 4 / 8 on the new D4 | within +-1 % of each other | RAST 4 |
| the wave-quantisation hypothesis (idle SIMDs in the last round) | D1 TF/s at M = 1024 equals M = 2048 for `down` (2.2 against 4.4 rounds) | not a visible loss; stream-K / split-K for M >= 1024 not built |

### Where the time goes (ablation: `abl_d1d4.csv`, hot weights, HIP 5, M = 2048)

Modes of the sweep kernel (`gemm_variant.hip`): m0 full, m1 LDS reads + dot2 only, m3 dot2 only, m4 global loads +
LDS stores + barriers only. Cycle efficiency = FLOPs / (time x clock x 18432 per cycle):

| design / cell | m0 full | m1 LDS+dot2 | m3 dot2 only |
| --- | --- | --- | --- |
| D4 gate_up | 13361 us, 1696 MHz, 177 W, 87.4 % | 12775, 1748 MHz, 88.7 % | 10755, 2022 MHz, 178 W, 91.1 % |
| D1 gate_up | 14154 us, 1587 MHz, 175 W, 88.2 % | 12459, 1740 MHz, 91.4 % | 10642, 2049 MHz, 178 W, 90.8 % |
| D4 down | 6440 us, 1713 MHz, 89.8 % | 6334, 1747 MHz, 89.5 % | 5374, 2041 MHz, 90.3 % |
| D4 in_proj_qkvz | 6408 us, 1710 MHz, 85.1 % | 6201, 1752 MHz, 85.8 % | 5224, 2030 MHz, 87.9 % |

The full kernels spend about as many dot2 slots per cycle as the dot-only build (85-90 % against 88-91 %): **the
loss against the 33-34 TF/s a dot-only build delivers is the clock** (1.59-1.71 GHz against 2.02-2.05 GHz at the same
175-178 W), i.e. energy per dot2, not issue slots. Every extra instruction class (LDS reads 17-19 %, global loads +
LDS stores + barriers 4-14 %) shows up as watts. Cutting the same-bank dot2 sources and the VALU overhead (BROT, BUF)
buys the +1..2 % above; the rest needs fewer LDS read bytes per dot2 (0.75 B at 8x16 per thread; 16x16 does not fit
the 255 VGPRs) [inference for the power split; the ablation numbers are measured]. Hot weights run 6-8 % faster than
the cold-weight tables (28.4 against 26.1 TF/s for `down` M = 2048, D4).

### Corrected ISA accounting

Round 1 quoted "1.275 instructions per dot2" for D1: that is a **static** ratio over the two-stage loop text. Counting
what a k-step iteration really issues (256 dot2 + 6 VALU for the swizzled fragment addresses + 12 `ds_read2st64_b64` +
11 `s_waitcnt lgkmcnt` + 6 SALU), the dynamic non-dot VALU share is 2.3-3 % and all instructions together ~1.17 per
dot2; BUF takes D1's VALU per dot2 from 1.062 to 1.031 and the DMA+BROT build to 1.023.

## What remains

1. **Fuse the tail into the main launch.** The tail blocks (136 column tiles at gate_up) appended to the main grid
   would run in the idle slots of its last partial round (0.44 of a round at M = 1024 + r), hiding most of the
   ~0.9 ms (gate_up) the separate tail-design launch costs; 3.5-5.9 % for r in 17..64 [derived, not built]. The tail
   designs are latency-bound (one stage of prefetch, 128 threads): BN = 64 tiles or split-K would raise their memory
   parallelism.
2. **Energy per dot2** is the limiter (clock 1.6-1.7 GHz against 2.0 GHz dot-only): fewer LDS bytes per dot2 (a larger
   register tile needs a different accumulator scheme), a vectorized epilogue (the fp16 stores are 2-byte scalars, ~1 %
   of the traffic) and `ds_read_b128` fragments (the LW = 16 build still needs a bank-aware register assignment: the
   compiler allocates 608 B of scratch for it).
3. Split-K for `down` (K = 8704) at M <= 700: 160 blocks against 144 resident slots at M = 512 (`down` M = 512 is 23.8
   TF/s against 25.8 at M = 1024); not on the scoreboard path.
4. Fused decode-in-GEMM was not built: decoding each weight inside every tile row costs 12-23 % of the dot2 issue
   budget at BM = 128..256 [derived]; the standalone dequant pass is 1.6 % of TP2 prefill time.

<!-- COMPOSITE -->

## Reproduce

```bash
source ~/w4a8_runs/hi2-gemm2/gpu5.sh          # HIP 5, asserts V620 bus e2, exports TREE PY SW HG2
# build (shared lock, MAX_JOBS <= 8, nice 19; staged install + atomic mv)
TREE=$HOME/w4a8_runs/leapquant/tree-hi2g BLD=$HOME/w4a8_runs/hi2-gemm2/build \
  bash tools/rdna/rdna_hint_flex/gemm_sweep/build_rocm_c.sh
# production grid: forced designs 0-3 against heuristic + tuned rocBLAS, cold rotating weights
$PY $SW/bench_prod.py --shapes all --m 384,448,...,2048 --designs 0,1,2,3 --no-auto --rows <both tunableop csvs> \
    --csv prod_b1.csv
# table from the CSV (later CSVs override earlier rows); needs the `regex` package
python3 $SW/gen_dispatch.py --csv prod_b1.csv
# composite (transform + dequant + GEMM [+ tail]) against rocBLAS heuristic / tuned
$PY $SW/bench_linear.py --shapes all --modes 1,4 --m 512,700,1023,1024,1031,1045,1500,2046,2048 --rows <both csvs>
# dev variants: D1 / D4 with knobs, e.g. D4+swz=1+pad=0+buf=1+brot=1+skip=1
$PY $SW/bench_prod.py --shapes gate_up --m 1024,2048 --designs 0,3 --no-auto --dev "D1+buf=1+brot=1"
```

Files: `bench_results/2026-11-15_hi2-prefill-gemm2/` (`prod_*.csv` production grids, `lin*.csv` composite and tail
runs, `screen*.csv` lever screens, `abl_d1d4.csv` ablation, `err_table_*.csv` error tables, `*.log` run logs).
