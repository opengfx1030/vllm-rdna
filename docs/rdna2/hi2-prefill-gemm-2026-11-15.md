# Dense fp16 prefill GEMM for HI2 on the V620: can a custom kernel beat rocBLAS? (2026-11-15)

Branch `feat/hi2-prefill-gemm` (from `feat/hi2-27b-opt` 74a27bb5d3). Box `par1-cs25`, tree
`~/w4a8_runs/leapquant/tree-hi2g`, **HIP 5 only** (PCI bus e2, a slice of lease b4 held by HI2;
`sweep.py` refuses any device that is not a V620 on the expected bus). Format and kernels:
`rdna-hint-flex-weights.md`; the prefill section there ("fused decode-in-GEMM against dequant +
rocBLAS") is the starting point. Labels: **[measured]** = produced by a command listed here,
**[derived]** = arithmetic on measured values, **[inference]** = reasoning a run must confirm.

**Measurement window.** `pgrep -f '^(/usr/sbin/|/usr/bin/)?docker buildx build'` returned nothing at
the start and at the end of every run in this note (logged in `bench_results/2026-11-15_hi2-prefill-gemm/`:
`phase1.log`, `phase2_markers.log`, `run_probe2.log`, `phase4.log` and `phase9.log`; phases 5 to 8
and the baseline sweep logged it the same way, but those logs are not in the repository, see the file
table at the end). No number here was taken during an image build. HIP 5 carried only this work; HIP 2-4 belonged
to other agents. The box load average was 3-20 (other agents' CPU jobs); the timed work is GPU-only,
CUDA-graph launched. The re-measure list at the end names what a quieter window should repeat.

**Result in one screen.**

* Best design **D1** (`128x128k32t8x16n2p0r8g16b0w0s1m0u1l8z`: 128x128 tile, 8x16 per thread, swizzled LDS,
  grouped block order), re-measured in one interleaved window against rocBLAS over 5 shapes x 5 M: **+7.1 %
  geometric mean over the better of rocBLAS's heuristic and TunableOp-tuned solution, +13.5 % over the
  heuristic that serving runs, never slower (worst cell +1.2 %)**; best of four finalists per cell: +8.6 % / +15.1 %.
* **Bar not met** (>= +10 % on gate_up, down, in_proj_qkvz at M = 512 / 1024 / 2048): 3 of the 9 cells with
  D1, 5 of 9 with the best of four; in_proj_qkvz stays at +7.0-8.2 %. **Negative result: nothing is wired,
  no fused variant was built.** The "no cell > 3 % slower" clause is met.
* The cheaper win is not a kernel: the **heuristic is 8-18 % slower than TunableOp-tuned rocBLAS in 13 of the
  25 cells** and the shipped rows do not cover these shapes.
* Why rocBLAS is hard to beat: its Tensile kernels already run a 128-accumulator register tile at 1.13
  instructions per dot2; both it and the best dense kernels sit at the 175 W cap (1.5-1.8 GHz, 13.4-14.8 TF/s
  per GHz of a possible 18.43). The existing fused and AWQ kernels are VALU-issue bound (1.8-1.9 VALU instructions
  per dot2, ~10 TF/s per GHz) and cannot win by tuning.

## The question and the contract

HI2 prefill today = `rdna_hint_flex` dequant to an fp16 scratch, then a rocBLAS GEMM. A fused
decode-in-GEMM kernel exists (`RDNA_HINT_FLEX_PREFILL=1`) and loses end to end. The TP2 profile
(rank 0) has rocBLAS GEMMs at 60 % of kernel time and the dequant at 1.6 %: fusion alone saves at most
that 1.6 %, so the only gain is **a GEMM that is itself faster than rocBLAS**. This note builds a
parametrized dense kernel (fp16 A and W, fp32 accumulation through `v_dot2_f32_f16`, fp16 C), sweeps
its design space on the 27B TP2 shapes, and compares it with rocBLAS under identical conditions.
Bar: >= 10 % over rocBLAS at M = 512-2048 on the three biggest shapes (gate_up, down, in_proj_qkvz)
and never more than 3 % slower in any tested cell.

GEMM shapes (N x K per rank): gate_up 17408 x 5120, down 5120 x 8704, in_proj_qkvz 8192 x 5120,
qkv_proj 7168 x 5120, out_proj / o_proj 5120 x 3072; M in {512, 1024, 2048} plus the odd M 700 and
1500 (tile quantization, rocBLAS rows that do not exist).

## The hardware, as measured and as it bounds a dot2 GEMM

| item | value |
| --- | --- |
| compute | 72 CUs = 36 WGPs, wave32, 2 SIMD32 per CU (4 per WGP), up to 16 waves per SIMD; no matrix cores |
| memory path | L0 16 KB per CU, L2 4 MB, Infinity Cache (L3) 128 MB, cacheline 128 B |
| LDS | 128 KB per WGP (unified), at most 64 KB per workgroup |
| packed dot rate | 4608 lanes x 4 flop = **18.43 TFLOP/s per GHz** of shader clock: 47.4 at the 2570 MHz DPM maximum, 35.3 at 1915 MHz, 33.7 at 1830 MHz |
| power | socket limit 175 W (`amd-smi static --limit`; max 250, min 120); `pp_dpm_sclk` levels 500 / 1915 / 2570 MHz |

The task text said "80 CU, ~50 TF/s": the V620 has 72 CUs, and ~50 TF/s is the *2570 MHz* peak the
card never holds. **Under a dot2-bound load the card sits at the 175 W cap and runs the shader clock at
1560-2040 MHz depending on the kernel** (probe table below), so the usable ceiling is
18.43 x (1.6..2.0) = 29-37 TF/s and rocBLAS's 23-24 TF/s is 63-80 % of it [derived].

## Method

* Weights rotate through enough sets to exceed 640 MB (five times the Infinity Cache): every launch
  of a cell reads cold weights; one CUDA-graph launch per set; fused / dequant rows use the
  compressed HI2 sets (same 640 MB rule). `--ws-mb` changes it.
* A cell (shape, M) times every contender in **interleaved rounds** (forward, then backward order) after
  a warm-up that brings the card to its power limit; each window is >= `--win-ms`; the CSV keeps the
  minimum and the median over rounds (the tables use the median). Short or sequential timings differ by
  25-35 % for identical work (an idle card boosts to 2.4 GHz, a loaded one settles under the cap), which
  is why TunableOp's own tuning runs and a 5-call profiler window are optimistic [measured: the
  profiler's warm rocBLAS gate_up M=2048 ran 11.8 ms, 30 TF/s; the sustained cold row below is 16.5 ms].
* **Reference = the faster of rocBLAS's heuristic and its TunableOp-tuned solution**, both cold and
  both in the cell's interleaved window. Serving today runs the heuristic for these shapes (see the
  side finding), so every table also gives the speed-up over the heuristic.
* Clocks: phase-1/2 rows carry a cell-level median (amd-smi); the probes sample the amdgpu hwmon files
  at ~200 Hz and attribute clock and power to each contender's own window.
* Correctness: every design, 5 shapes (two production N x K, N off the tile grid, tiny N, short K) x
  13 M values (1-2047 incl. odd M and M below one tile) = 65 cases against an fp32 reference built from the
  same fp16 operands; pass = relative Frobenius error <= 1e-3 (rocBLAS itself: 2.1e-4).

## Baseline: rocBLAS heuristic, TunableOp rows, the fused kernel, the dequant pair

`confirm_f.csv` (the confirmation window below: every contender of a cell interleaved, 9 rounds x 60 ms,
medians; cold rotating weights, CUDA-graph timed; the first baseline run `baseline_a.csv` agrees cell by cell
within 5 %). `us` and TF/s; `dequant` = the HI2 dequant kernel alone; `dq+mm` = dequant followed by the
rocBLAS GEMM (what the server runs), with the heuristic and the tuned solution.

| shape (N x K) | M | rocBLAS heur us (TF/s) | tuned us (TF/s) | fused_ref us (TF/s) | dequant us | dq+mm heur | dq+mm tuned |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gate_up 17408 x 5120 | 512 | 4015 (22.7) | 4013 (22.7) | 4707 (19.4) | 603 | 4410 | 4474 |
| gate_up | 700 | 6666 (18.7) | 5824 (21.4) | 6815 (18.3) | 603 | 7162 | 6119 |
| gate_up | 1024 | 9275 (19.7) | 8100 (22.5) | 9236 (19.8) | 604 | 9754 | 8313 |
| gate_up | 1500 | 14111 (18.9) | 11968 (22.3) | 13755 (19.4) | 604 | 14430 | 12240 |
| gate_up | 2048 | 18892 (19.3) | 16432 (22.2) | 18393 (19.9) | 604 | 19389 | 16486 |
| down 5120 x 8704 | 512 | 2293 (19.9) | 2289 (19.9) | 2506 (18.2) | 122 | 2370 | 2353 |
| down | 700 | 3138 (19.9) | 3130 (19.9) | 3549 (17.6) | 122 | 3209 | 3201 |
| down | 1024 | 4070 (22.4) | 4061 (22.5) | 4656 (19.6) | 123 | 4140 | 4141 |
| down | 1500 | 5859 (22.8) | 5864 (22.8) | 7087 (18.9) | 123 | 5927 | 5931 |
| down | 2048 | 7834 (23.3) | 7830 (23.3) | 9263 (19.7) | 124 | 7848 | 7854 |
| in_proj_qkvz 8192 x 5120 | 512 | 1991 (21.6) | 1987 (21.6) | 2330 (18.4) | 120 | 2094 | 2092 |
| in_proj_qkvz | 700 | 3022 (19.4) | 2786 (21.1) | 3276 (17.9) | 121 | 3102 | 2878 |
| in_proj_qkvz | 1024 | 3737 (23.0) | 3711 (23.1) | 4439 (19.4) | 121 | 3824 | 3799 |
| in_proj_qkvz | 1500 | 6148 (20.5) | 5387 (23.4) | 6576 (19.1) | 122 | 6218 | 5476 |
| in_proj_qkvz | 2048 | 7971 (21.6) | 7289 (23.6) | 8705 (19.7) | 122 | 8090 | 7328 |
| qkv_proj 7168 x 5120 | 512 | 1862 (20.2) | 1862 (20.2) | 2044 (18.4) | 106 | 1918 | 1917 |
| qkv_proj | 700 | 2641 (19.4) | 2408 (21.3) | 2944 (17.4) | 106 | 2695 | 2486 |
| qkv_proj | 1024 | 3524 (21.3) | 3275 (22.9) | 3881 (19.4) | 107 | 3565 | 3360 |
| qkv_proj | 1500 | 5399 (20.4) | 4794 (23.0) | 5715 (19.3) | 107 | 5443 | 4847 |
| qkv_proj | 2048 | 7050 (21.3) | 6433 (23.4) | 7559 (19.9) | 108 | 7138 | 6420 |
| out_proj 5120 x 3072 | 512 | 839 (19.2) | 817 (19.7) | 887 (18.2) | 58 | 841 | 833 |
| out_proj | 700 | 1126 (19.6) | 1129 (19.5) | 1256 (17.5) | 58 | 1162 | 1164 |
| out_proj | 1024 | 1541 (20.9) | 1431 (22.5) | 1653 (19.5) | 58 | 1575 | 1469 |
| out_proj | 1500 | 2253 (20.9) | 2253 (20.9) | 2511 (18.8) | 58 | 2275 | 2276 |
| out_proj | 2048 | 2953 (21.8) | 2725 (23.6) | 3278 (19.6) | 58 | 2965 | 2755 |

* The better of the two rocBLAS rows runs 19.5-23.6 TF/s (the heuristic alone 18.7-23.3). The fused kernel runs
  17.4-19.9 TF/s: it is slower than the best rocBLAS row in **every** cell, and it matches or beats the
  *heuristic* only at gate_up M >= 1024 (by 0.4-2.6 %), where the heuristic is weak.
* The dequant pass is 58-604 us: 3.1 % of dq+mm (heuristic) at gate_up M=2048 and 13.7 % at M=512 [derived];
  the profile's 1.6 % is the all-layers average at the chunk sizes served.

### Side finding: TunableOp rows for the TP2 shapes (free, no new kernel)

The heuristic is **8-18 % slower than the TunableOp-tuned solution in 13 of the 25 cells** (gate_up M >= 700:
14.5 / 14.5 / 17.9 / 15.0 %; in_proj_qkvz M = 700 / 1500 / 2048: 8.5 / 14.1 / 9.4 %; qkv_proj M >= 700: 9.7 / 7.6 /
12.6 / 9.6 %; out_proj M = 1024 / 2048: 7.7 / 8.4 %) and equal elsewhere, `down` included [derived from the table
above]. The shipped TunableOp rows (`tunableop/rocm7.14-rocblas5.5/`) contain no row for these N x K x M, and a row
only applies to its exact M, so an untuned shape runs the heuristic [derived; not checked against a live
server]. The tuned rows used here are `bench_results/2026-11-15_hi2-prefill-gemm/tunableop_results0.csv` (key
`GemmTunableOp_Half_TN,tn_<N>_<M>_<K>_...`). Given the 60 % GEMM share this is a larger free win than a custom
kernel over tuned rocBLAS; it needs rows for the M values the scheduler actually produces.

Which Tensile kernel rocBLAS runs (torch profiler, `prof_summary.md`; all HHS, ISA1030, KLA assembly,
GLVW8/LRVW8, PGR1/PLR1, GSU1): **every pick at these shapes is a 128-accumulator-per-thread kernel
(TT8_16)**, never the 64-accumulator TT8_8 one. Two macro tiles occur: MT128x256x16 (WG16_16_1, 256
threads, **199 VGPR, 28 KB LDS**) and MT128x128x16 (WG16_8_1, 128 threads, **203 VGPR, 16 KB LDS**);
WGM is Tensile's grouped workgroup order, the same knob as `RAST` in the kernel below.

| shape | M=512 heur / tuned | M=1024 heur / tuned | M=2048 heur / tuned |
| --- | --- | --- | --- |
| gate_up | 128x256 WGM1 / same | 128x128 WGM1 / 128x256 WGM8 | 128x128 WGM1 / 128x256 WGM8 |
| down | 128x256 WGM1 / same | 128x256 WGM1 / same | 128x256 WGM1 / same |
| in_proj_qkvz, qkv_proj | 128x128 WGM4 / same | 128x256 WGM8 / 128x256 WGM1 | 128x128 WGM1 / 128x256 WGM1 |
| out_proj | 128x128 WGM8 / 128x128 WGM1 | 128x128 WGM4 / 128x256 WGM4 | 128x256 WGM4 / 128x256 WGM1 |

The tuned rows differ from the heuristic in macro tile and WGM (the heuristic's 128x128 / WGM1 picks at
gate_up M=1024-2048 and at M=2048 for in_proj / qkv are the slow ones).

## Gap study: why 18-20 TF/s from the existing kernels, and where rocBLAS's 23-24 comes from

### Instruction mix of the hot loops (built binaries, `isa_mix.py`; per `v_dot2`)

| kernel | VGPR | LDS B | waves/SIMD | instr | ds_read | ds_write | global_ld | other VALU | VALU-issue ceiling TF/s per GHz | same-bank dot2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AWQ `q_gemm_rdna2_prefill` ConfigA<128,4,32,16,0> | - | - | - | 2.33 | 0.062 | 0 | 0.012 | 0.887 | 9.8 | 192/1024 |
| fused `rdna_hint_flex` gemm2_kernel<128,128,false> | - | - | - | 2.29 | 0.086 | 0.016 | 0.023 | 0.801 | 10.2 | 296/512 |
| this work 128x128 BK32 8x8, 2 stages, pad 8 (b128 reads) | 166 | 40960 | 5 | 1.234 | 0.062 | 0.008 | 0.016 | 0.023 | 18.0 | **512/512** |
| this work 256x128 BK32 16x8 | 228 | 61440 | 4 | 1.160 | 0.047 | 0.006 | 0.012 | 0.016 | 18.2 | 1024/1024 |
| this work 128x128 BK32 16x8, 8-byte reads (`l8`) | 249 | 40960 | 3 | 1.279 | 0.047 | 0.016 | 0.031 | 0.080 | 17.1 | 256/512 |
| this work 128x128 BK64 8x8, XOR-swizzled LDS (`z`) | 159 | 32768 | 6 | 1.152 | 0.062 | 0 | 0 | 0.016 | 18.2 | 0/256 |
| this work 128x128 BK32 8x8 + B-pair rotation (`q1`) | 172 | 40960 | 5 | 1.244 | 0.062 | 0.008 | 0.016 | 0.035 | 17.8 | **0/512** |
| **D1** 128x128 BK32 8x16, swizzled LDS, LW8, RAST 8 (`isa_final.txt`) | 255 | 32768 | 4 | 1.275 | 0.047 | 0.016 | 0.031 | 0.062 | 17.4 | 256/512 |
| D4 256x128 BK32 8x16, pad 8, LW8, RAST 4 | 232 | 61440 | 4 | 1.246 | 0.047 | 0.012 | 0.023 | 0.059 | 17.4 | 256/512 |
| D3 128x64 BK64 8x8, 1 stage, pad 8, LW8, RAST 4 | 180 | 27648 | 4 | 1.211 | 0.094 | 0 | 0 | 0.047 | 17.6 | 64/128 |
| rocBLAS Tensile MT128x256x16 TT8_16 WG16_16_1 WGM8 | 199 | 28672 | 4 [inference] | 1.128 | 0.047 | 0.023 | 0.003 | 0.004 | 18.4 | 256/1024 |
| rocBLAS Tensile MT128x128x16 TT8_16 WG16_8_1 WGM1 | 203 | 16384 | 4 [inference] | 1.139 | 0.047 | 0.031 | 0.004 | 0.004 | 18.4 | 256/1024 |

(`isa_summary.txt`; "same-bank" = the two VGPR sources of a `v_dot2c` share `reg mod 4`; the VALU-issue
ceiling is 18.43 / (VALU instructions per dot2).)

* **The two production prefill kernels are VALU-issue bound**: 1.8-1.9 VALU instructions per dot2
  (per-group scale / decode FMAs, `v_add_f32` accumulation across groups, bit extraction) cap them at
  ~10 TF/s per GHz = 18-20 TF/s at 1.8-2.0 GHz. That matches the measured 17.4-19.8 TF/s of the fused
  kernel [derived]. Neither can reach rocBLAS by tuning; a decode-once-into-LDS fused kernel would
  remove most of that overhead but then has the dense kernel's ceiling minus the decode, so it is moot
  unless the dense kernel clears the bar (it does not, below).
* A plain dense micro-kernel has **1.13-1.28 instructions per dot2**: the instruction economy of my
  kernels equals rocBLAS's (1.128 / 1.139). The 128 accumulators per thread of the 16x8 / 8x16 tiles (rocBLAS's
  TT8_16) lower the LDS-read instructions per dot2 from 0.062 (8x8 tile) to 0.047 [measured, `isa_summary.txt`].
* **Occupancy as measured** (`sweep.py occ`: every block logs (start, end); the peak overlap over the GPU
  divided by 36 WGPs): 128x128 BK32 8x8 (168 VGPR, 40 KB LDS) 2 blocks of 8 waves per WGP = 4 waves/SIMD
  although the compiler reports 5; 128x64 8x4 (112 VGPR) 4 blocks; 64x64 4x4 (71 VGPR) 6; 64x64 BK64 8x4
  (138 VGPR) 7; 128x256 (154 VGPR, 60 KB) 1; 128x128 16x8 (239 VGPR) 3. **LDS is one 128 KB pool per WGP and
  the VGPR file limits waves**: the HIP occupancy API (1-3 blocks per WGP) assumes 64 KB per multiprocessor
  and under-reports; the compiler's waves/SIMD is right when VGPR-limited.

### Roofline probes (`MODE 1-4` builds of the same kernel: one pipe at a time)

128x128 BK32 8x8 pad 8, in_proj_qkvz M=2048 (rocBLAS heuristic 7880 us, tuned 7279 us), phase-1 clocks 1910 MHz:

| build | what runs | us | TF/s-equivalent |
| --- | --- | ---: | ---: |
| m0 | the GEMM | 8686 | 19.8 |
| m1 | LDS reads + dot2, no global loads / barriers | 7472 | 23.0 |
| m2 | LDS reads only | 4102 | 41.9 |
| m3 | dot2 only (operands in registers) | 6837 | 25.1 |
| m4 | global loads + LDS stores + barriers only | 3763 | 45.7 |

gate_up M=2048: m0 19554 / m1 15810 / m2 8392 / m3 14393 / m4 15258 us (rocBLAS tuned 16205); gate_up
M=512: m0 5418 / m1 4045 / m2 2240 / m3 3670 / m4 6313 us. Reading: the LDS pipe is 60 % busy at most (m2 over
m3) and **the dot2-only build itself reaches only 25 TF/s**: the dot pipeline falls short of 18.43 x clock
(next two subsections). The memory-only build is *not* cheap on the big shape: **at gate_up it takes 78 %
of the GEMM at M=2048 and 117 % at M=512** (the 178 MB weight matrix is re-streamed once per M tile
through the 4 MB L2 / 128 MB Infinity Cache with the default N-fastest block order), against 43 % / 69 % on
in_proj_qkvz. That is why a grouped block order (`RAST`, rocBLAS's WGM) is the biggest single knob on
gate_up below.

### Sustained clock and power per kernel (`probe`: each build alone for 5 s, hwmon at 200 Hz)

in_proj_qkvz M=2048, last 3 s of each run. Every row is at the 174-175 W cap.

| contender | TF/s | MHz | W | TF/s per GHz | % of dot2 rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| rocBLAS heuristic | 21.5 | 1564 | 175 | 13.7 | 74.5 |
| rocBLAS TunableOp | 23.7 | 1633 | 175 | 14.5 | 78.9 |
| 128x128 8x8 GEMM (m0) | 19.8 | 1816 | 175 | 10.9 | 59.1 |
| 128x128 16x8 8-byte reads GEMM | 23.2 | 1576 | 175 | 14.7 | 79.8 |
| 256x128 16x8 GEMM | 21.4 | 1891 | 175 | 11.3 | 61.5 |
| m1 (LDS + dot2) | 23.2 | 1929 | 175 | 12.0 | 65.3 |
| m3 (dot2 only) | 25.0 | 2041 | 175 | 12.2 | 66.4 |
| m3, B-pair rotation (`q1`) | **29.2** | 1886 | 175 | 15.5 | 84.0 |
| m3, 8-byte fragments (`l8`) | **32.1** | 1810 | 174 | 17.7 | 96.1 |
| m1 / m0 with `q1` | 26.4 / 21.9 | 1758 / 1694 | 175 | 15.0 / 12.9 | 81.6 / 70.1 |

At the cap the work is **energy per flop**, not the dot rate: the same 175 W buys 25.0 TF/s from the
dot2-only build and 32.1 TF/s from the identical dot count with a different register layout, and the
clock *falls* as the kernel gets faster (2041 -> 1810 MHz) [measured].

### The VGPR-bank effect (`dot2_bank`: asm microbenchmark, 144 blocks x 4 waves, no compiler allocation)

`v_dot2c_f32_f16` over the 8x8 micro-tile pattern with the three register groups offset by (dB, dC)
banks (`gen_dot2_bank.py`; short run, boost clock, so only the *ratios* matter):

| pattern | order | TF/s | shader MHz |
| --- | --- | ---: | ---: |
| src0 and src1 in the same bank (dB 0), any dC | k-pair outermost (independent accumulators) | 30.3 | 2449 |
| different banks (dB 1, 2, 3), any dC | same | 35.0 | 2266 |
| src0 / src1 same bank | k-pair innermost (4 dependent dots in a row) | 39.9 | 2232 |
| different banks | same | 39.4 | 2218 |

* **A dot2 whose two sources share a bank issues 13-15 % slower** than one whose sources do not
  (35.0 vs 30.3 TF/s while the clock *rose*) when the dots are independent. A dependent-chain order hides it
  (39.9) in this microbenchmark. [measured] Mechanism (VGPR read ports) [inference].
* The compiler always puts a ds_read_b128 result in a 4-aligned tuple, and pairs A's dword t with B's
  dword t: **every dot2 of a 8x8 / 16x8 LW16 kernel has its sources in the same bank** (512/512 above),
  and emits the dots in the independent order. rocBLAS's TT8_16 kernels pair 4x4 blocks of different rows /
  columns, so 25 % are same-bank by construction.
* Fix in the kernel (`CFG_BROT=1`): the loader rotates the four dwords of every B chunk by one on its way
  into LDS, so A's dword t pairs with B's dword (t+1) mod 4: 0/512 same-bank in the binary, bit-identical
  results. Measured: m3 25.0 -> 29.2 TF/s (+17 %), m1 23.2 -> 26.4 (+14 %), m0 19.8 -> 21.9 (+11 %).
  8-byte fragments (`lw=8`) already land in different banks in the compiler's allocation (m3 32.1 TF/s).
* Forcing the dependent-chain order with inline asm (`CFG_ORD=1`) is **slower** in the real kernel
  (below): at 3-4 waves per SIMD the exposed dependent-issue latency costs more than the saved bank reads.

## The design space (`tools/rdna/rdna_hint_flex/gemm_sweep/`)

`gemm_variant.hip` is one kernel, every knob a `-DCFG_*` define, one `.so` per design; `sweep.py` builds
(`flock ~/w4a8_runs/audit/build.lock`, nice 19, 8 jobs), checks, times, probes, reports occupancy
(`occ`), ISA (`isa`); `report.py` makes the tables from the CSVs; `isa_mix.py`, `gen_dot2_bank.py`,
`prof_rocblas.py` are the analysis helpers.

| knob | values swept |
| --- | --- |
| block tile BM x BN | 64x64, 128x64, 64x128, 128x128, 128x256, 256x128 (+256x64, 128x64 in `t168`) |
| thread micro-tile TM x TN | 4x4, 8x4, 4x8, 8x8, 16x4, 4x16, 16x8, 8x16 (accumulators per thread 16..128) |
| K step BK, LDS stages | 16 / 32 / 64, single or double buffer (one barrier per tile) |
| global load width | 4 / 8 / 16 bytes; LDS fragment width 8 / 16 bytes (`lw`) |
| LDS layout | padded rows (pad 0 / 8 halves) or XOR-swizzled rows without padding (`z`) |
| k-loop unroll KU, tile rasterization RAST (grouped), waves-per-EU hint, split-K | 1/2/4; 0/2/4/8/16/32; 0/2/3/4/5/6; 1 and 2/4 |
| B-pair rotation BROT, dot2 chain order ORD | 0/1, 0/1/2 |

**334 distinct dense designs were timed** (`sweep_a` 170, `sweep_c` 44, `sweep_d` 99, `sweep_e` 17, `sweep_f` 12,
`sweep_g` 12, overlaps removed) on 5 shapes x 5 M = 25 cells, and **360 designs pass the 65-case correctness
check** (`check_all.csv`; worst abs 0.0078, worst relative Frobenius error 2.12e-4, i.e. rocBLAS's own 2.1e-4).
A kernel bug found on the way:
`__builtin_bit_cast` of a vector-element *lvalue* read element 0 for every k-pair (clang 23, gfx1030),
which made early numbers look like 27.5 TF/s; the kernel now passes unsigned scalars to `dot2u`.

## What moved the needle: six sweeps, 334 designs

Each sweep took the best family of the previous one and varied what the data pointed at (the decision
rule was the per-cell table, not intuition). "Best single design" = the highest geometric mean over the 25
cells of (best rocBLAS us / design us), `report.py`'s ranking at the time the sweep finished.

| sweep | new designs | varied | best single design afterwards | geomean vs best rocBLAS / vs heuristic | worst cell |
| --- | ---: | --- | --- | ---: | ---: |
| `sweep_a` | 170 | tiles 64x64..256x128, micro-tiles 4x4..16x8, BK 16/32/64, 1-2 stages, load widths, pad / swizzle, unroll, split-K | `128x128k32 t16x8 pad8 LW8` | 1.004 / 1.061 | 0.858 (gate_up M=512) |
| `sweep_c` | 44 | B-pair rotation (BROT), dot2 chain order (ORD), 16x8 on three tile shapes | `256x128k32 t16x8 pad8 LW8` | 1.026 / 1.079 | 0.926 (gate_up M=512) |
| `sweep_d` | 99 | the 128-accumulator family, **block order RAST 0/4/8/16** on the best design, unroll, waves-per-EU, BK, pad | `256x128k32 t16x8 pad8 LW8 r4` | 1.047 / 1.101 | 1.010 (out_proj M=2048) |
| `sweep_e` | 17 | RAST on 8x16, swizzled, 128x256, 128x64 | `128x128k32 t16x8 swizzle LW8 r8` | 1.065 / 1.123 | 0.997 (out_proj M=2048) |
| `sweep_f` | 12 | 8x16 x swizzled LDS x RAST x unroll | **`128x128k32 t8x16 swizzle LW8 r8`** | **1.082 / 1.141** | **1.012 (out_proj M=2048)** |
| `sweep_g` | 12 | around it: RAST 2/16/32, KU 2/4, LW16 (+BROT), BK 64, 128x256, 64x128, 128x64 | no change | 1.082 / 1.141 | 1.012 |

Fastest design in each sweep at three cells (us, medians; rocBLAS best of heuristic / tuned in the same
sweeps is 3816-3894 / 15315-16438 / 7071-7403):

| sweep | gate_up M=512 | gate_up M=2048 | in_proj_qkvz M=2048 |
| --- | ---: | ---: | ---: |
| a | 4277 | 16555 | 7117 |
| c | 4169 | 15237 | 7017 |
| d | 3650 | 14696 | 6938 |
| e | 3581 | 13881 | 6705 |
| f | 3570 | 13863 | 6613 |

What each step was worth (measured, with the mechanism where I have it):

* **Block order (`RAST`, rocBLAS's WGM) helps where the weight matrix is big and M small.** Same design
  (`256x128 t16x8 pad8 LW8`), RAST 0 -> best of 4 / 8 / 16 (us, same-design rows): gate_up M=512
  3973 -> 3650 (**+8.9 %**), M=1024 7687 -> 7336 (+4.8 %), M=2048 15262 -> 14696 (+3.9 %); in_proj_qkvz
  M=512 1968 -> 1868 (+5.4 %), M=1024 +1.6 %, M=2048 +0.6 %; qkv_proj M=512 +2.8 %, out_proj M=512 +4.8 %;
  `down` 0 % at every M. With the default N-fastest order every M tile re-streams the 178 MB gate_up weight
  matrix through the 4 MB L2 / 128 MB Infinity Cache; grouping 4-8 M tiles lets the concurrently resident
  blocks share each B tile (memory-only probe below). Geometric mean over the 25 cells of that design:
  0.995 -> 1.030 vs best rocBLAS. The same design measured in two different sweeps differs by up to 5 %
  at gate_up M=512 (4169 vs 3973 us at RAST 0), which is why the finalists are re-measured in one window below.
* **8x16 instead of 16x8 thread tiles** (same 128 accumulators): geomean over the 25 cells 1.030 -> 1.061 for
  `256x128 pad8 r4` and 1.063 -> 1.082 for `128x128 swizzle r8` (+3.0 % / +1.8 %) [measured]; why is
  [inference] (the longer B fragment is read as two 8-byte pairs per k step, the shorter A fragment once).
* **Swizzled LDS (no padding) with 128x128 tiles**: 32 KB per block instead of 40 KB, so **four** blocks
  fit one WGP's 128 KB LDS exactly and 255 VGPR give four waves per SIMD: 4 x 4 = 16 waves per WGP against
  3 blocks / 3 waves per SIMD for the padded tile (`remarks`: 255 VGPR, 0 spills, 0 scratch) [derived
  from the build remarks and the occupancy probe].
* **8-byte LDS fragments (`LW8`) beat 16-byte ones** at 128 accumulators (the compiler lands the two dot2
  sources in different banks and wastes no VGPRs on a second fragment): LW16 `q1` rows are 0.75-0.93 of
  rocBLAS at gate_up; BROT is a +10-17 % kernel-level fix for LW16 but LW16 still loses to LW8 [measured].
* **ORD=1 (inline-asm dot2 chains) is 15-30 % slower** at M=2048 (`128x128 t16x8 pad8 LW16 q1`: gate_up
  16837 -> 19703 us, in_proj_qkvz 7078 -> 8430, down 7568 -> 8970, qkv_proj 6289 -> 7381, out_proj 2732 ->
  3185); split-K, BK 16, waves-per-EU hints (3/4/5), RAST 2/16/32 and unroll 2/4 on the winners were flat
  or worse [measured].

### Memory path against block order (`MODE 4` = global loads + LDS stores + barriers only, no dot2)

`diag_b.csv` (HIP 5, shader clock 2.44-2.48 GHz at 142-162 W: this build is not power limited; rocBLAS rows of
the same runs, heuristic / tuned: gate_up M=512 4010 / 3991 us, M=2048 18322 / 15954, in_proj_qkvz M=2048 7667 / 7115):

| cell | design (all LW8) | RAST 0 | RAST 4 | RAST 8 | RAST 16 |
| --- | --- | ---: | ---: | ---: | ---: |
| gate_up M=512 | 256x128 t8x16 pad8 | 3878 | **1796** | - | 1814 |
| gate_up M=512 | 128x128 t16x8 swizzle | 5271 | - | **2131** | - |
| gate_up M=2048 | 256x128 t8x16 pad8 | 9678 | 5504 | - | **5394** |
| gate_up M=2048 | 128x128 t16x8 swizzle | 13411 | - | **6539** | - |
| in_proj_qkvz M=2048 | 256x128 t8x16 pad8 | 3083 | **2641** | - | 2740 |
| in_proj_qkvz M=2048 | 128x128 t16x8 swizzle | 3706 | - | **3357** | - |

With the default order the memory path of gate_up alone costs about as much as rocBLAS's whole GEMM at M=512
(3878 vs 3991 us) and 61 % of it at M=2048; grouped order cuts it **1.8-2.5x** [measured]. On in_proj_qkvz
(84 MB of weights, which fits the Infinity Cache) the gain is only 1.1-1.2x [measured]. That the weights then
cross the cache hierarchy once per group of 4-8 M tiles instead of once per M tile is the likely mechanism behind
the RAST row of the table above and behind rocBLAS's own WGM8 choice for the tuned gate_up rows [inference].

## Confirmation: the finalists against rocBLAS in one interleaved window

The screening tables above take the best of 334 noisy designs per cell, so they are biased upwards (and the
same design differs by up to 5 % between two sweeps at the smallest cells). The finalists were therefore fixed
*before* a last window (`run_phase9.sh`: the two best designs by geomean, the best 128x64 design from the M = 512
cells and the best 256x128 design) and re-measured with rocBLAS heuristic, rocBLAS tuned, the fused kernel and the
dequant rows in one interleaved window per cell (`confirm_f.csv`, 9 rounds x 60 ms, medians, graph, cold weights):

* **D1** `128x128k32t8x16n2p0r8g16b0w0s1m0u1l8z`: 128x128 tile, 8x16 per thread, BK 32, 2 stages, XOR-swizzled
  LDS (32 KB), 8-byte fragments, RAST 8; 255 VGPR, 0 spills, 4 waves/SIMD, 1.275 instr per dot2.
* D2 the same with RAST 4; D3 `128x64k64t8x8n1p8r4g16b0w0s1m0u1l8` (BK 64, 1 stage, pad 8, LW8, RAST 4;
  180 VGPR, 27.6 KB, 1.211 instr per dot2); D4 `256x128k32t8x16n2p8r4g16b0w0s1m0u1l8` (pad 8, RAST 4; 232 VGPR,
  61 KB, 1.246 instr per dot2).

| shape | M | rocBLAS heur us | tuned us | best rocBLAS TF/s | D1 us (TF/s, MHz) | D1 vs best / vs heur | best finalist | us | vs best / vs heur |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | --- |
| gate_up | 512 | 4015 | 4013 | 22.7 | 3745 (24.4, 1635) | 1.071 / 1.072 | `256x128 k32 t8x16 r4 pad8 LW8` | 3705 | 1.083 / 1.084 |
| gate_up | 700 | 6666 | 5824 | 21.4 | 5271 (23.7, 1575) | 1.105 / 1.265 | `128x128 k32 t8x16 r8 swz LW8` | 5271 | 1.105 / 1.265 |
| gate_up | 1024 | 9275 | 8100 | 22.5 | 7378 (24.7, 1553) | 1.098 / 1.257 | `256x128 k32 t8x16 r4 pad8 LW8` | 7358 | 1.101 / 1.261 |
| gate_up | 1500 | 14111 | 11968 | 22.3 | 11271 (23.7, 1495) | 1.062 / 1.252 | `256x128 k32 t8x16 r4 pad8 LW8` | 10910 | 1.097 / 1.293 |
| gate_up | 2048 | 18892 | 16432 | 22.2 | 15142 (24.1, 1465) | 1.085 / 1.248 | `256x128 k32 t8x16 r4 pad8 LW8` | 14537 | 1.130 / 1.300 |
| down | 512 | 2293 | 2289 | 19.9 | 1946 (23.5, 1746) | 1.177 / 1.178 | `128x128 k32 t8x16 r8 swz LW8` | 1946 | 1.177 / 1.178 |
| down | 700 | 3138 | 3130 | 19.9 | 2775 (22.5, 1659) | 1.128 / 1.131 | `128x128 k32 t8x16 r8 swz LW8` | 2775 | 1.128 / 1.131 |
| down | 1024 | 4070 | 4061 | 22.5 | 3665 (24.9, 1574) | 1.108 / 1.111 | `128x128 k32 t8x16 r8 swz LW8` | 3665 | 1.108 / 1.111 |
| down | 1500 | 5859 | 5864 | 22.8 | 5505 (24.3, 1602) | 1.064 / 1.064 | `256x128 k32 t8x16 r4 pad8 LW8` | 5467 | 1.072 / 1.072 |
| down | 2048 | 7834 | 7830 | 23.3 | 7364 (24.8, 1529) | 1.063 / 1.064 | `256x128 k32 t8x16 r4 pad8 LW8` | 7151 | 1.095 / 1.096 |
| in_proj_qkvz | 512 | 1991 | 1987 | 21.6 | 1802 (23.8, 1727) | 1.103 / 1.105 | `128x128 k32 t8x16 r8 swz LW8` | 1802 | 1.103 / 1.105 |
| in_proj_qkvz | 700 | 3022 | 2786 | 21.1 | 2654 (22.1, 1672) | 1.050 / 1.139 | `128x128 k32 t8x16 r8 swz LW8` | 2654 | 1.050 / 1.139 |
| in_proj_qkvz | 1024 | 3737 | 3711 | 23.1 | 3433 (25.0, 1626) | 1.081 / 1.089 | `128x128 k32 t8x16 r8 swz LW8` | 3433 | 1.081 / 1.089 |
| in_proj_qkvz | 1500 | 6148 | 5387 | 23.4 | 5294 (23.8, 1583) | 1.018 / 1.161 | `256x128 k32 t8x16 r4 pad8 LW8` | 5138 | 1.048 / 1.197 |
| in_proj_qkvz | 2048 | 7971 | 7289 | 23.6 | 7088 (24.2, 1533) | 1.028 / 1.125 | `256x128 k32 t8x16 r4 pad8 LW8` | 6870 | 1.061 / 1.160 |
| qkv_proj | 512 | 1862 | 1862 | 20.2 | 1665 (22.6, 1755) | 1.118 / 1.118 | `128x64 k64 t8x8 r4 pad8 LW8` | 1641 | 1.134 / 1.134 |
| qkv_proj | 700 | 2641 | 2408 | 21.3 | 2379 (21.6, 1689) | 1.012 / 1.110 | `256x128 k32 t8x16 r4 pad8 LW8` | 2350 | 1.025 / 1.124 |
| qkv_proj | 1024 | 3524 | 3275 | 22.9 | 3117 (24.1, 1658) | 1.051 / 1.131 | `128x128 k32 t8x16 r8 swz LW8` | 3117 | 1.051 / 1.131 |
| qkv_proj | 1500 | 5399 | 4794 | 23.0 | 4619 (23.8, 1590) | 1.038 / 1.169 | `256x128 k32 t8x16 r4 pad8 LW8` | 4573 | 1.049 / 1.181 |
| qkv_proj | 2048 | 7050 | 6433 | 23.4 | 6126 (24.5, 1534) | 1.050 / 1.151 | `256x128 k32 t8x16 r4 pad8 LW8` | 6046 | 1.064 / 1.166 |
| out_proj | 512 | 839 | 817 | 19.7 | 751 (21.5, 1830) | 1.089 / 1.118 | `128x64 k64 t8x8 r4 pad8 LW8` | 712 | 1.147 / 1.178 |
| out_proj | 700 | 1126 | 1129 | 19.6 | 1075 (20.5, 1766) | 1.047 / 1.047 | `128x64 k64 t8x8 r4 pad8 LW8` | 1057 | 1.065 / 1.065 |
| out_proj | 1024 | 1541 | 1431 | 22.5 | 1394 (23.1, 1726) | 1.027 / 1.106 | `128x128 k32 t8x16 r8 swz LW8` | 1394 | 1.027 / 1.106 |
| out_proj | 1500 | 2253 | 2253 | 20.9 | 2032 (23.2, 1684) | 1.109 / 1.109 | `128x128 k32 t8x16 r4 swz LW8` | 2000 | 1.126 / 1.126 |
| out_proj | 2048 | 2953 | 2725 | 23.6 | 2689 (24.0, 1619) | 1.013 / 1.098 | `256x128 k32 t8x16 r4 pad8 LW8` | 2605 | 1.046 / 1.133 |

("vs best" = better of the two rocBLAS rows / design; "vs heur" = heuristic row / design; MHz = shader clock
during that design's window; all windows at 174-175 W.)

| summary of the 25 cells | D1 alone | best of the four finalists per cell |
| --- | ---: | ---: |
| geomean vs better rocBLAS row | **1.071** | 1.086 |
| geomean vs heuristic (what is served) | **1.135** | 1.151 |
| worst cell vs better rocBLAS | 1.012 (qkv_proj M=700) | 1.025 (qkv_proj M=700) |
| cells >= 1.10 vs better rocBLAS | 7 of 25 | 10 of 25 |
| cells < 0.97 | 0 | 0 |
| big three (gate_up, down, in_proj_qkvz) x M = 512 / 1024 / 2048: cells >= 1.10 | 3 of 9 | 5 of 9 |
| same nine cells, worst / geomean per shape (gate_up, down, in_proj_qkvz) | 1.028 / 1.085, 1.115, 1.070 | 1.061 / 1.105, 1.126, 1.082 |

Median shader clock over the 25 cells: rocBLAS heuristic 1683 MHz, tuned 1676, D1 1626, D4 1710, all at 174-175 W.
Median throughput: D1 23.8 TF/s at 1.63 GHz (14.6 TF/s per GHz) against 22.5 TF/s at 1.68 GHz (13.4) for the
better rocBLAS row, i.e. the gain is efficiency per clock under the power cap, not higher clocks [derived].

## Decision

> **Superseded (round 2, `docs/rdna2/hi2-prefill-gemm2-2026-11-15.md`).** The +10 % bar below was an internal
> contract; the user overruled it because the kernel already wins by the measured **+7.1 % geomean over the better
> of heuristic and tuned rocBLAS (+13.5 % over the heuristic serving runs)** and never loses more than 3 %. The
> kernel is wired behind `RDNA_HINT_FLEX_PREFILL_GEMM=hip` (off in the code, on in the `27b-hi2-tp2*` recipes) and was optimized further in round 2.
> The "no-go" text below is kept as the record of the original decision against the old bar.

**The bar is not met; this is a negative result and nothing is wired.**

* Bar clause 2 (no cell more than 3 % slower than rocBLAS): **met** by D1 alone (worst cell 1.012).
* Bar clause 1 (>= 10 % over rocBLAS at M = 512 / 1024 / 2048 on the three biggest shapes): **not met**. Against
  the better of heuristic and tuned rocBLAS only 3 of the 9 cells reach +10 % with one design (5 of 9 with the
  best of four finalists); the cells that fall short are gate_up M=512 (+7.1 % / +8.3 %), down M=2048
  (+6.3 % / +9.5 %), in_proj_qkvz M=1024 (+8.1 %) and M=2048 (+2.8 % / +6.1 %). Per shape over the three M
  values, in_proj_qkvz stays at +7.0 % (D1) / +8.2 % (best of four).
* Against the heuristic that serving runs today the same kernel is +13.5 % geomean (6 of the 9 big cells >= +10 %,
  worst of the nine +6.4 % at down M=2048, worst of all 25 cells +4.7 % at out_proj M=700); most of that gap is
  what TunableOp rows close without any new kernel (8-18 % in 13 cells).
* No fused decode-in-GEMM variant was built and no `rdna_hint_flex.py`, C++ or env-switch change was made: the
  contract ties both to clearing the bar, and fusion could only remove the 1.6 % dequant share.

What the numbers support, for whoever decides next:

1. **Ship TunableOp rows for the 27B TP2 shapes at the served chunk sizes** (zero code): +8-18 % per GEMM in
   13 of 25 cells, 0 elsewhere. A 60 % GEMM share of prefill kernel time turns that into roughly +5-10 % prefill
   throughput at those cells [derived from the profile share; not measured end to end].
2. A custom kernel on top is worth **+7.1 % (D1) to +8.6 % (a four-design table) per GEMM over the better rocBLAS
   row**, never slower, i.e. about +4 % to +5 % of prefill kernel time at the 60 % share [derived, unweighted
   by the real shape mix]. It was not judged worth the production wiring (torch op registration, per-shape
   dispatch table, env switch, tests, maintenance of a 255-VGPR kernel) against the pre-agreed bar; if wiring is
   decided anyway, D1 is the single design to ship and `confirm_f.csv` the table to dispatch from.
3. The measured ceiling argument: the better rocBLAS rows and D1 both sit at the 175 W cap; the dot2 pipe could
   do 18.43 TF/s per GHz but the card delivers 13.4-14.8 per GHz under load, so further wins need less energy
   per flop (fewer LDS reads and register-file reads per dot2), not more issue slots [inference].

## Reproduce

Box `par1-cs25`, HIP 5 only (`sweep.py` refuses a device that is not a V620 on `EXPECT_BUS`; env:
`HIPCC`, `SWEEP_OUT` (build / csv directory), `AMDSMI_GPU`, `EXPECT_BUS`).

```bash
source ~/w4a8_runs/hi2-gemm/env.sh     # TREE, PY, SW=<tree>/tools/rdna/rdna_hint_flex/gemm_sweep, EXPECT_BUS=e2
export HIP_VISIBLE_DEVICES=5
$PY $SW/sweep.py build --preset rast3  # takes ~/w4a8_runs/audit/build.lock itself, nice 19, --jobs 8
$PY $SW/sweep.py check --preset rast3 --csv check_f.csv
$PY $SW/sweep.py run --preset rast3 --shapes gate_up --m 512,1024,2048,700,1500 \
    --csv sweep_f.csv --label sweep_f --tunable-file tunableop_results0.csv
# finalists, one interleaved window per cell (9 rounds x 60 ms), with the fused kernel and the dequant rows
$PY $SW/sweep.py run --preset "<name>+<name>" --shapes gate_up --rounds 9 --win-ms 60 --fused \
    --csv confirm_f.csv --label confirm_f --tunable-file tunableop_results0.csv
$PY $SW/report.py sweep_a.csv sweep_c.csv sweep_d.csv sweep_e.csv sweep_f.csv sweep_g.csv \
    --check check_all.csv --top 12 --designs <name>,<name> --bar 1.10 --floor 0.97
$PY $SW/sweep.py probe --preset fin --shape in_proj_qkvz --m 2048 --seconds 4 --tail 2 --csv probe_c.csv
$PY $SW/sweep.py occ; $PY $SW/sweep.py isa --preset "<name>"; $PY $SW/sweep.py remarks --preset "<name>"
```

`isa_mix.py` (hot-loop instruction mix), `gen_dot2_bank.py` (the bank microbenchmark generator),
`prof_rocblas.py` (Tensile kernel names under the torch profiler; `rocprofv3` fails to attach to the
venv runtime on this box with "api registration failed with error code 16") and `clk_log.sh` (hwmon clock
and power log) are described in their file headers. Test (CPU part anywhere, GPU part on a V620 with `HIPCC`
pointing at a hipcc): `pytest tests/kernels/quantization/test_rdna_prefill_gemm_sweep.py`.

## What needs a quiet-window re-measure

1. **The decision table itself.** Every number here is HIP 5 alone on its card, but HIP 2-4 carried other
   agents' GPU jobs and the box load was 3-20 (CPU builds, pytest). All rows run at the 174-175 W cap with
   shader clocks of 1.45-1.85 GHz, so a few percent of clock decide a few percent of TF/s. Re-run the
   confirmation command above with the neighbouring cards idle and compare the 25 `us` values and the
   `gfx_mhz` column; whether the neighbours move the clock at all is [inference] (shared PSU / airflow).
2. **End to end.** Nothing is wired, so there is no tok/s number. If a wiring is decided on, the number to take
   is chunked prefill at 512 / 1024 / 2048 against TunableOp rows added for these shapes (not against the
   heuristic, which would credit the kernel with the rows' gain).
3. **TunableOp rows.** `tunableop_results0.csv` was tuned for exactly M = 512 / 700 / 1024 / 1500 / 2048 by this
   sweep's own tuner; rows apply to the exact M only, so the M values the scheduler really emits need rows
   too (and the rows should be tuned on a quiet card).

## Not done, deliberately

* No fused decode-in-GEMM variant of the dense kernel: the bar for the dense kernel was not met (decision above),
  and the fusion could only remove the 1.6 % dequant share.
* No production wiring: no `rdna_hint_flex.py` / C++ / env-switch change; the kernel lives in the tools tree
  and is loaded by `sweep.py` only.
* Not tried (ideas, [inference] on the payoff): a persistent / stream-K tile scheduler for the wave-quantization
  tail (e.g. in_proj_qkvz M=2048 is 1024 tiles over 144 resident blocks), `s_setprio` bracketing of the dot2
  block as in Tensile's loops, non-temporal weight loads.

## Files

| path | what |
| --- | --- |
| `tools/rdna/rdna_hint_flex/gemm_sweep/gemm_variant.hip` | the one kernel, ~25 `-DCFG_*` knobs, MODE 0-5 (full GEMM and the one-pipe probes) |
| `.../sweep.py`, `report.py` | build / check / run / probe / occ / isa / remarks; tables |
| `.../isa_mix.py`, `gen_dot2_bank.py`, `prof_rocblas.py`, `clk_log.sh` | analysis helpers |
| `tests/kernels/quantization/test_rdna_prefill_gemm_sweep.py` | name grammar, presets, report tables (CPU); correctness vs fp32 and rocBLAS, strided / odd shapes (GPU) |
| `bench_results/2026-11-15_hi2-prefill-gemm/` | all CSVs (`sweep_a/c/d/e/f/g`, `confirm_f`, `baseline_a`, `check_*`, `diag_a/b`, `probe_a/c`), the logs of phases 1, 2, 4 and 9 and of the probes, `isa_summary.txt`, `isa_final.txt`, `dot2_bank.txt`, `prof_summary.md`, `tunableop_results0.csv`. Left out of the repository: the raw logs `phase5.log` .. `phase8.log` and `baseline_a.log` (0.6 MB); they are in commit `b47eb6c384` of branch `feat/hi2-integration` |
