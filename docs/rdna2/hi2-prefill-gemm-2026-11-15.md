# Dense fp16 prefill GEMM for HI2 on the V620: can a custom kernel beat rocBLAS? (2026-11-15)

Branch `feat/hi2-prefill-gemm` (from `feat/hi2-27b-opt` 74a27bb5d3). Box `par1-cs25`, tree
`~/w4a8_runs/leapquant/tree-hi2g`, **HIP 5 only** (PCI bus e2, a slice of lease b4 held by HI2;
`sweep.py` refuses any device that is not a V620 on the expected bus). Format and kernels:
`rdna-hint-flex-weights.md`; the prefill section there ("fused decode-in-GEMM against dequant +
rocBLAS") is the starting point. Labels: **[measured]** = produced by a command listed here,
**[derived]** = arithmetic on measured values, **[inference]** = reasoning a run must confirm.

**Measurement window.** `pgrep -f '^(/usr/sbin/|/usr/bin/)?docker buildx build'` returned nothing at
the start and at the end of every run in this note (logged in `bench_results/2026-11-15_hi2-prefill-gemm/`:
`phase1.log`, `run_probe2.log`, `phase4.log`; the baseline and screening sweeps logged it the same way).
No number here was taken during an image build. The box load average was 3-13 (other agents' CPU
jobs); the timed work is GPU-only, CUDA-graph launched.

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

`baseline_a.csv` / `sweep_a.csv`, medians, cold rotating weights, CUDA-graph timed. `us` and TF/s;
`dequant` = the HI2 dequant kernel alone; `dq+mm` = dequant followed by the rocBLAS GEMM (what the server
runs; heuristic / tuned rows). clk/W = cell median.

| shape (N x K) | M | rocBLAS heur us (TF/s) | tuned us (TF/s) | fused_ref us (TF/s) | dequant us | dq+mm heur | dq+mm tuned | MHz / W |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gate_up 17408x5120 | 512 | 3937 (23.2) | 3972 (23.0) | 4661 (19.6) | 603 | 4384 | 4407 | 1910 / 174 |
| gate_up | 700 | 6933 (18.0) | 6121 (20.4) | 6908 (18.1) | 603 | 7270 | 6229 | 1780 / 175 |
| gate_up | 1024 | 9312 (19.6) | 8132 (22.4) | 9259 (19.7) | 603 | 9698 | 8303 | 1785 / 175 |
| gate_up | 1500 | 14318 (18.7) | 12501 (21.4) | 13861 (19.3) | 603 | 14575 | 12409 | 1725 / 175 |
| gate_up | 2048 | 19105 (19.1) | 16508 (22.1) | 18430 (19.8) | 603 | 19412 | 16568 | 1720 / 175 |
| down 5120x8704 | 512 | 2264 (20.2) | 2263 (20.2) | 2491 (18.3) | 119 | 2334 | 2332 | 1950 / 175 |
| down | 700 | 3144 (19.8) | 3142 (19.9) | 3575 (17.4) | 122 | 3240 | 3221 | 1885 / 175 |
| down | 1024 | 4050 (22.5) | 4039 (22.6) | 4649 (19.6) | 122 | 4118 | 4124 | 1850 / 175 |
| down | 1500 | 5896 (22.7) | 5914 (22.6) | 7097 (18.8) | 122 | 5964 | 5947 | 1875 / 175 |
| down | 2048 | 7801 (23.4) | 7845 (23.3) | 9253 (19.7) | 123 | 7853 | 7851 | 1850 / 174 |
| in_proj_qkvz 8192x5120 | 512 | 1967 (21.8) | 1976 (21.7) | 2315 (18.6) | 119 | 2084 | 2088 | 1975 / 174 |
| in_proj_qkvz | 700 | 2881 (20.4) | 2830 (20.8) | 3300 (17.8) | 122 | 2982 | 2907 | 1940 / 175 |
| in_proj_qkvz | 1024 | 3751 (22.9) | 3709 (23.2) | 4449 (19.3) | 121 | 3832 | 3808 | 1888 / 175 |
| in_proj_qkvz | 1500 | 6156 (20.4) | 5338 (23.6) | 6590 (19.1) | 122 | 6258 | 5507 | 1875 / 174 |
| in_proj_qkvz | 2048 | 8028 (21.4) | 7400 (23.2) | 8725 (19.7) | 122 | 8128 | 7352 | 1865 / 175 |
| qkv_proj 7168x5120 | 512 | 1858 (20.2) | 1854 (20.3) | 2047 (18.4) | 106 | 1926 | 1921 | 1960 / 175 |
| qkv_proj | 700 | 2665 (19.3) | 2428 (21.2) | 2961 (17.4) | 107 | 2743 | 2510 | 1915 / 175 |
| qkv_proj | 1024 | 3506 (21.4) | 3270 (23.0) | 3895 (19.3) | 107 | 3561 | 3352 | 1882 / 174 |
| qkv_proj | 1500 | 5427 (20.3) | 4905 (22.4) | 5746 (19.2) | 108 | 5491 | 4872 | 1872 / 174 |
| qkv_proj | 2048 | 7067 (21.3) | 6453 (23.3) | 7588 (19.8) | 108 | 7165 | 6442 | 1860 / 175 |
| out_proj 5120x3072 | 512 | 853 (18.9) | 818 (19.7) | 893 (18.0) | 58 | 846 | 842 | 1995 / 175 |
| out_proj | 700 | 1133 (19.4) | 1133 (19.4) | 1268 (17.4) | 59 | 1172 | 1177 | 1940 / 174 |
| out_proj | 1024 | 1545 (20.9) | 1433 (22.5) | 1659 (19.4) | 59 | 1578 | 1472 | 1890 / 175 |
| out_proj | 1500 | 2263 (20.9) | 2267 (20.8) | 2515 (18.8) | 59 | 2288 | 2290 | 1895 / 175 |
| out_proj | 2048 | 2971 (21.7) | 2681 (24.0) | 3283 (19.6) | 59 | 2976 | 2772 | 1865 / 175 |

* rocBLAS runs 18-24 TF/s; the fused kernel 17.4-19.8 TF/s: it is slower than the best rocBLAS row in
  every cell (it ties the *heuristic* only at gate_up M >= 700, where the heuristic is weak).
* The dequant pass is 58-603 us: 3.1 % of dq+mm at gate_up M=2048, 13.7 % at M=512 [derived]; the profile's
  1.6 % is the all-layers average at the chunk sizes served.

### Side finding: TunableOp rows for the TP2 shapes (free, no new kernel)

The heuristic is **7-16 % slower than the TunableOp-tuned solution** at gate_up / in_proj_qkvz / qkv_proj
for M >= 700 and out_proj M = 1024 / 2048 (equal on `down`). Serving takes the heuristic for these
shapes: the shipped rows (`tunableop/rocm7.14-rocblas5.5/`) were curated for the 4B / 0.5B models and
contain no row for these N x K x M, and a row only applies to its exact M. The tuned rows used here are
`bench_results/2026-11-15_hi2-prefill-gemm/tunableop_results0.csv` (key
`GemmTunableOp_Half_TN,tn_<N>_<M>_<K>_...`). Given the 60 % GEMM share this is a larger free win
than any custom kernel in this note; it needs rows for the M values the scheduler actually produces.

Which Tensile kernel rocBLAS runs (torch profiler, `prof_summary.md`; all HHS, ISA1030, KLA assembly,
GLVW8/LRVW8, PGR1/PLR1, GSU1): **every pick at these shapes is a 128-accumulator-per-thread
kernel (TT8_16)**, never the 64-accumulator TT8_8 one: MT128x256x16 (256 threads, **199 VGPR,
28 KB LDS**) for `down` at all M, for gate_up / in_proj / qkv at M >= 1024 once tuned (WGM1 or WGM8
grouped rasterization), and MT128x128x16 (128 threads, **203 VGPR, 16 KB LDS**, WG16_8_1) for the
heuristic's gate_up / in_proj / qkv at M = 512, 2048 and most M = 512 rows. The tuned rows differ from
the heuristic mostly in macro tile (128x256 instead of 128x128) and WGM.

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
| rocBLAS Tensile MT128x256x16 TT8_16 WG16_16_1 WGM8 | 199 | 28672 | 4 [inference] | 1.128 | 0.047 | 0.023 | 0.003 | 0.004 | 18.4 | 256/1024 |
| rocBLAS Tensile MT128x128x16 TT8_16 WG16_8_1 WGM1 | 203 | 16384 | 4 [inference] | 1.139 | 0.047 | 0.031 | 0.004 | 0.004 | 18.4 | 256/1024 |

(`isa_summary.txt`; "same-bank" = the two VGPR sources of a `v_dot2c` share `reg mod 4`; the VALU-issue
ceiling is 18.43 / (VALU instructions per dot2).)

* **The two production prefill kernels are VALU-issue bound**: 1.8-1.9 VALU instructions per dot2
  (per-group scale / decode FMAs, `v_add_f32` accumulation across groups, bit extraction) cap them at
  ~10 TF/s per GHz = 18-20 TF/s at 1.8-2.0 GHz. That matches the measured 17.4-19.8 TF/s of the fused
  kernel [derived]. Neither can reach rocBLAS by tuning; a decode-once-into-LDS fused kernel would
  remove most of that overhead but then has the dense kernel's ceiling minus the decode, so it is moot
  unless the dense kernel wins (it does not, below).
* A plain dense micro-kernel has **1.13-1.28 instructions per dot2**: the instruction economy of my
  kernels equals rocBLAS's (1.128 / 1.139). The 128 accumulators per thread of the TT8_16 / 16x8 tiles
  lower the LDS read traffic from 0.25 to 0.19 B per flop (4 x ds_read_b128 per dot2) [derived].
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

gate_up M=2048: m0 19554 / m1 15810 / m2 8392 / m3 14393 / m4 15258 us (rocBLAS tuned 16205). Reading: the
memory pipe is not the limit at M=2048 (m4 is half the GEMM time on in_proj; it is ~the whole time on
gate_up M=512, where the GEMM is smaller than a wave of tiles) and the LDS pipe is 60 % busy; **the
dot2-only build itself reaches only 25 TF/s**: the dot pipeline, not the data path, is what falls
short of 18.43 x clock. The next two subsections explain that.

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
| k-loop unroll KU, tile rasterization RAST (grouped), waves-per-EU hint, split-K | 1/2/4; 0/4/8/16; 0/6; 1 and 2/4 |
| B-pair rotation BROT, dot2 chain order ORD | 0/1, 0/1/2 |

Designs: 170 (screening sweep `sweep_a`), 44 more (BROT / ORD sweep `sweep_c`), 87 built for the
neighbourhood and rotation studies; **257 distinct MODE-0 designs pass the 65-case correctness check**
(worst abs 0.0078, worst relative Frobenius error 2.12e-4, `check_*.csv`). A kernel bug found on the way:
`__builtin_bit_cast` of a vector-element *lvalue* read element 0 for every k-pair (clang 23, gfx1030),
which made early numbers look like 27.5 TF/s; the kernel now passes unsigned scalars to `dot2u`.
