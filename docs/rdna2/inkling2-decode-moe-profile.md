# Inkling-2-Mini decode (b = 1..4, TP = 1, gfx1030): MoE expert kernels, Infinity Cache, prefetch, attention decode

Measured on `par1-cs25`, V620 (HIP 3), merge tree `feat/inkling2-merge@5a5331fddb` (`csrc` unchanged), ROCm 7.14. Every
number below is from a log under `data/decode/` (raw paths in section 9); `[INFERENCE]` marks arithmetic, not a measurement.
All kernels are timed in HIP graphs over cold rotating weights (more than 128 MB between two reads of one weight) unless
the text says "hot".

## 1. Headline

* Native-exact (no requantisation, no calibration) b = 1 decode moves **5.34 GB of weights per step** (dense attention
  2.58 GB, lm_head 0.92 GB, NVFP4 experts 1.58 GB, MXFP8 experts 0.26 GB). The V620 read ceiling measured with a plain
  streaming kernel is **440.7 GB/s** (80 CTA/unroll/non-temporal configurations, 437-441 GB/s), so the DRAM floor of one
  step is **12.1 ms** (10.4 ms even at the 512 GB/s spec). The DRAM-bound kernels of the step (dense GEMV 8.22 ms + expert
  GEMM 5.38 ms = 13.6 ms) already sit at the launch-aware floor (12.1 ms + 190 launches x 6.5 us = 13.4 ms).
  **A decode step of 10 ms/token at b = 1 with native weights is below the DRAM floor.** It needs fewer bytes per token
  (opt-in quantised dense weights) or more than one token per weight stream (speculation).
* Exact-native spec-off is therefore a 17.8 ms (GPU graph) -> about 14.5-15 ms problem: the non-DRAM part of the step
  (5.1 ms of small kernels, each at least ~3 us of launch and ramp) and the last 0.5 ms of the NVFP4 kernel.
* The expert GEMV kernels are almost done: shipped NVFP4 is 80 % of the stream floor, DecodeGlue's bit-identical fast
  dequant (`moe_fast`) 90-97 %, MXFP8 is at the floor (95-99 %).
* The Infinity Cache cannot give cross-visit reuse (about 1 GB of other traffic between two visits of a layer, cache
  128 MB LRU-like), and a side-stream prefetch of the next visit's dense weights is worth at most 3-4 % (measured twice
  independently: here and in DecodeGlue's `prefetch_chain`).

## 2. Step breakdown as run by the engine (ctx 22, b = 1..4, `FULL_AND_PIECEWISE`)

`decode_profile_ctx.py` (real engine, torch profiler, kernel table by name). Times are sums of kernel durations per step.
`GPU graph` = DecodeGlue's un-profiled `GRAPH_REPLAY` of the b = 1 step in its tree (17.78 ms): equal to the kernel-duration
sum, so the "idle between kernels" of the profiled run (2.4 ms) is profiler overhead, not real gap time [INFERENCE: same graph].

| family (ms per step / launches) | b=1 | b=2 | b=3 | b=4 |
|---|---|---|---|---|
| dense GEMV: attn_in 48 x 93.5 us, o_proj 48 x 34.5 us, lm_head 1 x 2095 us | 8.22 / 97 | 8.33 | 8.41 | 8.47 |
| NVFP4 expert GEMM (44 visits x w13 + w2) | 4.73 / 88 | 7.68 | 10.38 | 12.05 |
| MXFP8 expert GEMM (4 visits x w13 + w2) | 0.65 / 8 | 1.17 | 1.61 | 1.83 |
| moe route (`inkling2_glue_moe_route`, 15.2 us per visit) | 0.73 / 48 | 0.78 | 0.90 | 0.92 |
| moe swiglu | 0.21 / 48 | 0.23 | 0.27 | 0.26 |
| glue (attn_res, gate_mix, gate_down, ...) | 1.90 / 226 | 1.92 | 2.08 | 2.07 |
| kv_prep + sconv | 0.64 / 144 | 0.65 | 0.66 | 0.66 |
| attention decode + combine | 0.54 / 96 | 0.62 | 0.67 | 0.74 |
| other + sampler | 0.11 / 28 | 0.10 | 0.10 | 0.11 |
| **kernel-duration sum** | **17.75** | 21.45 | 25.13 | 27.06 |
| MoE family (experts + route + swiglu) | 6.32 (36 %) | 9.86 | 13.16 | 15.06 |

Per extra verified row (b = 1 -> 4): +2.4 ms in the NVFP4 kernels, +0.4 ms MXFP8, +0.1 ms route/glue: every row brings
its own ~8 experts (distinct experts per visit at T = 1/2/3/4: 8.0 / 15.5 / 22.6 / 29.2).

One core-layer visit (kernels 386-403 of the b = 1 trace, 18 launches, 343 us of kernel time):
attn_in 93.2 | kv_prep 5.9 | decode 8.4 | combine 3.3 | o_proj 34.8 | sconv 3.4 | attn_res 5.8 | gate_down 4.6 |
gate_mix 14.0 | route 15.2 | w13 71.8 | swiglu 4.8 | w2 36.7 | sconv 3.4 | attn_res 7.7 | attn_res 5.7 | gate_down 7.0 |
gate_mix 17.3 (us). The four DRAM-bound kernels are 237 us of the visit; the 14 small kernels 106 us.

## 3. Expert GEMV kernels vs the bytes roofline (cold rotating weights, graph-timed, us per launch)

Floor = `stream_read` kernel reading the same number of bytes from a 3 GB rotating buffer, best of 36 CTA/unroll/nt
configurations (all within 2 % of each other): the per-launch cost of a decode-sized read is bytes / 440.7 GB/s + 6.5 us.

| launch (8 distinct experts) | bytes | bytes / 440.7 GB/s | stream-read floor | shipped | DecodeGlue `moe_fast` | shipped / floor |
|---|---|---|---|---|---|---|
| NVFP4 w13 (2304 -> 2304) | 23.9 MB | 54.2 us | **60.7 us** (394 GB/s) | 75.3 us (317 GB/s) | 67.1 us (356 GB/s) | 124 % |
| NVFP4 w2 (1152 -> 2304) | 11.9 MB | 27.1 us | **34.0 us** (352 GB/s) | 40.0 us (299 GB/s) | 35.1 us (340 GB/s) | 118 % |
| NVFP4 pair (one visit) | 35.8 MB | 81.3 us | **94.7 us** | 119.5 us | 106.2 us | 126 % |
| MXFP8 w13 | 43.8 MB | 99.4 us | **105.4 us** (416 GB/s) | 110.5 us (397 GB/s) | not built | 105 % |
| MXFP8 w2 | 21.9 MB | 49.7 us | **56.3 us** (389 GB/s) | 56.9 us (385 GB/s) | not built | 101 % |
| MXFP8 pair | 65.7 MB | 149 us | **161.7 us** | 174.0 us | not built | 108 % |

Rows 2-4 (BM = 1 is the best block size at every T <= 4 for both formats; BM 2/4 are never faster):

| T (distinct experts) | NVFP4 shipped pair | NVFP4 `moe_fast` pair | stream floor of those bytes | MXFP8 shipped pair |
|---|---|---|---|---|
| 1 (8.0) | 119.5 us | 106.2 us | 94.7 us | 174.0 us |
| 2 (15.5) | 203.5 us | 185.2 us | 175 us [INFERENCE: 2 x (bytes/440.7 + 6.5)] | 318.4 us |
| 3 (22.6) | 277.3 us | 259.7 us | 244 us [INFERENCE] | 452.2 us |
| 4 (29.2) | 356.1 us | 329.3 us | 311 us [INFERENCE] | 577.7 us |

* Shipped NVFP4 -> `moe_fast`: -13.3 us per visit at T = 1 = **-0.59 ms per step** (44 visits). DecodeGlue's end-to-end
  switch of the same flag gives 18.65 -> 18.07 ms/token (-0.58 ms), the two agree.
* `moe_fast` -> floor: 11.5 us per visit at T = 1 = **0.51 ms per step** at most (w13 +6.4 us, w2 +1.1 us). At T = 4 it is at 94 %.
* MXFP8: 4 visits x 12 us = 0.05 ms from the floor: nothing to gain.
* What the ISA of the shipped NVFP4 BM = 1 loop showed (`scratch/decode/nvfp4_bm1.s`, 99 VGPR, 9 waves/SIMD): 8
  serialised `global_load_sbyte` scale loads with `s_waitcnt vmcnt(0)` per 32-K iteration before the four 128-bit weight
  loads, divergent exec-mask branches in the e4m3 subnormal decode, ~840 VALU per iteration (about 45 us of ALU at 1.9 GHz
  against ~50 us of DRAM): the kernel waits on ALU and on the scale latency chain, not on the stream. `moe_fast`
  (v_perm dequant, branch-free bit-identical e4m3 scale, 32-bit scale loads) removes most of it; that is where its 13 us
  come from. What is left (0.5 ms/step): the 10 % in w13 (grid of 216 CTAs of 8 waves = 1.33 rounds over 1296 resident
  wave slots, idle y = 2 N-tile) and the 6.5 us launch ramp of each of the two launches.
* Launches per visit: 4 (route, w13, swiglu, w2), each followed by a graph dependency; swiglu fused into w2 (DecodeGlue
  `moe_nvfp4_gemm_swiglu_rdna2`) removes one launch (4.4 + 4.8 us per visit = up to 0.44 ms/step).
  `moe_route` costs 15.2 us per visit (0.73 ms/step) for a 128-way top-8: the largest non-bandwidth item of the MoE path.

## 4. Infinity Cache and the loop visits

Eager event-timed `stream_read` (`decode_moe.log`, section `ic`): read A (36 MB), then d MB of other data, then A again.

| d (other traffic between) | 0-64 MB | 96 MB | >= 112 MB |
|---|---|---|---|
| second read of A = 36 MB | 32.6 us (1158 GB/s, 3.0x faster) | 64 us (partial) | 99 us (miss, 380 GB/s) |

A = 90 MB hits at d <= 32 MB (61.8 us, 1526 GB/s) and misses at d >= 64 MB: effective capacity about 128 MB, LRU-like.

* The model runs the core layers 6-17 three times (48 visits). Two visits of the same layer are 11 layers apart: 11 x
  (54 MB dense + 36 MB experts) = ~1 GB of other traffic, 8x the cache. **No second-visit weight read hits the cache.**
  The order cannot be changed (each pass consumes the previous pass's output).
* A hit does not make the real dense GEMV 3x faster: `gemv_kernel` on the 39.6 MB attn_in weight takes 93.2 us cold
  (425 GB/s) and 49.9 us hot (794 GB/s; `gemv_hot.log`); the kernel, not the cache, limits it above ~800 GB/s. [The
  o_proj rows of that file used 8 sets x 14 MB = 113 MB, i.e. inside the cache: they are hot in both columns (15.1 us);
  the in-situ cold o_proj is 34.5 us.]
* Prefetch of the next visit's dense weights (attn_in + o_proj = 53.8 MB, statically known) on a second graph branch: real
  attn_in / o_proj / w13 / w2 kernels on cold rotating weights, the 14 small kernels as spin kernels of their measured
  duration, 48 visits per graph (`visit_replay.log`; 406 us per visit against 424 us for the profiled engine visit; each
  spin kernel carries its own ramp on top of the duration taken from the profile, so the replay, if anything, overstates the
  hide-able small-kernel time [INFERENCE]):

| variant (prefetch issued after kernel k, joined before the next visit) | us per visit | vs base |
|---|---|---|
| base (no prefetch) | 406.0 | 1.000 |
| o_proj weights only, after o_proj (pf4wo) | 394.3 | 0.971 |
| both, after o_proj, no join (pf4n) | 393.8 | 0.970 |
| attn_in weights only, after o_proj (pf4wi) | 410.0 | 1.010 |
| both, after o_proj (pf4w) / non-temporal loads (pf4wt) | 416.4 / 416.3 | 1.026 / 1.025 |
| both, after w2 (pf12w) / after the router (pf9w) / after w13 (pf10w) / after attn_in (pf0w) | 415.5 / 452.4 / 444.5 / 443.6 | 1.023 / 1.114 / 1.095 / 1.093 |

  DecodeGlue's independent chain (`prefetch_chain.log`: its own spin glue, 8 routed experts' w2 prefetch included):
  pfA +0.55 ms (worse), pfW -0.19 ms, a throttled window -0.8 ms of 18.3 (-4.4 %). Both agree: **a prefetch saves 0.6-0.8
  ms per step at best.** Why not the ~3 ms the hot-GEMV numbers suggest (63 us per visit): the 54 MB still costs 123 us of
  DRAM time, the DRAM-idle windows of a visit are 31 + 65 + 9 + 63 us in four fragments, and while the MoE kernels
  (DRAM-bound too) run the prefetch only steals bandwidth from them.

* A side effect for anyone profiling these graphs: `torch.profiler` on a graph with a forked stream hangs ("Async
  signal handler still waiting on signal" from the queue interposer); time such graphs with events only.

## 5. Attention decode (b = 1, TP = 1)

| ctx | attention decode + combine per step | `i2_decode_kernel` mean (48 launches) | `i2_combine_kernel` | share of the step | GPU kernel sum |
|---|---|---|---|---|---|
| 22 | 0.54 ms | 8.2 us | 3.1 us | 3.0 % | 17.75 ms |
| 1024 | 1.03 ms | 17.2 us | 4.2 us | 5.6 % | 18.37 ms |
| 16384 | 2.27 ms | 42.8 us | 4.5 us | 11.5 % | 19.72 ms |

At 16k the 8 full-attention visits cost about 170 us each [INFERENCE: (2054 us - 40 sliding-window visits x 17.2 us, the
1k cost, window 1024) / 8]; 16384 keys x 8 KV heads x 512 B = 67 MB per visit -> 395 GB/s, at the DRAM floor. At 22 and 1k
context the kernel is latency-bound (4.2 MB in 17 us) but the whole family is 0.5-1.0 ms. Attention decode is not a lever
below ~16k context (a free kernel would save < 3 % of the step at 22 and 5.6 % at 1k); at 16k the K/V stream is
bandwidth-bound already.

## 6. Where this leaves the b = 1 decode target

| item (all exact math, native weights) | per step | status / evidence |
|---|---|---|
| DRAM floor of 5.34 GB at 440.7 GB/s | 12.1 ms | measured ceiling (`peak_sweep.log`) |
| launch-aware floor of today's 190 DRAM-bound launches | 13.4 ms | 12.1 + 190 x 6.5 us |
| GPU graph now / bench per token | 17.8 / 18.65 ms | DecodeGlue `GRAPH_REPLAY`, bench; Lead 19.3 |
| fast exact NVFP4 dequant (`moe_fast`) | -0.58 ms | microbench -0.59, e2e -0.58 |
| NVFP4 w13/w2 to the stream floor | -0.51 ms max | not started: 10 % of w13 |
| swiglu fused into w2 | -0.2 .. -0.44 ms | DecodeGlue |
| `moe_route` 15.2 us -> ~5 us [INFERENCE] | -0.5 ms | glue TU |
| windowed prefetch | -0.6 .. -0.8 ms | two independent measurements |
| glue / small-kernel fusion (226 + 144 + 96 launches = 4.1 ms busy + route/swiglu) | -2 .. -3 ms | DecodeGlue |
| **realistic spec-off b = 1** | **~14.5-15 ms (67-69 tok/s)** [INFERENCE: not all levers land in full] | |
| <= 10 ms/token | below the 12.1 ms floor | needs fewer bytes (opt-in W8 / W4 attention + lm_head: 3.50 GB -> 1.75 / 0.88 GB, step floor 8.1 / 6.2 ms [INFERENCE: scales ignored]) or >= 1.3 tokens per weight stream |

## 7. Not planned / ruled out

* More time on the NVFP4 kernel beyond `moe_fast`: <= 0.5 ms/step for a persistent software-pipelined rewrite.
* Cross-visit weight caching: no (section 4).
* Re-ordering the loop passes: dependency chain, impossible at b = 1.
* Exact lm_head pruning for greedy decoding (norm/cluster bounds): bounds are far looser than the logit gaps at 2304
  dimensions [INFERENCE], and sampling needs every logit; not pursued.

## 8. Reproduction (all on `par1-cs25`, work dir `~/w4a8_runs/inkling2-work/Inkling2PrefillProfile/`)

```bash
S=~/w4a8_runs/inkling2-work/Inkling2PrefillProfile; export TREE=$S/tree I2_JIT_DIR=$S/jit ROCM_HOME=/opt/rocm/core-7.14 PATH=/opt/rocm/core-7.14/bin:$PATH
G="bash ~/w4a8_runs/inkling2-work/Lead/gpu_sub.sh 3 600"
$G $S/scripts/decode_moe_bench.py --mode roofline,experts,ic --json $S/out/decode/decode_moe.json      # sections 3, 4
$G $S/scripts/decode_moe_bench.py --mode experts --formats nvfp4 --tokens 1,2,3,4 --bms 1 \
     --variants nv_fast=$S/scratch/decode/variants/moe_fast_glue_snapshot.cu                         # section 3, moe_fast
$G $S/scripts/decode_profile_ctx.py                                                                  # sections 2, 5
$G $S/scripts/visit_replay.py --variants base,pf4w,...                                               # section 4 replay
$G $S/scripts/gemv_hot.py ; $G $S/scripts/peak_sweep.py                                              # hot GEMV, DRAM ceiling
```

## 9. Raw artefacts

`logs/decode_moe.log`, `out/decode/decode_moe.json`, `logs/decode_moe_fast.log`, `logs/decode_profile_ctx.log`,
`out/decode/profile_ctx/results.json`, `logs/visit_replay.log`, `logs/gemv_hot.log`, `logs/peak_sweep.log` under the work
dir; local copies in `data/decode/`; DecodeGlue's `~/w4a8_runs/inkling2-work/Inkling2DecodeGlue/out/{prefetch_chain,
prefetch_win,overlap_probe,moe_fast_rows,fast_e2e_chain}.log`.
