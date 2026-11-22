# Inkling-2-Mini prefill profile on gfx1030 (V620), TP=1

Measured on `par1-cs25`, HIP device 3, real checkpoint (`~/hfcache/inkling2-mini`), merged tree
`feat/inkling2-merge@5a5331fddb` (csrc identical to `0722fa5075`), eager prefill. Scratch only: no file of the worktree was
changed. Everything marked **[INFERENCE]** is arithmetic or an estimate, not a measurement.

## 1. Summary

1. **The prefill is MoE-bound, and the hypothesis is confirmed.** At T=1024 the two NVFP4 expert GEMMs are 72.0 % of the
   forward (w13 47.6 %, w2 24.4 %) and run at 9.1 / 8.9 TFLOP/s, i.e. 29 % of the dot2 peak (about 31 TFLOP/s at
   1.8 GHz, measured). The dense GEMMs run at 25-28 TFLOP/s (83-90 % of that peak), so they are not the problem.
2. **Current numbers** (clean wall): 1000-token prompt 873-883 ms = 1136 tok/s (the 1262 ms in the docs is stale);
   chunk curve flat at 1070-1136 tok/s from 512 to 4096 tokens; 1024-token chunks at depth: 895 ms (0), 1124 (24.6k),
   1336 (49k), 1545 (73.7k), 1764 (100k); a 101k-token prompt averages **764 tok/s**.
3. **Root cause of the experts**: the kernel is a decode kernel (BM = 1..8 rows per block, K split with fp16
   atomics, about 5 VALU ops of dequant per weight per 8 rows, so only about 40 % of the issue slots are dot2). At
   about 62 rows per expert every expert tile is re-streamed and re-dequantized about 8 times per call.
4. **A scratch rewrite (whole-K, chained dot2, BM16, cheap dequant on the SHIPPED weight layout) is 1.6-2.0x faster per
   launch (1.84x on the captured real routing) and 1.47x end to end** on the real checkpoint: 1000 tokens 880 -> 597 ms = **1675 tok/s**, no weight
   repack, no decode-kernel change, accuracy better than the shipped kernel against an fp32 reference (kernel rel L2
   3.7e-4 / 5.9e-4 vs 6.1e-4 / 1.06e-3).
5. **At depth, attention takes over**: 51 % of a chunk at 100k (889 ms, 12.0 TFLOP/s). Two exact, cheap changes (skip the
   relative-bias block for key tiles outside the bias window; `splits_tile=3` for deep full-attention chunks) give
   -10 % attention. Together with the expert rewrite of item 4, on a 101,376-token prompt: **764 -> 1023 tok/s average**,
   chunk at 100k 1764 -> 1383 ms.
6. **2.5k tok/s at TP=1 is reachable only if every lever lands** (section 6) and only for short or medium
   prompts: at 100k depth the attention alone is about 427 ms per 1024-token chunk even at 25 TFLOP/s.
7. **TP=2/TP=4 1M-token prefill is attention-bound (65 %)**; the attention tile at 13.35 TFLOP/s is the lever there
   (20 TFLOP/s = 1.28x at TP=4); its occupancy is LDS-limited (2 CTAs per WGP) and timing probes show +15 % at 3 and +28 %
   at 4 CTAs with the same instructions; the TP=2 chunk is capped by memory (96..184 tokens), not by kernels.

## 2. Method and commands

Harness: `profile_prefill.py` (scratch; builds on `tools/rdna/inkling2/profile_decode.py` labels). It boots the engine
(`LLM(..., enforce_eager=True, max_num_seqs=2)`), runs clean (unprofiled) steps for wall times, then one
`torch.profiler` step per size and groups the kernel table by op family; GPU kernel sum is 98-99 % of wall, about 1027
kernels per forward. Prompts are real text (repo docs) unless stated.

```bash
P=~/w4a8_runs/inkling2-work/Inkling2PrefillProfile
G="env TREE=$P/tree I2_JIT_DIR=$P/jit ROCM_HOME=/opt/rocm/core-7.14 PATH=/opt/rocm/core-7.14/bin:$PATH \
   bash ~/w4a8_runs/inkling2-work/Lead/gpu_sub.sh 3"           # HIP 3 only; check the log for "python exit="
# baseline: capture routing, 1000 tokens, chunk curve (kernel tables + clean walls)
$G 1200 $P/scripts/profile_prefill.py --ckpt ~/hfcache/inkling2-mini --out-dir $P/out/prefill_a \
   --scenarios capture,p1000,chunks --mnbt 4096 --max-model-len 8192
# baseline at depth (1024-token chunks, 101,376-token prompt)
$G 1200 $P/scripts/profile_prefill.py ... --out-dir $P/out/prefill_depth --scenarios depth --mnbt 1024 \
   --max-model-len 110000 --fill 101376 --profile-at 0,24,48,72,98
# any scratch expert kernel dropped in (sections 5 and 13): staged BM16 on the shipped layout
$G 1200 $P/scripts/profile_prefill.py ... --out-dir $P/out/prefill_v4_ch2_bm16 --scenarios p1000,chunks --mnbt 4096 \
   --moe-plugin $P/scripts/moe_plugin_v4.py --plugin-arg src=$P/scripts/moe_v4.cu --plugin-arg bm=16 --plugin-arg cheap=2
# everything: v4 experts + attention variant f1 + splits_tile 3 at depth
$G 1200 $P/scripts/profile_prefill.py ... --out-dir $P/out/prefill_v4_depth --scenarios depth --mnbt 1024 \
   --max-model-len 110000 --fill 101376 --profile-at 0,24,48,98 --moe-plugin ... --plugin-arg bm=16 --plugin-arg cheap=2 \
   --attn-variant $P/scratch/attn/attn_f1.cu --attn-split-tile 3
# expert microbenchmarks (repo suite, scratch v4)
$G 600 $P/scripts/moe_v4_bench.py --src $P/scripts/moe_v4.cu --tokens 16,...,512 --stages w13,w2 --cfgs 8:0:2,16:0:2 --base-bms 4,8
```

Clock/power: sustained sclk 1.72-1.95 GHz at 120-178 W under the 175 W cap; microbenchmarks ran 1.58-2.1 GHz. The
instruction calibration (`calib_dot.log`, 174 W) gave **dot2 = 31.2 TFLOP/s at 1824 MHz (17.1 TFLOP/s per GHz, 0.99
issue/clk/SIMD)**, `v_dot4_i32_i8` exactly 2x dot2 (62.9 TFLOP/s-eq), fp32 FMA 16.1. Numbers below that compare two
variants were taken in the same job unless noted; run-to-run clock noise is about +-4 %.

## 3. Baseline profile (shipped kernels)

### 3.1 Clean wall by chunk size (text prompt)

| chunk tokens | ms | tok/s |
|---:|---:|---:|
| 184 | 215.7 | 853 |
| 512 | 478 | 1071 |
| 1000 (prompt) | 873-883 | 1136 |
| 1024 | 904.8 | 1132 |
| 2048 | 1836 | 1116 |
| 4096 | 3744 | 1094 |

The curve is flat: chunking is not the problem. `MAX_MOE_TOKENS = 1024` splits the MoE calls (about 62 rows/expert).
The cost per token rises only below 512 (184 tokens: 1.17 ms/token vs 0.88).

### 3.2 Per-family breakdown, T = 1024, depth 0 (GPU 886.3 ms, 1027 kernels)

Shapes: attention input 2304 -> ~8.6k rows, o_proj 3072 -> 2304, experts w13 2304 -> 2304 (gate+up 1152 each) and
w2 1152 -> 2304, 128 experts top-8, NVFP4 on layers 0-19 (44 visits), MXFP8 on layers 20-23 (4 visits).
Ceiling = dot2 peak 30.8 TFLOP/s at 1.8 GHz [calibration x clock].

| family | ms | share | launches | us/launch | GFLOP | TFLOP/s | of peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| moe w13 nvfp4 | 422.1 | 47.6 % | 44 | 9593 | 3826.8 | 9.07 | 29 % |
| moe w2 nvfp4 | 215.9 | 24.4 % | 44 | 4906 | 1913.4 | 8.86 | 29 % |
| attn_in GEMM | 73.9 | 8.3 % | 48 | 1539 | 1949.0 | 26.4 | 86 % |
| attention (tile+combine) | 36.3 | 4.1 % | 144 | 252 | 309.5 | 8.5 | 28 % |
| moe w13 / w2 mxfp8 | 32.6 / 16.9 | 3.7 / 1.9 % | 4 / 4 | 8150 / 4225 | 348 / 174 | 10.7 / 10.3 | 34 % |
| o_proj GEMM | 27.5 | 3.1 % | 48 | 574 | 695.8 | 25.3 | 82 % |
| glue msg_gate (gate_mix 13.6 + gate_down 11.7) | 25.4 | 2.9 % | 96 | 264 | 1.81 per G=3 call | 6.6 (mix), 7.7 (down) | 21-25 % |
| moe route (sort + zero fills) | 12.8 | 1.4 % | 48 | 267 | - | - | |
| glue attn_res | 11.9 | 1.3 % | 120 | 99 | - | - | |
| moe swiglu | 5.0 | 0.6 % | 48 | 103 | - | 570 GB/s | |
| sconv / lm_head / kv_prep / conv_write / align | 2.5 / 2.3 / 1.1 / 1.1 / 0.6 | | | | | | |

MoE 77.6 % (NVFP4 72.0 %), dense GEMMs 11.4 %, attention 4.1 %, glue + route + small ops 7 %.
Per visit at T=184: w13 2.18 ms, w2 1.11 ms; per launch at T=4096: w13 9.9 ms, w2 5.05 ms (the same 9 TFLOP/s).

### 3.3 Depth (1024-token chunks, 101,376-token prompt, shipped)

| depth | chunk ms | attention ms (share) | attention TFLOP/s |
|---:|---:|---:|---:|
| 0 | 895 | 36 (4 %) | 8.5 |
| 24,576 | 1124 | 268 (24 %) | 11.4 |
| 49,152 | 1336 | 467 (35 %) | 11.8 |
| 73,728 | 1545 | ~675 (44 %) | 11.9 |
| 100,352 | 1764 | 889 (51 %) | 12.0 |

About 8.7 ms per 1k tokens of depth; the whole prompt: 101,376 tokens in 132.7 s = **764 tok/s**. A full-attention visit
of a 1024-token chunk costs 104 ms at 100k and 26.7 ms at 24.6k; sliding visits about 1.35 ms (6 % at depth).
`bench_attn` per visit (q = 1024): ctx 16,384 14.9 ms (13.4 TFLOP/s), 65,536 64.1 ms, 102,400 102 ms (12.6),
131,072 131.4 ms (12.5). Attention FLOPs of a chunk at 100k = 10.67 TFLOP (harness count: 10.41 full-attention + 0.26
sliding); at 20 TFLOP/s that is 533 ms (-355 ms vs 889), at 25 TFLOP/s 427 ms.

## 4. Expert GEMMs: microbenchmark vs ideal

Uniform routing, all 128 experts, per launch, shipped `moe_nvfp4_gemm_rdna2` (best of BM4/BM8) vs scratch v4
(sections 5 and 9.1), same job (`v4_rows.log`, `v4_rows.json`). Ideal compute = 2 x T x 8 x K x N at 30.8 TFLOP/s
(w13 84.9 MFLOP/token, w2 42.5); weight-stream floor = 382 MB (w13) / 191 MB (w2) per launch (4 bit + 1/16 byte scale)
at the 512 GB/s spec bandwidth = 0.75 / 0.37 ms.

| T | rows/expert | w13 shipped | v4 BM8 | v4 BM16 | ideal compute | w2 shipped | v4 BM8 | v4 BM16 | ideal |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 4 | 1182 (BM4) | 1034 | 1310 | 176 | 548 (BM4) | 498 | 606 | 88 |
| 128 | 8 | 1880 | 1216 | 1526 | 353 | 869 | 575 | 692 | 177 |
| 192 | 12 | 2603 | 1663 | 1650 | 529 | 1266 | 734 | 711 | 265 |
| 256 | 16 | 3129 | 2010 | 1899 | 706 | 1565 | 959 | 944 | 353 |
| 384 | 24 | 4305 | 2801 | 2664 | 1059 | 2034 | 1246 | 1279 | 530 |
| 512 | 32 | 5548 | 3530 | 3197 | 1412 | 2715 | 1638 | 1675 | 706 |
| 1024 | 64 | 8970 | 6102 | 5497 | 2823 | ~4780 | | ~2680 | 1412 |
| 4096 | 256 | 37422 | 21659 | 18919 | 11290 | 18747 | 11449 | 10070 | 5646 |

(us per launch; T=1024/4096 rows from `v4_a/b.json` with captured or uniform routing, clocks 1.8-2.0 GHz.)
Real routing (captured, T = 1000, 9 visits): 0-49 of 128 experts get zero rows, max 323-999 rows per expert; padding
x1.04-1.06 at BM8, x1.10-1.11 at BM16, x1.22-1.26 at BM32. w13, captured visit 6: shipped 9698 us, v4 BM8 5879,
v4 BM16 5271 us (16.1 TFLOP/s); w2: 4786 / 2929 / 2681.
In the engine the shipped kernel reaches 29 % of the compute ideal (w13 9.1 of 30.8 TFLOP/s), v4 BM16 53 % (16.4). The
weight-stream floor is never the limit for T >= 256 (0.75 ms vs 2.8 ms compute at T=1024): the kernels are VALU-bound.

### 4.1 Why: instruction model (measured on the ISA)

Per 32-K step per thread (4 columns): shipped BM8 = 8 dot2 per 2 weights + about 10 dequant ops (about 5 per weight),
about 40 % dot2, 112 VGPR / occupancy 9, 92 % VALU-issue efficient: it is bound at about 12.5 TFLOP/s and reaches 9.
v4 BM16 CH=2, counted from the assembly (`scratch/isa/v4_bm16_ch2.s`, `census2.awk`), per iteration (16 rows x 4
columns x 32 K = 2048 MAC):

| class | instr | note |
|---|---:|---|
| v_dot2_f32_f16 | 1024 | 64 % of 1602 |
| bit ops (dequant) | 246 | 1.9 per weight |
| v_pk_mul (scale) | 64 | 0.5 per weight |
| other VALU (e4m3 scale decode, address) | 99 | |
| ds_read (b128) | 64 | |
| s_waitcnt / SALU / VMEM | 33 / 46 / 6 | |

VALU-slot bound = 1024 / (1024 + 410) = 71 % -> 22 TFLOP/s at 31 peak; the kernel reaches 16.1-16.6 (73 % of that
bound; 187 VGPR / 5 waves, LDS 9.5 KB). Dequant is 2.4 ops per weight in both the bit ops and the pk_mul; with BM rows the
per-row cost is 64 + 410/BM slots: BM8 115, BM16 90, BM32 77 [INFERENCE: weighting by the real padding 1.05 / 1.105 /
1.24 predicts BM32 4 % better than BM16; measured BM32 is 4 % worse (192 VGPR, 19 KB LDS), so scheduling, not the
instruction count, limits BM32].

## 5. Prototype results: expert kernel and end to end

Variants measured at kernel level (`variants_a.log`, `v4_a.log`, `v4_b.log`, uniform T=1024 w13, us): shipped BM8 8970;
v2 chained BM16 (1.28x); v4 CH=0 BM16 (adjacent pairs, no cheap dequant) 6480; **CH=2 BM8 6102**; CH=1 BM16
(repacked words) 5418; **CH=2 BM16 5497**; CH=2 BM32 5545. Register double-buffer prefetch costs 3-5 %; BM32 with
`__launch_bounds__(256)` spills. CH=2 (nibble pairs (p, p+4) on the **shipped** layout, activations permuted with
`v_perm` while they are staged to LDS) is within 1.5 % (w13) / 4.5 % (w2) of the repacked layout, which is why no
weight repack is needed.

End to end (real checkpoint, `Inkling2LoopMoE.forward_hip` patched on the 20 NVFP4 layers, MXFP8 layers untouched,
harness re-sorts with `moe_align_block_size(BM)` after the glue route; clean wall):

| config | p1000 ms | tok/s | 184 | 512 | 1024 | 2048 | 4096 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| shipped | 873-883 | 1136 | 216 | 478 | 905 | 1836 | 3744 |
| v4 CH=2 BM8 (stage 1: no route, workspace or loader change) | 642.3 | 1557 | **162.0** | 349.9 | 656.6 | 1335.8 | 2727 |
| v4 CH=2 BM16 (stage 2) | 597.2 | **1675** | 171.2 | 335.6 | 610.6 | 1243.4 | 2547 |
| v4 CH=1 BM16 (repacked, reference only) | 589.8 | 1695 | 165.3 | 331.9 | 603.2 | 1227.4 | 2517 |

BM8 wins below about 192 tokens per call (162 vs 171 ms at 184), BM16 from 192-256 on (table in section 4). Family
table of BM16 CH=2 at T=1000 (577.6 ms GPU): w13 225.4 ms (16.6 TFLOP/s), w2 116.6 (16.0), attn_in 67.1 (28.4),
attention 33.6, mxfp8 30.9 + 16.0, o_proj 27.8, msg_gate 24.2, route 12.35, attn_res 11.4, swiglu 4.7.

**At depth, with the attention changes of section 7.2** (`prefill_v4_depth`: v4 BM16 + attention variant f1 +
`splits_tile=3` for window-0 chunks with >= 32768 keys, 152 MiB scratch workspace; `compare_depth.py`):

| depth | shipped ms | new ms | speedup | attention ms (new, TFLOP/s) |
|---:|---:|---:|---:|---:|
| 0 | 895.1 | 605.8 | 1.48x | 34.8 (8.9) |
| 8,192 | 982.2 | 686.2 | 1.43x | |
| 24,576 | 1123.8 | 815.1 | 1.38x | 251.6 (12.1) |
| 49,152 | 1336.4 | 993.2 | 1.35x | 418.4 (13.2) |
| 65,536 | 1474.7 | 1114.8 | 1.32x | |
| 81,920 | 1613.9 | 1236.9 | 1.30x | |
| 98,304 | 1747.5 | 1365.2 | 1.28x | |
| 100,352 | 1763.7 | 1382.8 | 1.28x | 799.2 (13.35), 58 % of the chunk |

101,376 tokens: 132.7 s -> 99.1 s = **764 -> 1023 tok/s** (+34 %). Of the 381 ms saved at 100k, 278 ms are the experts
and 90 ms the attention.

Numerics: kernel vs fp32 reference rel L2 3.7e-4 (w13) / 5.9e-4 (w2) vs shipped 6.1e-4 / 1.06e-3 (whole-K fp32
accumulation is the reason: shipped K-split fp16 atomics 1.05e-3 vs whole-K 4.6e-4 on a full MoE layer CPU model);
layer-0 MoE output vs the shipped path 1.2e-3, gate_up 5.4e-4. End-to-end numerics: section 5.1.

### 5.1 End-to-end numerics (real checkpoint, one 1000-token text prompt, two prefills per configuration)

`--scenarios acc` captures the fp16 input of the lm_head of two identical prefills in one process (`acc_shipped`: shipped
kernels; `acc_v4`: v4 BM16 + attention f1), applies the lm_head in fp32 with the model's soft cap (30) and scores the next-token
NLL of the prompt itself; `compare_acc.py` -> `logs/acc_compare.log`.

| metric | shipped vs shipped (noise) | v4+f1 vs v4+f1 (noise) | v4+f1 vs shipped |
|---|---:|---:|---:|
| final hidden state rel L2, all rows | 8.7e-2 | 8.1e-2 | 8.7e-2 |
| per-row rel L2 p50 / p90 / p99 | 5.2e-2 / 0.12 / 0.38 | 5.1e-2 / 0.12 / 0.35 | 5.5e-2 / 0.12 / 0.38 |
| top-1 agreement over the 1000 rows | 96.5 % | 96.4 % | 97.1 % and 96.9 % |
| mean NLL (nats) | 2.3428 / 2.3447 | 2.3490 / 2.3438 | |
| per-token NLL delta: mean (std) | +0.0020 (0.180) | -0.0052 (0.186) | +0.0062 (0.171), -0.0010 (0.158) |
| last row: top-1 id, top-10 overlap, max abs logit delta | | | 13 = 13 (29.461 vs 29.458), 10/10, 0.34 |

The shipped prefill is itself not bit-reproducible (two identical prefills differ by 8.7e-2 rel L2 in the final hidden state
and flip 3.5 % of the top-1 predictions; fp16 atomics in the MoE reduce plus top-8 near-ties **[INFERENCE: cause not isolated]**).
The scratch kernels sit inside that noise: top-1 agreement with the shipped run 97.1 % / 96.9 % vs 96.5 % between two shipped
runs, mean NLL within 0.001-0.006 nats vs 0.002-0.005 between two runs of the same configuration. One prompt, not an eval.

## 6. Budget at 2.5k tok/s (T = 1024, TP=1)

2.5k tok/s = 410 ms per forward = 8.5 ms per visit (shipped 18.6 ms per visit, v4 now 12.4). Gaps are at depth 0;
the unmeasured rows are **[INFERENCE]**.

| family | shipped ms | now (v4 BM16, measured) | budget for 2.5k | what has to happen |
|---|---:|---:|---:|---|
| NVFP4 experts (w13 + w2) | 638 | 355 | 235 | int8 dot4 and a tuned dot2 kernel; fp16 dot2 alone ideal is 186 ms |
| MXFP8 experts (4 visits) | 49.5 | 47.4 | 30 | port the v4 structure (1.5x expected) |
| attn_in + o_proj | 101.4 | 97.5 | 78 | int8 dense GEMM (dot2 GEMMs are already 83-90 % of peak) |
| attention (depth 0) | 36.3 | 34.8 | 30 | tile tuning |
| msg_gate | 25.4 | 24.5 | 10 | dot2 GEMM at >= 15 TFLOP/s (G=3 calls are at 6.6-7.7 now) |
| attn_res | 11.9 | 11.9 | 8 | fuse into neighbours |
| route + zero | 12.8 | 11.5 | 5 | rewrite the tail of the route kernel |
| swiglu | 5.0 | 4.8 | 0 | fuse into the w13 epilogue |
| sconv, lm_head, kv_prep, conv_write, align | 7.6 | 7.6 | 7 | |
| **total GPU** | **886** | **596** | **~403** | |

Consequences: (a) after the measured v4 step the remaining gap to 410 ms is 186 ms; the budget above takes 120 ms from the
experts (int8 or equivalent), 20 ms from the dense int8, 17 ms from the MXFP8 port, 30 ms from glue, 5 ms from attention
(192 ms of changes for 186 ms needed). (b) There is no slack: the dot2 compute ideal of the NVFP4 experts alone is 186 ms
(45 % of the 410 ms budget). (c) At T=184 (the TP=2 chunk) the weight-stream floor alone is 1.12 ms per NVFP4 visit (573 MB
at 512 GB/s; 49 ms per forward of the 74 ms that 2.5k tok/s would allow), so 2.5k tok/s is impossible there; the per-token
cost rises 52 % at 184 tokens vs 1024 even with v4 (0.90 vs 0.589 ms/token). (d) With depth the attention term grows by
100.7 MFLOP x (depth + 512) per 1024-token chunk (full-attention visits): at 100k even 25 TFLOP/s leaves about 427 ms for
attention, so a 100k prompt cannot average 2.5k tok/s at TP=1 (about 1.2k tok/s at the end of the prompt, about 1.6k
averaged over it) **[INFERENCE from FLOP counts]**.

## 7. Findings per component

### 7.1 Dense GEMMs (attn_in, o_proj): no lever above 5 %

`bench_linear.csv`: native attn_in M=1024 24.65 TFLOP/s (rocBLAS 18.2), M=4096 24.98 (19.1), M=184 16.3; o_proj M=1024
23.83 (20.3), M=4096 24.97; in the engine 25-28 TFLOP/s at the higher clock. Native = 83-90 % of the dot2 peak.
`docs/rdna2/hi2-prefill-gemm-2026-11-15.md`: rocBLAS best 23.6, hi2 D1 +7.1 % over the best rocBLAS (about 25.3),
TunableOp +8-18 % over the rocBLAS heuristic, i.e. nothing above 5 % over the native kernel. Only a different
arithmetic (int8 dot4, 2x the dot2 issue rate) moves these: **[INFERENCE]** 78 ms budget from 97.5.

### 7.2 Attention at depth

Kernel `i2_tile_kernel<3>` (`csrc/rocm/inkling2_attn_rdna2.cu`:741-1171): 128 threads, 16 query tokens x G=3 heads = 48
rows, 64-key tiles, 6 rows per thread, 24 QK scores and 8 output dims per thread, **255 VGPR, LDS 55,552 B per CTA**.
The compiler reports occupancy 2 waves/SIMD; with 255 VGPR the register file would allow 4 (1024 VGPR per lane), so **LDS is the
limiter**: sQ 13,056 B + sK 17,408 + sV 16,384 + sP 6,912 + sRel 1,536 + s_page 256 = 55,552 B per CTA, 2 CTAs in the 128 KB
of a WGP = 72 resident CTAs (`__launch_bounds__(128, 3|4)` changes nothing, measured).
Dynamic census per key tile per thread (`scratch/isa/tile3.s`):

| phase | instructions | content |
|---|---:|---|
| A: K/V load + stage | 548 | 413 non-dot2 VALU (64-bit address math, bounds checks), 16 ds_read, 16 ds_write |
| B: QK | 2128 | 1536 dot2 + 160 ds_read_b128 + ~288 loop overhead + 88 waitcnt |
| C: rel-bias, mask, softmax, P | 838 | 192 dot2 (bias) + 24 bpermute + 30 exp + 478 other VALU |
| D: PV | 1797 | 1536 dot2 + 112 ds_read_b128 + 66 other + 76 waitcnt |

5311 instructions of which 3264 dot2 (61.5 %): issue-bound ceiling 19.7 TFLOP/s at the calibrated peak, measured 12-13.5
(61 % of it). QK needs 10 b128 per 96 dot2: LDS pipe about 83 % busy at peak VALU **[INFERENCE]**. Tail: 512 CTAs on 72
slots = 8 rounds for 7.11 -> 11 % loss at q=1024 **[INFERENCE from round timing]**.

Exact variants built and measured against the shipped op (rel err 0, base control bit-identical; `attn_a/b/c.log`):

| variant | ctx 16,384 | ctx 102,400 | sliding 1024/2048 | full 1000/1000 |
|---|---:|---:|---:|---:|
| shipped | 14.49-14.75 ms | 98.5-99.8 ms | | |
| **f1**: skip the rel-bias block for tiles whose distances are all >= rel_extent | 13.83 (-4.6 %) | 93.3 (-5.3 %) | same | same |
| f2 (+ mask skip) / f3 (+ lazy rescale) / f4 (+ bounds skip) | 14.01 / 14.11 / 14.29 | 94.6 / 95.1 / 96.7 | 0.96-0.98x | 0.95-0.97x |

(f2-f4 drop instructions but lose to control flow and code growth; f1 has 226 VGPR and no spills.)
`splits_tile` sweep on the shipped op (policy returns 1 for q=1024 because it assumes about 144 CTAs fill the GPU, but
only 72 are resident): ctx 16k 14.75 -> 14.60 ms (st 3), 65k 62.7 -> 59.3 (st 4), 102k 99.6 -> about 94.5 (st 3-6,
-5 %); it hurts sliding visits (1.10 -> 1.17-1.35 ms) and the 1000-token prefill (0.60 -> 0.68-0.83 ms), so apply it only
to window-0 chunks with >= 32768 keys (`vllm/_custom_ops.py:1894-1922`). The split-K workspace of
`glue.py` (`_ATTN_WS_ROWS_NUM = 4975`, built for splits only below 144 CTAs) then has to grow: splits_tile 3 x 1024
tokens needs (32 + 3072) x 24 heads x 130 fp32 = 39 MB per call instead of 8 MB (the first depth run died on this
check: "workspace must be fp32 with at least ...").
**Measured together (e2e, depth 100k): attention 889 -> 799 ms (-10 %, 12.0 -> 13.35 TFLOP/s).**

**Occupancy probe: the LDS limit is a first-order bottleneck.** The kernel was rebuilt with only its LDS allocation changed
(`make_attn_variants.py` variants `occ1`, `occ3p`, `occ4p`) and run against the shipped op on HIP 3 (`attn_occ.log`,
`attn_occ.json`, q = 1024, full attention, time per call):

| variant | LDS per CTA | CTAs per WGP | ctx 16,384 | ctx 65,536 | ctx 102,400 |
|---|---:|---:|---:|---:|---:|
| shipped | 55.5 KB | 2 | 14,665 us (13.6 TFLOP/s) | 62,490 (13.1) | 99,167 (12.9) |
| occ1: +8 KB of live LDS, exact (control) | 63.5 KB | 2 | 14,511 (x1.01) | 62,657 (x1.00) | 99,232 (x1.00) |
| occ3p: sK cut to 1/4, **timing probe, wrong results** | 42.5 KB | 3 | 12,665 (x1.16) | 54,540 (x1.15) | 86,323 (x1.15) |
| occ4p: sK 1/16 and sV 1/4, **timing probe, wrong results** | 26.9 KB | 4 | 11,524 (x1.27) | 48,664 (x1.28) | 77,506 (x1.28) |
| f1 (exact) | 55.5 KB | 2 | 13,840 (x1.06) | 59,533 (x1.05) | 94,416 (x1.05) |

The probes keep the instruction stream and change only the number of resident CTAs; their results are garbage (rel err 0.8-1.0),
so they are an upper bound for an LDS diet that costs no extra instructions (garbage data may also lower power and raise the clock
a little **[INFERENCE]**). Occupancy is therefore a first-order limiter: +15 % at 3 CTAs per WGP, +28 % at 4. The obvious diet, 32-key
tiles (35 KB, 3 CTAs), doubles the per-key barrier / loop / softmax-reduction overhead: about +600 instructions per 64 keys (+11 %)
from the census, which eats most of the 15 % **[INFERENCE, not built]**; the target is a layout that frees 13 KB (3 CTAs) or 23 KB
(4 CTAs) of the 55.5 KB without adding instructions (section 9.2).

### 7.3 Glue (7 % at depth 0, 62 ms)

| kernel | ms | per call | rate | note |
|---|---:|---:|---|---|
| gate_mix<float,8,true> | 13.6 | 276 us | 6.6 TFLOP/s (fp32 FMA, 41 % of the fp32 peak; 1.81 GFLOP per G=3 call at T=1024) | fp32 weights (the checkpoint's gate weights are not all fp16-exact); the fp16/dot2 variant exists |
| gate_down<4> | 11.7 | 236 us | 7.7 TFLOP/s (fp16 dot2, 25 % of the dot2 peak; 2 x T x G x R x 2H with R = 64) | |
| route (moe_route_kernel<float,4>) | 12.8 | 267 us | | fp32-weight router GEMV (FMA, 0.6 GFLOP) + parallel counting sort/align + zero fills; the glue doc measured 152 us for the fp16 (dot2) router vs 223 us fp32 at T=1024 |
| attn_res<1,3,*> | 11.9 | 86-97 us | | |
| swiglu | 5.0 | 103 us | 570 GB/s | memory-bound, fusable into the w13 epilogue |

The `zero1=m.gu` fill (36.9 MB per visit) is not the cost of the route kernel: skipping it for a kernel that writes every
row saves only 0.9 ms per forward (route 12.35 -> 11.46 ms, `prefill_v4_depth`). Estimates **[INFERENCE]**: msg_gate
-10 to -14 ms at the dense rate, route -5 to -7, swiglu fusion -5, attn_res -3: about 25-30 ms (4-5 % at depth 0).

### 7.4 MXFP8 layers (4 visits, 47 ms = 8 %)

Same structure as the NVFP4 kernel (BM 8, split-K, 10.7 TFLOP/s). Not ported: the v4 structure should give about 1.5x
**[INFERENCE]**, -15 to -20 ms (about 3 % at depth 0).

## 8. Integration recipe for the expert kernel (patch list)

No weight repack and no decode-kernel change are needed with CH=2 (the repacked CH=1 layout is only 1.5-4.5 % faster
and would need a second 11.5 GB copy of the NVFP4 experts for the decode kernels; in-place repack of the 20 layers took
4.0 s with no extra memory but breaks the decode path, so it is not recommended).

Stage 1 (BM8, +37 % e2e, no route / workspace change):

1. `csrc/rocm/moe_lowp_dot2_rdna2.cu`: add the v4 kernel (`moe_v4.cu`, CH=2, BM 8 and 16, whole-K with 288-K LDS chunks,
   192 threads x 4 columns = 768-column tile, N = 2304 = 3 tiles) and dispatch it from the host function
   `moe_nvfp4_gemm_rdna2` for `block_size_m` 8 and 16 (BM 1/2/4 stay on the old kernel; the host switch at about
   lines 281-298 rejects anything but 1/2/4/8; `num_token_blocks = sorted_ids.size(0) / block_size_m` at line 339 is
   BM-generic). Write the gate/up scratch with plain stores (every row is written) and only the w2 top-k reduce with CAS
   fp16 atomics, so `zero1` is no longer needed.
2. `vllm/models/inkling2_loop/amd/moe.py`: `_BLOCK_M_BY_TOKENS["nvfp4"]` (tokens per MoE call -> BM) is
   `((4,1),(12,2),(64,4))` with `_BLOCK_M_WIDE = 8`; with the new kernel use `((4,1),(12,2),(48,4),(192,8))` and
   `_BLOCK_M_WIDE = 16` (section 4: v4 BM8 beats the shipped BM4 from T=64, BM8 wins below about 192 tokens, BM16 from
   192-256 on). Re-measure the table on the real route before committing.
3. Tests: `tests/models/test_inkling2_loop_moe.py::test_every_block_size_m` and
   `tests/models/inkling2_loop/test_moe.py::test_kernel_block_size_m` parametrize `[1, 2, 4, 8]`: add 16.

Stage 2 (BM16, another 7.5 %):
4. `csrc/rocm/inkling2_glue_rdna2.cu` lines 2024-2026: `TORCH_CHECK(block_size_m == 1 || 2 || 4 || 8, ...)`: accept 16
   (the align code of the route kernel is BM-generic: `nb = ceil(cnt/BM)`, `post_pad = total*BM`; `kAlignSmemBytes` and the
   `tb` template are independent of BM). The invalid-size test (`test_inkling2_glue_rdna2.py:980`) uses 3 and stays valid.
5. `vllm/models/inkling2_loop/amd/glue.py:169`: `ops.inkling2_glue_moe_route_sizes(chunk, E, K, 8)` sizes
   `sorted_ids`/`expert_ids` for BM8 (9088 ints); size for 16 (10112 ints, 632 blocks; the helper in
   `vllm/_custom_ops.py:2776` is pure Python and BM-generic); `ws.moe_views` follow.
6. MoE call chunk stays <= 1024 tokens (`kRouteMaxEntries = 8192`, `glue.py` `MAX_MOE_TOKENS`). Larger MoE chunks need a
   route-kernel rewrite and are worth 5-14 % of the expert time at kernel level (w13 uniform BM16: 5497 us at T=1024 vs
   18919 / 4 = 4730 us per 1024 tokens at T=4096; w2 2650 -> 10070 / 4 = 2518 us per 1024 tokens).

Attention (independent): ship f1 in `inkling2_attn_rdna2.cu` (guard the rel-bias block by "all key distances of the tile
>= rel_extent", exact), change the `ops.inkling2_attention_splits` policy for `window == 0 and max_query_len >= 512 and
max_seq_len >= 32768` to `splits_tile = 3`, and size the split-K workspace in `glue.py` (`_ATTN_WS_ROWS_NUM`) for it (+31 MB).

Load cost: none. Memory: none for the experts (LDS 4.7-9.5 KB per block), +31 MB attention workspace.
TP > 1: the v4 tile assumes N = 2304 (3 x 768); w13 at TP=2/4 has N = 1152 / 576, so the kernel needs a tile-width variant
(576 = gcd of 576/1152/2304: 144 threads x 4 columns, or 192-column tiles with an in-block K split) **before the TP=2/4
numbers can follow**.

## 9. Kernel designs

### 9.1 Expert GEMM (lever 1): as built (`moe_v4.cu`) and what is next

Built (fp16 dot2): grid = (expert-aligned token blocks of BM rows, N/768 = 3, 1); 192 threads x 4 adjacent columns,
accumulators acc[BM][4] fp32 (64 registers at BM16), whole K per block in 288-K LDS slabs ([BM][288] fp16 = 9.5 KB at
BM16, 4.7 KB at BM8, 19 KB at BM32), 32-K inner step. Weights go from global straight to registers: per 32-K step four
coalesced 16-byte loads (`int4` = the u32 words of 4 adjacent columns for one 8-K row; layout [expert][K/8][N]) and two
u32 scale loads (4 columns of e4m3 per 16-K row), no LDS for B; each word is dequantized **once per block** (2.4 VALU ops
per weight: shift/mask extract of the (p, p+4) nibble pair into a half2 plus one `v_pk_mul` with the e4m3 block scale),
then 16 `v_dot2_f32_f16` against the A rows (ds_read_b128 broadcast, chained accumulation) reuse it. The A
slab is permuted at staging with `v_perm` so the (p, p+4) pairing needs no weight repack. No K-split, so the gate/up
output is written once with plain stores; the w2 top-k reduce uses CAS fp16 atomics. Register budget gfx1030 wave32
(1024 VGPR per lane): BM8 123 VGPR / 8 waves per SIMD, BM16 187 / 5, BM32 192 / 5, no spills.

Remaining headroom of the fp16 dot2 kernel: VALU utilization 73 % -> about 90 % (waitcnt/latency; 5 waves) and fewer
non-dot2 slots (410 -> about 300: precompute fp16 scales, LUT/`v_perm` dequant) = about 18-20 TFLOP/s **[INFERENCE]**,
which is 273 ms for the 44 visits (now 355).

Next (owner `Inkling2PrefillInt8`): `v_dot4_i32_i8` at 2x the dot2 issue rate; dequant falls from 2.4 to about 0.5 op per
weight but each (row, column, 16-K group) needs an int32 -> scale flush: per 32-K step 512 dot4 + about 100 dequant +
256-384 flush = about 1000-1100 slots vs 1434 now = **1.3-1.43x over v4 [INFERENCE from the calibration, 41.6-47.5
TFLOP/s-eq vs 31.2 for dot2 with fp32/integer-mad flush]**. The CPU error simulation of the MoE layer (synthetic
activations; `int8_err_l6.log`): rel L2 shipped K-split fp16 1.05e-3, fp16 whole-K 4.6e-4, int8 per (row, 16-group)
1.19e-2, per 32-group 1.57e-2, per-row 6.6e-2: int8 needs 16-groups and a real-activation check.

### 9.2 Attention tile (lever 2): tuned vs redesigned **[INFERENCE, not prototyped]**

Tuned current structure (target 16-18 TFLOP/s at depth): (1) LDS diet below 42.6 KB (3 CTAs per WGP, measured ceiling x1.15) or
below 32 KB (4 CTAs, x1.28; 255 VGPR still allow 4 waves/SIMD). The 55.5 KB are sQ 13.1 + sK 17.4 + sV 16.4 + sP 6.9 + sRel 1.5

+ pages 0.3. Candidates **[INFERENCE, none built]**: (a) 32-key tiles, 35 KB, +11 % instructions; (b) K and V sharing one 17.4 KB
buffer with V staged after QK, which needs about 32 VGPR of V prefetch that the 255-VGPR kernel does not have (226 with f1);
(c) K staged in two 64-dim passes (8.7 KB buffer, one more barrier per tile); (d) CTAs of 8 query tokens (sQ 6.5 + sP 3.5 KB, 45 KB)
at half the K/V reuse per LDS read. Any of them also cuts the tail loss (512 CTAs on 108 slots = 5 rounds, 5 % instead of 11 %).
(2) phase A 548 -> about 150 instructions: one page per 64-key tile for block size >= 64 (uniform base in SGPRs, 32-bit offsets
through `buffer_load`), `v_perm_b32` for the V pair interleave; (3) keep f1 (the bias block only for tiles inside `rel_extent`);
(4) full unroll only where it did not lose (f5 lost on sliding). Non-dot2 slots 2047 -> about 1100 gives dot2 density 75 %,
bound about 23 TFLOP/s, 17 at 75 % issue efficiency.
Redesign for 20 TFLOP/s (new structure): a larger register tile for QK (6 rows x 8 keys instead of 6 x 4: dot2 per LDS read
9.6 -> 13.7), K/V double buffering with one barrier per tile instead of three, persistent CTAs over (q-tile, kv-head) with the
split count chosen from the 72/108 resident slots instead of the 144-CTA rule, `v_exp_f32` on pre-scaled scores. The K|V per
row `[bs][2 x 128]` layout stays (V needs the key-pair interleave).

Attention bounds (harness FLOP count, 10.67 TFLOP per 1024-chunk at 100k): 12.0 (shipped) / 13.35 (f1 + splits) / 17 / 20 /
25 TFLOP/s = 889 / 799 / 628 / 533 / 427 ms per chunk; per 942k-token prompt at TP=4 (10.9 PFLOP per rank) 826 s measured
(13.2 effective), 641 s at 17, 545 s at 20, 436 s at 25.

## 10. Chunk size, TP=2 and the 1M-token case

Measured: TP=2 chunk 184, 942k tokens in 2492.9 s = 378 tok/s; TP=4 mnbt 4096: 744 tok/s (1267 s); TP=1 162k prompt 648 tok/s.
Attention FLOPs for N tokens = 98,304 x N^2/2: 43.6 PFLOP at N = 942k, i.e. 10.9 PFLOP per rank at TP=4 and 21.8 at TP=2, so the
measured attention times (TP=4: 826 s of 1267 s; TP=2: 1651 s of 2493 s; **65 %**) are an effective 13.2 TFLOP/s per rank.

| attention rate | TP=4 attention s | TP=4 total s (tok/s) | TP=2 attention s | TP=2 total s (tok/s) |
|---|---:|---:|---:|---:|
| measured shipped (13.2 TFLOP/s effective) | 826 | 1267 (744) | 1651 | 2493 (378) |
| f1 + splits (-10 %, 14.7 effective) | 743 | 1184 (796) | 1486 | 2328 (405) |
| 17 TFLOP/s | 641 | 1082 (870) | 1282 | 2124 (444) |
| 20 TFLOP/s | 545 | 986 (955) | 1090 | 1932 (488) |
| 25 TFLOP/s | 436 | 877 (1074) | 872 | 1714 (550) |

Non-attention time (441 s at TP=4, 842 s at TP=2) is held constant in this table; a v4-class expert kernel at TP>1 needs the
tile-width variant of section 8 and would reduce it **[INFERENCE: unmeasured at TP>1]**.

Why the TP=2 chunk is 96..184 tokens: it is memory-limited, not kernel-limited (`docs/rdna2/inkling2-tensor-parallel.md`
lines ~290-353). The KV margin is a fixed 3 GiB (`KV_CACHE_MARGIN_FLOOR_BYTES`), per-rank weights 9.26 GiB with the
sharded embedding, full-attention KV 16 KiB/token per rank at TP=2 (32 KiB at TP=1); the per-chunk transient (logits
[chunk, 200058] fp32 = 0.76 GiB per 1024 tokens at TP=1, glue workspace 0.28 GiB per 1024, conv window) must fit in
what is left: chunk 256 needs 16.49 GiB and holds 15.98 (refuses), 192 refuses, 128 refuses by 0.06 GiB, 112 by 0.01,
96 and 64 boot; the newer note says 184 boots (text only). TP=4 fits any chunk (4096 used); TP=1's longest request is
179,136 tokens (chunk 1024). Cost of the small chunk: v4 shows 0.90 ms/token at 184 vs 0.589 at 1024 (+52 %; shipped
1.17 vs 0.88, +33 %).

## 11. Qwen / Flash-Next W4A8 v2 MoE comparison

Same skeleton as the Inkling kernel (256 threads x 4 columns, grid (blocks, N/1024, K/k_block)). Differences: int8
`sdot4` on masked nibbles with the activations pre-quantized once per call; LDS A interleaved [chunk][row]; BM in
{4, 8, 16}; `k_block` up to the whole K (BM x K <= 32 KiB LDS); plain stores when there is a single K block and no top-k
reduce; 121 VGPR at tile 8. Their tile 4 -> 8 with K = -1 gave 1.49-1.54x at 2048/4096 tokens (knobs
`VLLM_RDNA2_MOE_BSM_LARGE=8` from 512 tokens, `TILE8_MIN_TOKENS=4096`). The scratch v4 kernel converged on the same
structure (whole K, no K-split, plain stores for gate/up, BM up to 16) but stays on fp16 dot2 and the original
accuracy; BM16 is the sweet spot, BM32 is no better (padding x1.24 and 192 VGPR), and the int8 route is the way to the
2x dot2 issue rate.

## 12. Ranked levers

**Path to 2.5k tok/s (TP=1, short and medium prompts), in order:**

| # | lever | evidence | e2e gain | risk |
|---:|---|---|---|---|
| 1 | NVFP4 expert kernel, whole-K chained dot2, BM8/16, no repack (stage 1 BM8, stage 2 BM16) | measured: 1.6-2.0x per launch, p1000 880 -> 597 ms | **1136 -> 1675 tok/s (+47 %)**; stage 1 alone 1557 | low (accuracy better than shipped) |
| 2 | attention f1 + `splits_tile=3` at depth | measured at 100k: 889 -> 799 ms | +5-6 % at 100k, +0 at depth 0; 764 -> 1023 tok/s average with lever 1 | low (exact; +31 MB workspace) |
| 3 | int8 dot4 experts (and tuned dot2) | calibration 1.3-1.5x issue; CPU error sim 1.2e-2 | -90..-130 ms at T=1024 (about +15 % tok/s) [INFERENCE] | accuracy on real activations unproven |
| 4 | MXFP8 layers on the v4 structure | same kernel family, 8 % of the forward | -15..-20 ms (+3 %) [INFERENCE] | low |
| 5 | glue: msg_gate dot2 GEMM, route tail, swiglu fusion, attn_res | per-kernel rates, section 7.3 | -25..-30 ms (+5 %) [INFERENCE] | medium (several kernels) |
| 6 | dense int8 for attn_in/o_proj | native GEMMs at 83-90 % of dot2 peak | -20 ms (+4 %) [INFERENCE] | activation quant accuracy |
| 7 | MoE call chunk 4096 (route rewrite) | kernel level -5..-14 % per token | -15..-30 ms [INFERENCE] | route kernel rewrite |

**For long prompts and 1M tokens (attention-bound):**

| # | lever | evidence | gain |
|---:|---|---|---|
| A | attention f1 + `splits_tile` | measured -10 % at >= 64k | 1M TP=4: 1267 -> 1184 s (+7 %) |
| B | attention LDS diet to 3-4 CTAs per WGP, phase-A trim, f1 | timing probes x1.15 (3 CTAs) / x1.28 (4 CTAs) with the same instructions; 5311 instr/tile, 61.5 % dot2 | 3 CTAs: TP=4 1087 s (+17 %); 4 CTAs: 1021 s (+24 %) [INFERENCE: stacks with A] |
| C | attention redesign | bound 23 TFLOP/s at 100 % issue | 20 TFLOP/s: TP=4 986 s (+28 %), TP=2 1932 s [INFERENCE] |
| D | expert tile-width variants for TP=2/4 shapes, then lever 1 | N = 1152 / 576 | non-attention 35 % of the time [INFERENCE] |
| E | TP=2 chunk (memory) | 96..184 tokens | per-token cost +33..52 % vs 1024 |

Not levers: dense GEMM tuning (TunableOp/hi2/rocBLAS: nothing above 5 %), chunk size at TP=1 (flat), the
`zero1` fill (0.9 ms), BM32, prefetch, the int8 per-row scheme (6.6e-2 error).

## 13. Files, logs, raw data

Scratch tooling (not in the repo): local `/Users/kletorch/Projects/infrastructure/gfx1030_optimized/Inkling2PrefillProfile/`
and the box `$P/scripts/`: `profile_prefill.py` (harness; `--moe-plugin file.py --plugin-arg k=v`, `--attn-variant`,
`--attn-split-tile`, scenarios `capture,p1000,chunks,depth,acc`), `moe_plugin_v4.py`, `moe_v4.cu`, `moe_v4_bench.py`,
`moe_variants_bench.py`, `make_attn_variants.py`, `attn_variants_bench.py`, `build_attn.py`, `build_isa.py`,
`compare_depth.py`, `compare_acc.py`, `suite0.py`, `calib_dot.py`, `int8_error_cpu.py`. Plug-in API: `BM`, `FORMATS`,
`setup(args)`, optional `repack(m, stage)` and `ZERO_GU`, and `gemm(m, stage, a, c, topk_weights, sorted_ids,
expert_ids, ntpp, top_k, bm, mul_topk_weight, output_topk)`, the contract of `Inkling2LoopMoE._gemm`.
Logs (`$P/logs/`, copies in local `data/`): `suite0.log`, `prefill_a.log`, `prefill_depth.log`, `prefill_v4.log`,
`prefill_v4_ch2_bm8.log`, `prefill_v4_ch2_bm16.log`, `prefill_v4_depth.log`, `v4_rows.log`, `v4_a.log`, `v4_b.log`,
`variants_a.log`, `calib_dot.log`, `int8_err_l6.log`, `attn_a.log`, `attn_b.log`, `attn_c.log`, `attn_tile_sweep.log`,
`attn_isa_build.log`, `moe_v4_isa.log`, `acc_shipped.log`, `acc_v4.log`. Outputs (`$P/out/`): `prefill_a/results.json` +
`moe_capture.pt` (real routing), `prefill_depth/`, `prefill_v4*/`, `v4_rows.json`, `moe_rows_base.json`,
`moe_rows_skew2.json`, `bench_linear.csv`, `bench_attn.json`, `attn_*.json`, `acc_*/acc.pt`.
ISA: `$P/scratch/isa/{isa_base.s, tile3.s, v4_bm16_ch2.s, census.awk, census2.awk}`.

## 14. Open risks

1. Engine numbers with the plug-ins use the harness re-sort (`moe_align_block_size`) instead of the native route kernel;
   the native BM16 route is not built (stage 2 step 4).
2. Clock and power variance (1.58-2.1 GHz, 175 W cap): comparisons are same-job, but absolute TFLOP/s moves by +-4 %.
3. Real-weight power may differ from the random-weight microbenchmarks.
4. The int8 accuracy is unproven on real activations; the CPU simulation used synthetic ones.
5. The attention redesign and the MXFP8 / TP>1 kernel variants are not prototyped.
6. The route kernel limits a MoE call to 1024 tokens (`kRouteMaxEntries = 8192`).
7. Only the 20 NVFP4 layers run the new kernel in the harness; the engine's graph-capture path (decode) was not exercised, and
   the plug-in harness runs prefill only (repacking plug-ins break decode). `ZERO_GU = False` was exercised only in the depth run.
8. The TP=2/4 projections hold the non-attention time constant; they are arithmetic, not measurements, and the v4 tile does
   not fit the TP>1 expert shapes yet.
9. End-to-end accuracy was checked on one 1000-token prompt only and the baseline itself is noisy (section 5.1); a real eval
   (perplexity on a longer text, long-context retrieval) is still needed before shipping. The `splits_tile=3` path changes the
   split-K summation order (rel diff 3e-4 vs `splits_tile=1` at the kernel level) and was only exercised in the depth run.
