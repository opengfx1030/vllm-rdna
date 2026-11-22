# Inkling-2-Mini b=1 decode: glue launches and speculation cost (V620, gfx1030, TP=1)

Measured on one V620 (HIP 4, TP=1, CUDA graphs on) at head `e937e1abb0`
(`feat/inkling2-merge`).  Everything here is a measurement unless it is
labelled `[INFERENCE]`.  The levers are behind one flag,
`VLLM_INKLING2_FUSE_GLUE` (default **on**: the SwiGLU fusion is bit-identical
to the two-launch path, the gate-mix split only reorders an fp32 sum, and the
real-model NLL is inside the run-to-run floor, sections 4 and 5; `=0` restores
the shipped kernels and tables unchanged).

* Kernels: `csrc/rocm/inkling2_glue_rdna2.cu` (`gate_mix_split_*`),
  `csrc/rocm/moe_lowp_dot2_rdna2.cu` (`swiglu_act`, fused w2 activation load),
  ops `inkling2_glue_msg_gate_mix` (dispatch), `moe_nvfp4_gemm_swiglu_rdna2`,
  `moe_mxfp8_gemm_swiglu_rdna2`
* Python: `amd/glue.py` (`FUSE_GLUE`), `amd/moe.py` (`_gemm_swiglu`,
  `_BLOCK_M_BY_TOKENS_SPEC`)
* Tests: `tests/kernels/core/test_inkling2_glue_rdna2.py`
  (`*_f32w*` gate / route cases), `tests/kernels/core/test_inkling2_moe_rdna2.py`
  (`test_moe_swiglu_gemm_*`)
* JIT harness (no rebuild): `tools/rdna/inkling2/glue_jit.py`, `moe_jit.py`

## 1. Result in one table

| step (b=1, TP=1, graphs, 24 prompts x 128 tokens, rebuilt `_rocm_C`) | `=0` (shipped) | `=1` (default) | delta |
|---|---|---|---|
| graph replay, min of 200 (ms) | 18.425 | 17.675 | -0.749 |
| e2e ms/token (tok/s) | 19.30 (51.8) | 18.72 (53.4) | -0.58 (-3.0%) |
| DSpark K=2 e2e ms/token (tok/s) | 15.72 (63.6) | 15.50 (64.5) | -0.22 |

(Same session; replay-time noise is ~0.1 ms.  An earlier run with JIT-built
ops gave 19.29 -> 18.78.)  The glue is a 3% lever, not the road to 10
ms/token: the step is 76% weight-streaming kernels at DRAM speed, see
section 2.

## 2. Anatomy of the 782-kernel step

Graph profile (`profile_decode.py --batch 1`, 8 decode steps, per step,
`out/prof_graph_b1.log`): 782.75 kernels, GPU busy 18.48 ms, step wall 20.1 ms.

| family | launches | GPU ms | us / launch |
|---|---|---|---|
| `inkling2_linear::gemv_kernel` (attn input pack, o_proj, lm_head) | 97 | 8.25 | 85.1 |
| NVFP4 expert GEMMs (w13 + w2) | 88 | 4.78 | 54.3 |
| MXFP8 expert GEMMs | 8 | 0.65 | 81.8 |
| `gate_mix_kernel` (message gate mix + norm) | 48 | 0.76 | 15.9 |
| `moe_route_kernel` (router GEMV + top-k + align) | 48 | 0.75 | 15.7 |
| `i2_decode_kernel` (paged decode attention) | 48 | 0.43 | 8.9 |
| `i2_sconv_kernel` | 96 | 0.34 | 3.5 |
| `attn_res_kernel` (5 variants) | 120 | 0.90 | 5.8 - 15.7 |
| `gate_down_kernel` | 48 | 0.29 | 6.1 |
| `i2_kv_prep_kernel` | 48 | 0.27 | 6.7 |
| `swiglu_kernel` | 48 | 0.22 | 4.6 |
| `i2_combine_kernel` | 48 | 0.15 | 3.1 |
| copy_states / rms_norm / runner (sampler, metadata, ~20) | ~35 | ~0.2 | |

Per visit: 13 kernels on a no-message visit, 18 on a message visit.  The
glue, attention and small families together are ~4.1 ms in 535 launches
(23% of the step); GEMV + MoE are 13.7 ms.

**The 9 extra graph launches** (eager 774 vs graph 783 kernels per step, same
profile in `--cudagraph-mode eager`, `out/prof_eager_b1.log`): 8 of them are
`i2_combine_kernel` on the 4 full-attention layers x 2 loop visits (eager
skips the split-K combine at the short context of the profile, the graph is
captured with the maximal split count and always runs it; 3.1 us each,
0.025 ms in total) and 1 is a runner `hipMemcpyAsync` (`copyBuffer`).  Nothing
else differs.

### 2.1 What a launch costs

| measurement | us |
|---|---|
| dependent chain of trivial kernels inside a graph | 3.05 per kernel (3.07 eager) |
| one trivial 1-element kernel added after each of the 48 swiglu launches of the real step | **1.16 per kernel** (+0.0555 ms) |
| `dup` of a real glue launch inside the graph (kernel runs twice; hot cache, so a lower bound of its marginal cost) | swiglu 3.75, kv_prep 3.9, sconv 4.1, attention (decode + combine) 4.4, attnres_raw 4.7, attnres_norm 5.4, down 5.4, attnres_msg 4.4, route 11.6, mix 9.2 |

Removing one in-situ launch from the step is worth ~1.2 us of launch tax
plus that kernel's own latency chain, i.e. ~3.5 - 5 us for the small ones; the
788-launch floor is ~0.9 ms (`1.16 us x 788`), not the 2.4 ms a back-to-back
chain suggests.  The skip ablation (kernel removed, upper bound of what fusing
it away can give): swiglu -0.355, sconv -0.511, kvprep -0.330, attention
-0.95, gates -1.46, attn_res -2.69, copy ~0 ms (`out/abl_a1_*.log`).

## 3. What was implemented

### 3.1 SwiGLU folded into the w2 expert GEMM (`moe_{nvfp4,mxfp8}_gemm_swiglu_rdna2`)

The router GEMM `w13` writes `gate_up` as `[gate | up]` rows of 2*K fp16.
`inkling2_glue_swiglu` read that, wrote the activated `[rows, K]` fp16 tensor
and the w2 GEMM read it again.  `load_activations` of the lowp-dot2 GEMMs now
takes `(gate_cap, up_cap)`: with `gate_cap > 0` it reads `gate` and `up` of the
row, applies the identical softcapped SwiGLU in fp32 and rounds to fp16 exactly
as the standalone kernel did, so the result is **bit for bit** the old
swiglu-then-GEMM result (test `test_moe_swiglu_gemm_is_bit_identical_to_swiglu_then_gemm`,
bm 1/2/4/8, both formats).  One launch and one activation round trip fewer
per visit; the plain ops are unchanged (caps 0).

Lab (graph-timed, cold rotating weights, `out/moe_lab.log`), us per w2 launch,
unfused swiglu + GEMM -> fused:

| rows (bm=1) | NVFP4 w2 | saved |
|---|---|---|
| 1 | 46.36 -> 42.29 | 4.08 |
| 2 | | 2.99 |
| 3 | | 2.42 |
| 4 | | 1.84 |

MXFP8 saves 4.3 - 5.2 us.  The fused GEMM costs +0.3 us over the plain GEMM.

The fused load recomputes the activation once per N tile (3 tiles of 1024
columns), which is free while a launch is latency bound and costs ALU time once
the GEMM has rows to chew on.  Same lab at prefill-sized chunks (default block
widths, cold rotating weights, us per launch, `out/moe_lab_big.log`):

| tokens (bm) | NVFP4 swiglu + w2 | NVFP4 fused | | MXFP8 swiglu + w2 | MXFP8 fused |
|---|---|---|---|---|---|
| 13 (4 / 2) | 196.7 | 195.9 | | 325.6 | 319.4 |
| 64 / 48 (4 / 2) | 581.3 | 587.5 (+1.1%) | | 501.2 | 500.7 |
| 256 (8) | 1454.3 | 1534.6 (+5.5%) | | 1330.0 | 1392.0 (+4.7%) |
| 1000 (8) | 4955.1 | 5308.5 (+7.1%) | | 4185.2 | 4478.3 (+7.0%) |

so `amd/moe.py` fuses only chunks of at most `_FUSE_SWIGLU_MAX_TOKENS = 16`
tokens (decode, DSpark verify, small batches) and keeps the separate swiglu
kernel above that.

### 3.2 8-wave split of the message-gate mix (`gate_mix_split_kernel`)

`inkling2_glue_msg_gate_mix` ran R=64 rank terms of the fp32 gate weights in
one wave per token column and was DRAM-latency bound.  For `T <= 4`, `R == 64`
and fp32 weights the CTA now splits R over 8 waves (8 ranks each), sums the
partials in wave order through LDS (a fixed order, deterministic) and the last
arriving CTA normalises (split over tokens / columns).  The only numerical
change is the fp32 summation order of the rank sum.

Lab (graph, cold weights, us per launch, old -> new, `out/glue_lab.log`):

| kernel | old | new |
|---|---|---|
| gate_mix G=3, T=1 | 15.92 | 13.48 |
| gate_mix G=1, T=1 | 12.29 | 7.72 |
| gate_mix G=3, T=3 | 17.79 | 15.56 |
| gate_mix G=1, T=3 | 12.32 | 8.99 |

(48 launches per step; qkv gates G=3, mlp gates G=1.)  The kernel is bound by
the 3.5 MB of fp32 weights (G=3); at DRAM speed it would be ~7 us, so ~6 us
per launch is still latency.

### 3.3 Verify rows: two-row expert blocks (`_BLOCK_M_BY_TOKENS_SPEC`)

The production table keeps one routed row per expert block for T <= 4, tuned
with uniformly random routing (`moe_rows_bench.py`).  The real routing of
neighbouring tokens is correlated: adjacent rows share 3.54 of their 8 experts
(`out/route_stats_k4.log`, 142 verify steps x 48 visits), so the number of
distinct experts for 1 - 5 rows is 8.0 / 12.3 / 16.1 / 19.3 / 22.1 (uniform
routing would give 8 / 15.5 / 22.5 / 29.1 / 35.3), and a block of 2 rows
streams a shared expert once.  NVFP4 w13 + w2 per layer visit on real expert
sets (cold, 4 rotating weight sets, `out/moe_rows_real2.log`):

| rows | distinct experts | bm=1 (current) | bm=2 | bm=4 |
|---|---|---|---|---|
| 1 | 8.0 | **119.2** | 123.8 | 128.3 |
| 2 | 12.4 | 181.3 | **174.1** | 187.3 |
| 3 | 16.3 | 238.2 | **225.2** | 236.5 |
| 4 | 19.7 | 302.1 | **268.4** | 274.0 |
| 5 | 22.3 | 362.1 | 312.6 | **309.7** |

(us per visit; bold = fastest.)  With the flag on 2 - 12 rows use bm=2 (a
single row keeps bm=1): -13 us/visit at 3 rows (K=2 speculation) = 0.57 ms per
step over 44 NVFP4 visits, -34 us (1.5 ms) at 4 rows.  The kernels run at only
270 - 326 GB/s on these sets (53 - 64% of the 512 GB/s peak): the expert
GEMMs are the biggest remaining lever in a verify step, see section 6.

### 3.4 NVFP4 decode expert GEMM: VALU-bound loop, exact cheaper dequant

The 8-expert launches (`w13` 71.6 us, `w2` 37.0 us) read their weights at only
~330 GB/s of the 512 GB/s peak.  Diagnostics on real expert sets (cold
rotating weights, graph-timed, `w13 + w2` per visit, us;
`moe_rows_var.py`):

| variant (diagnostic; output is wrong unless noted) | 1 row | 3 rows |
|---|---|---|
| shipped loop | 118.9 | 244 |
| CAS atomic add replaced by a plain store | 118.1 | - |
| dequant + dot2 replaced by an xor of the loaded words (memory only) | 98.9 | 199.0 |
| depth-1 software-pipelined weight loads (exact) | -2% | -6% |

The atomics are not the limit and pipelining buys 2 - 6%; the loop is
ALU-bound.  The ISA has ~990 VALU instructions per thread and iteration, ~48
per packed word for the E2M1 dequant (704 of them) and ~100 for the scale
decode, against 64 `v_dot2`; `[INFERENCE]` ~44 us of VALU time in each 74 us
`w13` launch.

`dequant_e2m1_8_fp16_fast` (`moe_lowp_dot2_rdna2.cu`) spreads the sign and 3
magnitude bits of 4 nibbles into one byte each and places those bytes with
`v_perm_b32` at the high byte of each fp16 value x 2^-14: 19 VALU ops per
word instead of ~48; the multiply by the scale pair is unchanged.  The scale
bytes are loaded as one dword per 4 columns and `e4m3_scale_to_f32` decodes a
normal e4m3 byte by moving its exponent and mantissa bits up (bias +120; the
subnormals below 8 take the converted integer): its float bits equal the
shipped decode for all 256 bytes.  The fp16 scale keeps the shipped arithmetic
in the shipped operation order, `RNE16((e * global) * 2^14)`.  The first
version of this patch hoisted `global * 16384` (the fp32 product is the same
bit for bit), but the compiler then fuses the multiply into the fp16 convert
(one rounding instead of two) and the scale changed by one fp16 ulp for
~6e-5 of the (byte, global) pairs.  Comparing the production kernels of the
two builds (K = 256: one CAS add per output, deterministic) found it: 0 of
329,472 outputs differ on random data, 168,630 of 329,472 on data forced onto
tie-prone pairs.  With the shipped order restored the kernel is bit-identical
to the shipped one on both (0 of 658,944 outputs).  Dequant exact (2^32
words), scale rounding identical to shipped.

| rows (bm) | shipped us/visit | fast us/visit | delta |
|---|---|---|---|
| 1 (1) | 119.3 | 104.6 | -12.3% |
| 3 (1) | 246.8 | 208.0 | -15.7% |
| 5 (1) | 377.0 | 288.6 | -23.4% |
| 1 (2) | 124.5 | 106.5 | -14.5% |
| 3 (2) | 233.8 | 208.4 | -10.9% |
| 5 (2) | 335.9 | 288.5 | -14.1% |

(300 -> 342 GB/s at 1 row, 311 -> 369 at 3 rows, 292 -> 381 at 5 rows;
`out/moe_fast_rows.log`.)  End to end with the rebuilt ops (b=1, graphs,
`out/e2e_e3_{off,on}.log`) the microbench projection of 44 NVFP4 visits x
14.7 us = 0.65 ms came out as 0.648 ms of graph replay:

| config | graph replay min (ms) | e2e ms/token (tok/s) |
|---|---|---|
| shipped | 18.425 | 19.30 (51.8) |
| fast dequant, `FUSE_GLUE=0` | 17.776 | 18.65 (53.6) |
| fast dequant + glue levers (default) | 17.010 | 18.07 (55.3) |

Gates: the dequantised weights are identical to the shipped ones (device
check of `dequant_e2m1_8_fp16_fast` against the shipped function over all 2^32
words x 4 scale pairs x 4 fp16 pairs: 0 mismatches; production kernels 0 of
658,944 outputs differ), so the fp64-oracle error is unchanged.  The 163
oracle / bit-identity tests of the moe suite pass on the new kernel; the new
`test_nvfp4_scale_keeps_the_shipped_double_rounding_on_tie_cases` (8 tie-prone
(byte, global) pairs, block_m 1/2) passes on the shipped kernel and on this
one and fails on the hoisted first version; the new
`test_nvfp4_dequant_is_exact_for_every_nibble_and_scale_class`
(one expert, one-hot activations: the kernel output must equal the fp16
dequantised weight exactly, for all 16 E2M1 nibbles and scale bytes of every
e4m3 class including subnormals and negatives, block_m 1/2/4/8) passes; 4
hand mutants are killed (selector byte, magnitude mask, subnormal threshold,
wrong scale row; the subnormal-threshold mutant survives the random-scale
oracle tests and is killed by the exactness test only).  NLL, real ops with
the default flags against the shipped run: mean |dlogp| 0.0477 vs run-to-run
floor 0.0504 (p99 0.3653 vs 0.3612, max 0.7664 vs 0.4929), PASS (measured with the
hoisted scale; the final kernel computes the shipped scale bit for bit).

### 3.5 Tried and dropped

Prefetching the gate_down and router weights before the activation staging
(`PF` variants) was bit-identical but gave nothing at T=1 (route 13.81 vs
13.87 us) and was slower at T=3 (gate_down +1.2 - 1.5 us): removed.
Software pipelining of the NVFP4 weight loads (-2 to -6% at bm=1, +3 to -1%
at bm=2) is not worth the registers.

## 4. Correctness evidence

Oracles: the fused GEMM against an fp64 swiglu + dequantised matmul (relative
error <= 0.9% of rms, the same class as the plain kernel), bit-identity against
the old two-launch path, and the gate-mix split against the fp32 reference
(plain kernel and split kernel both inside the same ulp tolerance).  New unit
cases: 42 `swiglu` tests, 11 `_f32w` gate cases (T = 1..5, huge / tiny
activations so the eps path matters), 4 `_f32w` route cases.

Hand mutants, each killed by the suite (`out/mutants_run.log`,
`MUTANT_GATE PASS`; `out/mm6.log`):

| id | mutant | killed by |
|---|---|---|
| MM1 | up read from row 0 | `test_moe_swiglu_gemm_matches_fp64[nvfp4-1-1]` |
| MM2 | sigmoid sign flipped | same |
| MM3 | up softcap dropped | same |
| MM4 | gate uses up_cap | same |
| MM5 | wrong row stride | same |
| MM6 | `__float2half_rz` instead of `_rn` (1-ulp rounding) | bit-identity tests (4) and one fp64 case |
| GM1 | wave rank base `(wave & 3) * RPW` | `test_msg_gate[qkv_full_t1_f32w]` |
| GM2 | last wave skipped in the sum | same |
| GM3 | partial slot dropped | same |
| GM4 | last-arriver counter not reset | same |
| GM5 | eps dropped from the norm | `test_msg_gate[qkv_full_t4_f32w_tiny]` (new case) |

Unmutated: moe 42 passed, glue 20 passed (`I2_GLUE_JIT`, flag on).

Real-model quality gate (`tools/rdna/inkling2/real_check.py` NLL mode in
process with the JIT ops, 800 tokens of `inkling2-port-spec.md`, flag on vs a
flag-off run; `out/nll_n1_*.log`): mean |dlogp| 0.0433 (p99 0.305, max 0.613)
against a run-to-run floor of 0.0504, **NLL AGAINST PASS** (limit
2 x floor + 0.01).  Mean logp -1.9862 on, -1.9836 / -1.9826 off.  The NLL
text runs as one 799-token prefill, so it exercises the SwiGLU fusion but not
the decode-only gate-mix split; that is covered by the unit cases, the lab and
the greedy stream comparison in section 5.

## 5. End to end with the rebuilt ops

Real rebuilt `_rocm_C` (no JIT), HIP 4, same engine configuration
(`out/e2e_e2_{off,on}.log`, `out/spec_s2{off,on}_k2.log`,
`out/pytest_p1.chain.log`, `out/nll_n2_on.log`, `out/gen_streams*.json`):

| check | `=0` (shipped) | `=1` |
|---|---|---|
| graph replay min (ms) | 18.4248 | 17.6754 |
| b=1 e2e ms/token (tok/s) | 19.30 (51.8) | 18.72 (53.4) |
| DSpark K=2 ms/token (tok/s), accepted / step | 15.72 (63.6), 1.037 | 15.50 (64.5), 0.999 |
| glue pytest, installed ops | 227 passed | 227 passed |
| moe pytest, installed ops | 159 passed | 159 passed |
| NLL: mean abs dlogp against `=0` (floor 0.0504, limit 0.1108) | - | 0.0402 (p99 0.2605, max 0.5679), PASS |
| greedy streams, 16 x 160 tokens: identical streams / identical-prefix tokens | `=0` vs `=0`: 4/16, 1167/2560 | `=1` vs `=0`: 5/16, 1199/2560; vs second `=0` run: 5/16, 1278/2560 |

The K=2 row includes the two-row expert blocks and both fusions; at constant
acceptance the step is 1.04 ms shorter (32.02 -> 30.98 ms) [derived].  The
divergence of the greedy streams with the flag is inside the divergence of
the shipped path against itself (two shipped runs differ after 4 - 117
tokens: the fp16 CAS atomic adds of the expert GEMMs are order dependent).
The NLL text is an 800-token prefill, so it does not exercise the
decode-only gate-mix split; that is covered by the unit cases, the lab runs
and the stream comparison.  No float atomics are involved in the new kernels
(integer last-arriver counters only), so they add no nondeterminism.

## 6. Speculation: cost structure and what 1.6x needs

DSpark (greedy draft, K draft tokens, 1 + K verify rows) at b=1, TP=1, graphs
on, 24 prompts x 128 tokens (`out/spec_s1_k*.log`, `VLLM_RDNA_PHASE_TIMING`;
cumulative acceptance by position):

| K | rows | forward ms | draft ms | step GPU ms | accepted / step | e2e ms/token | tok/s | speedup |
|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 17.12 | - | 19.31 | - | 19.42 | 51.5 | 1.000 |
| 1 | 2 | 21.62 | 4.29 | 28.18 | 0.652 | 17.41 | 57.4 | 1.115 |
| 2 | 3 | 24.90 | 4.40 | 31.60 | 1.025 | 15.83 | 63.2 | 1.227 |
| 3 | 4 | 28.08 | 4.56 | 34.94 | 1.254 | 15.83 | 63.2 | 1.227 |
| 4 | 5 | 29.69 | 4.68 | 36.68 | 1.353 | 15.74 | 63.5 | 1.234 |

Cumulative acceptance (P that the first i drafts are all accepted), K=4:
0.628 / 0.375 / 0.222 / 0.128.  lm_head is 2.1 ms and the sampler 0.06 ms
(identical for every K).  The draft costs ~4.4 ms; the traced draft pass of
`inkling2-dspark.md` splits into the LM-head base-logits sweep 2.15 ms (922 MB
at 429 GB/s), the three layers 1.7 ms, ~0.13 ms per Markov step and the
context path 0.2 ms (= 4.3 ms at K=2).  A draft position pays only if it costs less than the ms/token before
it: position 1 14.3 (vs 19.4), position 2 8.8 (vs 17.4), position 3 15.8
(vs 15.8: break-even), position 4 13.7 (vs 15.8: +0.09 ms), so K=2 is the
design point and K >= 3 buys nothing.

Why 1 -> 3 rows adds 7.8 ms of forward: it is the **expert union**, not the
dense or glue work.  The per-visit NVFP4 cost grows from 119 us (8 distinct
experts, 1 row) to 238 us (16.3 distinct, 3 rows); x 44 visits that is +5.2 of
the 7.8 ms, at ~300 GB/s (section 3.3).  Dense GEMVs are row-independent (same
bytes), attention and glue add the rest.

**What 1.6x needs at K=2** (step = GPU + host = 32.06 ms measured):

* at the measured step: 1.64 accepted / step (measured 1.025), i.e. an equal
  conditional acceptance of 0.875 per position (measured 0.63 / 0.62);
* at the measured acceptance: a step of <= 24.6 ms (7.4 ms below today);
* both halfway: acceptance 0.80 per position (1.44 / step) with the step
  reduced by the levers below (~6.4 ms with all of them) gives 1.85x
  `[INFERENCE]`.

tokens/s grid at K=2 (`tok/s = 1000 (1 + accepted) / step`), the 100 tok/s
line:

| step ms | acc 1.00 | acc 1.25 | acc 1.50 | acc 1.75 | acc 2.00 |
|---|---|---|---|---|---|
| 32.1 | 62.3 | 70.1 | 77.9 | 85.7 | 93.5 |
| 28.0 | 71.4 | 80.4 | 89.3 | 98.2 | 107.1 |
| 26.0 | 76.9 | 86.5 | 96.2 | 105.8 | 115.4 |
| 24.0 | 83.3 | 93.8 | 104.2 | 114.6 | 125.0 |
| 20.0 | 100.0 | 112.5 | 125.0 | 137.5 | 150.0 |

Levers at K=2 (ms per step; cumulative from today's 63.2 tok/s):

| lever | ms/step | cumulative tok/s | basis |
|---|---|---|---|
| SwiGLU folded into w2 | 0.125 | 63.4 | measured |
| gate_mix split | 0.133 | 63.7 | measured |
| NVFP4 two-row blocks | 0.572 | 64.9 | measured on real expert sets (kernel level) |
| draft lm_head over a 32k-token subset | 1.75 | 68.7 | `[INFERENCE]` 0.92 GB -> 0.15 GB |
| draft layers 1.7 -> 1.2 ms | 0.5 | 69.9 | `[INFERENCE]` |
| MoE kernels 300 -> 450 GB/s on the verify rows | 3.3 | 78.9 | `[INFERENCE]` |

The glue levers together are worth about 0.26 ms of a K=2 step (0.5 ms of a
K=0 step).  Even all the levers listed reach ~79 tok/s at today's acceptance;
100 tok/s at b=1 needs acceptance >= 1.5 per step (equal rate 0.78) **and** a
step near 26 ms, or a different draft.

### 6.1 Prompt-lookup (n-gram) drafting costs no draft pass

Offline replay of exact greedy streams (8 chat-template prompts + 8
copy-heavy prompts, 160 greedy tokens each, K=0 graphs, the same streams for
every row below; `tools/rdna/inkling2/ngram_sim.py`): at every step the longest suffix n-gram (n = 3..2) that
occurred earlier in prompt + output is looked up and the tokens that followed
it (at most K) are verified; the verify cost is the measured forward time for
1 + len(draft) rows and there is no draft pass.  Results (16 streams, K=4):

| policy | chat streams | copy-heavy streams |
|---|---|---|
| prompt lookup n 3..2, K=4 | 1.21x (62.4 tok/s) | 1.17x (60.3 tok/s) |
| prompt lookup n 3..2, K=2 | 1.19x | 1.14x |
| prompt lookup n 3..1, K=4 | 1.13x | 1.01x |
| DSpark K=2 (measured) | 1.227x | |

The model's replies start with a thinking section that quotes the prompt, so
a free lookup reaches the same gain as the 4.4 ms draft on these streams
(per-stream 0.98x - 1.76x; the verbatim-copy prompts spend their first 160
tokens still thinking).  The two are complementary: the lookup has no draft
cost and a high acceptance on repeated spans, the draft model is the fallback;
running the draft only when the lookup misses saves the 4.4 ms draft on the
steps the lookup serves `[INFERENCE]`, not measured.

### 6.2 Vocabulary-pruned draft head: coverage check (CPU, `out/vocab_cov.log`)

The draft's base-logits sweep (2.15 ms, 0.92 GB) could read only the N most
frequent rows.  The draft can then never propose a token outside the set, so
`coverage = P(target greedy token in the set)` bounds the acceptance it keeps.
Set built from the repo docs + python sources (1500 files, 7.2 M tokens with the
model's tokenizer, 21.3 k distinct ids), evaluated on the 2560 greedy tokens of
the 16 streams of section 6.1 (655 distinct ids):

| N | share of the head | head ms at 429 GB/s | chat coverage | copy-heavy coverage |
|---|---|---|---|---|
| 4096 | 2.0% | 0.04 | 0.486 | 0.823 |
| 8192 | 4.1% | 0.09 | 0.567 | 0.908 |
| 16384 | 8.2% | 0.18 | 0.673 | 0.990 |
| >= 21.3k (all ids of the corpus) | 10.6% | 0.23 | 0.713 | 0.996 |

A static technical-text set is a poor proxy for chat: 29% of the chat tokens
(conversational words and the special tokens of the reply format) never occur
in it, and the BPE ids are not frequency ranked (the first 100k ids cover only
46% of the generated tokens).  Replacing the 2.15 ms sweep by a ~0.3 ms one
pays only if the kept acceptance stays above ~89% of today's (K=2: break-even
`(1 + c x 1.025) / (32.06 - 1.75) = 2.025 / 32.06`), an upper bound on the
loss because the full-vocabulary draft is rarely right on rare tokens anyway.
The set therefore has to come from model outputs on a calibration corpus; that
was not available here, so the lever stays `[INFERENCE]`.

## 7. Next smallest steps

1. Expert GEMM bandwidth on 2 - 5 row verify steps (the single biggest lever,
   3.3 ms per K=2 step if 300 -> 450 GB/s): the kernels use bm blocks that
   re-stream shared experts per row group; a block that holds all rows of an
   expert for the 8 distinct-expert case, or a K-split of w13 over more CTAs.
2. Vocabulary-pruned or int8 draft lm_head (2.15 ms of the 4.4 ms draft).
3. A hybrid draft: prompt lookup first, DSpark only on misses.
4. Per-visit glue is ~4.1 ms in 535 launches; with the measured 1.16 us launch
   tax the remaining fusion candidates are each ~0.2 ms or less
   (sconv + attn_res, kv_prep + decode, combine + decode); `moe_route` is
   0.75 ms (15.7 us per launch) and the largest single glue kernel.
