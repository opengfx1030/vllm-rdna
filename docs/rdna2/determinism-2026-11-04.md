# Run-to-run determinism of Flash-Next on gfx1030 (2026-11-04)

Question: why do identical requests to `flashnext-mtp2` (Qwen3.8-Flash-Next
AWQ, TP4/EP4, tree `4a3db777ae`) return different prompt logprobs, by 10-19
nats at single positions ([w4a8-quality-2026-11-04.md, nondeterminism
section](w4a8-quality-2026-11-04.md#nondeterminism-of-identical-requests))?

## Summary

- **Where it starts:** the routed MoE of layer 0
  (`layers.0.mlp.experts`) is the first op whose output differs between two
  identical requests while all its inputs match. The trace shows this on all
  four ranks, for both windows and both prefill chunks. Each expert's partial
  is computed the same way every time; the epilogue that adds a token's
  experts into one row with atomics rounds in arrival order. 2.0-2.3 % of the
  layer-0 MoE output elements differ, 81 % of them by 1 or 2 fp16 ULPs.
- **Verdict: benign.** This is rounding-order noise, not a race, an
  uninitialized read or a missing sync. With the order fixed, every hooked op
  of the step is bit-identical (eager) and the compiled server returns
  bit-identical prompt logprobs in 32 of 32 repeat pairs. The fp32 epilogue
  (`VLLM_RDNA2_MOE_FP32_ACCUM=1`) is still order-dependent, which is why it did
  not help in the quality run.
- **Why 15 nats:** the model amplifies any ULP. One ULP in one element of
  layer 0 reproduces the full effect: mean 0.17 nats per position, 88 moves
  of 1 nat or more, up to 9.5 nats, at positions the quality run also flagged
  (1990, 2009). Router top-10 sets with near-tied 10th and 11th scores, some
  exactly tied, flip along with it. In the noise run they flip in 15 rows at
  layer 1, 112 at layer 10 and 390-598 of 1024 in every layer from 30 on.
  GDN and attention carry each flip into every later position.
  [INFERENCE] The flips are the amplifier; routing was not pinned to
  isolate them.
- **Knob:** `VLLM_RDNA2_MOE_DETERMINISTIC=1` (default off), commit
  `d6384840c2`. Measured cost on `flashnext-mtp2`: +0.8 % ITL at c=1 and +0.4
  to +1.3 % at c=8, with prefill unchanged within noise. That server run
  predates the second part of the knob, the tie-stable radix top-k (see
  "Radix top-k tie membership" below), whose cost was measured on the kernel
  only.

| arm | tree | mode | knob | phases |
|---|---|---|---|---|
| T1 | production `4a3db777ae` | eager + trace hooks | — | trace |
| P0 | production | compiled, as served | — | identity, perf |
| T2 | fix `d6384840c2` | eager + trace hooks | `VLLM_RDNA2_MOE_DETERMINISTIC=1` | trace, noise, elem |
| P1 | fix | compiled, as served | `VLLM_RDNA2_MOE_DETERMINISTIC=1` | identity, perf |
| P0r | production | compiled, as served | — | perf (rerun of an invalid P0 cell) |

All arms ran on HIP 2-5 with the unchanged `flashnext-mtp2` recipe (W4A8 v2
MoE: `VLLM_RDNA2_W4A8_SDOT4=1`, `VLLM_RDNA2_W4A8_BLOCK_M=8`,
`VLLM_RDNA_W4A8_KBLOCK=-1`; MTP 2; `MAXBAT=2048`; prefix caching and mamba
align) in venv `7.14.0_0.32.0`, one boot per arm. The fix tree is
`git archive` of `d6384840c2` with the production tree's built libraries;
`csrc/` is unchanged between the two commits.

## Order-dependent kernels on the 2048-token prefill path

Code audit of every op a fresh 2048-token prefill runs under the production
recipe. "Order-dependent" means the result can change with the scheduling of
identical inputs. The eager trace below shows which ones actually fire.

| op / kernel | file | order-dependent? | read before write? | knob |
|---|---|---|---|---|
| Routed MoE w2, W4A8 (steps >= 512 tokens): top-k reduce into the output row | `csrc/rocm/moe_accum_rdna2.cuh` `moe_accum_rows` (packed-fp16 CAS-64), `csrc/rocm/moe_w4a8_rdna2.cu` (`output_topk = 10`) | **yes**: a token's local experts add into one fp16 row in arrival order, one rounding per add | no | `VLLM_RDNA2_MOE_DETERMINISTIC=1` |
| Same, fp32 epilogue (`VLLM_RDNA2_MOE_FP32_ACCUM=1`) | `moe_accum_rdna2.cuh` fp32 `atomicAdd` into a cached scratch | **yes**: fp32 adds in arrival order; the final fp16 cast hides most of it | no | `VLLM_RDNA2_MOE_DETERMINISTIC=1` |
| Routed MoE w1, W4A8 | `moe_w4a8_rdna2.cu` v2 kernel, `VLLM_RDNA_W4A8_KBLOCK=-1` | no: one K block, direct row stores (`KBLOCK` unset or positive would add split-K atomics) | no | keep `KBLOCK=-1` |
| Routed MoE w1 + w2, W4A16 (steps < 512 tokens: decode, MTP verify, the 1-token tail of a prompt) | `csrc/rocm/moe_q_gemm_rdna2.cu`: split-K over K/256 slices plus the top-k reduce, both atomic | **yes** | no | `VLLM_RDNA2_MOE_DETERMINISTIC=1` (Q32 fixed point) |
| Router `topk_softmax`, `moe_align_block_size` | `csrc/moe/` | no (ties go to the lower expert id; the align sort only permutes rows inside an expert segment, which reorders the atomics above) | no | — |
| Dense fp16 GEMMs (GDN/QSA projections, router, shared expert, PLE, lm_head) | `F.linear` → TunableOp rocBLAS rows in `tunableop/rocm7.14-rocblas5.5` | not decidable from the code (rocBLAS may pick atomic split-K solutions); measured deterministic below | no | `PYTORCH_TUNABLEOP_ENABLED=0`, `VLLM_BATCH_INVARIANT=1` |
| GDN prefill chain (prep, KKT, solve-WY, delta-h, o), gated RMSNorm, causal conv1d | `csrc/rocm/gdn_prefill_*_rdna2.cu`, Triton `_causal_conv1d_fwd_kernel` | no: fixed reduction order, no atomics | no (a null-block sequence left its `out` rows unwritten; only dummy rows hit it; fixed, see below) | — |
| QSA indexer, top-k, expand, sparse attention + split merge | `csrc/rocm/qsa_rdna2.cu`, `csrc/sampler.cu` | no with at most 2048 tokens of context (the top-k keeps every block, so QSA is exact causal attention). Above that, **yes** for exactly tied scores at the top-k cut: which tied columns are kept follows atomic arrival order. The order of the kept blocks does not matter, the RDNA2 expand step sorts them (`canonical_order`, `qsa_rdna2.cu:725`) | no | `VLLM_RDNA2_MOE_DETERMINISTIC=1` (the lowest tied columns win; see below) |
| HC mix prefill (ticketed split-K down GEMM) | `csrc/rocm/hc_prefill_rdna2.cu` | no: the last block sums the fp32 partials in fixed z order | no | `VLLM_RDNA_HC_DOWN_SPLITS` |
| HC norm/combine, RMS/Gemma norms, PLE lookup/gate/conv, input prep, prompt-logprob kernels | various | no | no | — |
| TP all-reduce, 5 MiB per call (RCCL) | `cuda_communicator.py` | no in practice (fixed algorithm for a fixed size and topology); measured below | no | `VLLM_BATCH_INVARIANT=1` |
| rdna_ar one-shot all-reduce (<= 256 KiB) | `csrc/rocm/rdna_allreduce.cuh` | no: fixed rank order | no | — |

## Where the divergence starts

Eager trace of the production tree and recipe (arm T1: `EAGER=1`, otherwise
`flashnext-mtp2` as served, HIP 2-5). A `sitecustomize` shim
(`bench_results/2026-11-04_determinism/scripts/dtrace/`) hooks every module of
the target model, the hyper-connection methods and the TP collectives, and
records an exact per-row hash of every input and output. Two windows (mix 28,
docs 0) are scored twice, interleaved, one request at a time. The scheduler
runs each 2048-token prompt as three forwards: 1024 tokens, 1023 tokens, then
the last token on its own. Rows 1024-2046 read the KV cache and GDN state of
the first chunk. All prompt logprobs come from the first two forwards, which
take the W4A8 MoE path (512 tokens or more).

In all four forward pairs (2 windows x 2 chunks) and on all four ranks the
first op whose output differs while all its inputs match is the routed MoE of
layer 0 (`language_model.model.layers.0.mlp.experts`). Everything before it in
the step is bit-identical: the HC mix, the whole GDN layer including its
`out_proj` all-reduce, the HC combine, the router logits and the shared
expert. The first differing tensor is the input of the MoE all-reduce, i.e.
each rank's local routed sum. From there on 696 of the 774 calls of the step
differ.

| fwd pair (window, chunk) | layer-0 MoE elements that differ | rows | max \|d\| | rel. L2 | ulp distance (count) | layer-1 MoE elements that differ | layer-1 rel. L2 |
|---|---:|---:|---:|---:|---|---:|---:|
| mix 28, rows 0-1023 | 54,423 (2.08 %) | 630 / 1024 | 4.9e-4 | 7.2e-5 | 1: 28,413; 2: 15,715; 4: 5,330; 5-16: 3,691; >16: 1,235; sign: 6 | 96.2 % | 1.2e-2 |
| mix 28, rows 1024-2046 | 55,604 (2.12 %) | 627 / 1023 | 4.9e-4 | 7.2e-5 | 1: 29,391; 2: 15,830; 4: 5,290; 5-16: 3,811; >16: 1,215; sign: 12 | 96.3 % | 8.6e-3 |
| docs 0, rows 0-1023 | 60,441 (2.31 %) | 658 / 1024 | 4.9e-4 | 8.3e-5 | 1: 31,561; 2: 17,889; 4: 5,657; 5-16: 3,948; >16: 1,338; sign: 3 | 96.2 % | 7.8e-3 |
| docs 0, rows 1024-2046 | 52,825 (2.02 %) | 630 / 1023 | 4.9e-4 | 7.0e-5 | 1: 27,810; 2: 15,046; 4: 5,156; 5-16: 3,563; >16: 1,193; sign: 12 | 96.1 % | 1.2e-2 |

Row 0 differs at layer 0 already. That is why no CAS arm of the quality run
ever returned an identical logprob at position 0, a row whose computation
depends on no other row.

## Single-layer replay

`scripts/repro_moe.py` runs `_rdna2_fused_moe` with the per-rank Flash-Next
geometry: hidden 2560, intermediate 640, 128 of 512 experts local, top-10,
group 128, fp16, synthetic int4 weights and random distinct routing. It
repeats the same call 20 times and compares every output bitwise with the
first one. Times are per call, one call captured in a CUDA graph and replayed
(HIP 2, idle, `results/repro_moe-idle.log`).

| tokens | path | epilogue | runs that differ from the first (of 19) | elements that differ (mean over those runs) | us per call | vs CAS |
|---:|---|---|---:|---:|---:|---:|
| 2048 | W4A8 | CAS (default) | 19 | 0.60 % | 2,222 | |
| 2048 | W4A8 | fp32 | 19 | 0.001 % | 3,000 | +35.0 % |
| 2048 | W4A8 | deterministic | **0** | 0 | 2,260 | +1.7 % |
| 512 | W4A8 | CAS (default) | 19 | 5.11 % | 1,061 | |
| 512 | W4A8 | fp32 | 19 | 0.002 % | 1,295 | +22.1 % |
| 512 | W4A8 | deterministic | **0** | 0 | 1,080 | +1.8 % |
| 24 | W4A16 | CAS (default) | 19 | 53.5 % | 173 | |
| 24 | W4A16 | fp32 | 19 | 0.020 % | 228 | +31.8 % |
| 24 | W4A16 | deterministic | **0** | 0 | 237 | +36.7 % |
| 3 | W4A16 | CAS (default) | 19 | 79.7 % | 48 | |
| 3 | W4A16 | fp32 | 19 | 0.29 % | 61 | +27.2 % |
| 3 | W4A16 | deterministic | **0** | 0 | 66 | +38.2 % |
| 1 | W4A16 | CAS (default) | 19 | 82.4 % | 33 | |
| 1 | W4A16 | fp32 | 13 | 0.21 % | 47 | +41.4 % |
| 1 | W4A16 | deterministic | **0** | 0 | 51 | +51.5 % |

The CAS rate at 2048 tokens (0.6 % of one rank's partial) fits T1's 2.1 %
after the all-reduce of four ranks. The W4A16 path also splits K into
256-wide slices that meet in the same atomics, so at decode sizes most
outputs differ.

## Deterministic mode makes the server bit-reproducible

`VLLM_RDNA2_MOE_DETERMINISTIC=1` (fix tree `d6384840c2`) changes only the
routed-MoE epilogue of the RDNA2 compressed-tensors path:

- W4A8 steps run w2 with `output_topk = 0`: one fp16 row per (token, expert)
  pair, written with plain stores. `moe_sum` then adds each token's local
  rows in fp32, in top-k order. This needs the single-K-block w1 (`KBLOCK=-1`,
  as in the recipe); with any other `KBLOCK` the step falls back to W4A16.
- W4A16 steps accumulate w1 and w2 in Q32 fixed point (`int64` atomics, the
  existing expert-cache epilogue), which is associative.

Arm T2 repeats the T1 trace on the fix tree with the knob: both windows score
**bit-identically** in both passes (100.00 % identical positions), and the
per-row hashes of all 774 hooked calls per step match on all four ranks, for
both chunks of both windows. That includes every rocBLAS GEMM, the GDN and
QSA kernels and all TP all-reduces, so on this path the MoE epilogue was the
only source of run-to-run variation.

The compiled server behaves the same way. Arm P1 is the fix tree with the
knob, compiled with CUDA graphs as served. It scored 16 windows (mix 24-31,
docs 0-7) three times each, interleaved. All 32 repeat pairs are
bit-identical: 100.00 % of positions, position 0 in 32 of 32 pairs. P0 is the
production tree in the same configuration without the knob. There, 0.46 % of
positions are identical and position 0 is identical in 0 of 32 pairs. Top-1
agreement is 95.97 %, the mean \|dlogp\| is 0.100 nats, 1,004 positions move by
1 nat or more and 23 by 5 nats or more. The largest move is 19.04 nats (mix
26 position 2029, -0.60 vs -19.64).

## Why ULP noise moves single logprobs by 10-19 nats

Arm T2 then re-adds controlled noise to the now deterministic server:

- **noise**: +-1 ULP on a random 2 % of the elements of every layer's MoE
  output (the size of the CAS noise measured in T1), a different pattern in
  each pass and the same on all ranks;
- **elem**: +1 ULP on one element, column 0 of the layer-0 MoE output at the
  first row of each chunk (positions 0 and 1024), nothing else.

| run (mix 28 unless noted) | identical positions | mean \|dlogp\| | moves >= 1 / >= 5 nats | max \|dlogp\| (position) |
|---|---:|---:|---:|---:|
| P0 production, compiled, pass 2 / 3 vs 1 | 0.00 % / 0.00 % | 0.180 / 0.193 | 83 / 6, 95 / 7 | 13.03 (2009), 12.09 (2009) |
| T1 production, eager | 0.00 % | 0.179 | 86 / 5 | 7.82 (2005) |
| T2 knob, clean | 100.00 % | 0 | 0 / 0 | 0 |
| T2 knob + noise, pass 2 vs pass 1 | 0.00 % | 0.162 | 65 / 4 | 10.72 (1673) |
| T2 knob + noise vs clean (2 passes) | 0.00 % | 0.166 | 138 / 10 | 12.51 (1990) |
| T2 knob + one ULP in one element vs clean | 0.00 % | 0.175 | 88 / 3 | 9.53 (1990) |
| docs 0: P0 / T1 / T2 noise pass 2 vs 1 | 0.05-0.15 % / 0.05 % / 0.05 % | 0.159-0.162 / 0.158 / 0.172 | 45-52 / 0, 52 / 0, 57 / 0 | 3.48, 3.41, 3.74 |

One ULP in one element reproduces the whole effect, at the same positions
the quality run reported (1990, 2009). The trace of the elem pass shows how:
the perturbed row stays the only differing row through layer 1, the GDN
recurrence of layer 2 carries it into all 1024 rows of the chunk, and every
later layer differs in every row.

The growth goes together with discrete router decisions (rank-0 dumps of
every decoder layer and router logits, noise passes, rows 0-1023):

| layer | 1 | 2 | 3 | 10 | 20 | 30 | 40 | 47 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| median row-wise rel. difference of the layer output | 1.6e-4 | 7.4e-4 | 1.8e-3 | 6.9e-3 | 2.9e-2 | 7.7e-2 | 1.2e-1 | 1.0e-1 |
| rows whose router top-10 set differs (of 1024) | 15 | 27 | 50 | 112 | 269 | 517 | 572 | 515 |

A changed top-10 set swaps whole experts in that row's MoE output instead of
moving it by ULPs, and GDN and attention pass the changed rows on to every
later position. [INFERENCE] That makes the flips the amplifier. The run that
would prove it, noise with the routing pinned to the clean pass, was not
made. At the rows with the largest moves the two passes no longer
compute the same hidden state at all: at positions 1673 and 1990 the
row-wise relative difference is 0.8-2.0 in every layer from 31 on. Both rows
change their top-10 set in every layer from 18 to 47. The 10th-11th router
logit margin in those layers is 0.000-0.25, with exact fp16 ties (margin 0)
in two layers each.

## Verdict

**Benign, not a bug.** The run-to-run difference has one source, the
routed-MoE epilogue. It adds each token's expert partials into a shared row
with atomics, in whatever order the blocks finish. Each run's sum is a valid
fp16 (or fp32) rounding of the same value. There is no race on data, no read
of unwritten memory and no missing synchronization. Removing the order
dependence makes every op of the step bit-identical (T2) and the compiled
server bit-reproducible (P1). The fp32 epilogue does not help because its
adds are still ordered by arrival.

The size of the moves is a property of the model, not of the noise. One ULP
in one element of layer 0 gives the same 0.17 nats mean change and moves of
the same size (up to 9.5 nats) as the full CAS noise. [INFERENCE] The likely
path is the router. It has near-tied 10th and 11th expert scores, some
exactly tied. Through 48 layers a ULP reaches one of them, the expert set of
a row changes, and GDN and attention carry that into every later position. A
position whose prediction sits between two continuations (in the quality
run often `<|im_end|>` against the next word) can then land on either side.

## The knob and what it costs

`VLLM_RDNA2_MOE_DETERMINISTIC=1` (default `0`, read once at import) in the
server environment. It covers the RDNA2 compressed-tensors MoE method
(`_rdna2_fused_moe`: Flash-Next AWQ, including the resident-weights path).
It does not cover the modular `RDNA2W4A16MoEExperts` path
(`rdna2_w4a16_moe.py`, `moe_wna16` checkpoints). Calls that bring their own
fixed-point accumulator (the expert cache) were already order-independent.
With a `VLLM_RDNA_W4A8_KBLOCK` other than `-1`, W4A8 steps fall back to
W4A16 under the knob, which costs the W4A8 prefill gain.

The knob also covers the radix top-k that QSA uses to select context blocks
(`top_k_per_row_decode` and `_prefill`, `csrc/sampler.cu`): among exactly
tied scores at the cut the lowest columns win, instead of the first threads to
reach an atomic. Unlike the MoE switch, the C++ launcher reads the variable
(`vllm::topKDeterministicTies`, `sampler.cu`), once, at the first launch of
either op, and forwards to the tie-stable kernels, which live in a second
translation unit (`csrc/sampler_deterministic.cu`, the same source compiled
with `VLLM_TOPK_DETERMINISTIC_TU`). The value is fixed for the process, so a
CUDA graph captured at any time replays the variant an eager call runs. Set it
in the server environment before the process starts; a later `os.environ`
change does not reach the top-k. Every caller of the two ops is covered (QSA,
the sparse indexers, the `topk.cu` fallback). The gfx950 device-length-aware
kernels are not covered (they keep the default tie order); only gfx1030 was
run.

Server cost: `flashnext-mtp2`, TP4 on HIP 2-5, one boot per arm, warmup on,
every cell inside a clean KFD window. P0's first 16k cell lost 10-24 s of GPU
queue time to KFD evictions and was rerun in a second production boot (P0r).

| cell | metric | P0 production | P0r production, 2nd boot | P1 knob |
|---|---|---:|---:|---:|
| 16384 in / 256 out, c=1 | TTFT mean | invalid | 5.62 s | 5.63 s |
| | prefill tok/s/req | invalid | 2,917 | 2,912 |
| | ITL median / mean (one MTP step) | invalid | 24.8 / 24.6 ms | 25.0 / 24.8 ms |
| | MTP acceptance length | invalid | 2.13 | 2.24 |
| 1024 in / 512 out, c=8 | TTFT mean | 0.93 s | 0.94 s | 0.91 s |
| | prefill aggregate tok/s | 2,756 | 2,731 | 2,736 |
| | ITL median / mean | 45.3 / 53.0 ms | 46.4 / 53.5 ms | 46.8 / 53.7 ms |
| | MTP acceptance length | 2.16 | 2.10 | 2.02 |
| | output tok/s/req (aggregate) | 40.67 (273.7) | 39.13 (267.3) | 37.55 (256.2) |

- Per MTP step the knob costs +0.2 ms at c=1 (+0.8 % mean ITL) and +0.2 to
  +0.7 ms at c=8 (+0.4 to +1.3 %). The two production boots differ from each
  other by 0.5 ms at c=8.
- Prefill throughput moves by less than 1 % (2,912 vs 2,917 tok/s/req at 16k
  c=1). TTFT moves by +0.2 % at c=1 and by -2 to -3 % at c=8 (0.91 vs 0.93 and
  0.94 s). [INFERENCE] The knob only adds work, so the faster c=8 TTFT is
  noise.
- Output tokens per second is not a usable comparison here. MTP acceptance
  depends on the generated text, which the epilogue's rounding also changes.
  It moves by 3 % between the two production boots (2.16 vs 2.10 at c=8).
- The replay above puts the kernel cost at +38 us per MoE call at 2048
  tokens (+1.7 %), and +17 to +18 us at 1 to 3 tokens (+38 to +52 %).
  [INFERENCE] Over 48 layers the 3-token figure would be about 0.9 ms per
  decode step, more than the +0.2 ms the c=1 cell measured. The
  deterministic W4A16 call runs about twice as many kernels as the default
  one (block alignment instead of the fused route-and-zero, two int64 zero
  fills, two fixed-point conversions), which is the likely source of its
  small-batch cost.

Scope of the guarantee:

- **Measured:** prompt scoring (2048-token windows, prompt logprobs), eager
  and compiled.
- **Not measured at the server level:** generation. For the MoE call itself,
  the replay shows the deterministic call bit-identical at 1, 3 and 24
  tokens. The other decode kernels (MTP drafter, GDN and QSA decode, sampler)
  were not audited.
- **Not addressed:** batch invariance, i.e. the same request scored inside a
  different batch.

## Issues found by the audit

None of these fires in the runs above. All five are fixed, each with a test
that fails before the fix and passes on it, run on one V620 (HIP 9) from
exact-SHA trees; logs are on the box under `~/w4a8_runs/audit/`. The fifth,
the top-k tie fix, acts only under the deterministic knob; the default
kernels are unchanged.

- **Causal conv1d null-block rows** (`1711e69f13`). A sequence mapped to the
  null block returned without writing its rows of `out`
  (`torch.empty_like`), so they held leftover allocator bytes. This applied
  to the Triton prefill (`causal_conv1d.py:122-127`), the Triton update
  (`:768-773`, only with a caller `out` that does not alias `x`), and the HIP
  prefill and spec-verify kernels (`csrc/rocm/causal_conv1d_rdna2.cu:271,
  541`). They now store zeros for those rows.
    - Test: `tests/kernels/mamba/test_causal_conv1d_null_block.py`, 7 failed
    before, all 55 (with the update tests) pass after.
    - Production: no real request maps to the null block in align mode; only
    dummy, warm-up and capture rows do, and their output is discarded.
    Kernel time is unchanged (Triton prefill 112 µs, HIP prefill 415 µs).
    - Left as is: the HIP single-token update `causal_conv1d_update_rdna2`
    still treats the null block as a scratch line (defined output).
- **QSA metadata ring rule accepted block 0** (`7921a896ea`).
  `qsa_metadata_rdna2.cu:147` used `>= 0` where the Triton kernel and the
  torch reference use `> 0`, so dummy steps (profile run, CUDA-graph warm-up
  and capture, zeroed block tables) wrote their keys into the null block's
  ring. It now uses `> 0`.
    - Test: `tests/kernels/rocm/qwen4_exp/test_qsa_metadata_null_block_rdna2.py`,
    4 of 5 failed before, all pass after.
    - Production: real requests own their ring block before their first step
    and nothing real reads block 0, so no output was affected. [INFERENCE]
    This is from the code paths, not measured at serve level.
- **FA-RDNA2 int8 prefill split-K workspaces** (`a948ee01de`). At
  `fa_rdna2.cu:4559-4571` the int8 path took the fp16 split-K path's
  persistent slots (`g_pref_*`) and sized them 16x too large
  (`[N, H_q, BR_PREFILL, kv_splits, D]`, while every kernel indexes
  `[N, H_q, kv_splits, D]`). So each int8 call zero-filled or regrew the fp16
  buffers. The int8 path now owns right-sized `g_pref_i8_*` slots. The stale
  comments are corrected, including the one claiming the fp16 layout
  corrupts memory.
    - Test: `tests/kernels/attention/test_fa_rdna2_int8_prefill_workspaces.py`
    (the int8 op is now registered for it), 1 failed before, 18 pass after.
    - Even before the fix, a captured fp16 graph replayed bit-identically
    after an int8 call: the old buffer stays alive.
    - Production: unreachable. The int8 op had no registration or caller,
    flashnext runs fp16 KV, and every attention layer is QSA.
- **fp32 MoE scratch never freed** (`c3321e922a`, test `e6d71e470c`). With
  `VLLM_RDNA2_MOE_FP32_ACCUM=1`, `moe_fp32_scratch` kept one buffer per
  (rows, n, device) in a static map: 61,440 B × T per distinct token count T
  at the Flash-Next shapes. It is now an `at::empty` shaped like the output
  on every call. Eager calls return it to the caching allocator, and graph
  capture takes it from the graph's private pool, which keeps the address
  for replays.
    - Test: `tests/kernels/quantization/test_rdna2_moe_fp32_scratch.py`
    (eager calls at 40 token counts keep no memory; two graphs in one pool
    replay correctly after eager calls at other sizes), red before, green
    after.
    - Cost: within ±0.7 % of the old cached buffer (1-2048 tokens, two
    interleaved rounds).
    - Production: the flag is off in every recipe.
- **Radix top-k tie membership** (`fix/topk-ties-deterministic`; acts only
  under `VLLM_RDNA2_MOE_DETERMINISTIC=1`). In `csrc/sampler.cu` step 3 and the
  sorted-bin gather, exactly tied logits at the k-th value were kept in
  `atomicAdd` race order, and the insertion sort kept the highest tied
  slots.
    - Fix (`09b48711bb`, cherry-picked as `2cd3e7fc41`, gated and
    restructured up to `45331fa50c`): equal logits at the cut keep the lowest
    stored indices (the column; for a merged split row, the column its block
    stored). `emitLowestKeys` (`sampler.cu:201`) picks them with a
    three-level radix select over the stored index. It replaces the atomic
    pass of step 3 (`:481`); the insertion comparator ranks ties by stored
    index (`:698`) and the sorted-bin gather refills the tied slots (`:773`).
    The selected set then depends on the scores alone. The order of the
    returned indices still follows atomic order; QSA sorts it
    (`canonical_order`).
    - Where it lives: `csrc/sampler_deterministic.cu` is `sampler.cu`
    compiled a second time with `VLLM_TOPK_DETERMINISTIC_TU`, exporting
    `top_k_per_row_{decode,prefill}_deterministic`. The ops in `sampler.cu`
    (`:1126`, `:1328`) forward to them when `topKDeterministicTies()`
    (`:1046`) reads the knob, once, at the first launch. Both files are in
    the `_C_stable` source list (`CMakeLists.txt`); hipify rewrites the
    `#include "sampler.cu"` of the second to `sampler.hip`. The second unit
    adds one more compile of `sampler.cu` to the build.
    - Test: `tests/kernels/test_top_k_per_row_ties.py`, 75 cases (decode on
    four tie patterns × five shapes from 4096 to 2.2 M columns, one of them
    strided and two on the split-and-merge path; prefill through the
    insertion and the radix kernel; CUDA-graph replay; random rows). Each mode
    runs in a child process, because the launcher reads the variable once.
    One V620 (HIP 9), exact-SHA trees: on `a5a762999f` plus the file, 36
    failed and 39 passed (every knob-on case fails); on `45331fa50c`, 75
    passed.
    - Default path unchanged. All nine loaded sections of the default unit
    in `_C_stable_libtorch.abi3.so` (`.note`, `.dynsym`, `.gnu.hash`,
    `.hash`, `.dynstr`, `.rodata`, `.text` with 226,008 B, `.data.rel.ro`,
    `.dynamic`) are byte-identical to those built from `a5a762999f`; only the
    unloaded DWARF differs. Selections on 522 tie-free rows (60 cases) are
    identical in the baseline, the fix's default and the fix with the knob on,
    and equal `torch.topk` apart from the defect listed below. Interleaved
    A/B of the default against the baseline
    (`benchmarks/kernels/benchmark_qsa_top_k_rdna2.py`, top-k 512,
    1-2048 rows × 1024-16384 columns, three rounds, plus six rounds on the
    tie-heavy shapes): median difference +0.03 % (QSA-like scores), +0.05 %
    (all zeros), -0.07 % (1/16 grid); the six-round zeros run is within
    ±0.04 % at every shape. Three cases outside ±0.35 % have round spreads
    larger than the difference (bimodal timings at 24 × 16384 and
    2048 × 8192, and 1 × 16384 on zeros).
    - Why a second unit. The first versions kept the deterministic kernels in
    `sampler.cu`. The default kernels' instructions stayed the same (the two
    jobs differed only in PC-relative offsets to the shared-array table), but
    the code object moved: the `.text` start by 0x3800 bytes, the table to
    another alignment. The default path then measured +1.3 to +2.8 % per call
    on all-zero rows of 4096 and 8192 columns at 256 and 2048 rows, in every
    round and at four data placements (0, 2, 6 and 14 MiB allocated first).
    With the second unit the code object is identical and the difference is
    gone. [INFERENCE] The slowdown came from code placement, not from the
    instructions; the mechanism was not isolated.
    - Cost of the knob, kernel only (same benchmark, knob on against knob
    off, three interleaved rounds):
        - QSA-like scores (relu of a normal): 3-6 % faster at 1-24 rows
      (0.4-1.0 µs), 20-24 % faster at 256-2048 rows (2048 × 16384: 628 to 482
      µs). [INFERENCE] The deterministic unit inlines the top-k job into its
      kernels, which the default unit cannot do without moving its code
      object; the speedup was not isolated.
        - All-zero rows (the cut inside a run of exact zeros): +20 to +26 %
      at 1-24 rows (+2.7 to +20 µs), +3 to +10 % at 256-2048 rows from 4096
      columns, -3 to -4 % at 1024 columns.
        - Scores on a 1/16 grid: +12 to +37 % at 1-24 rows (+1.8 to +6.2 µs),
      -8 to -16 % at 256-2048 rows.
        - Long rows (1 and 24 rows × 32768-131072 columns, three rounds):
      QSA-like scores 0 to -11 %, all zeros +24 to +28 % (+33 to +138 µs),
      1/16 grid +0.5 to +38 % (up to +68 µs at 131072 columns). The default
      against the baseline on these rows: -0.17 to +0.20 %.
        - Not measured: the cost in a server run. [INFERENCE] Small against
      the ITL: QSA only selects above 2048 tokens of context, and the worst
      measured decode call costs +20 µs up to 16384 columns.
    - Existing suites (`tests/kernels/test_top_k_per_row.py`,
    `tests/kernels/rocm/qwen4_exp/test_qsa_rdna2.py`,
    `tests/models/qwen4_exp/test_qsa_amd.py`): the same 29 cases fail on
    `a5a762999f`, on `45331fa50c` by default and with the knob on (29 failed,
    535 passed, 624 skipped in both of the latter runs, identical failing
    sets). None involves `sampler.cu`: 28 are the DeepSelect marker
    misfiring on gfx1030 (compute capability 10.3 matches
    `is_device_capability_family(100)`), one is a stale QSA test stub that
    rejects the `canonical_order` keyword. A test-side triage fixed all 29
    (`69c203dcad` for the marker, `a8821731e1` for the stub) and is merged at
    `1154318cf0`. The three suites now give 0 failed, 540 passed, 652
    skipped on `1154318cf0` on HIP 9, by default (4 min 7 s) and with the
    knob on (3 min 43 s).
    - The CUDA-gated radix tests (`test_top_k_per_row`, `_decode`,
    `_decode_large_vocab_size`, 62 cases), skipped on ROCm, pass with the
    skip lifted (the `skipif(not is_cuda())` markers removed): 62 passed on
    `45331fa50c` with the knob on (2 min 57 s) and by default (15 min 40 s),
    and 62 on `a5a762999f` (15 min 41 s, run by the triage). The default
    time matches the baseline's, as the identical code object predicts. The
    knob-on run is 5 times shorter, not because the kernel is faster (the
    microbenchmarks above show the opposite on tied rows). The slow cases are
    the `10LSBits` decode ones (22 fixed high bits, 1024 distinct values per
    row; 2048 batch × 8 steps): 142-250 s by default, 19-27 s with the knob;
    the `random` cases take the same in both modes (`--durations`).
    [INFERENCE] The suite's `compare_top_k_results` skips a slow
    value-by-value comparison when the selected set equals `torch.topk`'s,
    and the knob's set does: `torch.topk` keeps the lowest tied columns (64 of
    64 sampled `10LSBits` rows equal the lowest-tied set), the knob's
    selection equals it on 64 of 64, the default's on 45 of 64 at k = 2048
    and 41 of 64 at k = 3000. The comparison itself was not profiled.
    - Fixed (`2808f36487`, test `01e42e043c`, on `fix/topk-strided-split`):
    `top_k_per_row_decode` on column-major logits (`stride1 != 1`) with
    200,000 columns or more (the split-and-merge path) returned wrong
    indices, the same in the baseline, the default and the knob. Two
    defects. The strided loops handed the absolute column to code that adds
    `rowStart` again, so every block after the first counted its start twice:
    161 to 272 of 512 indices fell past the row (1034 of 2048 at k = 2048)
    and the set was wrong. The shortcut for a block no longer than k read
    `logits[rowIt + rowStart]` and ignored `stride1`, so rows under 10 k live
    columns selected the wrong set. The split kernels now pass the strided
    loops' column relative to `rowStart` (`stridedLoopIdx`) and the shortcut
    reads `(rowIt + rowStart) * stride1`; the row-major loops are as they
    were.
    - Evidence for that fix (HIP 9, exact-SHA trees).
    `test_top_k_per_row_strided.py` (12 row sets, a row-major control and
    the column-major copy, default and knob: 48 cases) gives 24 failed, 24
    passed on `1154318cf0` (every column-major case) and 48 passed with the
    fix. A random check (120 trials, 587 rows per layout, k = 512, 1024 and
    2048, widths 200,000-400,000) finds 587 of 587 column-major rows wrong
    on `1154318cf0` and 0 with the fix, default and knob on; no row-major row
    is wrong in either. The three suites give 0 failed with the fix (540
    passed, 652 skipped, by default and with the knob on) and
    `test_top_k_per_row_ties.py` 75 passed.
    - Cost on row-major rows. The single-block kernels, the ones QSA calls,
    are the same code at the same addresses (25 of 27 gfx1030 code objects
    are byte-identical) and the QSA grid differs by +-0.2 % (order noise).
    The split kernel keeps its row-major blocks (519 of 538 equal up to
    register names), but each strided loop is 16 B shorter (the double
    offset's induction variable is gone), which moves the large vectorized
    loops after it 16 B earlier modulo 64. On split shapes (1-256 rows ×
    204,800 and 262,144 columns, three rounds, both arm orders, two boots)
    the median of the 8 shapes is +0.63 to +0.74 % on QSA-like scores
    (relu4; -0.3 to +3.2 % per shape, slower in 31 of 32 shape runs), +0.0
    to +0.1 % on all zeros and -0.1 % on the 1/16 grid; with the knob on the
    medians stay within +-0.5 %. Layout control: the baseline rebuilt with
    `s_nop` padding in the strided loops only (the same instructions run;
    the large vectorized loops at the fix's offsets modulo 64) is slower by
    +1.4 to +1.6 % on relu4, +0.4 % on zeros and +0.5 to +0.65 % on the 1/16
    grid, while a rebuild with the baseline's own layout differs by +0.07 and
    -0.05 % on relu4. Layout alone moves this kernel by more than the fix's
    delta. [INFERENCE] The alignment of the large vectorized loops is the
    cause of the fix's delta; the control does not isolate it. The gfx950-only
    split kernels share the source change but did not run here. QSA passes
    row-major logits with fewer columns, so it does not reach this path.
    - Production: QSA only selects once the context exceeds 2048 tokens.
    [INFERENCE] Exact fp32 ties at the cut need equal positive scores or a
    cut inside the exact zeros, so they should be rare.

## Reproduce

Scripts and text results are in `bench_results/2026-11-04_determinism/`
(`scripts/`, `results/`). The trace dumps themselves (895 tensors, rank 0)
stayed on the box under `~/w4a8_runs/determinism/arms/`.

```bash
# Box: HIP 2-5 under the lease, exact-SHA trees ~/vllm-w4a8q-4a3d (4a3db777ae)
# and ~/vllm-det-d638 (d6384840c2 + the same built libraries).
cd ~/w4a8_runs/determinism
setsid nohup bash run_det_arms.sh > run_det_arms.log 2>&1 < /dev/null &   # T1 P0 T2 P1
# Perf-only rerun (P0r) while still holding the cards through
# Determinism-rerun.hip2..5 holder files:
WAIT_PID=<run_det_arms pid> ARMS=P0r setsid nohup bash run_perf_rerun.sh > run_perf_rerun.log 2>&1 < /dev/null &
# Analysis (CPU only):
export PYTHONPATH=$PWD/dtrace DTRACE_CHAIN_SITECUSTOMIZE= HIP_VISIBLE_DEVICES=
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES   # CPU only: set-but-empty, never -1 or unset
PY=~/Apps/vllm/venv-7.14.0_0.32.0/bin/python
$PY dgroups.py arms/T1/trace                       # forwards grouped by input hash
$PY dcompare.py arms/T1/trace --a 1 --b 5          # first SOURCE op, per rank
$PY ddump_diff.py arms/T1/trace --a 1 --b 5 --match layers
$PY dident.py arms/P1/ident.npz                    # repeat identity
$PY dident.py arms/T2/elem.npz --ref arms/T2/trace.npz
$PY dgrowth.py arms/T2/trace --a 10 --b 12 --offset 1024 \
    --lp arms/T2/noise_mix.npz:0,arms/T2/noise_mix.npz:1 --rows 1673,1990,2009
$PY dspec.py arms/P1                              # ITL, TPOT, MTP acceptance per cell
# Single-layer replay (one GPU):
REPRO_GPU=2 bash run_repro.sh
```

The tracer (`scripts/dtrace/dtrace_hooks.py`, loaded through
`sitecustomize.py` in every worker) is controlled by `DTRACE_DIR`,
`DTRACE_TOKENS=0` (trace every forward with at least `DTRACE_MIN_TOKENS`
rows), `DTRACE_DUMP_RANKS`, and two files that the hook rewrites between
phases: `dump_names.txt` (module name patterns to dump) and `perturb.txt`
(`name elem ROW COL ULPS` or `name noise FRAC SEED`).
