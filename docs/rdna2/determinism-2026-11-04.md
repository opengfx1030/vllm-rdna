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
  to +1.3 % at c=8, with prefill unchanged within noise.

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
| GDN prefill chain (prep, KKT, solve-WY, delta-h, o), gated RMSNorm, causal conv1d | `csrc/rocm/gdn_prefill_*_rdna2.cu`, Triton `_causal_conv1d_fwd_kernel` | no: fixed reduction order, no atomics | no (latent: a null-block slot returns early and leaves `out` unwritten; not hit) | — |
| QSA indexer, top-k, expand, sparse attention + split merge | `csrc/rocm/qsa_rdna2.cu`, `csrc/sampler.cu` | no; with at most 2048 tokens of context the top-k keeps every block, so QSA is exact causal attention | no (latent: radix top-k tie membership above 2048 tokens) | — |
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

## Latent issues found by the audit (not on this path)

None of these fires in the runs above; they are listed so that the knob's
guarantee is not over-read.

- `csrc/sampler.cu:305-312`: when several QSA block scores are exactly tied
  at the top-k threshold, `atomicAdd` on the shared counter decides which of
  them are kept. QSA only selects blocks once the context is longer than
  2048 tokens. [INFERENCE] Above 2048 tokens of context, exact ties can
  therefore make QSA nondeterministic even with the knob; the measurements
  here cover only 2048-token windows.
- `vllm/model_executor/layers/mamba/ops/causal_conv1d.py:122-127, 466`: the
  Triton conv kernel returns early for a sequence mapped to the null block
  and leaves its rows of `out` (`torch.empty_like`) unwritten. Only padding
  sequences map there.
- `csrc/rocm/qsa_metadata_rdna2.cu:147`: the circular-slot rule accepts
  physical block 0 (`>= 0`). [INFERENCE] Block 0 is vLLM's null block, so a
  padded or unmapped request would write its slots there.
- With `VLLM_RDNA2_MOE_FP32_ACCUM=1`, the fp32 scratch is cached per
  (rows, n, device) and never freed, so every distinct token count adds a
  buffer.
- `csrc/rocm/fa_rdna2.cu:4559-4571` (int8 prefill split-K): the comment
  promises persist slots distinct from the fp16 split-K path, but the code
  uses the same `g_pref_Op/Mp/Lp` globals as `fa_rdna2_prefill_paged_varlen_splitk`
  (3873-3878). It also still says that the fp16 wrapper's
  `[N, H_q, kv_splits, D]` layout corrupts memory, although that wrapper's
  kernel now indexes exactly that layout. [INFERENCE] Sharing the slots only
  matters for a process that runs both paths, where one can reallocate a
  buffer the other captured in a graph.

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
export PYTHONPATH=$PWD/dtrace DTRACE_CHAIN_SITECUSTOMIZE= HIP_VISIBLE_DEVICES=-1
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
