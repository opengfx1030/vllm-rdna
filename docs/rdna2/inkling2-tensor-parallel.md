# Inkling-2-Mini tensor parallelism (TP = 1, 2, 4) on the HIP serving path

`Inkling2LoopForConditionalGeneration` serves with `--tensor-parallel-size 2` and `4` on
the same native kernels as TP=1 (`csrc/rocm/inkling2_*_rdna2.cu`, the expert GEMMs); the
only new kernel is the vocabulary-shard embedding lookup (`inkling2_glue_embed_lookup`),
no torch compute in the forward.  FULL / PIECEWISE graphs at TP = 2 / 4 and the DSpark
draft at TP = 2 were added on the merged tree (`inkling2-serving.md`,
`inkling2-dspark.md`); the text below is the eager-era record.  Code:
`vllm/models/inkling2_loop/amd/{parallel,loader,attention,moe,glue,linear_pack,model}.py`,
tests `tests/models/inkling2_loop/test_tensor_parallel.py`, tools
`tools/rdna/inkling2/{tiny_vllm_check,real_check,tp_step_bench,bench_tp_comm}.py`.
Run it with fp16 and the one-shot HIP all-reduce:

```bash
VLLM_RDNA_AR=1 vllm serve ~/hfcache/inkling2-mini --dtype float16 \
  --tensor-parallel-size 2   # or 4; HIP_VISIBLE_DEVICES selects the cards
# the config's maximum length (1,048,576) on TP=2 needs a small chunk, see "Memory":
#   --max-model-len 1048576 --max-num-batched-tokens 64
```

`VLLM_RDNA_AR=1` (`rdna_all_reduce.py`, opt-in) keeps decode-sized messages on the HIP
one-shot kernel and sends larger ones to RCCL (`VLLM_RDNA_AR_MAX_KB` /
`VLLM_RDNA_AR_ONESHOT_KB`, default 64 KiB each); without it every collective is RCCL.

## Layout

The unit of sharding is the **KV-head group**: 24 q heads / 8 KV heads (GQA group 3) give
12 + 4 per rank at TP=2 and 6 + 2 at TP=4.  `parallel.check_supported` refuses every other
degree with the reasons (TP=8: three q heads per rank, the attention kernels need an
even count).

| tensor | TP > 1 |
|---|---|
| attention input projections (q, output gate, relative bias, sink, k, v, dynamic conv taps), per-head `q` / `rel` scales, k / v short-conv weights | row slice of this rank's heads (`loader.ATTENTION_SPLIT_DIM`) |
| `o_proj` | column slice (K); output is an **fp32 partial** `[T, H]`, all-reduced |
| KV cache, k / v conv state | local heads only |
| router | replicated (every rank takes the same decision) |
| expert `w13` (gate / up, interleaved rows) | rows `[2 lo, 2 hi)` of the intermediate split, then de-interleaved as at TP=1 |
| expert `w2`, scales | split along the intermediate dimension (K); fp16 partial, all-reduced |
| `lm_head` and embedding | vocabulary split: padded to a multiple of 64, even shards; `lm_head` padding logits `-inf`; the embedding lookup returns zero rows for the ids of other ranks, one fp16 all-reduce adds the owner's row back (exact), then `embed_norm` runs on the full rows |
| residual stream, block states, attention-residual queries, message gates, norms, the short convolutions of the attention / MLP outputs and their conv state, vision tower | replicated |

The local MoE intermediate is 576 at TP=2 and 288 at TP=4 (kernels need `K % 32 == 0`:
`test_moe.py` runs both shapes on the GPU against the dequantised reference, 16 tests).

**Two all-reduces per layer visit** are the whole communication of the layer stack (96
per decode step: 48 visits): after attention (fp32, feeds the replicated short
convolution `sconv_static`, which takes fp32 or fp16) and after the MoE (fp16).  The
embedding adds a 97th (fp16 rows of the tokens).  The consumers are nonlinear, so
nothing can be deferred or fused.  Everything between is replicated compute on
bit-identical all-reduced inputs, which is what keeps the router decisions of all ranks
equal.  Logits: every rank computes its vocabulary shard and `compute_logits`
all-gathers it (blocks of 64 rows straight into the final contiguous `[M, vocab]`);
`greedy_token_ids` gathers the per-block argmax partials (value + int32 id bits in one
buffer) and reduces them with `inkling2_linear_argmax_reduce`, the logits are never
gathered.

The embedding at TP > 1 is `inkling2_glue_embed_lookup` (this rank's rows, zeros for
other ranks' ids, exact copy of fp16 rows) into a workspace buffer, the all-reduce, and
`inkling2_glue_rms_norm(..., round_fp16=True)`, which is bit-identical to the fused
`inkling2_glue_embed_norm` of TP=1 (tested).  TP=1 keeps the fused op.

Refused / not done at TP > 1: the probabilistic (sampled) DSpark draft raises
`NotImplementedError` (greedy drafting is vocabulary-shard aware, see "DSpark at TP > 1"
below); the vision tower is replicated (every rank encodes the image, 0.41 GiB per rank at
every TP degree).

### KV and conv state

| | TP=1 | TP=2 | TP=4 |
|---|---:|---:|---:|
| page size | 256 KiB | 128 KiB | 128 KiB |
| bytes per block id (8 layers share it) | 2 MiB | 1 MiB | 1 MiB |
| block size, windowed / full attention | 64 | 64 | 128 |
| block size, conv groups (6 of the 12 groups) | 8 | 4 | 4 |
| blocks one request needs, `L` tokens, chunk 1,024 | 945 + `L`/64 | 1,713 + `L`/64 | 1,633 + `L`/128 |
| of which conv groups | 780 | 1,548 | 1,548 |

The attention KV shrinks with TP (4 KiB per token and layer slot at TP=1, 2 KiB at TP=2,
1 KiB at TP=4), the **conv state does not**: it holds the replicated streams, 32 KiB per
token and layer slot at every TP degree and a window of about 1,030 tokens per request,
about 1.5 GiB per request on every rank at any TP.  A single request therefore pays
1.6 - 1.85 GiB of constant state before its first KV page (`conv groups` row), which
limits concurrency at 8,192 tokens (planner: 3.5x at TP=1, 7.9x at TP=2, 10.5x at TP=4).
The conv groups also grow with the chunk: they hold the window plus one chunk (1.6 MiB
per chunk token at TP=2), so a smaller `max_num_batched_tokens` lowers the need of a
request.  Shrinking the state is the conv-state work of the `feat/inkling2-conv` branch;
sharding it would need the short convolutions of the attention / MLP streams to be
sharded as well.

> **Superseded on the merged tree** (`feat/inkling2-merge`, section "merged tree" of
> `inkling2-port-spec.md`): this section and the memory numbers below measured the old
> per-token fp32 conv layout.  With the fp16 conv ring of `feat/inkling2-conv` a rank's
> conv page is `[8 / tp, 3, W]` fp16 with `W` = 832 / 1408 / 2560 (39,936 / 33,792 /
> 30,720 B per layer and 64-token block at TP = 1 / 2 / 4), a request pays 34 blocks per
> conv group (1 / 2 / 3 groups: 51 - 68 MiB at mnbt 1024) instead of about 1.5 GiB, and the
> model's 1,048,576 tokens boot on TP=2 with `--max-num-batched-tokens` 184 (the largest
> tried; text only: the vision tower's 0.6 GiB per rank does not fit) and on TP=4 with 4096
> (2.17x concurrency).  The chunk-64 requirement and "the conv state does not shrink with
> TP" are the old layout's.

## Validation

**CPU** (`test_tensor_parallel.py`, 128 tests, 177 s; the existing suite of
`tests/models/inkling2_loop` and `test_inkling2_loop_moe.py` passes together with it:
582 tests pass, 92 GPU-only ones skipped, 522 s: the 454 existing tests and these 128):

* sharding math: partition helpers, `lm_head` shard tiling at TP=1/2/4 (padding rows,
  empty tail shards), the supported degrees of the real config (1, 2, 4; 8 refused),
  attention slices reassemble exactly and are views, `w13` slices de-interleave to the
  rank's gate and up rows, `w2` slices split K, TP=1 slicing is the identity;
* the per-rank model weights reassemble to the TP=1 model (the embedding and the
  `lm_head` as vocabulary shards), the owners hold only the rank's heads, the sharded
  embedding returns the TP=1 rows of every id (shard edges, the table's ends);
* a thread-per-rank in-process group runs the paged model on 2 and 4 ranks (tiny1 and
  tiny8 checkpoints, 24 q / 8 KV heads, head_dim 128, router bias x100, 8 experts top-2)
  against the fp32 transformers reference: prefill of 5 / 40 / 70 tokens, chunked prefill
  (1 / 7 / 33), cached decode, a batch of three requests with chunked prefill and
  decode mixed, a forward without metadata (profiling run), block-wise logits gathers
  (blocks of 1 and 9 rows): max |dlogit| <= 1.7e-4 (`FP32_TOL`), bitwise equal on every
  rank;
* greedy argmax over the sharded head with ties across shards (the lowest id wins).

Mutation checks (scratch copy of the tree, each mutant must fail the suite): attention
all-reduce skipped, MoE all-reduce skipped, `w13` rows not interleaved, `o_proj` split on
the wrong dimension, `lm_head` shard offset by one, `lm_head` padding not masked; for the
embedding the lookup offset by one, the shard's upper bound missing, the load offset by
one, the all-reduce skipped, foreign ids not zeroed, the table not sharded: **12 of 12
killed**.  The lookup kernel has its own 17 GPU tests, 10 of 10 kernel mutants killed and
a six-architecture compile (`inkling2-glue-kernels.md`, 7.5); the whole glue test file
passes on the TP build (212 tests).

**GPU engine, tiny checkpoints** (HIP kernels, fp16, eager, `tiny_vllm_check.py
--tensor-parallel-size N --against TP1.npz`; TP=1 against TP=1 is bitwise identical on
these: two experts per token commute; run before the embedding was sharded).  Max / mean
|dlogp| against the TP=1 run:

| checkpoint | scenario | TP=2 rdna_ar | TP=4 RCCL | TP=4 rdna_ar |
|---|---|---:|---:|---:|
| tiny1 (6 layers) | last position | 0.046 / 0.0092 | 0.040 / 0.0089 | 0.039 / 0.0088 |
| | decode (8 steps) | 0.057 / 0.0105 | 0.059 / 0.0101 | 0.059 / 0.0101 |
| | batch | 0.057 / 0.0108 | 0.058 / 0.0109 | 0.083 / 0.0104 |
| tiny8 (8 layers, loop x3) | last position | 0.119 / 0.0263 | 0.136 / 0.0266 | 0.135 / 0.0223 |
| | decode | 0.230 / 0.0263 | 0.209 / 0.0263 | 0.221 / 0.0237 |
| | batch | 0.230 / 0.0263 | 0.189 / 0.0252 | 0.176 / 0.0242 |

The greedy ids of all prompts equal the TP=1 ids (8 of 8 steps) in every run, and the
mean |dlogp| against the HF fp32 reference is the same as TP=1's (tiny1 0.0107 at TP=1,
0.0113 at TP=2, 0.0106 / 0.0110 at TP=4; tiny8 0.034, 0.031, 0.029 / 0.032).  RCCL and
`rdna_ar` give bitwise identical results at TP=2.  Chunked prefill with prefix caching
(`--chunked --prefix-caching`, same-step hit, cross-step hit, hit versus cold) passes at
TP=2 on both checkpoints (RESULT PASS, hit versus cold max |dlogp| 0.0).  The check
`batch/logprobs` (a request alone versus in a batch, tolerance 0.05 / 0.15 chosen at
TP=1) is a rounding-noise test and the only one that does not scale: 0.035 / 0.051 at
TP=1, 0.039 / 0.059 at TP=2, and at TP=4 0.052 (rdna_ar) and 0.069 (RCCL) on tiny1,
0.105 (rdna_ar, within 0.15) and 0.164 (RCCL, over 0.15) on tiny8.  The one-shot kernel
sums in fp32 in a fixed rank order; RCCL's ring reduction rounds at every hop in the
message dtype [INFERENCE: not inspected], which fits the larger TP=4 difference.  Every
other check passes (`RESULT FAIL` of those runs is that line only).

**Real checkpoint** (`~/hfcache/inkling2-mini`, `max_model_len` 8,192, chunk 1,024,
`gpu_memory_utilization` 0.92, fp16, eager, `VLLM_RDNA_AR=1`; the TP=1 runs A and B are
the noise floor, see below):

| | TP=1 | TP=2, replicated embedding | TP=2, sharded embedding | TP=4, replicated embedding | TP=4, sharded embedding |
|---|---|---|---|---|---|
| answer to both prompts (32 tokens, greedy) | reference | identical (32 / 32) | identical (32 / 32) | identical (32 / 32) | prompt 0 32 / 32; prompt 1 16 / 32 (the near-tie of token 16 that separates the TP=1 runs A and B) |
| first-step top-8 sets shared with TP=1 | | 8/8, 8/8 | 8/8, 8/8 | 8/8, 8/8 | 7/8, 8/8 |
| first-step max \|dlogp\| of the shared top-8 (prompt 0 / 1), limit `2 x floor + 0.1` = 0.245 / 0.209 | floor (run B): 0.073 / 0.055 | 0.060 / 0.069 | 0.231 / 0.148 | 0.377 / 0.182 and 0.369 / 0.154 (two runs) | 0.334 / 0.054 |
| teacher-forced logprobs of 799 positions: mean \|dlogp\| / p99 / max against TP=1 run A (limit on the mean 0.0898) | floor (run B): 0.0399 / 0.313 / 0.631 | 0.0439 / 0.300 / 0.840 | 0.0425 / 0.303 / 0.476 | 0.0473 / 0.328 / 0.531 | 0.0515 / 0.370 / 0.547 |
| mean logp of those positions | -1.9860 (A), -1.9828 (B) | -1.9805 | -1.9850 | -1.9786 | -1.9811 |
| prompt of 1,510 tokens, prefix cache cold / warm | 1.95 s / 0.12 s | 1.18 s / 0.08 s, same ids | 1.17 s / 0.08 s, same ids and top-1 (top-8 tail, hit vs cold 1.58, cold vs cold 1.22) | not run | not run |
| image request | answered (part A) | answered ("a red circle centered on a blue ...") | answered (same words) | not run | not run |

At TP=1 two runs of one configuration differ (fp16 CAS atomics of the expert GEMMs, see
part A, finding 2): run B diverges from run A at token 16 of the second prompt, a
near-tie.  The first step is a degenerate test (`<|message_model|>` has logprob 0.0, the
top-8 are tail tokens 19 to 22 nats below), so it is much more sensitive than the text.
At TP=4 prompt 0 differs from TP=1 by 0.377 and 0.369 (two runs) where the single TP=1
pair gives a floor of 0.073, which looked like a TP=4 error.  **It is noise**: two TP=4
runs differ from each other on the same step by up to 0.629 (prompt 0, 7 of 8 top-8
tokens shared; prompt 1 0.032), more than TP=4 differs from TP=1.  The teacher-forced
comparison over 799 positions of this document settles it: TP=4 against TP=1 mean
|dlogp| 0.0473 (limit 0.0898), TP=4 against a second TP=4 run 0.0479, and every
all-reduce configuration agrees at the same level (gate 256 KiB: 0.0423, p99 0.317, max
0.599; RCCL only: 0.0422, p99 0.313, max 0.703); the mean logp of the positions is
-1.9786 / -1.9856 (two TP=4 runs), -1.9835 (gate 256 KiB), -1.9794 (RCCL) against
-1.9860 / -1.9828 at TP=1.  The 32 generated ids equal TP=1 on both prompts in every
TP=4 run (the second run of prompt 1 diverges at token 16, the same near-tie that
separates the TP=1 runs A and B).  So no bisect and no fp32 MoE reduction was needed:
the fp16 partial sums of the MoE (and the replicated routers deciding on them) change
with the summation order exactly as they change run to run at TP=1 [INFERENCE for the
mechanism, measured for the size].

The third TP=4 configuration, with the vocabulary-sharded embedding (default gate), gives
the same picture: first step 0.334 on prompt 0 (7 of 8 top-8 shared) and 0.054 on prompt
1, 32 of 32 ids on prompt 0 and the near-tie divergence at token 16 on prompt 1, and the
teacher-forced comparison passes (mean |dlogp| 0.0515 against the limit 0.0898, p99
0.370, max 0.547, mean logp -1.9811); a second run of this build gives 0.367 / 0.625 on
the first step and 0.0491 (p99 0.378, max 0.565, mean logp -1.9822).

**The first-step criterion of `real_check.py` is statistically meaningless: its floor is
one pair.**  Two more TP=1 runs (C, D) against run A fail it as well: first step 0.426 /
0.626 (C) and 0.337 / 0.081 (D) against the 0.073 / 0.055 of the pair A / B that set the
limit, with teacher-forced means 0.0454 (C) and 0.0487 (D).  All 15 pairs of the four TP=1
runs and the two sharded-embedding TP=4 runs, mean |dlogp| over the 799 positions:

| pairs | n | range of the mean | average | p99 range | max range |
|---|---:|---|---:|---|---|
| TP=1 against TP=1 | 6 | 0.0399 - 0.0519 | 0.0470 | 0.31 - 0.41 | 0.51 - 0.79 |
| TP=4 against TP=1 | 8 | 0.0331 - 0.0515 | 0.0468 | 0.25 - 0.39 | 0.48 - 0.72 |
| TP=4 against TP=4 | 1 | 0.0356 | | 0.27 | 0.79 |

TP=4 is as far from TP=1 as TP=1 is from itself.  The first-step deviation of TP=4 on
prompt 0 (0.33 - 0.38 in four runs) is the size of TP=1's own (0.34 and 0.43 in two of
three pairs), and the shard returns exactly the rows of the fused lookup (tested bit for
bit), so the difference to the replicated-embedding runs is run-to-run noise of the
expert accumulation as well.  The teacher-forced limit (2 x floor + 0.01 = 0.0898) holds
in all 15 comparisons; the first-step limit would need a floor of several TP=1 pairs.

## Performance (real checkpoint, eager)

Wall time of `LLMEngine.step()` (`tp_step_bench.py`), prompts of 22 tokens, median ms,
chunk 1,024.  **Indicative**: the box (load average 7 - 13) ran the other agents' GPU jobs
on the neighbouring cards at the same time; no KFD `evicted_ms` check was made.

| | TP=1 | TP=2 rdna_ar | TP=2 RCCL | TP=2 no all-reduce | TP=2 rdna_ar, sharded embedding | TP=4 rdna_ar | TP=4 no all-reduce | TP=4 rdna_ar, sharded embedding, chunk 256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decode step, batch 1 | 23.40 | **18.90** | 20.05 | 17.13 | 18.92 | **17.54** | 14.28 | 16.77 |
| decode step, batch 4 | 36.86 | **26.65** | 28.09 | 24.84 | 26.96 | **22.28** | 19.19 | 21.58 |
| prefill, 22 tokens | 76.8 | 50.5 | 50.1 | 51.1 | 50.6 | 41.5 | 38.2 | 41.4 |
| prefill, 4 x 22 tokens | 175.4 | 104.0 | 104.0 | 104.9 | 104.8 | 74.8 | 67.4 | 74.0 |
| prefill, one 1000-token prompt | 1316 (760 tok/s) | 753 (1328) | 754 | 728 | not run | 488 (2051) | not run | not run |

"No all-reduce" replaces every all-reduce of the model by the identity (outputs are
meaningless, it is a timing experiment): the all-reduce costs 1.77 ms per decode step at
batch 1 with `rdna_ar` at TP=2 (96 calls, 18 us each) and 2.92 ms with RCCL (30 us each),
1.81 and 3.25 ms at batch 4; at TP=4 3.26 ms at batch 1 and 3.09 ms at batch 4 with
`rdna_ar` (34 us a call: the call itself is 14 - 27 us, the rest is the wait for the
slowest of four launch-jittered ranks).  The vocabulary-sharded embedding (lookup +
all-reduce + norm instead of one fused kernel) changes the TP=2 step by +0.02 ms
(batch 1) and +0.31 ms (batch 4), inside the run-to-run spread.  At batch 16 (TP=2, one
run each, range 42 - 67 ms) the median is 52.3 ms with the default gate, **49.1 ms with
the gate raised to 256 KiB** (`VLLM_RDNA_AR_MAX_KB=256 VLLM_RDNA_AR_ONESHOT_KB=256`: the
fp32 message of 16 tokens is 144 KiB) and 50.7 ms with RCCL.

TP=4 with the sharded embedding (last column; chunk 256, as the replicated-embedding
TP=4 run at chunk 256 measured 16.94 / 21.48 ms): the same step time within the noise.
The chunk-1,024 run of the same session measured 25.13 ms (batch 1) and 27.06 ms (batch
4) with spikes to 45 ms while another job was compiling on the box (load 7 - 13): that
run is disturbed and not repeated [INFERENCE for the cause]; the chunk-1,024 TP=4
figures of the table are the ones of the replicated-embedding build.

GPU busy time per decode step by kernel family (rank 0, torch profiler, batch 1, ms;
kernels per step 779 / 877 / 877; measured before the embedding was sharded):

| | TP=1 | TP=2 | TP=4 |
|---|---:|---:|---:|
| attention | 1.86 | 1.91 | 1.94 |
| glue (norms, gates, conv, residual) | 1.45 | 1.43 | 1.48 |
| linear (`attention_input`, `o_proj`, `lm_head`, router) | 8.49 | 4.52 | 2.55 |
| experts | 7.35 | 4.68 | 3.67 |
| other | 0.88 | 0.89 | 0.87 |
| `lm_head` all-gather (RCCL) | | 0.06 | 0.30 |
| all-reduce kernels (`rdna_ar`, incl. waiting for the other ranks) | | 5.27 | 8.39 |
| busy total | 20.04 | 18.75 | 19.20 |

The attention, glue and "other" families (4.2 ms) are replicated and do not shrink; with
the launch-bound eager step this is why TP=2 gains 1.24x and TP=4 1.33x at batch 1
(1.38x and 1.65x at batch 4).  The all-reduce kernel's own duration is 6 - 9 us at the
minimum (`min` 6.2 us at TP=2, 8.1 us at TP=4); the family totals above are dominated by
the wait for the slower rank (eager launch jitter), which a graph removes: the
microbenchmark of the collectives (`bench_tp_comm.py`, 106 cases, all results correct)
measures the call itself:

| message | TP=2 rdna_ar graph / eager | TP=2 RCCL graph / eager | TP=4 rdna_ar graph / eager | TP=4 RCCL graph / eager |
|---|---:|---:|---:|---:|
| fp32 `[1, 2304]` (attention, batch 1) | 13.3 / 19.5 | 29.4 / 25.5 | 21.5 / 26.4 | 74.7 / 53.0 |
| fp32 `[4, 2304]` | 14.6 / 20.6 | 27.3 / 27.4 | 26.6 / 32.5 | 68.5 / 57.1 |
| fp32 `[16, 2304]` (144 KiB) | 22.8 / 30.1 | 33.6 / 36.2 | 62.6 / 67.7 | 72.5 / 63.6 |
| fp16 `[1, 2304]` (MoE, batch 1) | 11.0 / 18.2 | 22.0 / 24.7 | 14.1 / 20.2 | 68.8 / 64.0 |
| fp32 `[1024, 2304]` (prefill chunk) | | 737 / 740 | | 1060 / 1066 |

(us per call.)  Both backends are capture-safe (graph rows).  `all_gather` of the argmax
partials `[2, 64]` costs 10.8 us (graph) at TP=2, 16.5 us at TP=4; of the logits `[1,
100032]` 48.6 us (graph) / 116.7 us (eager) at TP=2.  There is no TP=4 run with RCCL
only; from the call times above (eager, batch 1: 48 fp32 + 48 fp16 calls) it would cost
5.6 ms per step against 3.3 ms measured for `rdna_ar` [INFERENCE].

## Memory and the longest context

Per GPU ("Model loading took", includes the glue workspace, 0.28 GiB for a 1,024-token
chunk and 0.03 GiB for 96 tokens): TP=1 18.08 GiB; TP=2 9.69 GiB, **9.26 GiB with the
sharded embedding** (-0.43); TP=4 5.49 GiB, **4.84 GiB with the sharded embedding**
(-0.65; 4.63 GiB at chunk 256).  What stays replicated
per rank is the vision tower (0.41 GiB), the workspace, and a few tens of MiB of gates,
queries, norms and routers.  The engine sizes the extensible KV cache after the profiling
run as `min(requested - resident, headroom - margin) - transient`: the margin is a fixed
3 GiB on ROCm (`KV_CACHE_MARGIN_FLOOR_BYTES` in `extensible_kv_cache.py`), the transient
is what the profiling run allocated at its peak, and it scales with the chunk
(`max_num_batched_tokens`): the profiling run computes the logits of every token of a
chunk (`[1024, 200058]` fp32 = 0.76 GiB at TP=1) [INFERENCE for the composition].  Raising
`gpu_memory_utilization` to 0.99 does **not** add blocks at any TP (the cap is the device
headroom, not the request; TP=1 3,744 blocks at 0.92 and 0.99, TP=2 12,512 and 12,512).

The first sharded `compute_logits` held three copies of the full logits at the peak
(local shard, gather output, reshape): 1.91 GiB at TP=2 instead of 0.77 GiB at TP=1.  It
now gathers blocks of 64 rows into the final buffer (peak 0.93 GiB at TP=2, chunk 1,024),
which is worth 2,000 more blocks (+1.95 GiB of KV) at TP=2.

Free KV blocks and the longest single request the planner admits
(`get_max_concurrency_for_kv_cache_config`; the engine refuses a larger `max_model_len`).
The blocks a request needs at 1,048,576 tokens depend on the chunk (conv groups: window
plus one chunk):

| | chunk | weights per GPU | blocks held | KV per GPU | longest single request | blocks needed at 1,048,576 |
|---|---:|---:|---:|---:|---:|---:|
| TP=1 | 1,024 | 18.08 GiB | 3,744 | 7.31 GiB | 179,136 | 17,329 |
| TP=1 | 256 | | 4,168 | 8.14 GiB | 246,976 | |
| TP=2, replicated embedding (first `compute_logits`) | 1,024 | 9.69 GiB | 12,512 | 12.22 GiB | 691,136 | 18,097 |
| TP=2, replicated embedding | 1,024 | 9.69 GiB | 14,512 | 14.17 GiB | 819,136 | 18,097 |
| TP=2, replicated embedding | 256 | | 15,936 | 15.56 GiB | 987,840 | 16,885 |
| TP=2, sharded embedding | 1,024 | 9.26 GiB | 14,960 | 14.61 GiB | 847,808 | 18,097 |
| TP=2, sharded embedding | 256 | 9.02 GiB | 16,368 | 15.98 GiB | 1,015,488 | 16,885 |
| TP=2, sharded embedding | 128 | 8.99 GiB | 16,624 | 16.23 GiB | 1,044,736 (engine) | about 16,680 (16.29 GiB) |
| TP=2, sharded embedding | 96 | 8.97 GiB | 16,752 | 16.36 GiB | 1,056,064 | 16,635 |
| TP=4, replicated embedding | 1,024 | 5.49 GiB | 19,200 | 18.75 GiB | 2,248,576 | 9,825 |
| TP=4, replicated embedding | 256 | 5.28 GiB | 20,592 | 20.11 GiB | 2,578,048 | 8,643 |
| TP=4, sharded embedding | 1,024 | 4.84 GiB | 19,872 | 19.41 GiB | 2,334,592 | 9,825 |
| TP=4, sharded embedding | 256 | 4.63 GiB | 21,264 | 20.77 GiB | 2,664,064 | 8,643 |

The 1,048,576-token config maximum **fits at TP=4** at any chunk (it needs 9,825 blocks
at most), and **on TP=2 with the sharded embedding and a small chunk**.  Boot attempts at
`max_model_len` 1,048,576 on TP=2 (engine message "estimated maximum model length"):
chunk 256 refuses (needs 16.49 GiB, holds 15.98, 1,015,424 tokens), 192 refuses (16.39 /
16.11, 1,030,080), 128 refuses by 0.06 GiB (16.29 / 16.23, 1,044,736), 112 refuses by 0.01
GiB (16.27 / 16.26, 1,048,320), **96 boots** (`tp_step_bench.py`, 1.01x concurrency at
1,048,576 tokens) and 64 boots and answers (`real_check.py`, 32 / 32 ids equal to TP=1
on both prompts; first-step top-8 of prompt 0 0.247 against the 0.245 limit, the noise
discussed above).  `real_check` keeps the default async scheduling and needs 16.39 GiB at
chunk 96 (16.36 held, refuses with 1,046,464 tokens), the same as the benchmark engine
(async scheduling off) at chunk 192 [INFERENCE: the async scheduler budgets two chunks of
conv window], so **the serving recipe for 1,048,576 at TP=2 is `--max-num-batched-tokens
64`** (chunked prefill at 64 tokens per step: slower prefill; the prefill speed at that
chunk was not measured cleanly).  Before the embedding was sharded the gap at chunk 256
was 949 blocks (16,885 needed, 15,936 held, 0.93 GiB); an earlier version of this note
compared the chunk-1,024 need (18,097) with the chunk-256 cache and reported 2.1 GiB.
At TP=1 the longest single request is 179,136 tokens (chunk 1,024), 246,976 with chunk
256 (planner; not booted at that length).

TP=4 boots at `max_model_len` 1,048,576 (sharded embedding, chunk 256, `tp_step_bench.py`
with 4 decode steps): 21,216 blocks (20.72 GiB), 2,573,942 tokens, 2.45x concurrency at
1,048,576 tokens per request, boot in 28 s.  The shard saved 0.65 GiB per rank at TP=4
(5.49 to 4.84 GiB at chunk 1,024), as predicted from the table size (0.86 GiB x 3/4).

TP=2 at `max_model_len` 1,048,576 with `--max-num-batched-tokens 64` also boots with
prefix caching enabled (`real_check.py --prefix-caching`, 47 s): both prompts answered, a
608-token prompt prefills in 0.87 s cold and 0.11 s warm.  Its warm run generates other
ids than its cold run (top-8 max |dlogp| hit versus cold 3.31, cold versus cold again
1.09).  The same run at `max_model_len` 8,192 and chunk 64 shows the same at TP=1 (cold
1.34 s, warm 0.11 s, hit versus cold 0.81, cold versus cold again 1.65) and at TP=2 (0.84
s, 0.07 s, 2.78 and 5.04): the continuation of that prompt has near-ties and the
cold-versus-cold spread is as large as the hit-versus-cold difference, so the
difference is not a TP, chunk or length effect [INFERENCE for the cause; measured: the
three pairs].

Levers for more room on TP=2, none needed for 1,048,576 any more [INFERENCE]: the
conv-state work (about 1.5 GiB per request at any TP), a smaller planner margin
(`KV_CACHE_MARGIN_FLOOR_BYTES`, 3 GiB on ROCm), a profiling run that computes the logits
of the scheduled sequences instead of every token of the chunk (runner code).

## Open items

* **TP=4 prefix cache, image request and the profiler table** were not repeated with the
  sharded embedding; the TP=4 run of this build covers the numerics (NLL, first step), the
  step times (chunk 256), the planner (chunks 1,024 and 256) and the 1,048,576 boot.  The
  rank-0 profile table above and the no-all-reduce column are of the replicated-embedding
  build.
* **DSpark at TP > 1** (greedy draft) is implemented in `amd/dspark.py`: the draft's
  weights, K/V and taps are replicated on every rank, the anchors' embedding is the shard
  lookup above plus one all-reduce of the `[rows, hidden]` block, and the folded head runs
  over this rank's vocabulary rows: `inkling2_dspark_argmax_partial` feeds an all-gather
  and the reduce kernel of `greedy_token_ids` (ties to the lowest id).  Its engine runs
  with graphs are in `inkling2-dspark.md` ("Merged tree").  The sampled draft would need
  the full-vocabulary draft logits on every rank and still raises
  `NotImplementedError`.
* **Graphs**: done on the merged tree (`inkling2-serving.md`): the collectives are
  capture-safe, and the FULL / PIECEWISE graphs capture the two all-reduces per layer
  visit plus the embedding's.
* The step is launch- and wait-bound at TP=4 (rank 0 kernels without the all-reduce add
  up to 10.8 ms of the 17.5 ms wall step): a graph is where TP=4 gains most.
* Vision at TP > 1 works (image request answered at TP=2) but encodes the image on every
  rank and keeps the tower (0.41 GiB) on every rank.
* TP=8 is refused (see Layout).
* `real_check.py`'s first-step limit rests on one floor pair (`--floor` takes one dump)
  and false-fails TP=1 against TP=1 (see Validation); it should take several floor runs.
  The teacher-forced limit is the decision criterion.
* After a run the process can abort at interpreter exit (`terminate called without an
  active exception`, rc -6, after all output and the PASS line; twice in about 50 engine
  runs, the TP=2 NLL run of the replicated build and the first TP=4 run of the sharded
  one): an engine-shutdown race, not isolated.
