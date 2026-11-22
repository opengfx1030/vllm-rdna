# Inkling-2-Mini DSpark draft on gfx1030 (V620)

Native-HIP port of the Inkling-2-Mini draft (`mtp.safetensors`, architecture
`Inkling2DSparkModel`) for the V2 model runner of `opengfx1030/vllm-rdna`.  Everything
that runs at serving time is a HIP kernel in `csrc/rocm/inkling2_dspark_rdna2.cu` (plus the
repo's existing dense RDNA2 GEMM designs); PyTorch / CPU code exists only as the offline
fp64 oracle in tests and tools.  Numbers below are copied from runs on the box
(`par1-cs25`, b4 HIP 2, gfx1030) unless marked `[INFERENCE]`.

## Status

| Item | State |
| --- | --- |
| Draft model, loader, speculator (`amd/dspark.py`) | written; HIP-vs-fp64-oracle run on the REAL `mtp.safetensors` + real target head passes (numbers below) |
| 15 native ops (`inkling2_dspark_*`) plus one Python helper | written, registered (ops.h, torch_bindings.cpp, CMake whitelist, `_custom_ops.py`), reference-tested on b4 |
| Exact-probability gate of the rejection path | CPU oracle + negative controls pass; the production HIP rejection sampler passes on b4 (4 GPU cases) |
| Mutation checks of the kernels | 10 of 10 mutants killed (table in Results) |
| Official `build_lq.sh` build of the new TU | built (exit 0 after 804 s, official `-O2` flags); all 15 ops register in the official `_rocm_C`; on that build (plain pytest on the in-tree `_rocm_C.abi3.so`, b4 HIP 2) `test_inkling2_dspark_rdna2.py` gives 130 passed (48.19 s) and `test_dspark_model.py` + `test_dspark_verifier.py` give 31 passed (37.93 s) with the same draft-vs-oracle errors as the dev `-O3` build |
| Target tap hook in `amd/model.py` | applied and run on the real target through the runner's EAGLE3 interface (part B): `supports_eagle3` was false because the class lacked the `SupportsEagleBase` flags, and the draft config listed the final-output id 48 as a layer to tap (the hook raised); both fixed, CPU tests added |
| Runner-owned Triton (`prepare_dflash_inputs`) | HIP op written and tested against a Python port of the kernel; the dispatch is applied in `dflash/speculator.py` and gated by `DFlashSpeculator.uses_hip_prepare_inputs` (True only for `Inkling2DSparkSpeculator`), so every other DFlash / DSpark draft keeps the Triton kernel |
| V2 runner end-to-end (model load + `propose` through `LLM(...)`) | **run** on the real checkpoint, `FULL_AND_PIECEWISE`, `num_speculative_tokens=7`, TP=1: acceptance, speed and greedy equality in "Part B: the draft on the V2 runner" of `inkling2-port-spec.md` |
| FULL HIP-graph capture inside the runner | done: `DFlashCudaGraphManager` captures the draft step (graph memory 1.95 GiB with the target graphs) |
| TP > 1 | written and unit-tested, **not run through the engine**: the draft is replicated on every rank, the embedding of the anchors is the target's vocabulary-shard lookup plus one all-reduce, the head's argmax is a per-rank partial plus an all-gather (greedy drafting only); thread-emulated TP=2/4 drafts equal the single-rank draft (29 passed); the engine run waited for the cards (see "Merged tree") |

## What the draft is

`mtp_config` of the real checkpoint: 3 draft layers (hidden 2304, 24 q / 8 kv heads, head
dim 128, MLP 9216, RoPE theta 1e6, eps 1e-6), layers 0 and 2 sliding-window (1024), layer 1
full attention, block of 8 rows (row 0 = the last committed token, rows 1..7 = mask),
Markov rank 256, taps `target_layer_ids [5, 23, 41, 48]`.

* **Taps.** Three taps are the fp32 attention-residual partial block at the *entry* of
  target layer visits 5, 23 and 41 (global visit index over the 48 visits of the looped
  stack); the fourth is the final normalized output.  The speculator stores them as bf16
  bits (`inkling2_dspark_store_taps`, round-to-nearest-even) in `[max_num_tokens, 4*2304]`.
* **Context path.** For every draft layer: DFlare mix
  `rmsnorm(sum_t softmax(fuse_weight[l])[t] * tap_t) * w[l]` (`mix_rms`), K/V projection
  (dense fp16 GEMM), per-head RMSNorm + RoPE + paged KV write (`qkv_prep`).  The draft owns
  three KV caches (`model.layers.{0,1,2}.self_attn.attn`), page layout `[nb, H, bs, 256]`.
* **Block path.** Anchor/mask embedding with the target's embedding + `embed_norm`
  (`embed_block`), then per layer RMSNorm -> QKV GEMM -> `qkv_prep` -> paged split-KV
  attention (`attention`, bidirectional inside the block, window 1024 on layers 0 and 2) ->
  `o_proj` (fp32 split-K partials) -> `add_rmsnorm` -> gate/up GEMM -> `silu_mul` -> down
  (split-K partials) -> `add_rmsnorm`.
* **Markov head.** `f_rnn` from the last committed row (alpha row 3 of `fuse_weight`),
  base logits from the **target's `lm_head`** (shared), the Markov `hidden_out` folded into
  the head (`fold_w = lm_head . hidden_out_t`, `fold_b`, recomputed by
  `refresh_derived_weights`), then 7 sequential steps `markov_step` -> `head_logits`
  (softcap 30) -> `argmax` (greedy) or the runner's sampler (probabilistic, which also
  fills the draft-logits cache the rejection sampler reads).

## Kernels (`inkling2_dspark_*`, namespace `inkling2_dspark`)

`store_taps`, `mix_rms`, `embed_block`, `add_rmsnorm`, `silu_mul`, `gather_rows`, `gemv`
(skinny fp16, up to 32 rows, optional gathered rows and split-K fp32 partials), `gemm`
(the existing dense RDNA2 designs 0..5, fp16 in/out, fp32 accumulate), `qkv_prep`,
`attention` (split-KV, wave32, `V_DOT2_F32_F16`, pages of 16 or 64), `markov_step`,
`head_logits`, `fold_bias`, `argmax`, and `prepare_inputs` (HIP port of the runner's Triton
`_prepare_dflash_inputs_kernel` + `_pad_dflash_inputs` for context-parallel size 1).  All
launch on the current HIP stream, none synchronise the host, all shapes are static per
captured graph.  The TU is built at the official flags (`-O2`); the dev JIT builds used for
the tests above use `-O3` `[the A/B is below]`.

## Numerics and deviations from the checkpoint

| Quantity | Reference (NVIDIA / transformers) | Here | Why |
| --- | --- | --- | --- |
| Draft weights | bf16 | fp16 (`bf16 -> fp16` is exact in the normal range) | gfx1030 has no bf16 dot; overflow raises at load, flushed-to-zero values are counted |
| Residual stream, reductions, softmax, accumulators | bf16 storage / fp32 math | fp32 | match the reference accumulation |
| GEMM / attention inputs and outputs | bf16 | fp16 | `V_DOT2_F32_F16` |
| Taps | bf16 | bf16 bits (from the target's fp32 blocks and the fp32 final output) | exactly the reference's input precision |
| RoPE angle | fp32 | float64 `pos * inv_freq`, then fp32 sin/cos | long contexts (up to 1,048,576) |
| Markov state | bf16/fp32 | fp32 | recurrence over 7 steps |
| Draft-logits cache (probabilistic drafting) | bf16 | fp16 (`draft_logits_spec -> (float16, 0.0)`) | the sampler samples from the very values it caches, which is what keeps rejection sampling lossless |

fp16 weight report of the real `mtp.safetensors` (`fp16_weight_report`):
`max_abs 38.5`, `overflow 0`, `flushed_to_zero 446`, `max_rel 0.0`.  The fp16 headroom of
the draft *activations* on real target taps is **not measured** (only synthetic taps were
available); the fp32 residual stream makes an overflow unlikely `[INFERENCE]`.

## Results (what was run)

Commands run on the box with the dev-built library at `-O3` and the official kernel flags.

### Draft vs the fp64 oracle on the REAL checkpoint

`dev_run.py tools/rdna/inkling2/dspark_check.py --hip --context 300 1100` (b4 HIP 2): real
`mtp.safetensors` (rounded to the served fp16), the real target embedding / `embed_norm` /
unembedding, synthetic taps (no target forward exists yet), two requests of 300 and 1100
context rows (the second is past the 1024-key window), 7 draft steps each.  The oracle is
teacher-forced with the HIP draft's own tokens, so every step is compared on identical
Markov inputs.

| Quantity | Measured |
| --- | --- |
| K cache, max abs error (largest oracle magnitude 29.58) | 9.44e-3 |
| V cache, max abs error | 1.28e-2 |
| draft hidden, max abs / relative L2 | 5.53e-2 / 1.43e-3 |
| `f_rnn`, max abs | 1.03e-2 |
| softcapped logits (`abs(z) <= 30`), max abs / mean abs | 3.42e-2 / 4.39e-3 |
| greedy tokens differing from the oracle's argmax | 0 of 14 (`token_gap` 0.0) |

The draft's greedy chains were `[62, 2689, 82, 25, 196997, 58, 6]` and
`[25, 196997, 50, 12, 1539, 63, 150220]`.  Taps are synthetic, so these numbers bound the
draft's own arithmetic, not its acceptance length (that needs the target).

### Tests

| Command (box `par1-cs25`) | Result |
| --- | --- |
| `pytest tests/models/inkling2_loop/test_dspark_oracle.py` (CPU) | 15 passed, 16.33 s |
| `pytest tests/models/inkling2_loop/test_dspark_model.py` (CPU, GPU tests skipped) | 14 passed, 8 skipped, 7.51 s |
| `pytest tests/models/inkling2_loop/test_dspark_verifier.py` (CPU, GPU leg skipped) | 5 passed, 4 skipped, 4.54 s |
| `pytest tests/models/inkling2_loop/test_dspark_target_taps.py` (CPU, tiny1 + tiny8) | 4 passed, 3.14 s |
| same file with `visits_seen += 2` (hand mutation of the tap counter) | 4 failed (restored, 1 occurrence of `+= 1` verified) |
| lead's `test_paged_model.py -k "single_step or without_metadata or greedy"` with the hook in place | 10 passed |
| `test_inkling2_dspark_rdna2.py` on b4, earlier build (before mixed-dtype `store_taps`, `prepare_inputs`) | 111 passed, 63.01 s |
| same file, first run with `prepare_inputs` | 122 passed, 8 failed (the new `prepare_inputs` tests: my test layout used a query block of `steps + 1` rows with `sample_from_anchor=True`, which the Triton kernel itself writes out of bounds; the test now uses the runner's layout and the op rejects the other one) |
| same file, clean build (baseline of the mutation run) | 130 passed, 53.65 s; after restoring: 130 passed, 47.91 s |

### Hand mutations of the kernels

Each mutant patches the kernel TU in the box tree copy, rebuilds the dev library, runs
`tests/kernels/rocm/test_inkling2_dspark_rdna2.py -x` and must fail a test that passes on
the clean build (`mutate.py`; the clean source is restored and re-verified at the end).

| Mutant | Killed by |
| --- | --- |
| bf16 round-to-nearest-even -> truncation (`store_taps`) | `test_store_taps_rounds_to_nearest_even` |
| RoPE angle `pos * inv_freq` formed in fp32 instead of float64 | `test_qkv_prep_block_matches_fp64[16]` |
| window start `p - window` -> `p - window + 1` (attention) | `test_attention_matches_fp64[seq_lens2-1024-1]` |
| wrong softmax row of `fuse_weight` (`l ^ 1`) | `test_mix_rms_three_layers_matches_fp64` |
| `gemv` ignores `x_index` | `test_gemv_gathers_rows_through_x_index` |
| split-K consumer drops the last partial | `test_add_rmsnorm_split_partials_sum_in_order` |
| argmax ties resolve to the higher index (4 sites) | `test_argmax_matches_reference[1-200058-200064]` |
| `prepare_inputs` writes slots of the null block | `test_prepare_inputs_matches_the_triton_kernel[3-8-idx_dtype0-False]` |
| softcap without the `cap` scale | `test_head_logits_matches_fp64[1]` |
| Markov gate roles swapped | `test_markov_chain_matches_fp64[prev_dtype0]` |

All ten were killed.  The mutation of the verifier gate is its own negative control (four
wrong rules, run on the CPU).

### Model tests, verifier gate, dispatch and timings (b4 HIP 2, one window)

`pytest tests/models/inkling2_loop/test_dspark_model.py
tests/models/inkling2_loop/test_dspark_verifier.py
tests/kernels/rocm/test_inkling2_dspark_rdna2.py`: **161 passed in 90.34 s**
(22 + 9 + 130).  HIP draft vs the fp64 oracle in the three model-level comparisons:

| Case | hidden rel. L2 | logit max abs | `f_rnn` max abs | K / V cache max abs (largest magnitude) | tokens differing |
| --- | --- | --- | --- | --- | --- |
| random weights of the real shapes, 3 requests (37, 1100, 2100 rows) | 5.56e-4 | 2.19e-2 | 7.80e-3 | 3.88e-3 / 3.17e-3 (7.55) | 0 of 21 |
| the same weights, 40 requests (two head chunks, dense GEMM designs) | 6.33e-4 | 2.43e-2 | 1.44e-2 | 3.95e-3 / 3.06e-3 (7.34) | 1 of 280 (logit gap 1.1e-3, a near tie) |
| real `mtp.safetensors`, 3 requests (300, 1500, 90 rows) | 1.25e-3 | 2.33e-2 | 1.53e-2 | 9.44e-3 / 1.38e-2 (29.58) | 0 of 21 |

Also passing in that window: the page-size test (16 vs 64: same keys, only the split-KV
summation order may differ), padded requests, `f_rnn` after rejections (int32 and int64
`num_rejected`), whole-draft-step HIP graph capture + replay bit-exact against eager, and
the four production-rejection-sampler cases of the exact-probability gate (draft logits
cached as fp32 and fp16, two seeds).  The tolerances in `test_dspark_model.py` are about
10x the observed errors (`f_rnn` 1.3x), i.e. they are loose; the kernel mutants above are
the measured evidence that the tests can fail.

`prepare_inputs`: with the dispatch patch applied to the box tree copy,
`tests/v1/spec_decode/test_dflash_prepare_inputs.py` (the runner's own test of the Triton
kernel; `cp_size == 1` cases go through the HIP op) gave 6 passed in 4.26 s.

Draft timings (`bench_run.py`, mean of 30 runs, HIP events, `-O3` dev build; `context` is
the context pass over *all* listed rows, i.e. prefill-like for the long cases, `block` is
the per-proposal block pass + Markov head at that KV length):

| Requests x context rows | context pass | block pass |
| --- | --- | --- |
| 1 x 8 (steady state of one request) | 0.136 ms | 4.85 ms |
| 8 x 8 | 0.258 ms | 7.42 ms |
| 1 x 300 | 0.542 ms | 4.99 ms |
| 1 x 2100 | 2.14 ms | 5.45 ms |
| 1 x 4096 | 4.36 ms | 5.77 ms |
| 8 x 300 | 2.51 ms | 8.01 ms |
| 4 x 2100 | 9.86 ms | 13.16 ms |

At one request the block pass is ~5 ms for any context up to 4096 rows (layer 1 reads
the whole context; the other two read at most 1024 keys), so a proposal costs about 5 ms
at the steady state `[INFERENCE: the per-kernel split is not profiled]`.  The `-O2` vs
`-O3` A/B of this TU was **not run**; the TU is built with the repo default (`-O2`) and
is not in the `-O3` list of `CMakeLists.txt`.

## Verifier gates

1. **fp64 oracle** (`tools/rdna/inkling2/dspark_check.py`, `tests/models/inkling2_loop/test_dspark_oracle.py`): a
   plain-PyTorch float64 implementation of the whole draft (mix, K/V, window attention,
   MLP, Markov head with teacher forcing) that is also run against the real checkpoint.
2. **Kernel reference tests** (`tests/kernels/rocm/test_inkling2_dspark_rdna2.py`): every
   op against fp32/fp64 PyTorch (plus a Python port of the Triton input-preparation kernel).
3. **Model tests** (`tests/models/inkling2_loop/test_dspark_model.py`): the HIP draft
   against the oracle on random weights (contexts shorter than, just past and far past the
   window; 40 requests = two Markov-head chunks; padded requests; 16 vs 64 page size; `f_rnn`
   after rejections; HIP graph replay bit-exact) and on the real checkpoint.
4. **Exact-probability enumeration gate of the rejection path**
   (`tests/models/inkling2_loop/test_dspark_verifier.py`).  On a 4-token vocabulary with
   Markov target `p` and draft `q` the output law of one verification round (K = 3) is
   enumerated with `fractions.Fraction` (every draft block, every accept/reject path, every
   resample) and shown equal to the closed form
   `P(e_0..e_{i-1}, n >= i) = prod_{j<i-1} min(q_j, p_j)(e_j) * p_{i-1}(e_{i-1})`.  The
   production rejection sampler (HIP ports of the V2 runner) is then run on 400,000 random
   blocks (draft logits cached as fp32 and as fp16, as served) and every level's prefix
   counts are chi-square tested against the exact probabilities (threshold
   `df + 6 sqrt(2 df)`).  The same statistic must reject four deliberately wrong rules
   (always accept, resample from `p` instead of the residual, accept with `sqrt(p/q)`, emit
   the draft token on rejection); this negative control passes on the CPU.
   It covers token-wise verification; block verification is not covered.
5. **Hand mutations** of the kernels (table in Results).

## Wiring outside the model files (all applied in the repo)

### 1. Registry (`vllm/model_executor/models/registry.py`, `_SPECULATIVE_DECODING_MODELS`)

```diff
     "Gemma4DSparkModel": ("gemma4_dspark", "Gemma4DSparkForCausalLM"),
+    "Inkling2DSparkModel": (
+        "vllm.models.inkling2_loop.amd.dspark",
+        "Inkling2DSparkModel",
+    ),
```

### 2. `prepare_dflash_inputs` HIP dispatch (`vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`)

Applied in part B and gated: the condition in the repo is `if hip_kernel and _ON_RDNA2 and
cp_size == 1:`, `prepare_dflash_inputs(..., hip_kernel=self.uses_hip_prepare_inputs)` and
`Inkling2DSparkSpeculator` is the only class that sets `uses_hip_prepare_inputs = True`.
The diff below is the first, ungated version.  The runner's own
`tests/v1/spec_decode/test_dflash_prepare_inputs.py` therefore runs the Triton kernel again.

```diff
@@ -7,12 +7,14 @@
 import torch
 import torch.nn as nn
 
+from vllm import _custom_ops as ops
 from vllm.config import VllmConfig, get_layers_from_vllm_config, replace
 from vllm.config.compilation import CUDAGraphMode
 from vllm.forward_context import BatchDescriptor, set_forward_context
 from vllm.logger import init_logger
 from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
 from vllm.platforms import current_platform
+from vllm.platforms.rdna import on_rdna2_hip
 from vllm.triton_utils import tl, triton
 from vllm.v1.attention.backend import AttentionCGSupport
 from vllm.v1.attention.backends.utils import PAD_SLOT_ID
@@ -37,6 +39,9 @@
 from vllm.v1.worker.utils import AttentionGroup
 
 logger = init_logger(__name__)
+
+# Read once at import: Dynamo-traced hot paths must not re-evaluate the gate.
+_ON_RDNA2 = on_rdna2_hip()
 
 
 class DFlashSpeculator(DraftModelSpeculator):
@@ -992,6 +997,44 @@
     # per-request query length, not the total token count across the batch.
     max_target_query_len = int(input_batch.num_scheduled_tokens[:num_reqs].max())
     max_tokens_per_req = max_target_query_len + num_query_per_req
+    if _ON_RDNA2 and cp_size == 1:
+        # native HIP port of the kernel below (csrc/rocm/inkling2_dspark_rdna2.cu)
+        ops.inkling2_dspark_prepare_inputs(
+            input_buffers.input_ids,
+            input_buffers.positions,
+            input_buffers.is_padding,
+            input_buffers.query_start_loc,
+            input_buffers.seq_lens,
+            query_slot_mapping,
+            context_positions,
+            context_slot_mapping,
+            sample_indices,
+            sample_pos,
+            sample_idx_mapping,
+            temperature,
+            seeds,
+            input_batch.positions,
+            input_batch.query_start_loc,
+            input_batch.idx_mapping,
+            last_sampled,
+            next_prefill_tokens,
+            num_sampled,
+            num_rejected,
+            input_temperature,
+            input_seeds,
+            block_table,
+            block_size,
+            parallel_drafting_token_id,
+            num_query_per_req,
+            num_speculative_steps,
+            max_num_reqs,
+            max_num_tokens,
+            max_model_len,
+            num_reqs,
+            max_tokens_per_req,
+            sample_from_anchor,
+        )
+        return
     BLOCK_SIZE = 128
     num_blocks = triton.cdiv(max_tokens_per_req, BLOCK_SIZE)
     _prepare_dflash_inputs_kernel[(num_reqs + _NUM_PAD_PROGRAMS, num_blocks)](
```

### 3. Target tap hook (`vllm/models/inkling2_loop/amd/model.py`) - APPLIED

`Inkling2Model.aux_hidden_state_layers` (empty by default); `forward` counts layer visits
across the prelude, every pass of the loop and the coda, appends the fp32 block state at
the entry of each selected visit and returns `(out, [*taps, out])` (the final normalized
output is the last tap) when the set is non-empty.  `Inkling2LoopForConditionalGeneration`
gets `supports_eagle3 = True`, `set_aux_hidden_state_layers`,
`get_eagle3_default_aux_hidden_state_layers() -> (5, 23, 41)` and returns
`(hidden, taps)` from `forward` when taps were requested.  The structural
`SupportsEagle3` check of the runner needs `has_own_lm_head` / `has_own_embed_tokens` as
well (the protocol is runtime-checkable over every member; added in part B), and
`Inkling2DSparkConfig.eagle_aux_hidden_state_layer_ids` is `target_layer_ids[:-1]` =
(5, 23, 41): the target appends the final output itself and the id 48 is not a layer.

### 4. Test tethering (`tools/pre_commit/test_tethering_allowlist.txt`) - APPLIED

Four fork-only entries (`test_dspark_model.py`, `test_dspark_oracle.py`,
`test_dspark_verifier.py`, `test_dspark_target_taps.py`).

### 5. Draft quantization (`vllm/v1/worker/gpu/spec_decode/dspark/config.py`) - APPLIED

`DSparkSpeculatorConfig.default_model` copied the target's `modelopt_fp4` to the draft
(DeepSeek DSpark ships its weights inside the quantized target checkpoint).
`mtp.safetensors` is bf16, so `get_draft_quant_config` tried to parse a ModelOpt config
for it and failed (`Missing 'quant_algo'`).  Like Mistral Large 4, an `inkling2_loop` target
leaves the draft unquantized; the draft dtype needs nothing (fp16 like the target, the
bf16 weights are converted at load).

## What remains / risks

* Measured on the V2 runner (`inkling2-port-spec.md`, part B): the draft is correct (the
  acceptance is far above chance and the greedy output of a speculative run stays inside
  the run-to-run floor of the target), but with all 7 draft tokens verified the step costs
  2.4x a plain decode step (48.2 ms against 20 ms at batch 1: every verified row runs the
  MoE kernels, 8 rows pay the 8-row instance), so `num_speculative_tokens=7` alone gives
  0.98x at batch 1 over eight prompts and loses from batch 4.  The scheduler's dynamic
  speculative decoding verifies fewer tokens while the draft still proposes 7 (the model
  requires `num_speculative_tokens = 7`, the draft block is 8 rows):
  `num_speculative_tokens_per_batch_size=[[1, 1, 3], [2, 8, 0]]` measured 1.21x at batch 1
  (16.50 ms per token against 20.03 ms; K = 1, 2, 4, 7: 1.11x, 1.11x, 1.03x, 0.98x) and the
  no-draft speed from batch 2 up.  Part B2 (`inkling2-port-spec.md`) changed the two causes the
  profile found for the fifth-row cost (the row policy picked the 8-row MoE instance from 5
  tokens, and the old mxfp8 kernel spilled registers; per-launch numbers there): the K table
  above was measured before it, and its engine-level re-measurement is not in this document.
* `draft_logits.py` (`_assign_rows_kernel` / `_stale_drafts_kernel`) is runner-owned Triton
  used only for probabilistic drafting; greedy drafting (default) does not touch it.
* Prefix-cache hits: the draft K/V blocks are cached with the target's, so a hit reads
  K/V written when the block was produced; the acceptance of a warm run equals that of the
  cold run (part B).  Blocks produced by a run without DSpark would be stale (quality
  only, never correctness).
* The sliding-window manager may free the block holding key `P-1024`; the kernel then reads
  the null block (finite zeros), a one-key mismatch for that single key.
* The dense GEMM design table (`_gemm_design`) is not tuned for the draft's shapes.
* TP > 1: no engine run yet (see "Merged tree"); `max_model_len` is capped at the draft's
  1,048,576 positions by the model, and the draft's full-attention layer keeps its K/V
  replicated on every rank (about 4 KiB per token and rank, 4 GiB at 1,048,576 tokens
  `[INFERENCE: 8 kv heads x 128 x 2 B x 2]`), so DSpark at TP=2 does not leave the 1M
  context the room it has without it; the long-context draft cost is measured only up to
  the contexts in Results.

## Merged tree (conv ring + TP + graphs): DSpark on `feat/inkling2-merge` b94fb11a56

What was run: the real checkpoint, TP = 1, `FULL_AND_PIECEWISE`, prefix caching on,
`max_model_len` 4096, `max_num_batched_tokens` 1024, `max_num_seqs` 8,
`speculative_config={"method": "dspark", "num_speculative_tokens": 7,
"num_speculative_tokens_per_batch_size": [[1, 8, K]]}` (the model requires
`num_speculative_tokens == 7`, so every configuration drafts 7 and the schedule picks how
many are verified), 24 chat prompts x 128 tokens, 2 repetitions
(`tools/rdna/inkling2/dspark_bench.py --bench --spec-k 7 --schedule '[[1,8,K]]'`; the
box was shared with other agents, so the times are indicative).  The draft boots (30 s),
captures its FULL graphs for 1..8 requests, the conv ring has `3 + 7 = 10` rows, and the
KV cache with the draft holds 20,430 tokens against 42,142 without it (K = 5 boot).

Milliseconds per generated token of one request (speed-up over no draft):

| | b1 | b2 | b4 | b8 |
| --- | --- | --- | --- | --- |
| no draft | 19.04 | 23.40 | 31.06 | 43.09 |
| K = 3 | **15.83 (1.20x)** | **21.87 (1.07x)** | 32.60 (0.95x) | 44.01 (0.98x) |
| K = 5 | 16.31 (1.17x) | 24.08 (0.97x) | 35.83 (0.87x) | 49.00 (0.88x) |
| K = 7 | 17.75 (1.07x) | 25.42 (0.92x) | 39.27 (0.79x) | 55.92 (0.77x) |

K = 1, 2, 4, 6 are not measured on this tree.  Accepted drafted tokens per step at batch 1
(the other batch sizes are within 0.02): K = 3 1.239, cumulative per position
`[0.632, 0.378, 0.228]`; K = 5 1.390, `[0.620, 0.353, 0.207, 0.125, 0.085]`; K = 7 1.474,
`[0.605, 0.363, 0.212, 0.125, 0.084, 0.050, 0.034]`.  The effective step of the bench is
`ms/token x (1 + accepted)`: 35.4 ms (K = 3), 39.0 ms (K = 5), 43.9 ms (K = 7) at batch 1.
Policy these numbers support: `[[1, 2, 3], [3, 8, 0]]` (K = 3 for one or two requests, no
draft above).  Batch 1 and 2 are measured with K = 3; batch 3 is not measured and the
neighbours (1.07x at b2, 0.95x at b4) put it at break-even `[INFERENCE]`; K = 1 and 2 at
b2 / b4 could still win and were not run.  The earlier tree measured K = 1, 2, 3, 4, 7 at
batch 1 only (K = 3 best, 1.21x).

Where a verification step goes (`profile_decode.py --spec-k 7`, async scheduling off,
random prompts, so the numbers are of a K = 7 step, 8 rows per request): batch 1 clean step
37.7 ms (36.8 ms in a second run), GPU kernels 33.2 / 36.4 ms; the draft is 3.6 ms of the
profiler's per-step mean and 4.7 ms in the traced step (68 kernels, 5.2 ms from the first
to the last).  Within a traced draft pass: the base-logits sweep of the LM head 2.15 ms
(one 922 MB read, 429 GB/s), the three layers 1.7 ms (qkv 71 - 78 us, o_proj 54 - 61 us,
gate/up 212 - 216 us, down 128 - 129 us per layer), seven Markov steps about 0.13 ms each
(`markov_step` 16 us, `head_logits` 62 us after a 249 us cold first one, `argmax` 21 us,
launch gaps), the context path 0.2 ms.  The target's verification of 8 rows is 29.7 ms
(MoE NVFP4 13.3 ms, dense GEMV 8.4 ms, its own LM head 2.1 ms).  At batch 4 the step is
69.5 ms and the draft's kernels 11.0 ms (the 28-row base GEMV and the backbone on
`gemv_kernel<32>`: 8.3 ms), which only matters where the policy drafts.  The bench's
effective step at K = 7 and batch 1 (43.9 ms) is 6 - 7 ms above the profile's clean step;
where that goes is open (`dspark_bench.py --async-scheduling off` switches the engine's
async scheduling to compare).  The profiler traces show 5 - 7 ms gaps inside a step that
clean steps do not have (busy / wall 98.9% in the second run) `[INFERENCE: profiler
artifact]`.

Levers, by what the trace shows.  Done: the Markov chain stops at the schedule's largest
verified K (`scheduled_draft_tokens`, `_markov_steps`; the tokens after it are never
read: the runner passes `draft_tokens[:, :num_spec_tokens]` on), 0.13 ms per skipped step
(0.5 ms, 1.4% of a K = 3 step); CPU tests pass, the GPU oracle test
(`test_the_markov_chain_stops_at_the_largest_verified_token`) is written and **not run**.
Not done: an 8-bit copy of the LM head for the base sweep (about -1 ms per step, 3%, costs
0.46 GB and acceptance risk `[INFERENCE]`), a frequency-pruned draft vocabulary (about
-1.7 ms; needs an acceptance measurement on real outputs).  The sweep is at the memory
roofline, the layers are at 250 - 395 GB/s: the draft is about 14% of a K = 7 step and the
K = 3 gain is bounded by the verification rows, not the draft.

Bugs found on the merged tree: FULL capture of the draft (verify batches of 8 rows with
the capture's dummy metadata, `seq_lens = 1`, zero block tables) made the conv ring write
a negative row (`Memory Fault ... i2_state_write_kernel`): `i2_ring_keeps` accepted
`p < 0`, and padded PIECEWISE tokens read as request 0 and wrote its ring.  The merge
worktree fixes both (`csrc/rocm/inkling2_attn_rdna2.cu`: `0 <= p < p1`, no write to block
0; `amd/state.py`: `seq_idx = -1` for padded tokens); the K = 3 / 5 / 7 runs above ran with
the first fix only (the other kernels in `_rocm_C` rebuilt from that tree).

DSpark at TP > 1 (`amd/dspark.py`, `tests/models/inkling2_loop/test_dspark_tp.py`):
the draft weights, its K/V and the taps are replicated, so no draft collective sits inside
the layers.  The anchors' embedding rows come from the target's vocabulary shard
(`inkling2_glue_embed_lookup`) and one all-reduce of the `[rows, hidden]` block (at most
64 KiB); the folded head runs over this rank's vocabulary rows and
`inkling2_dspark_argmax_partial` (new op: value and global id of the best row of the
shard, padding rows excluded) feeds an all-gather and `inkling2_linear_argmax_reduce` (ties
to the lowest id, as at TP = 1).  Greedy drafting only: the probabilistic draft raises
`NotImplementedError`.  `Inkling2DSparkModel` no longer refuses TP > 1.  The TP = 2
no-draft engine on `FULL_AND_PIECEWISE` (collectives inside the graphs) was measured as
the baseline: 14.63 / 17.22 / 21.28 / 29.69 ms per token at batch 1 / 2 / 4 / 8.  The
draft at TP = 2 / 4 through the engine, its speed-up and its equality with no draft were
**not run** (HIP 4, 5 were in another agent's session).

Gates run on the box (HIP 2): `test_inkling2_dspark_rdna2.py` 139 passed (130 + 9 for the
new op); `test_dspark_tp.py` + `test_dspark_model.py` 29 passed (thread-emulated TP = 2 / 4
drafts equal the single-rank draft); `test_dspark_model.py -k markov_steps` on the CPU
7 passed.  Six-arch compile of the TU (gfx90a, gfx942, gfx950, gfx1030, gfx1100,
gfx1200): exit 0 and no errors on all six (34 kernels on gfx1030, none elsewhere by
design); gfx1100 failed first on five unused constants of the RDNA2-only block, which now
sit inside its guard.  Mutation check of `argmax_partial`: baseline 133 passed (its 6
failures are, `[INFERENCE]`, the six cross-shard reduce cases, whose op is not in the dev library); the
mutant `partial_drops_first_id` was killed; the other five (`partial_value_row_zero`,
`partial_id_row_overlaps_values`, `partial_scans_the_shard_padding`,
`partial_ties_to_higher_index`, `partial_first_id_off_by_one`) were **not run**.  Not run:
equality of the draft with no draft on this tree (the earlier tree: 20 of 26 identical,
`|dlogp|` <= 0.081 against a floor of 0.082), the prefix-cache cold / hit / duplicate
check (`dspark_bench.py --prefix-check`; the sweeps above ran with prefix caching on and
accepted as before), and the remaining K values.  No-draft `FULL_AND_PIECEWISE` against
eager (6 prompts, 64 tokens): 13 of 24 identical, the divergences at index >= 10 on
near-ties of at most 0.17 nats.
