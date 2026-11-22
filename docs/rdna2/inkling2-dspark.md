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
| Target tap hook in `amd/model.py` | applied (inert unless the runner asks for taps); 4 CPU tests on the tiny checkpoints; **never run on the real target** |
| Runner-owned Triton (`prepare_dflash_inputs`) | HIP op written, tested against a Python port of the kernel; with the dispatch below the runner's own `test_dflash_prepare_inputs.py` passes (6 passed) on b4; the dispatch is **not applied** to the repo |
| V2 runner end-to-end (model load + `propose` through `LLM(...)`) | **not run** (needs the tap hook, the registry entry and the dispatch patch; see "What remains") |
| FULL HIP-graph capture inside the runner | **not run**; capture + replay of the whole draft step is tested bit-exact on b4 in the harness |
| TP > 1 | **not tested** (draft is replicated, the head is vocab-shard aware) `[INFERENCE]` |

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

## Wiring that is NOT in the repo (patches for files outside the contract)

### 1. Registry (`vllm/model_executor/models/registry.py`, `_SPECULATIVE_DECODING_MODELS`)

```diff
     "Gemma4DSparkModel": ("gemma4_dspark", "Gemma4DSparkForCausalLM"),
+    "Inkling2DSparkModel": (
+        "vllm.models.inkling2_loop.amd.dspark",
+        "Inkling2DSparkModel",
+    ),
```

### 2. `prepare_dflash_inputs` HIP dispatch (`vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`)

Applied only in the box tree copy; the diff below is the script's exact output.

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
`SupportsEagle3` check of the runner has not been exercised.

### 4. Test tethering (`tools/pre_commit/test_tethering_allowlist.txt`) - APPLIED

Four fork-only entries (`test_dspark_model.py`, `test_dspark_oracle.py`,
`test_dspark_verifier.py`, `test_dspark_target_taps.py`).

## What remains / risks

* The V2 runner has not run this draft end to end: apply the registry and dispatch
  patches above (the tap hook is already in), then
  `LLM(model=~/hfcache/inkling2-mini, dtype="float16", speculative_config={"method":"dspark",
  "num_speculative_tokens":7})`.  Open questions that only that run answers: the draft KV
  owners' registration in `static_forward_context` and the KV-cache config, the draft
  `ModelConfig` dtype (bf16 checkpoint vs fp16 target), `spec.quantization` propagation,
  FULL-graph capture in `DFlashCudaGraphManager`, and the acceptance length.
* `draft_logits.py` (`_assign_rows_kernel` / `_stale_drafts_kernel`) is runner-owned Triton
  used only for probabilistic drafting; greedy drafting (default) does not touch it.
* Prefix-cache hits on blocks written without DSpark leave the draft context K/V stale
  (quality only, never correctness of the output).
* The sliding-window manager may free the block holding key `P-1024`; the kernel then reads
  the null block (finite zeros), a one-key mismatch for that single key.
* The dense GEMM design table (`_gemm_design`) is not tuned for the draft's shapes.
* TP > 1 (head all-gather) is untested; `max_model_len` is capped at the draft's
  1,048,576 positions by the model; the long-context draft cost is measured only up to the
  contexts in Results.
