# Inkling-2-Mini (thinkingmachines) — serving port spec

Checkpoint: `thinkingmachines/Inkling-2-Mini`, 18.8 GB, one shard + `mtp.safetensors`.
Reference implementation: `transformers` main, `models/inkling2_loop/modeling_inkling2_loop.py`
(1840 lines) + `configuration_inkling2_loop.py`. Upstream vLLM has no Inkling model.

Verified on the box before this spec: NVFP4/MXFP8 expert dequant and the
on-the-fly `V_DOT2` GEMM match an fp64 reference on the real checkpoint
(0 mismatches; dot error 0.033 % of max for NVFP4, 4e-6 for MXFP8).

## Text architecture

| Item | Value |
|---|---|
| Layers | 24, hidden 2304, GQA 24 q / 8 kv, head_dim 128 |
| Attention type | layers 0-4, 6-10, 12-16, 18-22 sliding (window 1024); 5, 11, 17, 23 full |
| Position encoding | **no RoPE** — hidden-state-conditioned relative bias only |
| Relative bias | `linear: hidden -> heads*d_rel(16)`, learned `proj [16, 1024]`, gathered by distance; zero outside `rel_extent` |
| Q/K | learned per-head `scale`, L2 clip at 22.63 (`q`/`k`), log-scale `1 + 0.1*log(max((pos+1)/8192, 1))` on full layers |
| V | unweighted RMS norm |
| k/v short conv | depthwise k=4 over the kv width; **dynamic** (weights from hidden) on the 4 full layers |
| Attention sinks | one extra zero logit per head from `sink_proj: hidden -> heads` |
| Output | `linear(sigmoid(hidden @ gate_attn) * attn_out)` |
| Post blocks | depthwise conv k=4 on the attention output and on the MLP output |
| Attention residual | every layer queries the current block's layer outputs + a partial state with a learned vector (`attention_input`, `mlp_input`) |
| Looped core | layers 6-17 run 3 passes; passes 2-3 restart from the pre-loop state and receive four per-slot messages through rank-64 gates (`q/k/v/mlp_message_gate`) |
| Final readout | `final_input` block-attention over all block states, then RMSNorm |
| MoE | all 24 layers, 128 experts, top-8 sigmoid routing with a score-correction bias, `route_scale 8`, intermediate 1152, fp32 accumulate |
| Experts activation | `tanh(g/4) * sigmoid(g) * tanh(u/25)` |
| Vocab | 201024 embedding rows, 200058 output rows (untied), final logit softcap 30 |
| MTP | separate `mtp.safetensors`: 3 layers, markov rank 256, `target_layer_ids [5, 23, 41, 48]` |

## Expert weights (measured)

| Layers | Format | Weight | Block scale | Global |
|---|---|---|---|---|
| 0-19 | NVFP4 | U8 `[E, N, K/2]` E2M1, low nibble = even k | F8_E4M3 `[E, N, K/16]` | per expert `amax / outer_scale_denominator`, denominator = 2688 |
| 20-23 | MXFP8 | F8_E4M3 `[E, N, K]` | UE8M0 `[E, N, K/32]` | none |

`w13` is `[E, 2304, 2304]` (gate+up fused), `w2` is `[E, 2304, 1152]`; dequantised
both are bf16 48.8 GB for all layers.

### `w13` row layout (2026-11-21)

The real checkpoint stores the rows of `w13_weight` **and** `w13_weight_scale`
interleaved: row `2i` is gate unit `i`, row `2i + 1` is up unit `i`
(`[g0, u0, g1, u1, ...]`); `amax` is per expert and unaffected, `w2` is plain. The
transformers reference de-interleaves with `Interleave(dim=1)`
(`conversion_mapping.py`, `inkling2_loop`: `out[:, j * I + i] = in[:, 2 * i + j]`, i.e.
`gate_up = [even rows; odd rows]`, then `gate, up = gate_up.chunk(2)`).

Checked on the real weights (CPU, a few experts of layers 0, 3, 7, 12, 19, 20, 23,
dequantised): mean |cos| of rows `(2i, 2i + 1)` is 0.206 / 0.196 / 0.155 / 0.118 /
0.108 (NVFP4) and 0.088 / 0.088 (MXFP8); rows `(i, i + 1152)` and the off-by-one
controls are ~0.019 (random level). Every row has unit norm, so row norms say
nothing; which of the pair is gate comes from the reference conversion, not from the
data. Anything that read `w13` verbatim treated gate and up as two halves, i.e. mixed
units.

Consequences: `Inkling2LoopForConditionalGeneration.load_weights` de-interleaves
`w13_weight` / `w13_weight_scale` while copying (`copy_gate_up_deinterleaved`), so
`Inkling2LoopMoE` and the kernels keep `[gate; up]` along N. The tiny checkpoints
(`make_tiny_checkpoint.py`) now write the interleaved layout, and the reference
oracles (`tiny_reference.dequant_experts`, `ref_checkpoint.expert_pair`) de-interleave;
both fill the transformers model directly and do not run its conversion, so nothing is
de-interleaved twice. The real-checkpoint oracle numbers below and the old
`l<layer>.safetensors` expert caches predate this finding (read the raw order); the
cache file is now `l<layer>.deint.safetensors`. Header-level proof that the real
checkpoint loads: `tests/models/inkling2_loop/test_real_checkpoint_names.py`.

## Serving implications

* **KV cache = 48 layer-equivalents, not 24.** Each loop pass keeps its own KV for
  layers 6-17 (3 x 12) plus the 12 non-loop layers. 20 of the layer-caches are
  sliding-window 1024, so the hit is bounded but real.
* **No RoPE** — the KV cache is the post-conv, post-clip k/v; nothing is rotated.
* **Conv state x4 per layer** (`k_sconv`, `v_sconv`, `attn_sconv`, `mlp_sconv`).
* **Per-layer extra attention**: `attention_input`/`mlp_input` attend over at most
  `n_layers_per_attn_res_block + 1` states (one head, no cache) — cheap but on the
  critical path of every layer.
* **Relative bias cost**: `[heads, 1024]` logits per query token per layer
  (24 x 1024 floats = 98 KB/token/layer). Must be fused into attention, not
  materialised per chunk.
* **MoE in every layer**, so the NVFP4/MXFP8 decode kernels matter for every step.

## Checkpoint name mapping (serving layout -> reference)

| Checkpoint | Reference |
|---|---|
| `model.llm.embed.weight` / `embed_norm` | `model.language_model.embed_tokens.weight` / `.embed_norm.weight` |
| `model.llm.unembed.weight` | `lm_head.weight` (untied) |
| `model.llm.norm.weight` | `model.language_model.norm.weight` |
| `model.llm.attn_res.output_query` | `final_input.query` |
| `model.llm.attn_res.attn_queries.{i}` | `layers.{i+1}.attention_input.query` (three slots on 5/11/17/23) |
| `model.llm.attn_res.mlp_queries.{i}` | `layers.{i}.mlp_input.query` |
| `middle_cycle_cross_pass_queries` (12,2,H) + `..._kv_queries` (12,2,H) | `message_input.query` (12,4,H) in (q, k, v, mlp) order |
| `attn.wq_du` / `wr_du` / `wk_dv` / `wv_dv` / `wo_ud` | `q_proj.linear` / `rel.linear` / `k_proj.linear` / `v_proj.linear` / `o_proj.linear` |
| `attn.gate_attn` | `o_proj.gate_proj` |
| `attn.attn_sink_proj` | `sink_proj` |
| `attn.{q,rel}_l2_clip.scale` | `q_proj.norm.scale` / `relative_position_bias.norm.scale` |
| `attn.rel_logits_proj.proj` | `relative_position_bias.proj` |
| `attn.{k,v}_sconv` / `attn_sconv` / `mlp_sconv` | `k/v_proj.sconv`, `post_attention_convolution`, `post_mlp_convolution` |
| `attn_norm` / `mlp_norm` | `input_layernorm` / `post_attention_layernorm` |
| `mlp.gate.weight` / `mlp.gate.bias` | `mlp.gate.weight` / `mlp.gate.e_score_correction_bias` |
| `controllers.{attn,k,v,mlp}.{down,forget,gain}_weight` | `{q,k,v,mlp}_message_gate.{down_proj,forget_gate_proj,message_gate_proj}` |

The loader implementing this table is `tools/rdna/inkling2/ref_checkpoint.py`.

## Open questions

1. Cross-pass message slot order (q,k,v,mlp vs another packing) — resolved by
   whether the reference produces coherent text with the mapping above
   (`INKLING_MSG_ORDER` switches it).
2. `middle_cycle_cross_pass_kv_queries` may instead feed the full layers' separate
   k/v inputs; the reference has a single 4-slot tensor, so the split above is the
   working assumption.
3. Vision tower (`model.visual.*` in the checkpoint vs `vision_tower.*` in the
   reference) is not mapped yet — text only.

## Delivery ladder

1. Reference oracle on the real checkpoint (CPU) — in progress.
2. vLLM model file: text-only, fp16/bf16, dequantised experts first, eager.
3. NVFP4/MXFP8 `dot2` MoE kernels replace the dequantised experts for decode.
4. MTP (needs the markov head) and vision.

## 2026-11-21 — MoE kernels done and proven; format metadata pinned

`csrc/rocm/moe_lowp_dot2_rdna2.cu` adds two RDNA2 (gfx1030) fused-MoE kernels
that share the MXFP4 skeleton (same tiling, V_DOT2 accumulate, packed-fp16 CAS
epilogue) and dequantise weights in registers, so no weight wider than 8 bits is
ever materialised:

| op | weight layout | scales | extra |
|---|---|---|---|
| `moe_nvfp4_gemm_rdna2` | `[E, K/8, N]` uint32, 8 E2M1 nibbles LSB-first | `[E, K/16, N]` fp8_e4m3 | `[E]` f32 globals |
| `moe_mxfp8_gemm_rdna2` | `[E, K/4, N]` uint32, 4 E4M3 bytes | `[E, K/32, N]` uint8 UE8M0 | — |

Dequant: `e2m1(nibble) * e4m3(scale) * global` and `e4m3(byte) * 2^(ue8m0-127)`.
Activations stay **fp16**: gfx1030 has neither fp4 nor fp8 dot hardware, so the
checkpoint's per-token/per-32 activation quantisation is deliberately not
applied (weights-only dequant, higher activation precision than the original).

Checkpoint metadata (from `hf_quant_config.json`, verified against the real
tensors):

| group | tensors | format |
|---|---|---|
| model group 2 | `layers.0-19.mlp.experts.{w13,w2}_weight` + `_weight_scale` + `_weight_amax` + `_weight_outer_scale_denominator` (=2688) | `nvfp4_e2m1`, block 16 |
| model group 3 | `layers.20-23.mlp.experts.{w13,w2}_weight` + `_weight_scale` | `mxfp8_e4m3`, block 32 |

`global = weight_amax / weight_outer_scale_denominator`. The checkpoint stores
`[E, N, K/2]` (NVFP4) / `[E, N, K]` (MXFP8) plus `[E, N, K/16]` / `[E, N, K/32]`
scales; the wrapper repacks both to the K-major layouts above at load.

Build gotcha: `CMakeLists.txt` whitelists `csrc/rocm/*.cu`; a new file that is
not added to that list links silently as if it did not exist (the ops are simply
absent at runtime).

Evidence (gfx1030, HIP 1, `tests/kernels/quantization/test_rdna2_moe_lowp.py`,
quantise with the checkpoint formula -> dequantise in torch -> compare per
`(m,k)` expert output): **24/24 pass**, E in {4,8}, K in {256,512}, N in
{128,256}, top_k 2, M in {1,4,8}, block_size_m in {1,4}, both formats.

Reference oracle on the real checkpoint (CPU, fp16, 24 layers of experts
dequantised from cache): "The capital of France is" -> top-3 `' '`, `' is'`,
`' France'`; logits mean 5.23, std 5.04, max 26.2. That is a coherent next-token
distribution, so the `(q, k, v, mlp)` message-slot mapping is at least
self-consistent (a bogus mapping produced punctuation/digit floods in the
chat-templated probes, which use a template the model does not expect).

Remaining before serving the quantised path:

1. Per-layer split: vLLM's `modelopt` config reads only `quant_algo: NVFP4`
   from `config.json`; layers 20-23 are MXFP8 and must be routed from
   `hf_quant_config.json`'s `parameter_groups`.
2. `Rdna2NvFp4MoEMethod` / `Rdna2MxFp8MoEMethod`: register the K-major packed
   weights, repack at load, call the kernels (w13 -> silu*mul -> w2), gated on
   `on_gfx10x()`.
3. The model itself: `Inkling2LoopForConditionalGeneration` is not in vLLM
   (`vllm/models/inkling/` is Inkling v1). First baseline: vLLM's generic
   `--model-impl transformers` backend on the HF implementation.

## 2026-11-21 — paged vLLM model: first boot and CPU-side validation (tiny checkpoints)

`vllm/models/inkling2_loop/amd/` is the serving model (`Inkling2LoopForConditionalGeneration`,
fp16 only, eager, text only). Its own name map is `model.py::_map_name` /
`load_weights` (the table above is still the contract; `ref_checkpoint.py` stays the
reference-side loader). Per-layer expert format (NVFP4 vs MXFP8) comes from
`hf_quant_config.json` `parameter_groups[*].compute_dtype` (`loader.py`), not from a
layer-index constant, so the tiny checkpoints (NVFP4 on layers 0-4, MXFP8 on 5) and
the real one (NVFP4 0-19, MXFP8 20-23) load through the same code.

Bugs the first boot and the CPU harness found:

| # | symptom | root cause | fix |
|---|---|---|---|
| 1 | `RuntimeError: shape '[8, 256, 128]' is invalid for input of size 524288` at load | MoE format was `layer >= 20 -> MXFP8`, hard-coded for the real model | read the formats from `hf_quant_config.json` |
| 2 | HIP page fault in `conv_taps` (`index_elementwise_kernel`) on the first forward | the conv-state owner of a layer lives in a *different KV-cache group* (4-token blocks, its own block ids / slot units) than its KV owner (64-token blocks); the layer used the KV owner's metadata for both, so conv slots were in 64-token units, ~16x too large; the extensible KV cache only commits a prefix of its blocks (`vllm/v1/worker/extensible_kv_cache.py`), so those indices were not present [INFERENCE: the commit size was not measured; the fault is gone with the fix] | `md_kv` / `md_conv` per owner, `Inkling2Attention.forward(..., kv_owner, conv_owner, kv_md, conv_md)` |
| 3 | HIP memory fault in `topk_log_softmax_kernel<float, 4>` during sampler warmup | `csrc/rocm/v1_sample_rdna2.cu`: the fp32 VEC=4 path bound-checks only the first 512-group of a partial last 1024-block; triggered when `vocab % 16 == 0 && vocab % 1024 != 0` (measured max abs error vs `torch.log_softmax`: V=500 4.8e-7, 512 0.767 + fault, 1024 9.5e-7, 1040 0.0226, 1536 0.300, 2048 9.5e-7, 4000 0.0235, 151936 0.0027) | not fixed in csrc here. The model exposes the *unpadded* vocab (500 tiny / 200058 real; padded 512 / 201024 would hit it): `padded_vocab_size` keeps the embedding rows, `lm_head` is trimmed to `vocab_size` rows |

CPU validation (no GPU): `tests/models/inkling2_loop/paged_harness.py` emulates the
engine's KV-cache manager around the real model class (per-group block free lists,
null block, sliding-window block freeing, real `Inkling2MetadataBuilder`, hybrid block
sizes 64/4/4/64) with NaN-poisoned `[B, bs, H, C]` pages, so reading an unwritten,
freed or null slot, or writing the null block, fails a test. `test_paged_state.py`
(29 tests, max |diff| 2e-7..1e-6 vs the dense oracle) covers `conv_taps`, `kv_write` /
`kv_read` and the paged attention; `test_paged_model.py` runs the whole model through
`load_weights` (exact dequantised experts) for `tiny1` and the 8-layer `tiny8`
(message-slot order observable): fp32 max |dlogit| vs the transformers reference is
2e-5..1.7e-4 for single-step prefill, chunked prefill (chunk 1/3/7/16/33 x KV block
4/16/64), cached and greedy decode, mixed prefill/decode batches and random schedules
with recycled blocks. Mutating the serving code (conv in-chunk condition, window by
one, KV write offset, conv owner reading the KV metadata, k/v message slots swapped)
fails these tests.

fp16 is the serving dtype, and the random tiny model is router-chaotic: a near-tie in
the top-2 expert choice flips under fp16 rounding and is amplified through the loop.
CPU emulation with exact experts, last-position mean |dlogp| vs the fp32 reference
(free routing / expert choice forced to the fp32 run's): tiny1 5 tokens 6.2e-3 /
6.2e-3, 40 tokens 1.053 / 1.5e-2, 70 tokens 0.139 / 7.9e-3; tiny8 (seed 0, the
checkpoint of the GPU runs) 1.7e-2 / 1.7e-2, 6.045 / 6.1e-2, 1.777 / 2.8e-2. So "mean
|dlogp| < 0.05" is a statement about routing stability of the prompt, not only about
the kernels; the forced-route number is the noise floor of the fp16 path (mean over all
positions 7e-3 .. 1e-2 for tiny1 and 1.9e-2 .. 2.2e-2 for tiny8, also for chunked
prefill, mixed batches and cached decode: `test_fp16_*_given_the_same_expert_choice`).

Still open: the engine/GPU acceptance (`tools/rdna/inkling2/tiny_vllm_check.py`), the
csrc sampler fix, and replacing the PyTorch loop in `Inkling2Attention._attend` by a
kernel. The `(q, k, v, mlp)` message order is validated against the transformers
modeling code (tiny8), not against the real checkpoint packing (open question 1).

## 2026-11-22 — real checkpoint boots and answers on one V620 (b4 HIP 2, TP=1, eager, fp16)

`tools/rdna/inkling2/real_check.py` (offline `LLM`, `--max-model-len 2048
--max-num-batched-tokens 1024`, `skip_tokenizer_init`, chat-templated prompts like the
oracle). Box tree `~/w4a8_runs/leapquant/tree-inkling2`, log
`~/w4a8_runs/inkling2-work/Lead/real3.log`.

* Weights load in 6.5 s, "Model loading took 17.74 GiB", engine boot 32 s.
* First-step top-8 of "What is the capital of France?" is the CPU oracle's top-8 in the
  same order (`<|message_model|>` 0.0, then 䁱, `user`, `<|content_thinking|>`, 鋽, ...;
  logprobs −21…−22). Greedy text: `<|message_model|><|content_thinking|>The user asks a
  simple factual question: capital of France. Need answer accurately. France's capital
  is Paris. ...`. Second prompt also coherent. Not yet a numeric comparison against the
  oracle's full logits (oracle prints raw logits, the engine log-probs).
* Speed is not a result: 22-token prompt + 32 decoded tokens took 11 s (eager,
  PyTorch attention loop, `block_size_m=8` expert kernels).

Findings of this session:

| # | symptom | root cause | fix |
|---|---|---|---|
| 1 | oracle top-8 was punctuation/digit noise for every message-slot order | `ref_checkpoint.py` copied `w13` verbatim; the checkpoint interleaves gate/up rows (`Interleave(dim=1)`) | de-interleave in the oracle (cache files `l<N>.deint.safetensors`, converted from the old cache by a pure row permutation) and in `model.py::copy_gate_up_deinterleaved` |
| 2 | `torch.OutOfMemoryError` in `process_weights_after_loading` (29.6 GiB allocated) | `load_weights` kept `params` / `buffers` dicts alive, so the checkpoint-layout experts were never freed while the repacked copies were allocated | `del params, buffers, target, tensor` before the repack loop |
| 3 | `ValueError: ... 6.77 GiB KV cache is needed` at `--max-model-len 4096`, 8192-token chunks | conv-state owners use 8-token blocks and `SlidingWindowSpec` worst case `window + max_num_batched_tokens`: 6 groups x ~1025 blocks of a 2 MiB page set | smaller `--max-num-batched-tokens`; **open design item**: one conv state slot per request instead of per token |
| 4 | tiny acceptance `last/top1 0/3` | `tiny_vllm_check.py` prompts (seed 1234) differ from the `prompts.json` that `I2Real/regen.sh` fed to `tiny_reference.py` | regenerate refs with the seed-1234 prompts; after that tiny1/tiny8 top-1 3/3 and tiny1 greedy ids equal the reference for 8/8, 8/8, 7/8 steps; the abs(dlogp) failures are the router-chaos noise described above (tiny1 last-position mean 0.009 / 1.05 / 0.139, same as the CPU fp16 emulation) |

The oracle's message-slot order sweep cannot separate the orders at the first token
(all give `<|message_model|>` at +21 logits); `q,k,v,m` (the order validated against the
transformers code on tiny8) stays.

## 2026-11-22 — tiny acceptance: router near-tie flips, and a same-step prefix-hit bug

Signals of the engine acceptance (`tiny_vllm_check.py`): tiny8 with
`max_num_batched_tokens=16`, 40-token prompt: last-position top-1 False, max abs dlogp
32.04, mean 6.33 (the 5- and 70-token prompts fine); concurrent requests differing from
the single-request runs (tiny8 default: other ids, max 5.14; tiny8 chunked 12.9; tiny1
default 6.35); tiny8 default decode of the 40-token prompt leaving the reference ids
after 2 steps (mean 0.72).

Method (throw-away tooling outside the repository, no debug code in the model:
`~/w4a8_runs/inkling2-work/I2Tiny/{i2debug,engine_debug,engine_analysis,make_teacher}.py`):
the engine runs in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) and class-level patches
record, per engine step and layer visit, the layer output, the attention output, the
router top-k ids and their margin (k-th minus (k+1)-th selection score). They can also
*impose* router decisions taken from the fp32 CPU run (the teacher: the paged model with
exact fp32 experts, equal to the transformers reference to 2e-3) by position, matching the
requests to teacher sequences by token content. The free runs reproduce the acceptance
numbers (tiny8 chunked, 40 tokens: top-1 False, mean 6.3268, max 32.039), so these are the
acceptance runs.

1. **Every deviation starts at a router near-tie.** In every free run the first router
   decision that differs from the fp32 run has an fp32 margin of 6e-5 .. 1.9e-3 (fp16
   engine margin 1.1e-4 .. 1.5e-3); the layer outputs before it are at fp16 noise
   (relative error <= 1.6e-2 tiny8, <= 5e-3 tiny1), after it O(1) (0.85 .. 1.6):

   | run | first flip | margin fp32 / engine | layer-output rel. error before / after |
   |---|---|---|---|
   | tiny8 chunked, p40 alone (signal 1) | chunk 2, layer 3 visit 1, position 28 | 1.9e-4 / 1.4e-3 | 8.5e-3 / 1.43 |
   | tiny8 default, p40 alone (signal 3) | first decode token, position 40, layer 3 visit 2 | 6.1e-5 / 3.5e-4 | 1.6e-2 / 1.58 |
   | tiny8 default, batch of 3, p40 (signal 2) | prefill, layer 3 visit 1, position 28 | 1.9e-4 / 3.7e-4 | 5.9e-3 / 1.35 |
   | tiny8 default, batch of 3, p70 | prefill, layer 2 visit 1, position 45 | 6.2e-4 / 2.0e-4 | 4.1e-3 / 1.55 |
   | tiny1 default, p40 alone | prefill, layer 2 visit 1, position 5 | 2.3e-4 / 1.1e-4 | 2.7e-3 / 1.01 |
   | tiny1 default, p70 alone | prefill, layer 3 visit 0, position 10 | 1.3e-4 / 1.9e-4 | 3.0e-3 / 1.26 |

   Which near-tie flips depends on the GEMM shapes: the 40-token prompt of tiny8 has no
   flip in its default single-request prefill (last-position mean 0.0525), 40 flipped
   decisions in the batch of 3 and 60 when chunked. The decode signal is one decision at
   the first decode token (position 40, margin 6.1e-5): decode step 0 is off by mean
   1.39 and the ids leave the reference one token later.

2. **With the fp32 router decisions imposed every scenario agrees with the reference**:
   ids equal the reference for 8/8 steps for every request of every forced run, and the
   last-position mean abs dlogp (prompts 5 / 40 / 70) is at the noise floor:

   | scenario | tiny8 default | tiny8 chunked | tiny1 default | tiny1 chunked |
   |---|---|---|---|---|
   | alone, free routing | 0.0216 / 0.0525 / 0.0420 | 0.0216 / 6.327 / 0.0423 | 0.0090 / 1.052 / 0.139 | 0.0090 / 1.053 / 0.139 |
   | alone, forced | 0.0216 / 0.0525 / 0.0352 | 0.0216 / 0.0447 / 0.0343 | 0.0090 / 0.0240 / 0.0126 | 0.0090 / 0.0258 / 0.0120 |
   | batch of 3, forced | 0.0241 / 0.0584 / 0.0300 | 0.0234 / 0.0481 / 0.0297 | 0.0090 / 0.0186 / 0.0115 | 0.0090 / 0.0250 / 0.0105 |

   The worst layer-output relative error over all visits and positions of the forced runs
   is 1.0e-2 for the 5-token and 1.4e-2 .. 1.8e-2 for the 40- and 70-token prompts
   (tiny8), 4.2e-3 .. 6.6e-3 (tiny1), the same alone, batched and chunked, and of the
   order of the CPU fp16 emulation with exact experts (1.3e-2 tiny8, 4e-3 .. 6e-3
   tiny1): the expert kernels add no visible noise on top of fp16.

3. **No cross-request state.** A batch of 5 with a dummy 23-token request in front and
   the 40-token prompt twice (rows 1 and 4), the 70- and 5-token prompts in between,
   forced: 8/8 ids for every request; the two copies differ per visit by a relative
   1.0e-2 (tiny8) / 4.4e-3 (tiny1), less than each differs from the teacher (1.6e-2 /
   5.5e-3), and in the logprobs by at most 0.147 / 0.050 over 8 steps (chunked: 0.113 /
   0.044). That is rounding noise from the batch row. A contaminated state would be
   O(1). The same holds for the chunked batches (copies cannot be told apart by row
   there; every request is within 1.8e-2 of the teacher per visit).

4. **The flips are noise-level events.** On the path pinned to the teacher the engine's
   own (unforced) decisions differ from the fp32 ones in 0 .. 5 of 96 .. 1232 decisions
   per prompt, all at fp32 margins <= 1.9e-3 (0.1 .. 2.4 % of all decisions have a margin
   that small).

5. **What the real engine hands over** (block tables of the 70-token prompt, tiny1
   chunked, dumped from the model): the hybrid 64 / 4 / 4 / 64 layout with scattered
   block ids; the null block 0 for the blocks below the window of the first scheduled
   token (a conv group, block size 4, at the second 16-token chunk: columns 0-2 null,
   live blocks from position 12 on); slot mapping
   `block * block_size + position % block_size`; *stale non-null block ids in the unused
   tail columns* of the persistent block-table row (ids of the previous request in that
   row), never read. This is what `paged_harness.py` emulates, except that it pads the
   tail with the null block (a read there fails the NaN check).

6. **A state bug found on the way, fixed: a same-step prefix hit read unwritten conv
   history.** `KVCacheManager.allocate_slots` registers the full blocks of the tokens it
   schedules in the prefix cache *before* they are computed
   (`coordinator.cache_blocks(request, computed + new)`), so a request scheduled later in
   the same pass (identical prompts, `n > 1` samples, a shared system prompt) can hit
   blocks that the first request only writes in the very step both run in. `conv_taps`
   read the history of the first tokens of the hit (the previous three positions, owned
   by the other request) before it wrote the inputs of the step: unwritten or stale
   slots. K/V were not affected (`kv_write` of the whole batch precedes `_attend`).
   `conv_taps` now writes first. Prefix caching is on by default for this architecture
   (`ModelConfig.is_prefix_caching_supported` is true for generative models). CPU, on
   HEAD: `test_conv_taps_of_a_prefix_hit_on_blocks_written_in_the_same_step` (state
   level) and `test_prefix_cache_hit_on_blocks_written_in_the_same_step` (whole model)
   fail with non-finite output (the poisoned page is read; real memory would give wrong
   finite values); with the fix they match the dense oracle / the reference (state: max
   abs diff 1e-7 .. 1e-6; model, fp32: max abs dlogit 3e-5 .. 1.5e-4).

   On the engine (fp32 router decisions imposed, prefix caching on, three requests `p2`,
   `p2`, `p2[:67]` in one `generate` call: the scheduler hands the model one 79-token
   step, `70 + 6 + 3` tokens, the two hits with `cached 64`; the runner puts the short
   requests first):

   | conv_taps | checkpoint | request | ids equal ref | last top-1 | mean / max abs dlogp | first layer visit above 5 % |
   |---|---|---|---|---|---|---|
   | HEAD (reads first) | tiny1 | owner | 8/8 | True | 0.0120 / 0.055 | none (6.5e-3 worst) |
   | HEAD (reads first) | tiny1 | hit | **0/8** | **False** | **2.72 / 11.09** | visit 0 (layer 0), rel. 0.90 at position 64 |
   | fixed | tiny1 | hit | 8/8 | True | 0.0125 / 0.052 | none (5.2e-3 worst) |
   | HEAD (reads first) | tiny8 | hit | **0/8** | **False** | **3.83 / 15.50** | visit 0 (layer 0), rel. 0.75 at position 64 |
   | fixed | tiny8 | hit | 8/8 | True | 0.0319 / 0.122 | none (9.0e-3 worst) |

   Hits on blocks of a *finished* request (five sequential `generate` calls, `cached 64`
   for the repeated 70-token prompt and for its 67-token prefix) were correct with both
   versions: ids 8/8, mean 0.0126 (tiny1) / 0.0347 (tiny8). With chunked prefill
   (16 tokens per step, tiny1) the three requests no longer share a step; both full
   prompts are correct (cached 64, ids 8/8, mean 0.0121 / 0.0122).
   The same-step tests are the GPU-free guard; the engine runs are
   `engine_debug.py --prefix-caching [--old-conv]` (`--old-conv` swaps the HEAD
   `conv_taps` in).

7. **The engine is deterministic run to run.** Three identical free-routing runs of the
   40-token prompt alone and of the batch of three, on tiny8, give bitwise equal ids and
   logprobs and a maximum layer-output difference of exactly 0: the router flips of
   finding 1 are deterministic functions of the GEMM shapes, not races.

Consequences. `tiny_vllm_check.py` judges a *chaotic* checkpoint with free routing: its
ids / top-1 / mean checks on the 40- and 70-token prompts and its batch-vs-single check
(`worst < 0.05`, below the 0.11 .. 0.15 that two identical forced tiny8 requests differ
by at different batch rows) fail for reasons that are properties of the random model, not
of the serving code. The deterministic verdict is the forced-route run. CPU side, the
fp16 tests of `test_paged_model.py` now impose the fp32 run's expert choice by position
for chunked prefill (chunk 7 / 16, KV block 4 / 64), a mixed batch and cached decode
(`test_fp16_*_given_the_same_expert_choice`: mean abs dlogp 7e-3 .. 2.2e-2), and the
8-layer fixture uses the seed of the GPU checkpoint, so the CPU free-routing numbers show
the same kind of deviation as the engine (last position of p40: 6.0, p70: 1.8).
`PagedWorld.fork_prefix` emulates a prefix-cache hit (shared reference-counted blocks, the
null block before the window of the windowed groups, optionally on blocks the owner only
writes in the same step); `test_prefix_cache_hit_*` run a hit request and the owner of its
blocks in one batch and decode them, let the owner finish while the hit is alive, and
schedule the hit and its owner in the same step in both batch orders (fp32 max abs dlogit
4e-5 .. 1.5e-4). The scheduler's own hit logic (`find_longest_cache_hit`, alignment) is
vLLM code; the engine runs above exercise it (hit length 64 = the 64-token block of the
hybrid layout).
`test_long_sequence_past_every_window_matches_reference` runs 200 tokens (chunks of 16 /
33, then single-token steps through positions 79 and 143, where the 64-token blocks of
the windowed groups become free) with the freed pages poisoned: max abs dlogit
7e-5 .. 1.7e-4. The tiny prompts of the engine runs (<= 78 tokens) never free such a
block, so that part is CPU-only.

## 2026-11-22 — integration part A: the serving path on the HIP kernels (eager, TP=1)

`Inkling2LoopForConditionalGeneration` now serves from the native kernels
(`csrc/rocm/inkling2_{attn,glue,linear,vit,dspark}_rdna2.cu` plus the existing expert
GEMMs); the eager torch implementation stays as the reference.  Graph capture is not
enabled (part B).  Kernel schemas and wiring recipes are in `inkling2-attn-kernels.md`,
`inkling2-glue-kernels.md`, `inkling2-linear-kernels.md`, `inkling2-vision.md`,
`inkling2-dspark.md`.

### Design

1. **Two paths, one class, no fallback.** `Inkling2Model.forward` dispatches to
   `forward_reference` (the torch code of the port, unchanged: what the CPU harness tests
   run and the oracle of the GPU parity tests) or `forward_hip`.  The mode is HIP when the
   default device is cuda/hip and reference on CPU; `VLLM_INKLING2_PATH=reference|hip`
   overrides it (the tools take `--path`), `Inkling2LoopTextForCausalLM(serving=...)` is
   the switch of the tests.  `prepare_hip` repacks and then deletes the checkpoint-layout
   weights, so the torch code cannot run on a serving model (a test asserts the
   `AttributeError`) and `forward_hip` has no torch compute: a missing op raises.
2. **One layer visit** (`Inkling2DecoderLayer.forward_hip`, 16 kernels on average per
   visit, 48 visits per forward): block residual + layer norm
   (`inkling2_glue_attn_res`, with messages followed by `msg_gate_down` / `msg_gate_mix`),
   the fused attention-input GEMV pack (q, o-gate, relative bias, sink, k, v and the
   dynamic-conv taps of full layers in one launch), `inkling2_kv_prep`,
   `inkling2_attention` (split count and workspace from the metadata), `inkling2_sconv_static`
   (post-attention), the MLP-input `attn_res` (+ gates), `inkling2_glue_moe_route`, expert
   GEMM 1, `inkling2_glue_swiglu`, expert GEMM 2, `inkling2_sconv_static` (post-MLP).
   The model ends with the final norm and the `lm_head` pack (softcap, vocab mask,
   optional fused argmax: `greedy_token_ids`).  MessageGate and the router use the fused
   glue ops, not the standalone linear-pack router / controller (fewer launches).
3. **Weights are repacked once and freed.** The attention pack, the o_proj pack, the
   stacked gate / layer-norm weights, the router and the lm_head pack replace the
   checkpoint tensors (`del module.weight`).  Real checkpoint: "Model loading took
   18.08 GiB" against 17.74 GiB on the torch path, and the engine logs the persistent
   workspace of 0.33 GiB for 1024 tokens per step, so the packs add 0.01 GiB.  The router
   and gate weights of the real checkpoint are not exactly representable in fp16 (0/24
   routers, 0/12 loop layers), so they stay fp32 (FMA path) and no value is ever rounded.
   Tiny models, census of the tensors each model owns (cache owners and the workspace
   excluded): HIP 12.44 / 16.44 MiB against 12.52 / 16.67 MiB for the reference (tiny1 /
   tiny8), `test_serving_weights_are_held_once`.
4. **Persistent scratch, no host syncs.** `Inkling2GlueWorkspace` (`amd/glue.py`) owns every
   per-step buffer (block states, messages, MoE scratch, attention workspace and split
   buffers, taps), sized for `MAX_MOE_TOKENS` / `max_num_batched_tokens`; `SeqIdxScratch`
   (`amd/state.py`) the token -> request map.  `test_forward_allocates_nothing_and_never_syncs`
   warms one forward up (memoised workspace views, lazy module state), then runs the HIP
   forward under `set_sync_debug_mode("error")` and checks that the allocator count does
   not move.  `embed_input_ids`, `compute_logits` and `greedy_token_ids` allocate (they run
   on the runner side, outside any future graph).
5. **Same-step prefix hits.** A hit request whose conv taps live in blocks the owner only
   writes in the same step needs the conv write phase to finish before the read phase.
   The ops cannot order the two inside one launch, so `conv_passes` launches `kv_prep` and
   `sconv_static` twice when `max_query_len > 1`.  Mutation check: with one pass
   (`conv_passes = lambda md: 1`) all 8 `test_prefix_cache_hit_on_blocks_written_in_the_same_step`
   cases fail.  Residual corner: a batch of single-token chunks only where a one-token
   tail hit shares the step with the owner that completes that block; a csrc write /
   read-phase split would remove it and the double launch.
6. **Vision.** The served class is `Inkling2LoopForConditionalGeneration(Inkling2MultiModalMixin,
   Inkling2LoopTextForCausalLM, SupportsMultiModal)`; with `--limit-mm-per-prompt
   image=0` the tower is not built (the real run boots that way).  The vision kernels
   take the real 20 x 20 patch geometry only, so the tiny engine checks boot with
   `hf_overrides={"vision_config": None}`.
   Real checkpoint with the tower (`real_check.py --vision`): "Model loading took 18.68
   GiB" (0.60 GiB for the tower), the encoder cache is profiled at start-up with one
   maximum-size image through the vision kernels, the text answers are unchanged, and a
   synthetic 448 x 336 image of a red disc on blue (132 prompt tokens) is described as "a
   red circle centered on a blue background".

### Measurements (real checkpoint, one V620, eager, TP=1, `profile_decode.py`)

22-token prompts, 8 decode steps; "clean" = unprofiled wall time.

| | torch path (baseline) | HIP path |
|---|---:|---:|
| decode step, batch 1 | 341.7 ms | **23.7 ms** (14.4x) |
| decode step, batch 4 | 417.0 ms | **35.8 ms** (11.6x) |
| prefill, 22 tokens, batch 1 | 366.8 ms | 75.6 ms (73.4 .. 76.7) |
| prefill, 4 x 22 tokens | 481.7 ms | 170.1 ms (166.4 .. 172.1) |
| prefill, one 1000-token prompt | | 1262 ms (792 tokens/s) |
| kernels per decode step, batch 1 / 4 | 23,826 / 28,242 | **779 / 779** |
| kernels per prefill step, batch 1 / 4 | | 978 / 978 |
| GPU busy share of the clean decode wall, batch 1 | | 93.9% |
| model memory ("Model loading took") | 17.74 GiB | 18.08 GiB (incl. 0.33 GiB workspace) |

Batch-1 decode is 22.3 ms of GPU time in 779 kernels (mean 28.6 us): attention-input pack
6.15 ms (27.6%), expert GEMM 1 4.56 ms (20.4%), expert GEMM 2 2.40 ms (10.7%), `lm_head`
2.13 ms (9.5%), `o_proj` pack 1.94 ms (8.7%), message gates 1.13 ms, `attn_res` 0.99 ms,
attention 0.85 ms, route 0.79 ms, `sconv` 0.45 ms, `kv_prep` 0.43 ms, swiglu 0.27 ms,
the rest below 0.2 ms.  The step is GPU-bound now: [INFERENCE] a HIP graph can remove at
most the 1.4 ms of idle time (6%) plus the host side of the step; the levers are kernel
time (attention-input GEMV, expert GEMMs, `lm_head`).  The launch count fell 31x (decode);
the forward, launch to completion, fell from 333.8 ms to 20.5 ms (11.1 ms of it host launch
time).

### Validation

| gate | result |
|---|---|
| build, all five `.cu` files in one tree | exit 0 (831 s); `ops_check` missing: `[]` |
| CPU suite (`tests/models/inkling2_loop` + the three `test_inkling2_*` files) | 519 passed, 76 skipped |
| `test_inkling2_linear_rdna2.py` (the DenseLinear auto-pick, first run on a GPU) | 677 passed |
| `test_serving_path_gpu.py` (116 tests, tiny1 / tiny8), final tree | 116 passed (152 s); the run before the last tolerance change had 1 failed, fixed |
| `real_check.py` (real checkpoint, 32 tokens) | coherent, see below |
| `real_check.py --prefix-caching --long-prompt-tokens 1500` | PASS |
| `real_check.py --vision` (tower loaded, one image request) | coherent text answers, correct image description |
| `tiny_vllm_check.py` tiny1-stable: default / chunked / prefix / chunked + prefix | 4 / 4 `RESULT PASS` |
| `tiny_vllm_check.py` tiny8-stable: default / prefix | `RESULT PASS` (`--tol-batch 0.15`) |
| `tiny_vllm_check.py` tiny8-stable: chunked, chunked + prefix | `decode/ids` fails on one flip (finding 5), everything else passes |

`test_serving_path_gpu.py` replays every layer visit of the reference path on the HIP layer
from the recorded inputs (teacher forced, so errors cannot compound) and compares the
router choices: worst relative error 1.07e-3 .. 1.26e-3 per token on tiny8-stable with 0
router rows differing in all four scenarios (prefill 70, chunked 70/16, prefill + decode,
batch).  Model level against transformers: mean |dlogp| 0.028 .. 0.033 (prefill, chunked,
random schedules, prefix hits, same-step hits); HIP against the reference path 0.017 ..
0.025.  The 200-token window test is teacher forced as well: the free-running comparison
flips a router near-tie on one chunking (HIP against reference 0.27 where both are 0.27
from transformers on another).

Engine, tiny checkpoints (HIP shaped, router bias x100, last-position mean |dlogp| against
transformers per prompt of 5 / 40 / 70 tokens, greedy ids equal out of 8):

| checkpoint, mode | HIP path | torch path (same engine, same checkpoint) |
|---|---|---|
| tiny1, default | 0.0068 / 0.0133 / 0.0128, ids 8/8 x3 | 0.0072 / 0.0140 / 0.0104, ids 8/8 x3 |
| tiny1, chunked 16 | 0.0068 / 0.0125 / 0.0163, ids 8/8 x3 | 0.0072 / 0.0118 / 0.0096, ids 8/8 x3 |
| tiny8, default | 0.0218 / 0.0936 / 0.0286, ids 8/8 x3 | 0.0238 / 0.0884 / 0.0408, ids 8/8 x3 |
| tiny8, chunked 16 | 0.0218 / 0.0518 / 0.0307, ids 8/8, 3/8, 8/8 | 0.0238 / 0.0735 / 0.0355, ids 8/8 x3 |

The old tiny checkpoints (head dim 32, d_rel 8) are rejected by the kernels, so the
statistics quoted before (tiny1 [0.009, 1.05, 0.139], ids 8/8, 8/8, 7/8) cannot be
reproduced on the HIP path; `make_tiny_checkpoint.py --head-dim 128 --d-rel 16
--num-heads 6 --num-kv-heads 2 [--router-bias-scale 100]` writes the HIP-shaped ones.
On them both paths are statistically the same (the torch path's tiny8 default mean, 0.0510,
is just above the 0.05 tolerance of the check, the HIP path's 0.0480 just below) and better
than the old free-routing numbers because the stable checkpoints do not flip routes.
Free-routing HIP-shaped tiny8 (statistics only): reference 2.965, HIP 0.413 (chaotic, see
the previous section).

Real checkpoint, first-step top-8 after the chat template, torch path -> HIP path:

* "What is the capital of France?": `<|message_model|>` 0.0, 䁱 -21.04 -> -21.09, user
  -21.13 -> -21.17, `<|content_thinking|>` -21.40 -> -21.22; ranks 5 .. 8 hold the same
  four tokens (鋽, two byte-fallback tokens, суш) in a different order at -21.9 .. -22.7.
  The 32 generated tokens are identical: "The user asks a simple factual question:
  capital of France. Need answer accurately. France's capital is Paris. Need final
  concise. Ensure".
* "Write one word: hello": `<|message_model|>` 0.0, 鋽 -19.35 -> -19.32, U+200B -20.60 ->
  -20.29, `<|content_thinking|>` -22.43 -> -22.28, U+FEFF -22.57 -> -22.32, then the tail
  permuted.  Both answers are coherent and agree up to "Need answer", then differ at a
  near-tie.

### Findings

1. **A dtype bug the GPU run found.** `inkling2_glue_msg_gate_mix` needs the forget and
   the gain weights in one dtype; the first version picked fp16 per tensor (46 failures,
   all tiny1).  `_stack_gates` now takes fp16 pair-interleaved weights only when both are
   exactly representable, else fp32 for both.
2. **The HIP path is not run-to-run reproducible with more than two experts per token.**
   The expert GEMMs add a token's expert outputs into the pre-zeroed output with a packed
   fp16 CAS atomic add (`moe_lowp_dot2_rdna2.cu`; `moe_accum_rdna2.cuh` documents it as
   "order-dependent at fp16 precision"), both paths share them.  Two experts commute, so
   `test_run_to_run_difference_by_experts_per_token[2]` is bitwise equal (and every tiny
   engine rerun above is); with eight of sixteen experts and the choice fixed by the bias
   the max logit difference of five identical runs is 0.20 .. 0.24 on a scale of 19.6
   (1.0 .. 1.2%).  On the real checkpoint two cold runs of one prompt differ in the
   top-8 logprobs by 0.00 .. 0.14 (22 tokens) and 0.35 .. 1.19 nats (1510 tokens, tail ranks
   at -21), the router amplifying the rounding.  Greedy ids and top-1 stay the same in every
   HIP run we made.  A deterministic reduction (per-k rows summed in order by the next glue
   op, or a fixed-point accumulation like the one `moe_accum_rdna2.cuh` has for the W4A16
   kernels) is the way to bitwise
   reproducibility; it matters for graph replay comparisons and speculative verification.
3. **Prefix caching on the real checkpoint.** A 1510-token prompt (cold: chunks of 1024 and
   486 tokens), cold prefill 1.95 s, with a hit 0.12 s; the generated ids, the text and
   the first-step top-1 are identical cold, cold again and with the hit; top-8 logprob
   differences hit against cold 0.48 .. 1.74 against 0.35 .. 1.19 between two cold runs
   (finding 2), so `real_check.py` judges the hit against the measured floor
   (`hit <= 2 x floor + 0.1`).
4. **Engine boot of the tiny checkpoints.** The first GPU engine run failed on the vision
   tower (`kernels are written for 20x20 native patches`): the tiny `vision_config` is for
   the transformers reference; the engine checks drop it through `hf_overrides`.
5. **tiny8 chunked, n = 40: one greedy flip.** At decode step 3 the HIP path picks a
   token that is 1.02 nats below the reference's top-1 (the agreeing steps show
   |dlogp| <= 0.27); the torch path does not flip there but fails its own batch-versus-single
   check in that mode (max 0.164, ids differ).  The teacher-forced layer parity (validation
   section: 0 router rows differing, relative error 1.3e-3) is the kernel-level verdict; this
   flip is attributed to the router-chaotic random model [INFERENCE, not isolated to a
   layer visit].

### What part B (graphs, TP, long context) must handle

* `Inkling2MetadataBuilder._cudagraph_support` is `NEVER`; it needs a real value and
  `build_metadata_for_cudagraph_capture`.  `Inkling2Metadata` has to be built into
  persistent in-place `block_table`, `query_start_loc`, `seq_lens` and `slot_mapping`
  buffers; `seq_idx` already lives in `SeqIdxScratch` (filled per step by
  `build_qsa_metadata_rdna2` outside the graph).
* The attention split count and workspace depend on `max_query_len`, the request count
  and `max_model_len` (`inkling2_attention_splits`, `inkling2_attention_workspace_numel`):
  fix them at capture time (decode graphs with `max_query_len = 1`, spec-verify / mixed
  graphs with `1 + num_speculative_tokens`).  Padded requests need `seq_lens = 0`, padded
  tokens `slot_mapping = -1` (the attention doc also asks `seq_idx = -1`; the scratch reads
  padded tokens as request 0, harmless in eager where T is the real token count).
* Decide `conv_passes` (twice-launched `kv_prep` / `sconv_static` for multi-token steps):
  inside a graph it doubles those nodes; the clean fix is the csrc write / read-phase
  split (item 5).
* The V2 runner must capture FULL graphs around `forward` only: `embed_input_ids`,
  `compute_logits` and `greedy_token_ids` allocate; `ws.bind(T)` views have to be created
  per captured size before capture.  `greedy_token_ids` is not wired into the sampler yet
  (runner-owned).
* TP: o_proj emits an fp32 partial and needs an all-reduce, `lm_head` shards the vocab
  (`inkling2_linear_pack_lm_head(tp_size, tp_rank)`), the router is replicated,
  `Inkling2LoopMoE` asserts TP = 1, `q_scale` / `rel_scale` need per-rank slices, the KV
  and conv owners split by KV head.
* Long context: `check_kernel_shapes` and the workspace bound (`attention_workspace_numel`)
  were validated up to `max_model_len` 2048; TP = 1 maximum length and TP = 2/4 at
  1,048,576 need the split count and the workspace re-derived for those lengths.
* Speed: the step is GPU-bound (93.9% busy); [INFERENCE] a graph gains at most ~6% of a
  batch-1 decode step, so the next gains are kernel time (attention input GEMV 27.6%,
  expert GEMMs 31%, `lm_head` 9.5%).
