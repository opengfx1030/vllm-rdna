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

## 2026-11-22 — integration part B: HIP graphs (FULL_AND_PIECEWISE) and the DSpark draft on the V2 runner

Branch `feat/inkling2-mini-hip`.  Real checkpoint, TP = 1, one V620 (HIP 2) unless a tiny
checkpoint is named.  The neighbouring cards ran other jobs at the same time, so absolute
times are indicative; the modes are compared inside the same session.  Tools:
`tools/rdna/inkling2/graph_check.py` (engine-level eager / graph / speculative runs and
`--compare`, which is CPU-only), `profile_decode.py --cudagraph-mode ... [--spec-k 7]`
(kernel counts, GPU-busy time), and the test file
`tests/models/inkling2_loop/test_serving_path_graph_gpu.py`.

### What `FULL_AND_PIECEWISE` does for a model that is not torch.compile'd

* The model is not compiled, so there is no piecewise compilation.  The fork's breakable
  CUDA graphs (`vllm/compilation/breakable_cudagraph.py`) record the *eager* forward as a
  chain of graph segments separated by eager breaks.  An architecture opts in through
  `default_breakable_cudagraph_architectures` (`vllm/config/vllm.py`, one line for
  `Inkling2LoopForConditionalGeneration`); without the entry the config silently downgrades
  `FULL_AND_PIECEWISE` to `FULL_DECODE_ONLY`.
* FULL graphs serve the uniform batches: decode (q = 1) and speculative verify (q = 1 + k).
  The non-speculative default sizes are 1, 2, 4, 8 requests (`decode_cudagraph_capture_request_sizes`).
* Breakable (PIECEWISE) graphs serve prefill, chunked prefill and mixed batches, per padded
  token count (`prefill_cudagraph_capture_token_sizes`; the default is 1, 2, 4, 8, 16
  without a draft and 1, 2, 4, 8, 16, 24 .. 128 with one).  The one eager break per layer
  visit is `attend_hip` (`@eager_break_during_piecewise_capture`): its grid depends on the
  longest query and the live split-K plan of the step.  48 layer visits give 49 graph
  segments; `BreakableCUDAGraphCapture.replay` runs the segments and the eager calls in order.
* A FULL graph for a mixed batch is infeasible for the same reason (the attention grid and
  the split-K plan are data dependent).  A batch above the largest captured size runs eager.
* The V2 runner captures `forward` only; `embed_input_ids`, `compute_logits` and the greedy
  sampler run eagerly around it, so their allocations never meet a capture.

### Capture contract (the part-A list "What part B must handle")

| Part-A item | Part B |
|---|---|
| `Inkling2MetadataBuilder._cudagraph_support = NEVER` | `UNIFORM_BATCH`; the classmethod `get_cudagraph_support` still answers `NEVER` when the HIP ops are not used (reference path) |
| persistent metadata buffers, fixed splits / workspace | the V2 runner owns the persistent buffers.  Capture records `max_seq_len = max_model_len` and `max_query_len = q`, so the grids, split-K plan and workspace are frozen at capture; the live lengths, slots and sequence indices are device tensors read at replay |
| padded requests need `seq_lens = 0`, `slot_mapping = -1`, `seq_idx = -1` | padded tokens map to request 0 (`SeqIdxScratch.fill` covers the padded token count, `make_metadata` slices `slot_mapping` to it); the conv and KV writers act only for `slot >= 0`; the router is NaN-hardened; `Inkling2GlueWorkspace.attn_out` is zero-initialised |
| `conv_passes` double launch | isolated in `attention.conv_passes(md)`: 2 for `max_query_len > 1` and while tracing a breakable graph, else 1 (a q = 1 FULL decode graph runs the kernel pair once).  The clean fix is a write / read-phase split inside the kernel (csrc, not done) |
| FULL graphs around `forward` only | confirmed, see above |

Splits frozen at `max_model_len` are the one place where a graph differs from eager: the
attention kernel is documented as split-dependent to its tolerance, and the measured
difference (below) is of that order.

### Measurements

Decode, real checkpoint, `max_model_len` 2048, prefix caching on, `graph_check.py` bench: ms per
generated token of one request, `(t_64 - t_1) / 63`, median of 3, b1 = one prompt, b4 / b8 =
4 / 8 distinct prompts, tok/s is the total over the batch.

| mode | b1 | b4 | b8 | graph memory | KV cache |
|---|---|---|---|---|---|
| eager (runs A / B) | 21.07 / 21.06 ms (47.5 tok/s) | 34.79 / 34.82 ms (115.0 tok/s) | 56.30 / 55.86 ms (142.1 / 143.2 tok/s) | 0 | 4,409 tokens |
| `FULL_DECODE_ONLY` | 20.02 ms (49.9 tok/s) | 33.96 ms (117.8 tok/s) | 55.21 ms (144.9 tok/s) | 0.09 GiB | 4,352 tokens |
| `FULL_AND_PIECEWISE` | 20.03 ms (49.9 tok/s) | 33.93 ms (117.9 tok/s) | 54.95 ms (145.6 tok/s) | 0.57 GiB | 4,070 tokens |
| `FULL_AND_PIECEWISE`, prefill sizes 1 .. 1024, `max_model_len` 8192 | 20.01 ms (50.0 tok/s) | 33.99 ms | 55.17 ms | 1.16 GiB | 13,441 tokens |

The graphs save 1.0 ms of a 21 ms step (4.9% at batch 1), 0.9 ms at batch 4 (2.5%) and about
1 ms at batch 8 (1.2 - 1.9%).  Boot takes 37 - 41 s with graphs (26 - 37 s eager).

Kernel view (`profile_decode.py`, async scheduling off, 22-token random prompts, GPU-busy
from the profiler):

| | eager | `FULL_AND_PIECEWISE`, prefill grid 1 .. 1024 (doubling) | `FULL_AND_PIECEWISE`, grid 1, 2, 4, 8, 16, 24 .. 128, 256, 512, 1024 |
|---|---|---|---|
| decode b1 | 23.1 ms wall, 22.3 ms GPU, 779 kernels | 21.1 ms, 18.9 ms GPU, 788 kernels | 21.1 ms, 788 kernels |
| decode b4 | 35.1 ms, 33.3 ms GPU, 779 kernels | 33.3 ms, 29.3 ms GPU, 788 kernels | 33.2 ms, 29.8 ms GPU, 788 kernels |
| prefill b1 (22 tokens) | 74.6 ms (978 kernels) | 97.1 ms (padded) | 80.9 ms (padded to 24, 976 kernels) |
| prefill b4 (88 tokens) | 169.0 ms | 222.0 ms (padded to 128) | 167.7 ms (exact size, 973 kernels) |
| prefill 1000 tokens | 1259 ms (794 tok/s) | 1300 ms | 1290 ms (padded to 1024) |

* The GPU time of a decode step falls by 15% under the graph (22.3 to 18.9 ms at batch 1:
  the launch gaps disappear) and the wall time by 1 - 2 ms.  The step is GPU-bound: the
  eager step is 94% busy, so a graph cannot give more than the gaps. `[INFERENCE]`
* **The only cost of the breakable graphs is padding.**  Padded rows go through the full MoE
  and dense work, so a padded step costs the time of the padded token count (1000 tokens
  padded to 1024: 1259 ms eager, 1290 ms replayed, +2.4%; 88 tokens padded to 128: +31%; the
  same 88 tokens on a grid that has the exact size: 167.7 ms against 169.0 ms eager).
  The 49 segments themselves are free: a throw-away probe (not in the repo) of 833 tiny kernels
  took 2.76 ms as 49 segments with eager breaks, 5.45 ms eager and 2.53 ms as one FULL graph.
* So PIECEWISE gives no speed-up for this model; it is kept because the recipe policy is
  `FULL_AND_PIECEWISE` and because it is correct (below).  Do not add a prefill grid finer
  than the traffic needs: every captured size costs graph memory (0.57 GiB default, 1.16 GiB for
  1 .. 1024 doubling) and an off-grid batch pays its padding.

Numerics (`graph_check.py --compare`, greedy 24 tokens, logprobs top-2 near-tie gaps):

| comparison | identical generations | max `|dlogp|` before the first divergence |
|---|---|---|
| tiny8 engine, mixed batches with 96-token chunked prefill (25 generations): eager vs eager | 25 / 25 | 0.045 (CAS noise floor) |
| tiny8: `PIECEWISE` vs eager | 25 / 25 | 0.0000 |
| tiny8: `FULL_DECODE_ONLY` / `FULL_AND_PIECEWISE` vs eager | 21 / 25 | 0.027 (the divergences are at near-ties of 0.019 - 0.053 nats) |
| real, 26 generations: eager A vs eager B (the floor) | 17 / 26 | 0.082 |
| real: eager B vs `FULL_AND_PIECEWISE` | 17 / 26 | 0.056 |
| real: eager B vs `FULL_DECODE_ONLY` | 19 / 26 | 0.076 |

* The real checkpoint's run-to-run floor is the fp16 CAS accumulation of the expert GEMMs
  (top-8 order dependent) plus the batch-dependent split-K plan; the graph and eager runs
  stay inside it.
* `test_serving_path_graph_gpu.py` (5 tests, tiny checkpoints on HIP 2): a replay is bitwise equal to the
  same frozen eager step; padded eager equals unpadded eager (0.0); the split-K plan frozen
  at capture (`max_seq_len = max_model_len`) differs from the live plan by 2.3e-2 on tiny1
  (logit scale 19.4) and 7.9e-2 on tiny8 (scale 22.4), the only graph-vs-eager difference
  `[INFERENCE]`; with top-8 experts the CAS noise is 0.198 eager vs eager, 0.188 graph vs
  eager and 0.187 graph vs graph (scale 19.8), so graph replay adds nothing measurable.
* Prefix caching with graphs, 1510-token prompt, `max_model_len` 2048: cold 2.34 s, warm 0.44 s
  (eager: 2.37 s and 0.46 s).  `FULL_AND_PIECEWISE`: cold, cold again and warm generate the same
  tokens; `FULL_DECODE_ONLY`: no divergence, max `|dlogp|` 0.039 (cold vs cold again) and 0.089
  (cold vs warm); eager: 0.093 and 0.106.  A 6002-token prompt at `max_model_len` 8192 takes
  8.59 s cold and 0.50 s warm; cold vs warm and cold vs cold again both flip at token 2 on
  a near-tie (top-2 -0.62 / -0.77 against -0.49 / -0.95), the same place, so it is the run-to-run
  floor and not the cache.
* Maximum length at TP = 1 was not re-derived in part B: 8192 boots and serves (KV cache 13,441
  tokens beside 8.99 GiB of KV memory, 1.16 GiB of graphs); the largest fitting length remains
  the open item of the long-context work.

### Cost of a decode step by row count (eager, real checkpoint)

`graph_check.py --batches 1,2,...,8 --tokens 64`, `b` distinct prompts per batch, ms per
generated token of one request = the step time with `b` rows, median of 2:

| rows | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| step ms | 21.07 | 25.50 | 30.98 | 34.81 | 45.28 | 48.99 | 53.04 | 56.13 |

An extra row costs 3 - 5.5 ms, except the fifth: +10.5 ms.  A verification of `1 + k` tokens is the
same work as `1 + k` decode rows, so it costs the row count's step time plus the draft.

Where the fifth row's time goes (`profile_decode.py --batch 5` against `--batch 4`, eager decode,
random-token prompts, GPU ms per step; the step medians are 35.1 and 45.1 ms):

| kernel | launches / step | batch 1 | batch 4 | batch 5 |
|---|---|---|---|---|
| `moe_gemm_nvfp4_kernel_rdna2<half, M>` | 88 | 5.33 ms (M = 1, 60.6 us per launch) | 11.21 ms (M = 4, 127.3 us) | 13.82 ms (M = 8, 157.1 us) |
| `moe_gemm_mxfp8_kernel_rdna2<half, M>` | 8 | 1.62 ms (M = 1, 202.2 us) | 5.90 ms (M = 4, 737.0 us) | 11.72 ms (M = 8, 1464.9 us) |
| `inkling2_linear::gemv_kernel` | 97 | not read | 10.43 ms (107.5 us) | 10.59 ms (109.2 us) |

The MoE kernels are templated on the row count (instances 1, 2, 4, 8) and the policy picked the
instance by the token count (1 -> 1, 2 -> 2, 3 - 4 -> 4, 5 and above -> 8), so a 5-row batch ran
the 8-row instance, and the mxfp8 kernel cost about 184 us per row and launch at both M = 4 and M =
8.  This paragraph first explained that with "the kernel does not amortise the weight
dequantisation over the rows".  That was wrong: a block already dequantises each weight once for all
its rows.  The cause was the policy and the register spills of the old mxfp8 kernel (part B2
below).  The kernel is what set the price of a speculative verification: an 8-row batch ran the
same 8-row instance `[INFERENCE]`, so 11.7 ms of its step were the mxfp8 kernel and 13.8 ms the
nvfp4 one.  The step times also vary with the routing (min 29.7 ms, max 43.3 ms at batch 4, same
prompts).

### Decode capture sizes: a padded row is real work

`graph_check.py --decode-sizes`, 64 tokens, ms per generated token of one request = the step time of
`b` rows (b = 1, 2, 4, 8 are exact sizes of the default grid and need no comparison):

| rows | eager | `FULL_AND_PIECEWISE`, default decode sizes 1, 2, 4, 8 | `FULL_AND_PIECEWISE`, `decode_cudagraph_capture_request_sizes` 1 .. 8 |
|---|---|---|---|
| 3 | 30.98 ms | 35.44 ms (+14%, padded to 4) | 30.81 ms (-0.5%) |
| 5 | 45.28 ms | 61.04 ms (+35%, padded to 8) | 43.78 ms (-3.3%) |
| 6 | 48.99 ms | 60.70 ms (+24%, padded to 8) | 47.65 ms (-2.7%) |
| 7 | 53.04 ms | 58.51 ms (+10%, padded to 8) | 51.59 ms (-2.7%) |

The padded rows of a FULL decode graph run the whole MoE and dense work, so a batch between two
captured sizes is *slower* than eager (up to 35% here) and a dense grid costs nothing: the graph
memory is 0.57 GiB for both grids.  The recipe for `max_num_seqs` 8 is
`--compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE",
"decode_cudagraph_capture_request_sizes": [1, 2, 3, 4, 5, 6, 7, 8]}'`; with a larger
`max_num_seqs`, list every request count the traffic reaches (counts above 8 were not measured).

### The DSpark draft on the V2 runner

`LLM(speculative_config={"method": "dspark", "num_speculative_tokens": 7},
compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE"}, enable_prefix_caching=True)`,
real checkpoint, TP = 1, greedy.  What it took (no csrc change; the details are in
`inkling2-dspark.md`):

* The taps: `Inkling2DSparkConfig.eagle_aux_hidden_state_layer_ids` is the three entry visits
  (5, 23, 41); it also listed the final-output id 48 and `configure_taps` raised.  The target
  appends the final output as the fourth tap itself.
* `Inkling2LoopTextForCausalLM.has_own_lm_head = False` and `has_own_embed_tokens = False`: the
  runtime-checkable `SupportsEagle3` protocol needs every member, without them the runner said
  "Model does not support EAGLE3 interface".
* Draft dtype and quantization: the draft is the bf16 `mtp.safetensors`, converted to fp16 at
  load like the target (weight report `max_abs 38.5`, no overflow).
  `DSparkSpeculatorConfig.default_model` copied the target's NVFP4 `quantization` to it, which
  made `get_draft_quant_config` raise `Missing 'quant_algo'`; an `inkling2_loop` target leaves
  the draft unquantized.
* The HIP `prepare_dflash_inputs` is dispatched only for `Inkling2DSparkSpeculator`
  (`uses_hip_prepare_inputs`); every other DFlash / DSpark draft keeps the Triton kernel.
* Draft KV owner registration, the draft FULL graph (`DFlashCudaGraphManager`: "Capturing model
  for Inkling2 DSpark speculator"; graph memory 1.95 GiB for the target and the draft, KV cache
  8.17 GiB at `max_model_len` 2048) and the verify batches (`1 + 7` rows, FULL graphs) boot and
  replay without further change.  Boot 33 s.
* Stale draft K/V after prefix-cache hits: the draft K/V sits in the same hybrid cache blocks
  as the target's, so a hit block carries the K/V written when it was produced; the
  acceptance of a warm run is that of a cold one (1510-token prompt, 128 tokens: 41 / 42 / 42
  drafts, mean accepted 2.10 cold, 2.02 cold again, 2.02 warm, same per-position profile) and
  the three runs generate the same tokens (max `|dlogp|` 0.061 and 0.084).  Blocks cached by
  a run without DSpark would carry no draft K/V (quality only) `[INFERENCE]`.

Acceptance, `k = 7`, 8 chat-templated prompts, each alone at batch 1, 192 tokens: 639 drafts,
mean accepted 1.41 per step (2.41 tokens per step), cumulative per position
`[0.50, 0.30, 0.20, 0.16, 0.10, 0.08, 0.06]`; per prompt 0.75 - 3.04 (an arithmetic question
accepts 3.04, open-ended answers 0.75 - 0.80).  Over the first 128 tokens of the same prompts
(the speed bench below): 361 drafts, 1.85 accepted per step (2.85 tokens per step),
`[0.60, 0.39, 0.29, 0.21, 0.16, 0.12, 0.10]`.

Speed, `FULL_AND_PIECEWISE`, same bench and session (ms per generated token of one request, 128
tokens, prompts as in the table above, b = 1 over all 8 prompts alone; tok/s is the total):

| batch | no draft | draft `k = 7` for batch 1, plain decode above (`num_speculative_tokens_per_batch_size=[[1,1,7],[2,8,0]]`) | ratio |
|---|---|---|---|
| 1 | 20.03 ms (49.9 tok/s) | 20.36 ms (49.1 tok/s) | 0.98x |
| 2 | 25.07 ms (79.8 tok/s) | 25.60 ms (78.1 tok/s) | 0.98x |
| 4 | 35.29 ms (113.3 tok/s) | 35.47 ms (112.8 tok/s) | 0.99x |
| 8 | 58.47 ms (136.8 tok/s) | 58.02 ms (137.9 tok/s) | 1.01x |

and with `k = 7` at every batch size (first 64 tokens, one prompt at b1, 4 / 8 distinct prompts
at b4 / b8; the first 64 tokens of the first prompt accept 2.15 per step, so this is the
optimistic case):

| batch | no draft | `k = 7` | ratio | mean accepted per step |
|---|---|---|---|---|
| 1 | 20.01 ms (50.0 tok/s) | 17.13 ms (58.4 tok/s) | 1.17x | 2.15 (20 drafts) |
| 4 | 33.93 ms (117.9 tok/s) | 43.54 ms (91.9 tok/s) | 0.78x | 1.93 (89 drafts) |
| 8 | 54.95 ms (145.6 tok/s) | 62.84 ms (127.3 tok/s) | 0.87x | 1.75 (190 drafts) |

* The engine step (`profile_decode.py --spec-k 7`, async scheduling off, random-token prompts) is
  48.2 ms at batch 1 (41.5 - 55.6) and 99.8 ms at batch 4, against 21.1 / 33.3 ms without a
  draft: the verification of 8 rows (about the 56 ms of an 8-row decode) plus the draft
  (about 5 ms).  The break-even is 2.4 accepted tokens per step; the measured 2.41 (192 tokens)
  and 2.85 (128 tokens) sit on it, and the effective step of the bench is 20.36 x 2.85 =
  58 ms, 10 ms above the profile's step (host work the profile does not count
  `[INFERENCE]`).  **DSpark at `k = 7` is therefore not a net gain on this stack**: +17% on a
  high-acceptance start of an answer, 0.98x over 8 different prompts at batch 1, a loss at batch >= 4.
* The scheduler's dynamic speculative decoding (`num_speculative_tokens_per_batch_size`, entries
  `[first batch, last batch, verified draft tokens]`) works with the DSpark speculator: the draft
  always proposes 7 tokens (the model requires `num_speculative_tokens = 7`, the draft block is
  8 rows), the scheduler verifies as many as the batch size allows, and with 0 above batch 1 the
  engine runs at the speed of the no-draft engine (table above, within 1.5%).  Verified tokens `K`
  at batch 1 (`[[1, 1, K], [2, 8, 0]]`, same bench, 8 prompts x 128 tokens; no draft: 20.03 ms):

  | K verified | 1 | 2 | 3 | 4 | 7 |
  |---|---|---|---|---|---|
  | ms per token (tok/s) | 18.07 (55.3) | 18.12 (55.2) | **16.50 (60.6)** | 19.51 (51.3) | 20.36 (49.1) |
  | speed-up | 1.11x | 1.11x | **1.21x** | 1.03x | 0.98x |
  | mean accepted per step | 0.67 | 0.90 | 1.23 | 1.55 | 1.85 |
  | rows of the verification | 2 | 3 | 4 | 5 | 8 |

  `K = 3` is the best: its 4 rows run the 4-row MoE instance and the fifth row of `K = 4` costs
  the step of the 8-row instance (the kernel table above) `[INFERENCE]`.  A second schedule,
  `[[1, 1, 3], [2, 2, 1], [3, 8, 0]]`, gave 16.25 ms at batch 1 (61.5 tok/s, 1.23x) but nothing at
  batch 2 (25.09 ms against 25.07 ms: two requests with one verified token each pay what they
  gain), and 36.66 ms at batch 3 because the default grid pads 3 rows to 4 (see the capture sizes).
* Recipe for one V620, TP = 1: the compilation config above and
  `--speculative-config '{"method": "dspark", "num_speculative_tokens": 7,
  "num_speculative_tokens_per_batch_size": [[1, 1, 3], [2, 8, 0]]}'`: 1.21x for a single stream,
  the speed of the no-draft engine from batch 2 up (a dense decode grid together with a draft was
  not measured; the graph memory with the draft is 1.94 GiB).
* Greedy equality (26 generations x 24 tokens, `--compare`): speculative vs eager B 19 / 26
  identical, max `|dlogp|` before the first divergence 0.064, the divergences at positions 16 and
  23 on near-ties of 0.012 - 0.061 nats; speculative vs no-draft `FULL_AND_PIECEWISE` 20 / 26
  (0.081).  The floor of the checkpoint on this box (eager A vs eager B) is 17 / 26 (0.082).

### Gates run

| command | result |
|---|---|
| CPU: `tests/models/inkling2_loop` + `tests/models/test_inkling2_loop_blocks.py`, `test_inkling2_loop_moe.py`, `test_inkling2_tiny_harness.py` | 522 passed, 77 skipped in 410 s (part A: 519 passed, 76 skipped; +3 new CPU tests) |
| GPU (HIP 2): `tests/models/inkling2_loop/test_serving_path_graph_gpu.py` | 5 passed in 5.57 s |
| GPU (HIP 2): `tests/models/inkling2_loop/test_serving_path_gpu.py` | 116 passed in 136.08 s |
| GPU (HIP 2): `tests/v1/spec_decode/test_dflash_prepare_inputs.py` (the runner's Triton path, which every non-Inkling draft keeps) | 6 passed in 5.08 s (6 failed first: `propose` reads the new `uses_hip_prepare_inputs`, and the test's speculator stand-in got that one attribute) |

### Open items after part B

* `conv_passes` still launches `kv_prep` / `sconv_static` twice for `q > 1` and while tracing a
  breakable graph (needs a write / read-phase split inside the kernel, csrc).
* The breakable graphs give no speed-up for this model; their cost is the padding.  The default
  decode grid is dense since part B2 (`dense_decode_cudagraph_architectures`): the recipe needs no
  `decode_cudagraph_capture_request_sizes`.
* DSpark: the fifth-row cliff of a verification step is removed at the kernel level in part B2 (the
  row policy and the register spills of the mxfp8 kernel, microbenchmarks and tests there).  The
  `K = 1 .. 7` table above (best `K = 3`, 1.21x at batch 1) was measured before that change; its
  engine-level re-measurement (decode ms per row count, the K table with the dense default grid, the
  per-launch engine profile) was queued on the box (`final_chain.sh` of the GraphDspark work
  directory, results in `out/fin_*`) and is not in this document.
* TP = 1 maximum length (8192 served; the largest length that fits is not measured) and TP =
  2 / 4 at 1,048,576 are not covered by part B; neither is TP > 1 with graphs.
* `vllm serve` itself was not started: the runs use `LLM(...)` with the same
  `compilation_config` dictionary.

## 2026-11-22 — integration part B2: the cost of a verification row (MoE) and the default decode grid

Part B left a fifth decode row at +10.5 ms (eager, 45.3 ms against 34.8 ms), an mxfp8 MoE kernel at
0.74 ms (M = 4) and 1.46 ms (M = 8) per launch, a best DSpark verification of 3 tokens (the 4-row
instance) and a decode grid that pads unless the recipe lists every size.  This part removes the
first three causes and the fourth.  Files: `csrc/rocm/moe_lowp_dot2_rdna2.cu` (the mxfp8 kernel),
`vllm/models/inkling2_loop/amd/moe.py` (the row policy and a loader guard), `vllm/config/vllm.py`
(the default grid), `tests/kernels/core/test_inkling2_moe_rdna2.py`,
`tools/rdna/inkling2/{moe_rows_bench,moe_jit}.py`.

### Where the cliff came from

* **The policy.**  A kernel block holds the routed rows of ONE expert (`block_size_m` of them, padded).
  A chunk of `T` tokens routes `8 T` rows over 128 experts, so up to a few dozen tokens nearly every
  block holds a single row and a wider block only adds padded rows of dot2 work.
  `Inkling2LoopMoE._default_block_size_m` chose the width 1 / 2 / 4 / 8 by the token count (1, 2, 3 - 4,
  5 and more): every step of 5 - 8 rows (a decode batch of five, a verification of `K >= 4`) ran the
  8-row instance.  The kernels themselves have no cliff (bench below).
* **The old mxfp8 kernel spilled registers.**  `-Rpass-analysis=kernel-resource-usage` on gfx1030, old
  against new source (`moe_jit.py --resource-usage`):

  | instance | old VGPRs / spilled VGPRs / scratch B per lane / waves per SIMD | new |
  |---|---|---|
  | mxfp8 M = 1 | 128 / 121 / 256 / 8 | 85 / 0 / 0 / 10 |
  | mxfp8 M = 2 | 128 / 158 / 336 / 8 | 93 / 0 / 0 / 10 |
  | mxfp8 M = 4 | 128 / 229 / 496 / 8 | 128 / 2 / 12 / 8 |
  | mxfp8 M = 8 | 128 / 396 / 1108 / 8 | 124 / 0 / 0 / 8 |
  | nvfp4 M = 1 / 2 / 4 / 8 (unchanged) | 99 / 99 / 80 / 112 VGPRs, no spills, 9 / 9 / 12 / 9 waves | same |

  The dequantisation `e4m3 -> f32 -> * 2^(scale - 127) -> f16` of 4 columns x 32 K per thread kept
  more values live than 128 registers hold, and every row added accumulators and activation reads.

### The new mxfp8 kernel

`moe_gemm_mxfp8_kernel_rdna2` dequantises on the 16-bit integer pipe: the 7 magnitude bits of an e4m3
byte moved to fp16 bits 13..7 are the fp16 encoding of `value * 2^-8` (normals and subnormals alike),
one `half2` multiply by `2^(scale - 119)` per 32-K block gives `value * 2^(scale - 127)` rounded to
fp16 once (two `v_perm_b32` and a few logic ops per 4 bytes instead of the float conversion chain),
and the activations of all `M` rows are read from LDS once per 8-K group as one `uint4` each.  The
accumulation order of the four `v_dot2` calls, the fp16 CAS accumulation and `write_out` are
unchanged.

* **Exactness.**  The result of the dequantisation is bit for bit the old `__float2half_rn(e4m3_to_f32(b)
    * 2^(scale - 127))` for scale bytes 95..134 (where `2^(scale - 119)` is an fp16 number);
  `test_mxfp8_dequant_is_exact_for_every_byte_and_supported_scale`checks all 256 byte values, spread
  over eight columns with the scales 95, 104, 105, 118, 120, 127, 130, 134, with one-hot activations
  for each of the four instances.  The loader refuses other scales (`MXFP8_SCALE_RANGE` in
  `repack_mxfp8`); the real checkpoint's four mxfp8 layers span 113..118 (checked over every tensor).
* **The nvfp4 kernel is unchanged** and does not need to change: it already shares the dequantisation
  of a block among its rows (bench: 52 -> 77 us for M = 1 -> 8 at one token) and from 5 tokens on it is
  at the memory stream rate (T = 8, M = 1: 64 expert blocks x 2.98 MB = 191 MB in 402 us, 475 GB/s
  against the 512 GB/s of the card `[INFERENCE: spec figure of 256-bit GDDR6 at 16 Gbps, not measured]`).
* **Code objects (gfx1030, `llvm-objdump -d --symbolize-operands` of the two builds, per kernel):** the four
  nvfp4 instances (M = 1 / 2 / 4 / 8) are **identical** (1176 / 1414 / 1870 / 2776 instructions in both);
  the four mxfp8 instances **changed on purpose**: 2981 -> 778, 3364 -> 1017, 4073 -> 1479 and
  5735 -> 2388 instructions.  No instance was added.  The kernels have templates for 1 / 2 / 4 / 8 rows
  and `inkling2_glue_moe_route` (the router and aligner, `inkling2_glue_rdna2.cu`) refuses other
  widths, so 3 / 5 / 6 / 7 would be dead instances; the table below shows they are not needed.
* **Six architectures.**  `python tools/rdna/inkling2/moe_jit.py --archs
  "gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1200" --resource-usage` compiles old and new source
  with exit 0 (no errors): gfx1030 builds the kernels, the other five the stubs inside the
  `lowp_dot2` namespace (40 stub blocks, no registers, the same as before).

### The row policy

`_BLOCK_M_BY_TOKENS` in `moe.py`, per expert format, by the token count of a chunk:

| tokens | 1 - 4 | 5 - 12 | 13 - 48 | 49 - 64 | 65 and more |
|---|---|---|---|---|---|
| NVFP4 layers (0 - 19) | 1 | 2 | 4 | 4 | 8 |
| MXFP8 layers (20 - 23) | 1 | 2 | 2 | 8 | 8 |

(old policy for both formats: 1 row for 1 token, 2 for 2, 4 for 3 - 4 and 8 for 5 and more).  Chosen from the bench
below as the fastest width at every measured token count or within 3% of it.

### Per-launch numbers (`moe_rows_bench.py`, real layer shapes, V620, HIP 2)

Hidden 2304, expert width 1152, 128 experts, top 8, one launch of the up (w13: K = 2304, N = 2304) and
down (w2: K = 1152, N = 2304) GEMM, microseconds, uniform random routing, 50 iterations per cell.
The weights of one launch are re-read every iteration, so the cells where they fit the 128 MB
infinity cache (one token: 24 MB) are cache-warm and faster than in the engine
`[INFERENCE: not isolated]`.  Old kernel -> new kernel, per instance M = 1 / 2 / 4 / 8:

| format / stage / tokens | M = 1 | M = 2 | M = 4 | M = 8 |
|---|---|---|---|---|
| mxfp8 w13, 1 token | 213 -> 35 | 257 -> 37 | 360 -> 40 | 639 -> 56 |
| mxfp8 w13, 5 tokens | 1097 -> 470 | 1142 -> 471 | 1538 -> 487 | 2590 -> 482 |
| mxfp8 w13, 8 tokens | 1740 -> 629 | 1893 -> 630 | 2501 -> 649 | 4224 -> 642 |
| mxfp8 w13, 16 tokens | 3328 -> 1098 | 2988 -> 1092 | 3568 -> 1126 | 5984 -> 1103 |
| mxfp8 w13, 64 tokens | not run -> 1994 | 8899 -> 1674 | 7793 -> 1673 | 10120 -> 1648 |
| mxfp8 w13, 128 tokens | not run -> 3461 | 16380 -> 2421 | 13064 -> 1947 | 14425 -> 1832 |
| mxfp8 w2, 5 tokens | 534 -> 63 | 613 -> 66 | 843 -> 83 | 1484 -> 118 |
| mxfp8 w2, 8 tokens | 869 -> 238 | 906 -> 237 | 1234 -> 263 | 2067 -> 244 |
| nvfp4 w13, 1 token (unchanged) | 48 / 52 | 49 / 49 | 59 / 59 | 76 / 77 |
| nvfp4 w13, 5 tokens (unchanged) | 183 / 171 | 158 / 150 | 187 / 180 | 252 / 246 |
| nvfp4 w13, 8 tokens (unchanged) | 398 / 402 | 397 / 401 | 400 / 400 | 514 / 510 |
| nvfp4 w13, 64 tokens (unchanged) | - | 1616 / 1663 | 1286 / 1300 | 1311 / 1284 |
| nvfp4 w2, 5 tokens (unchanged) | 93 / 92 | 96 / 91 | 111 / 105 | 150 / 141 |

(nvfp4 rows: two measurements of the same code, before / after, to show the noise: within 5%.)
What the engine pays per layer visit at the token counts of a verification (w13 + w2, old policy and
kernel -> new policy and kernel):

| tokens | mxfp8 layer | nvfp4 layer |
|---|---|---|
| 1 | 326 -> 58 us (M = 1) | 84 -> 84 us (M = 1, same instance) |
| 4 | 1988 -> 348 us (M = 4 -> 1) | 235 -> 203 us (M = 4 -> 1) |
| 5 | 4074 -> 537 us (M = 8 -> 2) | 387 -> 241 us (M = 8 -> 2) |
| 8 | 6291 -> 867 us (M = 8 -> 2) | 713 -> 535 us (M = 8 -> 2) |

### Tests and mutation checks

`tests/kernels/core/test_inkling2_moe_rdna2.py` (117 tests, HIP 2): both formats x up / down x
`block_size_m` 1 / 2 / 4 / 8 x tokens 1, 2, 3, 5, 8, 13, 40 (every row of the output against an fp64
reference built from the dequantised weights; K = 288 and N = 1056 so both tails are covered; 4
routed experts per token, experts with more rows than one block holds), the exactness test above,
and the rejection of a block width of 3.  The old kernel passes 117 (68 s), the new 117 (105 s with
the JIT build).  `I2_MOE_JIT=<path to a .cu>` runs the same tests on a JIT build of that file
(`moe_jit.py`): fifteen mutants of the new source were each run that way and **all fifteen are killed**
(first failing case in brackets): mxfp8 sign bit dropped, exponent top bit dropped, factor `2^(s -
120)` instead of `2^(s - 119)`, bytes 0/1 taken from lanes 2/3, lanes swapped, every row reading row
0's activations (`mxfp8-up-2-3`), the last row not accumulated, scale row stride doubled, 8-K group
offset not advanced (all `mxfp8-up-1-1`); nvfp4 block-global scale dropped, second scale row
replaced (`nvfp4-up-1-1`); router weight dropped (`nvfp4-down-1-1`), output row divided by the wrong
top-k (`nvfp4-down-1-2`), expert id read from the wrong block, K tail dropped (`nvfp4-up-1-1`).
CPU tests (`test_serving_path_cpu.py`): the policy table (widths in {1, 2, 4, 8}, never narrower with more
tokens, 1 up to 4 tokens, 2 for 5 - 8), the loader guard, and the default decode grid below.

### The default decode grid

`dense_decode_cudagraph_architectures()` (`vllm/config/vllm.py`; Inkling-2 on RDNA only, every other
architecture and Inkling-2 off RDNA keep the shared-size default) makes the default
`decode_cudagraph_capture_request_sizes` every request count up to 16
(`DENSE_DECODE_CUDAGRAPH_REQUESTS`) united with `decode_cudagraph_capture_grid(max_num_seqs)`, up to
`max_num_seqs`.  The sizes are request counts, so plain decode and the verification batches of a
draft (query length `1 + K`) both get the dense counts; an explicit list still wins.  Part B measured
what this avoids: 3 rows replayed in the 4-row graph 35.4 ms against 30.8 ms in an exact graph, 5 rows
in the 8-row graph 61.0 ms against 43.8 ms.  CPU tests (`test_serving_path_cpu.py`): the default sizes
for `k = 0` and `k = 7` at `max_num_seqs` 8 (1 .. 8) and 64 (1 .. 16 then the grid), the gating
(another architecture, Inkling-2 off RDNA: 1, 2, 4, 8 as before) and the explicit override.

### Engine: decode with the new kernel, policy and default grid

`graph_check.py --mode FULL_AND_PIECEWISE --max-model-len 2048 --prefix-caching --batches 1,2,...,8
--tokens 128 --reps 2 --skip-outputs`, real checkpoint, TP = 1, HIP 2, **no** `compilation_config`
sizes (the default grid is dense), `_rocm_C` rebuilt from the new `.cu`; ms per generated token of
one request (= the step time of `b` rows), tok/s is the total over the batch.  The part B columns
are the numbers in the tables above (eager and default-grid rows of the 64-token bench; b = 1, 2 from
the 128-token no-draft bench), so the comparison is not token-for-token the same bench; the longer
sequences of the new run should cost slightly more per step, i.e. the comparison is conservative
`[INFERENCE]`:

| rows | part B eager | part B `FULL_AND_PIECEWISE`, default grid 1, 2, 4, 8 | part B2 `FULL_AND_PIECEWISE`, default (dense) grid | tok/s total (B2) |
|---|---|---|---|---|
| 1 | 21.07 | 20.03 | 18.97 | 52.7 |
| 2 | 25.50 | 25.07 | 22.93 | 87.2 |
| 3 | 30.98 | 35.44 (padded to 4) | 26.71 | 112.3 |
| 4 | 34.81 | 33.93 | 30.09 | 133.0 |
| 5 | 45.28 | 61.04 (padded to 8) | 32.53 | 153.7 |
| 6 | 48.99 | 60.70 (padded to 8) | 34.71 | 172.9 |
| 7 | 53.04 | 58.51 (padded to 8) | 37.62 | 186.1 |
| 8 | 56.13 | 54.95 | 40.24 | 198.8 |

The fifth row now costs +2.4 ms (30.09 -> 32.53 ms) instead of +10.5 ms, and every row count is
replayed in its own graph (3 rows no longer pay the 4-row graph).  Graph memory 0.50 GiB (0.57 GiB
with the part B default grid).  Not measured in this document: the eager column with the new
kernel, the engine profile per launch, and the DSpark `K` table (queued on the box, see the open
items).

Gates on the rebuilt `_rocm_C` (HIP 2): `test_inkling2_moe_rdna2.py` 117 passed (67.5 s);
`test_serving_path_gpu.py` + `test_serving_path_graph_gpu.py` + `inkling2_loop/test_moe.py` +
`test_inkling2_loop_moe.py` 186 passed (204.8 s); CPU `test_real_checkpoint_names.py` +
`test_serving_path_cpu.py` 51 passed, 4 skipped (the full CPU suite gave 521 passed, 77 skipped and 7
errors before the meta-tensor fix of the MXFP8 scale guard, which the two files above then
cleared; the full suite was not repeated).
