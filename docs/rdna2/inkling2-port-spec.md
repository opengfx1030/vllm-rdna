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
6.2e-3, 40 tokens 1.053 / 1.5e-2, 70 tokens 0.139 / 7.9e-3; tiny8 1.0e-2, 1.8e-2,
1.2e-2 (no flips). So "mean |dlogp| < 0.05" is a statement about routing stability of
the prompt, not only about the kernels; the forced-route number is the noise floor of
the fp16 path.

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
