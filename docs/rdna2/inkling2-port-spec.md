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
   read-phase split would remove it and the double launch.  **Superseded** (2026-11-22,
   conv ring section below): the ops have `write_state` and the separate
   `inkling2_conv_state_write`, `conv_passes` is gone, and a request that would hit blocks
   written in the same step is deferred by the scheduler's conv-state manager.
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
* ~~Decide `conv_passes`~~ **Done** (2026-11-22 conv ring section): the conv ops no longer
  need two passes; `write_state` is a launch constant (fused for `max_query_len == 1`
  graphs, `inkling2_conv_state_write` after the conv for verify / mixed graphs).
* The V2 runner must capture FULL graphs around `forward` only: `embed_input_ids`,
  `compute_logits` and `greedy_token_ids` allocate; `ws.bind(T)` views have to be created
  per captured size before capture.  `greedy_token_ids` is not wired into the sampler yet
  (runner-owned).
* TP: done for the eager path (`inkling2-tensor-parallel.md`, the section "tensor
  parallelism" below): o_proj fp32 partial + all-reduce, `lm_head` and embedding
  vocabulary shards, replicated router, MoE intermediate split, per-rank `q_scale` /
  `rel_scale` slices, KV and conv owners by KV head.  Left: DSpark and graphs at TP > 1.
* Long context: `check_kernel_shapes` and the workspace bound (`attention_workspace_numel`)
  were validated up to `max_model_len` 2048; TP = 1 maximum length and TP = 2/4 at
  1,048,576 need the split count and the workspace re-derived for those lengths.
* Speed: the step is GPU-bound (93.9% busy); [INFERENCE] a graph gains at most ~6% of a
  batch-1 decode step, so the next gains are kernel time (attention input GEMV 27.6%,
  expert GEMMs 31%, `lm_head` 9.5%).

## 2026-11-22 — conv and KV memory: a position-addressed conv ring, and the long-context budget (branch `feat/inkling2-conv`)

Question: which cache layout holds Inkling-2-Mini's per-request state in the least memory
on a V620 (29.98 GiB usable, 18.08 GiB of weights at TP=1), keeps prefix caching,
speculative decoding and the tensor-parallel split, and what `max_model_len` does that buy
per GPU?  Every planner number below is the engine's own planner
(`vllm.v1.core.kv_cache_planning`) run on the real checkpoint config with the real cache
owners (`tools/rdna/inkling2/kv_plan.py`: meta-device model, no weights, no GPU).  "One
max-length request" is the memory the planner demands before it accepts a length (all
groups; the pool block is 2 MiB at TP=1, 1 MiB at TP=2, 0.5 MiB at TP=4).  `max_num_seqs`
2, layout `BLNHC`.  The planner holds `F` = 2 x `max_num_batched_tokens` ("mnbt") in-flight
tokens per request (async scheduling), so mnbt appears in every table.

### The problem in part A (HEAD 62b10cdd29)

The conv owner was a `SlidingWindowSpec(window 4)` of per-token slots: fp32 rows
`[H = 8, 1024]` per token (the four stream widths 128 / 128 / 288 / 288 padded to a power of
two), i.e. 48 layers x 8 x 1024 x 4 B = **1.5 MiB per token**, in 8-token blocks of 2 MiB
(6 groups of 8 layers).  The planner holds `window - 1 + F` tokens of it per request: 6 x
513 blocks at `max_model_len` 4096 and 6 x 1025 blocks (12.0 GiB) from 8192 up, whatever the
length:

| max_model_len | TP = 1 | TP = 2 | TP = 4 |
| ---: | ---: | ---: | ---: |
| 4096 | 3467 blocks = 6.771 GiB | 3.386 GiB | 1.693 GiB |
| 262,144 | 10,977 blocks = 21.439 GiB | 10.720 GiB | 5.360 GiB |
| 1,048,576 | 23,265 blocks = 45.439 GiB | 22.720 GiB | 11.360 GiB |

(mnbt 4096, 12 groups: five sliding-window KV groups of 8 layers, six conv groups, one
full-attention group.)  The engine leaves 7.3 GiB for KV at TP=1 (see below), so the old
layout could not serve a request of 8192 tokens (the conv groups alone are 12.0 GiB), and
TP=2 needed 22.7 GiB at the model's maximum length.

### The layout

* **State per cache block, addressed by position.**  Position `p` of a request lives in the
  cache block `block_table[req, p // 64]` at ring row `p % S` of a `[H, S, W]` fp16 page
  (`W` = 832 = 128 + 128 + 288 + 288 per kv head, no padding): 8 x 3 x 832 x 2 B =
  **39,936 B** per layer and block at TP=1, 48 layers = 1,916,928 B in one cache group (one
  2 MiB pool block).  Only the raw inputs of the last `k - 1 = 3` tokens are conv state, so
  a ring of 3 rows per block is enough, and because every block keeps its own tail a prefix
  cache hit on any block resumes the conv exactly (the conv group is a window of 4 tokens,
  so outside the in-flight tokens the hybrid allocator keeps only the newest block of a
  request for it).
* **`S = 3 + num_speculative_tokens`** (10 with 7 drafts).  A verify step computes `1 + k`
  tokens and keeps a prefix; the rejected tokens' inputs also go into ring rows, so the ring
  needs `k` rows more than the history or a rejection would overwrite history the next step
  reads.  The ops are tested with rings of 3 and 10 rows; the model-level rollback tests run
  4 drafts (a 7-row ring) at block sizes 4 / 16 / 64, and their negative control, a 3-row ring
  under the same 5-token verify steps, fails the same check (mean |dlogp| 3.17 in that run
  against about 0.03).  Cost: the page grows to 133,120 B at 7 drafts and the conv group
  splits into 4 groups of 12 layers (the pool block does not change).
* **fp16 state.**  The conv inputs of the serving path are fp16 GEMM outputs, so fp16 is
  exact at half the bytes of fp32.  An fp32 input that is not fp16-representable (for
  example a TP fp32 all-reduce result passed to the conv) is stored rounded to nearest even:
  only its history taps carry the rounding.
* **Writer rule.**  A step writes the raw input of every token that is among the last `S`
  positions of the intersection of its chunk with its block (`p + S >= min(seq_len,
  block_end(p))`): one writer per ring row, and exactly the rows the next reader needs (the
  history of the next chunk and the tail of every completed block).  The conv ops read the
  history first.  One-token steps write in the same launch (a thread reads its three history
  rows, then writes its own); multi-token steps follow with `inkling2_conv_state_write` per
  stream, because a long chunk would otherwise overwrite a row another thread of the same
  launch has still to read.  This removed `conv_passes` (the twice-launched `kv_prep` /
  `sconv_static` of part A).
* **Layout `BLNHC` / `BLHNC`.**  Mixed page sizes (256 KiB x 8 layers of KV, 39.9 KiB x 48
  layers of conv) only share pool blocks in the block-outermost layouts;
  `Inkling2Backend.supported_kv_cache_layouts` lists them and the engine logs "Using BLNHC
  KV cache layout".
* **Same-step prefix hits are deferred.**  The scheduler registers the blocks of the tokens it
  schedules before they are computed, so a request of the same batch can hit a block whose
  tail rows the owner writes in that very step, and its conv history would be read before it
  was written.  `Inkling2ConvStateManager` (a `SlidingWindowManager` registered for
  `Inkling2ConvSpec`, as `MambaManager` is for state specs) reports "does not fit" for such a
  request, so it is scheduled one step later and hits finished blocks.
  `test_conv_state_manager.py` drives the real `KVCacheManager` (the plain sliding-window
  manager is the control: it serves the hit in the same step).  Cost: one step of latency
  for requests that arrive together with an identical prefix.
* **Rejected.**  A per-request slot (Mamba style) cannot resume at a block boundary, i.e. it
  loses prefix caching; a per-block tail of exactly 3 rows needs a copy-back for draft
  rejection; fp32 or power-of-two padded rows cost 2x / 1.23x (79,872 / 49,152 B per page).

### Planner numbers after the change

mnbt 4096, `max_num_seqs` 2 (`F` = 8192), one max-length request:

| max_model_len | TP | k = 0 | k = 7 (ring 10) | old |
| ---: | ---: | ---: | ---: | ---: |
| 4096 | 1 | 454 blocks = 0.887 GiB | 649 blocks = 1.268 GiB | 6.771 GiB |
| 4096 | 2 / 4 | 0.443 / 0.222 GiB | 0.634 / 0.317 GiB | 3.386 / 1.693 GiB |
| 262,144 | 1 | 4951 blocks = 9.670 GiB | 5341 blocks = 10.432 GiB | 21.439 GiB |
| 262,144 | 2 / 4 | 4.835 / 2.417 GiB | 5.216 / 2.608 GiB | 10.720 / 5.360 GiB |
| 1,048,576 | 1 | 17,239 blocks = 33.670 GiB | 17,629 blocks = 34.432 GiB | 45.439 GiB |
| 1,048,576 | 2 / 4 | 16.835 / 8.417 GiB | 17.216 / 8.608 GiB | 22.720 / 11.360 GiB |

Seven groups at k = 0 (ten at k = 7): five sliding-window KV groups of 8 layers with
`cdiv(min(1023 + F, L), 64) + 1` blocks (145 from L = 9216 up), one conv group of all 48
layers with `cdiv(min(3 + F, L), 64) + 1` blocks (130), one full-attention group of 8 layers
with `L / 64` blocks.  The formulas reproduce the planner's block counts (454 at 4096, 4951
at 262,144, 17,239 at 1,048,576).  What remains is inherent: full-attention KV costs **32
KiB per token** at TP=1 (8 layers x 4 KiB), and the five sliding groups hold 725 blocks
(1.42 GiB) whatever the length.  The conv state is 0.25 GiB at any length of 8192 or more
(was 12.0 GiB).  mnbt is the other lever: mnbt 1024 (`F` = 2048) needs 49 blocks per sliding
group and 34 for the conv group, and a request of 1,048,576 needs 16,663 blocks (32.545 GiB
at TP=1, 16.272 / 8.136 GiB at TP=2 / 4; k = 7: 32.744 / 16.372 / 8.186 GiB).  mnbt 256
needs 16,519 blocks (TP=2: 16.132 GiB; k = 7: 16.161 GiB).

### What the engine leaves for KV (measured boots, real checkpoint, TP = 1)

The extensible KV cache sizes itself after warmup (`measure_kv_cache_bytes` in
`vllm/v1/worker/extensible_kv_cache.py`): KV = (free memory + committed KV) - margin -
transient activations, with the margin `max(3 GiB on ROCm, 0.75 x transient)`.
`gpu_memory_utilization` is not the binding term at 0.92 or above (the TP port's boots give
the same 7.31 GiB at 0.92 and 0.99).  HIP 3, one V620, util 0.92, eager, k = 0,
`--max-model-len -1` (the engine fits the length twice: before warmup from the available
memory, and again after warmup from what is really left; only the second counts):

| | mnbt 4096 | mnbt 1024 |
| --- | ---: | ---: |
| "Model loading took" (weights + the T-sized glue workspace) | 19.48 GiB | 18.68 GiB |
| glue workspace (`Inkling2GlueWorkspace.nbytes()`, meta-device sum) | 1.030 GiB | 0.303 GiB |
| available KV memory before warmup | 7.29 GiB | 8.11 GiB |
| first auto-fit length (before warmup) | 184,064 | 247,680 |
| free + committed after warmup | 8.76 + 0.89 GiB | 10.23 + 0.23 GiB |
| kept for transient activations | 0.77 GiB | 0.77 GiB |
| KV cache after warmup | 3012 blocks = 5.88 GiB | 3430 blocks = 6.70 GiB |
| **final auto-fit `max_model_len`** | **137,984** | **201,600** |
| engine "GPU KV cache size" | 138,029 tokens | 201,658 tokens |

The planner gives the same lengths from the same budgets (`kv_plan.py --budget-gib`): 5.88
GiB hold 137,920 tokens at mnbt 4096 (engine: 137,984), 6.70 GiB hold 201,664 at mnbt 1024
(engine: 201,600), 7.29 / 8.11 GiB hold 184,128 / 247,872 (engine: 184,064 / 247,680).  The
3 GiB margin is the largest slice that could still go to KV; it is an engine constant, not
part of this change.  The part-A layout could not have booted at either length.

### TP = 1: the needle run at the longest length that boots (HIP 3, mnbt 1024)

`tools/rdna/inkling2/long_context_check.py` boots the checkpoint with `--max-model-len
201600` (explicit; util 0.92, eager, prefix caching, k = 0, `max_num_seqs` 2) and sends a chat
prompt of 181,434 tokens, 90% of it: numbered filler facts with one passphrase
(`maple-quartz-47`, "Fact 6519") at depth 0.5.  The run is cold (`reset_prefix_cache`), then
warm, then a 96-token decode on the warm cache:

| Phase | Result |
| --- | --- |
| boot at 201,600 | OK in 40.4 s; device VRAM used 26.24 GiB of 29.98 GiB |
| cold prefill, chunks of 1024 | 382.2 s = **475 tok/s** (181,434 tokens) |
| warm prefill (hit on all but the last block) | 0.29 s |
| decode at 181,434 tokens of context | **35.6 ms per token** (96 tokens in 3.7 s, minus the 0.29 s warm one-token run) |
| needle | FOUND: the answer quotes "Fact 6519: the secret passphrase is maple-quartz-47" |
| control, same needle in 4088 tokens (8.0 s with 96 decoded tokens) | FOUND |
| peak device VRAM during the cold prefill / during the decode | 26.24 / 26.24 GiB (sysfs `mem_info_vram_used`, polled every 0.5 s) |

Part A measured 1000 tokens of prefill in 1262 ms (792 tok/s) and 23.7 ms per decode step
at a short context, so the 181k-deep run is at 60% of the prefill rate and +11.9 ms per
decode step [INFERENCE: the attention benchmark gives 1.2 ms per full-attention layer visit
at 131,072 tokens at TP=1, about 1.7 ms at 181k, x 8 visits = 13 ms, which is the order of
the difference].  The device peak equals the post-boot level: of the 3.74 GiB the engine
keeps free (the 3 GiB margin floor plus 0.77 GiB of transients) the run touched none that
the 0.5 s polling could see.  Each GiB of KV is 32,768 tokens of full-attention context at
TP=1 (32 KiB per token), so the margin floor is the lever for the TP=1 length and for TP=2
at 1,048,576.

**Experiment, not a shipped change: a 1 GiB margin.**  `long_context_check.py
--kv-margin-gib 1.0` replaces `KV_CACHE_MARGIN_FLOOR_BYTES` in-process (the engine then runs
inside the tool's process) and nothing else: util 0.92, mnbt 1024, k = 0.  After warmup the
KV is 3884 blocks = 7.59 GiB, now bounded by the 27.59 GiB that util 0.92 requests (the
margin would allow 10.46 - 1.0 - 0.77 = 8.69 GiB); the engine's auto-fit is **230,656**
(planner for 7.59 GiB: 230,848) and the boot at that length is OK.  The needle run at 90%,
207,586 tokens:

| Phase | Result |
| --- | --- |
| cold prefill, chunks of 1024 | 459.8 s = 451 tok/s |
| warm prefill | 0.23 s |
| decode at 207,586 tokens of context | 37.4 ms per token |
| needle / control (4088 tokens) | FOUND / FOUND |
| device VRAM after boot / peak during prefill / peak during decode | 27.14 / 27.14 / 27.14 GiB (2.84 GiB of the card untouched) |

With util 0.99 as well the same arithmetic leaves 8.69 GiB, i.e. 266,880 tokens at mnbt 1024
[INFERENCE: planner only, that boot was not run].

### TP = 2 / TP = 4: 1,048,576 tokens per GPU

A rank holds `8 / tp` kv heads of every KV and conv cache, so the TP=1 numbers scale by
`1 / tp` (16 KiB per token at TP=2, 8 KiB at TP=4, plus the fixed sliding and conv groups).
Per-rank weights and budgets are the real-checkpoint boots of `Inkling2TP`
(`runs/real/tp{2,4}_real_rdna.log`, `tp2_kv99_c256.log`; util 0.92, old conv layout, which
changes neither the weights nor the activations); the table applies the planner of this
branch:

| TP | weights / rank | KV after warmup (engine sizing) | one max-length request at 1,048,576 (k = 0 / 7) | `max_model_len` that budget holds (k = 0 / 7) |
| ---: | ---: | ---: | ---: | ---: |
| 4 (mnbt 1024) | 5.49 GiB | 17.34 - 17.38 GiB | 8.14 / 8.19 GiB | 1,048,576 with 9.2 GiB to spare |
| 2 (mnbt 1024) | 9.69 GiB | 12.22 GiB | 16.27 / 16.37 GiB | 782,976 / 776,448 |
| 2 (mnbt 256) | 9.45 GiB | 15.33 GiB (15.56 GiB in the lean run) | 16.13 / 16.16 GiB | 995,968 / 994,048 (1,011,072 / 1,009,152) |
| 1 (mnbt 1024) | 18.08 GiB | 6.70 GiB | 32.54 / 32.74 GiB | 201,600 |

* **TP = 4 holds the model's 1,048,576 tokens per GPU** with the engine's own sizing at mnbt
  1024 (the TP boots' setting), with or without drafts.  mnbt 4096 needs 8.42 / 8.61 GiB,
  also inside the 17.3 GiB [INFERENCE: that boot was not run].  Full-attention decode at
  1,048,576 tokens is 2.5 ms per layer visit at TP=4 (kernel benchmark in
  `inkling2-attn-kernels.md`), i.e. about 20 ms per token for the 8 visits [INFERENCE:
  kernel time only, on top of the 23.7 ms part-A step].
* **TP = 2 does not reach 1,048,576 with the current engine sizing**: it needs 16.27 GiB
  (mnbt 1024) and gets 12.22 GiB; at mnbt 256 it needs 16.13 GiB and gets 15.33 GiB.  The
  headroom after warmup is 17.14 GiB at mnbt 1024 (15.53 free + 1.61 committed) and 18.81
  GiB at mnbt 256, of which the 3 GiB ROCm margin and the 1.91 / 0.48 GiB of transients are
  taken first.  Even a zero margin cannot fit it at mnbt 1024 (17.14 - 1.91 = 15.23 GiB <
  16.27); at mnbt 256 it fits if the margin is at most 2.2 GiB (18.81 - 0.48 - 16.13)
  [INFERENCE: not booted at that length; it needs a smaller `KV_CACHE_MARGIN_FLOOR_BYTES`
  on ROCm, a shared engine constant, and mnbt 256].  With the shipped margin the TP=2 length
  is 995,968 (mnbt 256) or 782,976 (mnbt 1024).
* **TP = 1 cannot**: 1,048,576 needs 32.5 - 34.4 GiB of KV per GPU on a 29.98 GiB card that
  holds 18 GiB of weights; the 201,600 measured above is the length.

### Validation

Box `par1-cs25`, GPU runs on HIP 3 (one V620), other agents' jobs sharing the machine (the
latencies are indicative).

| Check | Result |
| --- | --- |
| CPU `tests/models/inkling2_loop` (`HIP_VISIBLE_DEVICES=` empty, the GPU files skip) | 491 passed, 54 skipped, 0 failed in 371 s (544 tests collected; 507 at HEAD: +34 ring cases in `test_paged_state.py`, +5 in the new `test_conv_state_manager.py`, -2 in `test_paged_model.py`, where the same-step prefix-hit cases became hits on blocks the owner completed inside a chunk) |
| kernels `tests/kernels/attention/test_inkling2_attn_rdna2.py` | 668 passed in 327 s (604 conv-state cases over (block size, ring rows) = (4, 3), (64, 3), (16, 10), padded rows, one-token and multi-token chunks, fp16 and fp32 inputs, both page layouts; a fake-tensor trace of the four ops) |
| GPU `test_serving_path_gpu.py` | 122 passed in 160 s (tiny HIP-shaped models at TP=1: prefill, chunked prefill, decode, prefix hits, draft rollback with a 7-row ring, and a negative control that fails the rollback check with a 3-row ring) |
| engine, tiny checkpoints (`tiny_vllm_check.py`) | default, prefix caching, chunked, chunked + prefix: RESULT PASS; 8-expert prefix run RESULT PASS (two identical 70-token requests in one batch on a cold cache: ids equal the single run, max abs dlogp 0.091 against 0.243 run-to-run noise; warm-cache hit 0.0; hit against cold 0.061) |
| engine, real checkpoint (`real_check.py --prefix-caching`) | PREFIX HIT PASS: first-step top-8 equal to part A (`<\|message_model\|>` 0.0, 䁱 -21.08, user -21.17, `<\|content_thinking\|>` -21.22), 1510-token prompt cold 1.98 s / warm 0.12 s, hit and cold runs generate the same ids; the log shows "Using BLNHC KV cache layout" and seven groups of 64-token blocks |
| six-arch compile | gfx1030 gfx1100 gfx90a gfx942 gfx950 gfx1200: exit 0 and `llvm-objdump --offloading` lists all six code objects; gfx1030 conv kernels: `i2_sconv_kernel` 33-41 VGPR, `i2_state_write_kernel` 8-10, `i2_kv_prep_kernel` 61-74, no scratch, no LDS |
| mutation checks | 13 of 13 conv mutants fail the conv tests: 8 ring mutants (keep rule, history / write ring row, history block, write before read, stream offset, head stride, hard-coded 3-row ring) and 5 re-applied tap-math mutants (`inkling2-attn-kernels.md` section 8) |
| `uvx pre-commit run --files <changed files>` | clean, run twice |

Conv op cost (CUDA-graph replay, microseconds, new / part A): `kv_prep` static q=1 4.9 / 4.4,
q=1024 with its two state writes 24.7 / 35.1; `sconv_static` after the conv q=1 4.2 / 3.8,
q=1024 with its write 23.7 / 39.5.  Ring addressing costs about 0.5 us on a decode launch; a
prefill chunk gets cheaper because only the last rows of each block are written.

### Risks and open items

* TP=2 at 1,048,576 tokens needs an engine change outside this branch (the 3 GiB ROCm margin
  floor in `extensible_kv_cache.py`, and mnbt 256); the numbers above say by how much, the TP=2
  boot at that length is the next measurement.
* An fp32 conv input (the TP all-reduce result) is stored rounded to fp16, history taps only
  (relative 2^-11); a TP=1 against TP=2 logits comparison on the TP branch should expect it.
* `Inkling2ConvStateManager` makes a request whose prefix is computed in the same batch wait
  one scheduler step.
* Graph capture: the conv ops read only `seq_idx`, `seq_lens`, `query_start_loc` and the conv
  block table (no conv slot mapping); padded requests need `seq_idx = -1`; `write_state` is a
  launch constant, fused for `max_query_len == 1` graphs and replaced by an
  `inkling2_conv_state_write` launch after the conv in verify and mixed graphs
  (`conv_writes_in_op(md)`).
* `UniformTypeKVCacheSpecs.from_specs` would fold a conv spec into a plain
  `SlidingWindowSpec` of window 4 if one existed; the real model has only window 1024.
* `tools/pre_commit/test_tethering_allowlist.txt` (shared file) got the new
  `test_conv_state_manager.py` entry.
* Beyond 201,600 tokens only kernel tests cover the lengths (decode and verify kernels up to
  1,048,576 tokens at TP=2 / 4 shapes); TP=2 / 4 serving at those lengths has not been booted.

### Reproduction

```bash
# planner: groups, pool block, one max-length request, and the length a budget holds
python tools/rdna/inkling2/kv_plan.py --ckpt ~/hfcache/inkling2-mini \
    --max-model-len 262144 1048576 --max-num-batched-tokens 1024 --tp 1 2 4 \
    --drafts 0 7 --budget-gib 6.7 12.22 17.34
# TP=1 boot with the auto-fitted length, then the needle prompt at 90% of it
HIP_VISIBLE_DEVICES=3 python tools/rdna/inkling2/long_context_check.py \
    --ckpt ~/hfcache/inkling2-mini --chat-template-file \
    ~/hfcache/inkling2-mini/chat_template.jinja --max-num-batched-tokens 1024 \
    --max-model-len -1 --boot-only        # read "--max-model-len auto reduced ... to Y"
HIP_VISIBLE_DEVICES=3 python tools/rdna/inkling2/long_context_check.py \
    --ckpt ~/hfcache/inkling2-mini --chat-template-file \
    ~/hfcache/inkling2-mini/chat_template.jinja --max-num-batched-tokens 1024 \
    --max-model-len Y --frac 0.9 \
    --vram-file /sys/bus/pci/devices/0000:dc:00.0/mem_info_vram_used
```

## 2026-11-22 — tensor parallelism: TP=2 and TP=4 on the HIP path (eager)

`--tensor-parallel-size 2` and `4` serve the real checkpoint on the same HIP kernels (one
new kernel, the vocabulary-shard embedding lookup; no torch compute in the forward);
layout, validation and every table are in `inkling2-tensor-parallel.md`.  Run with
`VLLM_RDNA_AR=1` (one-shot HIP all-reduce below 64 KiB, RCCL above; `VLLM_RDNA_AR_MAX_KB`
/ `_ONESHOT_KB` = 256 helps at batch 16).

* **Sharding**: by KV-head group (12 + 4 heads at TP=2, 6 + 2 at TP=4); `o_proj` emits an
  fp32 partial that is all-reduced; the MoE intermediate is split (1152 -> 576 / 288) with
  an fp16 all-reduce; `lm_head` and the embedding by vocabulary (padded to 64; the
  embedding lookup gives zero rows for other ranks' ids, one fp16 all-reduce adds the
  owner's row back exactly); the router, residual stream, norms, the short convolutions
  of the attention / MLP outputs and their conv state and the vision tower are
  replicated.  97 all-reduces per decode step.  `parallel.check_supported` refuses TP=8;
  the DSpark draft at TP > 1 raises `NotImplementedError` (it shares the full-vocabulary
  head and embedding).
* **Real checkpoint** (eager, fp16, `max_model_len` 8,192, chunk 1,024; indicative: box
  load 7 - 13, the other agents' jobs ran on the neighbouring cards):

| | TP=1 | TP=2 | TP=4 |
|---|---:|---:|---:|
| model memory per GPU | 18.08 GiB | 9.26 GiB (9.69 before the embedding shard) | 4.84 GiB (5.49 before) |
| decode step, batch 1 / batch 4 | 23.40 / 36.86 ms | 18.92 / 26.96 ms | 16.77 / 21.58 ms (chunk 256) |
| prefill, one 1000-token prompt | 1316 ms | 753 ms (replicated embedding) | 488 ms (replicated embedding) |
| KV blocks held (chunk 1,024) | 3,744 | 14,960 | 19,872 |
| longest single request, planner | 179,136 | 847,808 | 2,334,592 |
| config maximum 1,048,576 (blocks needed at chunk 256 / 1,024) | no | **yes with `--max-num-batched-tokens 64`**: boots, answers, also with prefix caching (96 boots, 112 is 256 tokens short); 16,885 / 18,097 | yes, booted at chunk 256 (21,216 blocks, 2.45x concurrency): 8,643 / 9,825 |

* **Correctness**: CPU `test_tensor_parallel.py` 128 tests, 582 with the existing suite
  (thread-per-rank model against the fp32 reference <= 1.7e-4; 12 of 12 mutants killed);
  GPU MoE shard shapes 16 tests; the lookup kernel 17 tests, 10 of 10 mutants killed,
  six-architecture compile, the whole glue test file 212 passed; tiny engine checks at
  TP=2 / 4 (greedy ids equal, mean |dlogp| at the TP=1 level).  Real TP=2: both answers
  identical to TP=1, teacher-forced logprobs of 799 positions mean |dlogp| 0.0425 against
  the TP=1 run-to-run floor 0.0399, prefix caching (1,510 tokens: 1.17 s cold, 0.08 s
  warm) and an image request answered.  Real TP=4 (sharded embedding): prompt 0 ids 32 /
  32 equal to TP=1, prompt 1 diverges at token 16 (the near-tie that also separates the
  TP=1 runs A and B).  Its first-step top-8 differs from TP=1 by 0.33 - 0.38 on prompt 0
  (limit 0.245) but so do TP=1 repeats (0.43 and 0.34 against run A; the limit came from
  the single pair A / B) and two TP=4 runs differ from each other by 0.63 on that step:
  noise.  The teacher-forced comparison over 799 positions: TP=4 against TP=1 mean
  |dlogp| 0.0331 - 0.0515 (8 pairs, average 0.0468), TP=1 against TP=1 0.0399 - 0.0519 (6
  pairs, average 0.0470), TP=4 against TP=4 0.0356, limit 0.0898; no bisect or fp32 MoE
  reduction was needed.
* **Findings**: (1) `rdna_ar` costs 1.77 ms per decode step at TP=2 (96 calls) against
  2.92 ms for RCCL (3.26 ms at TP=4), and sums in fp32 in a fixed order.  (2) The first
  sharded `compute_logits` held three copies of the logits at the profiling peak (1.91 GiB
  against 0.77 GiB at TP=1); it now gathers 64-row blocks into the final buffer: +2,000
  KV blocks at TP=2.  (3) `gpu_memory_utilization` 0.99 adds no KV block at any TP (the
  3 GiB `KV_CACHE_MARGIN_FLOOR_BYTES` and the profiling transient decide); a smaller
  chunk does, and the conv groups need window plus one chunk, so chunk 256 -> 64 lowers
  what a 1M request needs by 0.3 GiB (TP=2).  (4) The conv state (about 1.5 GiB per
  request) does not shrink with TP.  (5) The embedding shard saves 0.43 GiB per rank at
  TP=2 (+448 blocks) and, with chunk 64, is what lets TP=2 reach 1,048,576 (the gap was
  0.93 GiB at chunk 256 before it; an earlier note of 2.1 GiB compared different chunks).
* **Open**: DSpark and FULL / PIECEWISE graphs at TP > 1 (the collectives are
  capture-safe, graph replay measured in `bench_tp_comm.py`); TP=4 prefix-cache and image
  runs with the shard; attention kernels at lengths beyond 2,048 tokens (the
  long-context check, not this work); prefill speed at chunk 64.

## 2026-11-22 — merged tree: the conv ring and tensor parallelism together (branch `feat/inkling2-merge`)

`feat/inkling2-conv` (42fffe0e8c, the fp16 position-addressed conv ring) and
`feat/inkling2-tp` (6d25f71128, TP=2 / 4 on the HIP path, validated on the old fp32
per-token conv layout only) merged without a textual conflict (7d88271ade) but had never
run together.  They meet in three places: the conv owner of a rank is `[H_r, S, W_r]`
(`H_r = 8 / tp` kv heads, a row width that grows with TP), the post-attention conv input
of a TP rank is the fp32 all-reduce of the `o_proj` partials while the ring stores fp16,
and the planner (`tools/rdna/inkling2/kv_plan.py`) has to agree with what the model
allocates.  Box `par1-cs25`, V620 (gfx1030), eager, fp16, `VLLM_RDNA_AR=1` at TP > 1;
other agents' jobs ran on the neighbouring cards, so latencies are indicative (the KFD
`evicted_ms` of the TP=2 / TP=4 engine processes of the long runs stayed 0).  Every
number below was measured on this tree unless it says [INFERENCE].

### Found and fixed

1. **The tree did not build** (CMake step 145 of 157): `inkling2_glue_rdna2.hip:1868:
   error: expected unqualified-id`.  `embed_lookup_kernel`, the one kernel the TP branch
   added, was not a template, and the repo's clang-format breaks the launch of such a
   kernel after the namespace (`inkling2_glue::` / `embed_lookup_kernel<<<...>>>(a);`),
   which hipify turns into a statement clang rejects [INFERENCE for the mechanism; the
   failing line is the observation].  Every other launch in the file is `kernel<0>` with
   the `<<<...>>>` on the next line: those kernels carry a dummy `template <int DUMMY =
   0>` for exactly this.  `embed_lookup_kernel` is one now (body unchanged); the pre-commit
   clang-format keeps that form.  Rebuild of the one translation unit: `build_exit=0`
   (136 s).
2. **The planner derived wrong pages for TP > 1.**  `kv_plan.py` built the TP=1 model and
   divided the specs by TP: TP=2 got a 19,968 B conv page (the rank allocates 33,792 B)
   and one conv group at every TP (the rank's owners make 2 at TP=2 and 3 at TP=4).  The
   real model builds the conv owner of a rank from the streams the rank holds: K and V
   are per kv head (128 channels each), the post-attention and MLP streams are
   replicated, so their share per kv head grows with TP:
   `W = 128 + 128 + 2 x 2304 / H_r` = 832 / 1408 / 2560.
   `layer_specs(cfg, tp)` now builds the rank's own `Inkling2LoopTextForCausalLM` on the
   meta device and reads the owners' specs; the planner no longer patches the platform at
   import (`v620_platform()` context manager).  The model needed no change:
   `Inkling2DecoderLayer` builds its conv owner from `self_attn.num_kv_heads` already.
3. **Stale docstrings.**  `Inkling2ConvOwner` and `static_conv_hip` said the conv inputs
   are fp16 and the state exact; at TP > 1 the post-attention input is fp32 (below).
4. **`long_context_check.py`** had no tensor parallelism: `--tensor-parallel-size`,
   `--ar-backend rdna` (`VLLM_RDNA_AR=1`) and `--text-only` (no vision tower, see the
   long-context section) were added.
5. **A test-order leak in the new planner tests** (found by the whole-session CPU run,
   not by the files alone: 1 failure and 13 errors in `test_inkling2_tiny_harness.py`,
   `'Inkling2LoopTextConfig' object has no attribute 'pad_token_id'`).  Building the
   engine config of the real checkpoint registers the repo's config classes with
   `AutoConfig` for the rest of the process, so the transformers reference of the tiny
   harness, which runs later, got the repo's text config instead of transformers' own.
   The `plan` fixture of `test_tensor_parallel.py` now builds the config under a copy
   of `CONFIG_MAPPING._extra_content`; planner tests followed by the harness: 46
   passed (before: 1 failed, 13 errors).

### The caches of a rank (real config, fp16, 64-token blocks)

| TP | kv heads / rank | K\|V page (layer, block) | conv row `W` | conv page, ring 3 / 10 rows | pool block | conv groups, k = 0 / k = 7 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 262,144 B | 832 | 39,936 / 133,120 B | 2 MiB | 1 x 48 layers / 4 x 12 |
| 2 | 4 | 131,072 B | 1,408 | 33,792 / 112,640 B | 1 MiB | 2 x 24 / 6 x 8 |
| 4 | 2 | 65,536 B | 2,560 | 30,720 / 102,400 B | 0.5 MiB | 3 x 16 / 10 (8 x 5 + 2 x 4) |

The pool block is 8 full-attention pages; a conv group holds `pool block // conv page`
layers (52 / 31 / 17 at k = 0, 15 / 9 / 5 at k = 7), so the groups are `cdiv(48, that)`.
Every case has five sliding-window groups of 8 layers (49 blocks at mnbt 1024) and one
full-attention group of 8 layers (`L / 64` blocks); a conv group needs 34 blocks at mnbt
1024.  The conv state per rank does not shrink like the KV does (the replicated streams),
but it is 34 blocks per group and ends at 51 - 68 MiB whatever TP is.

One max-length request, from the planner (k = 0, mnbt 1024, `max_num_seqs` 2; the
planner holds `F` = 2 x mnbt in-flight tokens):

| L | TP=1 | TP=2 | TP=4 |
| ---: | ---: | ---: | ---: |
| 201,600 | 3,429 blocks = 6.697 GiB | 3,463 = 3.382 GiB | 3,497 = 1.708 GiB |
| 262,144 | 4,375 = 8.545 GiB | 4,409 = 4.306 GiB | 4,443 = 2.169 GiB |
| 1,048,576 | 16,663 = 32.545 GiB | 16,697 = 16.306 GiB | 16,731 = 8.169 GiB |
| 1,048,576, k = 7 | 16,765 = 32.744 GiB | 16,833 = 16.438 GiB | 16,969 = 8.286 GiB |

The conv-branch section above scaled the TP=1 pages by `1 / tp` (16.272 / 8.136 GiB at k =
0 and 16.372 / 8.186 GiB at k = 7 for TP=2 / 4); the numbers here are 0.03 - 0.10 GiB
higher.  The planner and the engine agree: TP=1 mnbt 1024 needs 3,429 blocks at 201,600
tokens and the engine's boot extends the cache to 3,430 blocks; for TP=2 at 1,048,576 the
lengths the engine's own budgets hold are 1,039,104 / 1,044,032 / 1,047,424 (budgets 16.00
/ 16.07 / 16.11 GiB at mnbt 256 / 224 / 192) and `kv_plan.py --budget-gib` gives 1,039,296
/ 1,044,288 / 1,047,360.

### fp32 -> fp16: the conv state at TP > 1

At TP=1 every conv input is an fp16 GEMM output, so the fp16 ring is exact.  At TP > 1 one
of the four streams, the post-attention one, is the all-reduced fp32 sum of the `o_proj`
partials (the K, V and MLP streams are fp16).  The conv kernels read the taps of the
current chunk from that fp32 input exactly and store it rounded to nearest even in the
fp16 ring, so only the *history* of a later chunk or step is rounded: the same rounding
TP=1 applies once, at the epilogue of the `o_proj` GEMM, to the same quantity (up to the
fp32 summation order).  `test_conv_state_rounds_fp32_inputs_to_fp16` pins the kernel
semantics for the real TP shapes (kv heads, per-head stream width) = (2, 128), (4, 576),
(2, 1152).

What the rounding costs, in three tiers.  (1) CPU, fp32 compute (nothing else rounds),
router-stable `tiny1`, the 70-token prompt, max |dlogit|
(`test_tp_fp16_conv_ring_only_rounds_the_history`):

| TP | chunk | fp32 rings: TP=N vs transformers | fp32 rings: TP=N vs TP=1 | fp16 ring costs at TP=N / at TP=1 | TP=N vs TP=1, fp16 rings | stored entries whose fp16 rounding differs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1 | 7.7e-5 | 5.5e-5 | 1.68e-2 / 1.75e-2 | 8.4e-3 | 1.10% |
| 2 | 7 | 5.6e-5 | 5.4e-5 | 8.7e-3 / 8.5e-3 | 1.8e-3 | 1.03% |
| 4 | 1 | 6.3e-5 | 5.8e-5 | 1.60e-2 / 1.75e-2 | 8.5e-3 | 1.14% |
| 4 | 7 | 6.1e-5 | 6.0e-5 | 8.4e-3 / 8.5e-3 | 3.1e-3 | 1.05% |

With equal rings TP=N equals TP=1 to the fp32 summation order (5e-5); the fp16 ring costs
the same on N ranks as on one; and what N ranks and one still differ by with fp16 rings
is smaller than that cost: the summation order moves a stored value by a few fp32 ulps,
which flips its fp16 rounding for about 1% of the stored entries.  On the free-routing
`tiny8` the same experiment is dominated by a router flip that the rounding causes at
chunk 1 (13.5 logits at TP=1 and TP=N alike on one variant, 2.75 at TP=1 only on the
router-biased one): a discrete decision, not a TP effect, and the reason the quantitative
test uses `tiny1`.  (2) GPU, the HIP kernels in fp16 with thread ranks on one card
(`test_serving_path_gpu.py::test_tp_*`, mean / max |dlogp| against TP=1, 70-token prompts):

| checkpoint | scenario | TP=2 vs TP=1 | TP=4 vs TP=1 | TP=1 vs transformers | TP=2 vs TP=2 with the attention stream cast to fp16 |
| --- | --- | ---: | ---: | ---: | ---: |
| tiny1 | prefill 70 | 1.01e-2 / 0.072 | 1.09e-2 / 0.091 | 1.18e-2 / 0.099 | |
| tiny1 | chunked 70 / 16 | 1.06e-2 / 0.066 | 1.16e-2 / 0.093 | 1.21e-2 / 0.117 | 1.08e-2 / 0.112 |
| tiny1 | decode from 30 | 0.98e-2 / 0.067 | 1.05e-2 / 0.091 | 1.13e-2 / 0.078 | 1.02e-2 / 0.084 |
| tiny8 | prefill 70 | 2.36e-2 / 0.178 | 2.64e-2 / 0.226 | 2.96e-2 / 0.210 | |
| tiny8 | chunked 70 / 16 | 2.54e-2 / 0.171 | 2.50e-2 / 0.209 | 2.78e-2 / 0.229 | 2.60e-2 / 0.173 |
| tiny8 | decode from 30 | 2.38e-2 / 0.180 | 2.60e-2 / 0.226 | 2.81e-2 / 0.207 | 2.43e-2 / 0.165 |

The control (the TP=1 conv input, an fp16 cast of the all-reduced attention stream, in
place of the fp32 one) moves the logits by the same amount as every other fp16-level
perturbation (TP against TP=1, TP=1 against the fp32 reference); the ring's rounding is
not resolvable against the pipeline's own noise.  (3) The engine on the tiny checkpoints
(below): TP=2 against TP=1 gives last-position 0.0455 / 0.00919 (max / mean |dlogp|) on
`tiny1` and 0.1192 / 0.0263 on `tiny8`, the old layout's 0.046 / 0.0092 and 0.119 /
0.0263.

### Validation of the merged tree

| Check | Result |
| --- | --- |
| build, all csrc together (`build_lq.sh`, MAX_JOBS 8) | first build failed at step 145 (above); second `build_exit=0` after 238 s; one-TU rebuild after the template fix 136 s |
| CPU `tests/models/inkling2_loop`, `test_inkling2_loop_blocks.py`, `test_inkling2_loop_moe.py`, `test_inkling2_tiny_harness.py` (one session, `HIP_VISIBLE_DEVICES=` empty) | final tree: **716 passed, 92 skipped, 0 failed in 683 s**.  Earlier runs of the same selection: 694 passed in 625 s on the tree before the planner and ring tests existed; 702 passed, 1 failed, 13 errors with the planner tests (the `AutoConfig` leak, item 5 above) |
| CPU `test_tensor_parallel.py` | 156 passed in 280 s: +28 tests, among them the per-rank cache specs and page sizes at TP=1 / 2 / 4 against the numbers above, the planner tool against the owners' pages (k = 0 / 7), the planner's groups against the pages, the runner's `allocate_kv_cache` of the planner's config binding to every owner (ring `[blocks, H_r, rows, W]`, no page overlap inside a block), and the fp16-ring test above |
| planner mutation check (`kv_plan.py`, each mutant must fail the planner tests) | 3 of 3 killed: TP=1 specs divided by TP (the old derivation) 8 tests fail, TP=1 row width in the draft ring 8 fail, drafts ignored 3 fail; the restored file 18 pass |
| ring-test mutation check (`test_tp_fp16_conv_ring_only_rounds_the_history`, 4 cases, mutants of `paged_harness.py` on a copy) | mutant A, the ring dtype ignored (rings stay fp32): 4 of 4 killed; mutant B, the "fp16" ring is bf16: 2 of 4 killed (the chunk-7 cases stay under the 0.1 bound); harness restored (`cmp`) |
| GPU `test_serving_path_gpu.py`, whole file on the final tree (rebuilt binary) | 138 passed in 174 s (the 122 of the TP=1 run + the 16 new TP tests) |
| GPU `test_serving_path_gpu.py::test_tp_*` (new, thread ranks on one card) | 16 passed in 24 s (the first version had two failures: a strict last-position argmax assert on the 70-token prompt, whose top-3 logprobs are within 0.036; it uses the clear-margin check now) |
| kernels `test_inkling2_attn_rdna2.py` | 818 passed in 370 s (668 on the conv branch + the TP shapes of the conv ops and the fp32-input rounding cases) |
| kernels `test_inkling2_glue_rdna2.py` | 212 passed in 81 s on the first build; 212 passed in 81 s and the 16 TP thread tests above passed again on the build with the template fix |
| six-arch compile of the glue TU (hipified source, the repo build's compile command with the arch list replaced by gfx1030 gfx1100 gfx90a gfx942 gfx950 gfx1200) | `compile_exit=0` before and after the template fix; `llvm-objdump --offloading` lists all six code objects; `embed_lookup_kernel<0>` on gfx1030: 8 VGPRs, 26 SGPRs, no scratch, no LDS, occupancy 16 (the TP doc's figures for the non-template kernel), the other five arches hold the 0-VGPR stub |
| csrc mutation checks | not re-run: the fix is a dummy template parameter and a line break, no semantic change; the 17 bit-exact embedding tests (inside the 212) pass on the rebuilt kernel; the TP branch's 10 of 10 mutants of that kernel are recorded in `inkling2-glue-kernels.md` 7.5 |
| `uvx pre-commit run --files <changed files>` | clean (ruff, ruff format, clang-format, mypy, SPDX, tethering ...), run twice |

### Engine on the tiny checkpoints (`tiny_vllm_check.py`, HIP kernels, fp16, eager)

`tiny1-tp` (6 layers) and `tiny8-tp` (8 layers, loop x3), 24 q / 8 kv heads, head_dim 128, router
bias x100, default / chunked (chunk 16) / prefix caching / chunked + prefix caching; TP=1 on
HIP 3, TP=2 on 4,5, TP=4 on 2,3,4,5 with `--ar-backend rdna`; `tiny8` with `--tol-batch
0.15` as in the TP doc.  The tool's RESULT line:

| checkpoint | TP | default | chunked | prefix | chunked + prefix |
| --- | ---: | --- | --- | --- | --- |
| tiny1 | 1 | PASS | PASS | PASS | PASS |
| tiny1 | 2 | PASS | FAIL: `last/top1`, `batch/ids` | PASS | FAIL: `last/top1`, `batch/ids` |
| tiny1 | 4 | PASS | FAIL: `batch/ids` | FAIL: `prefix/hit_vs_cold` | PASS |
| tiny8 | 1 | PASS | PASS | PASS | PASS |
| tiny8 | 2 | PASS | PASS | PASS | PASS |
| tiny8 | 4 | PASS | PASS | PASS | PASS |

All four failures are the known near-tie flips of the tiny random model, not the ring or
the TP split: every other check of those runs passes (last-position mean |dlogp| 0.0100 -
0.0115, decode 8 of 8 ids for the prompts that are not near a tie, `batch/logprobs` 0.037 -
0.043 against 0.05, `prefix/same_step_hit` and `prefix/cross_step_hit` equal).  The fp32
reference of the 70-token prompt has three tokens within 0.036 of each other (log-probs
-1.7655 / -1.7831 / -1.8014 for the tokens 110 / 377 / 333; the engine's own noise is
0.04 - 0.06).  TP=1 chunked and the default runs at TP=2 / 4 keep that order; TP=2
chunked gives 333, 110, 377 at -1.7826 / -1.7844 / -1.7844 (the three within 0.002) and
TP=4 chunked 110, 333, 377.  The `prefix/hit_vs_cold` branch prompt (67 shared tokens and
a new tail) has an fp32 top-2 gap of 0.0267 at decode step 6, so the cold and the warm
run (different chunking) land on different sides of it at TP=4 and not at TP=1 / 2.
Max / mean |dlogp| of the engine at TP=N against the engine at TP=1 (`--dump` files, same
statistics as `--against`), default mode; the TP doc's old-layout numbers in the last
column:

| checkpoint | TP | last position | decode (8 steps) | batch | old layout: last / decode / batch |
| --- | ---: | ---: | ---: | ---: | --- |
| tiny1 | 2 | 0.0455 / 0.0092 | 0.0574 / 0.0099 | 0.0600 / 0.0102 | 0.046 / 0.0092, 0.057 / 0.0105, 0.057 / 0.0108 |
| tiny1 | 4 | 0.0385 / 0.0088 | 0.0541 / 0.0097 | 0.0622 / 0.0097 | 0.039 / 0.0088, 0.059 / 0.0101, 0.083 / 0.0104 |
| tiny8 | 2 | 0.1192 / 0.0263 | 0.2326 / 0.0270 | 0.2326 / 0.0257 | 0.119 / 0.0263, 0.230 / 0.0263, 0.230 / 0.0263 |
| tiny8 | 4 | 0.1347 / 0.0223 | 0.1667 / 0.0233 | 0.1796 / 0.0232 | 0.135 / 0.0223, 0.221 / 0.0237, 0.176 / 0.0242 |

The last-position statistics are the old layout's to three digits and the decode / batch
means are within 7% of them (the maxima move by the engine's run-to-run noise; the greedy
ids of every prompt equal TP=1's in the default runs, 8 of 8).  In the chunked + prefix
modes `prefix/hit_vs_cold` is exactly 0.00000 at TP=1 / 2 / 4 on both checkpoints (the
hit leaves the chunking of the rest unchanged), and 0.04 - 0.09 in the plain prefix mode,
where the hit changes it.

### Engine on the real checkpoint (`real_check.py`, `--max-model-len 8192`, mnbt 1024, prefix caching, vision, eager)

Run A and B at TP=1 (HIP 3), TP=2 (HIP 4,5, `--ar-backend rdna`), TP=4 (HIP 2,3,4,5); the
teacher-forced NLL of 799 positions of this document's text against run A, run B (the
TP=1 repeat) being the noise floor; no `--against` (its first-step top-8 limit false-fails
and would skip the later phases, the criterion is the NLL one):

| | TP=1 run A | TP=1 run B | TP=2 | TP=4 |
| --- | ---: | ---: | ---: | ---: |
| boot | 46.2 s | 33.3 s | 44.4 s | 46.6 s |
| "Model loading took" per GPU (with the vision tower) | 18.68 GiB | 18.68 GiB | 9.86 GiB | 5.44 GiB |
| engine "GPU KV cache size" (`max_model_len` 8192) | 69,038 tokens, 8.43x | 69,038, 8.43x | 266,676, 32.55x | 663,983, 81.05x |
| "capital of France" | coherent, ends "France's capital is Paris" in its thinking text | the same | the same | the same |
| "write one word: hello" | coherent | diverges mid-sentence (near-tie) | the same as A | the same as A |
| image (red circle on blue) | "a red circle centered on a blue background" | the same | the same | the same |
| 1,510-token prompt, prefill cold / warm (prefix hit) | 1.99 s / 0.12 s | 1.99 s / 0.12 s | 1.16 s / 0.08 s | 0.75 s / 0.06 s |
| mean logp over the 799 positions | -1.9824 | -1.9844 | -1.9829 | -1.9796 |
| mean \|dlogp\| against A (limit 2 x floor + 0.01 = 0.1027) | | 0.0463 (the floor) | **0.0437 PASS** | **0.0416 PASS** |
| p99 / max \|dlogp\| against A | | 0.328 / 0.483 | 0.303 / 0.615 | 0.316 / 0.587 |

The TP=1 repeat differs from run A by 0.0463, the TP runs by 0.0416 - 0.0437: the TP split
and the fp16 ring are inside the run-to-run noise of the engine (the TP doc's old-layout
numbers: floor 0.0399, TP=2 0.0425 / 0.0439, TP=4 0.0473 / 0.0515).  The tool's
`PREFIX HIT FAIL` line appears in every run, TP=1 included: the warm run generates other
ids than the cold one while the first-step top-1 is the same, the known near-tie spread of
the continuation (TP doc).  Max |dlogp| of the top-8 tokens the two runs share, hit
against cold / cold against cold again: 0.455 / 0.340 (TP=1 A), 0.534 / 0.528 (TP=1 B),
1.747 / 0.500 (TP=2), 0.906 / 0.673 (TP=4).

### Long context on the merged tree (real checkpoint, `long_context_check.py` / `real_check.py`, eager, prefix caching, `max_num_seqs` 2)

The TP=2 and TP=4 rows are text only (`limit_mm_per_prompt={"image": 0}`); the TP=1 runs
carry the vision tower (18.68 GiB of weights, encoder cache profiled).  The planner
numbers of the table at the top of this section are the "needs"; the engine's outcome at
boot:

Settings of every run here: `dtype=float16`, `enforce_eager=True`,
`gpu_memory_utilization=0.92` (the extensible KV cache takes what its margin leaves),
`max_num_seqs=2`, prefix caching on, `VLLM_RDNA_AR=1` at TP > 1; command
`tools/rdna/inkling2/long_context_check.py --ckpt ~/hfcache/inkling2-mini
--chat-template-file <ckpt>/chat_template.jinja --tensor-parallel-size N --ar-backend rdna
--max-model-len 1048576 --max-num-batched-tokens M [--prompt-tokens P] --text-only` (TP=1:
`--max-model-len -1`, mnbt 1,024, no `--text-only`, `--ar-backend` not needed); the boot
searches used `real_check.py` with the same engine arguments.

| TP | `--max-model-len` | mnbt | KV blocks after warm-up | outcome |
| ---: | ---: | ---: | --- | --- |
| 1 | auto-fit | 1,024 | 3,430 (6.70 GiB) | **201,600**: "GPU KV cache size" 201,658 tokens, 1.00x; the fit before the warm-up forward was 247,680 (8.11 GiB free), cut to 201,600 after it; boot 27.5 s |
| 2 | 1,048,576 | 256 | 16,382 (16.00 GiB) | refused: 16.14 GiB needed, estimated maximum 1,039,104 |
| 2 | 1,048,576 | 224 | 16,452 (16.07 GiB) | refused: 16.13 GiB needed, estimated maximum 1,044,032 |
| 2 | 1,048,576 | 192 | 16,498 (16.11 GiB) | refused: 16.13 GiB needed, estimated maximum 1,047,424 |
| 2 | 1,048,576 | 184 | 16,530 (16.14 GiB) | **boots** (the largest tried that does; 185 - 191 not tried): 1,049,655 tokens, 1.00x; boot 35.7 s |
| 2 | 1,048,576 | 176 | 16,542 | boots: 1,050,417 tokens |
| 2 | 1,048,576 | 160 | 16,566 | boots: 1,052,260 tokens |
| 2, with the vision tower | 1,048,576 | 160 | 15,940 (15.57 GiB) | refused: 16.12 GiB needed, estimated maximum 1,012,160 |
| 2, with the vision tower | 1,000,000 | 160 | 15,940 (15.57 GiB) | **boots**: 1,012,127 tokens, 1.01x; short answers and the image request ("a red circle centered on a blue ...") coherent; boot 41.9 s |
| 4 | 1,048,576 | 4,096 | 38,008 (18.56 GiB) | **boots**: 2,277,517 tokens, 2.17x (one request needs 17,499 blocks, planner); boot 40.9 s |
| 4 | 1,048,576 | 8,192 | 35,968 (17.56 GiB) | boots: 2,034,768 tokens, 1.94x; short answers coherent; boot 45.0 s (not timed with a needle) |
| 4, with the vision tower | 1,048,576 | 4,096 | 36,708 (17.92 GiB) | **boots**: 2,197,701 tokens, 2.10x; weights 6.21 GiB (5.61 text only); short answers and the image request answered; boot 47.7 s |

A rank's budget grows by 1.92 blocks (MiB at TP=2) per token of `max_num_batched_tokens`
given up (16,382 -> 16,566 blocks from 256 to 160; the engine's log: 19.32 -> 19.44 GiB
free after warm-up and 0.36 -> 0.29 GiB kept for transient activations), while the need
of one request shrinks by only 0.22 blocks per token (16,529 -> 16,508): the memory of
the chunk's forward, not the conv groups, decides how large a chunk TP=2 can take; 184 is
the largest tried that boots.  The conv branch's shrinking of the conv group (34 blocks
per group) removed the TP agent's chunk-64 limit (old layout: mnbt <= 96, 16,752 blocks
held at 96).  With the vision tower the model loads 0.60 GiB more per rank (9.59 against
8.99 GiB) and TP=2 no longer fits the full length at mnbt 160: the engine's estimate was
1,012,160 tokens, and `--max-model-len 1000000` booted with 1,012,127 (row above).  Even
a chunk of 1 token would add at most 160 x 1.92 MiB = 0.30 GiB (15.87 GiB held against
16.12 GiB needed) [INFERENCE from the slope above, not run].  The margin the extensible
KV cache keeps free (`KV_CACHE_MARGIN_FLOOR_BYTES`, 3 GiB on ROCm) would have to fall to
about 2.45 GiB (18.83 + 0.03 - 0.29 - 16.12) for the full length to fit with vision
[INFERENCE, not run].

Needle runs (`long_context_check.py`: a passphrase in the middle of a numbered fact list,
the instruction at the end; `--prompt-tokens 262144` builds 261,052 tokens, the 1M timing
run uses this size so that a prefill takes minutes, not half an hour):

| | TP=1 | TP=2 | TP=4 |
| --- | ---: | ---: | ---: |
| `--max-model-len` / mnbt | 201,600 / 1,024 | 1,048,576 / 184 | 1,048,576 / 4,096 |
| prompt | 181,434 tokens (90% of the length) | 261,052 tokens (25%) | 261,052 tokens (25%) |
| cold prefill | 382.0 s = **475 tok/s** | 392.7 s = **665 tok/s** | 195.5 s = **1,335 tok/s** |
| warm prefill (full prefix hit) | 0.30 s | 0.22 s | 0.16 s |
| decode at that depth | **35.5 ms/token** | **26.4 ms/token** | **27.0 ms/token** |
| needle at depth 0.5 | FOUND | FOUND | FOUND |
| control (4,088 tokens) | FOUND | FOUND | FOUND |

The TP=1 numbers are the conv branch's (475 tok/s, 35.6 ms/token): the merge did not
change the single-card path.  KFD `evicted_ms` stayed 0 for the TP=2 and TP=4 engine
processes (`kfd_monitor.sh`; the TP=1 needle ran before the monitor started).  TP=4
prefills twice as fast as TP=2 with a chunk 22x larger (4,096 against 184), the TP=2
chunk being what the 16 GiB budget allows; the decode step does not get faster from TP=2
to TP=4 (26.4 against 27.0 ms/token): both are eager-mode launch and all-reduce latency
bound [INFERENCE, not profiled]; the graphs of the DSpark / graphs branch are the open
lever.

Depth scaling, every needle run above and the ~90% one at TP=2 (the instruction and the
needle at depth 0.5; the prefill rate is the average over the whole prompt):

| run | prompt | cold prefill | warm prefill | decode | needle / control |
| --- | ---: | ---: | ---: | ---: | --- |
| TP=1, limit 201,600 | 181,434 (90%) | 382.0 s = 475 tok/s | 0.30 s | 35.5 ms/token | FOUND / FOUND |
| TP=2, limit 1,048,576, mnbt 184 | 261,052 (25%) | 392.7 s = 665 tok/s | 0.22 s | 26.4 ms/token | FOUND / FOUND |
| TP=2, limit 1,048,576, mnbt 184 | **942,628 (89.9%)** | **2,751.8 s = 343 tok/s** | 0.42 s | **51.8 ms/token** | FOUND (Fact 33705) / FOUND |
| TP=4, limit 1,048,576, mnbt 4,096 | 261,052 (25%) | 195.5 s = 1,335 tok/s | 0.16 s | 27.0 ms/token | FOUND / FOUND |
| TP=4, limit 1,048,576, mnbt 4,096 | **942,628 (89.9%)** | **1,382.7 s = 682 tok/s** | 0.34 s | **32.1 ms/token** | FOUND (Fact 33705) / FOUND |

The ~90% prompt at TP=2 took 46 minutes to prefill: the average rate halves from 261k to
943k tokens (the full-attention layers read the whole history for every chunk of 184
tokens [INFERENCE: not profiled]), and the decode step grows by 25.4 ms from depth 261k
to 943k (0.037 ms per 1,000 tokens of context).  A per-token cost `a + b x depth` fitted
to the two TP=2 prefills (a = 0.96 ms, b = 2.1 ns per token of depth) gives about 3,290 s
(55 minutes) for a full 1,048,576-token prompt [INFERENCE from the fit, not run].  The
request fit in the 16,530 blocks the engine held (14,729 full-attention blocks plus the
window and conv groups' share, about 130 blocks at mnbt 184; the planner's need of the
full length at that chunk is 16,513); KFD `evicted_ms` stayed 0 for the engine processes.

TP=4 (mnbt 4,096) prefilled the same 942,628 tokens in 23 minutes (the average rate falls
from 1,335 to 682 tok/s, a factor 1.96; 1.94 at TP=2), and its decode step grew by 5.1
ms from depth 261k to 943k, a fifth of TP=2's growth (the factor is not 2; not
profiled).  The same two-point fit gives about 1,650 s (28 minutes) for a full
1,048,576-token prompt at TP=4 [INFERENCE, not run].

### Open risks and next steps (merged tree)

* **TP=2 with the vision tower does not fit 1,048,576 tokens** at mnbt 160 (15.57 GiB
  held, 16.12 GiB needed); `--max-model-len 1000000` boots with it (1,012,127 tokens,
  1.01x, image request answered).  The text only engine boots at the full length with 16
  blocks to spare at mnbt 184 (16,530 held, 16,508 - 16,529 needed).  Choices: TP=2 +
  vision at about 1.0M, lower `KV_CACHE_MARGIN_FLOOR_BYTES` (3 GiB on ROCm, all models),
  or TP=4 (8.2 GiB needed at 1M; booted with the vision tower, 2.10x).
* **Eager only.**  FULL_AND_PIECEWISE graphs and the DSpark draft at TP > 1 belong to the
  graphs / DSpark branch and were not run on this tree; every decode number above is
  eager (26 - 36 ms/token).  Graph memory comes out of the same KV budget, which at
  TP=2 and 1M is 17 blocks (MiB) wide at mnbt 184 (16,530 held, 16,513 needed) and 58 at
  mnbt 160 [INFERENCE: the capture sizes or the chunk will have to shrink].
* **hipify and clang-format.**  A non-template `__global__` launch that the repo's
  clang-format wraps after the namespace breaks the CMake build only (the JIT builds of
  the kernel tests do not go through hipify).  `embed_lookup_kernel` was the only one;
  a build of the glue TU belongs in whatever gates a csrc change.
* **csrc mutation checks were not re-run** for the template change (no semantic change;
  the 17 bit-exact tests pass on the rebuilt kernel).
* **Prefix-hit comparisons are noise-dominated**: `real_check.py` reports
  `PREFIX HIT FAIL` at TP=1 too (the continuation of a warm and a cold prompt differs
  at a near-tie while the first-step top-1 is equal), and `tiny_vllm_check.py`
  `prefix/hit_vs_cold` failed in 1 of the 8 TP > 1 runs with prefix caching (the fp32
  reference has a 0.0267 top-2 gap at decode step 6 of that prompt).  In the chunked +
  prefix runs `prefix/hit_vs_cold` and `prefix/cross_step_hit` are bitwise (0.00000).
* **The fp16 ring rounding at TP > 1 cannot be resolved on the GPU** against the
  pipeline noise (0.01 mean |dlogp| at TP=1 against transformers); the evidence is the
  CPU fp32-compute test.
* **Latencies are indicative**: other agents ran on the neighbouring cards; the TP=4 jobs
  held HIP 2 - 5 for long stretches.

Next steps: (1) merge the graphs / DSpark branch on this tree and re-run the TP=2 / 4
boots at 1M with graphs (the 16 GiB budget at TP=2 is the constraint), then the needle
decode with graphs; (2) decide the TP=2 + vision length; (3) a full 1,048,576-token
prompt at TP=2 (about 55 minutes of eager prefill, see above) once graphs are in.
