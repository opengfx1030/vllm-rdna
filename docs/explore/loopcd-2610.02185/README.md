# Explore / Work Queue: LoopCD (`arXiv:2610.02185`) Engine Support for Looped Transformers

**Status**: queued next in the RDNA engine roadmap (scoped against `Inkling2Loop` on `rdna_extras`).
**Reference**: *"Decoding Looped Transformers Better for (Almost) Free"* (`arXiv:2610.02185`, Liu et al., Oct 2026).

---

## 1. Motivation & Core Mechanism

Looped Transformers (`Inkling2Loop`, Huginn, Ouro) execute a shared recurrent block across $R$ passes (`loop_num_passes = 2` on `Inkling-2-Mini`: prelude `layers[0:6]`, recurrent loop `layers[6:18]` $\times 2$, coda `layers[18:24]`). Standard decoding reads only the final loop state $h_R$ and discards earlier loop states $h_1, \dots, h_{R-1}$.

**LoopCD** uses an earlier loop pass (typically pass 1, $h_{\text{first}}$) as an aligned, training-free "amateur / weak predictor" to contrast against the final pass $h_{\text{final}}$ with guidance strength $\omega \ge 0$:

1. **`LoopCD-Hidden` (pre-coda hidden-state contrast — zero coda / `lm_head` overhead)**:
   $$h_{\text{cd}} = h_{\text{final}} + \omega (h_{\text{final}} - h_{\text{first}}) = (1 + \omega)\, h_{\text{final}} - \omega\, h_{\text{first}}$$
   Combined hidden states then pass once through `coda_layers` and `lm_head`.
2. **`LoopCD-Logits` (logit-space contrast)**:
   $$z_{\text{cd}} = z_{\text{final}} + \omega (z_{\text{final}} - z_{\text{first}})$$
   with adaptive uncertainty gating $\omega_t$ (down-weighting contrast when the final distribution is already sharp) and plausibility masking ($z_{\text{final}}(v) < \log \beta + \max_u z_{\text{final}}(u) \implies -\infty$) so low-probability tail noise is never amplified.

---

## 2. Exact Fit with `Inkling2Loop` on RDNA2 (`vllm/models/inkling2_loop/amd/model.py`)

### 2.1 Zero-VRAM State Availability (`v.prev_blk`, `v.prev_x0`, `v.prev_x1`)

In `Inkling2LoopModel.forward_hip` (`vllm/models/inkling2_loop/amd/model.py`), right before `visit = 1` begins (`for visit in range(1, passes):`), the workspace **already copies and preserves** Pass 0's complete recurrent exit so Pass 1 can compute cross-pass `message_input`:

- `v.prev_blk[:, :n_prev]`: Pass 0's loop `AttentionResiduals` block states (`fp32`).
- `(v.prev_x0, v.prev_x1)`: Pass 0's lazy `fp32` partial-block output sum at layer 17 exit.
- When `passes == 2` (`Inkling-2-Mini`), `v.prev_blk`, `v.prev_x0`, and `v.prev_x1` are **never overwritten** during Pass 1 and remain live right at the `coda_layers` entry! Thus **`LoopCD-Hidden` requires `0 B` of new workspace memory**.

### 2.2 Critical Architectural Invariants for `Inkling2Loop`

1. **AttentionResiduals (`v.blk`) coupling**: Coda layers (`18..23`) and final readout attend over all prior block states (`blk[:, 0..n]`) via `inkling2_glue_attn_res`. `LoopCD-Hidden` must support contrasting both:
   - the loop exit partial block `(x0 + x1)` against `(prev_x0 + prev_x1)`, **and**
   - optionally the loop block-state stack `v.blk[:, num_pre + 1 : n]` against `v.prev_blk[:, :n_prev]` (controlled by an ablation flag `blend_block_states: bool = True`).
2. **Stateful Coda (`layers[18:24]`) & Prefix-Cache Safety**:
   - Coda layers `18..22` (SWA + `sconv`) and `23` (Full attention + `sconv`) write per-token KV cache and conv ring state.
   - **Rule for `LoopCD-Hidden`**: Pass a GPU-resident per-token scale `omega_per_token: float32[T]` (`InputBuffers`). Set `omega_per_token[t] = 0.0` on non-terminal prompt prefill rows (`t != query_start_loc[r+1] - 1`) and on requests with `loopcd_alpha == 0.0`. This guarantees **100% prefix-cache compatibility** across requests with different `loopcd_alpha` values while keeping a single `FULL_AND_PIECEWISE` HIP graph for mixed batches!
   - **Rule for `LoopCD-Logits`**: Do **not** run `coda_layers` twice with `write_state=True` (which would corrupt SWA/Full KV & `sconv` rings). Support two clean weak-logit modes:
     - *Mode L1 (Skip-coda / Final-Norm Readout)*: Apply `inkling2_glue_attn_res` (`self._final_query`, `self._final_norm_w`) directly to Pass 0's `(prelude_blk + prev_blk, prev_x0 + prev_x1)` for the `B` sampled positions only (`logit_indices`), then run `lm_head` (`+2.1 ms` at `b=1`, zero KV/conv state writes).
     - *Mode L2 (Read-only Coda Shadow Pass, optional)*: Run `coda_layers` on the `B` sampled tokens only (`T_weak = B`) with a read-only `skip_kv_and_conv_write` flag so canonical KV/conv state is untouched (`~+2.8 ms` coda + `+2.1 ms` `lm_head` at `b=1` **[INFERENCE]**).
3. **Compatibility with DSpark Speculative Decoding (`ADAPTIVE=1` + `32k draft_vocab`)**:
   - **With `LoopCD-Logits`**: Contrastive logits $z_{\text{cd}}$ are computed on all `1 + K` verified target positions before `rejection_sampler` / greedy verifier runs, so target verification remains **100% mathematically lossless with respect to the LoopCD distribution**! (Note: `LoopCD-Hidden` mutates coda KV states on drafted tokens, which stays exact on accepted tokens while rejected tokens roll back via `seq_lens` + conv ring naturally!)
   - **Bonus — Zero-Draft / Confidence Signal**: Pass 0 vs Pass 1 hidden cosine / logit KL divergence is an instant per-token uncertainty metric that can feed `AdaptiveK` (`speculator_options.adaptive_k`) to shrink $K$ before a hard reasoning branch!

---

## 3. Queued Implementation Plan (Phased)

| Phase | Scope | Key Files | Est. Effort |
|---|---|---|---|
| **P1: `LoopCD-Hidden` + Per-Request `SamplingParams`** | • Add `loopcd_alpha` (`float = 0.0`), `loopcd_mode` (`"hidden" \| "logits"`), `loopcd_adaptive` (`bool = False`), `loopcd_beta` (`float = 0.1`) via `SamplingParams.extra_args` (or engine config default) -> `InputBuffers.loopcd_alpha[max_num_tokens]` (`0.0` on non-last prefill tokens).<br>• Add native HIP kernel `inkling2_glue_loopcd_blend` (`csrc/rocm/inkling2_glue_rdna2.cu`, `~5 µs` inside `FULL` HIP graph) blending `(x0, x1)` + `blk[:, num_pre+1:n]` with `(prev_x0, prev_x1)` + `prev_blk` right before `coda_layers`. | `csrc/rocm/inkling2_glue_rdna2.cu`, `vllm/models/inkling2_loop/amd/model.py`, `vllm/v1/worker/gpu/input_batch.py` | ~2–3 hrs |
| **P2: `LoopCD-Logits` + Plausibility & Adaptive $\omega_t$** | • Save Pass 0 readout (`fp16 [B, H]`) at `logit_indices` into workspace.<br>• Compute weak logits + fused contrastive logit kernel (`z_cd = z_final + omega_t * (z_final - z_weak)` with `beta` plausibility mask and entropy/margin adaptive $\omega_t$) in `compute_logits` before `Sampler` & DSpark verifier. | `vllm/models/inkling2_loop/amd/model.py`, `vllm/v1/worker/gpu/model_runner.py`, `csrc/rocm/inkling2_glue_rdna2.cu` | ~3–4 hrs |
| **P3: Quality & Speed Qualification on `par1-cs25`** | • Verify `omega = 0.0` is **0-ULP bit-identical** to baseline.<br>• Sweep $\omega \in \{0.15, 0.25, 0.5, 0.75\}$ (fixed vs adaptive) across `LoopCD-Hidden` (blend exit vs blend exit+blocks) and `LoopCD-Logits` (skip-coda vs read-only coda) on GSM8K / AIME-subset / HumanEval + DSpark acceptance rate & tok/s on 4× V620 (`b4`).<br>• Test `loop_num_passes=1` fast mode vs `passes=2 + LoopCD` accuracy/FLOP trade-off. | `tests/models/inkling2_loop/test_loopcd.py`, `tools/rdna/inkling2/eval_loopcd.py` | ~2–3 hrs |
