# Gemma residual path: cached scale and in-place fused add-norm (HI2 27B, TP=2, 2026-11-15)

Branch `feat/hi2-residual` (from `feat/hi2-integration` @ `d3f2827e1b`). Python only, no
`_C` / `_rocm_C` rebuild. Code: `vllm/model_executor/layers/layernorm.py`
(`GemmaRMSNorm`, `gemma_fused_add_rms_norm_inplace`), `vllm/model_executor/models/qwen3_next.py`
(`Qwen3NextModel.forward` layer-0 guards), `vllm/envs.py`. Tests
`tests/kernels/core/test_gemma_residual_direct.py`, microbenchmark
`benchmarks/kernels/benchmark_gemma_residual_direct.py`. Labels: **[measured]** = produced by
a command listed here, **[derived]** = arithmetic on measured values, **[inference]** =
reasoning a run must confirm.

**Switch.** `VLLM_RDNA_GEMMA_RESIDUAL_DIRECT=1`, default **off**, gfx10x only, fp16, 2-D
activations only. Flag off: the same functional ops as before run (the only additions are
two unused attributes per norm). The switch is global for `GemmaRMSNorm` with a residual on
gfx10x; the audit below covers Qwen3.5 / Qwen3.8 (`qwen3_next`, `qwen3_5`, both MTP drafters).
The gemma / gemma2 / gemma3 / rnj1 decoders use the same call pattern but were not run.

## Why

Per call, the functional path (`gemma_fused_add_rms_norm`) does

| Step | Launches |
| --- | ---: |
| `weight.float()`, `+ 1.0`, `.to(fp16)` (scale rebuilt every call) | 3 |
| `activation_indices` clones of `x` and `residual` (IR `func_impl_fn`) | 2 |
| `_C.fused_add_rms_norm` | 1 |

`[measured]`: 6 kernels per call, eager and in a replayed graph (table below). A decode step
makes 128 such calls (63 input norms, 64 post-attention norms and the final norm; layer 0's
input norm has no residual) [fusion plan, R1].

## Change

- **Scale cache.** With the flag on, each `GemmaRMSNorm` owns a non-persistent fp16 buffer
  `rdna_scale16` and a `weight_loader` that runs `default_weight_loader` and then refreshes
  the buffer with the same arithmetic as before, `(w.float() + 1.0).to(fp16)`, **copied into
  the existing buffer** so a captured graph keeps reading the same address. Every loader that
  honours `param.weight_loader` refreshes it, reload included. Until a real load ran
  (dummy weights, a loader that writes `param.data` directly) `_rdna_scale_ready` stays
  False and the functional ops run: a missed refresh costs speed, never correctness.
  `Parameter._version` was **not** used: `param.data.copy_(...)`, which
  `default_weight_loader` does, leaves it at 0 `[measured, torch 2.12.0+rocm7.14.0]`
  (`p.copy_` bumps it, `p.data.copy_` does not), so it cannot invalidate a cache.
- **Direct op.** `torch.ops.vllm.gemma_fused_add_rms_norm_inplace(x, residual, scale, eps)`
  (`mutates_args=["x", "residual"]`) calls `torch.ops._C.fused_add_rms_norm(x, residual,
  scale, eps)`, the kernel the IR `vllm_c` impl reaches, with no clones. Declaring the
  mutation lets inductor keep the in-place update or copy where another consumer still needs
  the old value (see the compiled chain below). Taken for fp16 2-D `x` / `residual`; 3-D
  and other dtypes keep the functional ops.
- **Same bits as the clones.** The kernel takes its 8-wide vector path only for 16-byte
  aligned, row-contiguous buffers, and the functional path always hands it fresh clones. So
  the in-place call is used only for contiguous, 16-byte aligned `x` and `residual`; any
  other layout runs the functional IR op and copies the result back into `x` / `residual`.
  `x` and `residual` sharing storage raises.
- `Qwen3NextModel.forward`: with the flag on, a caller-owned `inputs_embeds` is cloned and an
  aux-layer-0 capture is cloned (aliasing audit, below).
- First weight refresh logs once: `VLLM_RDNA_GEMMA_RESIDUAL_DIRECT=1: GemmaRMSNorm with a
  residual uses the cached fp16 scale and the in-place fused add-norm`. Absent in the boot
  log with the flag on means the loader bypassed `weight_loader` and the arm measures nothing.

## Measurements

Box `par1-cs25`, HIP 4 = V620 bus `df` (resolved through torch device properties before
every run), venv `venv-7.14.0_0.32.0` (torch 2.12.0+rocm7.14.0), `_C` / `_rocm_C` from
`tree-hi2i`, under `flock ~/w4a8_runs/hi2-gpu4.lock`. **Shared box**: HIP 2,3 and others
were running other agents' jobs; the microbenchmark is a same-run old-vs-new comparison.
Logs `~/w4a8_runs/hi2-residual/bench3.log`, `bench3.json` (final code), `pytest-final.log`.
fp16, hidden 5120, eps 1e-6,
Gemma-style weights (N(0, 0.3) plus 16 channels at N(0, 3)) and activations with 4 massive
channels (x60). IR priority `vllm_c, native`, as `platforms/rocm.py` sets on gfx10x.

Command (inside the lock):
`HIP_VISIBLE_DEVICES=4 python benchmarks/kernels/benchmark_gemma_residual_direct.py --compile`

Time = one CUDA graph of 128 chained norm calls, replayed 30 times, minimum, divided by 128.
Launches from `torch.profiler` (kernel events per call).

| rows | bit-exact (output and residual) | old ms/call | new ms/call | saved ms/call | x128 ms | launches/call old -> new, eager | in replayed graph | inductor chain |
| ---: | :---: | ---: | ---: | ---: | ---: | :---: | :---: | :---: |
| 1 | yes | 0.0185 | 0.0033 | 0.0152 | 1.94 | 6 -> 1 | 5.0* -> 1 | 6 -> 1 |
| 2 | yes | 0.0186 | 0.0033 | 0.0153 | 1.96 | 6 -> 1 | 5.3* -> 1 | 6 -> 1 |
| 8 | yes | 0.0187 | 0.0034 | 0.0153 | 1.96 | 6 -> 1 | 6 -> 1 | 6 -> 1 |
| 24 | yes | 0.0188 | 0.0036 | 0.0153 | 1.96 | 6 -> 1 | 5.2* -> 1 | 6 -> 1 |
| 2048 | yes | 0.1291 | 0.0609 | 0.0682 | 8.72 | 6 -> 1 | 5.3* -> 1 | 6 -> 1 |

`*` ROCTracer dropped events in that profile (the log warns `duplicate flow start`); the
eager and inductor profiles of the same code count exactly 6 per call every time, and an
earlier run of the same script (`bench2.log`, before the helper refactor) shows 6 in the graph
for rows 2, 24 and 2048. Per-call times of the two runs agree within 0.0003 ms.

- The x128 column is the isolated norm chain, 128 calls per step: **1.94 to 1.96 ms per decode
  step at 1 to 24 rows** `[derived from measured]`. The fusion plan's estimate for the same
  change is
  2.2 ms at c1 (1.0 kernel + 1.2 gap) `[inference]`; the microbenchmark graph includes the
  inter-kernel gaps of a replayed graph, the model step has other neighbours.
- The 2048-row numbers are cache-warm: the 20 MiB activations stay in the on-die cache
  between calls `[inference]`, so a real prefill chunk, whose GEMMs evict them, saves
  more than 8.7 ms per chunk-forward `[inference]`. Not measured in the model.
- The "inductor chain" column compiles 8 chained calls with `torch.compile(backend=
  "inductor")` and counts launches per call: the functional op is opaque to inductor, so its
  6 launches stay; the mutating op compiles to exactly 1 launch per call, no functionalization
  copies, input mutation of the compiled function's arguments included.

Tests (`pytest tests/kernels/core/test_gemma_residual_direct.py`, HIP 4): 15 passed. They
cover bit-exactness (output and residual) at rows 1, 2, 8, 24, 2048, in-place-ness, the not
loaded fallback (inputs untouched), reload refreshing the buffer in place (same address),
row-strided and misaligned inputs, 3-D inputs, the alias error, and a 3-layer decoder chain
(layer-0 alias, add-norm pairs, final norm) eager and inductor-compiled, equal to the
functional chain bit for bit. The gating tests run off GPU; the rest skip off gfx10x.

## Aliasing audit

The in-place path overwrites `hidden_states` (`x`, becomes the normed value) and `residual`
(becomes `x + residual`) of every `GemmaRMSNorm(x, residual)` call. Safe only if nothing
else reads the old values afterwards.

| Consumer | Finding |
| --- | --- |
| Layer-to-layer chain (`qwen3_next.py` `Qwen3NextDecoderLayer.forward`) | `x` is the previous mlp / attention output (tensor-parallel all-reduce result), consumed once by the norm. `rdna_ar_all_reduce` returns a fresh `at::empty_like(in)` (`csrc/rocm/rdna_allreduce.cu:359`), never a shared staging buffer. The residual is the same tensor through all layers and is only read by the next add. Safe. |
| Layer-0 embedding alias | `residual = hidden_states` (the embedding) at layer 0, then the first add-norm mutates it in place. The embedding is `embed_tokens(input_ids)`, a fresh tensor, so nothing else holds it. Covered by the decoder-chain test (eager: the embedding buffer is overwritten, output equal to the functional chain). Safe, with the two guards below for the cases where something else does hold it. |
| Caller-owned `inputs_embeds` (multimodal / prompt-embeds runner buffers `inputs_embeds`) | Would be written through. Guard: `Qwen3NextModel.forward` clones it when the flag is on (one launch per step, only on that path). |
| Aux hidden-state capture (`_maybe_add_hidden_state`) | Layers >= 1 compute `hidden_states + residual` (a new tensor) before the next layer mutates, so they are safe. Layer 0 with `residual is None` stores the embedding itself, which becomes the mutated residual: guard clones that one entry. |
| MTP drafter (`qwen3_5_mtp.py` L205-237, `qwen3_next_mtp.py`) | The drafter gets the target's final-normed `hidden_states` and reads it only through `pre_fc_norm_hidden` (no residual, out of place) into `fc`, a fresh tensor that is the drafter layer's layer-0 residual; its final `self.norm(hidden_states, residual)` mutates only drafter-local tensors. `inputs_embeds` is normed out of place. Safe. |
| Target final norm (`qwen3_next.py` L747) | Mutates the last mlp output and the residual; the returned `hidden_states` is the only live reference. The residual is discarded. Safe. |
| Speculative verify | Same decoder layers with rows = requests x (1 + spec tokens), at most 24 in the recipe: contiguous 2-D, covered by the 24-row test. The GDN spec-verify kernel reads inputs inside its own layer, before the post-attention norm. Safe. |
| FULL / PIECEWISE cudagraph | Captured kernels run in the same order on the same static buffers; in-place updates of graph-owned tensors are what the clones' replacements do. The scale buffer is persistent and refreshed in place (same address), so capture never bakes a stale scale. Inductor sees the mutation through `mutates_args` and copies only where another consumer needs the old value (compiled chain: 1 launch per call). Not run in the full engine. |
| Other `GemmaRMSNorm` callers with a residual (gemma, gemma2, gemma3, rnj1, interns2_mobius) | Same `norm(x, residual)` rebinding pattern; **no guards** for a caller-owned `inputs_embeds` or aux layer 0 there. Do not set the switch for them. |
| AR + norm fusion passes (`allreduce_rms_fusion.py`) | They match the functional norm ops; with the flag on and gfx10x the opaque functional op is gone from the graph for the fp16 2-D case, so a pattern for it would not fire. None is active in the HI2 recipe (`rdna_ar` is a custom all-reduce) `[inference]`. |

## Scoreboard arm

```bash
bash tools/rdna/rdna_hint_flex/scoreboard.sh residual VLLM_RDNA_GEMMA_RESIDUAL_DIRECT=1
```

Baseline arm: the recipe with the flag off. Quality: bit-exact per norm call, so greedy
output, PPL and top-k KL should match the baseline exactly; any difference is a bug.

What the arm should show, relative to the baseline's c1 trace of 2,451 launches
(`order-breakdown` c1a) `[derived from the 6 -> 1 launch change, x128 calls]`: 2,451 ->
1,811 launches per step (-640), elementwise 633 -> 249 (-384) and copy / memset 370 -> 114
(-256) `[inference: those families' attribution]`. ITL: about -1.9 to -2.2 ms at c1
(microbenchmark and plan) `[inference]` until the arm measures it.
