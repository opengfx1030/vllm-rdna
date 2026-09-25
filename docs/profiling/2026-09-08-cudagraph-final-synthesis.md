# Cudagraph Diagnostic — Qwen3.5 Hybrid AWQ Int4 (Final)

**Branch**: `rdna_extras` @ `d71721c79` + GDN `qwen_gdn_full_forward` wrap
**Date**: 2026-09-08 (updated same day)

## TL;DR (updated)

There are **two independent bugs**, not one cudagraph-replay bug. FA-RDNA2 and HIP W4A16 are not the culprits (they fire in the correct cells).

| Configuration | `"The capital of France is"` | Verdict |
|---|---|---|
| `--enforce-eager` | `' Paris...'` | ✅ |
| Dynamo `backend=eager`, `cudagraph_mode=NONE` | `' Paris...'` | ✅ |
| Inductor, `cudagraph_mode=NONE` (no graphs) | `' not the only of the capital of...'` | ❌ inductor |
| Inductor + `custom_ops=["all"]`, no graphs | same garbage | ❌ not CustomOp default |
| Inductor + `ir_enable_torch_wrap=false`, no graphs | same garbage | ❌ not IR wrap |
| Inductor + `epilogue_fusion/pattern_matcher/split_reductions=false` | same garbage | ❌ not fusion |
| Dynamo `backend=eager` + `PIECEWISE` | `'ductductduct...'` | ❌ graph replay |
| Dynamo `backend=eager` + `PIECEWISE` + `cudagraph_copy_inputs` | crash `size 2048 vs 8` | ❌ |
| Default inductor + `PIECEWISE` | looping / CJK garbage | ❌ both bugs |

Dense AWQ (Qwen2.5-0.5B) + inductor + PIECEWISE remains correct. The inductor failure is hybrid-GDN specific. The `duct` failure is piecewise replay of eager-backend subgraphs (warmup tokens, input-blind).

**Correct kernels, no graphs:** `backend=eager`, `cudagraph_mode=NONE`, `VLLM_USE_RDNA2_FA=1`. That is the only compile-on configuration that matches `--enforce-eager`.

## Isolation evidence (this session)

1. **Not CUDA-graph replay (for the looping garbage).** `cudagraph_mode=NONE` + inductor still produces the exact same looping string. Graphs were confirmed skipped (`Skipping encoder and decoder CUDA graph capture`).
2. **Dynamo tracing is fine.** `backend=eager` (dynamo FX, no inductor codegen) is correct. Compile of range `(1, 2048)` takes 0.02 s.
3. **Inductor codegen is wrong** even with `combo_kernels=false`, `custom_ops=["all"]`, and `ir_enable_torch_wrap=false`. Output is bitwise-identical across those inductor variants.
4. **GDN `vllm::qwen_gdn_full_forward` splitting op fires** (warmup log) and is in `splitting_ops`. Wrapping the full GDN layer (OLMo pattern) does **not** fix inductor. It **does** prevent dynamo from tracing into GDN RMSNorm (`torch.accelerator.device_index` skip crash when the wrap is removed under `backend=eager`).
5. **Piecewise + eager backend** captures graphs (2–3 s, 0.71 GiB) then replays `'duct'` — the input-blind pattern. `cudagraph_copy_inputs=true` dies at capture with `The size of tensor a (2048) must match the size of tensor b (8) at non-singleton dimension 1`.

## Where the issue starts

`Qwen3_5Model.forward()` → `self.language_model.model(...)` → per-layer forward. The per-layer `QwenGatedDeltaNetAttention.forward()` at `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:888` is decorated with `@eager_break_during_capture` and dispatches to `forward_cuda` / `forward_hip`. **Neither of these downstream Python methods is invoked during the captured-graph replay path.** Two pieces of evidence:

1. The `_forward_core_decode_non_spec` env-gated `VLLM_LOG_GDN_PTRS` logger at line ~1745 (added in commit `ba4c50d08`) emitted 672 lines during the cell B probe, all with `capturing=False nd=0 nat=8 ...`. Zero entries captured during cudagraph graph capture, zero entries captured during replay — but the function fired 9 times per replay (matching 16 GDN layers stacked); therefore the function *was* called per step, but never inside `torch.cuda.is_current_stream_capturing() == True`.

2. The `forward_hip` env-gated logger added in this session (later reverted) had a `@torch._dynamo.allow_in_graph` allow_in_graph helper so it survived the captured graph compilation. It emitted zero entries during the cell probe. So `forward_hip` is also not on the captured-graph decode path on this configuration.

The captured-graph replay path therefore lives at a layer **above** `_forward_core_decode_non_spec` and `forward_hip`. The only plausible remaining levels are:

- `Qwen3_5Model.forward()` (line ~612 of `vllm/model_executor/models/qwen3_5.py`),
- the vLLM v2 runner / inductor-wrapper code at `vllm/compilation/decorators.py:680` and `vllm/compilation/wrapper.py:183`,
- or the kernel-side path captured into the cudagraph graph (CUDA Graph replay doesn't elevate `torch.cuda.is_current_stream_capturing()` to `True` because replay is not capture).

## What I ruled out with evidence

| Hypothesis | Test | Result |
|---|---|---|
| AWQ int4 GEMM kernels unsafe under cudagraph | Cell F: `VLLM_DISABLED_KERNELS=RDNA2W4A16LinearKernel` still garbage | disproven |
| FA-RDNA2 attention unsafe under cudagraph | Cell Afa: `VLLM_USE_RDNA2_FA=0` still garbage; dense AWQ cudagraph correct | disproven |
| Cudagraph dispatcher routes to cudagraph | `_forward_core_decode_non_spec` fires 672 times, graphs captured (`Graph capturing finished in 3 secs`) — dispatcher routes correctly | n/a (not a bug) |
| Pointer staleness on `non_spec_state_indices_tensor` | `--max-num-seqs 1` (single slot, no padding) still garbage | disproven |
| Static state-index buffers skipped | applied `use_static_state_buffers` for piecewise: still garbage + non-deterministic | disproven |
| Async copy race (`non_blocking=True`) | `non_blocking=False` on all 8 copies: still garbage + non-deterministic | disproven |
| Buffer sizing vs capture sizes | `--max-num-seqs 8` (buffers size to 8 = max capture size): still garbage | disproven |
| Dmesg / uncommitted VA page faults | no `UTCL`, `gfxhub`, `ring:24`, `FAULTY` events during probes | disproven |
| `combo_kernels=false` | already set; narrowing GEMMs to `splitting_ops` made output deterministic prefix-stable garbage but didn't fix it | partial |
| GDN decode core captured (`@torch.compiler.disable`) | forced eager: still garbage + non-deterministic | disproven |
| `@eager_break_during_capture` on `forward()` is the regression | removed decorator: still garbage | disproven |
| Forward path not in captured graph at all | 672 eager entries + 0 capture-time entries means it IS captured but dispatch lands elsewhere | disproven |

## Non-determinism finding

Across permutations, garbage at `temperature=0` is sometimes deterministic and sometimes not. The determinism is incidental (depends on memory allocator reuse landing on stable garbage), not diagnostic. **Non-determinism at `temperature=0` for identical prompts means the captured-graph replay is reading memory that varies between runs** — but the variance is in the uninitialised/contents dimension, not the pointer dimension (pointer invariant tested and held).

## What's pending

The captured-graph replay path goes through `qwen3_5.py`'s `forward()` at line 612, which then runs through `compilation/decorators.py:680` (`TorchCompileWithNoGuardsWrapper.__call__`) and `compilation/wrapper.py:183`. A diagnostic logger at this layer (with `@torch._dynamo.allow_in_graph` on the helper), gated by `VLLM_LOG_GDN_PTRS=1`, would catch the captured-graph entry. My `forward_hip` attempt proved the pattern is correct; what was missing was placement at the right Python level.

## Fix direction (next session)

Item 1/3 below was implemented this session (`vllm::qwen_gdn_full_forward` + `_attention_ops`). It did **not** fix inductor garbage. It is still required so `backend=eager` does not dynamo-trace into GDN RMSNorm (`device_index` skip).

Remaining:

1. **Bisect inductor codegen** on the hybrid residual/MLP subgraphs around the GDN split (dump FX with `debug_dump_path`, compare aten vs inductor kernels vs `backend=eager`). Dense AWQ inductor is correct, so the bad subgraph is hybrid-specific.

2. **Fix piecewise replay of eager-backend subgraphs** (`duct`). `CUDAGraphWrapper.__call__` ignores replay args and returns `entry.output` (weak ref to capture-time tensors). Need the model-runner static input copy to land in those addresses, or stop weak-ref'ing intermediate piecewise outputs.

3. **Do not use `cudagraph_copy_inputs=true`** on this model: capture dies with `size 2048 vs 8`.

## Recommended unblock path forward

Working correctness path today (TP=2 or TP=4, FA-RDNA2 + HIP W4A16):

```
--compilation-config '{"cudagraph_mode":"NONE","backend":"eager","compile_ranges_endpoints":[]}'
```

or `--enforce-eager`. Do not use default inductor + PIECEWISE on this hybrid.

To actually get cudagraphs: either (a) fix inductor lowering for the hybrid residual/MLP pieces, then re-enable PIECEWISE, or (b) make `CUDAGraphWrapper` replay of eager-backend piecewise subgraphs consume the runtime inputs (today it returns capture-time `entry.output` and yields `duct`).

## Files in this round

- `docs/profiling/2026-09-08-cudagraph-gdn-root-cause.md` (committed in `598a3f8ca` and parents) — the evidence base for ten+ disproven hypotheses.
- `docs/profiling/2026-09-08-cudagraph-dispatcher-routing.md` (committed in `d71721c79`) — dispatcher routing analysis.
- This file: final synthesis.

Plus the 12 commits on origin `rdna_extras` from this session (see `git log --oneline` from earlier turn for the full list).
