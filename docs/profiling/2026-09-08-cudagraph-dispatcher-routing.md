# Cudagraph Investigation — Dispatcher Routing Finding (2026-09-08)

**Branch**: `rdna_extras` @ `ba4c50d08`
**Key new finding**: With `cudagraph_mode: PIECEWISE` alone (no FULL / FULL_DECODE_ONLY component), vLLM's `cudagraph_dispatcher.dispatch()` always returns `CUDAGraphMode.NONE` because `mixed_mode()` is NONE for the standalone mode and the PIECEWISE key set is never populated.

This was unknown earlier in the session. It means every probe so far under `cudagraph_mode: PIECEWISE` was probably falling through to eager-mode dispatch, NOT actually capturing a piecewise cudagraph. The pointer-trace data (`VLLM_LOG_GDN_PTRS=1`) corroborates this: 288 replay-time entries, **zero** `capturing=True` entries — the function is invoked only outside any graph capture.

## Relevant code (`vllm/v1/cudagraph_dispatcher.py`)

`init` assertion (line 49-61): piecewise requires attention in `splitting_ops`
**OR** `is_breakable_cudagraph_enabled()`. Otherwise the dispatcher itself
aborts.

`dispatch()` (line 235-324): first looks up FULL (line 307), then PIECEWISE
(line 313-318), then falls back to NONE (line 320-324).

`initialize_cudagraph_keys()` (line 166-233): populates keys only when
`cudagraph_mode.mixed_mode() != CUDAGraphMode.NONE` (line 189). For
standalone `PIECEWISE`, this branch is skipped, leaving
`cudagraph_keys[CUDAGraphMode.PIECEWISE]` empty.

So under standalone PIECEWISE the dispatch loop at line 313-318 fails,
asserts `CUDAGraphMode.NONE in allowed_modes` (line 320) — which it always
is — and returns `(CUDAGraphMode.NONE, BatchDescriptor(num_tokens))`.
The forward path then runs in **eager mode** despite `cudagraph_mode`
reporting PIECEWISE elsewhere in logs.

## Implications for prior probes

Every "cudagraph garbage" output in this session (including the dense-AWQ
*correct* control row, the deterministic vs non-deterministic splits, the
`@torch.compiler.disable` test, and the pointer-trace test) was the eager-mode
output for that configuration. The bug is therefore not a cudagraph capture
issue per se but an eager-mode correctness issue on this v2 runner for
the GDN-hybrid AWQ path.

This also explains the `0 capturing=True` entry count: the function
`_forward_core_decode_non_spec` is called only in eager dispatch mode.

## Suggested next step (revised)

Probe `--enforce-eager` (Cell A control) on the current tree. If it is
correct on the current tree, the dispatcher finding explains why every
"cudagraph garbage" result actually came from eager mode and most prior
hypotheses (which assumed captured-graph replay effects) were misdirected.
The real fix is then in the compiled-graph layer (inductor-produced v2 runner
forward) or in the v2 runner's handling of GDN hybrid for AWQ int4.

If Cell A is also garbage on the current tree, my prior 11 commits
regressed something and the reverse-bisect that was started earlier becomes
the path.
