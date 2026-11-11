# Speculative target-logit NaN accounting

The V1 runner previously mapped logit row `i` directly to request `i`.
With MTP2, each request has verification rows plus a bonus row, so NaNs
in later rows could be omitted from the per-request report and fail-stop.
This fix sums each request's complete packed span using CPU draft lengths
already held by `SpecDecodeMetadata`. Async outputs keep an immutable copy
of the lengths for that step. Malformed speculative row layouts raise;
non-speculative missing-row behavior and pipeline ranks with no logits
are preserved.

No GPU kernel, transfer or synchronization is added. The existing NaN
counts and output-copy stream are reused. When NaN diagnostics are off,
the async path does not copy draft-length metadata for this purpose.
This fixes target-logit accounting for all V1 speculative methods, including
native MTP2; it changes no sampling arithmetic. V2 already reduces all rows
per request and is unchanged.

`run_cpu_ast.py` extracts the actual mapper and async methods from source
and exercises them with small CPU doubles. `cpu-ast-result.json` records
13 passing checks without importing PyTorch or initializing a GPU. This is
bounded logic evidence, not a real-device or full-module pytest run.
The tracked pytest file also checks synchronous late-row failure and the
actual async rejection-output parsing branch. Frozen-environment pytest v2 passed **7 tests**, importing the actual patched
module through an isolated one-file Python overlay. HIP6 alone was visible:
platform capability discovery initializes that device during module import,
while these tests use CPU tensors. Before/after source/core/boot pins match.
The first, GPU-hidden v1 attempt failed during capability discovery before
pytest; its log is retained separately. Neither test run changed serving
source or the installed core. Both ran between arms under the reservation lock.

This accounting fix alone does not check drafter logits. In particular,
greedy draft-token selection can discard NaNs, and dense MTP K2 currently
resets its control header on each invocation. That separate opt-in drafter
fault-reporting path requires its own fix and qualification. The regular
Mono target's timeout test poisons all rows, which may already be caught
by the earlier first-row check; do not describe every full-output timeout
as a demonstrated escape of the old mapper.

Current MonoKernel startup requires the V2 runner, whose target-row NaN reduction
is already complete. This V1 fix protects other V1 speculative configurations;
it does not establish current V2 drafter fault coverage.
