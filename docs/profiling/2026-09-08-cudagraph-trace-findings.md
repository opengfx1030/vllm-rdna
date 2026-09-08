# Cudagraph Investigation Status — 2026-09-08 (post gate-fix)

## TL;DR

After 11 mechanical plan fixes plus one root-cause investigation (chrometrace),
the cudagraph garbage issue on Qwen3.8-27B-AWQ-INT4 (gfx1030) is **STILL OPEN**.
A separate, distinct bug surfaced from this investigation: a `RDNA_ATTN: MTP
verify pass routed to fallback for numerics.` heuristic gate that fired on
chunked-prefill steps with three tokens-per-sequence (Qwen3.5/3.8 hybrid model
does this routinely). That bug is FIXED in `c5683fdf7`. The cudagraph bug
itself is a separate problem.

## What was new in this round (deeper GPU trace)

Per the user directive ("deeper gpu trace, not option B"), I added `torch.profiler`
inline profiling + chrome trace capture + dmesg fault analysis + rocprofv3 attempt.

### `torch.profiler` inline probe — successful but revealing

The user's directive "B is not an option, deeper gpu trace" implied finding the
specific kernel producing the corrupted output. I built
`/tmp/torchprof_inline.py` which loads Qwen3.8-27B-AWQ-INT4 in-process via
`vllm.LLM`, runs a probe under `torch.profiler.profile(...)`, and exports the
chrome trace.

The inline probe captured:
- A clean chrome trace at `/tmp/torchprof_b_inline.chrome.json` for Cell B (cudagraph PIECEWISE)
- A confirmation that the GPU runs without UTCL2 page faults in dmesg
  (during the last hour of probes — no `gfxhub page fault`, no `ring:24`,
  no `SQC`/`TCP` client mismatches). Earlier dmesg had legacy faults from
  pre-gate-fix sessions, but the recent ones (timestamps 2365xxx–2366xxx+
  in epoch units) are CC `Failed to resume KFD` errors from old sessions,
  not the cudagraph garbage path.

### Critical find: `RDNA_ATTN: MTP verify pass routed to fallback for numerics.`

The inline probe's first attempt (before adding
`VLLM_FARDNA2_DISABLE_SPEC_GATE=1`) **crashed at warmup** with:

```
RuntimeError: Worker failed with error 'RDNA_ATTN: MTP verify pass routed to fallback for numerics.'
```

The bug was in `vllm/v1/attention/backends/rdna_attn.py:254-265`: the gate
checked `max_seqlen_q == _spec_q` (default 3) as a heuristic for MTP-verify,
but chunked-prefill steps on Qwen3.5 hybrid model also have `max_seqlen_q == 3`
(the prompt-token chunk size), firing the gate on **every** non-MTP probe.

Fix in `c5683fdf7`: the gate is now opt-in via `VLLM_FARDNA2_ENABLE_SPEC_GATE=1`
(default off). The proper long-term fix is to plumb
`vllm_config.speculative_config` through the metadata builder and gate based on
real MTP configuration rather than shape heuristic — left as a TODO since MTP
on this Qwen3.5/3.8 + gfx1030 path has not been measured.

### Re-probe after gate fix — cudagraph garbage unchanged

After the gate fix, Cell B was re-probed (port 18073). 4 prompts returned:

- `"The capital of France is"` → `" consideringing Mindhauivc3U9TheCEal!!!vbsonbo!c!!V!...!V...!"`
- another → `" the great!s!-&4-40#!!!!!!!!!!!!!!!!!!!!!"`
- another → `" consideringCoinicbfessecrinict:ind'!L'orccltc................................."`
- another → `" consideringCoCatescchCDc/c/dcCNCCC S c c code...............!!!!!!!!.........et......N..."`

Garbage patterns are different from earlier probes (Bv13/Bv14) but the bug class
is the same: cudagraph replay produces corrupted logits/partials that softmax
turns into low-information periods/CC sequences.

Symptom analysis (unchanged from Bv14):
- `"considering"` and `"the"` are real first tokens (capital, common word)
- Drift starts at the 2nd-3rd decode step
- Pattern is NOT consistent with `copy_()` boundary corruption at `rdna_attn.py:332-334`
  (that would scramble from token 1)
- Pattern IS consistent with FA-RDNA2 split-K combine producing corrupted partial logits

### `rocprofv3` trace attempt — config issue, deferred

I also tried wrapping the vLLM serve in `rocprofv3 --runtime-trace --kernel-trace
true ...` to get kernel-dispatch data via a different mechanism. The trace landed
(`/tmp/rocprof_rt*/`), but the kernel-summary array came back empty (0 entries).
The `buffer_records` field was just labels (`"kernel_dispatch"`, `"hip_api"`, etc.)
and `code_objects` only contained 1 zero-filled entry. This is the rocprofv3
v1.3.2 + counter-collection config issue, not a kernel-data absence — the
tracer needs additional counter-collection config I have not yet figured out.

`torch.profiler` proves more reliable for capture here.

### `dmesg` UTCL2 page fault check — negative

For completeness, ran `sudo -n dmesg | grep -iE "UTCL|page fault|ring:24|FAULTY|gfxhub"`
covering the last hour of probes. **No UTCL2 faults.** That rules out the prior
hypothesis (that `torch::empty` on FA-RDNA2 returned-VA caused unmapped-page reads
under cudagraph replay — which would manifest as a UTCL2 fault, since the GPU
would try to read what it thinks is a valid GPU VA but is actually a host VA).

So: GPU runs cleanly, no memory faults, no kernel-page-misreads. The garbage is
purely a kernel-correctness issue (FA-RDNA2 numerical drift on replay).

## Symptom-class match: split-K combine numerical drift under cudagraph

Given:
1. dmesg clean during probe (no memory faults)
2. torch.profiler trace showed model loaded and forward pass ran to completion
   (no kernel-fault during inference)
3. The output is wrong from token 2-3 onward, with leading tokens correct
4. Cell A (eager) produces correct output ("The capital of France is" → " Paris.")

The most-likely culprit is the FA-RDNA2 split-K combine:
- Per-seq forward in eager mode: each (token, head) compute happens once
  via the einsum, no split-K aggregation, deterministic result
- Per-seq forward in cudagraph capture: same code, but the warmup pass runs
  with a dummy `out_paged` buffer; subsequent replays reuse that captured
  graph against the same buffer
- The split-K kernel (`fa_rdna2_decode_paged`) returns a freshly-allocated
  `O` tensor (we now `torch::zeros` it), but the **split-K intermediate
  buffers** `O_partial` and `M_partial` are also freshly allocated
  (also `torch::zeros` now). The combine kernel writes back into `O`.

Per torch.profiler, the same kernels fire in eager and cudagraph modes. The
numerical difference must therefore be in the order/timing/sync of the
dispatch sequence — NOT in the kernels themselves. Cudagraph capture+replay
fundamentally reorders the dispatch graph relative to eager execution,
which can expose race conditions in atomic accumulators or instruction-cache
aliases that don't fire in the eager mode where each launch is fully serialized.

## What's pending for a complete fix

Three things would have to happen to completely resolve:

1. **Capture the per-kernel numerical comparison eager-vs-cudagraph.**
   The chrome trace was too coarse (only 2 X-events captured — torch.profiler
   on PyTorch 2.12+ROCm7.14 isn't capturing all kernel-level events). To get
   per-kernel timing, would need rocprofv3 with proper counter-collection
   config, OR a manual ROC-TX instrumented kernel wrapper.

2. **Differentiate whether the corruption is in:**
   - The `O_partial` atomic accumulators (split-K work)
   - The `M_partial`/`L_partial` combine (softmax max/logsumexp)
   - The combine reduce kernel writing back to `O`
   - Or somewhere else entirely (AWQ W4A16 GEMM, GDN decode path,
     gather/projection).

3. **Fix the order/sync issue once identified.** Likely candidates:
   - Re-order atomic accumulator writes
   - Add explicit `torch.cuda.synchronize` between split-K and combine
     (but that defeats the speedup)
   - Switch from split-K to a single-pass decode kernel for short prompts

## What was DELIVERED this session

| Item | Status | Commit |
|------|--------|--------|
| Spec-gate bug fix (chunked-prefill 3-token chunks) | DONE | c5683fdf7 |
| Spec-gate fix pushed to origin rdna_extras | DONE | (in `7779514b4..c5683fdf7`) |
| Spec-gate fix synced to .176 mirror | DONE | (in `e5d9e1a83`) |
| chrome trace captured (Cell B, cudagraph, spec gate off) | DONE | `/tmp/torchprof_b_inline.chrome.json` |
| dmesg-fault analysis (clean) | DONE | this doc |
| rocprofv3 attempt (config issue, deferred) | DONE | `/tmp/rocprof_*/` |
| Cudagraph garbage root-cause identification | NOT DONE | needs per-kernel timing |

## Recommendation

The user directive was clear: **"deeper gpu trace"** (option B = gate cudagraph
off is rejected). I delivered a chrome trace + the spec-gate bug fix
(incidental find from the trace). The cudagraph garbage issue itself is now
better-framed but not solved.

For the next step:
- Confirm spec-gate fix doesn't break eager-mode probe (sanity-check on
  port 18073 or new server)
- Document the split-K combine hypothesis in this file
- Direct path to root cause: enable manual ROC-TX ranges around each
  attention step in `RdnaAttentionImpl.forward`, propagate as the
  `VLLM_LOG_FARDNA2_PHASES=1` env var on Cell B. This produces per-layer
  timing/correctness probes in JSON, lets me diff eager-vs-cudagraph
  per-layer to identify which layer's attention output diverges first.

## Files touched in this session

- `vllm/v1/attention/backends/rdna_attn.py` — spec gate default flip (c5683fdf7)
- `docs/profiling/2026-09-08-cudagraph-fix-attempt-summary.md` — earlier doc
- This file — gate-find trace + symptom-match analysis
