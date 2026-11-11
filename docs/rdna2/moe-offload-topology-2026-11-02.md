# MoE expert-offload topology (TP2/TP4), HIP collectives, and the v0.33 graph fix

Date: 2026-11-02. Branch `feat/moe-offload-topology`, built on v0.33.0.

## Summary

| Item | Result |
|---|---|
| PR #149 (crispyging) adapted | Env names normalised, one shared all-gather helper, topology validated in Python |
| HIP all-gather + fused vocab argmax | 3-14x faster per collective; about 1% decode ITL on the non-offload path; no measurable end-to-end change under offload |
| v0.33 offload regression | Fixed (`4af2aa0c2e` on `rdna_extras`): PLE ops are now eager breaks under breakable CUDA graphs |
| Qualified | TP2 (96 slots) and TP4 (48 slots), `FULL_AND_PIECEWISE`, prefix caching on, MTP2 + KV tier |

Topology: TP=2 is the MoE-offload mode (256 local experts/rank, default 96 slots).
TP=4 resident is the full-GPU mode; TP=4 offload (128 local experts/rank, 48 slots) is also qualified.

## Credit and adaptation of PR #149

The PR's own commits are authored by an agent identity, so the work was re-applied rather than merged.
`34727e4e1e` is a verbatim squash of the PR head; `fffd733287` is the adaptation, with a
`Co-authored-by: crispyging` trailer.

| PR name | Fork name |
|---|---|
| `FLASHNEXT_HEAD_ALLREDUCE` | `VLLM_RDNA_HEAD_ALLREDUCE` |
| `FLASHNEXT_GATHER_ALLREDUCE` | `VLLM_RDNA_GATHER_ALLREDUCE` |
| `FLASHNEXT_HEAD_GLOO` | dropped (host bounce in the hot loop, not graph-safe) |

- One shared `tensor_model_parallel_all_gather_via_all_reduce` in `vllm/distributed/communication_op.py`,
  routed through `tensor_model_parallel_all_reduce`, so the zero-pad fallback follows whichever all-reduce
  backend is active.
- `tools/rdna/mono/serve_offload.sh` validates the topology with `rdna_expert_cache_policy.validate_offload_topology`
  and sizes the cache with `default_cache_slots` (3/8 of local experts: 96 at TP2, 48 at TP4).
- The PR's source-text test was replaced by thread-fabric oracles (`tests/distributed/test_rdna_gather_allreduce.py`).
- The zero-pad fallback is for hosts without P2P. It cannot be exercised here (this box has working P2P),
  so it is covered by the oracles only.

## HIP all-gather and fused argmax (`19a010b476`)

`rdna_ar` gained a one-shot last-dim all-gather and a fused (value, index) vocab argmax sharing one
exchange phase with the existing all-reduce. Eligible when the staged payload is at most 64 KiB, contiguous,
and fp16/bf16/fp32. Bit-exact: gather is a copy; argmax ties go to the lowest rank and NaN wins.
Kill switch: `VLLM_RDNA_AR_GATHER=0`.

Microbench (microseconds per call, RCCL to HIP):

| Collective | RCCL | HIP | Speedup |
|---|---:|---:|---:|
| TP4 gather [1,640] | 67.1 | 13.6 | 4.9x |
| TP4 gather [24,640] | 87.1 | 30.5 | 2.9x |
| TP4 argmax, 1/8/24 rows | 171.7 / 191.6 / 187.3 | 21.8 / 22.4 / 23.0 | 7.9-8.6x |
| TP2 gather [1,640] | 64.5 | 12.5 | 5.2x |
| TP2 argmax | 171-185 | 13.1-13.4 | 13-14x |

End to end, non-offload (flash-next MTP2, TP4, ABBA, all probes pass): ITL median is about 1% lower on the
1k cells and neutral at 16k; aggregate tok/s is dominated by run-to-run noise (an A arm swung 75.5 to 64.9).

End to end, TP4 offload (MTP2, KV tier, 48 slots; B = HIP gather, A = kill switch):

| Cell | B1 | A1 | B2 |
|---|---:|---:|---:|
| 1k/512 c=1 decode agg tok/s | 31.94 | 35.71 | 34.04 |
| 1k/512 c=1 ITL med (ms) | 61.1 | 54.6 | 52.6 |
| 1k/512 c=8 decode agg tok/s | 68.47 | 72.33 | 67.87 |
| 16k/1k c=8 decode agg tok/s | 38.16 | 40.08 | 39.11 |

The two B arms differ by as much as B differs from A, so offload cannot resolve the effect: ITL there is
55-200 ms and is dominated by expert copies and the host break. No win and no loss is claimed under offload.

## The v0.33 offload regression

Symptom: with `--monokernel` and the expert cache, request 1 was correct and request 2 failed with
`NaNs detected in logits` (the runner fails stop under `--monokernel`). Plain Flash-Next with
`VLLM_USE_BREAKABLE_CUDAGRAPH=1` returned `!!!!` from request 2 onward.

Root cause: breakable CUDA graphs run with compile mode NONE, so no op is split out of a captured segment
unless it is an `@eager_break_during_capture` function. The AMD PLE ops read runtime request metadata
through the forward context (n-gram ids from `query_start_loc`, short-conv state routing). They were
captured with the first request's layout frozen in and replayed stale. The offload launcher is the only
path that forces breakable graphs for piecewise capture, which is why resident mono and the standard gates
(compiled piecewise) never showed it.

Fix: `qwen4_exp_amd_ple_ngram_ids`, `_ngram_embedding`, `_ngram_embedding_pinned`, `_embedding_rdna`
and `qwen4_exp_ple_short_conv` use `@eager_break_during_piecewise_capture` (`159df5e497`). They leave
PIECEWISE (mixed/prefill) segments, where the per-request layout would be frozen, and stay inside FULL
uniform-decode graphs, where their metadata lives in persistent buffers. The first version
(`4af2aa0c2e`) used `@eager_break_during_capture`, which on RDNA also breaks FULL captures and added an
eager segment per PLE layer to every breakable decode graph. Outside a breakable capture both decorators
call straight through, so compiled piecewise graphs are unchanged.

Every recipe now serves `FULL_AND_PIECEWISE`. Re-verified after the PIECEWISE-only change: TP2 offload
9/9, breakable plain Flash-Next 9/9, 27B EXL3 MTP=2 9/9 (previously only served PIECEWISE), TP4 offload
validation pass with decode at parity with the break-in-FULL build (1k c=1 32.2, 1k c=8 70.5,
16k c=8 38.7 tok/s; noise-dominated, so no speedup is claimed).

How it was localised (each arm is a boot and 3 prompts x N rounds):

| Experiment | Result |
|---|---|
| Pristine `rdna_extras` | fails: not caused by #149 or the HIP collectives |
| KV tier, MTP, GDN, QSA, K1, P2P, slot count, TP toggled | still fails |
| Per-pass expert-cache check (device slot == host expert, 100+ passes) | 0 mismatches |
| `--enforce-eager` offload | 18/18 clean |
| `FULL_DECODE_ONLY` offload, TP4 | 18/18 clean (decode graphs fine) |
| Resident mono + breakable, no offload | fails |
| Plain Flash-Next + breakable, no mono, no offload | fails (`!!!!`) |
| PLE ops break out of PIECEWISE segments | fixed |

`timing` in `rocm_mono_decode` now reports every nonfinite K2 input with the row count and whether the
call is a replay; this is what placed the first NaN at layer 1 attention input.

## Qualification (FULL_AND_PIECEWISE, prefix caching on)

| Config | Requests | NaN | Prefix cache |
|---|---|---|---|
| TP2 offload, 96 slots, MTP0, KV off | 18/18 | 0 | on |
| TP2 offload, 96 slots, MTP2, KV tier (launcher default) | 9/9 | 0 | 6144/20824 tokens hit |
| TP4 offload, 48 slots, MTP2, KV tier (launcher default) | 9/9 | 0 | 12288/20824 tokens hit; 5.2k prompt 5.2 s cold, 1.9 s warm |
| TP4 offload, 48 slots, validation driver | probes, PPL 4.84, concurrency pass | 0 | cold TTFT 8.9 s, warm 1.05 s (8.4x) |

Operational note: booting two servers at once on this box has stalled engine init (twice, TP2 after the
expert cache loaded). Cause not isolated; the same TP2 config boots in about 6 minutes when alone.
Boot servers one at a time. A boot also stalls servers that are already serving: its pinned-memory work
held the host for seconds, one rank waited 22 s at a collective, and the mono kernel's 2 s barrier
timeout then poisoned the output (NaN logits in one K2 split measurement). Measure with no
boot in flight: boots take an exclusive `flock` on `$HOME/w4a8_runs/boot.lock`, so a measurement can
hold it shared (`flock -s`).

## Merging with #148

PR 148 already conflicts with v0.33 `rdna_extras` in 7 KV files (it predates the v0.33 merge). This branch adds
two more conflicts, `tools/rdna/mono/serve_offload.sh` and `tools/rdna/mono/sweep_offload_policy.py`;
`rdna_expert_cache_policy.py` auto-merges. Land this branch first.
