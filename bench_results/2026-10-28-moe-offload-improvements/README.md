# MoE offload follow-up receipts

Target: Flash-Next AWQ INT4, gfx1030/HIP 7.14, TP2/EP2 on HIP GPUs 6/7.
Kernel/cache checks use GPU8. No GPU9 allocation was used.

**Current qualification blocker:** the long native-copy MTP2 arm produced NaN
logits after 810/1024 requested output tokens. The native backend remains opt-in;
the full-model optimization is not qualified. See `native-ab/serve.log.txt` and
the rejected `copy-long/qualification.json`. Causality is not established.

## Wider K2 gate

`wide-k2/arm.sh.txt` records the serving recipe. Both arms use tensor copies,
96 expert slots, LFU/LRU decay 128, MTP2, segmented FULL_AND_PIECEWISE graphs,
3.75 GiB GPU KV per rank and a 16 GiB aggregate native CPU KV pool. TunableOp
uses the same committed lookup rows. Prefix state resets before each benchmark.

`wide-k2/comparison.json` contains aggregate output throughput, TTFT, TPOT, ITL,
MTP acceptance and forced-output warnings. Aggregate output includes prefill:

- 1k/512 c1: 15.986 -> 19.076 tok/s (+19.33%).
- 1k/512 c8: 56.012 -> 48.149 tok/s (-14.04%).
- 16k/1k c1: 22.305 -> 21.954 tok/s (-1.57%).
- 16k/1k c8: 28.763 -> 27.279 tok/s (-5.16%).

Keep gate 8. Each arm ran once, serially; expert-cache histories and MTP
acceptance differ, so these numbers do not isolate a kernel speedup. Greedy,
PPL, prefix and concurrency checks passed in both arms. PPL was 4.81/4.79;
prefix cold/warm TTFT was 12.64/2.32 s and 12.34/2.31 s. Each concurrency run
passed 48 checks with zero hard failures, retaining near-tie/soft differences.
Both summaries report zero new GPU fault markers.

Detailed benchmark outputs are preserved. Random-prompt benchmarks force their
output lengths; repetition/garbage screening warnings are recorded separately
and must not be treated as normal-EOS semantic correctness evidence.

## Native copies

`native-cache-tests.txt`: 29 integrated scratch cache tests passed on GPU8,
including native copies of all six payloads, consumer-stream changes, stable
addresses and invalid inputs rejected before writes. `native-receipts/`
preserves standalone compilation/test drivers. The forced native-loop fallback
also passed on HIP 7.14; this is not older-HIP runtime qualification.

`copy-api-probe.jsonl` and its driver are a HIP API prototype, not the integrated
C++ backend or a full-model speedup. CPU enqueue medians include descriptor
construction where labeled `batch_rebuild`; `batch_prebuilt` excludes it.
GPU event intervals can include host starvation and are not pure DMA time.
The 16-miss tensor-copy timing needs broader repetition before generalization.

The integrated serving extension registered 159 ROCm ops; all 119 focused
policy/cache/model tests passed on GPU8 without skips (`integrated-native-tests.txt`).
`native-quality/` records the full TP2 native-copy quality run: greedy, PPL 4.78,
prefix caching (12.05 s cold / 2.29 s warm), 48 concurrency checks with zero
hard failures, and the after-load greedy probe all passed. No GPU faults were
reported. The client A/B hook failed on an unavailable, unexercised KV-load
metric; that run is not a completed performance comparison. The client gate
was corrected without changing engine code, retaining strict KV-boundary tests.

`native-startup-counters.json` is a snapshot after startup/probes/warmup, not a
steady workload benchmark. Both ranks report 48 native caches and equal native
and batch call counts. Expert payload pools are 11,767,037,952 device bytes and
31,378,685,952 host bytes per rank, including their small slot maps.

`native-ab/` preserves the replacement same-process run. Short ABBA cells
completed: c1 native 16.25–22.94 versus tensor 18.73–20.05 tok/s; c8 native
58.61–64.31 versus tensor 48.76–51.06 tok/s. These are observations, not a pure
copy speedup: MTP acceptance, generated trajectories and expert transfer bytes
vary. Native c8 submitted 500–582 GB across both ranks, tensor 683–706 GB.
Only the local prefix cache was reset; external CPU KV state was retained.
No KV loads were exported in the recorded short cells, so they do not qualify
CPU KV restoration. Forced-output warnings are preserved alongside the results.

The following cold 16k/c1 tensor arm completed 1024 tokens with TTFT 105.39 s;
this differs from older benchmarks that ran a full unmeasured ready-check
request before measurement. The native arm then ended at 810 tokens because
both workers detected all-NaN logits. The CLI reported `completed=1,failed=0`
for the partial stream, but our exact-length gate rejected it. Long c8 and CPU
profiling did not run. The replacement run is not a successful qualification.

`matched-copy/` isolates actual `PackedExpertCache._copy` with the same immutable
bank, destination addresses, stream and bytes for both backends, balanced ABBA
ordering and 30 samples per arm/cell. Median wall completion (tensor -> native):

- 1 miss: 196.003 -> 175.290 microseconds.
- 4 misses: 848.699 -> 606.602 microseconds.
- 16 misses: 2792.084 -> 1590.762 microseconds.

Raw CPU-enqueue/event/wall timings and p10/p90 are retained. Event intervals
include host submission gaps; they are not pure DMA time. Exact-byte, counter
and stable-address checks passed, as did a separate 96-expert / 576-copy batch
within 490.3 MB of host plus device payloads. This does not explain or rule out
a scheduling fault in the full model. A finite-check reproduction with optional
CPU KV disabled is the next diagnostic, not a performance measurement.

Tensor copies remain the default. Pool sizes and tensor addresses are
unchanged by selecting the native submission backend; HIP command metadata and
host descriptor allocations are not a measured zero-overhead claim.
