# Bounded expert-prefetch replay

## Method

This is causal CPU replay of captured routing decisions, not a deployed predictor or an ITL benchmark. Each ordinary replay reproduces the native demand policy exactly. Predictions never replace the learned router or omit demanded experts.

All modes use the same 115 real slots, 64 warmup calls per layer, and at most two predicted experts per eligible target call. Traces cover four ranks and ten selected layers (0, 1, 12, 13, 24, 25, 36, 37, 46, 47). Adjacent-layer transitions use only the recorded physical pairs. Training is online from the preceding prefix, not independently held-out task evaluation.

Fills are optimistically instantaneous; the replay models eviction pollution and transfer bytes, but not copy/compute contention, demand-load duration or predictor execution cost. Diagnostic timings are excluded from the performance comparison.

## Adjacent-layer top-two results

Useful/issued counts speculative loads subsequently demanded before eviction. Avoided is the net reduction in demand loads, including pollution elsewhere. Positive extra bytes mean more total demand-plus-speculative traffic than ordinary replay. Counts are distinct expert demands in bounded cache passes, not routed-token accuracy.

| Corpus / arm | Workload | Actual output/request | Baseline misses | Issued | Useful | Net avoided | Extra MiB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| random D | 1024-c8 | 512 | 631 | 0 | 0 | 0 | 0.00 |
| random D | 16384-c1 | 512 | 84 | 0 | 0 | 0 | 0.00 |
| random Z | 1024-c8 | 512 | 837 | 0 | 0 | 0 | 0.00 |
| random Z | 16384-c1 | 512 | 44 | 0 | 0 | 0 | 0.00 |
| coding 256 | 1024-c8 | 256 | 186 | 0 | 0 | 0 | 0.00 |
| coding 256 | 16384-c1 | 256 | 6433 | 7 | 5 | 4 | 7.31 |
| coding 512 | 1024-c8 | 512 | 682 | 0 | 0 | 0 | 0.00 |
| coding 512 | 16384-c1 | 512 | 6907 | 6 | 6 | 6 | 0.00 |

The corrected long coding trace avoided six of 6,907 demand loads (0.087%); all six issued loads were useful and total transfer bytes were unchanged. That is six observations, not evidence of general 100% prediction precision. Most candidates were already resident. No demand-load time or runtime ITL reduction was measured.

## Other policies

- Previous-step top-two issued nothing in the random-input traces or either short coding trace. On long coding tasks it avoided one load with 42 speculative loads at 256 output tokens, and three with 35 speculative loads at 512; the corrected trace added 77.93 MiB.
- Forcing predictions to be nonresident increased demand misses in every trace. On corrected coding tasks it added 113 misses and 25,310.10 MiB at 1k/c8, and 266 misses and 73,448.73 MiB on long/c1. A per-call budget alone does not prevent cumulative traffic or cache pollution.

## Scope and receipts

Random-input traces use source `0577aafba2bc09c074a2d2cc1721a9f232412bbe`; legacy coding traces use `20d601c98679e7972c7168fe6b56ed44435af348`; corrected coding traces use `ca5b72d89070b6d7a9076b943a8a78d853791204`.

The first custom-data run requested 512 but the benchmark default produced 256. It is retained and labelled honestly; the driver now sets `--custom-output-len` explicitly and rejects unexpected output lengths. All twelve corrected coding requests completed with exactly 512 output tokens. Nominal 16k prompts actually range from 15,917 to 19,580 tokens; short prompts range from 971 to 1,083.

[All policy counts and source links](prefetch-summary.json), [legacy manifest and corpus](semantic-r2-manifest.json), [corrected manifest](semantic512-r3-manifest.json). Compressed `stop.json.gz` files preserve the original trace bytes.

These cheap expert-ID-history predictors do not justify a runtime prefetch implementation on the sampled 90%-resident configuration. This does not rule out activation-based lookahead, other workloads or smaller resident caches. No production default was changed.
