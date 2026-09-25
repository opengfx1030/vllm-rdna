# V620 FP16 performance results — 2026-09-13

MTP2 is the fastest tested setting with four valid 16k/32k performance trials.
MTP0 with an 8k scheduled batch has the best measured prefill. Dense INT8 shadows
are disabled throughout this comparison. The existing Intel INT4 experts and
original BF16 PLE table in CPU RAM are unchanged.

This remains a draft candidate, not a replacement for the saved stable service.
All selected short single/four-request checks pass, but the separate long-context
quality suite passes only 2/4 cases. Performance validity checks do not establish
model accuracy. No configuration here establishes 90 tok/s sustained decode.

The tables in the initial campaign precede the conversation-cache correction.
See the recovery section below for the current cache-enabled deployment.

## Reused code and measured changes

- Reuse opengfx1030 PRs #6/#7/#8, selected PR #3 GDN tuning, and Leapdragon's
  Flash-Next/FP16 fusion implementation. The integration is based on
  `opengfx1030:rdna_extras` through `f86faadbd`, without mainline commit history.
- Enable the reused RDNA all-reduce after fixing the Torch device-context API
  that silently disabled it. Four-rank graph replay tests pass changing-input
  exact sums. The selected threshold is 64 KiB; larger transfers use RCCL.
- Reuse donor V620 MoE tiling. Alternative 4k/8k tile measurements did not beat
  the existing selection, so its configuration is retained.
- Qualify 28 FP16 rocBLAS solver rows against FP32 references on all four GPUs,
  with lookup-only execution and a library-hash guard. A 4k hyperconnection-down
  microbenchmark improved from 9.00 to 1.34 ms. Full-model code prefill in the
  campaign rose from approximately 1,195 to 1,470 tok/s; this includes intervening
  correctness changes and is not a single-variable numerical-parity comparison.
- Restore donor per-call `wvSplitK` output ownership. The target's shared output
  buffer allowed a later projection to overwrite retained results. The expanded
  native suite passes 230 cases, including FP16/BF16, eager/graph lifetime checks
  and the three/five-row shapes needed for higher MTP counts. No tolerances changed.
- Derive capture sizes from the speculative width: `(MTP + 1) * [1,2,4]`.
  MTP0, 1, 2, 3 and 4 were compared on the corrected FP16 base.

The optional general FP16 GEMV override did not improve MTP0 decode and remains
off. MTP1 with an 8k batch ran out of memory during PLE prefill; MTP-enabled
measurements use 4k. The separate mainline PLE gate fusion experiment reduced
temporary memory but did not meet its strict FP16 comparison. It was removed
from the candidate and retained only as an unqualified experiment. Dense INT8
experiments are excluded from the selected results and remain deferred.

## Benchmark method

Unmodified `llm-context-bench` at
`92286b24065565f4929e78c45f776029480e9939`, prose and code, one measured trial per
case, 1,024 requested output tokens after the runner's warmup. Thinking is off;
temperature 1, top-p 0.95, top-k 20, min-p 0, seed 3407. Input tolerance is
explicitly 11% for this tokenizer. All actual token counts are reported.

Prefill is prompt tokens / TTFT. Decode excludes the first output token's
latency. Early termination and repetition remain flagged in the raw results;
invalid trials are not retried until they pass. One trial per case does not
provide a confidence interval or prove a global optimum.

TP4/EP4, max context 262,144, four request slots, 4 GiB KV per GPU, full-decode
graphs, RDNA AR 64 KiB, qualified FP16 TunableOp lookup, hipBLASLt off, SDMA off.
Each V620 remains at its existing 180 W cap; clocks and power limits were not
changed. Configured capacity is distinct from a successful full-length request.

## MTP0 base versus selected MTP2

Every row in this table completed 1,024 output tokens and passed the runner's
performance checks. MTP0 uses batch 8,192; MTP2 uses batch 4,096.

| Workload | Prompt tokens | MTP0 prefill tok/s | MTP0 decode tok/s | MTP2 prefill tok/s | MTP2 TTFT s | MTP2 decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prose 16k | 16,750 | 1,418.95 | 42.21 | 1,364.47 | 12.28 | 56.94 |
| Prose 32k | 33,455 | 1,459.13 | 42.07 | 1,401.13 | 23.88 | 53.43 |
| Code 16k | 18,063 | 1,473.17 | 42.10 | 1,404.55 | 12.86 | 64.23 |
| Code 32k | 36,135 | 1,467.82 | 42.07 | 1,401.87 | 25.78 | 66.79 |

MTP2 code decode is 53–59% above this MTP0 base, while its smaller scheduled
batch costs roughly 4–5% prefill. Across these four measured requests, summed
TTFT plus decode duration is 168.78 s for MTP0, 147.16 s for MTP1 and 143.15 s
for MTP2. This comparison includes different generated text under speculative
sampling; it is not a fixed-output microbenchmark.

## MTP comparison

| Workload | MTP1 decode tok/s | MTP2 decode tok/s | MTP3 decode tok/s | MTP4 decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| Prose 16k | 53.92 | 56.94 | 85.36 | 58.52 |
| Prose 32k | 54.98 | 53.43 | 49.69 | 62.90 |
| Code 16k | 58.46 | 64.23 | 66.39 | Invalid: repetition |
| Code 32k | 58.65 | 66.79 | Invalid: repetition | 67.53 |

MTP3/MTP4 each have one invalid performance case and are not selected. Their
invalid observed rates (70.93 and 70.33 tok/s) are retained in raw data, not used
as speed achievements. Aggregate accepted draft-token fractions were 76.48%,
53.95%, 56.93% and 49.45% for counts 1–4; fractions across different widths are
not sufficient to choose throughput. All four MTP counts still pass only 2/4
long-context quality cases.

## Larger context

A fresh launch of the selected MTP2 configuration completed all four larger
trials with 1,024 output tokens each. All passed performance validity checks.

| Workload | Actual prompt tokens | Prefill tok/s | TTFT s | Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| Prose 64k | 66,913 | 1,376.67 | 48.60 | 54.67 |
| Prose 128k | 133,816 | 1,312.29 | 101.97 | 55.71 |
| Code 64k | 71,971 | 1,370.57 | 52.51 | 66.85 |
| Code 128k | 143,855 | 1,316.92 | 109.24 | 80.61 |

Code128k's 80.61 tok/s is this individual valid trial, not a general decode
guarantee. The larger tiers were performance tests; they do not override the
2/4 quality result at 16k/32k. Synthetic vision and a four-request answer check
after the long run passed. Video was not tested.

The engine reported KV capacity for 287,978 tokens, or 1.10 requests at the
configured 262,144-token limit. No 262k prompt was run; the longest measured
request contained 143,855 prompt tokens plus 1,024 generated tokens.

## Loading and memory

| Profile | Launch to healthy API s | Maximum worker loading s | Model-loading memory GiB/GPU |
| --- | ---: | ---: | ---: |
| MTP0 / batch 8192 | 252.27 | 96.63 | 19.18 |
| MTP1 / batch 4096 | 255.37 | 102.32 | 20.32 |
| MTP2 / batch 4096 | 259.36 | 112.50 | 20.45 |
| MTP3 / batch 4096 | 259.46 | 103.45 | 20.54 |
| MTP4 / batch 4096 | 269.77 | 120.09 | 20.67 |
| MTP2 / fresh larger-context run | 260.56 | 113.43 | 20.45 |

These are successive cached-filesystem starts, not controlled cold-storage
measurements. Worker loading is a stage of full startup, not an additional time
to add to readiness. Reported model-loading memory excludes the explicit 4 GiB
KV allocation and later graph/workspace allocations. The original CPU PLE table
is approximately 95.37 GiB. The explicit KV allocation bypasses startup profiling,
so enabling startup-plan caching would not skip another profiling stage here.

## Quality limitation and reproducibility

The long-context fixture expects `billing_units: 61`. Corrected MTP0 returns
59/59 for the code tiers; MTP1/2/4 return 54/59. The untuned corrected FP16 control
also returns 59/59. A separate step-by-step diagnostic computes the correct
54 + 7 = 61, but that does not repair the failed benchmark. Earlier apparent
quality passes under the shared-output kernel are not a safe reference. Packing,
weight sampling and isolated kernel checks do not establish end-to-end accuracy.

Model: `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`, revision
`4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897`, symmetric INT4/group128 experts.
PLE uses original BF16 checkpoint shard `model-00016-of-00017.safetensors`,
128 shards combined to `[320001536,160]`, in CPU RAM. No group16 sidecar was used.

See [candidate integration](V620-CANDIDATE.md) for reused commits, exact runtime
and CPU/native checks, and [FP16 tuning](../../tunableop/README.md) for qualified
solver rows. The campaign's raw benchmark JSON, generated output, metrics, logs,
source/native hashes, packages, source archive and replay helpers are preserved
under `/home/george/v620-vllm-testing/benchmarks/performance-2026-09-13`, with a
local copy under the workspace's `review-artifacts/performance-2026-09-13`.

The original installation and boot unit remain under `/home/george/v620-vllm`
and its saved release under `/home/george/v620-stable-releases`. This draft needs
human review and further model-quality investigation before stable promotion.
AI assistance was used for integration, testing and reporting.

## Follow-up latency correction (September 13)

The manual deployment initially recomputed entire conversations: identical
8,428-token prompts had zero cached tokens and 7.13 s TTFT; the short follow-up
had 10.05 s TTFT. Fresh-prompt benchmarks above did not cover this regression.

The correction enables aligned state checkpoints for Flash-Next/V2 at TP4,
uses the Mamba group's block size in the scheduler and worker resume path, and
retains both MTP replay boundaries. It adapts existing work from
[vLLM #54076](https://github.com/vllm-project/vllm/pull/54076),
[#53945](https://github.com/vllm-project/vllm/pull/53945),
[#54713](https://github.com/vllm-project/vllm/pull/54713), and
[#53798](https://github.com/vllm-project/vllm/pull/53798), with the aligned-prompt
replay stop described in [#50409](https://github.com/vllm-project/vllm/pull/50409).
The other-model TP>2 workaround remains. This is an integration/backport rather
than a competing upstream implementation. We did not import unconditional
splitting at every state boundary or the optional finer-grained MTP feature.

| Synthetic latency case | TTFT after | Cached / prompt tokens |
| --- | ---: | ---: |
| 8k identical resend | 1.718 s | 7,200 / 8,428 |
| 8k follow-up after identical resend | 1.679 s | 8,000 / 8,448 |
| Direct 8k follow-up, without resend | 1.649 s | 7,200 / 8,448 |
| 16k follow-up after identical resend | 0.306 s | 16,800 / 16,848 |
| 32k follow-up after identical resend | 0.380 s | 33,600 / 33,648 |
| Direct 32k follow-up, without resend | 4.486 s | 32,800 / 33,648 |

Each row is one bounded deterministic request, max output 16 tokens, measured
from the LAN stream's first content token. The subsecond rows benefit from the
preceding resend warming an additional boundary; they are not a general
follow-up latency guarantee. Uncached 32k TTFT varied from 29.76 to 41.35 s in
these synthetic trials; further prefill profiling remains necessary. These
latency probes do not replace the context-benchmark figures or quality limits.

192 CPU tests pass, three GPU-only cases skipped locally. Prose/code outputs
match exactly between cold and cached runs, and two concurrent chats returned
their separate requested words on the four V620s. Restore the original scheduler,
retention code, or worker divisor to reproduce the corresponding test failures.
The final service launch took 252.57 s to health; maximum worker loading took
97.55 s. Model precision, native kernels and the FP16/MTP2 settings are unchanged.

Reproduce on an otherwise idle endpoint with
[check_prefix_reuse.py](../../tools/rdna2/check_prefix_reuse.py):

```sh
.venv/bin/python tools/rdna2/check_prefix_reuse.py \
  --base-url http://127.0.0.1:8080 --skip-identical --lines 400 \
  --output /tmp/prefix-check
```

Omit `--skip-identical` for the three-request sequence; use `--lines 800` or
`1600` for approximately 16k/32k. The helper uses a fresh cache salt, saves raw
metrics/results, and checks output, cache reuse and TTFT. Artifacts and the
pre-fix Python rollback archives are saved in the testing workspace under
`followup-latency-2026-09-13`. The enhanced manual service remains on port 8080;
the original saved build and boot unit are preserved.

## Cache-enabled prefill recovery (September 13)

The cache correction changed ordinary prefill chunks from 4,096 to 4,000 tokens
on the automatic 800-token state grid. The original 28 exact-shape FP16 solver
rows did not cover these shapes. Fresh uncached 16k probes reproduced the
regression at 1,145 and 1,142 tok/s.

Reuse the existing donor TunableOp tooling, adding 42 rows while retaining the
original 28 unchanged. All 70 rows pass independent FP32 comparisons on all four
V620s. For example, the 4,000-token router improved from 10.063 to 1.244 ms;
the previously uncovered 3,072-token router improved from 8.579 to 0.859 ms.
Set `--block-size 1024 --max-num-batched-tokens 4096` to preserve ordinary
4,096-token chunks while retaining conversation caching. The 3,072-token rows
also cover intermediate stops on this grid. No native kernels, weight precision,
MTP count or all-reduce settings changed during this recovery.

The following are raw timings from the final configuration, one trial per case,
using the same unmodified benchmark revision and locked sampling described
above. A loopback streaming adapter adds a unique `cache_salt` per request;
server metric deltas confirm zero prefix-cache hits during the benchmark runs.
Every request produced the requested 1,024 output tokens.

| Workload | Actual prompt tokens | Prefill tok/s | Decode tok/s | TTFT s | Harness valid |
| --- | ---: | ---: | ---: | ---: | --- |
| Prose 16k | 16,750 | 1,369.83 | 59.23 | 12.23 | yes |
| Prose 32k | 33,455 | 1,390.23 | 55.71 | 24.06 | yes |
| Code 16k | 18,063 | 1,387.70 | 66.12 | 13.02 | yes |
| Code 32k | 36,135 | 1,391.18 | 67.94 | 25.97 | no: repeated-token flag |
| Prose 64k | 66,913 | 1,378.53 | 60.21 | 48.54 | yes |
| Prose 128k | 133,816 | 1,318.74 | 53.69 | 101.47 | yes |
| Code 64k | 71,971 | 1,372.64 | 65.95 | 52.43 | yes |
| Code 128k | 143,855 | 1,288.27 | 66.24 | 111.67 | yes |

A single fixed repeat of code 32k/128k measured 1,402.00/1,301.56 tok/s prefill
and 73.85/67.82 tok/s decode. Both code 32k trials triggered the unchanged
`dominant_repeated_token` check: generated TypeScript contains long hyphen
separator lines. Their raw timings are retained but excluded from valid
performance aggregates. The code 128k repeat passed. No retry-until-pass or
scoring changes were used. The earlier 80.61 tok/s code 128k decode result was
not reproduced: the two current trials measured 66.24 and 67.82 tok/s.

Final validation: eight arithmetic requests at concurrency four and two vision
requests pass; deterministic prose/code cold and cached outputs match exactly;
two concurrent chats retain separate requested words. A direct 16k follow-up
reused 15,360 tokens and started in 1.660 s, versus 11.988 s for its initial
16,828-token request. Launch to healthy API took 257.945 s with existing
filesystem caches. These checks do not resolve the earlier 2/4 long-context
quality result or establish BF16-equivalent accuracy.

Tuning tables and provenance are versioned with the launcher. Normal launches
warn about incomplete known-shape coverage and continue using default FP16
algorithms for missing shapes. Missing or incompatible library tables disable
TunableOp with a warning. `check_v620_tuning.py --strict` is an optional release
check, not a normal-service availability requirement. Four CPU regression tests
cover valid tables, missing shapes, final CLI overrides and incompatible-library
fallback. New runner shapes, arbitrary tails and mixed batches still require
benchmarking; passing static coverage does not guarantee optimal performance.

Reproduction artifacts, exact launch arguments, per-rank tuning hashes, raw
benchmark JSON, metrics and GPU telemetry are saved under
`/home/george/v620-vllm-testing/cache-prefill-recovery-2026-09-13` and the local
workspace's matching `review-artifacts/cache-prefill-recovery-2026-09-13`.
The current manual service uses the isolated testing installation on port 8080,
with model alias `active`, MTP2, FP16 dense weights and CPU BF16 PLE. The original
stable installation and boot unit remain preserved; this manual deployment
does not change which service starts at reboot.
