# Concurrent MTP decode: sequential W4A16 MoE

## Scope

Based on `rdna_extras` at `cd38a1d38`, including the merged RAM KV-offload
fixes from PR #24. Neither PR #22 nor PR #25 is incorporated. The experiment
uses RCCL, so existing all-reduce issues cannot confound this comparison.

With MTP2, three requests require nine target token rows. The existing decode
graph ladder `[3, 6, 12]` pads that batch to twelve. Sequential W4A16 MoE
dispatch currently caps HIP skinny decode at eight rows, even though the
native kernel accepts up to sixteen. The larger batch therefore uses tile
Triton. This change adds `VLLM_ROCM_MOE_SKINNY_MAX_M=16` as an opt-in while
retaining the default eight-row cap and a hard sixteen-row native bound.
It does not change arithmetic, packing, weights, precision, or large prefill
dispatch. Shuffled RDNA2 weights remain a separate, ineligible layout.

## Regression and numerical validation

All changes were committed/pushed locally and pulled into the inactive server
experiment checkout; no server-side source edits were made. The native
extension was incrementally rebuilt there. The active/default installation
and its service configuration were not modified.

```bash
python -m pytest --noconftest \
  tests/kernels/quantization/test_rocm_moe_skinny.py -q
python benchmarks/kernels/benchmark_rocm_moe_skinny.py --device 0
python benchmarks/kernels/benchmark_rocm_moe_skinny.py --device 1 --rows 9 12 16
python benchmarks/kernels/benchmark_rocm_moe_skinny.py --device 2 --rows 9 12 16
python benchmarks/kernels/benchmark_rocm_moe_skinny.py --device 3 --rows 9 12 16
```

Commands used the existing ROCm virtual environment with the experimental
checkout on `PYTHONPATH`. Local collection was blocked by a missing `regex`
dependency; this is not reported as a local test pass.

- Before dispatch implementation: **7 failed, 14 passed** in the new focused
  opt-in/bounds tests. Failures were exactly the rejected 9/12/16-row cases.
- After implementation: **57 passed**, including 24 GPU parameter cases
  covering M=1/3/6/9/12/16, two scale ranges, expert mapping/nonlocal rows,
  FP32 dequantized references, and exact eager/graph agreement with changing
  inputs. Existing numerical tolerances were retained, not relaxed.
- The benchmark validated 30 model-shaped cases across four GPUs against an
  FP32 reference and changing-input graph replay for each implementation.
- Commit hooks, Python lint/format, and shellcheck passed.

## Isolated kernel measurements

Four Radeon PRO V620 GPUs, gfx1030. Synthetic sequential symmetric W4A16
weights match the Intel checkpoint's per-rank dimensions: hidden 2560,
intermediate 640, 128 local / 512 global experts, top-k 10, group size 128.
These are not checkpoint-weight quality evaluations. Both implementations
use graph replay. Six timing samples of 100 replays alternate measurement
order; the table reports medians in microseconds per MoE operation.

| GPU | Rows | EP4 HIP | EP4 Triton | Ratio | All-local HIP | All-local Triton |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 12 | 257.1 | 1029.8 | 4.01x | 803.7 | 2575.4 |
| 1 | 12 | 267.3 | 1090.3 | 4.08x | 807.7 | 2508.6 |
| 2 | 12 | 258.6 | 1050.3 | 4.06x | 779.6 | 2417.8 |
| 3 | 12 | 264.3 | 1070.0 | 4.05x | 807.6 | 2477.0 |

EP4 routing includes a forced all-nonlocal final row to check graph padding;
the one-row EP4 case is consequently padding-only, not a single-chat speed
estimate. All-local routing stresses weight traffic rather than representing
balanced EP4. Reference/graph validation gates every reported timing.

This establishes an isolated kernel opportunity, **not a 4x model speedup**.
The real model uses the modular expert caller; full-model A/B is necessary
to account for its workspace use, shared experts, other layers, communication,
and speculative acceptance. Default dispatch remains unchanged; model results
for the opt-in follow below.

Raw server evidence:
`<v620-home>/v620-experiments/moe-decode-results-20260927/` contains
`build.log`, `dispatch-before.log`, `correctness.log`,
`correctness-after.log`, and `latency-gpu0.log` through `latency-gpu3.log`.

## Full-model qualification

The model A/B uses source `f22fee026`, the same experimental native binary
(`a202db733d6a887a580b3d12cc09f28194435c73079747598b0607f584150507`),
Intel AutoRound, TP4/EP4, MTP2, FULL_DECODE_ONLY `[3, 6, 12]`, RCCL,
and 64 GiB RAM KV offload. Only `VLLM_ROCM_MOE_SKINNY_MAX_M` changes.
All four candidate workers logged HIP skinny dispatch at M=12 during capture.

The unmodified `llm-context-bench` 0.4.0 runner uses three simultaneous
requests, 8K tiers, 1,024 output tokens, two measured repetitions, 12% input
size tolerance, and identical request-tag scope
`moe-concurrent-controlled-20260927` across independent service starts.
The configured sampling/seed is the benchmark's unchanged preset.

Both coding runs completed all six requests, but each had three
`dominant_repeated_token` validation failures and zero valid aggregate groups.
Keep these failures: they do not support a coding-throughput improvement claim.

Both regular-text runs passed all six performance-validity checks, with two
valid aggregate groups per setting. Actual prompt length was 8,440 tokens.

| Metric | Default cap 8 | Opt-in cap 16 | Observed change |
| --- | ---: | ---: | ---: |
| Aggregate generation tokens/s | 46.66 | 51.89 | +11.2% |
| Per-chat generation tokens/s | 18.11 | 20.05 | +10.8% |
| Mean inter-token latency, ms | 55.23 | 49.88 | -9.7% |
| Median TTFT, seconds | 15.76 | 15.69 | approximately unchanged |

Control group rates were 46.93 and 46.39 aggregate tokens/s; candidate group
rates were 52.61 and 51.18. Aggregate generation uses the benchmark's shared
generation window, which includes periods when some requests are still
prefilling. It is not three times one stream's rate.

Counter deltas over each regular benchmark, including its warm-up, show
3,479/6,346 accepted draft tokens (54.82%) for cap 8 versus 3,378/6,544
(51.62%) for cap 16. Better speculative acceptance does not explain the gain;
the candidate accepted less. Outputs/routing can still differ, and two
repetitions do not establish a universal percentage or a confidence interval.
This is an observed concurrent-decode improvement, not a 4x model speedup or
a single-chat improvement. Large prefill batches retain their existing path.
These are performance checks, not code-quality or long-context semantic
evaluations.

Both settings passed the three concurrent synthetic tool-call sessions and
their staggered returns, with exact tool arguments and final answers. The
existing committed helper was executed from
`f03b318a0:tools/rdna2/qualify_tool_turns.py`; its originating all-reduce changes
were not merged into this branch.

The first regular-text control startup was refused by the overlap guard before
inference. Automatic recovery started the unchanged default service; it was
then stopped, all inference processes were verified absent, and the isolated
control was retried without bypassing the guard. Both the refused-start log
and the retry log are retained. A blocked startup is not a model benchmark.

Model evidence: `bench-cap8-c3.json`, `bench-cap16-coding-c3.json`,
`bench-cap16-regular-c3.json`, `bench-cap8-regular-c3.json`,
`tools-cap8.json`, `tools-cap16.json`,
the corresponding service logs, and before/after metrics in the same result
directory. No full-VRAM long-session or multimodal stress test is claimed for
this dispatch change. It remains opt-in; production promotion is separate.

## Final service state

After testing, the experimental services were stopped and the unchanged
`v620-tp4-git.service` was restored. At 19:22 UTC its health check and a
deterministic `2 + 2` completion passed. The API reported Intel AutoRound at
`<v620-home>/v620-vllm/models/intel-autoround` with max length 262,144; the
existing 64 GiB RAM-offload configuration and enabled boot service remain.
The default runs its original KV-offload checkout, not this performance branch.

Restoration recorded one automatic retry: the first start was refused by the
overlap guard while an exiting process was still visible; ten seconds later
the retry loaded successfully. PID 258139 then reached health without further
restarts. `restored-models.json` and `restored-smoke.json` retain the API evidence.
