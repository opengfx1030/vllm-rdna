# V620 baseline before porting the fast service

> **Timing correction:** The first 16k trial below is invalid as a clean
> performance baseline. The regular-prose request overlapped Triton kernel
> compilation in the server logs, so its 1,029 tok/s prefill rate includes
> first-use compilation delay. Output validity does not validate timing.
> The corrected warmed measurements appear below.

The baseline source is upstream `rdna_extras` at `f3dd65fa70636566c11b249209d281c0b63c819e`.
That merge already contains PR #15's live-context QSA prefill bound, so this
port must not count QSA as a new change relative to this base.
Commit `9daf24d68949be29b9506ce640fd46367ca0b9cc` adds only the
benchmark launcher. No runtime source edits were present during this run.
The server cloned and pulled the branch from GitHub and built its own gfx1030
native modules from that checkout.

The server ran four V620 GPUs with TP4, EP4, MTP2, FP16, CPU PLE, a 4096-token
batch limit, and the launch options in `tools/rdna2/serve_v620_baseline.sh`.
The model was `<v620-home>/v620-vllm/models/intel-autoround`.
The active fast service was stopped and had zero running or waiting requests
before the baseline started.

Only `llm-context-bench` was used. The locked harness checkout was `92286b2`,
with SHA-256 `ec553141cd5f63d962ec1bbe6fbaf65effc44d032e5d055d4e6376824138e5ae`
reported by the result file. The command used both suites, the performance
lane, 16k inputs, one measured repetition per case, streaming, a 240-second
timeout, and disabled thinking. The harness performed its normal warmup.

| 16k suite | Prompt tokens | Prefill tokens/s | Generation tokens/s | TTFT s | Valid |
| --- | ---: | ---: | ---: | ---: | --- |
| Regular prose | 16,750 | 1,029.1 | 57.1 | 16.28 | Yes |
| Coding | 18,063 | 1,485.4 | 66.8 | 12.16 | Yes |

These are cold-start diagnostic observations, not performance conclusions. The
large prefill discrepancy between cases must not be used to rank changes. The full JSON
result remains on the V620 server at
`<v620-home>/v620-experiments/base-16k-f3dd65fa7.json`.

## Corrected 16k baseline

The server rebuilt the exact `9daf24d68` Git revision in a separate Git
worktree. One 16k pass warmed the prompt shapes and compiled Triton kernels.
The next pass accidentally reused a deterministic request tag and hit prefix
cache for regular prose (1.00-second TTFT). The harness did not report cached
tokens for this vLLM stream and therefore marked that trial valid. It must be
discarded. A further `llm-context-bench --repetitions 3` pass used fresh tags
`performance-02` and `performance-03`. The server logged no Triton inference
JIT during this final pass.

| 16k case | Prompt tokens | Fresh trial | Prefill tokens/s | Generation tokens/s | TTFT s | Output valid |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Regular prose | 16,750 | 02 | 1,411.5 | 55.5 | 11.87 | Yes |
| Regular prose | 16,750 | 03 | 1,424.0 | 54.8 | 11.76 | Yes |
| Coding | 18,063 | 01 | 1,478.1 | 64.7 | 12.22 | Yes |
| Coding | 18,063 | 02 | 1,474.7 | 65.3 | 12.25 | No: repeated-token output |
| Coding | 18,063 | 03 | 1,464.5 | 64.4 | 12.33 | Yes |

The clean prefill comparison is therefore about **1,411–1,424 tokens/s for
regular prose** and **1,464–1,478 tokens/s for coding**. Coding prompt length
is 7.8% greater. This is a small fixture-dependent gap, not the 44% gap in the
first cold-start trial. The coding trial with invalid output is excluded from
the valid measurement range. The full final JSON is on the server at
`<v620-home>/v620-experiments/base-16k-f3dd65fa7-fresh-repeats.json`.

## Resident MoE port on the same 16k suite

At source revision `fea652bcd` (later `9332972c5` changed this document only),
the launcher enabled `V620_ENABLE_RESIDENT=1`. The code bundle adds resident
W4A16 weights, FP16 shared-expert fusion, zero-preserving INT4 dequantization,
and a recurrent slot-bounds guard. It does **not** enable the eight-row MoE
prefill tile or grouped PLE normalization yet. A warm pass completed without
the earlier GPU fault. Three further repetitions logged no inference JIT or
GPU memory faults. Repetition 01 reused the earlier deterministic regular
prefix and gave an impossible 40,156 tokens/s; it is excluded even though the
harness marked it valid because this server did not report cached-token usage.

| 16k case | Fresh trial | Prefill tokens/s | Generation tokens/s | TTFT s | Output valid |
| --- | ---: | ---: | ---: | ---: | --- |
| Regular prose | 02 | 1,947.8 | 60.8 | 8.60 | Yes |
| Regular prose | 03 | 1,943.6 | 55.8 | 8.62 | Yes |
| Coding | 01 | 1,955.1 | 70.2 | 9.24 | Yes |
| Coding | 02 | 1,951.8 | 71.9 | 9.25 | Yes |
| Coding | 03 | 1,950.8 | 68.5 | 9.26 | Yes |

Against the exact-base fresh trials, the bundle improves 16k prefill by about
33–38%. This comparison does not isolate one of the bundled edits as the sole
cause. JSON:
`<v620-home>/v620-experiments/resident-16k-fea652bcd-fresh-repeats.json`.

## Qualified eight-row MoE prefill tile

Commit `85f22d518` selects tile8 only for FP16, at least 4,096 tokens, and
the served model's 2,560/640 hidden/intermediate size, 128/512 local/global
experts, 10 routed experts, and expected scale groups. Commit `178f107ad`
adds a GPU test comparing tile8 with the existing tile4 on this weight
geometry; it passed on V620. A startup log confirmed tile8 selection.
The first 16k pass compiled additional Triton shapes and was used as warmup.
The measured pass logged no inference JIT. As above, repeated tag 01 on the
regular fixture hit prefix cache and is excluded despite harness validity.

| 16k case | Fresh trial | Prefill tokens/s | Generation tokens/s | TTFT s | Output valid |
| --- | ---: | ---: | ---: | ---: | --- |
| Regular prose | 02 | 2,002.5 | 61.3 | 8.36 | Yes |
| Regular prose | 03 | 1,999.9 | 59.8 | 8.38 | Yes |
| Coding | 01 | 2,009.4 | 66.4 | 8.99 | Yes |
| Coding | 02 | 2,000.3 | 71.1 | 9.03 | Yes |
| Coding | 03 | 2,004.7 | 68.2 | 9.01 | Yes |

This is about 2.5–3% higher prefill than the preceding resident candidate in
these 16k runs. The decode trials vary and do not establish a decode gain.
JSON: `<v620-home>/v620-experiments/resident-tile8-16k-178f107ad-fresh-repeats.json`.

## Fused PLE grouped normalization

Commit `dbd999403` reuses the existing AMD grouped RMSNorm kernel for PLE
on supported FP16/BF16 contiguous GPU tensors. Eight focused CPU/GPU,
strided-input, and graph-replay tests passed on V620. One 16k pass warmed the
build; the measured pass logged no inference JIT or GPU memory faults. Again,
the first regular repetition reused a deterministic prefix and is excluded.

| 16k case | Fresh trial | Prefill tokens/s | Generation tokens/s | TTFT s | Output valid |
| --- | ---: | ---: | ---: | ---: | --- |
| Regular prose | 02 | 2,013.5 | 56.4 | 8.32 | Yes |
| Regular prose | 03 | 2,016.0 | 57.2 | 8.31 | Yes |
| Coding | 01 | 2,021.9 | 72.2 | 8.93 | Yes |
| Coding | 02 | 2,013.1 | 69.7 | 8.97 | Yes |
| Coding | 03 | 2,017.8 | 72.8 | 8.95 | Yes |

This is roughly 0.5–0.8% faster prefill than tile8 alone in these 16k trials.
Decode remains variable and no decode improvement is attributed to PLE.
JSON: `<v620-home>/v620-experiments/resident-tile8-ple-16k-dbd999403-fresh-repeats.json`.

## Served prefix checkpoint replay fixes

Commit `772dac40d` ports the served cache/replay changes for aligned Flash-Next
TP4 chunks and MTP resend/extension boundaries. Ten focused core tests passed.
The full-model 16k suite completed. The first repetition of both fixtures
reused cached prefixes and is excluded. The two later fresh trials measured:

| 16k case | Fresh trial | Prefill tokens/s | Generation tokens/s | TTFT s | Output valid |
| --- | ---: | ---: | ---: | ---: | --- |
| Regular prose | 02 | 1,950.2 | 54.8 | 8.59 | Yes |
| Regular prose | 03 | 1,949.1 | 56.7 | 8.59 | Yes |
| Coding | 02 | 1,969.0 | 73.6 | 9.17 | Yes |
| Coding | 03 | 1,967.4 | 71.9 | 9.18 | Yes |

Two Triton inference JIT events appeared early in the measured pass, during
the cached first repetition; these later fresh trials were stable. Compared
with the preceding PLE build, the checkpoint changes cost about 2.5–3.2%
prefill on these 16k fixtures. This step is for correct prompt reuse; the
served tuning table is evaluated separately below. JSON:
`<v620-home>/v620-experiments/checkpoints-16k-772dac40d-fresh-repeats.json`.

## Git-tracked served MoE tuning and launch settings

Commit `1e68a8b90` versions the served V620 MoE configuration and matches the
unit's `VLLM_ROCM_MOE_PREFILL=0` and `VLLM_GDN_HIP_PREFILL=0` settings. The
JSON content was compared with the live unit's tuning table by canonicalized
SHA-256 and matched exactly. The same 16k procedure gave these uncached trials:

| 16k case | Fresh trial | Prefill tokens/s | Generation tokens/s | TTFT s | Output valid |
| --- | ---: | ---: | ---: | ---: | --- |
| Regular prose | 02 | 1,951.7 | 56.2 | 8.58 | Yes |
| Regular prose | 03 | 1,950.8 | 56.0 | 8.59 | Yes |
| Coding | 02 | 1,971.5 | 70.6 | 9.16 | Yes |
| Coding | 03 | 1,970.5 | 73.1 | 9.17 | Yes |

The tuning table and launcher parity did not materially change 16k prefill
from the checkpoint-only build in this controlled run. The measured pass logged
two early Triton JIT events during cached repetition 01; later fresh trials
were stable. JSON:
`<v620-home>/v620-experiments/served-tuning-16k-1e68a8b90-fresh-repeats.json`.

## Served resident skinny decode

Commit `0821be8d7` ports the served native skinny decode kernel behind
`V620_ENABLE_SKINNY=1`. The six native reference and graph-replay tests passed
on V620. The full model logged that the skinny path was enabled, and the
fresh 16k pass had no GPU memory faults. Repetition 01 on each fixture reused
prefix cache and is excluded. The later fresh trials measured:

| 16k case | Fresh trial | Prefill tokens/s | Generation tokens/s | TTFT s | Output valid |
| --- | ---: | ---: | ---: | ---: | --- |
| Regular prose | 02 | 1,947.5 | 54.2 | 8.60 | Yes |
| Regular prose | 03 | 1,946.8 | 56.1 | 8.60 | Yes |
| Coding | 02 | 1,965.5 | 73.8 | 9.19 | Yes |
| Coding | 03 | 1,967.2 | 72.9 | 9.18 | Yes |

The prefill change from the tuned build is within a few tokens/s. Decode moved
down slightly on prose and up slightly on coding, so these full-model trials
do not establish a consistent decode benefit. The measured pass logged two
early Triton JIT events during cached repetition 01; later fresh trials were
stable. JSON:
`<v620-home>/v620-experiments/served-skinny-16k-0821be8d7-fresh-repeats.json`.

## Long-context check after served-source audit

Commit `5e4b5d580` restores the old service's one-shot RDNA2 GDN state
initialization guard. The native modules were rebuilt from that Git checkout,
and `llm-context-bench` then ran the regular and coding performance cases at
32k, 64k, and 128k with a unique request-tag scope. Each case requested 1,024
output tokens. The service reported a zero percent prefix-cache hit rate.

| Case | Prompt tokens | Prefill tokens/s | Generation tokens/s | TTFT s | Valid |
| --- | ---: | ---: | ---: | ---: | --- |
| Regular 32k | 33,475 | 1,977.1 | 58.6 | 16.93 | Yes |
| Coding 32k | 36,155 | 1,982.2 | 70.1 | 18.24 | Yes |
| Regular 64k | 66,933 | 1,918.7 | 56.9 | 34.89 | Yes |
| Coding 64k | 71,991 | 1,928.4 | 65.8 | 37.33 | Yes |
| Regular 128k | 133,836 | 1,778.8 | 55.7 | 75.24 | Yes |
| Coding 128k | 143,875 | 928.6 | 73.4 | 154.93 | Yes, after retry |

The coding 128k group took 410 seconds because its first request exceeded the
240-second client timeout. A separate coding 128k run with retries disabled
also timed out. Scheduler instrumentation showed that this was not a capacity
limit: at each 4,096-token prefill chunk, four Mamba managers allocated new
state blocks but did not retire their previous states across null gaps. Block
use therefore grew by eight per chunk instead of four. At 135,168 computed
tokens only three of 278 blocks remained, the next eight-block reservation
failed, and the only request was preempted and restarted from zero.

Commit `b619cf991` backports mainline vLLM commit `fadfe1c7d` (`[Bugfix][Core]
Retire Mamba states across null gaps (#55450)`) while retaining the served
branch's checkpoint/replay implementation. The same instrumented request then
crossed 135,168 computed tokens with 123 blocks free before allocation and 119
after it. The pinned 143,855-token coding prompt completed at 3.75 GiB with
1,832.6 prefill tokens/s and 70.2 generation tokens/s. Commits `5030766bc` and
`b88acd94d` remove the temporary scheduler instrumentation.

The clean 3.75 GiB Git service completed the pinned `llm-context-bench` matrix
on September 23, 2026. Retries were disabled. Coding 32k completed but was
excluded by the harness for a dominant repeated token, the same known fixture
behavior recorded by the historical combined run.

| Case | Prompt tokens | Prefill tokens/s | Generation tokens/s | TTFT s | Valid |
| --- | ---: | ---: | ---: | ---: | --- |
| Regular 32k | 33,455 | 1,977.2 | 67.5 | 16.92 | Yes |
| Coding 32k | n/a | n/a | n/a | n/a | No: repeated output |
| Regular 64k | 66,913 | 1,932.5 | 56.6 | 34.63 | Yes |
| Coding 64k | 71,971 | 1,928.0 | 69.8 | 37.33 | Yes |
| Regular 128k | 133,816 | 1,823.5 | 54.9 | 73.38 | Yes |
| Coding 128k | 143,855 | 1,813.9 | 78.2 | 79.31 | Yes |

An immediately preceding instrumented coding 128k run measured 1,832.6
prefill tokens/s and 70.2 generation tokens/s at 78.50 seconds TTFT. Clean
service startup took 352 seconds, compared with 373.879 seconds for the saved
combined build.

Raw server artifacts:

- `<v620-home>/v620-experiments/git-fast-5e4b5d580-32-128k-20260922.json`
- `<v620-home>/v620-experiments/git-fast-5e4b5d580-coding128k-repeat-20260922.json`
- `<v620-home>/v620-experiments/git-mamba-retire-375g-coding128-20260923.json`
- `<v620-home>/v620-experiments/git-final-mamba-retire-375g-32-128-20260923.json`
