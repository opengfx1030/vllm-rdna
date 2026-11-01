# RDNA tuning guide: fork knobs, modes, TunableOp

What the fork adds on top of upstream vLLM for RDNA2 (gfx1030, primary) and
RDNA3 (gfx11, secondary), and what to turn on or off when tuning a serve.
Defaults below are the code defaults. A recipe may override them, so
`PRINT=1 bash tools/rdna/serve_rdna.sh RECIPE=<name>` shows what a launch
actually sets. The human walkthrough of the launcher and every recipe is
[`docs/rdna/serving.md`](../rdna/serving.md).

## Ground rules

- **Start from a recipe** (`tools/rdna/recipes/*.env`). Change one knob at a
  time with `KEY=value` on the launcher command line. Precedence:
  recipe < caller environment < trailing `KEY=value`.
- **Prefer CLI/config flags over env vars.** Config flags propagate to the
  engine-core and worker processes. Env-only knobs need checking: confirm in
  the serve log that the path took effect (each RDNA path logs when it engages).
- **Production is `FULL_AND_PIECEWISE` cudagraphs at full prefill and decode
  speed.** Eager (`EAGER=1`) must stay correct but is not a tuning target.
- **A/B before keeping a change**: run `tools/rdna/port_v031/serve_validate.sh`
  on both arms. Report prefill tok/s, decode tok/s per request, TTFT and ITL
  for 1k/512 and 16k/1k at c=1 and c=8. Greedy probe, perplexity, prefix and
  concurrency gates must pass on both arms.
- **RDNA3**: most HIP kernels are gated on `on_gfx10x()` and fall back to the
  upstream/Triton path on gfx11. Nothing here is validated on RDNA3 yet, and
  the shipped TunableOp rows are for one gfx1030 rocBLAS build only.

## Fork CLI / config parameters

| Flag | Default | What it does |
| --- | --- | --- |
| `--decode-stall-budget-ms` | None (RDNA: `VLLM_RDNA_DECODE_STALL_MS`) | Decode-stall cap: bounds prefill tokens in steps that carry decodes, so a long prefill cannot freeze decoders for seconds. 0 disables. |
| `--decode-time-share` | 0.0 | With the cap, the minimum share of wall time kept for pure-decode steps while a prefill runs. |
| `--prefill-schedule-interval N` | 1 (off) | Admit prefills only every Nth step, so decoders get pure-decode steps (and MTP drafts). Upstream only applied this to data-parallel; the fork applies it to a single engine too. |
| `--linear-backend rdna2\|rdna_hybrid` | auto | Force the W4A16 linear kernel: `rdna2` = HIP decode+prefill; `rdna_hybrid` = skinny HIP + Triton (gfx10 dequant + rocBLAS at M ≥ 256). |
| `--attention-backend RDNA_ATTN` | auto | FA-RDNA2 attention (gfx10x). The launcher's `ATTN=fa` sets it. |
| `--speculative-config '{"method":"mtp","num_speculative_tokens":2,"use_local_argmax_reduction":true}'` | — | MTP=2. `use_local_argmax_reduction` avoids the drafter's vocab all-gather (required on TP=4). |
| `--monokernel` | off | Qwen4Exp (Flash-Next) decode on the RDNA2 MonoKernel: K1 (HC preamble, <= 24 rows), K2 (FFN block, <= 8 rows), fused GDN core, QSA decode, all on. `VLLM_ROCM_MONO_DECODE=1` is an alias (read in the API server). Knobs: `--additional-config '{"rocm_mono_decode":{"k2_max_rows":8,"gdn":true,...}}'`, only read with the flag. Needs `GPU_MAX_HW_QUEUES=4` (the Flash-Next recipes set it). Unsupported deployments refuse at startup (`--monokernel <reason>.`). See `vllm/models/qwen4_exp/amd/mono/rdna2/README.md`. The mono fused QSA sparse decode wins over the generic HIP QSA kernel (both HIP); its fused store and norm+RoPE are Triton, so the store stands down while `VLLM_RDNA_QSA_HIP=1` (the three HIP `qsa_store_cache_rows` run instead) and a provably zero-Triton run omits `--monokernel`. |

## Launcher knobs (`tools/rdna/serve_rdna.sh`)

| Knob | Values | Effect |
| --- | --- | --- |
| `MTP` | 0, 2, 3 | Speculative tokens. Capture ladder becomes `(MTP+1)×{1,2,4,8}`. Keep powers of two (`CG_SIZES` overrides); 9/15/18/21 GPU-fault the 16k cell. |
| `SPEC_CONFIG` | JSON | Replace the MTP speculative config while retaining its capture ladder. `flashnext-dflash2` sets `MTP=3` for four verification rows and supplies the DFlash2 config here. |
| `ATTN` | fa, triton, none | FA-RDNA2 (`RDNA_ATTN` + GQA decode) / Triton AMD FA / leave to vLLM. |
| `CG_MODE`, `COMPILE_MODE`, `CG_EXTRA`, `EAGER` | | Cudagraph mode, compile mode, extra compilation JSON, eager fallback. |
| `PLE` | bf16, int4, int4-disk | Flash-Next PLE table, see [PLE modes](#ple-table-flash-next). Needs `PLE_QUANT_DIR` for int4*. |
| `W4A8` | 0, 1 | → `VLLM_RDNA2_W4A8_SDOT4`. |
| `DFLASH_TOKENS` | 3, 2 | Flash-Next DFlash2 draft depth, set in the caller environment: 3 favors 1k/512 c=8; 2 favors 16k/1k c=8 and selects the 24-row verification capture ladder. |
| `RDNA_AR` | 0, 1 | → `VLLM_RDNA_AR` when the recipe leaves it unset (the flashnext/27B recipes set it to 1; override with `VLLM_RDNA_AR=0`). |
| `P2P` | 1, 0 | `0` = no PCIe peer-to-peer: `VLLM_RDNA_AR=0` (overrides the recipe) + `NCCL_P2P_DISABLE=1` (RCCL over host shared memory). For hosts where P2P is broken, and to bisect wrong output. Slower decode. |
| `TUNABLEOP`, `TUNABLEOP_PROFILE` | 1 / name | TunableOp lookup on/off, force a named profile. |
| `TUNABLEOP_ROOT` | directory | Optional alternate profile registry and rank files for a serving A/B; defaults to the fork's `tunableop/`. The same rocBLAS hash validation applies. |
| `TUNED_CONFIG` | 0, 1 | Use `tuned-moe/` MoE configs (`VLLM_TUNED_CONFIG_FOLDER`). |
| `SEQS`, `MAXBAT`, `MAXLEN`, `KV`, `GMEM`, `BLOCK_SIZE`, `LPTH`, `PREFILL_INTERVAL` | | `--max-num-seqs`, `--max-num-batched-tokens`, `--max-model-len`, `--kv-cache-memory-bytes`, `--gpu-memory-utilization`, `--block-size`, `--long-prefill-token-threshold`, `--prefill-schedule-interval`. `KV=auto` (needs `GMEM`) drops `--kv-cache-memory-bytes` so vLLM sizes the cache after profiling. An empty `MAXBAT` is derived from the KV and SSM dtypes (2048 for an FP16 KV cache, 4096 for a packed KV cache with an FP16 SSM state, 6144 with the model's own FP32 state): a prefill chunk ends on a mamba state block boundary, so the budget must cover two blocks or a long prompt drains at half rate at concurrency. The other recipes pin `MAXBAT=2048`, which is right for their FP16 KV cache; a recipe that switches to a packed KV cache has to let the launcher derive the budget instead. |
| `PP`, `PP_PARTITION` | | `--pipeline-parallel-size`; layers per stage → `VLLM_PP_LAYER_PARTITION`. See [Pipeline parallel](#pipeline-parallel-3-or-4-cards). |
| `ALLOC_CONF` | true, false, auto | `expandable_segments`. `auto` turns it off with the PLE sidecar. |
| `FEATURES` | tokens | `prefix-caching mamba-align expert-parallel language-model-only skip-mm-profiling vision-cap trust-remote-code reasoning auto-tool prompt-tokens-details`. |
| `HIP_VISIBLE_DEVICES` / `GPUIDS` | | GPUs. Keep a TP group on one PCIe switch. |

## Modes by subsystem

### Flash-Next with DFlash2

`flashnext-dflash2` serves the Flash-Next AWQ-W4A16 **target** with
`incoai/Qwen3.8-Flash-Next-DFlash2` as an external draft model. The draft is
not a standalone language model. The recipe defaults to three draft tokens
and accepts `DFLASH_TOKENS=2` for the faster long-context c=8 configuration.
It uses the V2 runner, TP4/EP4, text-only inputs, and `--monokernel`. Flash-Next's QSA
attention uses the native HIP sparse decode and indexer paths under MonoKernel;
the BF16 non-causal DFlash2 draft uses FA-RDNA2's GQA prefill kernel with a
symmetric 2048-token sliding window. Its single-token steps also use that
kernel because the split-K decode path remains FP16 and causal. The target
also uses HIP GDN prefill and RDNA2 W4A16 MoE. FA-RDNA2 accepts block-major
`BLHNC` pages through its stride-aware HIP kernels and cache writer. The
target's QSA attention does not call FA-RDNA2.
The target remains FP16 on RDNA2; the draft stays BF16 because its residuals
overflow in FP16 and produce NaN candidates. BF16 draft normalization uses
PyTorch's RMSNorm since the RDNA AOT RMSNorm kernel accepts FP16 only.
On 4× V620 with the AWQ target, the 2026-10-20 graph validation passed.
The initial exact-hash rocBLAS profile added 58 BF16 DFlash2 draft GEMM rows;
the four strided-batched BF16 shapes remain on rocBLAS defaults. The final
tuned run passed greedy, perplexity, prefix, and concurrency probes, with no
GPU faults. It measured 65.91/223.81 output tok/s at 1k/512 c=1/c=8 and
49.00/46.38 at 16k/1k c=1/c=8. Repeated 1k c=8 tests improved from
153.92–168.96 untuned to 223.81–229.95 tuned. The 16k c=8 runs ranged
37.98–49.23 untuned and 39.54–46.38 tuned, so a long-context gain is not
established. These figures used the earlier Triton draft route. A controlled
same-GPU comparison on the new FA-RDNA2 route (32 requests per c=8 cell,
temperature 0, HIP 6–9) measured:

| Draft attention | 1k in / 256 out c=8 | 16k in / 512 out c=8 |
| --- | ---: | ---: |
| Triton | 182.94 tok/s | 31.24 tok/s |
| FA-RDNA2, two-head BF16 tile | 170.41 tok/s | 36.42 tok/s |
| FA-RDNA2, four-head BF16 tile (recipe) | 178.19 tok/s | 33.31 tok/s |

The c=8 benchmark summaries also report per-request and aggregate rates,
mean time to first token (TTFT), and median/mean inter-token latency (ITL):

| Draft attention | Input → output | Prefill tok/s/req (agg estimate) | Decode tok/s/req (agg output) | Mean TTFT | ITL median / mean |
| --- | --- | ---: | ---: | ---: | ---: |
| Triton | 1,024 → 256 | 945.1 (2164.0) | 26.79 (182.94) | 1.08 s | 64.7 / 89.2 ms |
| FA-RDNA2 four-head (recipe) | 1,024 → 256 | 864.0 (2131.6) | 25.80 (178.19) | 1.19 s | 68.7 / 92.9 ms |
| Triton | 16,384 → 512 | 591.3 (1449.3) | 5.05 (31.24) | 27.71 s | 552.8 / 423.6 ms |
| FA-RDNA2 four-head (recipe) | 16,384 → 512 | 512.2 (1313.7) | 5.72 (33.31) | 31.99 s | 554.8 / 418.6 ms |

Each cell completed 32 requests. Per-request prefill is mean input tokens
divided by mean TTFT; aggregate prefill estimates one eight-request wave from
its p99 TTFT. Per-request decode is the inverse of mean time per output token;
aggregate decode is measured output throughput over the full run. The two
prefill figures are estimates based on end-to-end latency, not isolated GPU
prefill timings.

These 256/512-output cells are draft-attention diagnostics, not the standard
1k/512 and 16k/1k workload used by the merged Flash-Next PRs. Their
speculative acceptance rates for the four-head tile were 46.85% and 46.65%;
all 64 benchmark requests completed without failure. Greedy, perplexity
(4.80), prefix, concurrency, and after-load probes passed, with no GPU faults.

The three-token FA-RDNA2 recipe also completed the standard c=1 cells on
HIP 6–9 (eight requests per cell, temperature 0):

| Input → output | Prefill tok/s/req (agg estimate) | Decode tok/s/req (agg output) | Mean TTFT | ITL median / mean |
| --- | ---: | ---: | ---: | ---: |
| 1,024 → 512 | 2208.9 (2133.5) | 71.59 (67.35) | 0.464 s | 32.5 / 32.6 ms |
| 16,384 → 1,024 | 2056.2 (2045.5) | 73.66 (46.85) | 7.97 s | 34.3 / 34.3 ms |

The standard-length, same-GPU c=8 comparison (32 requests per cell,
temperature 0, HIP 6–9) is:

| Speculative path | Input → output | Prefill tok/s/req (agg estimate) | Decode tok/s/req (agg output) | Mean TTFT | ITL median / mean |
| --- | --- | ---: | ---: | ---: | ---: |
| DFlash2, Triton | 1,024 → 512 | 965.7 (2114.3) | 32.30 (227.55) | 1.06 s | 65.4 / 78.5 ms |
| DFlash2, FA-RDNA2 four-head | 1,024 → 512 | 940.2 (2130.8) | 31.33 (222.47) | 1.09 s | 69.1 / 82.2 ms |
| DFlash2, FA-RDNA2 three tokens, interval 4 | 1,024 → 512 | 833.3 (2097.9) | 30.47 (220.24) | 1.23 s | 69.5 / 81.9 ms |
| DFlash2, FA-RDNA2 two draft tokens | 1,024 → 512 | 915.9 (2086.0) | 23.88 (163.91) | 1.12 s | 82.3 / 92.4 ms |
| Native MTP2, current branch (BF16 PLE) | 1,024 → 512 | 1045.3 (2206.8) | 32.86 (226.63) | 0.98 s | 53.7 / 63.3 ms |
| Native MTP2, merged PR #94 (INT4 PLE) | 1,024 → 512 | 1003.2 (2254.8) | 32.54 (228.01) | 1.02 s | 55.9 / 65.7 ms |
| DFlash2, Triton | 16,384 → 1,024 | 424.4 (933.1) | 6.17 (39.31) | 38.61 s | 551.0 / 394.3 ms |
| DFlash2, FA-RDNA2 four-head | 16,384 → 1,024 | 369.7 (760.0) | 5.46 (34.98) | 44.32 s | 553.2 / 404.2 ms |
| DFlash2, FA-RDNA2 three tokens, interval 4 | 16,384 → 1,024 | 624.4 (1525.8) | 11.84 (70.24) | 26.24 s | 71.5 / 202.0 ms |
| DFlash2, FA-RDNA2 two draft tokens | 16,384 → 1,024 | 895.8 (1963.5) | 12.10 (78.29) | 18.29 s | 85.4 / 179.3 ms |
| DFlash2, FA-RDNA2 two tokens, interval 4 | 16,384 → 1,024 | 778.4 (1914.9) | 12.41 (75.44) | 21.05 s | 85.1 / 173.5 ms |
| Native MTP2, current branch (BF16 PLE) | 16,384 → 1,024 | 952.3 (2059.5) | 14.79 (92.27) | 17.20 s | 55.7 / 156.7 ms |
| Native MTP2, merged PR #94 (INT4 PLE) | 16,384 → 1,024 | 1459.6 (3585.4) | 14.35 (96.63) | 11.23 s | 58.3 / 160.5 ms |

To isolate the drafter choice, a subsequent same-recipe A/B used
`flashnext-dflash2` for both arms, switching only the speculative config
between native MTP2 and BF16 DFlash2. Both used two draft tokens, the 24-row
capture ladder at c=8, the same AWQ target snapshot, BF16 PLE, MonoKernel,
FA-RDNA2 selection, TP4/EP4 on HIP 6–9, temperature 0, seed 12345, and
`--ignore-eos`. Each arm ran c=1 (eight requests) then c=8 (32 requests) at
1k/512 and 16k/1k. The dry-run commands and actual API arguments matched
after accounting for the speculative config and port:

| Drafter | Workload | Prefill tok/s/req (agg estimate) | Decode tok/s/req (agg output) | Mean TTFT | ITL median / mean |
| --- | --- | ---: | ---: | ---: | ---: |
| Native MTP2 | 1,024 → 512, c=1 | 2221.2 (2176.3) | 81.74 (76.27) | 0.46 s | 23.8 / 23.9 ms |
| DFlash2, two tokens | 1,024 → 512, c=1 | 2091.2 (1953.3) | 43.59 (41.92) | 0.49 s | 49.6 / 49.6 ms |
| Native MTP2 | 1,024 → 512, c=8 | 1012.4 (2193.3) | 32.30 (233.38) | 1.01 s | 53.9 / 64.2 ms |
| DFlash2, two tokens | 1,024 → 512, c=8 | 885.9 (2096.6) | 23.62 (172.25) | 1.16 s | 82.0 / 92.6 ms |
| Native MTP2 | 16,384 → 1,024, c=1 | 2111.3 (2095.1) | 96.45 (55.75) | 7.76 s | 24.3 / 24.3 ms |
| DFlash2, two tokens | 16,384 → 1,024, c=1 | 1999.2 (1986.3) | 41.50 (31.18) | 8.20 s | 51.1 / 51.1 ms |
| Native MTP2 | 16,384 → 1,024, c=8 | 1558.7 (5363.9) | 15.31 (102.59) | 10.51 s | 55.7 / 142.7 ms |
| DFlash2, two tokens | 16,384 → 1,024, c=8 | 1706.1 (7356.6) | 12.53 (87.32) | 9.60 s | 85.7 / 168.3 ms |

All paired benchmark requests completed. DFlash2's aggregate output was
45.0%, 26.2%, 44.1%, and 14.9% below native MTP2 in table order. The long
c=8 cells followed long c=1 cells with prefix caching enabled; native and
DFlash2 reported 13.6% and 15.5% cache hit rates by the end, respectively.
Their large aggregate prefill estimates and TTFT differences include cache
effects and do not establish a prefill-kernel advantage. The earlier cold c=8
runs likewise showed about a 15% long-context output gap between two-token
DFlash2 (78.29 tok/s) and native MTP2 (92.27 tok/s).

Native MTP2 passed its paired concurrency probe. The paired DFlash2 probe and
an isolated recheck each flagged one divergence after the same completed
haiku and generated `<|endoftext|>` while forcing continuation with
`ignore_eos=True`. Saved solo/concurrent tokens match through the end-of-text
token; a separate normal-EOS probe was not run. Greedy, perplexity, prefix,
and after-load probes passed, and no V620 compute fault was recorded. The
native arm's dmesg delta contained corrected PCIe AER on a separate W7800
at startup, not an amdgpu compute fault on HIP 6–9.

Merged PR #94 used the same target, TP4, four V620s on HIP 6–9, 32 requests
per c=8 cell, and these standard lengths. It used INT4 PLE; the current-branch
runs used BF16 PLE. Native MTP2 speculative acceptance at 16k was 65.3% on
this branch versus 66.0% in PR #94. The current branch retains earlier native
MTP2 decode speed; the DFlash2 long-context gap is specific to its
drafting/verification path. The native control's 16k mean TTFT was 17.20 s
versus PR #94's 11.23 s, but their medians were 8.70 and 9.00 s. The mean
difference comes from the slow tail (p99 63.64 versus 36.56 s), so the
prefill-rate estimates in the table do not establish a prefill-kernel
regression; they include queueing and the slow tail.

All standard-length c=8 benchmark requests completed, with no V620 compute
faults recorded during the runs. Triton and an earlier two-token FA-RDNA2 run
passed their concurrency probes; the later strict paired two-token run had the
post-end-of-text failures described above. The first three-token FA-RDNA2
probe had two strong output divergences despite passing greedy, perplexity,
prefix, and after-load checks. After a server reboot, its repeat probe passed
with zero strong failures across 48 checks. A fixed eight-prompt FA-RDNA2/Triton
comparison found no systematic high-margin wrong-token pattern; first
candidate differences were in both backends' top-five lists. The cause of the
server reboot remains unknown. At 16k/1k, the three-token FA-RDNA2 draft
accepted about 40.3% of speculative tokens versus about 47.7% for Triton.
Reducing to two draft tokens raised FA-RDNA2 acceptance to about 58.6% and
kept accepted output per target verification step nearly unchanged (2.17
versus 2.21 tokens). It also reduced the c=8 verification batch from 32 to 24
rows, restoring MonoKernel K1, and lifted aggregate output from 34.98 to
78.29 tok/s. At 1k/512, two tokens fell from 222.47 to 163.91 tok/s, so the
better depth depends on the workload. These are single runs and the 16k cell
has shown variance; repeat before treating small FA-RDNA2/Triton differences
as stable. An instrumented c=8 phase run with eight 16k/1k requests recorded
286 mixed 1024-token prefill/decode steps averaging 551.5 ms, versus 418
pure decode steps averaging 51.6 ms. Mixed target forward consumed about
517.7 ms per step;
draft work consumed 30.9 ms and the inter-step gap 1.5 ms. These groups
account for most of the run's 193.2-second wall time. Its absolute throughput
is not directly comparable to the 32-request matrix.

`PREFILL_INTERVAL=4` raised the three-token 16k/1k c=8 aggregate output from
34.98 to 70.24 tok/s and reduced mean TTFT from 44.32 to 26.24 s. It did
not improve 1k/512 c=8 (222.47 versus 220.24 tok/s) or the two-token 16k
configuration (78.29 versus 75.44 tok/s, with TTFT rising 18.29 to 21.05 s).
The recommended settings are therefore the recipe defaults for short
contexts, and `DFLASH_TOKENS=2` for the measured long-context workload.

Profiling the two-token path with `VLLM_RDNA_PHASE_TIMING=50` separates its
context preparation from draft graph replay. At 1k/512, steady c=1 steps
averaged 48.24 ms: target forward 18.94 ms and drafting 28.36 ms, including
2.79 ms context processing and 25.53 ms replay (200 sampled steps). At c=8,
250 pure-decode steps averaged 77.36 ms: target forward 43.78 ms and
drafting 31.94 ms, including 29.08 ms replay. Input preparation and metadata
were small. At 16k/1k c=8, pure-decode steps averaged 80.81 ms, while 69 mixed
prefill/decode steps averaged 547.02 ms, including 492.10 ms target forward
and 53.27 ms drafting. These bounded profiling cells used one c=8 wave,
not the 32-request throughput matrix above.

A separate eager diagnostic used finer GPU-event markers. The BF16 draft
LM-head GEMM consumed 9.98 ms at c=1 and 9.69 ms at c=8; local top-k took
0.07/0.09 ms, candidate gather 0.19/0.21 ms, and the selector 0.43/0.42 ms.
The five draft layers took about 15.01/18.03 ms, including 5.73/5.96 ms in
MLP blocks and 1.69/4.09 ms in attention. Thus the full-vocabulary BF16 head
is the largest individual draft stage; top-k and candidate communication
are small. The eager model is not a production throughput baseline. Its
model/head/selector/context totals agree approximately with the graph-mode
draft span. GPU-event spans can include stream idle time; the torch-profiler
traces on this runtime contained CPU events only.

The observed head shapes are BF16 `M=2/16, N=62080, K=2560` at c=1/c=8.
Those keys and the two-token query widths `M=3/6/12/24` were absent from the
initial 58 BF16 rows, which cover the three-token capture widths. Missing
rows used the rocBLAS heuristic. An isolated exact-shape comparison on HIP 6
(five warmups, 30 repetitions, same library hash) confirmed the head gap:
current lookup took 8.670/8.772 ms for M=2/16, the heuristic 8.680/8.772 ms,
and fresh scratch lookup 1.636/1.716 ms. The tuner selected
`Gemm_Rocblas_-456` for both. This identifies about 7 ms per draft step of
potential savings, subsequently tested in the serving A/B below.
The canonical profile was unchanged by this diagnostic. A subsequent offline
campaign tuned 36 missing spec2 keys independently on all four GPUs and
added them to the existing `tunableop/rocm7.14-rocblas5.5/` rank files. The
profile now has 1860 rows/rank (1766 FP16 + 94 BF16), with every previous row
preserved. All 144 isolated measurements improved by at least 2.354x
(median 3.417x); 36/36 lookup hits were verified on each GPU. See the
[measurement evidence](../../bench_results/2026-10-20_dflash2-spec2-tunableop/README.md).
A full same-boot serving A/B then completed all four standard cells with
eight c=1 or 32 c=8 requests per cell. Tuned aggregate output was 64.62 /
208.78 tok/s at 1k/512 c=1/c=8 and 43.35 / 85.60 at 16k/1k, versus 40.99 /
173.83 and 33.66 / 77.28 for the old profile: gains of 57.7%, 20.1%, 28.8%
and 10.8%. Median ITL fell 37.2%, 22.6%, 36.1% and 21.0%. Greedy, PPL
(4.81 old / 4.87 new), prefix, normal-EOS and after-load probes passed;
all eight cache resets passed and no GPU faults were recorded. The linked
evidence includes per-request/aggregate prefill and decode, TTFT, ITL,
acceptance and provenance. These are one paired run per cell. An earlier
interrupted pre-reboot run is excluded from this comparison.
Variable auxiliary-FC/mixed-prefill context widths are outside these 36
keys. A subsequent same-boot short-context depth control with this profile
measured 69.78/213.38 aggregate output tok/s at depth three, versus
64.62/208.78 at depth two (c=1/c=8). Depth two had lower median ITL
31.2/63.4 ms versus 32.7/69.6 ms, and c=8 TTFT 1.06 versus 1.16 s.
Depth three remains the short-context throughput default; depth two is an
option for lower streaming latency and the explicitly selected long-context setting.
The 40 depth-three requests and all probes passed (PPL 4.79, normal-EOS
0 failures/48 checks, no recorded GPU faults). These are single paired runs.

The fresh native MTP2 control used the same boot, source, expanded profile,
target, GPUs, two draft tokens and captures [3,6,12,24]. With a verified cache
reset before each cell, native aggregate output was 90.98/236.85 tok/s at
1k/512 c=1/c=8 and 57.15/91.92 at 16k/1k. Tuned DFlash2 was 29.0%, 11.9%,
24.1% and 6.9% slower, respectively. Native median ITL was 23.9/54.3/24.4/55.4 ms
versus DFlash2's 31.2/63.4/32.7/67.3 ms. All native 80 benchmark requests,
four cache resets and probes passed (PPL 4.79, normal-EOS 0 failures/48,
no recorded GPU faults). Native versus DFlash2 had no matching-prefix
premature stops; near-tie token/stopping differences remain. Native added
client `--save-detailed`; server arguments matched except drafter and port.
The linked evidence contains all four cells' full metrics and retained
request data. These single paired runs establish an improvement from tuning,
while native MTP2 remains faster than this external drafter.

Download the draft into the Hugging Face cache before serving (the launcher
defaults to offline mode), then launch with the target checkpoint as `MODEL`:

```bash
hf download incoai/Qwen3.8-Flash-Next-DFlash2
MODEL=/path/to/Qwen3.8-Flash-Next-AWQ-W4A16 \
VENV=/path/to/rocm-venv HIP_VISIBLE_DEVICES=6,7,8,9 \
  bash tools/rdna/serve_rdna.sh RECIPE=flashnext-dflash2
```

For the measured 16k/1k c=8 profile, set `DFLASH_TOKENS=2` in the caller
environment before invoking the same recipe. The recipe then also selects the
24-row verification capture ladder:

```bash
MODEL=/path/to/Qwen3.8-Flash-Next-AWQ-W4A16 \
VENV=/path/to/rocm-venv HIP_VISIBLE_DEVICES=6,7,8,9 DFLASH_TOKENS=2 \
  bash tools/rdna/serve_rdna.sh RECIPE=flashnext-dflash2
```

For a draft stored outside the default cache, override `SPEC_CONFIG` with a
JSON object whose `model` is its directory. Use
`"attention_backend":"TRITON_ATTN"` in that object for a draft-attention
comparison. Set `EXTRA_ARGS=` to compare without MonoKernel and use
`flashnext-mtp2` for a native MTP baseline. The checkpoint's
five target layer IDs select the completed zero-based layer outputs, averaged
over the HyperConnection streams to the model's native hidden width.

### Dynamic MTP graph capture

Target verification and draft prefill consume scheduled verification tokens;
subsequent autoregressive draft steps consume one token per request. Applying
the target schedule to those steps can generate negative capture widths and
fail `InputBatch.make_dummy` during graph-memory profiling.

With eight sequences, MTP=4 and schedule
`[[1,1,4],[2,2,2],[3,4,1],[5,256,0]]`, steady verification token totals
are 5, 6, 6, 8, 5, 6, 7, 8. A dense 1–8 ladder covers those shapes, but
does not cover large prefill or mixed batches. Bound uniform FULL captures
independently of a larger PIECEWISE ladder; keep mixed/varlen fallbacks.

**Capture coverage is not reliability certification.** Dynamic MTP also needs
accepted-state handoff and matching zero-draft PLE metadata at capture/replay.
The bounded-only image faulted under concurrency; do not omit those fixes.
For the tested TP8/eight-sequence schedule, the best concurrent ladder was
1–8 plus 64, 128, 256, 512, 1024, 1032 and 2048 rows (2.28–2.38 GiB/rank).
The 1–8-only alternative saves ~1 GiB/rank but loses concurrent decode speed.
All three passed correctness gates; 54 benchmark cells informed selection.
Revalidate prefix reuse, churn and repetition; this is not long-soak certification.
QSA ring alignment requirements remain independent of graph sizes.

### Attention

| Env | Default | Notes |
| --- | --- | --- |
| `VLLM_USE_RDNA2_FA` | 0 | 1 pins `RDNA_ATTN` (FA-RDNA2) on gfx10x when no backend is set. |
| `VLLM_FA_RDNA2_GQA_DECODE` | 0 (launcher `ATTN=fa`: 1) | GQA-grouped decode kernel. |
| `VLLM_FA_RDNA2_GQA_MODE` | subgroup | GQA prefill mode. `subgroup` was measured faster than varlen/splitk. |
| `VLLM_FA_RDNA2_SPLIT_DECODE` | 1 | Split-K decode for MTP-verify and mixed batches (gated to ≤ 256 query×head rows). |
| `VLLM_FA_RDNA2_VERIFY_FULL_GRAPH` | 1 | Capture MTP-verify steps as FULL graphs (`UNIFORM_BATCH`). |
| `VLLM_FARDNA2_ENABLE_SPEC_GATE`, `VLLM_FARDNA2_SPEC_VERIFY_Q_LEN` | 0, 3 | Legacy gate that rejects verify batches. Leave off. |
| `VLLM_USE_RDNA2_MLA` | 0 | HIP sparse-MLA decode (DeepSeek-V4) instead of Triton. |

### Linear GEMMs (W4A16 / W4A8 / dense fp16)

| Env | Default | Notes |
| --- | --- | --- |
| `VLLM_RDNA2_W4A8_SDOT4` | 0 | **Opt-in** (the `27b-awq` recipe sets it to 1). int8-activation prefill GEMMs; decode stays W4A16. Dense from M ≥ 33 (27B AWQ: prefill +24–34 %, PPL 4.90 vs 4.94); MoE from 512 tokens (Flash-Next: prefill +14–17 %, PPL 4.84 vs 4.86). `tools/rdna/bench_moe_w4a8.py` A/Bs the MoE kernel. Re-measured on the 27B AWQ tree (2026-10-25, one boot, one knob changed, raw PASS): the fitted prefill cost falls from 0.94–1.15 to 0.70–0.93 ms/token, 16k c=1 prefill 860 → 1075 tok/s (TTFT 19.1 → 15.2 s), 16k c=8 aggregate output 42.5 → 52.3 tok/s (TTFT 48.8 → 34.4 s), paired PPL 4.94 → 4.91, greedy 3/3 on both sides. |
| `VLLM_RDNA2_W4A16_RUNTIME_DISPATCH` | 0 | Choose decode/prefill kernel per call at run time (custom op) instead of at trace time. |
| `VLLM_RDNA_DENSE_INT8`, `_ONLY`, `_MIN_ROWS` | 0, 0, 64 | int8 weight-only shadow copies of dense fp16 projections for decode. Off in all recipes. |
| `VLLM_RDNA_DENSE_GEMV` | 0 | Force `gemv_f16_rdna2` for n ≤ 5. **Never route anything to wvSplitK on gfx1030**: it asserts on device. |
| `VLLM_DISABLED_KERNELS` | — | Upstream registry: e.g. `RDNA2W4A16LinearKernel` to fall back to Triton/Exllama. |

### MoE

| Env | Default | Notes |
| --- | --- | --- |
| `VLLM_RDNA2_MOE_FP32_ACCUM` | 0 | 0 = packed-fp16 CAS epilogue (faster, run-to-run ULP flips). 1 = fp32 scratch (deterministic, for debugging/A-B). |
| `VLLM_RDNA2_MOE_BSM_LARGE`, `_BSM_MIN_TOKENS` | 8, 512 | Prefill tile width (rows) and the token count where it kicks in. 0 keeps the decode tile. |
| `VLLM_ROCM_MOE_SKINNY`, `_MAX_M` | 1, 8 | Wave-per-row skinny GEMV for **Triton WNA16** MoE weights only (max 16). Not used by the shuffled RDNA2 MoE path. |
| `VLLM_RDNA_MOE_RESIDENT`, `_SKINNY` | 0 | Keep RDNA2 W4A16 MoE weights in the native shuffled layout. Disables W4A8 MoE. |
| `VLLM_RDNA_FUSED_SE` | 1 | Fused shared-expert path (Qwen2/3-MoE family). |

### Hybrid models (GDN, conv1d, Qwen4Exp hyper-connection, QSA, PLE conv)

| Env | Default | Notes |
| --- | --- | --- |
| `VLLM_GDN_DECODE_RDNA2` | 1 | HIP GDN decode. |
| `VLLM_GDN_HIP_PREFILL` | 1 | HIP GDN prefill chain (prep/solve-WY/delta-h/o). On by default since the 2026-10-15 kernel rewrites; faster than Triton at every served shape. |
| `VLLM_RDNA_GDN_OPAQUE` | 1 | Whole GDN layer as one opaque op. 0 keeps the projections in the compiled graph. |
| `VLLM_CAUSAL_CONV1D_RDNA2_FWD`, `_UPDATE` | 1 | HIP causal conv1d. Recipes set 0 (Triton). |
| `VLLM_CAUSAL_CONV1D_RDNA2_SPEC_STATE` | 0 | Let those HIP kernels run on an MTP-sized conv state (`width - 1 + num_spec`). They read and write only the first `width - 1` columns, so the extra capacity is ignored -- opt-in until validated on the other recipes and MTP; off means an over-capacity state takes Triton. |
| `VLLM_RDNA_FUSED_HC` | 1 | Fused hyper-connection decode (Flash-Next). |
| `VLLM_RDNA_HC_PREFILL_FUSED` | 2 | HC prefill: 0 torch+Triton, 1 fused mix kernel, 2 split-K HIP down + fused kernel. |
| `VLLM_RDNA_HC_PAD_M` | 64 | Pad HC row counts to the 64-step grid that has TunableOp rows. 0 disables (and loses ~60 ms per 512-token mixed step). |
| `VLLM_RDNA_HC_PREFILL_HIP` | 1 | HIP hyper-connection prefill. On by default (end-to-end verified). |
| `VLLM_RDNA_PLE_CONV_HIP` | 0 | Opt-in HIP PLE dilated short-conv; not end-to-end verified. |
| `VLLM_RDNA_QSA_HIP` | 1 | HIP QSA path. On by default: faster than Triton at all nine served shapes. |
| `VLLM_RDNA_ALL_HIP` | 1 | Umbrella for the RDNA2 HIP kernel set: worker/sampler, top-k/top-p masking, QSA metadata, MTP conv1d + GDN spec-verify, unquantized MoE, vision pos-embed. Default on since 2026-10-16; set 0 to fall back to the pre-port Triton kernels. With it set the Triton audit shows zero launches and decode is ~99 vs ~89 tok/s (V620, TP4, MTP-2). Independent of the per-op flags above. |
| `VLLM_RDNA_MROPE_BLOCK` | auto | HIP M-RoPE block size. `auto` = one block per token, striding the rotary index across the block; `1` restores the one-thread-per-token launch, which at decode (3 tokens) runs 3 threads and makes the kernel ~50x slower (494 vs 10 us, gfx1030). |

### All-reduce (TP over PCIe)

| Env | Default | Notes |
| --- | --- | --- |
| `VLLM_RDNA_AR` | 0 (recipes 1) | RDNA one-shot push all-reduce for small messages, RCCL above. Check the log for `rdna_ar: … active` and no `rdna_ar: disabled`. |
| `VLLM_RDNA_AR_MAX_KB`, `_ONESHOT_KB` | 64, 64 | Size gate. **Do not raise**: two-shot is not boot-safe under PCIe load. |
| `VLLM_RDNA_AR_ALGO` | auto | `twoshot` forces two-shot (testing only). |
| `VLLM_RDNA_AR_BLOCKS`, `_PACE`, `_WAIT_MS` | auto, 0, — | Block cap, pacing, and peer-flag wait timeout. |
| `VLLM_FORCE_CUSTOM_ALL_REDUCE` | 0 | Stock custom AR past the XGMI gates. **Keep 0**: it has crashed the host (PCI SERR). |
| `NCCL_P2P_LEVEL=pxb`, `NCCL_PROTO=Simple`, `RCCL_P2P_NET_DISABLE=1`, `RCCL_P2P_BATCH_ENABLE=1`, `RCCL_MSCCL_ENABLE=0` | | RCCL settings for RDNA PCIe. LL/LL128 can deadlock. |
| `GPU_MAX_HW_QUEUES` | launcher 2; five Flash-Next recipes 4 | Launch-environment HIP queue count (`rdna_launcher_common.sh` defaults to 2). `flashnext`, `flashnext-exl3`, `flashnext-mtp0`, `flashnext-mtp2`, and `flashnext-dflash2` override to 4 — leave that. Recipes that omit the key keep 2. Do not drop the Flash-Next recipes to 2; a host-specific setting of 2 is only an unverified workaround reported in [ROCm/TheRock#9306](https://github.com/ROCm/TheRock/issues/9306). Not a client setting. |

### Scheduler

| Env | Default | Notes |
| --- | --- | --- |
| `VLLM_RDNA_DECODE_STALL_MS` | 0 | Default for `--decode-stall-budget-ms` on RDNA. Dense TP=4 recipes: 250 (max decode gap during 16k prefill 2672 → 320 ms, −8 % c=8 output). Opt-in on Flash-Next (−13 %), off at TP=2. The 27B AWQ recipe now sets 0: once the prefill budget is derived from the mamba state block, the cap only chunks the prefill below that boundary, and it costs 12–19 % TTFT at c=8 for 0–2.5 % output (2026-10-25, one boot, raw PASS). |
| `VLLM_RDNA_DECODE_SHARE` | 0 | Default for `--decode-time-share`. |
| `VLLM_DECODE_STALL_MAX_OVERHEAD` | 0.25 | Floor on the cap so per-step fixed cost stays below this fraction of per-token work. |
| `VLLM_DECODE_STALL_FIT_ALL` | auto | Fit step cost on all prefill steps (dense default) or mixed steps only (MoE default). |
| `VLLM_RDNA_DYNAMIC_PREFILL` | 0 | Experimental live tuning of chunk cap and cadence. |

### Pipeline parallel (3 or 4 cards)

Flash-Next has nothing that divides by 3 (heads, experts, `moe_intermediate`),
so three cards means PP=3 at TP=1. Recipes `flashnext-pp3` (MTP0) and
`flashnext-pp3-mtp2` follow
[alanoo81/flashnext-v620-pp3](https://github.com/alanoo81/flashnext-v620-pp3):
text-only, 262K, KV sized from free memory, int4 PLE.

- **Step budget vs the Mamba block.** With prefix caching in `align` mode the
  block is 2048 tokens and prefill chunks are whole blocks. When decode tokens
  share the step, `MAXBAT` must exceed the block by the decode tokens of a full
  batch, or admission stalls (16K c=4 at MTP2: 3 running + 1 waiting at 2048,
  4 running at 2112). PP3 MTP2: `MAXBAT=2112` (2048 + 4 seqs × 3). TP2×PP2
  MTP2 at 8 seqs: `MAXBAT=4096`. Sub-block chunks are not an option (they
  poison the prefix cache, upstream #43559).
- **Decode cadence.** Each request decodes every `pp_size` steps, so per-request
  decode at c>1 drops; MTP pays off up to ~4 streams, MTP0 above.
- **Memory.** Vision does not fit next to stage 0's layers (text-only), dense
  INT8 must be `ONLY` mode. The W4A16 MTP drafter checkpoint
  (`tools/rdna/quant_mtp_experts.py`) is required for MTP K2; compatible
  attention HyperConnections can use opt-in MTP K1 without it. Under PP the
  drafter shares the last stage's target embedding and `lm_head` (CPU
  placeholders until shared). On 30 GB V620s a full 262K request at MTP2
  needs `GMEM=0.975`
  (0.95 caps the length at ~188K); 32 GB cards fit it at 0.95.
- **Partition.** 16,16,16 for MTP0; 17,17,14 for MTP2 (the drafter sits on the
  last stage). PLE layers must stay on stage 0.

Measured on 3× / 4× V620 (int4 PLE, F&P, `--temperature 0`, 2026-10-13):

| Config | Cell | Prefill tok/s/req (agg) | Decode tok/s/req (agg out) | TTFT | ITL med |
| --- | --- | --- | --- | --- | --- |
| PP3 MTP0 (`flashnext-pp3`, KV 398K tok) | 1k/512 c=1 | 842 (835) | 38.4 (35.2) | 1.22 s | 26.1 ms |
| | 1k/512 c=8 | 204 (1078) | 9.0 (66.3) | 5.01 s | 106 ms |
| | 16k/1k c=1 | 1848 (1811) | 38.0 (28.6) | 8.87 s | 26.4 ms |
| | 16k/1k c=8 | 771 (1589) | 7.7 (53.3) | 21.3 s | 63.4 ms |
| PP3 MTP2 (`flashnext-pp3-mtp2`, KV 268K tok) | 1k/512 c=1 | 845 (842) | 57.7 (50.8) | 1.21 s | 36.3 ms |
| | 1k/512 c=4 | 535 (847) | 12.4 (46.1) | 1.92 s | 120 ms |
| | 16k/1k c=1 | 1866 (1839) | 70.9 (44.1) | 8.78 s | 37.0 ms |
| | 16k/1k c=4 | 1071 (3088) | 10.1 (34.7) | 15.3 s | 122 ms |
| TP2×PP2 MTP2 (`flashnext-mtp2 TP=2 PP=2 MAXBAT=4096`) | 1k/512 c=1 | 1496 (1453) | 52.6 (49.2) | 0.68 s | 37.8 ms |
| | 1k/512 c=8 | 565 (1309) | 13.7 (100.5) | 1.81 s | 141 ms |
| | 16k/1k c=1 | 2199 (2189) | 54.8 (39.2) | 7.45 s | 38.1 ms |
| | 16k/1k c=8 | 1055 (2399) | 8.6 (59.8) | 15.5 s | 132 ms |

On four cards TP=4 MTP2 decodes faster (16k c=1 76–87 tok/s, 16k c=8 ITL
~62 ms) while TP2×PP2 has higher aggregate prefill at 16k c=8 (2399 vs
1050–1350 tok/s). PP2's concurrency gate shows the known prompt-1 near-tie
flip. Raw runs: infrastructure `bench_results/2026-10-13_pp3-mtp2/` and
`2026-10-13_bundle-final-ab/` (TP4).

### PLE table (Flash-Next)

| Mode | Env | Host RAM peak | Notes |
| --- | --- | ---: | --- |
| `PLE=bf16` | (none) | ~160 GB | Upstream pinned BF16 table. |
| `PLE=int4` | `VLLM_PLE_QUANT_DIR=<ples_int4>` | ~90 GB | INT4 table in pinned RAM, dequantized in the lookup. |
| `PLE=int4-disk` | + `VLLM_PLE_CPU_OFFLOAD=1` | ~45 GB | CPU sidecar serves mmapped shards (page cache). Set `VLLM_PLE_OFFLOAD_READY_TIMEOUT=3600` (default 600 s) and `expandable_segments:False`. |

All three are validated with MTP0/MTP2 under agentic load at speed parity.
Tables come from `primitive-ai/Qwen3.8-Flash-Next-PLE-quant` (only `ples_int4`
is wired). `PLE_OFFLOAD_DOORBELL` (1) and `PLE_OFFLOAD_PREFAULT` (1) are
sidecar internals; leave them on.

### EXL3

`VLLM_EXL3_FOLDED_CACHE=<dir>` caches folded fp16 weights across loads.
`VLLM_EXL3_MEMORY_MODE` (`full`|`packed`), `VLLM_EXL3_M_MAX` (64),
`VLLM_EXL3_FUSED_PROJECT` (1), `VLLM_EXL3_PREFILL_DECODE` (1),
`VLLM_EXL3_GEMM_V3` (1), `VLLM_EXL3_V3_BLOCK_K` (0) and `VLLM_EXL3_HADAMARD`
(`both`) select kernel variants. `VLLM_EXL3_DEQUANT_ALL=1` forces the fp16
path (correctness A/B only).

### Other

- `VLLM_USE_BREAKABLE_CUDAGRAPH`: auto-enabled for models whose RDNA path
  needs eager breaks inside FULL graphs (DeepSeek-V4, Qwen3.5/3.8 hybrids).
  Set it explicitly only to override.
- `VLLM_HIPPIHX`, `_LIB`, `_CODE_OBJECT`: route ops through a prebuilt hippihx
  V1 library when its plan is ready.
- `VLLM_NO_PARTIAL_HASH=1`: disable partial-block prefix hits (debugging).

### Diagnostics (off by default; slow, do not bench with them)

NaN tracing: `VLLM_{LAYER,MOE,HC,GDN,PLE}_NAN_DEBUG`, `VLLM_RDNA2_MOE_DEBUG_NAN`.
Kernel launch logs: `VLLM_RDNA2_W4A8_DEBUG`, `VLLM_RDNA2_W4A8_MOE_DEBUG`,
`VLLM_RDNA2_PREFILL_DEBUG`, `VLLM_QSA_RDNA2_DEBUG`, `VLLM_CONV1D_DEBUG`,
`VLLM_EXL3_DEBUG`, `VLLM_EXL3_HADAMARD_DBG`, `VLLM_PAGED_MQA_DEBUG_OOB`.
Cache/scheduler traces: `VLLM_DEBUG_PREFIX_HIT`, `VLLM_TRACE_PREFIX_CACHE`.
Timing: `VLLM_RDNA_PHASE_TIMING=<every N steps>` (+ `VLLM_RDNA_OP_TIMING=1|2`),
`VLLM_RDNA_SYNC_DEBUG`. PLE sidecar:
`PLE_OFFLOAD_DEBUG_{HOPS,TRACE,DELAY_MS}`, `PLE_OFFLOAD_FUSED_CHECK`.
GPU profiling: the launchers set `ROCPROFILER_REGISTER_ENABLED=0`, so the HIP/HSA
runtimes never register with rocprofiler (no tool init, no queue interception)
while serving. Set `ROCPROFILER_REGISTER_ENABLED=1` for torch-profiler or
rocprofv3 GPU traces.

## TunableOp

gfx1030 GEMMs run on rocBLAS (hipBLASLt is off: `TORCH_BLAS_PREFER_HIPBLASLT=0`,
`PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0`). TunableOp replaces rocBLAS's heuristic
pick with a measured solver per exact GEMM shape `(op, M, N, K)`. Tuned rows
run ~15 % faster than the heuristic, and much faster on skinny shapes.

**What ships**: `tunableop/<profile>/tunableop_results{0..3}.csv` (one file per
rank) plus `tunableop/profiles.json`. Solver IDs only mean something to the
rocBLAS build that produced them, so one folder per build:
`rocm7.14-rocblas5.5` is the canonical serving profile and covers Flash-Next
MTP0/2 (including mixed-step and padded HC shapes), the DFlash2 BF16 draft,
27B AWQ (W4A8 0/1), 27B EXL3, Qwen3.6-35B-A3B GPTQ and DeepSeek-V4-Flash.
Keep every row for a build in that one folder; do not start parallel row sets.

**How it is consumed**: the launcher sources `tools/rdna2_028/tunableop_env.sh`,
which hashes the loaded `librocblas.so.5`. It then picks the profile whose
`lib_sha256[:12]` matches and sets `PYTORCH_TUNABLEOP_ENABLED=1`, `TUNING=0`
(lookup only) and `PYTORCH_TUNABLEOP_FILENAME` at those rows.
`TUNABLEOP_PROFILE=<name>` forces a profile and fails the launch on a hash
mismatch. If no profile matches, it falls back to
`$HOME/.cache/tunableop/tunableop_results.csv`. Look for
`TunableOp profile … auto-selected` in the log. **Never** point rows at `/tmp`
or the run CWD.

A shape without a row silently runs the heuristic. Shapes change when you
change `MAXBAT`, `BLOCK_SIZE` (chunk alignment), MTP, TP or the model. The
HC pad (`VLLM_RDNA_HC_PAD_M`) exists to keep mixed steps on tuned rows.

**Generating rows** (service stopped, GPUs exclusive, everything under a
persistent work dir):

1. **Census.** Serve the target workloads with
   `PYTORCH_TUNABLEOP_UNTUNED_FILENAME=<work>/untuned%d.csv` (lookup still on)
   to record shapes that have no row. Not `_RECORD_UNTUNED_*`; see
   `tools/rdna2_028/probe_record_untuned.py`.
2. **Tune.** `PHASE=tune tools/rdna2_028/tunableop_rows_pipeline.sh` (driver
   `tune_prod_shapes.py --mode tune`) tunes each shape on every rank into
   `<WORK>/scratch/`. It also adds the M ≤ 8 decode shapes of every (N, K)
   seen.
3. **Measure + curate.** `PHASE=measure` times each shape three ways (heuristic,
   current rows, new rows). `curate_tunableop_rows.py` keeps the fastest per
   shape, drops rows slower than the heuristic, and omits `Default` rows.
4. **Merge and prove.** Merge into `tunableop/<profile>/`, update
   `rows_per_rank` in `profiles.json`, and run `verify_tunableop_lookup.py
   --rows <file>` to show each shipped row is actually hit. Then rerun the
   serve A/B: tuning can change reduction order, so check correctness gates
   as well as speed.

For a new rocBLAS build (or gfx11), make a new profile folder with its own
`lib_sha256` and run the full census → tune → curate → merge campaign.
`tunableop/README.md` has the per-profile provenance details.
