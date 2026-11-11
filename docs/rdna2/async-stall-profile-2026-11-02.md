# Prefill/decode asynchrony, batching and stalls: rocprofv3 serve profile (2026-11-02)

Question: is prefill/decode work asynchronous end to end, does batching work,
and is there any stage that waits? Answer for the resident path (Flash-Next
MTP2, TP4): **asynchrony and batching work; the waste is inside the GPU
step** — RCCL all-reduces that miss the `rdna_ar` gate, about 2,800 kernels
per step with a fixed per-node gap, and decode steps that share a batch with
a 2,048-token prefill chunk. The offload path is not profiled yet.

Status: profiling tooling committed. All-reduce gate change measured under
the profiler; the unprofiled serve A/B is running. Fixes are in progress in
two parallel lanes (see "Improvement work and plan" at the end). No runtime
code changed yet.

## Setup

| Item | Value |
|---|---|
| Model | Qwen3.8-Flash-Next AWQ W4A16, PLE int4 offload, MTP=2 |
| Recipe | `flashnext-mtp2`: TP4, `FULL_AND_PIECEWISE`, prefix caching, `SEQS=8`, `MAXBAT=2048`, `LPTH=0`, `PREFILL_INTERVAL=1` |
| GPUs | 4x V620 (HIP 2-5), venv `venv-7.14.0_0.32.0`, tree `vllm-rdna-0.32.0` |
| Load | 1k/512 decode loop at c=6 plus one 16k-token prefill injector every 10 s |
| Trace | rocprofv3 kernel + HIP runtime + memcopy trace, 15 s collection window starting 200 s after tool init; 12.3 s analysed per rank |

## Findings

### 1. Async scheduling works: the host is not on the critical path

The step-marker kernel (`get_num_sampled_and_rejected`, once per engine step)
is followed by the next step's first kernel after **2.9-9.6 us median**, p90
about 10 us, on all four ranks. The next step is already queued when the
previous one finishes, so async scheduling (auto-on under the V2 runner) is
doing its job.

Long host calls are backpressure, not stalls:

- `hipEventRecordWithFlags`: median 4 us, but about 20 calls of about 110 ms
  (2.4 s total on ranks 0 and 2). Only 1 in 21 returns as the GPU drains, so
  the host is blocked on a full queue while the GPU is busy.
- `hipGraphLaunch`: median about 150 us, p90 about 4 ms (same cause).
- Rank 0 output thread: `hipEventSynchronize` 10.3 s total. This is the async
  output thread waiting for results, which is its job.

The baseline did show rare 22-30 ms step-boundary gaps (p99). They vanished
with the all-reduce change below (p99 10 us), so they were collective skew,
not scheduler waits.

Static-audit hypothesis 1 (pageable D2H in `async_utils.py`) is refuted:
`to("cpu", non_blocking=True)` uses pinned memory and no per-step block shows
in the trace.

### 2. Batching works; decode pays for co-scheduled prefill chunks

Decode and prefill chunks share steps as intended (`PREFILL_INTERVAL=1`, no
long-prefill threshold). The cost lands on the decoders in that step:

| Step type (rank 2, baseline) | Steps | Period mean | Period median | Period max | Kernels/step | GPU busy |
|---|---:|---:|---:|---:|---:|---:|
| decode-only | 86 | 62.2 ms | 62.0 ms | 76.2 ms | 2,797 | 48.7 ms (78%) |
| mixed (decode + prefill chunk) | 48 | 136.4 ms | 92.9 ms | 497.6 ms | 2,905 | 111.5 ms (82%) |

A decode stream that shares a step with a prefill chunk waits up to about
0.5 s for its next token. With `MAXBAT=2048` and decoders running, the
chunk is 1,024 tokens (the scheduler aligns chunks to the 1,024-token mamba
block; from code, not yet measured). The injector itself saw TTFT 8.9-11.2 s for
16k tokens and median ITL about 54 ms. With 8 decoders and a 16k prefill
already running when decoders arrive, the longest decode gap is 7.2 s
(stall probe, `serve-arab-A64`, reverse case). Existing knobs
(`--prefill-schedule-interval`, `VLLM_RDNA_DECODE_STALL_MS`,
`--long-prefill-token-threshold`) only ration prefill and trade its
throughput for decode latency; they are not the fix. The target is a
design where decode never waits for a prefill chunk (see "Stall-free mixed
prefill/decode" below).

### 3. TP all-reduces miss the `rdna_ar` gate and run on RCCL

The decode all-reduce is 18 verify tokens (6 sequences x (1 + MTP 2)) x
hidden 2560 x fp16 = **92 KB**, just above the 64 KB one-shot gate that every
recipe sets (`VLLM_RDNA_AR_MAX_KB=64`, `VLLM_RDNA_AR_ONESHOT_KB=64`). So:

| Collective kernel (rank 2, 12.3 s window) | Calls | Median | Total | Per decode step |
|---|---:|---:|---:|---:|
| `ncclDevKernel_Generic_4` (RCCL) | 13,809 | 79.8 us | 2,681 ms | **19.9 ms** |
| `rdna_ar_oneshot` | 503 | 23.5 us | 23.5 ms | 0.17 ms |
| `rdna_ar_argmax_oneshot` | 268 | 20.2 us | 5.4 ms | 0.04 ms |

That is 102 RCCL all-reduces per step (about 2 per layer over 48 layers).

The 64 KB gate is historical: it was set while the push two-shot kernel
existed (and was not boot-safe). Two-shot has since been removed
(`rdna_all_reduce.py` `resolve_gate_kb`), and the one-shot cap is a runtime
value (`VLLM_RDNA_AR_ONESHOT_KB` -> `g.oneshot_max`; staging buffer sized from
`VLLM_RDNA_AR_MAX_KB`), so raising both needs no rebuild.

Isolated sweep (4 ranks, HIP 2-5, serving NCCL env, fp16, 300 iterations,
CUDA-event timing including launch; one-shot result compared exactly against
an fp32 RCCL reference of integer-valued inputs):

| Size | rdna_ar one-shot median / p90 | RCCL median / p90 | Exact | Ratio |
|---:|---:|---:|---|---:|
| 16 KB | 34.4 / 51.6 us | 125.5 / 145.1 us | yes | 3.6x |
| 64 KB (current gate) | 53.2 / 68.1 us | 119.4 / 136.4 us | yes | 2.2x |
| **92 KB (MTP2 c=6 decode)** | **60.3 / 72.1 us** | **122.3 / 150.5 us** | yes | **2.0x** |
| 128 KB | 70.2 / 85.5 us | 126.8 / 152.1 us | yes | 1.8x |
| 256 KB | 99.0 / 111.9 us | 133.2 / 159.0 us | yes | 1.3x |
| 384 KB | 129.8 / 145.4 us | 136.9 / 164.8 us | yes | 1.05x |
| 512 KB | 168.4 / 187.7 us | 140.7 / 167.7 us | yes | 0.84x |
| 1 MB | 324.2 / 348.6 us | 169.3 / 196.1 us | yes | 0.52x |

No timeouts and no PCIe errors in `dmesg`. The crossover is about 384 KB; a
256 KB gate keeps margin and covers MTP2 decode up to about 17 concurrent
sequences (51 tokens).

Same serve profile with `VLLM_RDNA_AR_MAX_KB=256 VLLM_RDNA_AR_ONESHOT_KB=256`:

| Decode-only step (rank 2) | 64 KB gate | 256 KB gate |
|---|---:|---:|
| Steps in window | 86 | 217 |
| Period mean / median | 62.2 / 62.0 ms | 56.3 / 60.5 ms |
| GPU busy per step | 48.7 ms | 43.7 ms |
| Gaps < 10 us per step | 12.45 ms | 12.37 ms |
| Step-boundary p99 | 30.1 ms | 10.0 us |
| Collective kernels per step | 102 RCCL | 104 one-shot |

Collective kernel time in the window dropped from 2,383 ms (RCCL, 13,908
calls) to 1,078 ms (one-shot, 22,669 calls) plus 55 ms of RCCL for the 218
large (prefill) messages. Caveat: the injector did not fall inside the
256 KB window, so mixed steps are not compared, and the load mix differs.
**The unprofiled serve A/B is the source of truth.** Arms A (64 KB) and B
(256 KB), repeated in reverse order (B then A) for the 1k cells:

| Cell (TP4, flashnext-mtp2) | 64 KB | 256 KB | Delta |
|---|---:|---:|---:|
| 1k/512 c=6 aggregate output | 175.8 tok/s | 192.5 tok/s | +9.5% |
| 1k/512 c=6 ITL p50 | 54.4 ms | 52.0 ms | -4% |
| 1k/512 c=8 aggregate output (mean of 2 arms each) | 212.4 tok/s | 224.5 tok/s | +5.7% |
| 1k/512 c=8 ITL p50 | 57.1 / 57.5 ms | 55.6 / 55.7 ms | -3% |
| 16k/1k c=8 aggregate output (one arm each) | 89.0 tok/s | 91.3 tok/s | +2.6% |
| 16k/1k c=8 ITL p50 | 59.6 ms | 57.6 ms | -3% |

Part of the throughput delta is MTP acceptance (2.01-2.06 vs 2.07-2.10);
the ITL p50 drop is the cleaner signal. Prefill throughput and TTFT are
unchanged within noise. At c=1 the all-reduce message is 15 KB, below both
gates, so the c=1 delta is acceptance noise. Probes, PPL 4.85, prefix
cache and concurrency checks pass on both arms. **Shipped** for
`flashnext-mtp2` in `641e0c2288`; library defaults and the other recipes
stay at 64 KB until each is measured (hidden size and verify width
differ).

### 4. About 20% of each decode step is per-kernel gaps inside the graph

A decode step replays about 2,800 kernels through a FULL graph. Inter-kernel
gaps below 10 us add up to **12.4 ms per 62 ms step** (about 4.5 us per
node). The gaps stay the same with the all-reduce change, so they come from
how many kernels run, not from which ones.

Kernel composition of one mid-window decode step (256 KB run, rank 2, 2,803
kernels, 60.6 ms):

| Kernel | Per step | GPU time |
|---|---:|---:|
| HC / GDN / QSA custom kernels (`gemm_tn_splitk_k`, `hc_up_mix_prefill_k`, `hc_combine_norm_onepass_kernel`, `gdn_spec_verify`, QSA) | 387 | 11.9 ms |
| `direct_copy_kernel_cuda` | 290 | 1.16 ms |
| `FillFunctor` | 267 | 0.36 ms |
| rocBLAS/Tensile GEMMs (several tiles) | about 450 | about 11.5 ms |
| `__amd_rocclr_copyBuffer` | 120 | 0.21 ms |
| `rdna_ar_oneshot` | 104 | 5.4 ms |
| `BinaryFunctor` | 98 | 0.35 ms |
| `moe_gemm_q4_kernel_rdna2` | 96 | 8.7 ms |
| `_copy_kernel_cuda` | 57 | 0.12 ms |
| index | 54 | 0.24 ms |
| `topkGating` | 50 | 0.52 ms |
| sigmoid, nocast elementwise, `CatArrayBatchedCopy`, add | about 170 | about 0.6 ms |

About **950 generic PyTorch glue kernels** (copies, fills, binary, index, cat)
run per step: about 2.5 ms of execution plus about 4.3 ms of node gaps, or
about 11% of the step.

One source is confirmed: `rdna_hc_mix_prefill` (`csrc/rocm/hc_prefill_rdna2.cu`)
allocates its split-K fp32 accumulator with `at::zeros` on every call. At
M = 18 (> 8) the decode path takes it twice per layer, which gives about 100
fills per step. The other about 170 fills and 290 copies still need
attributing (graph replay hides their Python call sites from rocprofv3).

### Other idle sources (smaller)

- `copyBuffer -> copyBuffer` gaps of 24-133 ms per window: host preparation
  between consecutive copies.
- Gaps next to `_mamba_align_block_table_kernel` and `build_qsa_metadata_kernel`
  (per-step metadata preparation).

## Static audit hypotheses: status

| # | Hypothesis | Status |
|---|---|---|
| 1 | Pageable D2H in `async_utils.py:299` | Refuted (pinned; no per-step block) |
| 2 | Offload `k2.py:180-181` `route_ids.cpu()` drain per MoE layer | Pending offload profile |
| 3 | Offload padding `.cpu()` (`k2.py:186-190`) | Pending offload profile |
| 4 | Expert miss copies issued after the drain | Pending offload profile |
| 5 | `_map_upload_event.synchronize()` | Pending offload profile |
| 6 | Mixed prefill steps leave the FULL graph | Confirmed as a latency cost (finding 2) |
| 7 | No DBO, single compute stream | True; not the dominant cost here |
| 8 | Async scheduling active under V2 | Confirmed (finding 1) |

## How to profile (recipe and gotchas)

Driver and analyzers are in `tools/rdna/prof/`:

```bash
# rocprofv3 timeline (launch mode), 15 s window 200 s after tool init
P=<ples_int4 dir>
PERIOD=200:15:1 LOAD_S=90 PLE_QUANT_DIR=$P \
  bash tools/rdna/prof/serve_profile.sh RES 2,3,4,5 18402 rdna \
  RECIPE=flashnext-mtp2 PLE=int4 PLE_QUANT_DIR=$P
python3 tools/rdna/prof/rocprof_gaps.py ~/w4a8_runs/prof/RES/trace --top 6

# torch profiler with Python stacks (launch sites of graph contents)
PROFILER=torch LOAD_S=60 bash tools/rdna/prof/serve_profile.sh ATTR 2,3,4,5 18402 rdna \
  RECIPE=flashnext-mtp2 PLE=int4 PLE_QUANT_DIR=$P CG_MODE=NONE
python3 tools/rdna/prof/kernel_sites.py ~/w4a8_runs/prof/ATTR/torch/<rank0>.json.gz
```

`rocprof_gaps.py` reports GPU busy share, idle gaps by bracketing kernel
pair, host HIP calls overlapping gaps, and (`--step-marker`, default the
sampler kernel) a per-step decode/mixed breakdown with gap buckets and kernel
counts.

Gotchas, each of which cost a failed run:

- The launcher exports `ROCPROFILER_REGISTER_ENABLED=0`
  (`rdna_launcher_common.sh:86`); with it every trace is empty. The driver
  sets 1.
- Use the venv's rocprofv3
  (`_rocm_sdk_core/bin/rocprofv3`, v1.3.2). `/opt/rocm/core-7.14/bin/rocprofv3`
  is a different SDK build and aborts with "api registration failed with
  error code 16". The venv build has no `rocprof-attach`; attach mode does
  not work here.
- Wrap only the python server (`LAUNCH_WRAPPER` in `serve_rdna.sh`).
  Wrapping the bash launcher aborts: bash `exec`s python in the same PID and
  the tool cannot re-initialise.
- rocprofv3's SIGTERM handler in the API server waits on children that are
  never stopped. SIGTERM the `Worker_TP` PIDs first (each flushes its trace),
  then kill the rest.
- `--collection-period DELAY:DUR:1` counts from each process's tool init;
  workers init about 43 s after launch.
- vLLM's `capture_torch_profiler` exists only in the V1 runner; the recipes
  use V2, hence the `CG_MODE=NONE` attribution boot.

## Improvement work and plan

Direction set by the user on 2026-11-02:

- **No decode stall in mixed prefill/decode, by design.** Knobs that ration
  prefill (`VLLM_RDNA_DECODE_STALL_MS`, `--prefill-schedule-interval`,
  long-prefill thresholds) only trade prefill throughput for decode latency
  and are rejected as the fix. `MAXBAT=2048` is the established optimum;
  4096 is the one MAXBAT variant still worth measuring.
- **Optimize the rest:** fewer kernel launches and fewer Python calls per
  step, plus every other measured waste.

The eight V620s are split into two GPU lanes; two more lanes are analysis
only. Each lane is one agent. Server boots across lanes are serialized
with a lock on the box (`flock ~/w4a8_runs/boot.lock`, held from launch
until the server answers `/v1/models`), because two concurrent boots
stalled earlier; serving in parallel is fine.

| Lane | GPUs | Owner | Work, in order |
|---|---|---|---|
| Fix | HIP 2-5 (TP4) | agent `FixLane` | (1) all-reduce gate, (2) HC accumulator zeros, (3) kernel-launch and Python reductions from the map and attribution |
| Investigate | HIP 6-9 (TP4 / TP2) | agent `ProbeLane` | (1) glue-kernel attribution, (2) `GPU_MAX_HW_QUEUES` 4 vs 2, (3) offload profile, (4) stall baseline and MAXBAT 4096 |
| Kernel map | none (traces, py-spy on ProbeLane's servers) | agent `KernelMap` | per-layer kernel map of a decode step, ranked fusion/removal list, host Python breakdown per step |
| Async design | HIP 8-9 when lent by ProbeLane | agent `AsyncDesign` | design and feasibility of stall-free mixed prefill/decode |

HIP 2-5 sit behind PLX-B (BDF 47/4a/52/55) and HIP 6-9 behind PLX-A
(BDF 07/0a/12/15), so the lanes do not share a PCIe switch. The investigate
lane measures the unchanged baseline tree (`4c58ba25af`); the fix lane
builds and serves from its own tree and venv copy, and syncs a shipped fix
into the baseline only after the investigate lane agrees.

### Fix lane (HIP 2-5)

1. **All-reduce gate 64 KB -> 256 KB** (finding 3). Read the running A/B
   (`serve-arab-A64` vs `serve-arab-B256`: 1k/512 c=1, 1k/512 c=8, 16k/1k
   c=8, stall probe). If the gate wins or holds parity everywhere, raise
   `VLLM_RDNA_AR_MAX_KB` and `VLLM_RDNA_AR_ONESHOT_KB` in every recipe and
   launcher that pins 64, the library defaults (`_ONESHOT_DEFAULT_KB`,
   `RDNA_AR_ONESHOT_MAX`), and the `AGENTS.md` note.
2. **`rdna_hc_mix_prefill` accumulator** (finding 4): drop the per-call
   `at::zeros` fill (about 100 kernels per decode step) without breaking
   CUDA-graph safety or the split-K atomic accumulation. Kernel test,
   rocprofv3 before/after kernel count, serve bench.
3. **Kernel launches and Python per step**: implement the ranked list from
   the kernel map (fusions, removed copies/fills/casts, routing and norm
   fusions; HIP where it fits) and the host-side Python reductions, largest
   saving first.

Every fix: correctness probes (greedy 9/9, 0 NaN, prefix-cache hits),
serve bench (prefill, decode, TTFT, ITL) vs the same-day baseline, and a
rocprofv3 per-step comparison. Ship only on a win.

#### Prefill levers (2026-11-03)

A 2,048-token prefill step is GPU-bound (960.6 of 974.1 ms busy, median of 24
steps) and splits over five kernel families on one rank (rocprofv3, TP4
Flash-Next MTP2, 16k prompt): GDN 58.7 %, MoE 21.4 %, QSA 14.7 %,
attention/MHA 2.6 %, everything else 2.6 %. Inside GDN the `delta_h` prefill
kernel alone is 36.2 ms per 2,048-token chunk (36 calls).

Each lever was measured in one lease, one boot per arm, all four GPUs, against
arm A0 (the shipped tree, legacy QSA kernel, shipped TunableOp table): solo
TTFT at 4k and 16k (median of 3), 1k/512 at c=1/6/8, 16k c=8 aggregate
prefill, decode ITL, probes, PPL, prefix cache and a greedy agreement check
against A0's outputs. A0 reproduced the historical baseline to the hundredth
(solo 1.86 / 7.48 s, 16k c8 aggregate 2,033 tok/s, ITL 22.9 / 43.5 / 47.4 ms
at c=1/6/8, PPL 4.82), so the deltas below are same-day.

| Lever | Solo TTFT 4k | Solo TTFT 16k | 16k c8 agg prefill | Decode ITL (c1/c6/c8) | Quality |
|---|---:|---:|---:|---|---|
| A0 baseline | 1.86 s | 7.48 s | 2,033 tok/s | 22.9 / 43.5 / 47.4 ms | PPL 4.82, probes 3/3, prefix 19.9x |
| pipelined QSA kernel (now default) | 1.74 s (-6.5 %) | 6.95 s (-7.1 %) | 2,147 tok/s (+5.6 %) | 22.7 / 42.8 / 45.9 ms | PPL 4.83, probes 3/3, prefix PASS, 34/48 greedy identical, 13 near-tie, 1 special-token flip under churn |
| pipelined QSA + W4A8 MoE (`flashnext-mtp2` recipe) | 1.53 s (-17.7 %) | 6.04 s (-19.3 %) | 2,496 tok/s (+22.8 %) | 24.5 / 44.5 / 47.7 ms | PPL 4.86, probes 3/3, prefix PASS, 35/48 greedy identical, 10 near-tie, 3 soft, 0 failing |
| + GDN `delta_h` pipe (variant 1) | 1.74 s (±0) | 6.97 s (+0.3 %) | 2,157 tok/s (+0.5 %) | 24.2 / 44.5 / 47.5 ms | PPL 4.85, probes 3/3, prefix PASS, 31/48 identical, 14 near-tie, 3 soft, 0 failing; no win over QSA alone, kept off |
| + harvested TunableOp rows | 1.77 s (+1.7 %) | 7.08 s (+1.9 %) | 2,129 tok/s (-0.8 %) | 24.2 / 44.5 / 47.7 ms | PPL 4.82, probes 3/3, prefix PASS, 36/48 identical, 10 near-tie, 2 soft, 0 failing; neutral, rows are additive table entries |
| SHIP (all committed defaults together) | 1.54 s (-17.2 %) | 6.12 s (-18.2 %) | c1 2,377 / c6 2,476 / c8 2,487 agg (clean rerun; 16k c8 2,485) | 24.7 / 45.4 / 47.2 ms | PPL 4.81, probes 3/3, prefix PASS 16.7x, concurrency PASS (0 failing, 31/48 identical, 16 near-tie, 1 soft) |

The one greedy difference is a degenerate run of special tokens
(`248044 248045 248045 ...`): the churned run's own top-2 gap there is 0.20
nats, and the churn-vs-solo logit shift at that position is 0.5-0.7 nats with
the legacy kernel too (A0: `248046` -2.84 -> -2.31). The pipelined kernel
merely starts from a smaller gap (1.23 vs 2.76 nats), so the same shift flips
the token. No garbage, no logprob drop (continuation mean lp -0.19 vs -0.28),
PPL 4.82 -> 4.83.

Decode-only ABBA (`A0 / SHIP / SHIP / A0`, c1 + c6, no c8, one boot per arm,
same tree) settles the c1 ITL question: A0 read 24.7 / 25.1 ms at c1 and
46.0 / 46.6 ms at c6, SHIP read 24.5 / 24.9 ms at c1 and 44.1 / 44.9 ms at
c6. The committed defaults are equal-to-better on decode ITL, so the earlier
A1 22.7 ms c1 reading was box drift, not a QSA-pipe cost. The same run gives
SHIP +17-19 % c1 and +20 % c6 aggregate prefill over A0 (2,407/2,392 vs
2,061/2,003; 2,491/2,490 vs 2,083/2,063 tok/s). One SHIP boot hit a single
special-token churn flip (margin 1.016 nats, the degenerate `248045` run);
the other SHIP boot and both A0 boots passed 0 failing. c1/c6 were the clean
cells in that window. Every c8 cell after 11:26 there was invalid: PCIe AER
correctable events (41:00.0 Timeout, 42:04.0/42:0c.0 NonFatalErr, 40:03.1
BadTLP, links still 16 GT/s x16) and the CI docker build ran through it, and
both SHIP and A0b collapsed only in c8 (350/1,450 and 758/1,530 tok/s) while
c1/c6 stayed normal. A quiet-window rerun (`w1C8A0`/`w1C8SHIP`, c1, c6, c8
and 16k c8, one boot per arm, load and AER recorded at arm start/end) is
aer-clean on both arms and gives A0 c8 2,053 / 16k c8 2,018 tok/s (ITL 49.4 /
51.5 ms, TTFT 18.27 s) against SHIP c8 2,487 / 16k c8 2,485 tok/s (ITL 47.2 /
49.4 ms, TTFT 15.02 s): +21 %/+23 % aggregate prefill and equal-or-better
decode ITL at c8, PPL 4.89/4.86, probes 3/3, prefix PASS, concurrency PASS on
both. The SHIP arm's load rose to 34 during the c8 cells (a buildkitd docker
build), but its c1/c6 cells reproduce the ABBA (2,465/2,489 vs 2,407/2,491),
so the c8 cells are quoted.

### Investigate lane (HIP 6-9)

1. **Glue-kernel attribution**: `PROFILER=torch CG_MODE=NONE` boot (TP4),
   `kernel_sites.py` table of copy/fill/binary/index kernels per step by
   Python launch site; handed to the fix lane.
2. **`GPU_MAX_HW_QUEUES` 4 vs 2**: `flashnext-mtp2.env` sets 4, and the
   A/B boots on HIP 2-5 logged `Runlist is getting oversubscribed due to too
   many queues` on all four GPUs. AGENTS.md measured 4 as 10-30 % slower than
   2 on this hardware. **Result (ABBA, TP4, HIP 6-9, no `--monokernel`)**:
   at 2 queues, 16k/1k c=8 TTFT mean drops 20.0 → 17.7 s and per-request
   prefill rises 824 → 925 tok/s, at the same aggregate prefill (2065 vs
   2071 tok/s) and output. Decode loses 3-6 % in every cell (1k c=8
   aggregate output 216.2 → 205.1, ITL p50 +3 %, c=1 ITL 26.2 → 27.7 ms).
   The stall probe is unchanged, and neither arm logged oversubscription.
   Shipped for `flashnext-mtp2` by user choice (TTFT over decode) in
   `bbe36aab97`. `--monokernel` stays at 4 (the engine raises it; at 2 the
   persistent kernel skews TP ranks). **Monokernel 4 vs 2 (ABBA, HEAD
   `7860e9b4ec`)**: 2 queues cost 0.5-2.1% ITL p50 on average (c=1 24.4 vs
   23.9 ms) with prefill and stalls unchanged, and the skew is per boot:
   with in-serve timing marks over 10 boots, 0 of 4 four-queue boots were
   skewed, but 4 of 6 two-queue boots were (GDN core +9-11%, QSA and
   attention +6-8% on the slow ranks, c=1 ITL +3.6-5.5%); which ranks lose
   changes per boot. Monokernel keeps 4.
3. **Offload profile** (TP2 on HIP 6-7, K2 path, c=1, rocprofv3 window of
   224 steps plus a CPU-only torch trace). Step period p50 80.2 ms with the
   GPU busy 61%. Per MoE layer (48.3 per step): 838 us wall, 393 us GPU
   busy, 445 us bubble. Per step the bubbles add up to about 21 ms (26%).
   The gap between stage 1 and stage 2 of a layer (580 us mean) is 23 us
   for `route_ids.cpu()` to return, 69 us for the padding `.cpu()`, 435 us
   of host planning, `ensure` and copy issue after the drains, and 52 us
   other. It grows with expert misses: 321 us with none (53% of layers),
   1,930 us with 17 or more; fit 338 us + 64.5 us per host-to-device
   record. Hypotheses: the `route_ids.cpu()` drain is the serialization
   point (held), but the wait itself is short and the cost is the host work
   that follows; padding `.cpu()` held, small (about 3.3 ms per step); miss
   copies issued only after the drain held, and the cost is host issue time
   (about 65 us per record), not copy time; the map-upload synchronize is
   refuted (2 us). `expert_cache_copy=native` wedged the engine under
   rocprofv3, so its effect is not measured. Fix shipped (`898830cbd7`,
   receipts `e22be5c72f`, `002b1c1777`): the K2 expert pass is split, so
   resident experts run right after stage 1 while the missed experts load,
   then a masked pass covers the loaded ones; host planning is cheaper and
   copies go on one stream without per-pass events. TP2 MTP2 1024/512, two
   crossed pairs measured at the same time: c=1 decode 31.7 -> 36.7 tok/s
   per request (+15.7%), ITL p50 65.4 -> 56.4 ms, p99 107 -> 93.5 ms; c=4
   decode +6.9%, ITL p50 -5.4% (above the split's row gate, gain from the
   host work). Hit-only layers 536 -> 282 us, loading layers 1,284 ->
   1,172 us, stage-1-to-first-pass gap 774 -> 38 us. Greedy output and
   logprobs bit-identical in graph and eager mode; probes, PPL 4.84, prefix
   and concurrency pass. Loading layers are now bound by DMA (about
   566 us). Not measured: 16k, c=8, TP4 (the split fits at most 4 rows in
   TP4's 48 slots).
4. **Stall baseline and MAXBAT 4096** (finding 2): a clean baseline with
   6 decoders (fewer than `SEQS=8`, so injected prompts are not queued),
   including the reverse case (decoders arriving during a 16k prefill). It
   is the acceptance metric for the stall-free design. **MAXBAT 4096 (one
   arm)**: no prefill gain (16k c=8 aggregate prefill 1,965 vs 2,071 tok/s),
   the maximum decode gap grows 2.7-3x (1,423-1,558 vs 515-542 ms) and
   16k c=8 ITL p99 rises 548 -> 1,661 ms. 2048 stays.
5. **Glue attribution** (TP4, `CG_MODE=NONE`, torch trace with GPU kernels,
   baseline tree): a decode step is 2,605 kernels, of which 1,028 are glue
   (39% of launches, 4.6 ms of 59.4 ms GPU busy). Largest sites (kernels,
   us per step): shared expert 196 / 1,229; GDN `rearrange_mixed_qkv`
   144 / 925; `moe_align_block_size` expert-map gather and cast 96 / 651;
   GDN `forward_cuda` 108 / 369; MoE zero fills 96 / 217; HC fill 100 / 216.
   Batches 1-4 removed most of these. Mixed steps add 360 `index_select`
   kernels in GDN `_forward_core` (12.3 ms of about 490 ms); that is queued
   in the fix lane.

### Kernel map (analysis)

From the rocprofv3 traces: the ordered kernel sequence of one decode step,
segmented into prologue, the 48 layers by type, MTP draft passes and
epilogue, each kernel mapped to its Python call site and HIP source. Output:
a ranked list of fusions and removals with estimated savings (removed graph
nodes x about 4.5 us + execution time), plus a `py-spy` breakdown of host
Python time per engine step (scheduler, input preparation, graph launch,
sampling/MTP, output processing) for decode-only and mixed steps.

#### Kernel map results (2026-11-02)

Full tables (every kernel of every segment with duration, gap, grid and
source) are in `~/w4a8_runs/lanes/map/plan.md` on the box. Trace: RES256,
4 ranks, 217 decode-only steps per rank; the kernel sequences are identical
on all ranks.

Decode steps fall into classes by kernel count:

| Class | Steps | Period |
|---|---:|---|
| 2,803 kernels | 173 | 59-61 ms |
| 2,787 kernels | 15 | 53 ms |
| 2,392 kernels | 27 | 33.5-38.1 ms |

The 2,392-kernel class runs 100 fewer of each HC mix kernel (the fp32 fill,
`gemm_tn_splitk_k`, `hc_up_mix_prefill_k`) and 147 fewer split-K
`PostGSU4`: it is the M <= 8 HC decode path (`rdna_ops.py:28`,
`_DECODE_MAX = 8`). MTP2 verify batches of 18-24 tokens take the prefill
path instead. Measured split of the 21.4 ms difference (M <= 8 step 38.1 ms
vs verify step 59.5 ms): about 9.1 ms is inherent scaling with M (MoE q4,
all-reduce, QSA, GDN), about 8.8 ms is the dense GEMM family (not
separable from M scaling), HC 1.6 ms, glue 1.9 ms.

Anatomy of a 2,803-kernel step (48.1 ms of kernel execution, 13.2 ms of
gaps):

| Segment | Kernels | Exec | Gaps |
|---|---:|---:|---:|
| Post-sampler bookkeeping | 14 | 0.06 ms | 0.06 ms |
| MTP pass 1 (M=24, rocBLAS) | 86 | 3.05 ms | 0.41 ms |
| Inter-draft prep + MTP pass 2 (M=6, GEMV) | 77 | 1.93 ms | 0.36 ms |
| Next-step input prep (eager) | 96 | 0.32 ms | 0.45 ms |
| Main forward, 48 layers | 2,509 | 40.91 ms | 11.77 ms |
| Head mix + LM head + logits all-gather | 8 | 1.57 ms | 0.05 ms |
| Sampler / rejection | 13 | 0.24 ms | 0.06 ms |

The forward repeats every 4 layers as GDN, GDN, GDN, QSA. A GDN layer is
50 kernels (0.83 ms exec, 0.24 ms gaps); a QSA layer is 58 kernels
(0.92 ms, 0.27 ms). Only about 16 kernels per layer do model math (HC mix,
projections, conv, GDN or QSA attention, norm, all-reduce, router, MoE
GEMMs); the rest is glue.

Ranked candidates (savings per step are estimates: removed kernels x
4.5 us plus their measured execution time; not exactly additive):

| # | Candidate | Kernels/step | Est. saving | Effort |
|---:|---|---:|---:|---|
| 1 | ~~Raise `_DECODE_MAX` so dense GEMMs and HC use the GEMV path at M <= 24~~ **withdrawn**: the crossover was already measured (`b074874b54`, `layers/utils.py:353-356`); rocBLAS with TunableOp wins at M = 16-24. Only a new skinny-GEMM kernel that beats it in `gemv_crossover_bench.py` would help | - | - | - |
| 2 | GDN glue: `zeros`, 3x `contiguous`, `cat` + copies, 2x copyBuffer (`qwen_gdn_linear_attn.py`, `layernorm.py`) | -360 | about 2.5 ms | M |
| 3 | Shared-expert silu/mul/sigmoid/mul into one kernel (`rdna_ops.py`) | -196 | 1.6-1.8 ms | S |
| 4 | MoE expert-map index/cast glue (`moe_align_block_size.py`) | -98 | about 1.05 ms | S-M |
| 5 | Zero fills: HC (100, in progress), MoE output (96, zero inside the sort kernel), QSA (14) | -210 | about 1.2 ms | S each |
| 6 | Fuse `moe_align_block_size` + `count_and_sort` | -96 | about 0.69 ms | M |
| 7 | Split-K `PostGSU4` post-reduce (150) | -150 | about 0.87 ms | M |
| 8 | `gemma_rms_norm` f16/f32 triplets (28 calls) | -85 | about 0.54 ms | S-M |
| 9 | Inductor `triton_poi_*` glue (219) | -100 to -219 | 0.6-1.4 ms | M-L |
| 10 | Eager step prologue (110 kernels into 2-3) | about -95 | about 0.55 ms | M |
| 11 | Fuse `rdna_ar_oneshot` with `hc_combine_norm` | -99 | about 0.7 ms | L, multi-rank risk |

Also found: the GDN decode runs the Triton `_causal_conv1d_update_kernel`
although a HIP `causal_conv1d` kernel exists (the dispatch gate in
`causal_conv1d.py` does not take it), and the QSA scorer runs as Triton
because the HIP scorer rejects the shape (`ops/qsa.py`).

Host Python (py-spy, 20 s on a 16k/1k c=8 server, about 10-16% of samples
lost to read errors, no native frames):

- EngineCore spends 99.8% waiting for worker responses; scheduler and
  output processing are under 0.2% (about 0.5 ms per step, estimate).
- Worker rank 0 waits for the next RPC 87.2% of the time; `execute_model`
  is 10.1%, of which model-forward Python is 8.7% of worker time and
  attention-metadata build, sampling, MTP propose and input preparation are
  about 1% each.
- Host time never exceeds GPU time: mixed steps carry about 55 ms of worker
  Python (estimate) under about 525 ms of GPU work, and decode-only FULL
  steps are a graph launch. **Python is not on the critical path of FULL
  decode steps**. Eager execution is a different story: with graphs off,
  decode median ITL is 105.6 ms against 55.6-57.1 ms with
  FULL_AND_PIECEWISE (host-bound, about 1.9x). Mixed batches above the
  largest capture size (24 tokens) run eager.
- `BasevLLMParameter.__torch_function__` (`parameter.py:120`) runs 664
  times per eager step, but it is not a cost worth removing: its inclusive
  time (42.9 ms per step) is almost all the wrapped op itself
  (`F.linear` and similar); its self time is 0.73 ms per step, about
  1.1 us per call or 0.4% of eager `execute_model` (profiler-inflated upper
  bound).
- The other large Python costs are the same GDN glue as row 2, per-layer
  attention-metadata rebuilds, and the Triton conv launched from Python
  every GDN layer.
- Not yet measured: the API-server process, and how much of a mixed step's
  host time is exposed rather than hidden behind GPU work.

Kernel microbenchmarks from the fix lane (graph replay, per call):

| Kernel | M | Before | Fused | Per-step saving |
|---|---:|---:|---:|---:|
| Shared expert (49 per step) | 18 / 24 | 120.7 / 119.5 us | 44.0 / 52.3 us | 3.8 / 3.3 ms |
| HC mix (100 per step) | 18 / 24 | 90.0 / 92.3 us | 72.0 / 72.7 us | 1.8 / 2.0 ms |

Order handed to the fix lane: (1) HC fill removal, HC injection clones and
MoE expert-map glue (built, unit-tested), (2) shared-expert fused kernels
for M <= 32, (3) GDN glue (row 2), (4) rows 5-8, (5) rows 9 and 10, then
11.

#### Shipped decode batches (2026-11-02)

| Commit | Change |
|---|---|
| `b30757c257` | Batch 1: HC split-K without the fill, HC injection written by the op, shared expert fused to M <= 32, MoE align without the expert-map gather |
| `13fbe2f916` | Batch 2: GDN spec-verify glue (q/k/v, a/b and z read in place, direct output) |
| `5bda4c2450` | Batch 3: HC decode tile retune, `rdna_gemm_smallm` for 9-32 row GEMMs with weights up to 4 MiB |
| `27e5fa40da` | Batch 4: shared-expert gate_up tile, one-kernel Gemma norm |

rocprofv3, same load as the baseline trace (6 decoders + a 16k injector),
dominant decode step:

| | Baseline (`RES256`) | Batches 1-3 | Batches 1-4 |
|---|---:|---:|---:|
| Kernels per step | 2,803 | 1,823 | 1,739 (-38%) |
| Kernel busy time | 47.1 ms | 42.7 ms | 40.7 ms |
| Step period, median (profiled) | 60.5 ms | 51.3 ms | 49.5 ms |

Serve proof after batch 4 (TP4 flashnext-mtp2, unprofiled,
`GPU_MAX_HW_QUEUES=4`), against the 256 KB-gate baseline:

| Cell | ITL p50 | Aggregate output | Aggregate prefill | TTFT |
|---|---|---|---|---|
| 1k c=1 | 26.7 -> 23.5 ms (-12%) | - | 2,015 tok/s | p50 503 ms |
| 1k c=6 | 52.0 -> 45.0 ms (-13%) | 192.5 -> 205.3 (+7%) | 2,080 tok/s | p50 651 ms |
| 1k c=8 | 55.6 -> 48.2 ms (-13%) | 224.5 -> 239.9 (+7%) | 2,068 tok/s | p50 668 ms |
| 16k c=8 | 57.6 -> 49.4 ms (-14%) | 91.3 -> 90.1 (-1%) | 2,029 tok/s | - |

- Prefill is unchanged: aggregate prefill stays at 2,015-2,095 tok/s, and
  solo TTFT is 1.86 s (4k) and 7.48 s (16k) against 1.86 / 7.44 s.
- ITL p99 (about 500 ms) is unchanged: that is the mixed-step stall, which
  only the two-lane engine removes.
- Decode tok/s moves with MTP acceptance, which varies 0.1-0.3 between boots
  of identical code (16k c=8: 2.15 here vs 2.33 at baseline), so ITL p50 is
  the step-time signal.
- Probes 3/3, PPL 4.81, prefix cache 19.7x, concurrency gate pass, no faults.
- These numbers are at 4 queues. The recipe now uses 2, which costs 3-6%
  of decode (ABBA above).

Rejected after measurement: shared expert on an auxiliary stream inside the
graph (14.5-16.7 us per layer slower), HC K-loop pipelining, larger HC
tiles at M >= 18, merging MoE zero fills (aliasing risk for 0.28 ms).

Batch 5 (`815b16022c`): one HIP kernel for MoE routing plus the
accumulator zero fills (scratch sized at init from the largest capture size
times top-k), the shared-expert down projection as a split-K tile with the
gate in its epilogue, and GDN mixed steps slicing contiguous spec and
non-spec rows instead of `index_select`/`index_copy_`. Kernels per 24-token
decode step 1,739 -> 1,595; profiled period median 49.55 -> 47.65 ms.
Serve ITL p50 (c=1 / c=6 / c=8 / 16k c=8): 23.1 / 44.2 / 47.7 / 49.1 ms at
`GPU_MAX_HW_QUEUES=2`, 23.0 / 44.3 / 47.6 / 48.9 ms at 4; same-tree A/B
with the three new paths off vs on: -1.7% / -1.4% / -1.6% (c=1 / c=8 /
16k c=8). Prefill, TTFT, ITL p99 unchanged; probes, PPL 4.80-4.81, prefix
cache and concurrency pass. In this tree 2 and 4 queues differ by at most
0.4% on ITL.

All-reduce (`a359ddb4a8`): a 16-byte vector finisher for 2, 4 and 8 ranks
and retuned block counts (4 blocks up to 8 KB, 6 up to 32 KB, else 32); the
exchange protocol, flags and wedge detection are unchanged. Floor
measurements: one-way P2P flag latency 6.3 us, a 120 KB push to three peers
22-23 us (PCIe write rate), about 2.4 ms per step of rank-skew wait that no
kernel removes. In serving (rocprofv3, same tree with `VLLM_RDNA_AR_VEC=0`
as control) the 101 calls of 123 KB in a 24-token step drop 38.1 -> 32.1 us,
about 0.6 ms per step (1.3%); 15-31 KB calls are unchanged. Serve ITL p50
moved -0.4% to -2.3% (c=8 47.7 -> 46.6 ms, 16k c=8 49.1 -> 48.1 ms), inside
the boot-to-boot spread, so only the profiled gain is claimed. Probes, PPL
4.76, prefill, TTFT, ITL p99 and the stall probe are unchanged. Rejected:
per-chunk flags, 16-byte pushes, two-shot below about 256 KB, the `out=`
variant (0.12 ms per step), residual/norm fusion.

The decode list is essentially done (remaining items under 1% each).

#### Prefill profile (2026-11-03)

Solo 16k prefill, TP4 Flash-Next MTP2, rocprofv3 on all four ranks. A 16k
prompt runs as 7 chunks of 2048 tokens plus 2 of 1024. A 2048-token step
takes 974 ms (eager, 2,592 kernels, 13.5 ms of idle gaps), so prefill is
GPU-bound; piecewise graphs and host gaps are not levers.

| Family (2048-token step, rank 0) | ms | % |
|---|---:|---:|
| MoE int4 GEMM (tile 4) | 208.6 | 21.4 |
| Dense GEMM (rocBLAS) | 168.4 | 17.3 |
| RCCL all-reduce | 167.9 | 17.2 |
| HC (down, up-mix, combine) | 157.5 | 16.2 |
| QSA attention | 142.8 | 14.7 |
| GDN | 58.7 | 6.0 |
| MTP layer MoE | 17.6 | 1.8 |
| Other | ~39 | 4 |

The all-reduce time is mostly waiting, not transfer. Matching the 5,048
RCCL calls across ranks, the collective's own critical path is 650 us per
call at 2048 tokens (67.6 ms per step, about the PCIe Gen4 x16 rate); the
other ~113 ms is ranks waiting for the slowest one, 90 ms of it after MoE.
The cause is expert-parallel routing imbalance: the busiest rank's MoE GEMM
is 1.11-2.04x the rank mean per layer (median 1.39x), and 36 of 48 layers
have the same slowest rank in 70% or more of steps. The GPUs themselves are
matched (dense, HC, GDN within 2%). A custom prefill all-reduce is
therefore not worth doing.

Ranked fixes (estimated savings per 974 ms step):

1. MoE tile 8 from 1,024 tokens (today it starts at 4,096, which MAXBAT
   2048 never reaches): 35-45 ms.
2. W4A8 sdot4 MoE at prefill (existing opt-in op, kernel 1.75x vs tile-8
   W4A16 at 2048 tokens): a further 40-55 ms, with a quality gate.
3. QSA sparse-attention kernel (BLOCK_N 16 knob, then prefetch): 30-60 ms.
4. Static expert placement to balance ranks: up to 30-48 ms (in-sample
   bound).
5. HC down-GEMM grid (96 blocks on 72 CUs at 2048 tokens): 8-27 ms.
6. GDN `delta_h_packed` on 48 blocks: about 10 ms.
7. TunableOp rows for 35 missing prefill GEMM shapes: up to 10 ms.
8. MTP layer MoE at prefill: up to 10 ms.

If 1-6 land, solo 16k TTFT would go from 7.48 s to about 5.9-6.5 s
(estimate; items overlap). Next in the fix lane: these in order, each with
the full serve proof.

#### Prefill lever arms (2026-11-03, HIP 2-5, TP4 Flash-Next MTP2)

One boot per arm, pipelined QSA + W4A8 SDOT4 on both sides, load and PCIe
AER recorded at arm start/end. A vllm-image CI docker build ran throughout
(load 11-27, clang-23/ninja), so only within-window deltas are quoted.

| arm | c1 agg | c6 agg | 16k c8 agg | TTFT c8 | ITL c1/c6/c8 | quality |
|---|---:|---:|---:|---:|---|---|
| P0 baseline | 2,434.7 | 2,523.5 | 2,500.0 | 15.10 s | 24.3/44.2/48.7 ms | PPL 4.85, probes 3/3, prefix PASS, concurrency PASS |
| P1 static placement | 2,406.9 | 2,512.2 | 2,506.6 | 14.57 s | 24.3/44.2/49.3 ms | PPL 4.87, all PASS |
| V1 v2 build, shipped kernel | 2,440.1 | 2,532.1 | 2,507.0 | 14.36 s | 23.7/43.3/48.3 ms | PPL 4.84, all PASS (AER +2) |
| V2 v2 kernel, tile 8 + K=-1 | 2,639.5 | 2,744.2 | 2,722.7 | 12.83 s | 22.6/42.4/46.9 ms | PPL 4.80, all PASS (AER clean) |
| A2 tile 8 from 1,024 | 2,089.6 | 2,232.6 | 2,227.6 | 19.48 s | 24.2/44.8/49.0 ms | PPL 4.85, all PASS |

Verdicts:

1. Static placement: within noise against linear (c1 -1.1 %, c6 -0.4 %,
   16k c8 +0.3 %, ITL equal). It was applied on 48 decoder layers (the MTP
   layer stayed linear by design). Kept opt-in.
2. W4A8 v2 at tile 8: +8.2 %/+8.4 %/+8.6 % prefill (c1/c6/16k c8), TTFT -1.5
   s, decode ITL 1 ms better, PPL 4.80. Shipped as the `flashnext-mtp2` recipe
   default (`VLLM_RDNA2_W4A8_BLOCK_M=8`, `VLLM_RDNA_W4A8_KBLOCK=-1`).
3. MoE tile 8 from 1,024 tokens (`VLLM_RDNA2_MOE_TILE8_MIN_TOKENS=1024`): a
   clear loss (c1 -14 %, 16k c8 -11 %), so 4,096 stands.

### Stall-free mixed prefill/decode (design)

Today a decode token waits for the whole step that carries a prefill chunk
(about 0.5 s per 1,024-token chunk, 7.2 s in the reverse case). Two
architectures remove the wait instead of rationing prefill:

| Option | Mechanism | Open questions |
|---|---|---|
| A. In-instance split streams | Each step splits into a decode sub-batch (FULL graph, high-priority stream) and a prefill chunk (low-priority stream) running concurrently; decode outputs return without waiting for prefill | How far decode slows beside prefill on gfx1030 (stream priority, CU masking); two TP collectives in flight at once; scheduler, V2 runner and async output changes |
| B. Prefill/decode disaggregation | 4 GPUs prefill, 4 decode; KV, GDN/mamba and QSA state move over PCIe P2P | Transfer cost per 16k prompt; capacity of a 4+4 split; connector support on ROCm |

The design lane maps the current mixed-step flow, inventories what this
vLLM tree already has (dual-batch overlap, micro-batching, KV connectors),
and measures concurrent priority-stream decode against prefill-sized GEMMs
on gfx1030, with collectives in flight on both streams, before either
option is built.

#### Design study results (2026-11-02)

Full document: `~/w4a8_runs/lanes/async/design.md` on the box, with
file:line detail in `parts/*.md`.

Why a mixed step blocks decode:

- It is GPU-bound, not host-bound: 491 ms with 473 ms of kernel time (96%
  busy) for a 1,024-token chunk against 62 ms for a decode step (rocprofv3,
  rank 2). About 424 ms of extra work scales with prefill tokens.
- The chunk is 1,024 tokens: the 2,048 budget minus 6 decoders x 3 tokens
  leaves 2,030, floored to the 1,024-token mamba block
  (`scheduler.py:721-732`).
- A 1,030-token mixed batch is larger than every capture size
  (`[3,6,12,24]`, `rdna_launcher_common.sh:134-139`), so it runs with
  `CUDAGraphMode.NONE`, fully eager: 3,776 kernels in the traced mixed step
  (`cudagraph_utils.py:602-620`, `model_runner.py:2184-2186`; inferred
  from code and kernel count, not seen in a log).
- Decode rows share one stream, one layer loop and one set of GEMMs, MoE,
  QSA and all-reduces with the prefill rows, and one async output per step:
  a decoder's token is only available when the whole step ends.

What this tree already has:

- Dual-batch overlap is unusable: it needs DP > 1 and rejects speculative
  decoding, hybrid models and PIECEWISE graphs
  (`config/vllm.py:3263-3280`, `gpu/ubatch_utils.py:545-565`).
- No stream priority on the compute path (only the KV-offload copy
  streams).
- No `P2pNcclConnector` or shared-storage connector in this tree; Nixl is
  the only candidate (GDN state handling, ROCm code, UCX over RDNA PCIe
  unverified), and the QSA layer reports `supports_kv_connector() False`
  (`qsa.py:239-240`), so no stock connector can move this model's state.

Disaggregation estimate: a 16k prompt is 249.1 MiB of state per rank
(996.5 MiB over four ranks, 540.8 MiB unique). At 5-20 GB/s that is
13-52 ms per rank, or 52-210 ms through one shared uplink (estimate; the
measured P2P rate below is about 28 GB/s), under 2% of a 9-11 s prefill. Bandwidth is
not the obstacle. Capacity is: weights are about 19 GiB per rank at TP4
and about 37 GiB at TP2, which does not fit a 32 GiB V620, so a split is
TP4 prefill plus TP4 decode on all eight GPUs, with one group mostly idle
for most workloads (about half the throughput of two mixed TP4 instances,
estimate).

Provisional ranking:

| # | Option | Expected effect | Work | Main risks |
|---:|---|---|---|---|
| 1 | Two-lane engine on the same TP4 GPUs: decode lane (FULL graph, CU-masked stream), prefill lane (second stream on the remaining CUs, own communicator) | Decode ITL about 60-80 ms during prefill vs 425-548 ms today; prefill about 1,600 tok/s with 24 CUs reserved vs about 2,400 solo (estimates) | Second execution path in the V2 runner, two-pool scheduler, per-lane communicators; about 4-6 weeks (estimate) | Whether graph replay honours a CU mask; ordering of two communicators across ranks |
| 2 | TP4 prefill + TP4 decode (disaggregation) | No stall by construction | New connector for QSA, MTP and HC state; uses all 8 GPUs; about 6-10 weeks (estimate) | Halves per-phase capacity |
| 3 | Cheaper mixed steps (larger capture sizes, all-reduce gate) | Shorter mixed steps; does not remove the wait | 1-2 weeks | none major |
| 4 | Dual-batch overlap | not viable here | | |

Decision rule: option 1 if a CU-masked decode graph stays within about
1.3x of its solo latency while prefill keeps at least 55% of solo
throughput, and two communicators in flight do not deadlock; otherwise
option 2.

#### Feasibility microbenchmarks (emulated kernels, HIP 8-9)

Workloads: an emulated decode graph of 2,544 nodes (65.3 ms alone, 62.5 ms
with real `rdna_ar`; the real graph is 56-62 ms) and a stream of rocBLAS
GEMM units at M=1024 standing in for prefill. Raw results are in
`~/w4a8_runs/lanes/async/results/` on the box.

- PCIe: 27.9 GB/s P2P inside a PLX switch, 28.2 GB/s to the host; two GPUs
  streaming to the host together share 28.2 GB/s. A 16k request's state
  would move in roughly 40-75 ms (under 1% of its prefill).
- **Stream priority alone does nothing**: decode beside prefill is
  3.2-3.4x slower with or without it.
- **CU masking works when sized in WGPs.** The V620 exposes 36 mask bits of
  2 CUs each; bits above 35 are silently ignored. Graph replay honours the
  mask (replay tracks eager launches at every mask size). Masks only
  isolate when both lanes are masked and the streams are created once in a
  fresh process; a process with many leaked streams degraded them badly.

Single GPU, fresh process per arm, `GPU_MAX_HW_QUEUES=4`:

| Decode WGPs (CUs) | Decode vs solo | Prefill vs solo | Gate |
|---|---:|---:|---|
| 12 (24) | 1.34x | 75% | fail |
| 14 (28) | 1.24x | 69% | pass |
| 16 (32) | 1.24x | 67% | pass |
| 18 (36) | 1.17x | 62% | pass |
| 20 (40) | 1.19x | 60% | pass |
| 24 (48) | 1.10x | 48% | fail |

Unmasked, decode is 3.25x slower.

Two GPUs, real collectives (one `rdna_ar` handle in the decode graph; a
second `rdna_ar` handle and a second RCCL communicator for the prefill
lane): no deadlock, timeout or wrong result in 37 arm runs. At 16 decode
WGPs, decode takes 94 ms against 198 ms unmasked, and prefill keeps 58-59%.
The 94 ms decomposes as:

| Step | Decode |
|---|---:|
| Solo, high-priority stream | 62.5 ms |
| Normal-priority stream (masked or not, no prefill) | 83.3 ms |
| With the concurrent prefill lane | 94.1 ms |

The prefill lane's collective type (none, `rdna_ar`, RCCL) moved decode by
only about 0.6 ms at 16 WGPs. Against the normal-priority solo time the
result is 1.14x (pass); against the high-priority solo time it is 1.51x
(fail). A masked stream cannot carry a priority
(`hipExtStreamCreateWithCUMask` has no priority argument), and the real
decode graph already runs on a normal-priority stream in 56-62 ms, so the
priority cost is probably an artifact of the emulated graph.

Verdict: option 1 stays first, conditionally. Before building it:

1. A green-context probe (about 10 minutes): the HIP runtime here exports
   `hipGreenCtxCreate`, `hipDevSmResourceSplitByCount` and
   `hipExecutionCtxStreamCreate(stream, ctx, flags, priority)`, which
   combine a CU partition with a stream priority (estimate if it works:
   about 1.18x).
2. A real-kernel spike (2-3 days, TP4): the real decode FULL graph on the
   decode partition against a real 1,024-token prefill chunk on the
   complement, with the same gate (decode within 1.3x of the same graph
   alone on a normal-priority stream, prefill at least 55%), and a check
   of TP4 rank skew.

Option 1 also needs one engine batch queue per lane: the current queue is
FIFO (`core.py:795,810,815`), so a decode batch behind a prefill batch
cannot be released early. If the spike fails or decode needs more than
about 20 WGPs, option 2 (disaggregation) is the fallback.

Not yet measured: real-kernel CU scaling, the green-context API on
gfx1030, TP4 skew, RCCL-only interference, repeats per cell (each cell is
one run of 40 replays), and why `GPU_MAX_HW_QUEUES=8` slowed the TP
baselines (decode 114 ms vs 63 ms; keep it at 4 or lower).

#### Green-context probe and queue-count check (HIP 8-9, emulated kernels)

Full tables: `~/w4a8_runs/lanes/async/results/GCTX_REPORT.md` on the box.

- The HIP green-context API works on gfx1030 (72 partitionable CUs, minimum
  partition 2). Decode and prefill on disjoint partitions ran concurrently
  in every size tested.
- **The "priority cost" was not priority.** With the arm order reversed,
  whichever stream a process creates third runs about 10 ms slower (low
  62.3, normal 62.3, high 72.7 ms when high runs third). The slowdown
  tracks the number of KFD queues in the process (8 to 9), and runlist
  oversubscription messages appear on both GPUs during the two-GPU runs.
  Stream count and `GPU_MAX_HW_QUEUES` matter, priority does not.
- With `GPU_MAX_HW_QUEUES=2` and the serving HSA environment, both
  partitioning methods pass the gate on two GPUs with real collectives
  (prefill lane on RCCL):

| Lanes (hwq 2) | Decode | vs solo (63.4 ms) | Prefill vs solo (186-207 units/s) |
|---|---:|---:|---:|
| WGP mask, 16 WGPs decode | 75.2 ms | 1.19x | 70-78% |
| Green context, 32 CUs decode | 74.2 ms | 1.17x | 70-78% |
| Same lanes at hwq 4 | 84.8-86.5 ms | 1.34-1.37x | 64-71% |

No mismatches or timeouts in any arm. The two-lane design is feasible on
emulated kernels at hwq 2; the real-kernel TP4 spike is next in the queue.

#### Real-kernel spike (TP4, HIP 6-9, `GPU_MAX_HW_QUEUES=2`)

A worker extension (outside the vLLM tree) replays the real 24-token decode
FULL graph on a CU-masked stream while a real 1,024-token prefill chunk runs
eagerly on the complement with its own communicator. Full-GPU references:
decode graph 39.3-39.6 ms, prefill chunk 442-497 ms. Raw data:
`~/w4a8_runs/lanes/async/spike/RESULTS.md` on the box.

| Decode / prefill WGPs | Decode alone -> concurrent | x full | Prefill chunk alone -> concurrent | Share of full | Gate |
|---|---|---:|---|---:|---|
| 24 / 12 | 43.4 -> 49.0 ms | 1.24 | 1,025 -> 1,103 ms | 40% | fail (prefill) |
| 20 / 16 | 45.0 -> 52.7 ms | 1.33 | 797 -> 862 ms | 52% | fail (both, narrowly) |
| 18 / 18 | 48.9 -> 57.0 ms | 1.45 | 795 -> 916 ms | 53% | fail |
| 16 / 20 | 49.3-49.5 -> 58.1-60.3 ms | 1.48-1.53 | 693-724 -> 714-875 ms | 51-68% | fail (decode) |

- No split met both gates; 20 / 16 misses by 0.03x and 3 points, inside
  the boot-to-boot spread. No deadlock in six two-lane runs; the four ranks
  agree within 0.03 ms (decode) and 0.5 ms (prefill).
- Against today: the decode forward under a prefill chunk takes 49-60 ms
  instead of waiting 425-548 ms (a whole decode step with sampler and MTP
  draft about 66-83 ms, estimate). Prefill chunks take 714-1,103 ms
  instead of 442-497 ms, so a 16k prompt under decode load would take
  about 11-18 s instead of 7-8 s (estimate).
- Rules found: create each lane's masked streams once at start-up and never
  destroy them (an all-reduce after a stream destruction killed a worker
  with "invalid resource handle"; a seventh masked-stream creation hung
  RCCL); stay at 8 KFD queues or fewer (at 9 the full-GPU decode graph ran
  30% slower). Stock RCCL uses 2 channels here, so channel caps change
  nothing.
- Logits agreement is not established: the dummy decode batch is not
  bit-reproducible even between two plain replays, so correctness needs
  real prompts through the engine.

Queue order: the planned decode fixes (HC/MoE batch, shared expert, GDN
glue), the remaining probes (offload profile, attribution, queue A/B,
MAXBAT 4096) and the green-context probe come first. The two-lane engine
build (real-kernel TP4 spike, then the worker and engine slices) is queued
after them.

Results are added to this report as each item lands.
