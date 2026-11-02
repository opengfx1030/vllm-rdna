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

A decode stream that shares a step with a 2,048-token chunk waits up to
about 0.5 s for its next token. The injector itself saw TTFT 8.9-11.2 s for
16k tokens and median ITL about 54 ms. Mitigation knobs already exist
(`--prefill-schedule-interval`, `VLLM_RDNA_DECODE_STALL_MS`, a smaller
`MAXBAT`, `--long-prefill-token-threshold`). Each one trades prefill
throughput for decode latency, so the choice belongs to the user and will be
made from measured ITL vs TTFT, not changed by default.

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
**The unprofiled serve A/B is the source of truth** (cells 1k/512 c=1, 1k/512
c=8, 16k/1k c=8, plus the mixed-batch stall probe; 64 KB vs 256 KB).

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

The eight V620s are split into two lanes so fixes and the next
investigation run at the same time. Each lane is one agent. Server boots
across lanes are serialized with a lock on the box
(`flock ~/w4a8_runs/boot.lock`, held from launch until the server answers
`/v1/models`), because two concurrent boots stalled earlier; serving in
parallel is fine.

| Lane | GPUs | Owner | Work, in order |
|---|---|---|---|
| Fix | HIP 2-5 (TP4) | agent `FixLane` | (1) all-reduce gate, (2) HC accumulator zeros, (3) glue kernels from the attribution |
| Investigate | HIP 6-9 (TP4 / TP2) | agent `ProbeLane` | (1) glue-kernel attribution, (2) `GPU_MAX_HW_QUEUES` 4 vs 2, (3) offload profile, (4) prefill-stall trade-off |

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
3. **Glue kernels**: fuse or remove the largest copy/fill/binary groups that
   the investigate lane attributes, HIP where it fits.

Every fix: correctness probes (greedy 9/9, 0 NaN, prefix-cache hits),
serve bench (prefill, decode, TTFT, ITL) vs the same-day baseline, and a
rocprofv3 per-step comparison. Ship only on a win.

### Investigate lane (HIP 6-9)

1. **Glue-kernel attribution**: `PROFILER=torch CG_MODE=NONE` boot (TP4),
   `kernel_sites.py` table of copy/fill/binary/index kernels per step by
   Python launch site; handed to the fix lane.
2. **`GPU_MAX_HW_QUEUES` 4 vs 2**: `flashnext-mtp2.env` sets 4, and the
   A/B boots on HIP 2-5 logged `Runlist is getting oversubscribed due to too
   many queues` on all four GPUs. AGENTS.md measured 4 as 10-30 % slower than
   2 on this hardware. Serve A/B with the stall probe; if 2 wins, the fix
   lane ships the recipe change.
3. **Offload profile** (TP2 on HIP 6-7): rocprofv3 under the same load;
   confirm or refute audit hypotheses 2-5 (per-MoE-layer `route_ids.cpu()`
   drain, padding `.cpu()`, miss-copy ordering, map-upload sync), with
   bubbles per MoE layer.
4. **Prefill-stall trade-off** (finding 2): decode ITL p50/p99 vs injector
   TTFT and prefill tok/s for `PREFILL_INTERVAL`, `VLLM_RDNA_DECODE_STALL_MS`,
   `MAXBAT` 1024/2048 and `LPTH`. Numbers for a user decision; no default
   change.

Results are added to this report as each item lands.
