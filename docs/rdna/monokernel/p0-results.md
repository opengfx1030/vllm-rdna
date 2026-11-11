# MonoKernel P0 results (V620 / gfx1030, 2026-10-14)

Measured answers to [plan.md](plan.md) P0. Box: par1-cs25, Radeon PRO V620
(gfx1030), HIP 7.14.60850 (`/opt/rocm/core-7.14`), HIP devices 2 (probe) and
2,3,4,5 (serve, one PLX switch). Raw logs are on the box under
`~/w4a8_runs/monokernel-p0/` and `~/vllm-rdna-0.31.0-mono/tools/monokernel-p0-probe/logs/`,
and copied to the infra repo under `bench_results/2026-10-14_monokernel-p0/`.

## P0a — standalone probe

### Build

`tools/monokernel-p0-probe/build.sh` with `ROCM_PATH=/opt/rocm/core-7.14`.
One compile error on the first build (`_exit` undeclared: missing
`<unistd.h>`); otherwise only `[[nodiscard]]` warnings. Probe changes, design
unchanged:

- `#include <unistd.h>`.
- Clear `hipGetLastError()` before every plain launch. A refused cooperative
  launch leaves `hipErrorCooperativeLaunchTooLarge` as the last error, and the
  original code then reported the *next plain* launch (graph capture, busy
  kernel) as failed. Run 1 (`logs/p0-20261014-075958.log`, WGP section) shows
  that false failure.
- Measured residency: every WG stores `wall_clock64()` at entry/exit and a
  `done` counter; `[direct]` prints `peak_coresident` and `done=N/G`.
- `--inject-stall P:W` (WG W stops publishing at phase P) to exercise the
  bounded wait and the sticky record deterministically.
- `run_stress.sh`: oversubscribed grids, cooperative-launch limit, injected
  stall, and a busy kernel that fills every wave slot.

Logs: `logs/p0-20261014-080456.log` (run.sh), `logs/p0-stress-20261014-080503.log`
(run_stress.sh).

### Code-object metadata (both builds identical except the mode bit)

| Kernel | `.amdhsa_workgroup_processor_mode` wgp / cumode | VGPR | SGPR (`.sgpr_count`) | static LDS | scratch | spills | compiler occupancy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `flag_kernel<false>` (plain) | 1 / 0 | 13 | 38 | 4 B | 0 | 0 | 16 waves/SIMD |
| `flag_kernel<true>` (coop) | 1 / 0 | 13 | 40 | 4 B | 0 | 0 | 16 |
| `busy_kernel` | 1 / 0 | 2 | 12 | 0 | 0 | 0 | 16 |

`.amdhsa_float_denorm_mode_32 = 3`, `_16_64 = 3`, `ieee_mode 1`, `dx10_clamp 1`
in both builds. Device: `CUs=36` (HIP reports WGPs), `sharedMemPerBlock=65536`,
`coopLaunch=1`, wall clock 100 MHz, `hostNativeAtomics=1`.

### Residency (grid × 256 threads; dynamic LDS = 65536 − 4 static unless noted)

| Case | WGP build | CU-mode build |
| --- | --- | --- |
| Occupancy API | 1 block/"CU" × 36 = 36 (INSUFFICIENT) | same, 36 |
| Measured peak co-resident, grid 72 @ 64 KiB | **72** (all 72 overlap, start skew 0.000 ms) | **72** |
| grid 73 / 108 / 144 @ 64 KiB | peak 72 (rest dispatched as WGs retire) | peak 72 |
| grid 144 @ 32 KiB (+4 B static) | peak **108** (3/WGP: allocation rounds up past 32 KiB) | peak 72 |
| grid 144 @ 16 KiB | peak 144 | peak 144 |

The occupancy API under-reports by 2× in WGP mode (it counts 36 units with
64 KiB each; a WGP has 128 KiB). The hardware holds 72 × 64 KiB WGs in both
modes.

### Graph capture: plain vs cooperative launch

| | WGP build | CU-mode build |
| --- | --- | --- |
| Direct plain, grid 72 | PASS, 1.08 ms | PASS, 1.08 ms |
| Direct coop, grid 72 | **refused: `hipErrorCooperativeLaunchTooLarge`** | PASS, 1.06 ms |
| Graph plain (memset + memset + kernel, 3 nodes), 100 replays | PASS, 1.074 ms/replay | PASS, 1.073 ms/replay |
| Graph coop, 100 replays | capture fails (same refusal at capture) | PASS, 1.084 ms/replay |
| Coop at grid 144 @ 16 KiB | refused | PASS (peak 144) |

ms/replay includes a host `hipStreamQuery` poll per replay; the kernel itself
spans 0.26–0.37 ms (16 phases). The cooperative launch's admission check uses
the same 1-per-WGP count as the occupancy API, so **cooperative launch only
works with `-mcumode`** at this LDS size. In CU mode a cooperative kernel
captures into a hipGraph and replays like a plain one. A direct (non-graph)
cooperative launch blocked the host for ~23 ms the first time in a process
(progress test, flag-first: busy launched at 43–44 ms instead of 20 ms).

### Forward progress

| Case | WGP plain | WGP coop | CU plain | CU coop |
| --- | --- | --- | --- | --- |
| run.sh default (busy 144 WGs × 500 ms), busy-first | PASS, flag done 1.1 ms after launch | refused | PASS | PASS |
| run.sh default, flag-first | PASS | refused | PASS | PASS |
| busy 576 WGs (fills all wave slots) × 3000 ms, wait bound 500 ms, busy-first | PASS, flag done at **6000.8 ms** | refused | PASS, 6001.3 ms | PASS, 6000.5 ms |
| same, flag-first | PASS, 21.2 ms | refused | PASS | PASS |

No partial residency was observed in any order: when the busy kernel holds
the wave slots, the dispatcher starts *none* of the flag kernel's WGs until
the busy grid has fully drained (both 3 s rounds), then starts all 72
together. So the ring protocol never deadlocked, but the flag kernel was
**starved for 6 s**. The device-side bound cannot see that wait (it starts at
WG entry), so only the host backstop covers it.

### Bounded wait and sticky record (`--inject-stall 5:17`, wait bound 100 ms)

| | Result |
| --- | --- |
| Direct plain (both builds) | FAIL after **100.6 / 100.3 ms**; record `phase=5 wg=16 waiting_on=17 observed=5 expected=6` (exactly the expected one); `done=60/72`, all 72 co-resident; kernel exits, no hang |
| Graph plain replay 0 (WGP) | sticky error at replay 0, same record; replay loop stops |
| Direct coop (CU mode) | **hangs** in `grid.sync()` (the bailing WGs skip it); host backstop fires at 5000 ms, record intact, process exits 3 without teardown |

The system-scope CAS into coherent host memory works (`hostNativeAtomics=1`),
first reporter wins, and every other WG bails on seeing the record. The
README caveat holds: `grid.sync()` is unbounded, so one bailing WG hangs a
cooperative kernel.

### P0a conclusions

1. 72 WGs × 64 KiB LDS × 256 threads are co-resident on a V620 in both WGP
   and CU mode (device timestamps), even though the occupancy API says 36.
2. Persistent kernels capture and replay in hipGraph as plain launches (both
   modes) and as cooperative launches (CU mode only).
3. The phase-id sticky record and bounded agent-scope waits behave as
   designed for the plain launch. For 2a: use the plain launch with
   bounded flag barriers, not `grid.sync()` (unbounded, and coop needs
   `-mcumode`).
4. A persistent kernel can be starved, not deadlocked, by a concurrent
   wave-filling kernel. In a captured decode graph on one stream nothing runs
   concurrently, but any side-stream work (all-reduce helper, async copies)
   would delay it; keep the mono step on one stream.
5. Keep static + dynamic LDS at a power-of-two share (64 KiB or 32 KiB
   exactly); 4 extra static bytes cost a quarter of the residency at 32 KiB.

## P0b — Model Runner V2, live serve

### Setup

Tree `~/vllm-rdna-0.31.0-mono` = `rdna_extras` @ `15e9e7adbd` (+ these docs)
over the native build of `~/vllm-rdna-0.31.0-pp3` (`_rocm_C.abi3.so` of
2026-10-11 19:16; the only later `csrc` change, `f2cc25ca92`, is the W4A8 MoE
workspace and is not on this W4A16 path). Venv `venv-7.14.0_0.31.0-final`.
Recipe `flashnext-mtp2` (TP4/EP4, MTP=2, `FULL_AND_PIECEWISE`, V2 runner,
`MAXLEN=262144`, `MAXBAT=2048`, `SEQS=8`, prefix caching, mamba-align,
`VLLM_RDNA_AR=1` one-shot ≤ 64 KiB), INT4 PLE
(`PLE=int4 PLE_QUANT_DIR=…/Qwen3.8-Flash-Next-PLE-quant/…/ples_int4`),
`GPUS=2,3,4,5`, port 18790. Driver: `tools/monokernel-p0-probe/serve_p0b.sh`
over `tools/rdna/port_v031/serve_validate.sh`.

Runs (all under `~/w4a8_runs/monokernel-p0/`, copied to
`bench_results/2026-10-14_monokernel-p0/`):

| Run | What | Result |
| --- | --- | --- |
| `serve-mono-p0b` | production serve, gates + the four bench cells | all gates PASS; baseline below |
| `serve-mono-p0b-graphs` | production serve + `graphcount_shim` (kernel nodes of every captured graph) + `VLLM_RDNA_PHASE_TIMING=50` (CUDA-event step phases), c=1 / c=8 decode windows | used for launch counts and step time |
| `serve-mono-p0b-tp-eager` | same recipe with `CG_MODE=NONE`, torch profiler with GPU activities | used for the per-kernel time split |
| `serve-mono-p0b-tp`, `-tp-rccl`, `-rp` | kernel tracing on the graph-mode serve (kineto; kineto with RCCL; rocprofiler-sdk tool) | **all three hang the engine** (see Risks) |

### Baseline (production serve, `serve-mono-p0b/summary.txt`)

Server default sampling (no `--temperature 0`), `vllm bench serve` random
dataset, seed 12345. MTP=2: c=1 decodes at width 3, c=8 at width 24.

| Cell | Prefill tok/s/req (agg) | Decode tok/s/req (agg out) | TTFT | ITL med / mean |
| --- | --- | --- | --- | --- |
| 1k/512 c=1 | 2065.1 (2005.0) | 70.74 (66.32) | 0.50 s | 29.5 / 29.6 ms |
| 1k/512 c=8 | 960.8 (2035.4) | 28.01 (202.44) | 1.07 s | 62.3 / 73.4 ms |
| 16k/1k c=1 | 2027.2 (2023.2) | 75.60 (47.37) | 8.08 s | 29.8 / 29.8 ms |
| 16k/1k c=8 | 1242.7 (3540.3) | 13.97 (92.51) | 13.18 s | 63.7 / 162.8 ms |

Gates: greedy probe 3/3, PPL 4.86, prefix TTFT 9.75 → 0.72 s, concurrency
probe PASS (48 checks, 0 failing), probe after load PASS, 0 fault markers in
`serve.log`. These match the recent TP4 MTP2 runs (e.g.
`bench_results/2026-10-14_asym-mtpaccept`: 1k c=8 decode 28.8, 16k c=8 14.0).

### Answers

1. **Flash-Next serves under V2 at TP4.** Every worker logs `Using V2 Model
   Runner`; boot, all gates and all four cells pass (above).
2. **Captured widths are exactly 3, 6, 12, 24.** Config
   `cudagraph_capture_sizes: [3, 6, 12, 24]`, `FULL_AND_PIECEWISE`. Per rank
   the V2 runner captured 4 FULL target graphs and 4 × 53 PIECEWISE target
   graphs, plus 4 FULL drafter "prefill" graphs and 3 FULL drafter decode
   graphs (`graph_kernel_counts.txt`). The decode steps replay the FULL graph:
   the phase timer reports `forward@cpu = 0.26 ms` for a 23.8 ms GPU forward at
   width 3 (`phase_decode.txt`), the drafter phases are tagged
   `dprefill_FULL`, and the CPU-side trace of a decode step holds no model ops
   (0 `aten::mm`, one compiled region; `serve-mono-p0b/torchprof-cpu-only`).
   Only widths 3 (c=1) and 24 (c=8) were exercised at runtime.
3. **GDN / mamba-align do not refuse V2.** `--mamba-cache-mode align` is on,
   the workers pad the mamba page (`Padding mamba page size by 29.13%`), no
   refusal or fallback message appears, and the hybrid model serves.
4. **Launches per decode step** (kernel nodes of the captured graphs, rank 0):

   | Graph | Kernel nodes |
   | --- | --- |
   | target FULL, width 3 and 6 | 2523 |
   | target FULL, width 12 and 24 | 2908–2909 |
   | drafter FULL "prefill" pass | 88–100 |
   | drafter FULL decode pass | 86 |

   Widths 3/6 use the M ≤ 8 fused decode glue (`rdna_fused_glue.cu`);
   widths 12/24 fall back to the unfused path (Tensile GEMMs, split-K, extra
   copies), hence ~390 more kernels. (Width-to-graph mapping inferred from the
   descending capture order and the M ≤ 8 cut.) A c=1 step is therefore
   ≈ 2,700 GPU launches (target 2523 + drafter ~175 + a few eager sampling
   ops); the eager trace counts 2,656 kernels for the target forward plus
   logits.

   Per layer at width 3 (eager trace, split at the all-reduce kernels,
   `analysis-c1-rank0.txt`): a GDN layer is 23 (GDN block) + 1 (AR) + 21 (FFN
   block) + 1 (AR) = **46** launches, a QSA layer 46 + 1 + 20 + 1 = **68**.
   36 × 46 + 12 × 68 = 2472, plus embedding / PLE / final HC = 2523.
5. **This port does not need anything V2-specific.** The FFN kernel would be a
   custom op called from the model forward, so it is captured into the FULL
   decode graph by whichever runner captures it; production already runs V2
   and its FULL capture covers all four widths. What 2a does need: FULL
   capture (refuse PIECEWISE, as planned), a plain launch (P0a), and support
   for M up to 24 (MTP2 at 8 sequences). Prefill and mixed steps run the
   PIECEWISE graphs and stay on vLLM's layers.

### Decode-step time split

GPU time per step from the phase timer (CUDA events, production graphs,
rank 0; ranks agree to 0.05 ms):

| Width | target forward | lm_head + gather + sample | drafter (prefill + decode pass) | step total | gap |
| --- | --- | --- | --- | --- | --- |
| 3 (c=1) | 23.8 ms | 0.9 ms | 2.05 + 2.05 ms | **28.9 ms** | 0.65 ms |
| 24 (c=8) | 50.0 ms | 1.7 ms | 3.3 + 2.5 ms | **57.6 ms** | 0.7 ms |

28.9 + 0.65 ms matches the 29.5 ms ITL of the c=1 cells.

Kernel time inside the target forward (eager trace: kernel durations are
launch-mode independent, so the sum of kernel time is the graph's busy
time; per-step medians, rank 0):

| Block | width 3 (c=1) | | | width 24 (c=8) | | |
| --- | --- | --- | --- | --- | --- | --- |
| | launches | busy ms | % | launches | busy ms | % of non-AR |
| attention / GDN blocks (49) | 1497 | 12.33 | 55 % | 1546 | 24.04 | 55 % |
| FFN blocks (48) | 996 | 8.63 | 39 % | 1332 | 19.72 | 45 % |
| all-reduce (98) | 98 | 1.04 | 4.7 % | 98 | n/a | — |
| other (embed, logits) | 65 | 0.29 | 1.3 % | 66 | 0.33 | — |
| **total** | 2656 | 22.36 | | 3042 | 44.1 + AR | |

Graph forward 23.8 ms vs ~22 ms of kernel time: ~0.85 µs of gap per kernel
under graph replay. At width 24 the eager AR durations are spin time waiting
for peers (rank 3's AR at width 3 reads 1.5 ms/call the same way), so only
the count is usable; at width 24 the payload (24 × 2560 × 2 B = 120 KiB) is
above the 64 KiB one-shot gate and every AR goes to RCCL
(`ncclDevKernel_Generic_4`, p10 60 µs).

**One FFN block at width 3** (20–21 launches, median 181 µs):

| Part | Kernels | µs |
| --- | --- | --- |
| HC pre (2 fills, `_hc_combine_norm`, `gemv_act_k` down/inject, copy, `hc_up_gate_mix`) | 7 | 66 |
| shared expert (`se_gate_up_silu_k`, `se_down_gated_k`) | 2 | 19 |
| router GEMV + `topkGating` + align + count/sort + copy + index + fill | 7 | 39 |
| routed experts (`moe_gemm_q4` gate_up 32 µs, `act_and_mul`, fill, down 15 µs) | 4 | 52 |
| residual add | 1 | 2 |

At width 24 the block is 28 launches / ~400 µs and runs the unfused path
(`gemm_tn_splitk_k`, `hc_up_mix_prefill_k`, Tensile `Cijk_*`, rocBLAS gemvt).
The attention blocks carry the same 66 µs HC preamble, so HC alone is
~132 µs per layer, ~6.3 ms of the 28.9 ms c=1 step.

### 2a headroom estimate (against the baseline above)

* Width 3: the routed-expert GEMMs move ~18 MB per layer per rank (≈ 7.3 of
  128 local experts touched by 30 picks, ~2.5 MB each) in 47 µs, already
  ~380 GB/s. The headroom is in the glue: router/top-k/align/sort/index
  (39 µs), the two shared-expert launches (19 µs for ~2.5 MB) and ~15
  inter-kernel gaps. A fused FFN kernel at ~75 µs instead of ~115 µs (HC
  excluded) saves ~40 µs + ~13 µs gaps per layer ≈ **2.2–2.5 ms per step**:
  c=1 ITL 29.5 → ~27.1 ms (−8 %), decode 70.7 → ~77 tok/s/req. Folding the HC
  preamble in as well (66 µs, two GEMVs with fp16 weights) could add ~1 ms
  more (≈ −11 % ITL), unverified.
* Width 24: ~48 of 128 local experts are touched (240 picks), ~118 MB per
  layer, a ~280–300 µs floor against ~400 µs measured. Realistic saving
  ~60–100 µs per layer ≈ 2.9–4.8 ms of 57.6 ms (−5 to −8 % ITL at c=8).
* Prefill and TTFT are untouched by construction (PIECEWISE / mixed steps keep
  vLLM's layers); that still has to be shown with the same four cells in
  phase 3.

## Risks found

1. **Cooperative launch needs `-mcumode`.** In the default WGP build HIP
   refuses `hipLaunchCooperativeKernel` for 72 × 64 KiB WGs (and even 144 ×
   16 KiB), although 72 are co-resident; `grid.sync()` is also unbounded and
   hangs on a bail. Use the plain launch with bounded agent-scope flag
   barriers.
2. **Starvation is not covered by the device-side bound.** A wave-filling
   kernel on another stream delayed the persistent kernel by 6 s before any
   of its WGs started. Keep the decode step on one stream.
3. **Width 24 is a different code path.** At MTP2 c=8 the decode step uses
   the unfused M > 8 glue (+390 launches) and RCCL all-reduces (payload above
   the 64 KiB one-shot gate). A 2a kernel has to cover M = 24, and two
   cheaper levers exist for that cell: extend the M ≤ 8 fused glue to M ≤ 24,
   and the AR gate/protocol at 120 KiB.
4. **Kernel tracing deadlocks the graph-mode serve.** With torch profiler GPU
   activities (kineto) or the rocprofiler-sdk tool preloaded, the first traced
   decode on the FULL/PIECEWISE serve never completes: GPUs at 99 %, the
   worker blocks in `torch.cuda.Stream.synchronize`, rocprofiler logs
   `Async signal handler still waiting on signal`. Same with
   `VLLM_RDNA_AR=0` (RCCL). The eager serve traces fine. Per-kernel numbers
   for graph mode therefore come from eager kernel durations plus graph node
   counts plus CUDA-event phases, not from a graph-mode trace.
5. **Kineto records no GPU kernels in any vLLM worker on this stack.** The API
   server imports torch at startup (venv `sitecustomize`), its HIP runtime
   writes `ROCPROFILER_REGISTER_LIBRARY` into the C environment, every
   spawned worker inherits it, and rocprofiler-register then never finds
   libkineto, so `--profiler-config` traces are CPU-only (reproduced with a
   plain `subprocess` child of a torch-importing parent).
   `tools/monokernel-p0-probe/kineto_shim/sitecustomize.py` drops the variable
   before torch loads. `vllm/env_override.py`'s RTLD_GLOBAL fix does not
   cover this case.
6. `serve_validate.sh` reported 20–21 "new amdgpu dmesg faults" per run;
   they are the known PCIe AER correctable errors on root port `40:03.1`
   (BadTLP), not GPU faults (0 fault markers in `serve.log`).

## Recommendation

2a is worth starting only with a clear scope and target:

* Expected gain at width 3 (c=1): ~2.2–2.5 ms of the 28.9 ms step (−8 % ITL,
  ~+8 % decode tok/s/req) for router + shared expert + routed experts; ~−11 %
  if the HC preamble is folded in. At width 24 (c=8): −5 to −8 % ITL.
  Prefill/TTFT unchanged by design.
* The routed-expert GEMMs are already near bandwidth; the win is the ~16
  glue launches per FFN block and the latency-bound router/top-k/shared
  expert. Folding the HC machinery (~132 µs per layer, 22 % of the c=1 step,
  in both the attention and FFN blocks) is the larger lever and needs a
  roofline check before scoping.
* Cheaper first steps that target the same cells: fused glue for M ≤ 24
  (c=8 runs the unfused path today) and the width-24 all-reduce (98 RCCL calls
  per step).

Raw outputs: `bench_results/2026-10-14_monokernel-p0/` in the infra repo —
`probe/logs/p0-20261014-080456.log`, `probe/logs/p0-stress-20261014-080503.log`,
`probe/build/*`, `serve-mono-p0b/summary.txt` + `bench-*.json`,
`serve-mono-p0b-graphs/{graph_kernel_counts.txt,phase_decode.txt,graphs/}`,
`serve-mono-p0b-tp-eager/{analysis-c1-rank0.txt,analysis-c8-rank0.txt,kernels-c1-rank0.txt,torchprof/}`.
