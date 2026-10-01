# fp32-accumulator MoE default — Flash-Next full matrix (2026-09-30)

Part 3 acceptance: perf + correctness of the **fp32 MoE epilogue accumulator
default** (`rdna_extras` `09b7f33d0` + `86461060e` + `e3ced1495`). Flash-Next
AWQ-W4A16, TP=4, FULL_AND_PIECEWISE cudagraphs, prefix caching, FA-RDNA2
(`RDNA_ATTN`), frozen 765-row TunableOp set (`rocblas-f30bb442e9b5`,
lookup-only), 4× Radeon PRO V620 (gfx1030), PLE int4 sidecar on.

- Arm: **fp32 accum default ON** (`VLLM_RDNA2_MOE_FP32_ACCUM` unset/true).
- A/B: same cells with the legacy CAS (`VLLM_RDNA2_MOE_FP32_ACCUM=0`).
- W4A8: `VLLM_RDNA2_W4A8_SDOT4=1` (shared epilogue) single cell.
- Launcher: `tools/rdna2_028/fp32_accum_matrix.sh`; driver logs
  `driver_fp32.log` (fp32 arms) + `driver_cas_w4a8.log` (cas/w4a8 arms).

All 5 arms: **8/8 (or 2/2, 1/1) cells completed, server coherence 4 OK / 0 BAD**,
PCI-SERR unchanged (0) across every boot/teardown.

## 1. fp32 default matrix (8 cells)

### MTP=0 (arm `fp32-m0`)

| Cell | TTFT (s) | prefill tok/s | decode/req tok/s | ITL med (ms) | out tok/s (agg) | total tok/s |
|---|---:|---:|---:|---:|---:|---:|
| 16k/1k c=1 | 9.76 | 1677.9 | 42.07 | 23.79 | 30.04 | 510.75 |
| 16k/1k c=8 | 47.41* | 2753.6 | 13.88 | 38.56 | 66.64 | 1132.86 |
| 1k/512 c=1 | 0.65 | 1572.3 | 42.44 | 23.55 | 40.34 | 121.02 |
| 1k/512 c=8 | 2.92* | 2817.8 | 24.01 | 37.59 | 168.57 | 505.71 |

\* median TTFT (mean in mean-ttft column; see cells.csv for both).

### MTP=2 (arm `fp32-m2`)

| Cell | TTFT (s) | prefill tok/s | decode/req tok/s | ITL med (ms) | out tok/s (agg) | total tok/s | accept len / rate |
|---|---:|---:|---:|---:|---:|---:|---|
| 16k/1k c=1 | 10.19 | 1608.0 | 66.14 | 36.12 | 39.91 | 678.47 | 2.46 / 73–78% |
| 16k/1k c=8 | 51.52* | 2541.4 | 15.07 | 85.79 | 66.08 | 1123.32 | ~2.4 / ~75% |
| 1k/512 c=1 | 0.65 | 1571.7 | 72.76 | 36.06 | 66.71 | 200.12 | — |
| 1k/512 c=8 | 3.02* | 2722.4 | 23.49 | 83.01 | 143.86 | 431.58 | — |

Acceptance from `acceptance.txt` (`SpecDecoding metrics`): mean acceptance
length 2.10–2.95, avg draft acceptance rate 55–78%. MTP=2 beats MTP=0 at c=1
(16k out 39.91 vs 30.04; 1k out 66.71 vs 40.34) and is at parity at c=8.

## 2. A/B: fp32 default vs legacy CAS (c=1)

Both arms carry genuine mode flags (worker environ verified: `=1` fp32, `=0`
CAS). `Δ` = CAS relative to fp32 (positive = CAS faster).

| Arm / cell | out tok/s fp32 | out tok/s CAS | Δ out | prefill fp32→CAS | Δ prefill | ITL fp32→CAS | TTFT fp32→CAS |
|---|---:|---:|---:|---|---:|---|---|
| MTP=0 16k/1k c=1 | 30.04 | 30.78 | **+2.5%** | 1677.9→1745.1 | +4.0% | 23.79→23.36 ms | 9.76→9.39 s |
| MTP=0 1k/512 c=1 | 40.34 | 41.38 | **+2.6%** | 1572.3→1671.8 | +6.3% | 23.55→23.01 ms | 0.65→0.61 s |
| MTP=2 16k/1k c=1 | 39.91 | 44.60 | **+11.8%** | 1608.0→1684.7 | +4.8% | 36.12→35.37 ms | 10.19→9.73 s |
| MTP=2 1k/512 c=1 | 66.71 | 75.65 | **+13.4%** | 1571.7→1698.5 | +8.1% | 36.06→35.06 ms | 0.65→0.60 s |

Conclusion: the fp32 accumulator is **not free** — CAS is 2.5% faster at
MTP=0 and 12–13% faster at MTP=2, at c=1. Greedy outputs stay coherent on
both arms (france→Paris, 2+2→4, no garbage flags). The fp32 path's value is
run-to-run determinism (covered by `e3ced1495`), bought at a small c=1 cost.
No correctness regression on either side.

## 3. W4A8 arm check (`w4a8-m0`, `VLLM_RDNA2_W4A8_SDOT4=1`)

MoE marker asserted in the serve log:
`moe_w4a8_rdna2.hip:582 Warning: RDNA2 W4A8 sdot4 MoE path active (config
moe_a8_k32_ag)`. Coherent (4 OK / 0 BAD).

| Cell | TTFT (s) | prefill tok/s | decode/req tok/s | ITL med (ms) | out tok/s (agg) | total tok/s |
|---|---:|---:|---:|---:|---:|---:|
| 16k/1k c=8 | 45.71* | 2862.1 | 14.12 | 38.81 | 68.32 | 1161.40 |

W4A8 (shared fp32 epilogue) is at parity with the W4A16 16k/1k c=8 fp32 cell
(66.64 out / 2753.6 prefill → 68.32 / 2862.1). No regression.

## 4. Coherence (all arms)

`probe_w4a8.py` on every arm: **4 OK / 0 BAD / 4 total**. France→Paris and
2+2→4 both coherent; gpu/python prompts non-empty and free of garbage flags
(empty / dominant-token / repeat≥5 / char-loop / replacement-char). Per-cell
coherence (`cells/*/coherence.txt`) coherent on every measured cell.

## 5. Device placement + a real finding

The brief asked for GPUs 4–7. **On this stack `HIP_VISIBLE_DEVICES` and
`CUDA_VISIBLE_DEVICES` do not remap physical devices** — a standalone
`torch.ones(2 GB, device="cuda:0")` under `HIP_VISIBLE_DEVICES=4,5,6,7`
allocated on **physical GPU 0** (verified in `rocm-smi --showmeminfo vram`),
and `torch.cuda.device_count()` still reports 4. `--device-ids "4,5,6,7"`
(`assigned_physical_gpu_ids`) also failed to reach the workers. What *does*
work here is the documented GPU-assignment quirk: with
`HIP_VISIBLE_DEVICES=0,1,2,3` (the launcher default) the vLLM workers bind to
**physical GPUs 4,5,6,7** (`rocm-smi --showpidgpus` → DRM 4/5/6/7;
`showmeminfo` → 4–7 at ~26–29 GB, 0–3 at 0.02 GB). All measured cells
therefore ran on physical 4–7, matching the intent.

A second, load-bearing finding: **launcher-passed env vars are dropped at the
`api_server → EngineCore` spawn** (the EngineCore/worker `/proc/*/environ`
carry none of them, while serve-script `export`s survive). Without a fix, the
CAS arm silently fell back to fp32 and the W4A8 arm to W4A16 (both verified
dropped on a first attempt). Fix: an opt-in venv `sitecustomize.py` hook that
re-loads `~/.cache/rdna_fork_mode` into `os.environ` at every process start;
the launcher writes that file per arm. Worker environ then shows the correct
`VLLM_RDNA2_MOE_FP32_ACCUM` / `VLLM_RDNA2_W4A8_SDOT4` values, and the arms are
genuine.

## 6. Verdict

**The fp32-accum default is production-clean**: coherent output at both 16k
and 1k, MTP=0 and MTP=2, c=1 and c=8; no garbage; no PCI SERR; no arm
failures. Its cost versus the legacy CAS is small at MTP=0 (+2.5% for CAS)
and larger at MTP=2 (+12–13% for CAS), all at c=1. Production serving at c=8
is at parity. Recommend keeping fp32 ON for its determinism unless the
MTP=2/c=1 +13% matters to the workload.

## GPU cleanup

Before the run: stopped the leftover `VLLM::Worker_TP0..3` + EngineCore +
resource-tracker of the previous `2026-09-30_pr30-b1c` run
(PIDs 845292–845295, 844873, 844872), which held GPUs 4–7. GPUs 0–3 were free
throughout (no `tensorfold` process active; never touched). After the run:
all engines torn down by the launcher's per-arm `hygiene`/`teardown`; box left
with all 8 GPUs at ~0.02 GB (no vLLM processes). No co-tenant process was
stopped or restarted.

## Files

- `driver_fp32.log`, `driver_cas_w4a8.log` — driver console (cells.csv lines,
  PCI-SERR snapshots, READY/teardown).
- `<arm>/cells.csv` — flat metrics per measured cell (warmups included).
- `<arm>/cells/<cell>/coherence.txt` — per-cell France/math verdict.
- `<arm>/coherence.txt` — server-level garbage/coherence probe.
- `<arm>/markers.txt` — W4A8 + fp32 + TunableOp + rdna_ar + backend markers.
- `<arm>/acceptance.txt` — MTP `SpecDecoding metrics`.
- `<arm>/cold_compile_seconds.txt` — cold boot (286–327 s per arm).
- `<arm>/serve.log` — full server log.
