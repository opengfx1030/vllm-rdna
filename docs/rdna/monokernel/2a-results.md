# MonoKernel 2a — results (V620 / gfx1030, 2026-10-14)

Branch `monokernel/2a`. Raw results: `bench_results/2026-10-14_monokernel-2a/`
(infra repo); on the box `~/w4a8_runs/monokernel-2a/`. The M1 write-up of the
first kernel version (0.91x, stopped) is [`2a-m1-results.md`](./2a-m1-results.md);
this document covers the reworked kernel, M2, M3, and the in-serve timing.

**Verdict: recommend stopping 2a.**
* The kernel is correct, graph-captured and fail-stop.
* Standalone, it beats the M1 reference by 1.15-1.59x at every width.
* In the serve it shows no gain:
  * decode tok/s/req is −5% to +2.3%;
  * ITL at c=8 (width 24) is +4-6%;
  * prefill and TTFT are unchanged.
* In-serve timers show why:
  * at widths ≥ 12, today's FFN block is ~180 µs faster in the serve than the M1 reference measured, so the kernel loses at width 24;
  * at widths 1-8 it saves 5-15 µs per layer, which an unexplained slowdown outside the FFN region cancels.

## What changed since the first M1

Same 7-phase persistent kernel (`csrc/rocm/mono_ffn_rdna2.cu`):
* HC combine+norm, HC down+inject, HC up-mix, router/shared expert, top-k plus routed gate_up, routed down, and the partial.
* Six bounded grid barriers, a sticky error record, and NaN fail-stop.

Changes:
* **16 waves/CU:** 144 WGs × 256 threads, 128 VGPRs, 29 KiB LDS. The residency cap is computed from the code object.
* **LDS-tiled GEMVs at M ≥ 5:** the HC down, HC up-mix and router/shared-expert phases use WG tiles, ported from `gemm_tn_splitk_k` / `hc_up_mix_prefill_k`.
* **Routed-expert phases:** P5 runs as WG tasks (expert × 64 gate columns, LDS reduction, SwiGLU). P6 runs as wave tasks with a K split and fp32 atomics.
* **Bug fix:** an int4 group-scale preload bug for K slices spanning more than 4 quant groups is fixed.

## M1 — standalone (hipGraph of 16 distinct layers, cold Infinity Cache, per layer)

| width | today's block (M1 reference) | mono | speedup | mono P1..P6 (µs) |
| --- | --- | --- | --- | --- |
| 1 | 153.2 | 127.6 | 1.20x | 8.6/23.6/23.7/13.8/33.8/18.0 |
| 2 | 179.0 | 146.3 | 1.22x | 8.4/26.9/26.8/15.1/35.7/17.2 |
| 3 | 204.4 | 172.7 | 1.18x | |
| 4 | 225.9 | 195.8 | 1.15x | 8.2/31.5/27.8/19.1/71.2/32.6 |
| 6 | 280.3 | 243.2 | 1.15x | |
| 8 | 299.9 | 260.7 | 1.15x | 8.3/41.6/30.3/22.4/124.4/52.9 |
| 12 | 541.7 | 340.1 | 1.59x | |
| 24 | 683-707 | 513-532 | 1.33x | 9.8/43.5/34.4/25.2/253.3/107.1 |

* Occupancy is 16 waves/CU.
* Error vs the fp32 golden is about half of today's path's: fp32 expert accumulation. Rows that differ are fp32 routing near-ties (top-10/11 gap < 1e-3).
* GB/s per phase:

  | width | P2 | P3 | P4 | P5 | P6 |
  | --- | --- | --- | --- | --- | --- |
  | 3 | 235 | 234 | 240 | 263 | 260 |
  | 24 | 155 | 200 | 167 | 289 | 333 |

## M2 — integration and gates

* **Switch:** `--additional-config '{"rdna_mono_ffn": true}'`, off by default.
* **Gating:** gfx10x, Qwen4Exp MoE decoder layers, symmetric W4A16 RDNA2 MoE, and contiguous EP4 expert blocks. MTP layers are excluded.
* **Op structure:** each layer's FFN region becomes one opaque custom op (`rdna_mono_ffn_block`). It launches the kernel only while a FULL graph is being captured (M ≤ 24, fp16). Everything else keeps today's modules. The all-reduce stays outside the kernel.
* **Capture evidence:** widths 3, 6, 12 and 24 were captured on all 4 ranks.
* **Gates on HIP 6-9:**

  | Gate | ON | OFF |
  | --- | --- | --- |
  | Probe | 3/3 | pass |
  | PPL | 4.80 | 4.83 |
  | Prefix | pass | pass |
  | Concurrency | 0 failing | 0 failing |
  | Probe after load | pass | pass |

## M3 — serve A/B (4× V620 TP4/EP4, F&P, PLE int4, `--temperature 0`, OFF/ON alternating)

Means over 2 reps (MTP2) or 1 rep (MTP0). Columns are prefill tok/s/req, decode tok/s/req, TTFT and ITL median.

| recipe | cell | OFF | ON | decode Δ | ITL Δ |
| --- | --- | --- | --- | --- | --- |
| MTP2 | 1k/512 c=1 (w3) | 2069 / 65.20 / 0.49 s / 28.8 ms | 2052 / 64.71 / 0.50 s / 28.4 ms | −0.8% | −1.1% |
| MTP2 | 1k/512 c=8 (w24) | 950 / 29.62 / 1.08 s / 60.6 ms | 925 / 29.19 / 1.11 s / 64.0 ms | −1.5% | +5.7% |
| MTP2 | 16k/1k c=1 (w3) | 2032 / 78.16 / 8.06 s / 29.2 ms | 2034 / 79.94 / 8.06 s / 28.8 ms | +2.3% | −1.4% |
| MTP2 | 16k/1k c=8 (w24) | 1180 / 14.12 / 13.91 s / 62.4 ms | 1218 / 14.13 / 13.52 s / 65.1 ms | 0.0% | +4.4% |
| MTP0 | 1k/512 c=1 (w1) | 2005 / 44.97 / 0.51 s / 22.2 ms | 1991 / 44.16 / 0.51 s / 22.6 ms | −1.8% | +1.8% |
| MTP0 | 1k/512 c=8 (w8) | 521 / 26.10 / 1.96 s / 34.6 ms | 522 / 24.80 / 1.96 s / 36.7 ms | −5.0% | +5.9% |
| MTP0 | 16k/1k c=1 (w1) | 2105 / 44.44 / 7.78 s / 22.5 ms | 2104 / 43.61 / 7.79 s / 22.9 ms | −1.9% | +1.9% |
| MTP0 | 16k/1k c=8 (w8) | 1070 / 13.30 / 15.31 s / 35.7 ms | 1072 / 13.06 / 15.28 s / 37.3 ms | −1.8% | +4.5% |

* Prefill and TTFT are within noise.
* MTP2 c=1 decode tok/s/req swings ±10% between OFF reps because MTP acceptance varies (65 vs 60 at 1k). Read ITL for that cell.
* All 6 serves pass every gate (PPL 4.79-4.87).
* **Logprob parity** (greedy top-5, identical-prefix tokens of 772):

  | Pair | Identical-prefix tokens |
  | --- | --- |
  | OFF vs OFF | 539 |
  | ON vs ON | 595 |
  | OFF vs ON (two pairs) | 621, 500 |
  | MTP0 OFF vs ON | 610 |

  Every divergence is at a top-2 gap < 0.3 nats. ON vs OFF is inside the run-to-run noise.

**Estimated vs measured.** From the M1 per-layer savings over 48 layers:

| Width | Estimated ITL change | Measured ITL change |
| --- | --- | --- |
| 1 | −1.2 ms | +0.4 ms |
| 3 | −1.5 ms | −0.4 ms |
| 24 | −8.4 ms | +3.4 ms |

## In-serve timing (why)

Graph-mode tracing hangs the engine (P0), so the timing is built in, opt-in through `additional_config`:
* `rdna_mono_ffn_timing` puts one-thread timestamp kernels (`mono_ffn_rdna2_mark`) around the FFN region + its all-reduce inside the FULL graphs. It also turns on per-width kernel stats in the workspace tail: phase times, first and last WG entry, and mark-to-entry latency.
* `rdna_mono_ffn_modules` keeps today's modules under the same marks, giving the switch-off reference.

Means over 47k-379k replays per width. The four ranks agree within 1%. Tools: `serve_2a.sh ARM=off-t|on-t`. Raw: `serve-timing/timing_final.txt`.

| width | today, region (µs) | mono, region (µs) | Δ per layer | mono kernel (µs) | mono P1..P6 in serve (µs) | M1 mono kernel (µs) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 172 | 161 | −11 (−6%) | 122 | 7.5/25.2/23.5/14.0/34.9/17.1 | 128 |
| 3 | 211 | 196 | −15 (−7%) | 159 | 7.5/30.2/27.2/18.1/49.7/26.5 | 173 |
| 8 | 318 | 313 | −5 (−1.5%) | 256 | 7.1/43.8/32.1/23.1/103.3/46.1 | 261 |
| 12 | 365 | 354 | −11 (−3%) | 291 | 7.2/43.8/31.5/23.6/129.3/55.6 | 340 |
| 24 | 504 | 569 | +65 (+13%) | 449 | 7.9/45.2/31.7/25.6/219.5/119.1 | 513-532 |

The coordinator's four suspects:

1. **Start latency: not the cause.**
   * The first WG enters 3.8-4.2 µs after the preceding mark kernel ends, which is the graph's kernel-to-kernel gap.
   * The spread between the first and last WG entry is 0.1-0.3 µs, so all 144 WGs are resident at once.
   * P1 (combine+norm plus barrier 1) takes 7-8 µs, the same as standalone.
2. **Today's in-serve block: this is the cause at widths ≥ 12.**
   * Today's FFN region *including* its all-reduce is 365 µs at width 12 and 504 µs at width 24.
   * The M1 reference was 542 and 683-707 µs *without* the all-reduce.
   * The serve's M > 8 path is therefore much faster than the op sequence M1 replicated. The "+390 launches, prefill-style glue" penalty that M1 measured does not exist in the serve.
   * At widths ≤ 8, today's in-serve block matches the M1 reference: 172 vs 153 µs plus about 20 µs of all-reduce and marks.
3. **Cache context: the kernel is as fast in the serve as standalone, or faster.**
   * Per-phase times match M1 within a few µs at widths 1-8. At width 24 the serve is faster (P5 220 vs 253 µs), so the achieved GB/s is the same or higher.
   * The cold 16-layer bench was a fair model of the serve.
4. **Width 24: no barrier trouble.**
   * No sticky error ever fired; PPL and parity are clean.
   * Entry spread is 0.2 µs, so no co-scheduled RCCL kernel holds CUs when the grid launches.
   * The kernel simply takes 449 µs. Of that, 339 µs is the two routed-expert phases, while today's entire region is 504 µs including HC, router, shared expert and the RCCL all-reduce.

**Unexplained remainder.**
* At widths 1-3 the region saves 11-15 µs per layer, which is −0.5 to −0.7 ms per step.
* Yet ITL at MTP0 c=1 is +0.4 ms in M3 and +0.8 ms in the timing pair, and at MTP2 c=1 only −0.3 to −0.4 ms.
* So with the kernel on, the rest of the step (attention/GDN, its all-reduce, PLE, sampling) costs ~0.7-1.3 ms more, about 15-27 µs per layer.
* It reproduces across two serve pairs. It is not inside the marked region, and both arms run the same op and the same marks.
* Candidates, not measured:
  * the clock/power state after a 16-wave/CU persistent kernel;
  * the L2/MALL state the kernel leaves for the following attention kernels.
* Marks around the attention block (same tool) would separate these.

## Recommendation

Stop 2a.
* **Widths ≥ 12:** the kernel's routed-expert phases (P5 + P6 = 339 µs at width 24) are slower than today's whole in-serve block minus its all-reduce. Winning there would mean beating today's in-serve MoE GEMMs at M = 24 by more than 2x inside a persistent grid. Nothing in the phase data points to that.
* **Widths ≤ 8:** the best case is −15 µs per layer, about −0.7 ms or −2.5% ITL at c=1. Even that is cancelled by the outside-region slowdown.
* **Fallback option:** gating the kernel to M ≤ 8 would be ITL-neutral at best, so it is not worth carrying.
* **What to keep:** the kernel, the integration (default off) and the in-serve timing tools stay on the branch as reference. The timing hook is a reusable way to time any FFN-region change inside FULL graphs on this stack.
* **Next lever:** the gap the M1 reference suggested at M > 8 is not present in the serve. The decode levers worth chasing are the ones P0 listed for the all-reduce and attention side, not the FFN glue.


## Stage E and the all-on MonoKernel (2026-10-15)

Superseding the recommendation above: with the later work the MonoKernel parts
are all on together and faster than off. Raw results:
`bench_results/2026-10-14_monokernel-2a/stage-e/` (infra repo).

**All-on** = `rdna_mono_ffn` (max rows 8) + `rdna_mono_hc` (K1) +
`rdna_mono_gdn` (E1') + `rdna_mono_qsa` (E3), all off by default.

* **E1' (`gdn_fused_rdna2`)**: conv update + gated delta rule + gated RMSNorm
  in one launch for pure decode / MTP-verify batches (<= 64 rows); today's
  path for anything else, inside the same op (a splitting op, like the GDN
  core). Standalone 1.11-1.90x on the replaced region (64 cold layers),
  conv state exact, ssm state <= 7e-8.
* **E3 (QSA)**: live-context bound for host-run scoring, early exit of
  scoring tiles past the live context (valid inside CUDA graphs), all-padding
  tile skip in sparse attention, a one-launch HIP sparse decode
  (`qsa_decode_rdna2`, from 3 rows), the indexer norm + MRoPE in one launch
  and the three indexer cache stores in one launch.
* **Fix that made all-on additive**: the abandoned E1a GDN mode compiled into
  the persistent FFN kernel slowed every FFN / K1 phase ~25% (standalone
  w1/3/8 151.7/201.5/311.2 us -> 117.2/165.5/251.3 us without it); now behind
  `MONO_FFN_E1A=0`.

### A13: all-on vs off, same session, alternating (4x V620, TP4, FG=0)

Cells: prefill tok/s/req / decode tok/s/req / TTFT s / ITL median ms.

| recipe / cell | off | all-on | ITL |
| --- | --- | --- | --- |
| MTP2 1k c=1 | 2096 / 70.35 / 0.49 / 28.8 | 2248 / 75.98 / 0.46 / 25.3 | -12.2% |
| MTP2 1k c=8 | 893 / 31.21 / 1.15 / 61.3 | 1011 / 29.54 / 1.01 / 58.7 | -4.2% |
| MTP2 16k c=1 | 2034 / 87.24 / 8.06 / 29.1 | 2039 / 100.04 / 8.04 / 26.2 | -10.0% |
| MTP2 16k c=8 | 1374 / 13.94 / 11.93 / 62.1 | 1396 / 14.15 / 11.73 / 61.2 | -1.4% |
| MTP0 1k c=1 | 2001 / 44.99 / 0.51 / 22.2 | 2165 / 46.86 / 0.47 / 21.3 | -4.1% |
| MTP0 1k c=8 | 523 / 25.87 / 1.96 / 35.1 | 566 / 26.75 / 1.81 / 34.1 | -2.8% |
| MTP0 16k c=1 | 2105 / 44.47 / 7.78 / 22.5 | 2104 / 46.09 / 7.79 / 21.7 | -3.6% |
| MTP0 16k c=8 | 1074 / 13.33 / 15.26 / 35.8 | 1070 / 13.43 / 15.32 / 34.8 | -2.8% |

Prefill and TTFT at parity or better (16k MTP0 within 0.5%). All gates pass;
logprob parity inside off-vs-off noise.
