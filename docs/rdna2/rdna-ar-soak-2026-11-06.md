# `rdna_ar` soak on slot A next to slot-B boots: 2026-11-06

Status: **passed.** Production `flashnext-mtp2` (TP4, compiled mode, FULL_AND_PIECEWISE
cudagraphs, `GPU_MAX_HW_QUEUES=4`, `VLLM_RDNA_AR=1`) served 2 h 17 min of
continuous mixed load on slot A (HIP 1-4) from an exact-SHA tree of
`rdna_extras` `22483a656e` (with the rdna_ar wedge fix, the mlock fix and the
hier mode, which does not apply at TP4), while another agent booted
production TP4 servers on slot B (HIP 6-9) back to back: 19 boot.lock
windows of about 5 min each started during the measured load.

Result: 0 wedges, 0 `rdna_ar` disables, 0 late peers, 0 faults, 0 failed
requests, 0 ms of KFD eviction on every worker, ITL flat to within noise.
One probe (the prefix-cache speedup probe) reports FAIL on this tree
independently of the soak; see "Findings".

## Setup

| item | value |
|---|---|
| Tree | `~/vllm-soak-22483a656e` on par1-cs25: `git archive` of `22483a656e` plus the `.so` set of the `2d0f6ccdf9` build (`c5c36a5a88f6` `_rocm_C`; `csrc/` is identical between the two commits) |
| Recipe | `tools/rdna/recipes/flashnext-mtp2.env` unchanged; plan checked before the launch (`GPU_MAX_HW_QUEUES=4`, `VLLM_RDNA_AR=1`, `MAX_KB=ONESHOT_KB=256`) |
| One-shot gate | the recipe's 256 KiB, not the 64 KiB library default: MTP2 verify batches are 77-123 KB. The hier mode logs "does not apply, using the flat exchange" at TP4 |
| Slot | HIP 1-4 = V620 `ce d3 d8 db`, verified by PCI bus before the launch; lease `Soak.slotA` 22:49-01:14 |
| Boot | 290 s to READY, cold caches; PLE table 23.84 GiB per rank, mlocked on all four workers |
| Load | 92 mixed cells back to back, 1k/512 and 16k/1k alternating, concurrency 1-8 drawn at random, random seed per cell, temperature 0, plus 24 reference cells (1k/512 c1, 1k/512 c8, 16k/1k c1) every 15 min |
| Checks | every 15 min: greedy probe and PPL probe; every hour: concurrency probe; boot and final |
| Neighbour | no quiet window held; the slot-B boots were the load under test |
| Monitor | 2 s sampler: KFD `evicted_ms` per worker, `rdna_ar` / traceback / fault lines of the serve log, dmesg (timestamp-filtered), `boot.lock` and lease state, neighbour processes, host memory and reclaim counters; junction temperature every second with the abort rule (5 s mean >= 90 C or a sample >= 95 C) |

Scripts: `tools/rdna/ar_soak/` (`driver.sh`, `serve_wrap.sh`, `soak_hook.sh`,
`monitor.py`, `tempwatch.py`, `analyze.py`, `slots.sh`). Logs on the box:
`~/w4a8_runs/rdna_ar_soak_2026-11-06/` (`run/summary.txt`, `run/serve.log`,
`run/soak/{cells.tsv,checks.tsv,events.log,counts.json,meminfo.csv}`,
`run/temps1s.log`, `analysis.md`, `driver.log`).

## Results

| criterion | result |
|---|---|
| Wedges (`rdna_ar: WEDGED`, `rdna_ar wedged`) | 0; no wedge marker in the cache root |
| `rdna_ar: disabled` | 0 |
| `rdna_ar: late peer` | 0 |
| `rdna_ar` active on every rank | yes, 4 of 4 (`up to 256 KB, RCCL above`) |
| KFD `evicted_ms` since READY, workers TP0-TP3 and the parent | +0 ms on all five over the whole run (monitor: no jump in 2 s samples; `serve_validate`: `evicted_ms since READY ... -> ok`) |
| amdgpu faults, resets, AER, `Runlist is getting oversubscribed` | 0 (timestamp scan of dmesg from the launch; the driver's line-offset summary is unreliable on this busy log) |
| Failed requests | 0 of 1,151 benchmark requests in 116 cells; 0 client timeouts (cell cap 1,500 s, longest cell 163 s); 0 Tracebacks in the serve log |
| Greedy probe | PASS at boot, every 15 min and at the end (10 of 10) |
| PPL probe | 4.83 at boot; 10 checks min 4.78, max 4.85, mean 4.826 (the recipe's range is 4.80-4.86) |
| Concurrency probe | PASS at boot and at 00:00 and 01:00 (0 failing; near-ties only) |
| Junction temperature | max 83 C (d3), 80-81 C on the others over 8,511 samples: no abort |

### ITL drift

Median of the per-cell ITL median, first third of the cells against the last
third, per (context, concurrency) group:

| group | cells | first third (ms) | last third (ms) | change | slope %/h |
|---|---|---|---|---|---|
| 1k c1 | 15 | 22.39 | 22.44 | +0.2 % | +0.04 |
| 1k c2 | 6 | 26.31 | 26.30 | -0.1 % | +0.06 |
| 1k c4 | 5 | 35.31 | 35.14 | -0.5 % | -0.20 |
| 1k c6 | 7 | 41.25 | 40.97 | -0.7 % | -0.15 |
| 1k c8 | 12 | 43.97 | 44.00 | +0.1 % | -0.01 |
| 16k c1 | 14 | 22.85 | 22.77 | -0.3 % | -0.08 |
| 16k c2 | 9 | 26.88 | 27.08 | +0.7 % | +0.19 |
| 16k c5 | 8 | 40.66 | 41.38 | +1.8 % | +0.39 |
| 16k c8 | 5 | 46.18 | 45.69 | -1.1 % | -0.14 |

All 16 groups sit within -1.9 % to +2.8 % first-vs-last third with a
coefficient of variation of 0.2-1.9 % and no consistent sign (the c3 to c7
groups in `analysis.md`). The 15-minute reference cells are flat for 2 h: 1k
c1 22.4 ms every time, 16k c1 22.6-23.0 ms, 1k c8 43.3-44.9 ms (the c8 spread
is the 16-request sample). The p99 ITL per group does not move either (e.g.
1k c8 358.6 ms inside a boot window, 358.8 outside; 16k c8 404 vs 403 ms; the
~360-400 ms are the interleaved prefill steps, not stalls).

### Neighbour boots

The monitor saw 20 `boot.lock` held windows overlapping the load (a slot-B
boot of 298-333 s each, one 58 s aborted attempt, and a first 695 s window
that began at 22:51 before the load); the 19 that started during the load: 23:04,
23:10, 23:16, 23:22, 23:28, 23:34, 23:44, 23:55, 00:01, 00:03, 00:09,
00:15, 00:29, 00:35, 00:41, 00:47, 00:53, 00:59, 01:05 (box clock; the
correlations are in `run/soak/events.log`). Each window loads the 168 GiB
checkpoint and pins 4 x 23.8 GiB on the other slot.

- 98 of the 116 cells overlap one of the 20 windows; their ITL median is 0.9994 of the
  group median (range 0.979-1.028), the 18 cells outside 0.9968 (0.967-1.015).
  The worst cell inside a window (16k c6) is +2.8 %, within the 16k c6 group
  spread.
- Host pressure was real: lowest MemFree 11.0 GiB (23:41:22), 3.96 M pages
  migrated by compaction and 709 direct-reclaim page scans over the run.
  Still 0 ms evicted on every worker.
- Three `amdgpu 0000:ce:00.0: Trap debug id already reserved` lines at
  23:38:55, when a slot-B client process started; no consequence in the
  serve log, probes or ITL. They are the only amdgpu lines in the window.

## Findings

1. **Prefix probe FAIL (not rdna_ar).** `serve_validate.sh`'s
   `prefix_probe.py` reports `cold TTFT 0.69 s, warm 0.37 s, speedup 1.8x`,
   hits +1024 of 3,854 queries (the cache works: the 1,024-token block hits);
   the 0.5 ratio gate fails because the cold prefill is now fast (it was 4.2-4.4 s
   in the 2026-11-05 smoke). A second fresh boot on another port gave the same
   0.69 s / 0.36 s (1.9x), so it is not stray traffic. The probe's speedup gate
   needs a recheck against the current prefill speed; the soak's own
   correctness gates (greedy, PPL, concurrency) are unaffected.
2. **`rdna_ar: PIX topology query failed (name 'amdsmi_init' is not defined)`**
   on all four ranks at every boot: the venv has no `amdsmi`, so
   `is_pix_connected` fails and `pix` is reported `False`. The value only goes
   into the "active" log line; the one-shot path is unaffected.
3. `dmesg_summary.txt` of `driver.sh`-style runs counts new lines by line
   offset, which is wrong when the kernel ring is nearly full (it reported 0
   amdgpu lines while the ring had 3). `monitor.py` filters by timestamp
   instead; use that.
4. The first launch (try0) was killed by a monitor bug (the GECC boot line of
   the idle W7800s matched the fault regex); the 22:39 boot also took a short
   stray request burst from the Disagg driver (default port 18700). Both
   attempts are in `run.try0-monitor-bugs/`; the 22:49 relaunch is the soak.
   The prefix rerun is `~/w4a8_runs/rdna_ar_soak_2026-11-06-prefixrerun/`.

## What was not covered

- Hier mode (TP8 only) was not exercised: this was TP4.
- 256 KiB one-shot is what ran; the 64 KiB library default was not soaked.
- 2 h 17 min at one boot: the 600 s late-peer wait was never needed, so
  its path stayed idle; the earlier stress matrix is where that was
  exercised.
