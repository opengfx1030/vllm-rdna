# Hierarchical `rdna_ar` for two PCIe switch islands (TP8): 2026-11-06

Status: **opt-in, not shipped as the default.** `VLLM_RDNA_AR_HIER=auto|1`
turns on a topology-aware all-reduce for a group of two PCIe switch islands
(TP8 on the 2 x 4 V620 box); the default stays `0`. The run budget ended before
the last two rows below were filled in.

| item | state |
|---|---|
| Correctness at TP8, 4 KiB to 20 MiB, fp16 / bf16 / fp32 | proven: bit-exact against a CPU reference on the real 2 x 4 islands and on an interleaved override (4200 exact + 2400 soak results per layout), forced 2 + 2 islands on 4 GPUs pass, 11 new CPU test functions |
| Per-call latency | measured: 3.1x faster than the flat one-shot at 128 KiB (the MTP2 c=8 verify size), 5.8x at 1 MiB, 3.8x faster than RCCL at 128 KiB, tie with RCCL at 1.5 MiB |
| At least 1 h of stress with 0 wedges | **not complete**: the shakedown is clean (3 cells x 60,416 collectives); the 1 h block started at 11:16 box clock and was still running when the budget ended |
| Serving ABBA, Flash-Next TP8 MTP2, compiled | **not run**; the expected effect below is an inference from the microbench |
| Default | stays `0` until both of the last two rows are filled in |

## Why the flat one-shot is cross-root bound at TP8

The box has two PCIe switch islands of four V620 (slot A = HIP 1-4, buses
`ce d3 d8 db`, root complex `c0`; slot B = HIP 6-9, buses `0a 0f 14 17`, root
`00`), and the two roots meet in the SP3 (EPYC). The P2P matrix measured by the
PCIe audit ([pcie-iommu-audit-2026-11-06.md](pcie-iommu-audit-2026-11-06.md))
says what a TP8 collective can use:

- P2P writes (push) run at 24.5-27 GB/s per direction for every pair, cross-root
  included; a cross-root flag costs 6.9 us one way against 6.2 us inside a
  switch. Cross-root reads (pull) are 5.7 GB/s: never pull.
- The cross-root aggregate per direction is 25.2 GB/s for 1 pair, 27.6 for 2
  pairs, **18.7 for 3 and 17.0 for 4 concurrent distinct pairs** (above about
  64 KiB per flow; below it every concurrency level costs 7-10 us).
- Intra-switch traffic does not slow cross-root traffic and the other way
  round: the two switches are independent.

The flat one-shot (`rdna_ar_oneshot`, the rdna_ar kernel of PR 22 that the
recipes run with `ONESHOT_KB` = `MAX_KB`, so the two-shot kernel is unused) has
every rank push its whole payload to each of the seven other ranks. At world 8
that is four cross-root pushes per rank, 16 concurrent cross-root flows of the
whole payload per direction. Its measured per-call cost (table below) is
linear in the payload, 0.81 us/KiB at 1 MiB: 16 N bytes per direction in
18.6 GB/s at 128 KiB and 20-21 GB/s from 1 to 5 MiB, the size of the cross-root
aggregate the audit measured for 3-4 concurrent pairs (17-19 GB/s) [INFERENCE:
the same limit, with 16 flows instead of 4 pairs]. The time is set by the
cross-root path, not by the intra-switch links. The RCCL ring has the opposite
profile: bandwidth-optimal, one cross-root flow per direction per ring step, but
eight sequential steps, 180 us from 4 to 64 KiB. The flat one-shot beats RCCL
below about 150 KiB and loses above it (the TP8 audit of Tp8Pp2, on
`rdna_extras` as `docs/rdna2/tp8-pp2-flashnext-2026-11-06.md`, found 150-160 KiB).

An MTP2 verify step at c=8 reduces M = 24 rows of 2560 fp16, 120 KiB per
all-reduce, about 100 of them per step: above the flat gate, where the flat
one-shot is as slow as RCCL (113 us against 136 us) and both are far from what
the links can do.

## Design

`VLLM_RDNA_AR_HIER=auto` (or `1`) enables a topology-aware mode for a group that
forms two islands of equal size. Islands come from the sysfs PCI path of each
rank's GPU (host bridge + root port, the first two components below
`/sys/devices`); `VLLM_RDNA_AR_HIER_ISLANDS="0,1,2,3|4,5,6,7"` overrides it for
tests. `auto` applies to world 8 only (the measured topology), `1` to any
2 x G, G = 2..4. Anything else stays on the flat path. The default is `0`.

Three kernels, all on the existing rdna_ar protocol (uncached IPC staging with
two parity slots per source, posted P2P pushes, flags in each rank's own
uncached page, wall-clock-bounded waits with the same abort record, sticky
abort and late-peer record):

- `rdna_ar_oneshot_isl`: the flat exchange with island-grouped arithmetic
  (sum each island's G contributions in local order, round the partial to the
  working dtype, add the two partials). Used below `HIER_MIN_KB`
  (default 20 KiB), where latency is all that matters and a flat exchange is
  already one hop.
- `rdna_ar_hier`: reduce-scatter in the island, exchange across the root,
  all-gather in the island. Each island has X "owners" (local index 0..X-1);
  the payload is cut into per-block chunks and each chunk into X owner
  ranges. Phase 0: every island member pushes owner o's range of its input to
  o, which sums the G contributions in local order in fp32 and rounds the
  island partial to the working dtype. Phase 1: owner o pushes its partial to
  its partner (same local index in the other island), one flow of N/X bytes.
  Phase 2: owner o adds the two partials (fp32, one more rounding) and pushes
  the result to its island peers.
- Size dispatch (the sweeps below pick the values): payloads under
  `HIER_MIN_KB` = 20 KiB use the flat exchange with island arithmetic, larger
  ones `rdna_ar_hier`. Owners per island X = 2 up to 56 KiB, else X = G = 4;
  blocks 4 up to 16 KiB, 16 up to 96 KiB, 32 above.

Bytes: a rank sends 1.5 N inside its island and N / 4 across the root (X = 4);
the flat exchange sends 4 N across. At most four flows cross per direction, each of
N / 4 bytes, so the aggregate cap of 17-19 GB/s applies to a quarter of the
bytes of 16 whole-payload flows. The 1.75 N per rank is the bandwidth-optimal
volume (the ring's 2 (W - 1) / W N).

Blocks are independent pipelines over disjoint element ranges: no grid barrier,
flags per (phase, block, source) at `16 + (phase * 32 + block) * 8 + source` in
the 4 KB flag page (the flat flags use `[0, 8)`), and the last block to finish
bumps the sequence word. A block only waits for the same block of the ranks it
talks to, so blocks that are not resident yet cannot deadlock the resident
ones. Abort records carry phase codes 3 (island reduce-scatter), 4 (cross-root
exchange) and 5 (island all-gather) next to the flat 1 and 2; the abort path
announces the remaining phases so peers never wait on a rank that gave up, and
copies the input to the output like the flat kernel.

Slot reuse across collectives is the flat argument unchanged: collective c + 2
(same parity) starts after c + 1 completed on this rank, every output depends
on every rank's input, so every rank started c + 1, i.e. finished reading the
slots of c, whatever algorithms c, c + 1 and c + 2 used.

The kernels use no scratch memory (hipcc `-Rpass-analysis=kernel-resource-usage`:
29-31 VGPRs for `rdna_ar_hier`, 42-56 for `rdna_ar_oneshot_isl`, scratch 0).

### Numerics

A hier-enabled group computes `round(round(sum_island0) + round(sum_island1))`
in every algorithm and every size: all three kernels are bit-identical to each
other, so a batch-size change never flips between two roundings. The legacy
flat sum is `round(sum_rank_order)`, one rounding. The island-grouped result
has one more rounding to the working dtype (and an fp16 island partial can
overflow where the flat fp32 sum would not; the test pins this behaviour).
Measured against an fp64 sum (random payloads of 30720 and 1 Mi elements): max
absolute error 0.83 / 0.96 (fp16) and 3.65 / 7.22 (bf16) for the island
arithmetic, 0.96 / 1.76 (fp16) for RCCL, and every element within the analytic
bound of the island arithmetic.

### Gate, capacity and memory

The hier kernels are bandwidth-efficient up to 1 MiB, so a hier group gates
at `VLLM_RDNA_AR_HIER_MAX_KB` (default 1024, 0 = the flat gate). Staging
capacity follows the larger of the two gates (2 parity slots x 8 sources x
1 MiB = 16 MiB per instance, uncached; the TP and the EP group each have one).
`MAX_KB` / `ONESHOT_KB` keep gating the all-gather, the argmax and every group
the mode does not apply to. The mono kernel (a TP4 experiment path) gates on
`RdnaArDev.max_bytes`, which now holds the staging capacity: with the mode off
that equals the flat gate as before, `auto` never applies to world 4, and mono
uses the flat flags `[0, world)` only, never the hier flags.

The boot self-test covers both new algorithms: a hier-enabled group runs the
flat exchange with island arithmetic and `hier` at the largest size (the gate)
and at odd sizes (fp16 4099, bf16 2049, fp32 1537 elements) on position- and
rank-dependent integers, exact in every dtype, so a chunk that lands in the
wrong place fails the boot test; any failure leaves the group on RCCL.

### Knobs

| variable | default | meaning |
|---|---|---|
| `VLLM_RDNA_AR_HIER` | `0` | `0` legacy; `auto` 2 x 4 (world 8) only; `1` any 2 x G |
| `VLLM_RDNA_AR_HIER_MAX_KB` | 1024 | one-shot gate of a hier group (0 = flat gate) |
| `VLLM_RDNA_AR_HIER_MIN_KB` | 20 | below it the flat exchange with island arithmetic |
| `VLLM_RDNA_AR_HIER_ALGO` | `auto` | `flat` (island arithmetic) or `hier` forces one algorithm |
| `VLLM_RDNA_AR_HIER_X` | 0 (by size) | owners per island, 1..G |
| `VLLM_RDNA_AR_HIER_BLOCKS` | 0 (by size) | blocks per launch, 1..32 |
| `VLLM_RDNA_AR_HIER_ISLANDS` | sysfs | rank to island map, `0,1,2,3\|4,5,6,7` |

## Correctness

CPU, `tests/distributed/test_rdna_ar.py`: 64 passed on the box. The 11 new test
functions cover the island key (host bridge + root port from the sysfs path),
island classification, the override syntax and what it rejects, the env values
and bad values, the plan rules (which groups get the mode, gate, capacity,
`auto` only at world 8), the text of the hier abort records (phases 3-5), the
lost-write marker for a hier flag wait, and that the new kernels and ops are
wired.

GPU, `tests/distributed/test_rdna_ar_hier_gpu.py`, one process per GPU, every
result compared bit for bit with a CPU reference of the island arithmetic
(each island summed in rank order in fp32 and rounded to the dtype, then the
two partials added and rounded again):

- Algorithms: flat exchange with island arithmetic, `hier` with 1-4 owners and
  any block count, and the size dispatch; the flat algorithm of the same
  instance still gives the legacy rank-order result.
- Sizes: 4, 8, 15, 30, 60, 120, 256 KiB, 1, 5 and 20 MiB, each also with 1, 3
  and 7 extra elements (ragged tail), in fp16, bf16 and fp32, plus misaligned
  input and output views. 5 and 20 MiB run through the forced-algorithm op;
  the production dispatch hands them to RCCL.
- Payloads: order-sensitive magnitudes, zeros, -0, equal values, subnormals,
  an fp16/bf16 island sum past the dtype range while the total is in range,
  and NaN / Inf / signed zeros mixed in. Integer payloads must equal RCCL's
  `all_reduce`; random payloads must stay within the rounding bound of the
  island arithmetic against an fp64 sum.
- CUDA graph: capture once per algorithm, replay with new inputs.
- Soak: back-to-back collectives mixing algorithms and sizes, closed-form
  integer payloads, checked on the device (`VLLM_RDNA_AR_HIER_TEST_LOAD=1`
  adds a GEMM on a side stream), and `rdna_ar_timeout_info == 0` at the end.

| layout | GPUs | exact results | soak results | result |
|---|---|---|---|---|
| islands `[0,0,1,1]` (forced) | slot A, 4 | 2100 | 800 | passed |
| islands `[0,1,0,1]` (forced) | slot A, 4 | 2100 | 800 | passed |
| `[0,0,0,0,1,1,1,1]` (sysfs, the real 2 x 4) | HIP 1-4, 6-9 | 4200 | 2400 | passed |
| `[0,1,0,1,0,1,0,1]` (override) | HIP 1-4, 6-9 | 4200 | 2400 | passed |

Maximum absolute error against an fp64 sum of random payloads (30720 and
1 Mi elements) on the real 2 x 4: fp16 0.83 / 0.96 for the island arithmetic
(flat exchange and hier agree bit for bit), 0.96 / 1.76 for RCCL; bf16
3.65 / 7.22 against 6.81 / 9.82 for RCCL.

## Latency per call

Method: `benchmarks/kernels/benchmark_rdna_allreduce_hier.py`, 8 ranks on HIP
1-4 and 6-9, fp16, vLLM `init_distributed_environment` and a TP8 group. Each
configuration is a CUDA graph of 4 and of 20 back-to-back calls replayed 40
times; the per-call time is the slope (t20 - t4) / 16, per rank, and a row is
the slowest rank. Three repeats; the table gives the median and, in brackets,
the range. RCCL is `pynccl` in the same graphs with the production RCCL env
(`NCCL_P2P_LEVEL=pxb NCCL_PROTO=Simple RCCL_MSCCL_ENABLE=0
RCCL_P2P_NET_DISABLE=1 RCCL_P2P_BATCH_ENABLE=1`); `HSA_ENABLE_SDMA=0`. Every
row also checks the result (integer payloads, exact). These tables come from a
working copy synced to the sources of commit `10ea484499` (not the exact-SHA
tree) at `GPU_MAX_HW_QUEUES=2`; the rerun on the exact-SHA tree at the
production 4 is queued in the stress block (sets `bench` and `sweep`).

Per call in us, `flat` = the production one-shot, `auto` = hier mode with the
size dispatch:

| KiB | RCCL | flat one-shot | hier `auto` |
|---|---|---|---|
| 4 | 187.6 (185.8-192.7) | 15.8 (15.0-16.2) | 15.6 (15.5-16.5) |
| 8 | 184.8 (182.8-184.9) | 21.5 (18.8-22.7) | 22.7 (21.8-24.1) |
| 16 | 182.8 (174.1-187.1) | 29.6 (29.3-30.1) | 30.9 (29.8-31.1) |
| 24 | 184.3 (177.1-186.1) | 37.9 (37.0-38.6) | 34.3 (33.4-34.7) |
| 32 | 175.6 (175.5-178.2) | 44.1 (43.6-44.6) | 35.2 (34.2-35.5) |
| 48 | 181.4 (172.6-183.9) | 53.1 (51.9-53.2) | 34.7 (34.4-34.8) |
| 64 | 177.6 (177.4-178.7) | 64.3 (64.0-64.8) | 37.3 (37.2-37.9) |
| 128 | 135.8 (132.6-139.3) | 112.7 (111.8-112.8) | 36.2 (36.2-36.4) |
| 256 | 146.1 (145.0-149.9) | 204.6 (203.9-204.9) | 50.2 (49.2-50.5) |
| 512 | 166.1 (166.1-167.0) | 404.1 (402.9-404.7) | 76.9 (76.7-78.5) |
| 1024 | 170.7 (168.7-170.8) | 827.6 (826.0-831.8) | 143.7 (143.3-144.4) |
| 1536 | 213.1 (209.6-213.8) | 1198.4 (1187.7-1203.7) | 212.6 (210.7-224.1) |
| 2048 | 221.8 (220.0-231.1) | 1576.1 (1526.8-1616.3) | 274.8 (271.6-305.8) |
| 5120 | 399.9 (398.4-406.5) | 3899.1 (3849.3-3927.0) | 677.6 (632.7-684.4) |

Reading it:

- Up to about 20 KiB the island arithmetic costs nothing (`auto` = flat
  exchange with island sums): 15.6 against 15.8 us at 4 KiB, 30.9 against
  29.6 us at 16 KiB, inside the spread of both.
- The crossover with the flat one-shot is at 20-24 KiB; from there `auto` is
  flat at 34-39 us up to 128 KiB (three dependent one-way hops of about 7 us
  plus the launch [INFERENCE]) while the flat exchange grows by about 0.8 us
  per KiB: 1.7x faster at 64 KiB, **3.1x at 128 KiB (the MTP2 c=8 verify
  size, 120 KiB)**, 4.1x at 256 KiB, 5.3x at 512 KiB, 5.8x at 1 MiB.
- Against RCCL (180 us at 4-64 KiB, 136-170 us from 128 KiB to 1 MiB) `auto` is
  4.8-5.4x faster at 24-64 KiB, 3.8x at 128 KiB, 1.2x at 1 MiB and ties at
  1.5 MiB. From 2 MiB RCCL wins (1.2x at 2 MiB, 1.7x at 5 MiB): both move the
  bandwidth-optimal 1.75 N per rank, RCCL sustains more of the link. The gate
  of a hier group is therefore 1 MiB; prefill chunks (2 MiB and up) stay on RCCL
  and TTFT is not touched by this change.

Owners and blocks (`hier:xN:bM` = N owners per island, M blocks; per call in
us, forced algorithm), the sweep behind the dispatch table:

| KiB | flat-isl | x1:b8 | x2:b4 | x2:b8 | x2:b16 | x4:b4 | x4:b8 | x4:b16 | x4:b32 |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 22.6 (21.4-22.7) | 32.6 (31.9-32.8) | 31.6 (31.3-32.1) | 31.4 (31.1-31.6) | 31.6 (31.5-31.9) | 37.3 (37.1-39.1) | 36.6 (35.6-36.7) | 37.1 (36.9-37.4) | 36.4 (36.1-36.8) |
| 16 | 30.0 (29.8-31.3) | 34.1 (32.4-34.7) | 32.3 (31.7-32.8) | 33.5 (33.2-34.3) | 33.1 (32.2-34.8) | 38.8 (38.0-39.0) | 37.7 (36.8-37.8) | 37.5 (36.9-37.8) | 37.5 (37.2-38.1) |
| 24 | 38.5 (37.8-39.2) | 35.8 (35.0-35.9) | 35.2 (34.4-35.9) | 33.6 (33.5-35.5) | 33.0 (32.9-33.1) | 37.9 (37.5-38.0) | 39.3 (38.5-39.3) | 38.1 (37.6-38.6) | 37.2 (37.0-38.4) |
| 32 | 43.4 (41.8-43.7) | 37.8 (36.9-37.8) | 33.9 (33.7-35.4) | 34.8 (34.8-35.8) | 34.4 (34.0-35.7) | 39.5 (37.7-39.6) | 37.8 (37.2-37.9) | 37.2 (36.7-39.2) | 37.3 (37.1-37.9) |
| 48 | 52.6 (52.6-53.1) | 44.0 (43.2-44.1) | 36.9 (36.5-37.6) | 36.5 (35.9-36.7) | 36.0 (34.9-36.4) | 38.4 (37.2-39.1) | 38.5 (38.0-38.9) | 39.4 (39.2-39.7) | 38.2 (37.2-38.5) |
| 64 | 64.4 (64.2-65.0) | 52.9 (50.3-52.9) | 38.4 (37.6-38.5) | 38.7 (38.2-38.9) | 38.2 (38.0-38.6) | 39.6 (38.8-39.9) | 38.9 (37.7-39.7) | 37.2 (36.8-39.4) | 37.9 (37.6-38.0) |
| 128 | 112.9 (112.7-112.9) | 66.5 (65.2-66.5) | 49.2 (48.6-52.9) | 48.3 (48.1-49.6) | 49.8 (49.6-50.5) | 45.8 (45.6-45.9) | 42.5 (42.5-43.3) | 39.7 (39.5-40.6) | 35.8 (35.0-36.5) |
| 256 | 204.9 (204.4-205.0) | 100.9 (100.6-101.7) | 76.1 (75.8-76.2) | 75.1 (74.8-76.8) | 75.1 (74.4-75.5) | 59.0 (58.2-60.0) | 59.2 (58.1-59.5) | 56.7 (56.4-57.3) | 50.3 (50.1-50.3) |
| 512 | 404.3 (403.4-404.4) | 173.0 (172.8-173.3) | 124.5 (124.4-125.9) | 122.8 (122.5-123.2) | 120.8 (119.8-122.3) | 89.9 (88.2-90.0) | 82.9 (81.8-84.5) | 80.3 (79.7-80.5) | 77.9 (77.3-78.2) |
| 1024 | 831.1 (829.5-832.6) | 315.4 (315.4-318.0) | 228.5 (228.2-228.8) | 221.4 (220.9-221.9) | 219.3 (218.8-219.6) | 168.6 (168.5-168.7) | 160.7 (159.3-161.2) | 148.2 (148.0-151.0) | 142.3 (141.4-144.0) |

Up to 64 KiB two owners match or beat four (the payload per flow is tiny, so
fewer flows per phase help [INFERENCE]); from 128 KiB four owners with 32
blocks win and stay the best to 1 MiB. With four owners each cross-root flow
carries N / 4, at most 64 KiB up to a 256 KiB payload, the per-flow size below
which the audit saw no collapse with 4 concurrent pairs [INFERENCE: why four
owners scale]. One owner per island (a single cross-root flow of the whole
island sum) is 1.9-2.2x slower than four from 128 KiB to 1 MiB; the owner's
own island links carry 3 N in and 3 N out [INFERENCE: the likely limit].

## Robustness

Tool: `tools/rdna/rdna_ar_stress.py`, extended for hier: `--hier 0|1|auto`,
`--islands`, `--hier-algo`, `--hier-x`, `--hier-min-kb`, `--mixed` (three payload
sizes in one graph so both algorithms and both staging parities alternate),
`--progress-s` progress lines, the stall injection of the flat tool
(`--stall-rank/--stall-ms/--stall-every`) and the traffic generators
`p2p host mix cross xmix`. Every collective's input changes on every replay, every
result is checked on the device, and a wedge, a late peer, a wrong sum, a hang
and a KFD `evicted_ms` jump are counted per cell.

Observed: the shakedown (09:38-09:46 box clock, working copy, TP8, `--hier 1
--mixed`, no traffic) ran the 24 / 128 / 512 KiB cells of 60,416 collectives
each with 0 wedges, 0 hangs, 0 mismatches, 0 late peers and 0 ms evicted
(82.9 / 85.8 / 119.5 us per collective, in-graph check included). The first
`xmix` cell (cross-root plus host traffic, 24 KiB) did not finish 60k
collectives in 7 minutes and was stopped by hand: collectives under that
traffic are about two orders of magnitude slower [INFERENCE: head-of-line
blocking of the flags behind the traffic's posted writes]; the traffic cells
of the block use smaller counts.

The 1 h block is `~/w4a8_runs/hier_ar/stress_block.sh s1` on the exact-SHA tree
`10ea484499` at `GPU_MAX_HW_QUEUES=4`, `--hier auto`, started at 11:16 box
clock. Sets, in order: `none` (4 mixed and 3 single sizes, 2 M collectives per
cell), `algos` (forced `hier` / flat exchange, 1-3 owners), `test` (pytest soak,
10,000 iterations, side-stream GEMM load, both layouts), `late` (a rank sleeps
3 s: late-peer record, no wedge), `abort` (`--wait-ms 300` against a 1.5 s
stall: every injected wedge must be reported, none may hang), `traffic`
(`mix`, `cross`, `xmix`), `g3` (six GPUs, islands `0,1,2|3,4,5`), then the
latency reruns `bench` and `sweep`. Its results are on the box
(`stress/s1/summary.txt`, `stress_report.py stress/s1`) and are **not** part of
this document; pass criterion: 0 hangs, 0 mismatches, 0 unreported wedges, 0 ms
evicted, at least 60 minutes of collectives.

## Serving ABBA (not run)

Plan, fixed before any result: four fresh boots of `flashnext-mtp2` (TP8 / EP8,
MTP 2, compiled, `FULL_AND_PIECEWISE`, `MAXBAT` 2048, `SEQS` 8) on HIP 1-4 and
6-9 from the exact-SHA tree, order A1 B1 B2 A2, A = `VLLM_RDNA_AR_HIER=0`
(recipe flat gate 256 KiB), B = `VLLM_RDNA_AR_HIER=auto`; cells 1k/512 and
16k/1k at c=1 and c=8 with the seeds and the KFD `evicted_ms` / quiet-window
protocol of the HWQ ABBA ([hwq-abba-2026-11-06.md](hwq-abba-2026-11-06.md)).
Ship (default `auto`) only if the ITL median improves beyond the
within-condition spread at c=8, is not worse beyond spread at c=1, TTFT and the
prefill aggregate are not worse beyond spread, probes, PPL and the concurrency
probe pass in all four boots, and the stress block is clean. The harness copies
and the analysis script are on the box (`~/w4a8_runs/hier_ar/abba/`,
`analyze_abba.py`, which reproduces the HWQ ABBA table when pointed at that run).

Expectation [INFERENCE, from the per-call tables]: an MTP2 verify step at c=8
is 24 tokens (120 KiB) and about 100 all-reduces, each about 70 us cheaper
(107 -> 36 us), a few ms of the 41.6 ms TP8 step measured by Tp8Pp2 (one boot,
recipe gate 256 KiB); at c=1 (15 KiB) the dispatch picks the flat exchange and
nothing changes; prefill chunks (2 MiB and up) stay on RCCL, so TTFT is
unchanged.

## Limits

- fp16 only in the latency tables (the exactness tests cover fp16, bf16, fp32).
- Only 2 x 4 was run on the real topology; 2 x 2 is covered by forced islands
  on one slot, 2 x 3 by the `g3` cell (not yet run).
- Numerics: one more rounding than the flat sum (see Numerics); an fp16 island
  partial can overflow where the flat fp32 sum would not. Greedy tokens can
  differ from the flat arm in near-tie cases; the ABBA's PPL and quality
  collect are the check.
- Memory: hier capacity is 16 MiB of uncached staging per rdna_ar instance
  (TP and EP group: 32 MiB per GPU) against 2 MiB at the 128 KiB flat gate.
- Above about 1.5 MiB RCCL is faster (1.7x at 5 MiB); the gate keeps those
  all-reduces on RCCL.

## Reproduce

```bash
# CPU tests
python -m pytest tests/distributed/test_rdna_ar.py -q
# GPU exactness, real 2 x 4 (HIP 1-4 and 6-9) and the 4-GPU forced layouts
python -m pytest tests/distributed/test_rdna_ar_hier_gpu.py -k "29555 or 29557" -x -s -q
python -m pytest tests/distributed/test_rdna_ar_hier_gpu.py -k "29551 or 29553" -x -s -q
# per-call latency
python benchmarks/kernels/benchmark_rdna_allreduce_hier.py --world 8 \
  --sizes-kb 4,16,64,128,512,1024,2048,5120 --configs rccl,flat,auto
# stress
python tools/rdna/rdna_ar_stress.py --sizes-kb 24,128,512 --traffic none,mix \
  --hier auto --mixed --collectives 2000000 --label x --out stress.jsonl
```

## SHAs and data

Code: `feat/rdna-ar-hier` at `10ea484499250ed7dca1f5065ef1d7528af84139` (based
on `rdna_extras` `1ce3de8045`); exact-SHA tree on the box `~/vllm-hier-10ea484499`.
Raw data under `~/w4a8_runs/hier_ar/` on the box: `runs/bench-{a,b,c}.jsonl`
(latency, working copy), `runs/t4`, `runs/t8` (GPU tests),
`runs/stress-shake.jsonl` (shakedown), `stress/s1/` (the 1 h block).
