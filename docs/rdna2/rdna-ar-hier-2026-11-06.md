# Hierarchical `rdna_ar` for two PCIe switch islands (TP8): 2026-11-06

Status: **shipped, default `auto`.** `VLLM_RDNA_AR_HIER=auto` (the default, with
`VLLM_RDNA_AR=1`) turns on a topology-aware all-reduce for a group of 8 ranks
that forms two PCIe switch islands (TP8 and EP8 on the 2 x 4 V620 box); every
other group keeps the flat path, and `0` turns the mode off. Below 20 KiB (c=1
decode) the dispatch runs the old flat one-shot unchanged. Measured in
production compiled mode on Flash-Next MTP2: the ITL median at c=8 improves by
10 % (4.2 ms).

| item | state |
|---|---|
| Correctness at TP8, 4 KiB to 20 MiB, fp16 / bf16 / fp32 | proven: bit-exact against a CPU reference on the real 2 x 4 islands and on an interleaved override (7680 exact + 80,000 soak results per layout on the shipped commit), forced 2 + 2 islands on 4 GPUs and 3 + 3 on 6 GPUs pass, 11 new CPU test functions |
| Per-call latency | measured on the exact-SHA tree at the production queue count: 3.1x faster than the flat one-shot at 128 KiB (the MTP2 c=8 verify size), 5.9x at 1 MiB, 3.6x faster than RCCL at 128 KiB; the same flat kernel below 20 KiB; RCCL wins from 2 MiB (the gate of a hier group is 1 MiB) |
| At least 1 h of stress with 0 wedges | proven: 87.6 min and 20.5 M collectives on `10ea484499`, then 6.1 min and 2.1 M collectives on the shipped dispatch: 0 hangs, 0 mismatches, 0 ms KFD eviction, and no wedge but the 9 injected ones, all reported |
| Serving ABBA, Flash-Next TP8 MTP2, compiled | **won**: ITL median at c=8 -10.0 % at 1k and at 16k context, beyond the spread; TTFT and prefill unchanged; probes, PPL and the concurrency probe pass in all four boots. The tested tree cost +0.5 % (0.09 ms) ITL at c=1, 1k; the shipped commit runs the old kernel there (no new ABBA) |
| Default | `auto` |

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
18.7 GB/s at 128 KiB and 20-21 GB/s from 1 to 5 MiB, the size of the cross-root
aggregate the audit measured for 3-4 concurrent pairs (17-19 GB/s) [INFERENCE:
the same limit, with 16 flows instead of 4 pairs]. The time is set by the
cross-root path, not by the intra-switch links. The RCCL ring has the opposite
profile: bandwidth-optimal, one cross-root flow per direction per ring step, but
eight sequential steps, 180 us from 4 to 64 KiB. The flat one-shot beats RCCL
below about 150 KiB and loses above it (the TP8 audit of Tp8Pp2, on
`rdna_extras` as `docs/rdna2/tp8-pp2-flashnext-2026-11-06.md`, found 150-160 KiB).

An MTP2 verify step at c=8 reduces M = 24 rows of 2560 fp16, 120 KiB per
all-reduce, about 100 of them per step: the size at which the flat one-shot
(112 us) is nearly as slow as RCCL (129 us), and both are far from what the
links can do.

## Design

`VLLM_RDNA_AR_HIER=auto` (the default; `1` too) enables a topology-aware mode
for a group that forms two islands of equal size (and needs `VLLM_RDNA_AR=1`).
Islands come from the sysfs PCI path of each rank's GPU (host bridge + root
port, the first two components below `/sys/devices`);
`VLLM_RDNA_AR_HIER_ISLANDS="0,1,2,3|4,5,6,7"` overrides it for tests. `auto`
applies to world 8 only (the measured topology) and is silent elsewhere, `1` to
any 2 x G, G = 2..4 (it warns when the group is not two islands), `0` turns the
mode off. Anything else stays on the flat path. An invalid value turns the mode
off with a warning.

Three kernels, all on the existing rdna_ar protocol (uncached IPC staging with
two parity slots per source, posted P2P pushes, flags in each rank's own
uncached page, wall-clock-bounded waits with the same abort record, sticky
abort and late-peer record):

- `rdna_ar_oneshot_isl`: the flat exchange with island-grouped arithmetic
  (sum each island's G contributions in local order, round the partial to the
  working dtype, add the two partials). The size dispatch does not run it: the
  first serving ABBA used it below `HIER_MIN_KB` and paid +0.09 ms per decode
  step at c=1 for its +1.1 us per call at 16 KiB, so the dispatch now keeps the
  old flat one-shot there. It stays as the forced algorithm
  `VLLM_RDNA_AR_HIER_ALGO=flat` for tests, benchmarks and numerics comparisons.
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
  `HIER_MIN_KB` = 20 KiB use the old flat one-shot (`rdna_ar_oneshot`, the
  same kernel and block count as a group without the mode), larger ones
  `rdna_ar_hier`. Owners per island X = 2 up to 56 KiB, else X = G = 4;
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

Below `HIER_MIN_KB` the dispatch result is the legacy flat sum,
`round(sum_rank_order)`, one rounding. From `HIER_MIN_KB` to the gate it is
`round(round(sum_island0) + round(sum_island1))`: `rdna_ar_hier` and the forced
flat exchange with island arithmetic are bit-identical to each other, for any
owner and block configuration. The island-grouped result has one more rounding
to the working dtype (and an fp16 island partial can overflow where the flat
fp32 sum would not; the test pins this behaviour). A batch-size change across
20 KiB therefore changes the summation order, within the bounds below.
Measured against an fp64 sum (random payloads of 30720 and 1 Mi elements): max
absolute error 0.76 / 0.93 (fp16) and 4.75 / 6.90 (bf16) for the island
arithmetic, 1.25 / 1.96 (fp16) and 5.54 / 9.14 (bf16) for RCCL, and every
element within the analytic bound of the island arithmetic.

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
| `VLLM_RDNA_AR_HIER` | `auto` | `auto` 2 x 4 (world 8) only, silent elsewhere; `1` any 2 x G, warns; `0` off |
| `VLLM_RDNA_AR_HIER_MAX_KB` | 1024 | one-shot gate of a hier group (0 = flat gate) |
| `VLLM_RDNA_AR_HIER_MIN_KB` | 20 | below it the old flat one-shot |
| `VLLM_RDNA_AR_HIER_ALGO` | `auto` | `flat` (island arithmetic) or `hier` forces one algorithm |
| `VLLM_RDNA_AR_HIER_X` | 0 (by size) | owners per island, 1..G |
| `VLLM_RDNA_AR_HIER_BLOCKS` | 0 (by size) | blocks per launch, 1..32 |
| `VLLM_RDNA_AR_HIER_ISLANDS` | sysfs | rank to island map, `0,1,2,3\|4,5,6,7` |

## Correctness

CPU, `tests/distributed/test_rdna_ar.py`: 65 passed on the box (shipped commit). The 11 new test
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
  any block count, and the size dispatch (the legacy rank-order result below its
  threshold, the island result from it); the flat algorithm of the same
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

| layout | GPUs | exact results | soak results | tree | result |
|---|---|---|---|---|---|
| islands `[0,0,1,1]` (forced) | slot A, 4 | 3840 | 8000 | `2d0f6ccdf9` | passed |
| islands `[0,1,0,1]` (forced) | slot A, 4 | 3840 | 8000 | `2d0f6ccdf9` | passed |
| `[0,0,0,0,1,1,1,1]` (sysfs, the real 2 x 4) | HIP 1-4, 6-9 | 7680 | 80,000 | `2d0f6ccdf9` | passed |
| `[0,1,0,1,0,1,0,1]` (override) | HIP 1-4, 6-9 | 7680 | 80,000 | `2d0f6ccdf9` | passed |
| `[0,0,0,1,1,1]` (override, G = 3) | 6 GPUs | 3150 | 1800 | `2d0f6ccdf9` | passed |

Maximum absolute error against an fp64 sum of random payloads (30720 and
1 Mi elements) on the real 2 x 4: fp16 0.76 / 0.93 for the island arithmetic
(flat exchange and hier agree bit for bit), 1.25 / 1.96 for RCCL; bf16
4.75 / 6.90 against 5.54 / 9.14 for RCCL.

The TP4 flat-path tests (`test_rdna_ar_allreduce_gpu.py`,
`test_rdna_ar_gather_argmax_gpu.py`) pass with the new default on slot A of
`2d0f6ccdf9` (2 + 2 tests): `auto` never applies to a group of 4.

## Latency per call

Method: `benchmarks/kernels/benchmark_rdna_allreduce_hier.py`, 8 ranks on HIP
1-4 and 6-9, fp16, vLLM `init_distributed_environment` and a TP8 group. Each
configuration is a CUDA graph of 4 and of 20 back-to-back calls replayed 40
times; the per-call time is the slope (t20 - t4) / 16, per rank, and a row is
the slowest rank. Three repeats; the table gives the median and, in brackets,
the range. RCCL is `pynccl` in the same graphs with the production RCCL env
(`NCCL_P2P_LEVEL=pxb NCCL_PROTO=Simple RCCL_MSCCL_ENABLE=0
RCCL_P2P_NET_DISABLE=1 RCCL_P2P_BATCH_ENABLE=1`); `HSA_ENABLE_SDMA=0`. Every
row also checks the result (integer payloads, exact). These tables are from the
exact-SHA tree `2d0f6ccdf9` at the production `GPU_MAX_HW_QUEUES=4`, in a quiet
window (14:56-14:58 box clock: no image build, no other GPU job). Earlier runs
on `10ea484499` (13:20-13:26) and on a working copy at `GPU_MAX_HW_QUEUES=2`
agree within a few percent from 24 KiB up (128 KiB: 36.3 and 36.2 us against
35.7 us here; 1 MiB: 141.9 and 143.7 us against 142.0 us).

Per call in us. `flat one-shot` = the production one-shot of a group without
the mode, `flat-isl` = the flat exchange with island arithmetic (forced), `auto`
= the dispatch of the hier mode (the old flat one-shot below 20 KiB, `hier`
from there):

| KiB | RCCL | flat one-shot | flat-isl | hier `auto` |
|---|---|---|---|---|
| 4 | 191.1 (178.5-191.3) | 23.8 (23.0-24.1) | 16.2 (15.8-24.1) | 14.8 (14.6-14.9) |
| 8 | 185.3 (180.7-186.6) | 21.2 (20.6-22.0) | 21.1 (21.0-21.8) | 21.1 (19.8-22.7) |
| 16 | 187.0 (186.8-187.2) | 29.7 (29.6-29.8) | 31.0 (30.6-31.7) | 30.5 (30.0-31.3) |
| 24 | 182.1 (181.9-185.7) | 36.9 (36.4-38.3) | 38.8 (38.0-39.3) | 33.9 (33.8-34.4) |
| 32 | 178.7 (177.8-183.4) | 43.8 (43.4-44.7) | 45.0 (43.1-45.6) | 34.8 (34.5-35.8) |
| 48 | 180.6 (178.7-185.9) | 52.5 (52.4-52.9) | 52.7 (51.0-52.9) | 35.3 (34.4-36.4) |
| 64 | 179.6 (177.1-182.0) | 64.1 (63.9-64.4) | 64.3 (63.9-64.6) | 36.8 (36.6-37.6) |
| 128 | 129.2 (128.9-131.0) | 112.1 (112.1-112.3) | 112.0 (111.9-113.1) | 35.7 (34.7-37.4) |
| 256 | 150.1 (146.6-155.0) | 204.8 (204.4-204.8) | 204.7 (204.5-206.0) | 49.8 (49.5-50.0) |
| 512 | 168.7 (162.8-169.5) | 404.2 (404.0-405.0) | 406.2 (405.7-406.5) | 77.7 (77.7-78.7) |
| 1024 | 168.5 (168.1-169.6) | 832.8 (830.5-833.6) | 834.4 (832.2-835.8) | 142.0 (140.6-144.2) |
| 1536 | 211.4 (204.5-220.5) | 1199.2 (1199.1-1212.2) | 1223.8 (1202.9-1231.0) | 206.5 (203.9-214.1) |
| 2048 | 221.2 (207.8-231.9) | 1582.6 (1544.3-1591.2) | 1593.9 (1565.4-1617.0) | 287.1 (270.2-299.3) |
| 5120 | 398.1 (394.6-407.9) | 3958.5 (3757.7-3973.0) | 3908.8 (3887.7-3943.0) | 636.9 (623.0-668.4) |

Reading it:

- Below 20 KiB `auto` launches the same kernel with the same block count as the
  flat one-shot (`rdna_ar_pick_algo` returns `RDNA_AR_ALGO_FLAT`; the block
  count is a function of the size only), and the exactness test pins its
  arithmetic to the legacy rank-order sum, so a c=1 step is the flat group's by
  construction. The interleaved run below agrees: -0.2 to +0.4 us against a
  repeat-to-repeat spread of 2-3 us. The sequential run of the table shows the
  same kernel 0.8 us apart at 16 KiB (29.7 against 30.5 us) and 9 us apart at
  4 KiB, where the first rdna configuration the process runs is slow in all its
  repeats [INFERENCE: cold clocks]: order, not the kernel.
- The crossover with the flat one-shot is at 20-24 KiB; from there `auto` is
  flat at 34-37 us up to 128 KiB (three dependent one-way hops of about 7 us
  plus the launch [INFERENCE]) while the flat exchange grows by about 0.8 us
  per KiB: 1.1x faster at 24 KiB, 1.3x at 32 KiB, 1.5x at 48 KiB, 1.7x at
  64 KiB, **3.1x at 128 KiB (the MTP2 c=8 verify size, 120 KiB)**, 4.1x at
  256 KiB, 5.2x at 512 KiB, 5.9x at 1 MiB.
- Against RCCL (179-191 us at 4-64 KiB, 129-169 us from 128 KiB to 1 MiB)
  `auto` is 4.9-5.4x faster at 24-64 KiB, 3.6x at 128 KiB, 3.0x at 256 KiB,
  2.2x at 512 KiB, 1.2x at 1 MiB and ties at 1.5 MiB (206.5 against 211.4 us).
  From 2 MiB RCCL wins (1.3x at 2 MiB, 1.6x at 5 MiB): both move the
  bandwidth-optimal 1.75 N per rank, RCCL sustains more of the link. The gate
  of a hier group is therefore 1 MiB; prefill chunks (2 MiB and up) stay on
  RCCL and TTFT is not touched by this change.

Below 20 KiB, flat one-shot against `auto` (the same kernel), interleaved
`flat,auto,auto,flat` after a warm-up pass, 7 repeats per row, two passes: 28
values per cell (15:12 box clock, tree `2d0f6ccdf9`):

| KiB | flat one-shot | hier `auto` | auto - flat |
|---|---|---|---|
| 4 | 15.2 (13.4-16.6) | 15.6 (13.4-17.7) | +0.4 us (+2.4 %) |
| 8 | 21.2 (19.6-24.6) | 21.0 (19.5-24.7) | -0.2 us (-1.1 %) |
| 16 | 30.4 (29.6-32.0) | 30.6 (29.3-31.9) | +0.2 us (+0.7 %) |

Owners and blocks (`hier:xN:bM` = N owners per island, M blocks; per call in
us, forced algorithm), the sweep behind the dispatch table:

| KiB | flat-isl | x1:b8 | x2:b4 | x2:b8 | x2:b16 | x4:b4 | x4:b8 | x4:b16 | x4:b32 |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 21.7 (19.4-21.9) | 31.5 (30.4-33.4) | 31.1 (31.0-31.5) | 31.3 (31.1-31.3) | 31.4 (30.6-31.7) | 37.2 (36.8-37.4) | 36.1 (35.8-36.5) | 38.2 (37.0-38.6) | 36.0 (35.6-37.9) |
| 16 | 29.9 (29.5-30.4) | 32.9 (32.8-33.5) | 32.7 (32.5-33.4) | 32.1 (32.0-32.3) | 33.7 (33.2-34.0) | 38.5 (37.8-38.6) | 37.2 (36.5-37.3) | 37.0 (36.5-37.4) | 37.4 (36.6-37.7) |
| 24 | 38.0 (37.5-38.1) | 34.9 (34.7-35.5) | 34.8 (33.4-35.3) | 34.7 (32.2-35.1) | 33.6 (33.5-34.5) | 38.6 (37.8-39.9) | 38.2 (37.7-38.5) | 39.1 (38.3-39.5) | 38.0 (37.3-38.3) |
| 32 | 43.8 (43.8-44.3) | 36.8 (36.4-37.3) | 35.5 (34.7-36.5) | 34.9 (34.5-35.0) | 34.2 (33.9-35.9) | 38.7 (37.5-38.9) | 37.7 (37.5-39.8) | 37.8 (36.5-39.5) | 38.3 (37.1-38.4) |
| 48 | 52.8 (51.9-54.1) | 43.9 (43.1-44.1) | 36.5 (36.0-37.2) | 36.1 (35.2-36.4) | 35.0 (34.4-35.0) | 38.0 (37.7-38.9) | 39.6 (38.4-39.9) | 38.6 (36.7-38.8) | 37.9 (37.7-40.1) |
| 64 | 64.5 (64.2-64.7) | 51.3 (50.1-52.7) | 38.6 (38.0-39.0) | 38.6 (38.2-38.6) | 38.1 (37.6-38.5) | 39.9 (39.1-40.3) | 39.0 (39.0-39.3) | 37.7 (37.4-38.2) | 36.9 (36.5-37.7) |
| 128 | 111.5 (111.2-111.9) | 66.1 (66.1-66.2) | 50.6 (49.7-51.3) | 48.9 (48.8-50.3) | 49.5 (48.8-49.6) | 45.5 (44.5-47.1) | 43.1 (43.1-43.1) | 39.5 (38.8-40.3) | 35.5 (35.2-35.5) |
| 256 | 204.1 (204.1-204.4) | 100.7 (100.4-101.5) | 75.7 (74.3-76.6) | 75.8 (75.4-76.2) | 75.2 (75.0-75.6) | 58.0 (58.0-58.9) | 58.0 (57.4-58.8) | 57.4 (56.2-58.0) | 49.3 (49.1-50.2) |
| 512 | 404.7 (404.4-405.1) | 172.6 (171.9-172.8) | 125.8 (124.7-127.2) | 123.8 (122.4-125.0) | 119.9 (119.7-120.9) | 88.8 (88.5-90.3) | 82.7 (81.1-83.3) | 81.1 (81.1-82.1) | 78.2 (77.6-78.5) |
| 1024 | 824.5 (823.7-825.6) | 317.1 (315.9-318.5) | 228.7 (228.0-229.4) | 221.3 (221.2-222.0) | 218.5 (218.3-218.8) | 168.6 (167.5-168.6) | 161.7 (161.0-161.8) | 149.1 (147.8-149.3) | 143.5 (142.0-145.4) |

Up to 48 KiB two owners beat four (the payload per flow is tiny, so fewer flows
per phase help [INFERENCE]); from 64 KiB four owners with 32 blocks lead (36.9
against 38.1 us at 64 KiB, 35.5 against 49-51 us at 128 KiB) and stay the best
to 1 MiB. With four owners each cross-root flow carries N / 4, at most 64 KiB up
to a 256 KiB payload, the per-flow size below which the audit saw no collapse
with 4 concurrent pairs [INFERENCE: why four owners scale]. One owner per island
(a single cross-root flow of the whole island sum) is 1.9-2.2x slower than four
from 128 KiB to 1 MiB; the owner's own island links carry 3 N in and 3 N out
[INFERENCE: the likely limit].

## Robustness

Tool: `tools/rdna/rdna_ar_stress.py`, extended for hier: `--hier 0|1|auto`,
`--islands`, `--hier-algo`, `--hier-x`, `--hier-min-kb`, `--mixed` (three payload
sizes in one graph so both algorithms and both staging parities alternate),
`--progress-s` progress lines, the stall injection of the flat tool
(`--stall-rank/--stall-ms/--stall-every`) and the traffic generators
`p2p host mix cross xmix`. Every collective's input changes on every replay, every
result is checked on the device, and a wedge, a late peer, a wrong sum, a hang
and a KFD `evicted_ms` jump are counted per cell.

Results of block `s1` (exact-SHA tree `10ea484499`, `GPU_MAX_HW_QUEUES=4`,
`--hier auto`, both slots, no server, 11:16-12:49 box clock): 28 cell rows,
20,516,864 collectives, 87.6 minutes, **0 hangs, 0 mismatches, 0 ms KFD
eviction, no AER delta, and 6 wedges that were all injected** (the `abort`
cells). Sets: `none` (mixed and single sizes, 2 M collectives per cell),
`algos` (forced `hier` / flat exchange, 1-3 owners), `late` (one rank sleeps
3 s: late-peer records, no wedge), `abort` (`--wait-ms 300` against a 1.5 s
stall: every wedge must be reported, none may hang), `traffic` (`mix`, `cross`,
`xmix`), and the pytest soak and `g3` below.

| cell | KiB | traffic | collectives | wedges | hangs | mismatches | late (max ms) | KFD evicted ms | us/collective | wall |
|---|---|---|---|---|---|---|---|---|---|---|
| none-mixed | 24 | none | 2,000,896 | 0 | 0 | 0 | 0 (0) | 0 | 82.81 | 3.0 min |
| none-mixed | 128 | none | 2,000,896 | 0 | 0 | 0 | 0 (0) | 0 | 84.96 | 3.0 min |
| none-mixed | 512 | none | 2,000,896 | 0 | 0 | 0 | 0 (0) | 0 | 119.13 | 4.2 min |
| none-mixed | 1024 | none | 2,000,896 | 0 | 0 | 0 | 0 (0) | 0 | 162.62 | 5.6 min |
| none-single | 24 | none | 2,000,896 | 0 | 0 | 0 | 0 (0) | 0 | 81.41 | 3.0 min |
| none-single | 128 | none | 2,000,896 | 0 | 0 | 0 | 0 (0) | 0 | 102.76 | 3.6 min |
| none-single | 512 | none | 2,000,896 | 0 | 0 | 0 | 0 (0) | 0 | 178.9 | 6.2 min |
| force-hier | 128 | none | 1,000,448 | 0 | 0 | 0 | 0 (0) | 0 | 102.6 | 1.9 min |
| force-isl | 128 | none | 500,736 | 0 | 0 | 0 | 0 (0) | 0 | 165.83 | 1.6 min |
| force-isl | 512 | none | 500,736 | 0 | 0 | 0 | 0 (0) | 0 | 476.87 | 4.2 min |
| x1 | 128 | none | 1,000,448 | 0 | 0 | 0 | 0 (0) | 0 | 120.93 | 2.2 min |
| x2 | 128 | none | 1,000,448 | 0 | 0 | 0 | 0 (0) | 0 | 101.67 | 1.9 min |
| x2 | 512 | none | 1,000,448 | 0 | 0 | 0 | 0 (0) | 0 | 201.27 | 3.6 min |
| x3 | 128 | none | 1,000,448 | 0 | 0 | 0 | 0 (0) | 0 | 99.49 | 1.9 min |
| late | 24 | none | 100,352 | 0 | 0 | 0 | 98 (2992) | 0 | 500.33 | 1.1 min |
| late | 128 | none | 100,352 | 0 | 0 | 0 | 98 (2992) | 0 | 521.75 | 1.1 min |
| abort | 24 | none | 9,216 | 3 (injected) | 0 | 0 | 0 (0) | 0 | 564.13 | 0.8 min |
| abort | 512 | none | 9,216 | 3 (injected) | 0 | 0 | 0 (0) | 0 | 668.95 | 0.8 min |
| traffic-mix | 24 | mix | 100,352 | 0 | 0 | 0 | 0 (0) | 0 | 6541.99 | 11.2 min |
| traffic-cross | 24 | cross | 50,176 | 0 | 0 | 0 | 0 (0) | 0 | 5827.71 | 5.1 min |
| traffic-cross | 128 | cross | 50,176 | 0 | 0 | 0 | 0 (0) | 0 | 6722.54 | 5.9 min |
| traffic-cross | 1024 | cross | 50,176 | 0 | 0 | 0 | 0 (0) | 0 | 8996.97 | 7.8 min |
| traffic-xmix | 24 | xmix | 12,288 | 0 | 0 | 0 | 0 (0) | 0 | 7724.85 | 1.8 min |
| traffic-xmix | 128 | xmix | 12,288 | 0 | 0 | 0 | 0 (0) | 0 | 9838.95 | 2.3 min |
| traffic-xmix | 1024 | xmix | 12,288 | 0 | 0 | 0 | 0 (0) | 0 | 14105.31 | 3.1 min |

- Three further rows (`force-hier` at 4 KiB, `force-hier-8` and `force-isl-8` at
  8 KiB) have 0 collectives and are not results: the tool sets every gate to the
  cell size, and the boot self-test needs a gate above 8198 bytes (its odd-size
  cases), so `rdna_ar` disabled itself in those cells. Forced algorithms at
  those sizes are covered by the exactness tests, not by the stress.
- `late`: 98 late-peer records of up to 2992 ms per cell, no wedge, no
  mismatch. `abort`: each cell restarted on fresh processes after every wedge
  and ran to its third; all 6 were reported, none hung, no sum was wrong
  outside a wedged attempt.
- Traffic cells are robustness only: under `mix`, `cross` and `xmix` a
  collective costs 5.8-14.1 ms against 0.08-0.16 ms quiet [INFERENCE:
  head-of-line blocking of the tiny flag and payload writes behind the
  traffic's bulk posted writes]. All sums were right, nothing wedged. The
  `traffic-mix` run was stopped after its 24 KiB row to leave time for `g3`.
- Pytest soak (`VLLM_RDNA_AR_HIER_TEST_SOAK=10000`, side-stream GEMM load): both
  8-GPU layouts passed, 7680 exact + 80,000 soak results each, on `10ea484499`
  and again on the shipped `2d0f6ccdf9`. `g3` (six GPUs, islands
  `[0,0,0,1,1,1]`): 3150 exact + 1800 soak results, passed on both trees.

After the dispatch change (block `p1`, exact-SHA tree `2d0f6ccdf9`, same queue
count and tool, 14:49-14:56 box clock) these cells ran on the shipped dispatch.
`post-mixed` puts three payload sizes (N, N/8, N/64) into one graph, so the old
flat one-shot and `hier` (both owner counts) alternate back to back with
alternating staging parities; `post-single` is the flat one-shot alone in a
hier-enabled group (16 KiB).

| cell | KiB | collectives | wedges | hangs | mismatches | late (max ms) | KFD evicted ms | us/collective |
|---|---|---|---|---|---|---|---|---|
| post-mixed | 24 | 500,736 | 0 | 0 | 0 | 0 (0) | 0 | 82.75 |
| post-mixed | 128 | 500,736 | 0 | 0 | 0 | 0 (0) | 0 | 85.44 |
| post-mixed | 512 | 500,736 | 0 | 0 | 0 | 0 (0) | 0 | 119.46 |
| post-single | 16 | 500,736 | 0 | 0 | 0 | 0 (0) | 0 | 80.64 |
| post-late | 24 | 100,352 | 0 | 0 | 0 | 98 (2992) | 0 | 501.71 |
| post-abort | 24 | 9,216 | 3 (injected) | 0 | 0 | 0 (0) | 0 | 564.67 |

Earlier, the shakedown (09:38-09:46, working copy, `--hier 1 --mixed`) had run
the 24 / 128 / 512 KiB cells of 60,416 collectives each clean (82.9 / 85.8 /
119.5 us per collective).

## Serving ABBA

Four fresh boots of `flashnext-mtp2` (TP8 / EP8, MTP 2, compiled,
`FULL_AND_PIECEWISE`, `MAXBAT` 2048, `SEQS` 8, shipped TunableOp rows,
`GPU_MAX_HW_QUEUES=4`) on HIP 1-4 and 6-9 from the exact-SHA tree `10ea484499`,
order A1 B1 B2 A2 (13:25-14:23 box clock), no image build running, KFD
`evicted_ms` unchanged in every cell, 0 invalid runs. A =
`VLLM_RDNA_AR_HIER=0` (the recipe's flat gate of 256 KiB), B =
`VLLM_RDNA_AR_HIER=auto`; the serve logs of B show 16 lines (8 TP + 8 EP ranks)
`hier islands [[0, 1, 2, 3], [4, 5, 6, 7]], algo auto, hier from 20 KB`, those
of A have none. Cells 1k/512 and 16k/1k at c=1 (4 prompts) and c=8 (32 prompts),
greedy, fixed seeds; harness, seeds and quiet-window protocol of
[hwq-abba-2026-11-06.md](hwq-abba-2026-11-06.md). The rule was fixed before the
run: ship only if the ITL median improves beyond the within-condition spread
(max of |A1 - A2| and |B1 - B2|) at c=8, is not worse beyond spread at c=1, TTFT
and prefill do not get worse beyond spread, probes, PPL and the concurrency
probe pass in all four boots, and the stress is clean.

Per arm, and the mean of each arm pair (ms unless noted):

| cell | metric | A1 | A2 | B1 | B2 | mean A | mean B | B vs A | spread A / B | beyond spread |
|---|---|---|---|---|---|---|---|---|---|---|
| 1k c=1 | ITL median | 19.56 | 19.59 | 19.69 | 19.64 | 19.57 | 19.66 | +0.5 % | 0.02 / 0.05 | worse |
| 1k c=1 | ITL p99 | 20.24 | 20.21 | 20.16 | 20.23 | 20.22 | 20.19 | -0.2 % | 0.03 / 0.08 | within |
| 1k c=1 | TTFT median | 338.65 | 343.52 | 341.57 | 341.46 | 341.09 | 341.52 | +0.1 % | 4.87 / 0.11 | within |
| 1k c=1 | TTFT p99 | 418.50 | 425.52 | 420.53 | 421.24 | 422.01 | 420.88 | -0.3 % | 7.02 / 0.71 | within |
| 1k c=1 | output tok/s | 91.96 | 108.08 | 114.82 | 93.02 | 100.02 | 103.92 | +3.9 % | 16.13 / 21.80 | within |
| 1k c=8 | ITL median | 41.56 | 42.11 | 37.60 | 37.71 | 41.84 | 37.65 | **-10.0 %** | 0.54 / 0.12 | better |
| 1k c=8 | ITL p99 | 324.59 | 324.28 | 324.56 | 324.17 | 324.44 | 324.37 | 0.0 % | 0.31 / 0.39 | within |
| 1k c=8 | TTFT median | 466.89 | 468.00 | 455.25 | 454.43 | 467.45 | 454.84 | -2.7 % | 1.11 / 0.82 | better |
| 1k c=8 | TTFT p99 | 2676.84 | 2678.95 | 2677.63 | 2674.97 | 2677.89 | 2676.30 | -0.1 % | 2.11 / 2.66 | within |
| 1k c=8 | output tok/s | 289.43 | 318.06 | 336.50 | 321.92 | 303.75 | 329.21 | +8.4 % | 28.63 / 14.58 | within |
| 16k c=1 | ITL median | 19.99 | 20.06 | 20.05 | 20.13 | 20.03 | 20.09 | +0.3 % | 0.07 / 0.09 | within |
| 16k c=1 | ITL p99 | 20.49 | 21.10 | 20.57 | 20.51 | 20.79 | 20.54 | -1.2 % | 0.61 / 0.06 | within |
| 16k c=1 | TTFT median | 5271.06 | 5269.42 | 5272.17 | 5264.38 | 5270.24 | 5268.27 | 0.0 % | 1.64 / 7.80 | within |
| 16k c=1 | TTFT p99 | 5313.78 | 5297.02 | 5311.04 | 5301.77 | 5305.40 | 5306.41 | 0.0 % | 16.76 / 9.27 | within |
| 16k c=1 | output tok/s | 73.13 | 78.80 | 75.21 | 80.43 | 75.97 | 77.82 | +2.4 % | 5.67 / 5.23 | within |
| 16k c=8 | ITL median | 43.33 | 43.57 | 39.12 | 39.08 | 43.45 | 39.10 | **-10.0 %** | 0.24 / 0.04 | better |
| 16k c=8 | ITL p99 | 382.37 | 383.29 | 382.10 | 382.80 | 382.83 | 382.45 | -0.1 % | 0.92 / 0.70 | within |
| 16k c=8 | TTFT median | 6865.43 | 9123.88 | 6837.50 | 7102.64 | 7994.65 | 6970.07 | -12.8 % | 2258.45 / 265.15 | within |
| 16k c=8 | TTFT p99 | 46379.96 | 46383.41 | 46389.58 | 46110.06 | 46381.69 | 46249.82 | -0.3 % | 3.45 / 279.52 | within |
| 16k c=8 | output tok/s | 122.93 | 124.77 | 124.04 | 127.09 | 123.85 | 125.56 | +1.4 % | 1.84 / 3.05 | within |

Against the rule: at c=8 the ITL median improves beyond the spread in both
cells (-4.18 ms against 0.54 / 0.12 at 1k, -4.35 ms against 0.24 / 0.04 at
16k). At c=1 the 16k cell is within the spread and **the 1k cell is worse by
0.09 ms (+0.5 %) against a 0.05 ms spread**, a miss of "not worse at c=1" by
0.04 ms. TTFT and the prefill aggregate are within the spread everywhere except
the 1k c=8 TTFT median, which is 12.6 ms better. Output tok/s follows the MTP
acceptance length (1k c=1: 1.91 and 2.28 for A, 2.47 and 1.95 for B), which
differs by up to 0.5 between two boots of the same arm, so at this sample size
it is noise; the ITL median is the metric. Probe, PPL probe, concurrency probe
and probe after load pass in all four boots (PPL 4.83, 4.82, 4.83, 4.83 for A1
B1 B2 A2); no new `Runlist is getting oversubscribed`, amdgpu fault / reset,
AER or IOMMU lines in any arm, BMC event log unchanged.

The c=1 miss has a known cause: the tested tree ran the flat exchange with
island arithmetic below 20 KiB, +1.1 us per call at 16 KiB (30.9 against 29.8
us in that tree's microbenchmark) times the roughly 80 all-reduces of a c=1
step. The shipped commit `2d0f6ccdf9` runs the old flat one-shot there (one
line, `rdna_ar_pick_algo`), so the c=1 path is the A arm's kernel; the c=8 path
(24 tokens, 120 KiB, `hier`) is unchanged. No new ABBA was run for this change.

Default-env arm of the shipped code (tree `9b96ecfa7a`: `rdna_extras`
`80059a362c` merged into `2d0f6ccdf9`, the same `_rocm_C`; one fresh boot,
17:06-17:14 box clock, **no `VLLM_RDNA_AR_HIER` in the server environment**,
cells 1k c=1 and c=8, same seeds and protocol): the serve log shows the 16
hier lines (`hier islands [[0, 1, 2, 3], [4, 5, 6, 7]], algo auto, hier from
20 KB`), so `auto` is the default in the server. ITL median 19.54 ms at c=1
(A: 19.56 and 19.59; the tested tree's B: 19.69 and 19.64) and 37.17 ms at c=8
(A: 41.56 and 42.11; B: 37.60 and 37.71); ITL p99 20.17 and 324.67 ms; TTFT
median 344.4 and 453.8 ms; output 109.4 and 310.1 tok/s. Probe, PPL probe
(4.85), concurrency probe, probe after load and the quality collect (0
failing, 25 near-tie of 48 checks) pass, KFD `evicted_ms` +0, no new dmesg
faults. One boot is not an ABBA; it shows the c=1 cell back at the A level and
the c=8 gain kept.

## Limits

- fp16 only in the latency tables (the exactness tests cover fp16, bf16, fp32).
- Only 2 x 4 was run on the real topology; 2 x 2 is covered by forced islands
  on one slot and 2 x 3 by the six-GPU `g3` test (override, not a real island).
- Numerics: from 20 KiB up one more rounding than the flat sum (see Numerics);
  an fp16 island partial can overflow where the flat fp32 sum would not. Greedy
  tokens can differ from the flat arm in near-tie cases; the PPL probe of the
  ABBA (4.82 and 4.83 in all four boots) shows no measurable change.
- Memory: hier capacity is 16 MiB of uncached staging per rdna_ar instance
  (TP and EP group: 32 MiB per GPU) against 2 MiB at the 128 KiB flat gate.
- From 2 MiB RCCL is faster (1.6x at 5 MiB); the gate keeps those all-reduces
  on RCCL.
- Forced `hier` or flat-exchange stress below 16 KiB was not run (the
  self-test gate artifact above); those sizes are covered by the exactness tests.
- The serving ABBA and the 87.6 min stress ran on `10ea484499`, whose dispatch
  used the flat exchange with island arithmetic below 20 KiB; the shipped commit
  differs from it below 20 KiB only, and has its own exactness, stress and
  per-call runs above.

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
# serving ABBA (harness copies on the box: ~/w4a8_runs/hier_ar/abba)
SHA=<tree sha> bash abba_run.sh   # A1 B1 B2 A2 = VLLM_RDNA_AR_HIER 0, auto, auto, 0
```

## SHAs and data

Code on `feat/rdna-ar-hier`: `10ea484499250ed7dca1f5065ef1d7528af84139` (kernels,
tests, tools; the ABBA and the 87.6 min stress ran on this tree; based on
`rdna_extras` `1ce3de8045`), `2d0f6ccdf9fc0928a93a79198085beb5580e053c` (dispatch
below 20 KiB keeps the flat one-shot, default `auto`: the shipped code), and the
merge of `rdna_extras` `80059a362c`. Exact-SHA trees on the box:
`~/vllm-hier-10ea484499`, `~/vllm-hier-2d0f6ccdf9`. Raw data under
`~/w4a8_runs/hier_ar/` on the box: `stress/s1/` (the 87.6 min block;
`stress_report.py stress/s1`), `stress/s2/` (latency on `10ea484499`),
`stress/p1/` (pytest soak, g3, post stress and latency on `2d0f6ccdf9`),
`stress/p2/` (interleaved c=1 microbenchmark, forced 2 + 2 layouts and TP4
regression on `2d0f6ccdf9`), `abba/arms/{A1,B1,B2,A2}` and `abba/report.md`
(serving ABBA), `runs/bench-{a,b,c}.jsonl`, `runs/t4`, `runs/t8` (working copy),
`runs/stress-shake.jsonl`.
