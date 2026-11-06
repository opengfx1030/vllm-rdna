# One-shot all-reduce: latency floor, variants and the vector finisher (2026-11-02)

Question: `rdna_ar_oneshot` was 104 calls per Flash-Next MTP2 decode step at
about 41 us median (5.3 ms/step, ~11 % of the 47.7 ms M=24 step; profile PB5f,
see `async-stall-profile-2026-11-02.md`). What is its floor, and how much of
the gap to the floor is kernel design rather than PCIe?

Answer: the floor at the serving sizes is set by **three physical numbers**
(below); the old finisher was the only part of the kernel that was far from
them. A 16-byte finisher plus a retuned launch width removes **6.0 us (-16 %)
from the critical path of each 123 KB call** in serving, 0.61 ms per M=24 step
(about 1.3 % of the step), with an unchanged exchange protocol. At 15-31 KB
(c=1 and the three small calls of an M=24 step) serving already sits at 12-17
us, near the floor, and the change is neutral there. Everything that changes
the protocol (per-chunk flags, 16-byte pushes, two-shot) was measured and does
not pay at the sizes serving uses.

Box `par1-cs25`, 4x V620 on HIP 2-5 (PLX-B). Harness
`~/w4a8_runs/lanes/fix/arx/ar_exp.hip` (fork per rank, IPC-mapped uncached
staging like the real thing, hipGraph replay, `wall_clock64` probes);
bit-exactness checked against a host fixed-order fp32 reference for every
variant.

## The floor (measured, not modelled)

| Quantity | Value | How |
|---|---|---|
| One-way P2P flag latency | **6.25-6.4 us** | flag ping-pong between two GPUs; HIP 2,3 (sub-port pair), 2,4 and 3,5 (cross pair) are identical |
| Same, cold (kernel start after a 100-400 us idle or busy gap) | RTT 13.4-14.4 us vs 12.8-13.3 us back to back | link state is not a factor |
| Empty 32x256 kernel launched from a graph | **3.1 us** per launch | graph of 100 empty kernels |
| Push rate | 120 KB to one peer 8.4 us (14.6 GB/s); to three peers 22.1-23.2 us (**16.3 GB/s per GPU aggregate**) | probe timestamps inside the kernel; identical with 2-byte and 16-byte stores |

At 123 KB (M=24) on 4 ranks one rank pushes 369 KB, so the push alone is
~22.5 us; adding one 6.3 us flag hop and ~3 us launch bounds the exchange at
about 32 us before any barrier or finisher cost. The PB5f critical path (last
rank's start to the last rank's end, per call) was 38.9 us mean at 123 KB and
13.5 us at 15-31 KB. The remaining 20-40 us per call that rocprofv3 shows as
"duration" is rank skew (ranks reach the all-reduce 25-40 us apart because
their previous work differs); no all-reduce kernel can remove it.

## Variants measured (4 ranks, graph replay behind a 200 us all-CU busy kernel)

us per all-reduce after subtracting the gap kernel, median of 6 (ABBA x3),
all bit-exact. These are harness numbers: the gap kernel makes small sizes
look slower and the gains look larger than serving shows (see the serving
profiles below):

| Variant | 15 KB | 30 KB | 60 KB | 120 KB |
|---|---:|---:|---:|---:|
| `base`: old exchange, scalar finisher, old widths | 24.1 | 34.1 | 45.2 | 54.9 |
| `v1`: same exchange, 16-byte finisher | 21.3 | 29.5 | 39.4 | 44.9 |
| `x1`: v1 + 16-byte pushes | 21.7 | 23.9 | 37.9 | 49.0 |
| `v3`: per-chunk flags, no grid barrier, last block bumps the sequence | 24.6 | 30.5 | 39.0 | 43.4 |
| `v4`: v3 with the sequence bump before the wait | 24.3 | 28.4 | 37.8 | 41.7 |

- **Shipped: v1 plus a world-specialised unrolled finisher** (`v5`, equal or
  better than v1 everywhere) and new launch widths. The finisher was a scalar
  loop of 2-byte uncached loads with a dependent chain per element; with 16-byte
  loads the W staging reads issue back to back and the reduce phase drops from
  ~7-10 us to ~2 us.
- **Launch width with the vector finisher** (v1, medians of 3): 15 KB
  4/6/8/16/32 blocks = 18.1/18.7/20.1/21.1/20.1 us; 30 KB 6/8/16/32 =
  27.3/27.6/29.4/32.0; 60 KB 28/32/36 = 37.4/38.8/39.2; 120 KB 32/36/48 =
  43.6/42.1/50.9. The old rule (16 blocks above 8 KB) cost 2-3 us at 15 and
  30 KB; new rule 4 / 6 / 32 blocks at <= 8 / <= 32 / above 32 KB.
- **Per-chunk flags (v3/v4) rejected.** They hide the grid barrier (span 5 us
  shorter in the probes) but the per-iteration time only wins at >= 60 KB, by
  1-3 us over v1, and loses 2.4 us at 15 KB. They change the flag protocol that
  all-gather, argmax and the mono K2 all-reduce share.
- **16-byte pushes (x1) not shipped.** -5.6 us at 30 KB but +4.1 us at 120 KB:
  one thread issuing all three destinations' stores at once defeats the
  rank-staggered push order that keeps each destination written by one source
  at a time. Neither size is a serving cell (c=1 is M=3, c>=6 is M=24).
- **Two-shot (reduce-scatter + all-gather) not built.** Egress drops to 1.5N
  (184 KB, 11.3 us) but costs a second 6.3 us hop, a second handshake and a
  grid barrier between the phases; the estimate breaks even at ~60 KB and wins
  only above ~256 KB, the gate. (The earlier two-shot also returned zeros
  above the gate; see `rdna-ar-twoshot.md`.)
- **`out=` variant deferred.** The GDN post-all-reduce copy is
  `output[:num_tokens].copy_(result)` in the opaque `_full_forward`; removing it
  needs `out=` through `RowParallelLinear` and every `forward_hip` return path,
  for 36 copies/step (~0.12 ms, 0.25 % of the step), below serve noise.
- **Fusion with the residual add / norm not attempted.** The residual add is
  already inside the fused add-RMSNorm kernel, so only the launch (3.1 us) is
  left to save per fusion; the row-wise norm needs a different block layout than
  the finisher and a new multi-rank exactness argument.

## What shipped

- `rdna_ar_sum_vec<T, W>` (`csrc/rocm/rdna_allreduce.cuh`): one 16-byte load
  per rank slice, fp32 sum in rank order 0..W-1 starting from 0, one 16-byte
  store. W = 2, 4, 8 are unrolled; any other world, 16-byte-misaligned
  input/output and the `n % (16 / sizeof(T))` tail use the scalar loop, so the
  result is bit-equal to the scalar finisher everywhere. The exchange (push,
  grid barrier, flags, sticky abort, `report` records, late-peer record) is
  unchanged, so the wedge check, all-gather, argmax and the mono K2 all-reduce
  are untouched.
- `rdna_ar_reduce_blocks` (`csrc/rocm/rdna_allreduce.cu`): 4 / 6 / 32 blocks at
  <= 8 / <= 32 / above 32 KB for all-reduce only; `VLLM_RDNA_AR_BLOCKS` still
  caps it.
- `VLLM_RDNA_AR_VEC=0` restores the scalar finisher and the old widths (A/B and
  rollback), echoed in the `rdna_ar: one-shot all-reduce active` log line.

Harness A/B of the production kernel (`prod`) against the scalar kernel
(`base`), ABBA x3, 4 ranks, same conditions as the table above (harness
only; the serving critical paths are in the next section):

| Payload | VEC=0 | default | Delta |
|---|---:|---:|---:|
| 7.5 KB | 29.9 | 15.9 | -14.0 us |
| 15 KB (M=3) | 24.3 | 21.8 | -2.5 us |
| 30 KB (M=6) | 33.8 | 28.7 | -5.1 us |
| 60 KB (M=12) | 43.3 | 37.8 | -5.5 us |
| 120 KB (M=24) | 55.1 | 44.8 | -10.3 us |

## Exactness

`tests/distributed/test_rdna_ar_allreduce_gpu.py` (2 and 4 ranks): bit-equal to
the fixed-order fp32 reference (fp16, bf16, fp32; 21 sizes from 1 to 131072
elements including odd lengths and 2-byte-misaligned views; 48 collectives
issued back to back between syncs), integer-valued payloads equal to RCCL's
all_reduce at every size, CUDA-graph replays with new inputs, and a soak that
checks every collective against a device-side reference (RCCL all-gather of
the inputs summed in rank order). Final build, 4 ranks: 2,880 reference results
plus 600,000 soak results, 0 mismatches; 2 ranks: 720 plus 100,000.
`VLLM_RDNA_AR_TEST_ITERS`, `_RCCL_ITERS`, `_SOAK` scale the loops.

## Critical path per call in serving (rocprofv3) and in the harness

An M=24 decode step makes 104 all-reduce calls: **101 at 123 KB** (grid 8192)
and 3 at 15-31 KB (positions 3-5 of the step). At c=1 (M=3) every call is
15 KB. The kernel duration rocprofv3 reports includes waiting for late ranks,
so the number that matters is the critical path: last rank's end minus last
rank's start, with the four GPUs' clocks aligned by the collectives' ends
(local median offset; `ar_skew3.py` / `ar_pos_cp.py` in the lane directory).

Serving, same load and same tree with and without the change (hwq 2; PB5f is
the previous batch's tree, PB6o is `VLLM_RDNA_AR_VEC=0`, PB6 the default):

| Profile, step type | Calls | Critical path, mean / median |
|---|---:|---:|
| PB5f, M=24 steps, the 101 calls at 123 KB | 13,534 | 38.9 / 38.8 us |
| PB6o, M=24 steps, the 101 calls at 123 KB | 23,328 | 38.1 / 38.0 us |
| PB6, M=24 steps, the 101 calls at 123 KB | 22,018 | **32.1 / 32.4 us** |
| PB6o, M=24 steps, the 3 calls at 15-31 KB (positions 3, 4, 5; means) | 231 each | 17.5, 16.4, 17.6 us |
| PB6, same 3 positions (means) | 218 each | 17.2, 17.6, 17.2 us |
| PB6oc1, c=1, every call 15 KB, old: 16 blocks, scalar | 15,515 | 12.4 / 11.8 us |
| PB6c1, c=1, every call 15 KB, new: 6 blocks, vector | 32,031 | 12.1 / 11.4 us |

At 123 KB the change removes **6.0 us per call (-16 %)** from the critical
path: 101 x 6.0 us = **0.61 ms per M=24 step** (the sum of the 104 positions'
mean critical paths is 3.903 ms in PB6o and 3.290 ms in PB6), about 1.3 % of
the 47.9 ms step. Rank 3's summed all-reduce kernel duration per M=24 step is
5.389 ms (PB5f), 5.233 ms (PB6o), 4.927 ms (PB6). At 15 KB (c=1) and at the
three 15-31 KB calls of an M=24 step the real critical path is already
12-17 us, close to the floor (3.1 us launch + 6.3 us hop + the push of the
three slices at 16 GB/s), and the change moves it by -0.3 us at c=1 and by
-0.4 to +1.2 us at the three positions: neutral there, and no regression from
the narrower launch.

The profiled step period did not move: PB6o 231 steps, period mean 47.93 /
median 48.11 ms, busy 40.35 ms; PB6 218 steps, 47.86 / 48.08 ms, busy
40.33 ms. Between the two boots `moe_gemm_q4_kernel_rdna2` alone changed by
+0.28 ms per step (8.108 to 8.387 ms; PB5f had 7.849 ms with the old kernel),
a boot-to-boot swing of unrelated kernels as large as the all-reduce gain, so
a pair of profiles cannot show a 1 % step change. The per-call critical path
is the clean measurement; the step-level figure above is an accounting
estimate.

Harness (`ar_exp cp`, 4 ranks, graph replay behind a 200 us all-CU gap kernel,
two repeats, critical path in us):

| Payload | Old (scalar, old widths) | Shipped (vector, 4 / 6 / 32 blocks) | Delta |
|---|---:|---:|---:|
| 7.5 KB | 19.6 / 19.8 | 13.3 / 13.6 / 13.8 / 10.4 (4 samples, noisy) | about -6 |
| 15 KB (M=3) | 17.5 / 17.8 | 14.9 / 14.9 | -2.7 |
| 30 KB (M=6) | 29.2 / 29.6 | 20.2 / 20.2 | -9.2 |
| 60 KB (M=12) | 35.7 / 35.7 | 31.3 / 31.3 (32.9 / 32.5 in the width sweep) | -2.8 to -4.4 |
| 120 KB (M=24) | 45.6 / 45.6 | 40.2 / 40.1 | -5.4 |

The harness agrees with serving at 120 KB (-5.4 us against -6.0 us) but
**overstates the small sizes**: the 200 us all-CU gap kernel in front of every
collective inflates the absolute numbers (15 KB: 17.5 us old in the harness
against 12.4 us in serving) and the gain (-2.7 us against -0.3 us). The
harness found the finisher cost and picked the widths; the serving profiles
are the evidence for what the change is worth, and they say 15-31 KB is
neutral. Width sweep with the vector finisher (harness, critical path in us,
two repeats): 30 KB 4/6/8/16/32 blocks = 22.4/20.2/22.0/22.8/24.5 and
22.0/20.2/22.5/22.5/24.7; 15 KB 15.2/14.9/15.8/16.0/15.7 and
14.6/14.9/15.4/16.0/14.1; 60 KB 6 blocks 31.2 against 32.9/32.5 for 32 blocks
(inside the repeat-to-repeat spread of about 1.5 us, so the rule above 32 KB
is left at 32); 120 KB 32 blocks 40.2 against 41.9 for 6.

The in-serving gain at 60 KB (M=12) and 30 KB (M=6) was not profiled; by the
harness it lies between the neutral 15-31 KB result and the measured 123 KB
result.

## Serving proof (Flash-Next MTP2 TP4, HIP 2-5)

Recipe `flashnext-mtp2.env`, `GPU_MAX_HW_QUEUES` 2 (the recipe's setting at the time; 4 since 2026-11-06,
[hwq-abba-2026-11-06.md](hwq-abba-2026-11-06.md))
and 4, PLE int4, prefix caching, `serve_validate.sh` cells. Format: prefill
aggregate tok/s, decode aggregate tok/s, TTFT p50 / p99 ms, ITL p50 / p99 ms.

| Cell | b5h2f (old, hwq 2) | b6h2 (new, hwq 2) | b5h4f (old, hwq 4) | b6h4 (new, hwq 4) |
|---|---|---|---|---|
| c1 | 2002.3, 77.28, 505 / 511, 23.1 / 23.8 | 2017.7, 81.79, 506 / 508, 22.9 / 23.5 | 2038.5, 71.59, 491 / 502, 23.0 / 23.6 | 2034.5, 75.95, 495 / 503, 22.9 / 23.5 |
| c6 | 2068.1, 198.03, 651 / 2971, 44.2 / 496.0 | 2087.1, 213.97, 646 / 2944, 43.3 / 493.6 | 2100.3, 206.04, 646 / 2925, 44.3 / 491.8 | 2088.8, 206.07, 647 / 2941, 43.8 / 495.0 |
| c8 | 2064.9, 251.38, 669 / 3967, 47.7 / 502.9 | 2074.8, 246.81, 662 / 3948, 46.6 / 499.6 | 2077.0, 247.19, 665 / 3944, 47.6 / 500.7 | 2069.0, 245.75, 665 / 3959, 46.9 / 500.7 |
| 16k c8 | 2032.6, 85.22, 12359 / 64483, 49.1 / 555.5 | 2032.9, 94.06, 12406 / 64474, 48.1 / 556.6 | 2035.1, 91.49, 13181 / 64407, 48.9 / 556.6 | 2033.6, 94.22, 9231 / 64452, 48.9 / 557.3 |

Decode tok/s follows the sampled acceptance (1.87-2.25 per boot) and is not
comparable across boots; ITL p50 is the clean signal. Against the previous
batch's tree, ITL p50 at hwq 2 is -0.9 / -2.0 / -2.3 / -2.0 % (c1 / c6 / c8 /
16k c8) and at hwq 4 -0.4 / -1.1 / -1.5 / 0.0 %. Prefill throughput, TTFT at
1k and the ITL p99 (the prefill-interference tail) are unchanged within noise;
the 16k c8 TTFT p50 moves between 9.2 and 13.2 s from boot to boot with no
pattern while its p99 stays at 64.4-64.7 s.

Probes, new build at hwq 2: greedy 3/3 PASS, PPL 4.76 PASS, prefix cache 20.1x
(cold 5.95 s, warm 0.30 s), concurrency probe PASS (0 failing, 15 near-tie
warnings, 33 of 48 identical), mixed stall probe NO-GATES with solo TTFT 4k
1.82 s and 16k 7.34 s (previous tree 1.85 s and 7.48 s), probe after load PASS.
hwq 4: greedy 3/3, PPL 4.83, prefix 21.4x, concurrency 0 failing, probe after
load PASS.

**What the serve numbers can and cannot show.** The critical-path accounting
predicts about -0.6 ms per M=24 step (-1.3 %) and nothing at c=1, a smaller
effect than the boot-to-boot noise of this box. Same-tree controls at hwq 2
(`VLLM_RDNA_AR_VEC=0`) in ABBA order, c1 / c6 / c8 ITL p50: new b6h2
22.9 / 43.3 / 46.6, old b6h2o 24.2 / 45.8 / 48.7, old b6h2ob 24.2 / 45.6 /
48.6, new b6h2b 25.2 / 46.2 / 49.0 ms. Pooled, new is -0.6 / -2.1 / -1.7 %
against old, but the two new boots differ from each other by 10 % at c1 (the
15-minute load average on the 96-core host was 17 at the end of b6h2b, with
compiler jobs and another server boot running), and since c=1 is neutral at
the kernel level the -0.9 % at c1 against the previous tree is noise too.
The serve runs therefore do not measure the end-to-end gain; the per-call
critical path does, and the step-level figure is an accounting estimate
(-0.6 ms at M=24, 0 at c=1). What the serve runs establish is that nothing
regressed: probes, prefill, TTFT, tails and the stall probe are unchanged.

## Wedge detection

The exchange, flags, `report` records and the sticky abort are untouched; only
the final summation and the launch width changed. `tests/distributed/test_rdna_ar_gpu.py`
(wedge, abort, sticky, late peer) passes on the final build: 4 passed;
`test_rdna_ar_gather_argmax_gpu.py` 2 passed; `tests/distributed/test_rdna_ar.py`
17 passed.
