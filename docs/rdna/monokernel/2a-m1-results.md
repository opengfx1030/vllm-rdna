# MonoKernel 2a, milestone M1 (V620 / gfx1030, 2026-10-14)

> Superseded: the reworked kernel (1.15-1.59x standalone), M2, M3 and the
> in-serve timing are in [`2a-results.md`](./2a-results.md). This page records
> the first kernel version.

Standalone result of the FFN-only persistent kernel against today's Flash-Next
decode FFN block, before any integration. **Verdict: the kernel is correct but
does not beat today's block at widths 3 and 24 (0.91x at both). Per the plan
M2/M3 were not started.** Raw results:
`bench_results/2026-10-14_monokernel-2a/m1/` (infra repo); on the box
`~/w4a8_runs/monokernel-2a/`.

## What was built

`csrc/rocm/mono_ffn_rdna2.cu` (`mono_ffn_rdna2`): one launch per decoder layer
replaces everything between the attention all-reduce and the FFN all-reduce
for a pure decode step of 1–24 rows:

| Phase | Work | Today's kernels it replaces |
| --- | --- | --- |
| P1 | HC combine (`res + attn_out * 2 sigmoid(inj/4)`) + grouped Gemma RMSNorm | 2 fills, `_hc_combine_norm` |
| P2 | HC down+inject GEMV (336 x 10240), K split, fp32 atomics | `gemv_act_k` (+ copy) / split-K + Tensile at M > 8 |
| P3 | HC up GEMV + sigmoid + gated mean -> block input | `hc_up_gate_mix_k` / `hc_up_mix_prefill_k` |
| P4 | router GEMV (fp32 logits), shared-expert gate_up, expert gate | router `gemv_f16`, `se_gate_up_silu_k` |
| P5 | fp32 softmax top-10 (redundant per WG), local expert lists; routed gate_up + SwiGLU; shared-expert down | `topkGating`, align, count/sort, index, fills, `moe_gemm_q4` #1, `act_and_mul`, `se_down_gated_k` |
| P6 | routed down, * top-k weight, fp32 atomics | `moe_gemm_q4` #2 |
| P7 | `out = fp16(shared + routed)` (rank partial) | `triton_poi_fused_add` |

Plain launch, 72 WGs x 256 threads (one per CU; the grid cap is computed from
the code object, because `hipOccupancy*` under-reports by 2x on gfx10, see
P0), one stream, graph-capturable (the M1 bench times it inside a hipGraph).
Phases are separated by a grid barrier on a generation counter (agent-scope
atomics, relaxed polling, bounded wait). A timed-out wait writes a sticky
`{code, phase, wg}` record, every WG leaves, and later launches exit at
entry until the record is cleared (fail-stop). No `grid.sync`. The all-reduce
stays outside. EP: experts `[expert_base, expert_base + 128)` are local.

## Correctness (random weights in the checkpoint-derived layouts)

Weights are random in the layouts the current kernels read (symmetric int4
g128, exllama shuffle, synthesized zeros, fp16 HC/router/shared expert). The
reference is the exact op sequence production runs (`hc_combine_norm`,
`rdna_hc_mix`, `rdna_shared_expert`, `rocm_unquantized_gemm_impl`,
`fused_topk`, `moe_align_block_size`, `moe_gptq_gemm_rdna2` x2). The golden is
the same block in fp32 torch from the unshuffled weights. The FFN block is the
same code on GDN and QSA layers; the two input sets differ in the scale of
the attention output (0.5 for "GDN-like", 2.0 for "QSA-like").

| M | inputs | mono vs fp32 golden, max abs (rel) | today vs golden, max abs (rel) | mono vs today, max abs | res_out | routing differs |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | GDN-like | 1.6e-4 (5.9e-4) | 3.0e-4 (1.1e-3) | 2.4e-4 | bit-identical | 0/3 |
| 3 | QSA-like | 5.7e-4 (8.2e-4) | 8.0e-4 (1.2e-3) | 9.8e-4 | bit-identical | 0/3 |
| 6 | GDN-like | 3.3e-4 (8.0e-4) | 6.6e-4 (1.6e-3) | 7.3e-4 | bit-identical | 0/6 |
| 6 | QSA-like | 6.8e-4 (8.7e-4) | 1.0e-3 (1.3e-3) | 9.8e-4 | bit-identical | 0/6 |
| 12 | GDN-like | 3.7e-4 (6.4e-4) | 8.1e-4 (1.4e-3) | 7.3e-4 | bit-identical | 0/12 |
| 12 | QSA-like | 7.8e-4 (7.1e-4) | 1.4e-3 (1.3e-3) | 1.5e-3 | bit-identical | 0/12 |
| 24 | GDN-like | 3.2e-4 (7.6e-4) | 6.7e-4 (1.6e-3) | 7.3e-4 | bit-identical | 0/24 |
| 24 | QSA-like | 8.9e-4 (9.1e-4) | 1.7e-3 (1.8e-3) | 1.5e-3 | bit-identical | 0/24 |

The kernel is about twice as close to the fp32 golden as today's path (it
accumulates the expert GEMMs and the shared + routed sum in fp32 instead of
fp16 CAS partials). The injection output has exactly today's error against
the golden at every width. No sticky error in any run.

## Timing (hipGraph of 16 distinct layers, per layer, median of 20 replays)

16 layers x ~334 MB of per-rank weights, so the 128 MB Infinity Cache cannot
hold the working set across replays; both paths are timed the same way. (The
P0 eager kernel sum was 181 µs at width 3; with a cold IC the same block
takes ~211 µs here.)

| M | today's block | mono kernel | speedup | mono phases P1/P2/P3/P4/P5/P6 (µs) |
| --- | --- | --- | --- | --- |
| 3 | 211.1 µs | 232.9 µs | **0.91x** | 8.9 / 46.6 / 49.2 / 21.3 / 53.7 / 23.4 |
| 6 | 283.0 µs | 292.3 µs | 0.97x | 8.8 / 50.9 / 51.5 / 23.6 / 86.4 / 42.6 |
| 12 | 540.2 µs | 461.5 µs | 1.17x | 9.6 / 69.1 / 93.7 / 40.4 / 170.4 / 81.8 |
| 24 | 684.1 µs | 749.0 µs | **0.91x** | 14.8 / 103.2 / 141.3 / 55.1 / 266.2 / 133.7 |

With barrier-time prefetch of the next phase's weights (`MONO_FFN_PREFETCH=1`):
0.90x / 0.96x / 1.17x / 0.91x (`m1_final_pf1.json`).

## Why it does not win

* **Each phase is latency-bound, not bandwidth-bound.** The barrier itself is
  cheap (1–2 µs: the fastest WG reaches it 12–15 µs before the slowest, and
  the release follows the last arrival within ~1 µs). The phase bodies are the
  problem: P2 streams 6.9 MB in ~45 µs and P3 6.5 MB in ~48 µs (150 GB/s
  against 512 GB/s); today's `gemv_act_k` / `hc_up_gate_mix_k` do the same
  work in ~30 / ~20 µs. A persistent grid has 8 waves per CU for the whole
  step; today's separate kernels run at 2–4x that occupancy and hide the
  DRAM latency that our 8 waves cannot. Six dependent phases each pay that
  ramp.
* **Attempts that did not change the picture:** 4-row x 8-activation register
  tiling with a transposed wave reduction (fixed the M=24 activation traffic,
  P2 180 -> 103 µs, P3 170 -> 141 µs, but not M=3); deeper unrolling (8
  weight rows in flight, group scales preloaded); WG-level expert tasks with
  K split across waves and an LDS reduction (P5 176 -> 54 µs at M=3); two WGs
  per CU (128 VGPRs, 144 WGs: co-resident and correct, but slower from
  spills); barrier-time prefetch of the next phase's weights (the next phase
  then runs mostly from L2/IC, e.g. P2 work 32 -> 17 µs, but the prefetch
  traffic slows the WGs that are still working, net neutral).
* **The HC GEMVs dominate.** P2 + P3 are ~95 µs of the 233 µs at M=3; both
  weight sets are replicated per rank (13.4 MB per mix), so the floor at
  achievable bandwidth is ~32 µs and today already spends ~50 µs.

## Width 24 and the cheaper glue lever

Today's block jumps from 283 µs at M=6 to 540 µs at M=12 and 684 µs at M=24:
for M > 8 the decode glue falls back to the prefill-style path (split-K +
Tensile GEMMs, unfused HC and shared expert, +390 launches per step). The
kernel's only win, 1.17x at M=12, is that gap, not the kernel. At M=24 the
kernel's HC phases (245 µs) are slower than today's M > 8 HC kernels, so there
is no overlap to argue about: extending the fused M <= 8 glue to M <= 24 goes
after the same cost (the M > 8 penalty) more cheaply than this kernel does,
and this kernel would not add to it.

## Recommendation

Stop 2a here (no M2/M3). The kernel is correct, capturable and fail-stop, and
it is kept on the branch as a reference, but it does not reach a clear margin
at widths 3 or 24 and the analysis above does not point to a fix short of a
different decomposition (fewer, fatter phases, or HC weights sharded instead
of replicated). The two levers from P0 target the same cells with less risk:
the M <= 24 fused decode glue, and the width-24 all-reduce path.
