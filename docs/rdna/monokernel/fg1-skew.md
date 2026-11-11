# MonoKernel 2a: per-rank skew under HSA_FORCE_FINE_GRAIN_PCIE=1 (2026-10-15)

Branch `monokernel/fg1-skew`. Raw results on the box: `~/w4a8_runs/fg1-skew/`
(`serve-f1-*`, `serve-ab-*`). Tools: `tools/monokernel-2a/fg1_{run,ab,table}.sh|py`.

## Symptom

With the FFN MonoKernel in the decode graph (TP4, 4x V620, FG=1), one to three
of the four ranks ran their attention-side kernels slower per layer (HC
preamble +6-7 us, GDN body +10-12, QSA body +10-30) and the other ranks waited
20-40 us per layer at the attention all-reduce, which cancelled the kernel's
gain (`docs/rdna/monokernel/2a-results.md`, "unexplained remainder"). Kernel
off, or FG=0, every rank was symmetric.

## What was measured

Two new one-thread mark kernels per layer in the serve timing
(`rdna_mono_ffn_timing`): kind 8 right after mark 3 books the graph's
kernel-to-kernel dispatch gap per rank, kind 9 a 64-step dependent load chain
through the coarse workspace (shader cycles, MALL-latency scale). Means per
layer, w1, MTP0 1k/512 c=1, FG=1, kernel on (max_rows 8), from
`fg1_table.py`:

| serve | hc (us) dev0/1/2/3 | AR | GDN core | gap8 | lat9 (cyc) | kernel | ITL c1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S1 Q=2, rdna_ar | 64.1/63.8/63.1/**58.3** | 26/28/32/44 | 196/197/199/205 | 2.84-2.87 | 29.4-29.6k | 116-118 | 24.5 |
| S3 Q=2, `VLLM_RDNA_AR=0` (RCCL/SHM, no P2P) | 58.2/**65.8**/59.0/57.6 | 60/45/68/73 | 224/219/231/233 | 3.0-3.3 | 29.1-29.3k | 117 | 26.7 |
| S4 `GPU_MAX_HW_QUEUES=4`, rdna_ar | 57.7/57.6/57.3/57.3 | 25/26/27/28 | 188/188/190/190 | 2.82-2.85 | 29.4-29.8k | 116-118 | 23.0 |

(ITL in the timing arm carries ~0.9 ms of probe cost: two marks plus the
12 us latency chain per layer.) The S1/S3 "slow" ranks have the same dispatch
gap, the same memory latency, the same SCLK and the same MonoKernel time as the
fast ones; only today's many-small-workgroup torch kernels are slower on them.

## Ruled out, with evidence

* Fence/atomic scope: on gfx1030 `hipcc` 7.14 emits the same
  `buffer_gl1_inv; buffer_gl0_inv` for system and agent scope and there is no
  L2 writeback/invalidate instruction at any scope
  (`~/w4a8_runs/fg1-skew/isa/fence.s`). Scope cannot be what FG=1 changes.
* Peer traffic / IPC / fine-grained staging: S3 keeps the skew with every
  all-reduce on RCCL over SHM and `NCCL_P2P_DISABLE=1`.
* Allocation flags: workspace and caching-allocator blocks read HSA flag 4
  (coarse) on all ranks under FG=1; rdna_ar staging is `hipDeviceMallocUncached`
  (extended-scope pool, visible with or without FG).
* Launch latency, memory latency, clocks: probes above.
* What `HSA_FORCE_FINE_GRAIN_PCIE` does (ROCR `amd_gpu_agent.cpp`
  `InitRegionList`): it only makes the fine-grained VRAM pool user-visible;
  CLR then serves `CL_MEM_SVM_ATOMICS` device allocations
  (`hipDeviceMallocFinegrained`, RCCL's P2P buffers) from it instead of silently
  from the coarse pool. Nothing in this stack's per-layer decode path allocates
  that way.

## Cause

`GPU_MAX_HW_QUEUES=2` (launcher default). CLR hands out at most that many HSA
queues per priority and then shares a queue between streams, picking the
"lowest load" queue at stream-creation time (`rocdevice.cpp`
`getQueueFromPool`), so which streams share a queue and a CP pipe differs per
process and per run. With the persistent kernel occupying every CU for the FFN
block, the side work that used to overlap the FFN kernels (copies / side-stream
work on the shared queue) overlaps the attention block instead, and its
many-small-workgroup kernels pay the pipe arbitration: ~10 % per kernel on the
rank(s) whose stream placement lost the draw. Four HW queues (one process per
GPU, 8 HQDs) remove the sharing and the skew (S4).

## Fix

TBD from the A/B below: recipe `GPU_MAX_HW_QUEUES=4`, or scoped
(`mono_ffn.set_worker_hw_queue_budget`, engine-core process, before the workers
spawn, only when an `rdna_mono_*` switch is on; `rdna_mono_hw_queues` overrides,
0 leaves the launcher value).

## A/B (production FG=1, no timers, `--temperature 0`)

TBD: Q=2 off / Q=4 off / Q=4 all-on x MTP0, MTP2 x 1k/512, 16k/1k x c=1, c=8,
runlist-oversubscription dmesg count per arm.
