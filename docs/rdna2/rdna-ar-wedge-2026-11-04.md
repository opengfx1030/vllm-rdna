# rdna_ar wedge investigation (2026-11-04, par1-cs25)

## Symptom

On 2026-11-04 at 09:04, a production Flash-Next TP4 serve logged
`rdna_ar: WEDGED -- rank R timed out after ~131744/131872 ms ... collective #593530`
and the engine died. With the longer peer wait (below), the same event now
shows up as a late peer instead. At 10:52:34, ranks 3, 0 and 2 waited about
29.2 s for rank 1 at collective #446832, and the collective then completed.

## Root cause: KFD userptr evictions of the pinned PLE table

The peer did not lose a P2P write. Its GPU queues had been evicted by the
kernel driver.

- AER does not line up with the wedges. 40:03.1 shows BadTLP=86
  (correctable), 41:00.0 shows Timeout=4, uncorrectable errors are 0, and
  none of them fall at a wedge time.
- KFD exposes evictions in two places. Each process has
  `/sys/class/kfd/kfd/proc/<pid>/stats_<gpu_id>/evicted_ms`; the value is
  the same in every `stats_*` of a process. The raw KFD SMI event stream
  carries `QUEUE_EVICTION` records. On the serving workers, every
  `QUEUE_EVICTION` record has trigger 1, which is USERPTR.
- The userptr buffer is the pinned PLE n-gram table. Each worker allocates
  one in `Qwen4ExpPLEPinnedHostEmbedding` with `torch.empty(...,
  pin_memory=True)`. On a dGPU, ROCr backs paged hipHostMalloc memory with a
  KFD userptr BO over ordinary anonymous pages; this is the default
  (`HSA_USERPTR_FOR_PAGED_MEM=1`).
    - The table is 128 x [2500012, 160] bf16, 95.4 GiB in total, or 23.84 GiB
    per TP4 rank.
    - Torch's caching host allocator rounds each rank's share up to a 32 GiB
    block. That block shows up as a 32 GiB `rw-s /dev/zero (deleted)`
    mapping, and each worker's RssShmem is about 32.5 GiB.
- The host has 503.5 GiB of RAM.
    - Two TP4 servers pin 8 x 32.5 GiB, about 260 GiB, which is 52% of RAM.
    Shmem measured 260.2 GiB.
    - The 168 GiB checkpoint also sits in page cache.
    - During a boot, MemFree falls to 3-5 GB (from 194 GB to 4.8 GB in 3 min,
    10:34-10:37), and kswapd reclaim, compaction and zram swap-out follow.
- When reclaim or compaction moves a single page of a table, the MMU
  notifier invalidates the whole userptr BO. KFD then evicts every queue of
  that process and re-validates the whole 32 GiB BO before the queues run
  again.

Measured evictions (`evicted_ms`, raw events trigger USERPTR):

| Time | Cause | Evicted workers | Evicted for |
| --- | --- | --- | --- |
| 10:24-10:26 | neighbour TP4 boot | serving TP4 instance | 124-126 s |
| 10:38-10:40 | its own boot | booting instance | 18 s, then 38 s |
| 10:39:36-41 | neighbour TP4 boot | serving TP4 instance | 8.2, 13.4, 13.5, 14.0 s |
| 10:52:05 | reclaim burst: 1.68M pages by kswapd, 255k migrated, 329 compaction stalls, 1575 pages swapped out, in 15 s | serving instance A | 24.2, 29.0, 29.2, 24.1 s |
| 10:52:05 | same burst | serving instance B | 7.1-19.2 s |

A peer that stays evicted longer than the all-reduce wait wedges the
collective. That wait was 120 s at 4a3db777ae. The 09:04 wedge (a 131.8 s
wait) happened while a neighbour instance spent 445 s loading. [INFERENCE:
evictions were not being recorded before 10:00, so this one is matched by
time and mechanism only.]

## THP / compaction runtime experiment (not adopted)

- Baseline settings: THP `enabled=madvise`, `defrag=madvise`,
  `khugepaged/defrag=1`; `vm.compaction_proactiveness=20`,
  `vm.watermark_boost_factor=15000`, `vm.extfrag_threshold=500`.
- Arm tested: `defrag=defer` with `compaction_proactiveness=0`, applied at
  runtime only from 10:42:30 to 10:59:40.
- Result:
    - One quiet sample (10:48) evicted a single worker 33 times, about
    143 ms in total.
    - The 10:52:05 burst above happened with the arm still applied. Direct
    compaction still ran (329 stalls), and the evictions lasted 7-29 s.
- The arm did not measure better, so the original values were restored at
  10:59:40. It is not proposed as a default.

## Fix

1. `rdna_ar` survives a late peer.
   - A peer that has not reached the collective is waited for 600 s
     (`VLLM_RDNA_AR_WAIT_MS`), longer than the engine's 300 s execute
     timeout.
   - A peer that announced but whose write never landed aborts after 10 s.
   - Waits over 2 s are logged as late peers.
   - Workers also log jumps in their KFD `evicted_ms`.
2. Root cause, gfx10x only (`on_gfx10x()`):
   1. The pinned PLE table is allocated at its exact size
      (`vllm.utils.torch_utils.alloc_pinned_exact`, which calls
      hipHostMalloc). That is 23.84 GiB instead of 32 GiB per TP4 rank:
      8.16 GiB less per rank, 32.6 GiB less per TP4 server.
   2. Multiproc workers set `HSA_USERPTR_FOR_PAGED_MEM=0` before HIP
      initializes, and log the choice at startup.
      - ROCr then allocates paged host memory as GTT BOs. These are
        TTM-owned pages, not on the LRU, so reclaim and compaction do not
        move them and no MMU notifier fires.
      - A value the user already set is kept. `HSA_USERPTR_FOR_PAGED_MEM=1`
        opts out.
      - If HIP was already initialized, the worker logs a warning instead.
   3. GTT is capped by TTM `pages_limit`: half of RAM, shared by the whole
      host (251.8 GiB here).
      - TP rank 0 checks that table bytes x TP ranks fit in `pages_limit`
        minus the GTT already in use, before any rank allocates.
      - If they do not fit, every rank fails fast and the error states the
        numbers.
      - Two servers with 32 GiB tables (256 GiB) would not fit. With
        exact-size tables, 190.7 GiB fits.

## Open measurements

These need a GPU slot on HIP 2-5, comparing tree eb216eb35f with
88c1f15885:

- which device's `mem_info_gtt_used` grows, and whether the table is mapped
  from `/dev/dri/renderD*`. [INFERENCE: libhsakmt may place system-memory
  BOs on the first KFD GPU node, which is not one of the serving cards.]
- UVA gather speed, 16k-prefill TTFT and decode ITL.
- boot time.
- `evicted_ms` on the serving workers while a neighbour instance boots;
  0 is expected.
- the `rdna_ar` stress matrix.
