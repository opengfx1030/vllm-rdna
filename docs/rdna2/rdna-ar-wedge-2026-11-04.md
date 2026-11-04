# rdna_ar wedge investigation (2026-11-04, par1-cs25)

## Symptom

At 09:04, a production Flash-Next TP4 serve logged
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
  KFD userptr BO over ordinary pages; this is the default
  (`HSA_USERPTR_FOR_PAGED_MEM=1`).
    - The table is 128 x [2500012, 160] at 2 bytes, 95.4 GiB in total, or
    23.84 GiB per TP4 rank.
    - Torch's caching host allocator rounds each rank's share up to a 32 GiB
    block, which shows up as a `rw-s /dev/zero (deleted)` mapping.
- The host has 503.5 GiB of RAM.
    - Two TP4 servers pin 8 x 32.5 GiB, about 260 GiB, which is 52% of RAM.
    Shmem measured 260.2 GiB.
    - The 168 GiB checkpoint also sits in page cache.
    - During a boot, MemFree falls to 3-12 GB, and kswapd reclaim, direct
    reclaim, compaction and zram swap-out follow.
- When reclaim or compaction moves a page of the table, the MMU notifier
  invalidates the whole BO. KFD then evicts every queue of that process and
  re-validates the whole 32 GiB BO before the queues run again.

Measured evictions (`evicted_ms`, raw events trigger USERPTR):

| Time | Cause | Evicted workers | Evicted for |
| --- | --- | --- | --- |
| 10:24-10:26 | neighbour TP4 boot | serving TP4 instance | 124-126 s |
| 10:38-10:40 | its own boot | booting instance | 18 s, then 38 s |
| 10:39:36-41 | neighbour TP4 boot | serving TP4 instance | 8.2, 13.4, 13.5, 14.0 s |
| 10:52:05 | reclaim burst (1.68M pages, 255k migrated, 329 compaction stalls, swap-out) | serving instance A | 24.2, 29.0, 29.2, 24.1 s |
| 10:52:05 | same burst | serving instance B | 7.1-19.2 s |
| 13:16-13:22 | neighbour boot, below | booting instance (32 GiB userptr tables) | 62-97 s each |

A peer that stays evicted longer than the all-reduce wait wedges the
collective. That wait was 120 s at 4a3db777ae. The 09:04 wedge (a 131.8 s
wait) happened while a neighbour instance spent 445 s loading. [INFERENCE:
evictions were not being recorded before 10:00, so this one is matched by
time and mechanism only.]

## Options tried

- **THP / compaction tuning: not adopted.** Tried `defrag=defer` with
  `compaction_proactiveness=0` at runtime, 10:42:30-10:59:40.
    - The 10:52:05 burst still evicted workers for 7-29 s with it applied.
    - The original values were restored, and it is not proposed as a
    default.
- **Pinned host memory as GTT (`HSA_USERPTR_FOR_PAGED_MEM=0`): rejected.**
    - The GTT BO lands on the first KFD GPU node, 0000:c7:00.0. That is a
    W7800 belonging to another job, not the worker's own GPU.
    - The 0.25 GiB test grew `mem_info_gtt_used` there, whether HIP was
    pinned to bus 47 or ROCr was filtered with `ROCR_VISIBLE_DEVICES`. With
    the filter, rocminfo no longer lists the W7800s, yet the BO still lands
    on c7:00.
    - GPU gather speed was the same in both modes: 131072 rows of 320 bytes
    in 1.83 ms (22.9 GB/s).

## Fix

1. `rdna_ar` survives a late peer.
   - A peer that has not reached the collective is waited for 600 s
     (`VLLM_RDNA_AR_WAIT_MS`), longer than the engine's 300 s execute
     timeout.
   - A peer that announced but whose write never landed aborts after 10 s.
   - Waits over 2 s are logged as late peers.
   - Workers also log jumps in their KFD `evicted_ms`.
2. gfx10x pinned PLE table:
   - It is allocated at its exact size with hipHostMalloc: 23.84 GiB
     instead of 32 GiB per TP4 rank, 32.6 GiB less per server.
   - It is then `mlock()`ed. Locked pages are not reclaimed, and with
     `vm.compact_unevictable_allowed=0` compaction does not migrate them.
   - This needs `RLIMIT_MEMLOCK` above 23.84 GiB per worker. Below that,
     the worker logs the errno and keeps the unlocked table.
3. The drivers mark windows with evictions INVALID.
   - `tools/rdna/port_v031/kfd_evictions.py` reads each worker's
     `evicted_ms`.
   - `serve_validate.sh` reports it per bench cell and since READY.
   - `stall_probe.py --server-pid` tags every window, baseline and solo
     request that overlaps an eviction. Such a probe ends with verdict
     INVALID and exit code 3.

## Measurements (HIP 2-5, recipe flashnext-mtp2, 2026-11-04)

Settings common to all three arms:

- `serve_validate.sh` from tree 1eff2b5de0.
- Cells: 16k:1 at concurrency 1 (4 prompts) and 1k:256 at concurrency 8
  (32 prompts), temperature 0.
- Greedy correctness probe before and after load: PASS in every arm.
- No `evicted_ms` jump during any bench cell.

| Arm | Serving tree | Boot (READY) | 16k TTFT | c8 TTFT | c8 ITL median / mean |
| --- | --- | --- | --- | --- | --- |
| baseline | 88c1f15885 (32 GiB table, userptr) | 331 s | 5.57 s | 0.92 s | 44.8 / 61.3 ms |
| exact size only | 1eff2b5de0, `HSA_USERPTR_FOR_PAGED_MEM=1` | 311 s | 5.60 s | 0.96 s | 46.3 / 62.5 ms |
| exact size + mlock | 9bd03cfb9b, memlock unlimited, `compact_unevictable_allowed=0` | 330 s | 5.60 s | 0.99 s | 45.8 / 61.8 ms |

TTFT and ITL are within run-to-run noise across the arms. Boot time does
not change either; the boots are cold and their compile time varies.

Neighbour-boot test:

- Each arm with a load window served steady load (concurrency 4, 1k:256)
  for 13 min after its cells, while a TP4 instance booted on HIP 6-9.
- **exact size + mlock**: the neighbour booted at 13:14:20.
    - Host: MemFree fell from 136 to 11.6 GB, reclaim took 1.75M pages by
    kswapd and 2.6M pages directly in one minute, and 1.63M pages were
    migrated by 13:22.
    - The serving workers, all four tables locked, accumulated 526, 59, 442
    and 79 ms of USERPTR evictions in total over the whole storm. The
    booting neighbour's own workers lost 62-97 s each at the same time.
    - The serving instance's ITL stayed at a 45-46 ms mean (worst 5 s window
    49 ms) through the storm. For comparison, a userptr serving instance
    in the 10:39 storm had a 150.8 ms mean, 737.8 ms p99 and a 533 ms worst
    5 s window.
    - [INFERENCE: the remaining short evictions come from the workers' other,
    smaller pinned buffers, which are not locked.]
- **exact size only**: the neighbour boot (12:33) brought MemFree down to
  29 GB but caused no reclaim or compaction, so this arm saw no storm. It
  shows 0 evictions, which does not test the fix.

## Host settings (persisted on par1-cs25, 2026-11-04)

Both settings were measured as above and then persisted, with user
approval. See [serving](../rdna/serving.md#host-settings-for-the-pinned-ple-table).

- `/etc/security/limits.d/90-vllm-memlock.conf` sets `chenco_adm` soft and
  hard `memlock` to `unlimited`.
    - Unlimited rather than sized: a TP4 worker locks 23.84 GiB, a TP2
    worker 47.7 GiB. Only the PLE table path calls `mlock`.
    - It applies to new logins: `ulimit -l` shows `unlimited`. The old
    default was 8192 KiB.
- `/etc/sysctl.d/90-vllm-compact.conf` sets
  `vm.compact_unevictable_allowed = 0` (default 1).
- Restore:
  `sudo rm /etc/security/limits.d/90-vllm-memlock.conf /etc/sysctl.d/90-vllm-compact.conf && sudo sysctl -w vm.compact_unevictable_allowed=1`,
  then log in again.

Not run: the `rdna_ar` stress matrix (topology x size x load) on the
fixed tree.
