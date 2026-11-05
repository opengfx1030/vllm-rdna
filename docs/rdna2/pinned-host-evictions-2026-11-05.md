# Pinned host memory and KFD queue evictions: the other pinned buffers (2026-11-05)

## Symptom

With the wedge fix in place (exact-size PLE table, `mlock()`ed;
`vm.compact_unevictable_allowed=0`; `RLIMIT_MEMLOCK` unlimited), a TP4
Flash-Next server on HIP 2-5 still had KFD `evicted_ms` rise by 7.1 s, 1.3 s
and 0.04 s on all four workers at once, while TP2 servers on other GPUs
loaded weights. The host has 503 GiB of RAM, 370-487 GiB of it page cache,
little free memory, and swap on zram.
[The earlier investigation](rdna-ar-wedge-2026-11-04.md) showed the same
mechanism for the PLE table. This one asks what else a worker keeps in
evictable pinned host memory, and what to do about it.

## Mechanism

On a dGPU, ROCr backs paged `hipHostMalloc` memory (torch's
`pin_memory=True`) with a KFD userptr BO over ordinary shmem pages. It is not
pinned in the page-pin sense: `VmPin` is 0 on a live worker. Any invalidation
of any page of the BO (reclaim unmapping it for swap-out, compaction
migrating it) reaches `amdgpu_hmm_invalidate_hsa`, and KFD evicts every queue
of the process. `evicted_ms` grows when the restore worker has re-validated
the BO and the queues run again, so a big BO that was invalidated shows up as
seconds. `mlock()` keeps reclaim away from a page, and with
`compact_unevictable_allowed=0` it keeps compaction away too.

## What a worker pins

`/proc/<pid>/smaps` of a live TP4 worker (RidgeFill3, 12:45, a tree from before
the exact-size fix), during its weight load:

- 329 `rw-s /dev/zero (deleted)` mappings, 32.97 GiB in total, `Locked: 0`
  (the tree had no lock yet): one 32 GiB block (the PLE table, rounded up by
  torch's host allocator) and 328 small ones of about 1 GiB together. These
  are the `hipHostMalloc` blocks (torch pinned staging, model-runner inputs and
  so on).
- 312 private anonymous `rw-p` mappings, 11.93 GiB (Python and torch heap;
  none is known to be registered with KFD).
- the checkpoint, mmapped (95 GB, 14 GB resident at that time): file-backed,
  not a KFD buffer.
- `VmPin` 0, `RssShmem` 33.8 GiB.

Reading `smaps` of a worker cost about 1 s of system time, too much for a 1 Hz
sampler; `pinned_sampler.py` reads `pagemap` instead.

After the wedge fix the table is 23.84 GiB and locked. The 328 small blocks,
about 1 GiB per worker, were not locked by anything. That is the hypothesis
this change acts on; the proof below decides it.

## The fix: lock every pinned host mapping after warm-up

`vllm/utils/pinned_host.py`, called at the end of
`Worker._compile_or_warm_up_model` (after model load and graph capture):

1. read `/proc/self/maps`;
2. select the mappings that are `rw-s` and named `/dev/zero (deleted)`;
3. `mlock()` them, smallest first, so a short `RLIMIT_MEMLOCK` leaves the
   biggest block (the table, locked earlier) as the one that is unlocked;
4. log once per worker
   `Locked N of M pinned host mappings (X GiB); VmLck A -> B GiB.`, and one
   warning if some `mlock()` failed.

It never calls `mlockall`, never touches the mmapped checkpoint, files, the
heap, `/dev/shm` or device nodes, and a failed `mlock()` only warns.
`VLLM_ROCM_MLOCK_PINNED_HOST` is `auto` (default: on for ROCm gfx10x), `1`
(any ROCm GPU) or `0` (off). Locking needs `RLIMIT_MEMLOCK` above the table
plus about 1 GiB, so about 25 GiB per TP4 worker. Tests:
`tests/utils_/test_pinned_host.py` (parsing, selection, the lock pass with a
fake `mlock` and with a real shared anonymous `mmap`, gating, the hook).

What it does not cover: memory registered with `hipHostRegister` (it stays
where it was: `rw-p` anonymous or `/dev/shm`), pinned blocks allocated after
warm-up (the pass runs once; the sampler lists new pinned mappings it sees),
and anything in a process without a KFD context.

## Checking the mechanism on a real kernel

Not the box, and no GPU: a Linux 6.19.7 aarch64 VM, as root, with a stand-in
worker holding 19 `rw-s /dev/zero (deleted)` mappings (one 128 MiB block and
eight 256 KiB blocks locked, eight 256 KiB blocks and two 32 MiB blocks not),
`echo 1 > /proc/sys/vm/compact_memory` three times, and `pinned_sampler.py`
diffing `pagemap` once a second:

| `vm.compact_unevictable_allowed` | pages moved, unlocked mappings | pages moved, locked mapping |
| --- | --- | --- |
| 1 (kernel default) | 1,006 | 264 |
| 0 (par1-cs25) | 1,065 | 0 |

So the sampler sees moved pages and their lock state correctly, compaction
moves unlocked pinned mappings under either setting, and locking only protects
against compaction with the sysctl at 0. It says nothing about KFD.

## Evidence tooling

`tools/rdna/port_v031/pinned_sampler.py` (root, vLLM venv python for numpy):

```bash
sudo -n ~/Apps/vllm/venv-7.14.0_0.32.0/bin/python \
    tools/rdna/port_v031/pinned_sampler.py --out DIR --hours 1 [--root-pid PID]
sudo -n python3 ~/w4a8_runs/rdna_ar_wedge/kfd_smi_raw.py RAW.log HOURS  # as its header says
python3 tools/rdna/port_v031/pinned_report.py DIR --kfd-events RAW.log --md report.md
```

Every second, for each `VLLM::Worker`: KFD `evicted_ms`, `VmLck`, `VmSwap`,
`RssShmem`, and a `pagemap` diff of its pinned mappings (the small ones every
second, the big ones every 5 s, big private anonymous and `/dev/shm` mappings
every 5 s) that names every mapping with moved, swapped-out or dropped pages;
every 30 s `kpageflags` of a few pages of each mapping says whether it is
mlocked. Host side: the `vmstat` reclaim counters, PSI, zram, kswapd and
kcompactd CPU. `pinned_report.py` joins this with the raw `QUEUE_EVICTION`
records (trigger, restore time) into a Markdown report: per-worker totals,
reclaim bursts, and per eviction burst the mappings that moved with their lock
state.

## GPU proof plan

Not run: it needs a GPU slot (HIP 2-5, 4 cards) and a quiet window. Driver:
`tools/rdna/port_v031/mlock_pinned_proof.sh` (EXPECT_SHA of an exact-SHA tree
of this branch; `DRY=1` prints the arms and the neighbour and touches nothing).

- Arms, one boot each, on the same tree, differing only in
  `VLLM_ROCM_MLOCK_PINNED_HOST`: `base` (`0`: the table is locked, nothing
  else) and `fix` (`1`). Order `base fix fix2 base2`; the repeats are the
  boot-to-boot floor.
- Each arm: `serve_validate.sh` boots `flashnext-mtp2` (TP4), runs two
  eviction-checked bench cells (1024:512 at c=1 and c=8: TTFT, ITL), then the
  hook: a 60 s idle window, then three neighbour loads 90 s apart, each followed
  by a 45 s settle. During all of it: the sampler, the raw KFD reader, KFD
  `evicted_ms` every 0.5 s and a streaming decode load that logs stalls.
- The neighbour (`NEIGHBOUR=replay`, CPU only) replays a TP2 weight load: two
  ranks, each a 47.68 GiB shared-anonymous block (written, mlocked: the 95.4 GiB
  PLE table over TP2), anonymous memory so that the demand is 16 GiB above
  `MemFree` (forcing reclaim), and a read of the checkpoint; it refuses a demand
  above `MemAvailable` minus 60 GiB. `NEIGHBOUR=cmd` runs a real TP2 boot of the
  same model on other leased GPUs instead. Each load takes the box's
  quiet-window and `boot.lock` protocol.
- Verdict (`results/verdict.md`): PROVEN when the baseline evicted at least
  500 ms in a worker, no fix arm evicted more than 100 ms in any worker, every
  worker locked all its candidates, VmLck rose by what the fix logged (and is
  flat inside an arm), and TTFT/ITL are within 3 % (or the base spread).
  INCONCLUSIVE when the baseline did not evict: the neighbour did not provoke
  the problem. FAILED otherwise, and then `proof/report.md` of the fix arm
  names the mappings that moved.
- "VmLck unchanged" is read as: unchanged by anything but the fix's own
  logged GiB, and flat while the neighbour loads.
- A server is only stopped with SIGTERM here (`STOP_GRACE` 300 s before
  `serve_validate.sh` falls back to SIGKILL). A TP4 server killed in the
  middle of its KFD teardown is a suspect for a wedged box.

`serve_validate.sh` gained two opt-in variables for this (defaults keep its
behaviour): `QUIET_DIR`/`QUIET_OWN`/`QUIET_HOLD`/`QUIET_MIN`/`QUIET_POLL` (wait
for the quiet windows before and after taking `BOOT_LOCK`; keep a quiet file
fresh from READY) and `STOP_GRACE`.

## Tracing the invalidation (for the user to run)

Not installed on the box, and never run by the investigation: it adds a probe
to the kernel. Run as root while a neighbour loads:

```bash
sudo bpftrace -o /tmp/hsa_inval.txt -e '
kprobe:amdgpu_hmm_invalidate_hsa {
    $r = (struct mmu_notifier_range *)arg1;
    @by[comm, $r->event] = count();
    @who[kstack(8)] = count();
    printf("%llu %s %d [%lx,%lx) event=%d\n", nsecs, comm, pid,
           $r->start, $r->end, $r->event);
}
interval:s:60 { exit(); }'
```

`comm` is the task doing the reclaim or the migration: `kswapd0` (background
reclaim), `kcompactd0` (compaction) or the allocating process itself (direct
reclaim). The stack separates `shrink_folio_list`/`try_to_unmap` (swap-out)
from `migrate_pages`/`try_to_migrate` (compaction) and `zap_*` (an unmap).
`[start,end)` against the mapping starts in `pagemap.jsonl` names the buffer.
It needs BTF (`/sys/kernel/btf/vmlinux`); without it, `#include
<linux/mmu_notifier.h>` at the top of the script.

## Host settings

None applied. Current values on par1-cs25: `vm.swappiness=60`,
`vm.min_free_kbytes=91888`, `vm.watermark_scale_factor=10`,
`vm.watermark_boost_factor=15000`, `vm.compaction_proactiveness=20`,
`vm.compact_unevictable_allowed=0`, MGLRU on, THP `madvise`, zram 8 GiB (7 GiB
swap) at priority 100.

What the earlier measurements say
([wedge investigation](rdna-ar-wedge-2026-11-04.md)):

- Storm of 13:14-13:22, a TP4 neighbour booting with the serving instance's
  four tables locked: `MemFree` fell from 136 to 11.6 GB; in one minute kswapd
  took 1.75M pages and direct reclaim 2.6M (60 % of the scan was done by the
  allocating processes), and 1.63M pages (6.2 GiB) were migrated by 13:22. The
  serving workers were evicted 59-526 ms in total.
- Burst of 10:52, an unlocked table: 1.68M pages of reclaim, 255k migrated, 329
  compaction stalls, swap-out; evictions of 7-29 s.
- `defrag=defer` with `compaction_proactiveness=0`, tried 10:42-10:59 with the
  table unlocked: the 10:52 burst still evicted for 7-29 s. It was not
  adopted.

With 503 GiB of RAM, the gap between the kswapd wake-up (`low`) and its
sleep (`high`) is `max(min_free_kbytes/4, RAM x watermark_scale_factor/10000)`:
0.5 GiB today, so kswapd wakes below 0.59 GiB free and sleeps above 1.09 GiB.
A direct-reclaim share of 60 % is consistent with a runway that short
[INFERENCE: no run has varied it].

| Rank | Setting | Proposal | What it does | Risk |
| --- | --- | --- | --- | --- |
| 1 | `vm.swappiness` | 60 to 1 | reclaim takes page cache (370+ GiB of it) before anonymous or shmem pages, so unlocked pinned pages are not swapped to zram | an anonymous-heavy overload OOMs earlier instead of swapping; not tried on this box |
| 2 | `vm.watermark_scale_factor` | 10 to 100 | `low` 0.59 to 5.1 GiB, `high` 1.1 to 10.1 GiB: kswapd starts with a 5 GiB runway and the allocating processes reclaim less | 5-10 GiB of RAM kept free |
| 2 | `vm.min_free_kbytes` | 91888 to 2097152 (2 GiB) | with the above: `low` 7 GiB, `high` 12 GiB; direct reclaim only below 2 GiB | 2 GiB reserved |
| 3 | `vm.watermark_boost_factor` | 15000 to 0 | no boosted reclaim after fragmentation events | none known |
| 4 | `vm.compaction_proactiveness` | 20 to 0 | no background compaction migrating unlocked pinned pages | tried at 10:42 (with `defrag=defer`): no effect on the burst, table unlocked then |
| 5 | loader | `posix_fadvise(DONTNEED)` after each shard, or direct I/O | the checkpoint stops filling the cache that reclaim then has to empty | code change |

The order is the order of directness to the observed mechanism and of the
evidence above; it is a proposal until the sampler's totals decide it:
swap-out in the eviction bursts (`pswpout`, `swapped_out` pages) ranks
`swappiness` first; direct reclaim dominating (`pgscan_direct` against
`pgscan_kswapd`, `allocstall`) ranks the watermarks first; migrations
(`pgmigrate_success`, `compact_stall`) rank the compaction settings first. With
the fix in place none of them should matter for the workers: they are what
protects a worker whose pinned pages are not locked.

To try ranks 1 and 2 at runtime (not persisted; the values to go back to are
the ones above), the user runs:

```bash
sudo sysctl -w vm.swappiness=1 vm.watermark_scale_factor=100 vm.min_free_kbytes=2097152
sudo sysctl -w vm.swappiness=60 vm.watermark_scale_factor=10 vm.min_free_kbytes=91888  # back
```

## Status

- Done: the fix and its tests; the sampler, the report and their tests; the
  proof driver, its hook and their tests; the mechanism check above.
- Not done: the 30-minute sampling run on the box, the neighbour correlation
  table, the proof run and the host-setting numbers from measured reclaim
  totals. par1-cs25 was unreachable (`ssh` hung after the banner) from about
  12:50, and was then shut down for a power change.
