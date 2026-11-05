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

`/proc/<pid>/smaps` of a live TP4 worker (RidgeFill3's, around 12:45, a tree
from before the exact-size fix):

- 329 `rw-s /dev/zero (deleted)` mappings, 32.97 GiB in total, `Locked: 0`
  (the tree had no lock yet): one 32 GiB block (the PLE table, rounded up by
  torch's host allocator) and 328 small ones of about 1 GiB together. These
  are the `hipHostMalloc` blocks (torch pinned staging, model-runner inputs and
  so on).
- 312 private anonymous `rw-p` mappings, 11.93 GiB (Python and torch heap;
  none is known to be registered with KFD).
- the checkpoint, mmapped (a 95 GB safetensors blob): file-backed, not a KFD
  buffer.
- `VmPin` 0.

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

`tools/rdna/port_v031/pinned_evidence.sh` runs the evidence collection on the
host (no GPU, no server of ours):

```bash
HOURS=1 bash tools/rdna/port_v031/pinned_evidence.sh start    # OUT=~/w4a8_runs/evictfix/<ts>
bash tools/rdna/port_v031/pinned_evidence.sh status OUT
bash tools/rdna/port_v031/pinned_evidence.sh stop OUT         # SIGTERM, report.md
```

It starts, as root and with `setsid nohup`, `pinned_sampler.py` (the vLLM
venv's python, for numpy) and the raw KFD reader (`KFD_RAW`, default
`python3 ~/w4a8_runs/rdna_ar_wedge/kfd_smi_raw.py <log> <hours>`), writes
`host_start.txt` and `host_end.txt` (sysctls, zone watermarks, swap and zram,
THP, KSM, MGLRU, the proactive-reclaim module, `RLIMIT_MEMLOCK` and cgroup
limits of each worker, KFD system memory, `/proc/vmstat`), and `stop` renders
`report.md` with `pinned_report.py`:

```bash
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
reclaim bursts, the weight loads of the followed workers (a worker whose RSS
grows by 4 GiB or more at 0.05 GiB/s or more) with the host reclaim during each
and the evictions it coincided with in other servers' workers, and per
eviction burst the weight loads in progress and the mappings that moved with
their lock state.

## Evidence of 2026-11-05: 37 minutes, passive

`pinned_evidence.sh` ran as root (sampler at 1 Hz and the raw KFD reader) from
15:53:19 to 16:30:11, with no GPU and no server of ours. In that time
TopoSmoke booted, benched and shut down three TP4 `flashnext-mtp2` servers
(slot A twice, slot B once). Report: `~/w4a8_runs/evictfix/20261105-155311/`
on par1-cs25.

1. The host did not reclaim. Since the 14:03 reboot `/proc/vmstat` reads 0 for
   `pgscan_kswapd`, `pgscan_direct`, `pswpout`, `compact_stall`,
   `pgmigrate_success` and `allocstall_normal` (read at 16:23). `MemFree` never
   fell below 140.7 GiB (kswapd wakes below 0.59 GiB) and memory PSI stayed at
   0.00 s. The page cache is 282 GiB, not the 370-487 GiB of the report that
   started this: since the reboot the box has not been in that state, so a
   passive run could not reproduce it.
2. Twelve worker weight loads (one per worker per server): RSS +24.8-25.0 GiB
   in 11 s (24.1 GiB of it shmem), peak 20.8-21.3 GiB/s. Over each: 0 pages
   scanned, 0 swapped out, 0 migrated, and no jump of `evicted_ms` in any of
   the twelve workers.
3. 210 raw `QUEUE_EVICTION` events, all trigger USERPTR (one per process and
   GPU), none restored, in six bursts. All six are the shutdown of a server:

   | Burst | Time | Who | Workers left the samples | Pages of a followed mapping moved or swapped |
   | --- | --- | --- | --- | --- |
   | 1 | 16:07:33 | TopoSmoke server 162863, TP0-3 | 1.3 s later | none |
   | 2 | 16:07:38 | its other processes 162012, 162862, 162863 | not followed | n/a |
   | 3 | 16:18:23 | server 173975, TP0-3 | 0.5 s later | none |
   | 4 | 16:18:28 | its other processes 173276, 173974, 173975 | not followed | n/a |
   | 5 | 16:29:44 | server 186314 (slot B), TP0-3 | 3.8 s later | none |
   | 6 | 16:29:49 | its other processes 184614, 186313, 186314 | not followed | n/a |

   The queues of an exiting process are evicted when its memory is unmapped,
   a few seconds before its pid goes: that is not a reclaim eviction. The
   report calls a burst a teardown when nothing was restored and the worker
   left the samples within 30 s.
4. What a worker pins, at the last sample of each of the twelve workers:
   `VmLck` 23.84 GiB (the table, locked) and, in addition, 0.46 GiB of pinned
   mappings that nothing locks, in 595-632 mappings (0.35-0.52 GiB in 281-638
   mappings over the run). That is the population the fix locks. No page of it
   was swapped out, 3 were moved and 9,712 dropped (the unmapping at the
   shutdowns), because no reclaim ran. Anonymous mappings of 763 MiB moved
   1.5 M pages with no migration counter moving [INFERENCE: khugepaged
   collapsing them, THP is `madvise`].

What the 11-04 raw log says when it is paired the way the kernel counts
(`~/w4a8_runs/rdna_ar_wedge/thp/kfd_raw.log`, written by `kfd_smi_raw.py`,
10:41-13:23, GPUs 07:00.0 and 47:00.0; numbers below are for 47:00.0).
`amdgpu_amdkfd_restore_userptr_worker` reschedules itself while a userptr BO
cannot be revalidated, and a `QUEUE_RESTORE` with argument `R` is that
rescheduled attempt: the queues stay evicted, so an eviction ends at the first
restore without `R`. That gives 354 evictions, 279 restored and 75 never
restored; 112 of them had 900 rescheduled restores between them. The time
evicted is 1,852 s, against 173 s when the first restore of any kind closes
the eviction (the first version of the report did that). The pairing agrees
with the `evict.log` counters of that day: pid 1704020 +29,212 ms at
10:52:34.542 against 29.211 s here, pid 1704019 +102,204 ms at 11:01:23
against 102.2 s. The four consecutive pids 1704018-21 (stressed from 10:48)
were evicted 238-271 s each, longest 85-102 s (the all-reduce wait that wedges
a collective is 120 s), and the four pids 1983797-1983800 of the booting
neighbour of 13:16 103-118 s each, longest 21 s.

What this settles and what it does not. Measured: every worker holds hundreds
of pinned mappings that nothing locks (item 4); an invalidation evicts every
queue of the process; the pairing above matches the kernel's counter; and
compaction moves unlocked pinned pages (the VM check above). Not shown by the
passive run: a reclaim burst moving one of those mappings and evicting a
worker, because the host has not reclaimed since the reboot. The proof forces
that condition.

## GPU proof plan

Not run yet: it needs slot A (HIP 1,2,3,4 = the V620s on PCI buses ce, d3,
d8, db; the lease README of 2026-11-05 15:35) and the quiet windows below.
Driver: `tools/rdna/port_v031/mlock_pinned_proof.sh` (`EXPECT_SHA` of an
exact-SHA tree of this branch; `DRY=1` prints the arms and the neighbour and
touches nothing, no GPU, no lease).

- Slot and lease. `SLOT=A` (or `B`: HIP 6,7,8,9 = buses 0a, 0f, 14, 17); any
  other `GPUS` is refused. After `GO` and again before every arm the driver
  maps the slot's PCI buses to HIP indices (`pinned_proof.py buses`:
  `hipDeviceGetPCIBusId`, no GPU context) and refuses unless each bus is a V620
  (PCI id 1002:73a1) and the indices are exactly the slot's. The lease is
  `slotA.request` (created atomically, the queue), then `EvictFix.slotA` once no
  other `*.slotA` holder exists and every card has 28 GiB free; both are removed
  at the end.
- The W7800s. The driver resolves the buses of the W7800s (`cb`, `07`) to their
  HIP indices (0 and 5 on 2026-11-05) and refuses a slot that contains one, and
  hands those indices to `serve_validate.sh` as `REFUSED_GPUS`. The script's own
  default is `0,1` (the topology before 2026-11-05), which refuses HIP 1 of
  slot A: the first launch (16:32) failed both arms on that before any server
  booted.
- Temperatures. For each arm `pinned_proof.py temps` logs the four junction
  temperatures every second (`arms/<arm>/temps.log`). A sample at 95 C, or the
  mean of five at 90 C (TopoSmoke's rule of the same day), writes `ABORT_TEMP`
  and SIGTERMs the server's process group, and the arm ends with status 3.
- Arms, one boot each, on the same tree, differing only in
  `VLLM_ROCM_MLOCK_PINNED_HOST`: `base` (`0`: the table is locked, nothing
  else) and `fix` (`1`). The pair is `ARMS="base fix"`; the default order
  `base fix fix2 base2` adds the repeats that give the boot-to-boot floor.
- Each arm: `serve_validate.sh` boots `flashnext-mtp2` (TP4) and warms it, runs
  two eviction-checked bench cells (1024:512 at c=1 and c=8: TTFT, ITL), then
  the hook. During the hook: the sampler, the raw KFD reader, KFD `evicted_ms`
  every 0.5 s and a streaming decode load that logs stalls.
- Quiet windows (the README's measurer rules). The bench cells are one window.
  Each of the three neighbour loads is a window of its own: 30 s idle, the load,
  30 s settle. A window is the file `quiet/EvictFix`, touched again every
  minute, for 15 minutes at most and only while the script lives. Before it the
  driver waits for any `docker buildx build` to end (the pattern is anchored,
  `^(/usr/sbin/|/usr/bin/)?docker buildx build`: another agent's waiter, `bash -c
  'while pgrep -f "docker buildx build"; ...'`, matches an unanchored one for
  ever) and, after touching, for `boot.lock` to be free (`flock -n boot.lock
  true`: a boot that began first still runs, and a booter can take the lock just
  before the file appears), again before every bench cell, and 30 s settle after
  a wait. Two windows are at least 120 s apart. Booting an arm and every
  neighbour load follow the booter rule: wait for no fresh foreign file, take
  `boot.lock`, look again under it. An eviction in a
  gap between windows is not counted for or against an arm; the verdict lists
  those per worker ("outside the windows") and flags a window that outlasted
  its cap.
- The neighbour (`NEIGHBOUR=replay`, CPU only) replays a TP2 weight load: two
  ranks, each a 47.68 GiB shared-anonymous block (written, mlocked: the 95.4 GiB
  PLE table over TP2), anonymous memory so that the demand is 16 GiB above
  `MemFree` (forcing reclaim), and a read of the checkpoint; it refuses a demand
  above `MemAvailable` minus 60 GiB. `NEIGHBOUR=cmd` runs a real TP2 boot of the
  same model on other leased GPUs instead. Each load takes the `boot.lock`
  loader protocol inside its window.
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

`serve_validate.sh` gained opt-in variables for this (defaults keep its
behaviour): `QUIET_DIR`/`QUIET_OWN`/`QUIET_MIN`/`QUIET_POLL` (the booter side:
wait for the quiet windows, take `BOOT_LOCK`, look again under it),
`QUIET_HOLD=1` with `QUIET_CAP`/`QUIET_TOUCH`/`QUIET_GAP`/`QUIET_SETTLE` (the
bench cells as one window, then the gap before the hook) and `STOP_GRACE`.

Launch (the delta of this branch from the build of its base is Python only, so
the base's extensions are reused; nothing is built):

```bash
SHA=<full SHA of fix/mlock-pinned-host>; T=~/vllm-evictfix-${SHA:0:10}
BUILT=~/vllm-topo-f36f05b9b7          # exact-SHA tree of the base, extensions built
mkdir -p "$T" && git -C <clone> archive "$SHA" | tar -x -C "$T"
cp -p "$BUILT"/vllm/*.abi3.so "$BUILT"/vllm/vllm-rs "$BUILT"/vllm/_version.py "$T/vllm/"
mkdir -p "$T/vllm/third_party" && cp -rp "$BUILT"/vllm/third_party/triton_kernels "$T/vllm/third_party/"
echo "$SHA" > "$T/.sha"
# dry run: the tree, the host limits, the arms and the neighbour; no GPU, no lease
DRY=1 EXPECT_SHA=$SHA bash "$T/tools/rdna/port_v031/mlock_pinned_proof.sh"
# start (it touches no GPU before GO; from a fresh login shell), then GO once
# the slot is ours
RUN=~/w4a8_runs/evictfix-proof; mkdir -p "$RUN"
cd "$T" && EXPECT_SHA=$SHA SLOT=A ARMS="base fix" RUN=$RUN setsid nohup \
    bash tools/rdna/port_v031/mlock_pinned_proof.sh > "$RUN/proof.log" 2>&1 < /dev/null &
touch "$RUN/GO"    # or: touch "$RUN/STOP" to end it between arms
```

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
