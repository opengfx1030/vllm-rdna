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

## The fix: lock every pinned host mapping, as it appears

`vllm/utils/pinned_host.py`. `PinnedHostLocker.scan()` does:

1. read `/proc/self/maps` (and stop here when it is the same text as at the
   last scan);
2. select the mappings that are `rw-s` and named `/dev/zero (deleted)`, and
   drop the ones an earlier scan locked (key: start, end and inode, so a new
   block at a reused address is locked again and the 24 GiB table is not
   walked twice);
3. `mlock()` the rest, smallest first, so a short `RLIMIT_MEMLOCK` leaves the
   biggest block (the table, locked earlier) as the one that is unlocked.
   A block unmapped while it was being locked (`ENOMEM`) is not blamed on the
   limit.

It runs at the end of `Worker._compile_or_warm_up_model` (after model load
and graph capture) and logs once per worker
`Locked N of M pinned host mappings (X GiB); VmLck A -> B GiB.`, plus one
warning if some `mlock()` failed. A daemon thread then scans again every 2 s
for as long as the worker lives, because the blocks do not stop appearing at
the end of warm-up: in the passive run of 2026-11-05 a worker had 453 pinned
mappings when graph capture ended (16:24:26) and 638 ninety seconds later,
while the first requests ran, 0.52 GiB throughout. A single pass would have
left about 185 of them, the small blocks, unlocked. The rescans that locked
something are logged together, at most every 30 s
(`Pinned host mappings that appeared after warm-up: N now mlocked ...`); the
wording avoids `Locked`, which the GPU proof counts once per worker.

It never calls `mlockall`, never touches the mmapped checkpoint, files, the
heap, `/dev/shm` or device nodes, and a failed `mlock()` only warns.
`VLLM_ROCM_MLOCK_PINNED_HOST` is `auto` (default: on for ROCm gfx10x), `1`
(any ROCm GPU) or `0` (off). Locking needs `RLIMIT_MEMLOCK` above the table
plus about 1 GiB, so about 25 GiB per TP4 worker. Tests:
`tests/utils_/test_pinned_host.py` (parsing, selection, incremental scans,
address reuse, refused and vanished blocks, the rescan thread and its log
line, gating, the hook, and a real shared anonymous `mmap` created after the
first scan; 39 passed on the box, the real-`mmap` test is skipped on macOS).
The switch is not in `vllm/envs.py`, so a server started with it logs
`Unknown vLLM environment variable detected` once, like the other RDNA
switches of this fork; with `--fail-on-environ-validation` it would refuse to
start.

What it does not cover: memory registered with `hipHostRegister` (it stays
where it was: `rw-p` anonymous or `/dev/shm`), a block that appears and is
moved within the 2 s before the next scan, and anything in a process without
a KFD context.

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
and the evictions it coincided with in other servers' workers, and the eviction
bursts: one summary row per burst (who was evicted for how long, the host
counters of the burst, the moved and swapped pages per mapping class and lock
state, the weight loads in progress), then the details of each. The proof's
hook also passes its `windows.json` (`--windows`), which adds a column with the
window (idle, the neighbour running, settling, gap) each burst starts in.

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
- Quiet windows (the README's measurer rules). The bench cells are one window:
  the file `quiet/EvictFix`, touched again every minute, for 15 minutes at most
  and only while the script lives. Before it the driver waits for any `docker
  buildx build` to end (the pattern is anchored,
  `^(/usr/sbin/|/usr/bin/)?docker buildx build`: another agent's waiter, `bash -c
  'while pgrep -f "docker buildx build"; ...'`, matches an unanchored one for
  ever) and, after touching, for `boot.lock` to be free (`flock -n boot.lock
  true`: a boot that began first still runs, and a booter can take the lock just
  before the file appears), again before every bench cell, and 30 s settle after
  a wait.
- Each neighbour load is a booter first and a measurer once it may start. It
  waits for the 120 s gap after the previous window and for any build, then
  follows the booter rule (no foreign quiet file newer than 6 min, take
  `boot.lock`, look again under it), and only then touches its own file, holding
  the lock: 30 s idle, the load, 30 s settle. A load never holds a window while
  it waits for someone else's (two such drivers would wait on each other for
  ever), and the 15 minute cap counts measuring time only. The first launch of
  2026-11-05 opened the window first: it stayed up for 15.6 minutes, 10 of them
  waiting for another agent's window, and was flagged as outlasting the cap.
  Booting an arm follows the booter rule too. An eviction in a gap between
  windows is not counted for or against an arm; the verdict lists those per
  worker ("outside the windows") and flags a window that outlasted its cap.
- The neighbour (`NEIGHBOUR=replay`, CPU only) replays a TP2 weight load: two
  ranks, each a 47.68 GiB shared-anonymous block (written, mlocked: the 95.4 GiB
  PLE table over TP2), anonymous memory so that the demand is 16 GiB above
  `MemFree` (forcing reclaim), and a read of the checkpoint; it refuses a demand
  above `MemAvailable` minus 60 GiB. `NEIGHBOUR=cmd` runs a real TP2 boot of the
  same model on other leased GPUs instead.
- Verdict (`results/verdict.md`): PROVEN when the baseline evicted at least
  500 ms in a worker, no fix arm evicted more than 100 ms in any worker, every
  worker locked all its candidates, VmLck rose by what the fix logged (the
  first pass and the rescans; it is flat inside an arm), and the bench cells
  are within 3 % (the p99s of a 4- or 32-request cell within 9 %), or within
  the base spread.
  INCONCLUSIVE when the baseline did not evict: the neighbour did not provoke
  the problem. FAILED otherwise, and then `proof/report.md` of the fix arm
  names the mappings that moved.
- "VmLck unchanged" is read as: unchanged by anything but the fix's own
  logged GiB (first pass and rescans), and flat while the neighbour loads.
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
sudo bpftrace -o "$HOME/hsa_inval.txt" -e '
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

Which kcompactd path ran (the long compaction burst of the base arm had one
requested wake, `compact_daemon_wake` +1, and 36.8M pages scanned by the
daemon: proactive compaction, or one compaction pass asked for by a
high-order allocation?):

```bash
sudo bpftrace -o "$HOME/kcompactd_paths.txt" -e '
tracepoint:compaction:mm_compaction_kcompactd_wake {
    printf("%llu requested wake nid=%d order=%d\n", nsecs, args->nid, args->order);
}
tracepoint:compaction:mm_compaction_begin /comm == "kcompactd0"/ {
    @compaction_runs = count();
}
interval:s:120 { exit(); }'
```

Compaction runs with no `requested wake` line just before them are the
proactive path (`vm.compaction_proactiveness`); a wake line names the order
of the allocation that asked for it.

## Host settings

None applied. Current values on par1-cs25: `vm.swappiness=60`,
`vm.min_free_kbytes=91888`, `vm.watermark_scale_factor=10`,
`vm.watermark_boost_factor=15000`, `vm.compaction_proactiveness=20`,
`vm.compact_unevictable_allowed=0`, MGLRU on, THP `madvise`, zram 8 GiB (7 GiB
swap) at priority 100.

What the base arm of the proof measured (2026-11-05, one replay of a TP2 weight
load next to a TP4 server whose pinned blocks are not locked; replay 17:06:21
to 17:10:45, section 3 of `arms/base/proof/report.md`). The four workers were
evicted for 43.0 s in all (10.9, 11.1, 11.3 and 9.8 s), in seven bursts. The
host counters are the deltas of each burst (+-1 s):

| Bursts | Evicted, 4 workers | What the host did | `MemFree` |
| --- | --- | --- | --- |
| 3-6 (+755 to +805 s) | 3.0 s (7 %) | kswapd scanned 64.6M pages, direct reclaim 0.2M; 6,384 direct compaction stalls; kcompactd scanned 6.3M pages in 288 wakes; 2.7M pages migrated; 644 swapped out | 3.1 GiB, then 128-152 GiB |
| 7-8 (+811 to +821 s) | 1.7 s (4 %) | kswapd 5.6M; 265 stalls; kcompactd 2.0M pages in 12 wakes; 309k migrated; 6.7k swapped out | 149-152 GiB |
| 9 (+851 to +868 s) | 38.3 s (89 %) | no reclaim: kswapd 0, direct 0, no stall, no swap-out. kcompactd alone scanned 36.8M pages after one wake and migrated 4.98M pages (19 GiB) in 17 s | 142-147 GiB |

Over the whole replay: kswapd scanned 84.7M pages and direct reclaim 0.53M
(0.6 %), 7,006 direct compaction stalls, 8.04M pages migrated, 8,283 pages
swapped out, of which 172 pinned ones, while 380,529 pinned pages were moved
(0.05 % swapped). `MemFree` fell to 2.46 GiB. Whatever moves a page of an
unlocked pinned block evicts the worker, and the page is mostly moved by
compaction, not swapped.

Earlier measurements ([wedge investigation](rdna-ar-wedge-2026-11-04.md)):

- Storm of 13:14-13:22, a TP4 neighbour booting with the serving instance's
  four tables locked: `MemFree` fell from 136 to 11.6 GB; in one minute kswapd
  took 1.75M pages and direct reclaim 2.6M (60 % of the scan was done by the
  allocating processes), and 1.63M pages (6.2 GiB) were migrated by 13:22. The
  serving workers were evicted 59-526 ms in total.
- Burst of 10:52, an unlocked table: 1.68M pages of reclaim, 255k migrated, 329
  compaction stalls, swap-out; evictions of 7-29 s.
- `defrag=defer` with `compaction_proactiveness=0`, tried 10:42-10:59 with the
  table unlocked: the 10:52 burst still evicted for 7-29 s. That burst was
  reclaim and direct compaction, which this setting does not touch; it is not
  the 17 s burst above.

With 503 GiB of RAM, the gap between the kswapd wake-up (`low`) and its
sleep (`high`) is `max(min_free_kbytes/4, RAM x watermark_scale_factor/10000)`:
0.5 GiB today, so kswapd wakes below 0.59 GiB free and sleeps above 1.09 GiB.

Proposals, ranked by the share of the measured evicted time each setting
touches. The first is the only one the data single out; the rest are weak, and
with the fix in place none of them should matter for the workers (they are
what protects a worker whose pinned pages are not locked).

| Rank | Setting | Proposal | Evidence | Effect | Risk |
| --- | --- | --- | --- | --- | --- |
| 1 | `vm.compaction_proactiveness` | 20 to 0 | burst 9, 89 % of the evicted time: only kcompactd ran, with 142 GiB free, one wake, 36.8M pages scanned: the whole-zone proactive pass [INFERENCE: the `kcompactd_paths` trace above tells a proactive pass from a requested one] | no background defragmentation after the neighbour's churn | no background help for THP and high-order allocations; THP is `madvise` here; not tried on this box |
| 2 | `vm.swappiness` | 60 to 1 | swap-out in 4 of the 7 followed bursts (7.4k pages), 172 pinned pages swapped in the run against 380k moved; those 4 bursts hold 10 % of the evicted time, together with compaction | reclaim takes page cache (370+ GiB of it) before anonymous and shmem pages | an anonymous-heavy overload OOMs earlier instead of swapping; not tried on this box |
| 3 | `vm.watermark_scale_factor` | 10 to 100 | 6,384 direct compaction stalls (7,006 in the run) while `MemFree` fell to 3 GiB (bursts 3-6, 7 % of the evicted time), although kswapd did 99.4 % of the scan; the 11-04 storm had 60 % direct reclaim | `low` 0.59 to 5.1 GiB, `high` 1.1 to 10.1 GiB: kswapd starts with a 5 GiB runway | 5-10 GiB of RAM kept free |
| 3 | `vm.min_free_kbytes` | 91888 to 2097152 (2 GiB) | same | with the above: `low` 7 GiB, `high` 12 GiB; direct reclaim only below 2 GiB | 2 GiB reserved |
| 4 | `vm.watermark_boost_factor` | 15000 to 0 | no burst of the run is attributable to it | no boosted reclaim after fragmentation events | none known |
| 5 | loader | `posix_fadvise(DONTNEED)` after each shard, or direct I/O | the 168 GiB checkpoint read fills the page cache that kswapd then has to empty (all 84.7M pages of its scan) | the checkpoint stops filling the cache | code change |

Rank 3 cannot cure the replay's bursts: a demand 16 GiB above `MemFree`
exhausts any runway of 5-12 GiB. It only makes the allocating processes
reclaim and compact less often.

To try ranks 1-3 at runtime (not persisted; the values to go back to are the
ones above), the user runs:

```bash
sudo sysctl -w vm.compaction_proactiveness=0 vm.swappiness=1 vm.watermark_scale_factor=100 vm.min_free_kbytes=2097152
sudo sysctl -w vm.compaction_proactiveness=20 vm.swappiness=60 vm.watermark_scale_factor=10 vm.min_free_kbytes=91888  # back
```

A rerun of the proof's base arm (`ARMS=base`) under the new values, with a
fresh replay, shows whether burst 9 is gone.

## Status

- Done: the fix and its tests; the sampler, the report and their tests; the
  proof driver, its hook and their tests; the mechanism check above.
- Not done: the 30-minute sampling run on the box, the neighbour correlation
  table, the proof run and the host-setting numbers from measured reclaim
  totals. par1-cs25 was unreachable (`ssh` hung after the banner) from about
  12:50, and was then shut down for a power change.
