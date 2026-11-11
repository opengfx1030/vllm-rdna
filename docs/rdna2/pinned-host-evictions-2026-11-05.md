# Pinned host memory and KFD queue evictions: the other pinned buffers (2026-11-05)

## Result

par1-cs25, slot A, `flashnext-mtp2` TP4, one replay of a TP2 weight load per
window. Two pairs of boots, each pair on one tree and differing in
`VLLM_ROCM_MLOCK_PINNED_HOST` only: `base`/`fix` (three replays each, tree
`d892e6e7db`) and `fix2`/`base2` (one replay each, tree `efbc0ec34a`, which
adds the switch to `vllm/envs.py`). Details under "GPU proof".

- The other pinned memory is the cause. A worker's KFD host buffers are 630-660
  `rw-s /dev/zero (deleted)` mappings: the PLE table and 0.52 GiB of small
  `hipHostMalloc` blocks. The driver's count of them (`drm-total-cpu` of the
  worker's own GPU) equals the sum of the mappings to the KiB in all four
  workers, and `VmLck` was the table alone: 0.52 GiB per worker stayed
  evictable. In the base arm 44,241 pages of those blocks moved, 43,042 of them
  in the burst that evicted the four workers 4.2-4.5 s each; no locked page
  moved. Second by second, 28 of the 29 s the base workers were under eviction
  had a pinned page move, and all 28 s with a pinned move had an eviction (1 of
  45 s with moves of other pages only).
- Compaction moves them, more than swap-out does. Of the 70.9 s the workers
  were evicted over the three base replays that evicted (summed), 78 % lies in
  two kcompactd passes (36.8M and 22.3M pages scanned, in one wake and in three,
  almost no reclaim around them), 13 % in bursts of kswapd scan with compaction
  stalls and 8 % in one mixed burst with swap-out; in the first replay 172 pinned
  pages were swapped against 380,529 moved. All 26 kcompactd passes seen
  followed kswapd passes and none came with kswapd idle:
  `vm.compaction_proactiveness` is not the trigger.
- The fix. `PinnedHostLocker` (`vllm/utils/pinned_host.py`) `mlock()`s every such
  mapping, smallest first, at the end of warm-up and again every 2 s (453
  mappings after warm-up, 630-632 ten minutes later). Branch
  `fix/mlock-pinned-host`; the tree under test is `d892e6e7db`.
- Proof, evictions. 0 ms in all four workers through three replays of `fix`, in
  the three idle windows and between them. The second replay was as heavy as
  the base replays that evicted (6.25M pages migrated against 5.4-8.0M, more
  direct reclaim than any of them, 1.98M pages swapped out); the third was a
  long kcompactd pass (17.5M pages scanned, 2.31M migrated). The base arm
  evicted the workers 4.9-5.8 s each in its one heavy replay (43.0 s and 5.9 s
  in all in the first boot's two). The repeat pair agrees: `fix2` 0 ms in its
  replay (a lighter one: 1.82M pages migrated, 11.4M scanned by kcompactd),
  `base2` 2.5-3.5 s per worker in a heavy one (5.95M migrated, 10,577 stalls,
  72,929 pinned pages moved). The steady decode client saw no gap above 251 ms
  in either fix arm; in the base arms 8 and 4 requests waited more than 1 s
  for a chunk, up to 3.0 s.
- Proof, side effects. `VmLck` rose by 0.522 GiB, the 0.52 GiB the fix logged,
  and stayed flat, in both fix arms. The first pair failed the driver's 3 %
  gate on output throughput (c1 -9.2 %, c8 -4.8 %), which over 4 or 32 requests
  moves with the sampled tokens (acceptance length 2.01 to 1.79 at c1). The
  gate is now the decode steps per second (output over acceptance length),
  committed before the repeat pair ran. Over the four arms, ABBA, the verdict
  is PROVEN: c8 steps/s 135.4 (`base`), 130.3 (`fix`), 134.0 (`fix2`), 132.3
  (`base2`); fix against base -1.26 % (the boots of one configuration differ by
  2.3 %), `fix2` against `base2` +1.27 % (c1: +0.95 %). Median TTFT and ITL
  moved by less than 1 %, p99s by up to 2.9 %. `fix2` against `base2` alone
  has one row outside 3 %: median TTFT at c1, 3.1 % lower with the fix. See
  "Latency and throughput" and "The repeat pair".
- Host settings. None is shown to remove the kcompactd passes (78 %). Ranked
  proposals with numbers are under "Host settings", the two bpftrace commands
  that decide the rest under "Tracing the invalidation".

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

After the wedge fix the table is 23.84 GiB and locked. The small blocks (328 of
about 1 GiB in that worker; 659 to 663 of 0.52 GiB in each worker of the proof
server below) were not locked by anything. That is the population this change
locks; the GPU proof below measures what it changes.

Which host memory is a KFD buffer can be read off the driver. The fdinfo of
each render-node descriptor of a worker has `drm-total-cpu`, the size of the
process's system-memory buffer objects in that GPU's VM, userptr ones
included. On the worker's own GPU it equals, to the KiB, the sum of the
worker's `rw-s /dev/zero (deleted)` mappings: 4 of 4 workers of the base-arm
server (2026-11-05 19:35, 659 to 663 mappings, 25,547,324 to 25,547,360 KiB)
while `VmLck` was the table alone, 25,000,120 KiB. So about 0.52 GiB per worker
is KFD memory that nothing locks, and NCCL's `/dev/shm/nccl-*` segments
(118 MiB per worker; RCCL imports `hipHostRegister`) and vLLM's `psm_*`
segments are not part of that total. `pinned_fdinfo_check.py PID...`, next to
the sampler in `tools/rdna/port_v031/`, prints the comparison; for a worker of
the fix arm its last column (total minus `VmLck`) must be about 0.

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
(any ROCm GPU) or `0` (off). It is declared in `vllm/envs.py` with the other
choice variables, so a value outside those three raises instead of picking a
mode, and `--fail-on-environ-validation` accepts it (the first GPU proof ran
before that: its serve.log has 15 `Unknown vLLM environment variable
detected` lines for it). Locking needs `RLIMIT_MEMLOCK` above the table
plus about 1 GiB, so about 25 GiB per TP4 worker. Tests:
`tests/utils_/test_pinned_host.py` (parsing, selection, incremental scans,
address reuse, refused and vanished blocks, the rescan thread and its log
line, gating, the hook, and a real shared anonymous `mmap` created after the
first scan; the real-`mmap` test is skipped on macOS) and the registration in
`tests/test_envs.py` (default, the three values, the rejected ones).

What it does not cover: memory registered with `hipHostRegister` that is not
an `rw-s /dev/zero (deleted)` mapping (none was found: the worker's own-GPU
`drm-total-cpu` is the zero blocks to the KiB, and RCCL's `/dev/shm/nccl-*`
segments are not in it), a block that appears and is moved within the 2 s
before the next scan, and anything in a process without a KFD context.

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
window (idle, the neighbour running, settling, gap) each burst starts in, and a
table of the host counters (kswapd and direct scan, compaction stalls, pages
migrated and swapped out, lowest `MemFree`) over each proof window. A second
table lists the kcompactd passes: the sampler ticks in which the daemon scanned
1M pages or more, with its wakes and kswapd's work over the minute before. The
last table of the report counts the seconds of the run in which pinned pages
moved, in which only other pages moved and in which nothing did, with how many
of each had a worker under eviction.
`pinned_fdinfo_check.py PID... --every 30` logs, one line per worker and tick,
its KFD buffer total against `VmLck`. `spec_acceptance.py` reads the speculative
acceptance length of a bench cell off the server log, which tells a change of
the engine's step rate from a change of the sampled tokens in the cell's output
throughput.

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

## GPU proof

Driver: `tools/rdna/port_v031/mlock_pinned_proof.sh` (`EXPECT_SHA` of an
exact-SHA tree of this branch; `DRY=1` prints the arms and the neighbour and
touches nothing, no GPU, no lease). It ran on 2026-11-05 on slot A (HIP 1,2,3,4
= the V620s on PCI buses ce, d3, d8, db; the lease README of 2026-11-05 15:35),
inside the quiet windows below. First how it works, then the run.

### How the proof works

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
  the base spread. The cells are gated on TTFT, ITL and the decode steps per
  second (output throughput over the cell's acceptance length, read off the
  server log by `spec_acceptance.py`'s functions). Output throughput and
  acceptance length are in the table, not gated: over 4 or 32 requests they
  move with the sampled tokens (about 0.2 of acceptance length inside one
  boot), not with the engine. The first launch gated output throughput, which
  is what failed (see "Latency and throughput").
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

### The run of 2026-11-05

Both arms ran `flashnext-mtp2` (TP4, `Qwen3.8-Flash-Next-AWQ-W4A16`) on slot A
with `CELLS="1024:512:1 1024:512:8"`, `LOADS=3` and `NEIGHBOUR=replay`, on the
same tree, `d892e6e7db`. `arm.txt` of an arm records the SHA, the value of
`VLLM_ROCM_MLOCK_PINNED_HOST` (0 or 1), the host's load average at the start
(base 1.63 1.35 3.71, fix 10.68 14.87 13.59, on 96 CPUs) and the sysctls, the
same in both. Time line:

- base, first boot, 16:40:05-19:15:02: not part of the pair. After its first
  replay (17:06) the hook waited about two hours for another agent's image
  build; the second replay came at 19:09, when its sampler (a two-hour bound)
  had expired, and the hook ended `rc=1`. The driver booted base again. Both
  replays stay below as extra data (`arms/base/attempt1`);
- base, second boot, 19:15:44-19:48:54 (`rc=0`): replays at 19:23, 19:37 and
  19:44;
- fix, 19:49:09-20:42:57 (`rc=0`): READY at 19:59:21 after waiting for another
  agent's quiet window, replays at 20:04, 20:12 and 20:36 (the last after 16.5
  minutes of waiting for the quiet windows of others).

**Evictions.** KFD `evicted_ms` per worker over each replay, and the host's
counters over it (`windows.json` of the arms; the host counters of the first
base boot's replays come from the three-hour run, which sampled that day):

| arm, replay | MemFree at start, GiB | evicted ms, sum (TP0 / TP1 / TP2 / TP3) | kswapd scan | direct scan | compaction stalls | migrated | swapped out | kcompactd scan | lowest MemFree, GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base 1, 17:06 | 111.7 | 43,028 (10,893 / 11,093 / 11,251 / 9,791) | 84.7M | 528k | 7,006 | 8.04M | 8.28k | 45.1M | 2.5 |
| base 1, 19:09 | 198.7 | 5,912 (1,435 / 1,293 / 1,391 / 1,793) | 15.8M | 462k | 3,796 | 5.42M | 782k | 20.8M | 2.8 |
| base 2, 19:23 | 199.5 | 21,986 (5,599 / 4,878 / 5,818 / 5,691) | 46.2M | 729k | 4,112 | 5.87M | 76.5k | 26.4M | 2.6 |
| base 2, 19:37 | 88.2 | 0 | 11.6M | 73.1k | 76 | 25k | 2.58k | 252k | 2.6 |
| base 2, 19:44 | 107.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 79.2 |
| fix, 20:04 | 202.1 | 0 | 9.84M | 16.7k | 55 | 202k | 13.6k | 3.07M | 2.9 |
| fix, 20:12 | 221.4 | 0 | 16.2M | 1.67M | 8,862 | 6.25M | 1.98M | 9.57M | 2.8 |
| fix, 20:36 | 211.2 | 0 | 17.5M | 390k | 171 | 2.31M | 5.29k | 17.5M | 3.1 |

The replays are not alike: the host was in a different state at each start
(other agents' processes come and go; before the last base replay some exited
and freed 55 GiB, so there was nothing to reclaim). What counts is how much the
replay made the host move. The three base replays that migrated 5.4-8.0M pages
evicted the workers 5.9, 22.0 and 43.0 s in all; the two that migrated 25k and
none evicted nothing. The fix arm's second replay (6.25M pages migrated, 1.98M
swapped out, 8,862 stalls) and its third (kcompactd scanned 17.5M pages, 2.31M
migrated) are in that range of host activity, and no worker was evicted for a
millisecond, in them or in the first replay, in the three idle windows or
between the windows (the base arm's idle windows were clean too).

**What moved.** Pages that moved, were swapped out or were dropped over the
sampled run of each arm (`pagemap` at 1 Hz; section 4 of `report.md`), by class
and lock state:

| arm | class | lock | moved | swapped out | dropped |
| --- | --- | --- | --- | --- | --- |
| base 2 | pinned (`rw-s /dev/zero (deleted)`) | unlocked | 44,241 | 0 | 0 |
| base 2 | anonymous | unlocked | 130,884 | 4 | 0 |
| base 2 | shmem | unlocked | 5,407 | 0 | 0 |
| fix | pinned | locked or not | 0 | 0 | 0 |
| fix | anonymous | unlocked | 83,845 | 639,267 | 0 |
| fix | anonymous | unknown | 0 | 6 | 0 |
| fix | shmem | unlocked | 62 | 0 | 60,453 |

In the 14 s burst of the 19:23 replay that evicted the base workers 4.2-4.5 s
each, 43,042 pages of unlocked pinned blocks moved and none of a locked one. The
fix workers had no unlocked pinned block at any census (`24.36/0.00/0.00` GiB
locked, unlocked, unknown in 630-632 mappings; the base workers
`23.84/0.52/0.00` in 659-663). Their unlocked anonymous and shmem pages, which
are not KFD buffers, were moved, swapped out and dropped in large numbers
without an eviction.

Second by second the same tool (section 4 of `report.md`, 1 s bins of the
`pagemap` records against the counter jumps, each taken back to where its
eviction began) lines the moves up with the evictions:

| arm | seconds in which | seconds | with a worker under eviction |
| --- | --- | --- | --- |
| base 2 | pinned pages moved or swapped out | 28 | 28 (100 %) |
| base 2 | only other pages moved or swapped out | 45 | 1 (2 %) |
| base 2 | nothing moved | 1,431 | 0 |
| fix | pinned pages moved or swapped out | 0 | 0 |
| fix | only other pages moved or swapped out | 136 | 0 |
| fix | nothing moved | 2,182 | 0 |

Of the 29 s the base workers spent under eviction, 28 had a pinned page move; of
the 45 s in which only anonymous or shmem pages moved, 1 had an eviction.

**Lock coverage.** At 19:59:04 each fix worker logged `Locked 453 of 453 pinned
host mappings (24.36 GiB); VmLck 23.84 -> 24.36 GiB.` Seven or eight rescan
lines per worker (19:59:34 to 20:11:05; 20:35:39 for TP2 and TP3) then locked
177-179 more mappings of 0.9-1.0 MiB in all, 630-632 mappings at the end, and
`VmLck` read 24.363-24.364 GiB in every window of the hook.
`pinned_fdinfo_check.py --every 30` followed the four workers from 20:03:57 to
20:42:31 (312 worker lines): the KFD buffer total of the worker's own GPU minus
`VmLck` was 0 KiB on every line (619-632 mappings); in the base arm it was
547,204-547,240 KiB (0.52 GiB) on all four. The raw KFD log of the fix arm (180
`QUEUE_EVICTION`, all USERPTR) holds no record of a worker: they belong to other
agents' processes and to this server's API server and EngineCore (20 on 10
GPUs each, 20:16:14 and 20:16:27, in the second replay), which hold KFD contexts
and run no kernel, and the lock is not applied to them.

**What the steady decode client saw.** Four decode streams ran through each arm
and logged the longest gap between two chunks of a request. During the replays:
base 1, 5 of 211 requests waited more than 1 s for a chunk (longest 1.65 s);
base 2, 8 of 245 in the 19:23 replay (longest 3.0 s; four `rdna_ar: late peer`
warnings in its `serve.log`, up to 2.86 s) and 0 of 345 in the other two; fix, 0
of 769 in the three (longest 246 ms; outside the replays the p99 is 246 ms too).

**Latency and throughput.** The driver's table, fix against base 2 (4 and 32
requests per cell, gates 3 % for medians and throughput, 9 % for p99s):

| cell | metric | base | fix | diff % | gate % | ok |
| --- | --- | --- | --- | --- | --- | --- |
| c1 | median TTFT, ms | 369.32 | 374.63 | +1.44 | 3 | yes |
| c1 | p99 TTFT, ms | 376.20 | 378.73 | +0.67 | 9 | yes |
| c1 | median ITL, ms | 23.53 | 23.41 | -0.52 | 3 | yes |
| c1 | p99 ITL, ms | 24.03 | 23.98 | -0.22 | 9 | yes |
| c1 | output throughput, tok/s | 82.09 | 74.51 | -9.24 | 3 | NO |
| c8 | median TTFT, ms | 527.57 | 528.62 | +0.20 | 3 | yes |
| c8 | p99 TTFT, ms | 2890.79 | 3102.90 | +7.34 | 9 | yes |
| c8 | median ITL, ms | 45.15 | 45.42 | +0.59 | 3 | yes |
| c8 | p99 ITL, ms | 369.14 | 371.13 | +0.54 | 9 | yes |
| c8 | output throughput, tok/s | 274.74 | 261.59 | -4.78 | 3 | NO |

`VmLck` at the end: 23.842 GiB in the base arm, 24.364 in the fix arm, +0.522,
which is the 0.52 GiB the fix logged; flat inside each arm.

The two failed rows are output throughput, and over 4 or 32 requests it moves
with the speculative acceptance length (which tokens were sampled) as much as
with the engine. `spec_acceptance.py` reads it off the server log over a cell
(`python3 tools/rdna/port_v031/spec_acceptance.py serve.log 19:59:40 20:00:24
--bench bench-1024x512-c1.json`; the cells' times are in `summary.txt`):

| run | cell | acceptance length | output tok/s | decode steps/s | mean ITL, ms | mean TPOT, ms |
| --- | --- | --- | --- | --- | --- | --- |
| base 1 | c1 | 2.051 | 83.4 | 40.63 | 23.46 | 11.30 |
| base 2 | c1 | 2.011 | 82.1 | 40.81 | 23.51 | 11.48 |
| fix | c1 | 1.786 | 74.5 | 41.71 | 23.41 | 12.72 |
| base 1 | c8 | 2.083 | 280.8 | 134.77 | 52.66 | 25.35 |
| base 2 | c8 | 2.029 | 274.7 | 135.38 | 52.28 | 25.84 |
| fix | c8 | 2.007 | 261.6 | 130.32 | 52.48 | 26.41 |

At c1 the fix arm's 9 % lower throughput is its lower acceptance (1.79 tokens
per step against 2.01 and 2.05); the steps per second are the same. At c8 the
acceptance differs by 1 %, but the steps per second are 3.7 % lower than base 2's
and 3.3 % lower than base 1's, while the two base boots differ by 0.5 %. The
steady client, the same in all arms, gives a median ITL per
request of 35.10, 36.35 and 37.20 ms (base 1, base 2, fix) and a median of 107
chunks per request in all three: boots of the same configuration differ by 3.6 %
there, more than base 2 and fix do (2.3 %). Temperature does not explain a gap
(junction below 66 C in every cell), and the locker's thread cannot: one full
scan (read, parse, select) of a live worker's maps took 11.7 ms (2,810
mappings, another agent's worker), 6.4 ms of it in the parser, every 2 s at most.

**Verdict of the first pair.** The driver wrote `FAILED` (`results/verdict.md`)
because of those two throughput rows and nothing else: `evicted_ms` (base 4.9-5.8
s per worker in the one replay that migrated 5.9M pages; fix 0 in every window),
the lock coverage (4 of 4 workers, 453 of 453), `VmLck` and every TTFT and ITL
row passed. That verdict stands as written. Its open question, whether the fix
costs more than 3 % at c8, is answered by the repeat pair below, with the gate
moved from the output throughput to the decode steps per second.

What the proof does not show: the replay is a CPU-only stand-in for a TP2 weight
load (a real boot of the same model on other GPUs, `NEIGHBOUR=cmd`, was not
run), so the host's reaction to a real load, in particular the high-order
allocations of the GPU driver, was not reproduced; and the three base replays
that evicted came from two boots, not one.

### The repeat pair (fix2, base2)

Run after the first pair to measure the boot-to-boot floor of the c8 step rate:
`ARMS="fix2 base2" LOADS=1 LEASE_MINUTES=75`, one replay per arm to fit the
lease (the default is three). Tree `efbc0ec34a`: `d892e6e7db` plus the switch
declared in `vllm/envs.py` (12 lines; `pinned_host.py` reads it through `envs`,
18 lines), tests and tools; no native source differs. The step-rate gate
(`bc2d2e2e99`) was committed at 21:33, before the first bench cell of the pair
finished (21:41).

Time line: `fix2` 21:29:39-21:49:48 (boot behind another agent's boot, READY
21:34:52, cells 21:40:42-21:42:42, replay window 21:44:51-21:49:16); `base2`
21:50:03-22:06:42 (READY 21:54:46, cells 21:55:05-21:57:02, replay window
21:59:11-22:06:08). Both `rc=0`, no fault in dmesg, junction temperature at most
81 C (`fix2`) and 83 C (`base2`) against below 66 C in the first pair (the guard
stops a server at one sample of 95 C or a mean of five at 90 C). `fix2` logged
`Locked 453 of 453 pinned host mappings (24.36 GiB)` on all four workers and
held `VmLck` 24.364. Its serve.log names 14
unregistered variables of the recipe (`VLLM_RDNA_*` and others) and not the
switch; the first fix arm named 15, the switch among them.

Evictions and the host's counters over the replay window (`windows.json`):

| arm, window | evicted ms (TP0 / TP1 / TP2 / TP3) | kswapd scan | direct scan | compaction stalls | migrated | swapped out | kcompactd scan | lowest MemFree, GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fix2, 21:45 | 0 | 23.2M | 32.7k | 335 | 1.82M | 3.5k | 11.4M | 2.2 |
| base2, 21:59 | 2,454 / 3,072 / 2,669 / 3,489 | 28.4M | 517k | 10,577 | 5.95M | 2.07M | 17.8M | 3.1 |

The `fix2` replay was the lighter one (a third of the pages migrated, a thirtieth
of the compaction stalls), so by itself it does not discriminate; the first
pair's second `fix` replay (6.25M migrated, 8,862 stalls) and third (2.31M
migrated, 17.5M pages scanned by kcompactd) do. What moved: in `base2` 72,929
pinned pages (491 records, all unlocked) and
24 swapped, 8,106 anonymous moved and 687,540 swapped; in `fix2` no pinned page
moved or was swapped, 64,310 anonymous and 3,552 shmem pages moved. Second by
second: `base2` 33 s with a pinned move, 32 of them under eviction (97 %), 63 s
with only other pages (5 under eviction), 325 s with nothing moved (0); `fix2`
0 s with a pinned move, 24 s with other pages (0), 245 s with nothing (0). The
steady client: 4 of 326 `base2` requests waited more than 1 s for a chunk
(longest 1.5 s), 0 of 209 in `fix2` (longest 244 ms); neither server logged
`rdna_ar: late peer`.

The cells of all four arms (4 and 32 requests; steps/s is the cell's output
tok/s over its acceptance length; `base` and `fix` are the first pair):

| arm | cell | median ITL, ms | output tok/s | acceptance length | decode steps/s |
| --- | --- | --- | --- | --- | --- |
| base | c1 | 23.53 | 82.09 | 2.011 | 40.81 |
| fix | c1 | 23.41 | 74.51 | 1.786 | 41.71 |
| fix2 | c1 | 24.10 | 74.63 | 1.898 | 39.32 |
| base2 | c1 | 24.06 | 86.25 | 2.214 | 38.95 |
| base | c8 | 45.15 | 274.74 | 2.029 | 135.38 |
| fix | c8 | 45.42 | 261.59 | 2.007 | 130.32 |
| fix2 | c8 | 46.52 | 270.69 | 2.020 | 134.01 |
| base2 | c8 | 46.67 | 270.45 | 2.044 | 132.33 |

At c8 the steps/s against `base` are -3.74 % (`fix`), -1.01 % (`fix2`) and
-2.25 % (`base2`); `fix2` against `base2` is +1.27 % (median ITL -0.3 %), and at
c1 +0.95 % (median ITL +0.2 %). Two boots of the same configuration differ by
2.3 % (base) and 2.8 % (fix) at c8, by 4.7 % and 5.9 % at c1: the floor is of
the order of the 3 % gate, and the first pair's -3.74 % lies inside it. The
acceptance length at c1 spans 1.79 to 2.21 over the four boots in no order of
the lock (fix 1.79 and 1.90, base 2.01 and 2.21), which is why output
throughput cannot gate a cell of four requests. The second pair ran hotter than
the first (c8 median ITL 46.5-46.7 ms against 45.2-45.4).

Verdicts. The driver's own, at the end of the run, used the old gate and said
`FAILED` for the pair (`results/verdict.md` of the run: c1 median TTFT -3.1 %
and c1 output throughput -13.5 %). With the gate of `bc2d2e2e99`, `fix2` and
`base2` alone fail one row: c1 median TTFT, 3.10 % lower with the fix against a
3 % gate (one base arm, so no spread; the gate is two-sided). Over the four
arms in the order `base fix fix2 base2` the verdict is PROVEN: c1 median TTFT
-0.86 % (base spread 2.7 %), median ITL -0.18 % and +0.13 % (c8, spread 3.3 %),
steps/s +1.59 % (c1, spread 4.7 %) and -1.26 % (c8, spread 2.3 %), p99 TTFT up
to +2.85 % (gate 9 %). To recompute it from the arms of both runs:

```bash
G=~/w4a8_runs/evictfix-proof4; mkdir -p "$G/arms"
for a in base fix; do ln -s ~/w4a8_runs/evictfix-proof/arms/$a "$G/arms/$a"; done
for a in fix2 base2; do ln -s ~/w4a8_runs/evictfix-proof2/arms/$a "$G/arms/$a"; done
"$VENV/bin/python" tools/rdna/port_v031/pinned_proof.py verdict --run "$G" \
    --arms "base fix fix2 base2"
```

Provenance of the extensions. The tree has no build of its own: `git archive`
of the SHA plus the base's extensions (the Launch block). The nine files
(`_C`, `_C_stable_libtorch`, `_moe_C_stable_libtorch`, `_rocm_C`,
`cumem_allocator`, `fs_io_C`, `spinloop`, `vllm-rs`, `_version.py`) have the
same `sha256sum` in `~/vllm-topo-f36f05b9b7/vllm` (its `.sha` is f36f05b9b7), in
the first proof's tree (`d892e6e7db`) and in the trees of `efbc0ec34a` and
`bc2d2e2e99`; the first 16 hex digits are `f15e1b311b52db33`,
`af1eb83cdf18ce86`, `a75809fd41653e07`, `e74ecc5445c5e084`, `086124e5859d6725`,
`5912e85d59966885`, `9efcda53abcf5cb9`, `dcdcd2ec1e875701` and
`a2bc57d342164e80`. No native source differs between `f36f05b9b7` and those
trees (`git diff --stat f36f05b9b7 <sha> -- csrc cmake CMakeLists.txt setup.py
rust` is empty; `vllm/` differs in four Python files). The files come from
`~/vllm-audit-build-moe`, whose `vllm/*.so` are hard links of its cmake outputs
(link count 2) and byte-identical to the base tree's; there 962 of 962 native
sources of `f36f05b9b7` (`csrc`, `cmake`, `CMakeLists.txt`, `setup.py`, `rust`)
are byte-identical by git blob hash, plus two files no CMake list compiles
(`csrc/rocm/ridgefill_map_rdna2.cu` and `.cuh`). Freshness, per extension, from
that tree's compile database (108 translation units, 100 of them with
`--offload-arch=gfx1030`), ninja's deps log and the hipify source map, which give
each target's own sources and headers: none is newer than the extension's link
time (`_C` Nov 4 20:50, 2 TUs, newest source Nov 3; `_C_stable_libtorch` Nov 5
09:01, 32 TUs, newest `csrc/sampler.cu` 09:00; `_moe_C_stable_libtorch` Nov 4
23:05, 4 TUs, newest 23:04; `_rocm_C` Nov 5 07:14, 67 TUs with the skinny-GEMM
object library, newest 02:42, the all-reduce and `mono_k2_rdna2` files;
`cumem_allocator`, `fs_io_C` and `spinloop` Nov 4 20:45, newest Nov 3), and none
of the 108 objects is older than one of its own sources or headers
(`~/w4a8_runs/evictfix/so_freshness.txt`, made by `fresh_tu.py` beside it). That
is file times, not a rebuild: nothing was recompiled; the hipified `.hip` copies
there were regenerated at 15:55 (ninja's dry run wants to redo every target for
that reason alone, which says nothing about content); and the build flags were
not compared with those of the build, since `build.ninja` was regenerated then
too.

## Tracing the invalidation (for the user to run)

Not run by the investigation: it adds a probe to the kernel, and `bpftrace` is
not installed on the box. Checked read-only on the box's kernel (`sudo -n`):
`amdgpu_hmm_invalidate_hsa` is a local symbol of the `amdgpu` module in
`/proc/kallsyms`; tracefs has `vmscan/mm_vmscan_wakeup_kswapd` (fields `nid zid
order gfp_flags`), `compaction/mm_compaction_kcompactd_wake` (`nid order
highest_zoneidx`) and `compaction/mm_compaction_begin`; and
`/sys/kernel/btf/vmlinux` does not exist, so neither command below reads a
kernel struct. Run as root while a neighbour loads:

```bash
sudo bpftrace -o "$HOME/hsa_inval.txt" -e '
kprobe:amdgpu:amdgpu_hmm_invalidate_hsa {
    @by[comm] = count();
    @who[comm, kstack(10)] = count();
}
interval:s:60 { exit(); }'
```

`comm` is the task doing the reclaim or the migration: `kswapd0` (background
reclaim), `kcompactd0` (compaction) or the allocating process itself (direct
reclaim or direct compaction). The stack separates
`shrink_folio_list`/`try_to_unmap` (swap-out) from
`migrate_pages`/`try_to_migrate` (compaction) and `zap_*` (an unmap). Before
bpftrace 0.17 the probe is spelled `kprobe:amdgpu_hmm_invalidate_hsa`. The
invalidated range (`$r = (struct mmu_notifier_range *)arg1;`, then `$r->start`,
`$r->end`, `$r->event`, to name the buffer against the mapping starts in
`pagemap.jsonl`) needs BTF, missing here, or `#include <linux/mmu_notifier.h>`
from the headers of this exact kernel; the layout of that struct changes
between kernel versions, so the command leaves it out.

What starts the long compaction passes. Every kcompactd pass of the base runs
(a sampler tick in which the daemon scanned 1M pages or more: 15 in the three
hours 16:57-19:57, which hold the base arm's replays) and of the fix arm (11)
followed kswapd passes: 21-297 wakes of the daemon and 40-498 kswapd passes in
the minute before, the last kswapd pass 0-30 s earlier. None came with kswapd
idle, and the hours without a replay had none, so
`vm.compaction_proactiveness` is not what starts them. Before the 14 s pass of
the base replay of 19:23 kswapd ran one pass a second (`pageoutrun`; 287k pages
scanned each, the Normal zone's high watermark) with 2-3 kcompactd wakes a
second while 81 GiB were free, and `kswapd_low_wmark_hit_quickly` rose by 322 in
that replay: kswapd went to sleep and was woken again within 100 ms, over and
over. [INFERENCE from the kernel source: going to sleep clears the compaction
skip hints and wakes kcompactd with the order of the allocation that woke
kswapd (`kswapd_try_to_sleep`; order 0 wakes nothing), so the first compaction
after the last wake starts from scratch and scans the whole zone, 22-37M pages;
the wakes are then high-order allocations.] Which allocations wake kswapd, and
from which process:

```bash
sudo bpftrace -o "$HOME/kswapd_wakers.txt" -e '
tracepoint:vmscan:mm_vmscan_wakeup_kswapd { @wakers[comm, args->order] = count(); }
tracepoint:compaction:mm_compaction_kcompactd_wake { @kcompactd_wake[args->order] = count(); }
tracepoint:compaction:mm_compaction_begin /comm == "kcompactd0"/ { @compaction_runs = count(); }
interval:s:120 { exit(); }'
```

`@wakers` counts, by process and order, the allocations that woke kswapd: order
0 is a real shortage, orders 1 and above are the requests that can start the
chain (a transparent huge page fault is order 9; the GPU driver's TTM page pool
and the network stack allocate large orders too [from memory of those drivers,
not checked in this kernel's source]). `@kcompactd_wake` gives the orders
kcompactd was asked for, and `@compaction_runs` how many compaction runs it made
in the window.

## Host settings

**Applied 2026-11-06:** ranks 1 and 2 (`vm.watermark_scale_factor=100`,
`vm.min_free_kbytes=2097152`, `vm.swappiness=1`) are set on par1-cs25 in
`/etc/sysctl.d/91-vllm-reclaim.conf`, managed by the host's Ansible repo; rank 3
(`extfrag_threshold`) is not. No neighbour-load replay has run since.

Values before that change (read
2026-11-05 20:58): `vm.swappiness=60`, `vm.min_free_kbytes=91888`,
`vm.watermark_scale_factor=10`, `vm.watermark_boost_factor=15000`,
`vm.compaction_proactiveness=20`, `vm.extfrag_threshold=500`,
`vm.compact_unevictable_allowed=0`, MGLRU on, THP `madvise` (`defrag`
`madvise`, shmem THP `never`), zram 8 GiB (7 GiB swap) at priority 100, one NUMA
node, kernel 7.1.7.

Where the base arm's evicted time went. The three replays of the base runs that
evicted (17:06, 19:09 and 19:23 on 2026-11-05) evicted the four workers for 70.9
s in all (summed over workers; the counters of each replay are in the table
under "GPU proof"). The host counters below are those of each burst (+-1 s):

| Evicted, 4 workers | Share | What the host did |
| --- | --- | --- |
| 38.3 s (17:06, a burst of 16.3 s) and 17.3 s (19:23, 14.1 s) | 78 % | one kcompactd pass each. 17:06: no reclaim around it (kswapd 0, direct 0, no stall, no swap-out; 142-147 GiB free), 36.8M pages scanned after one wake, 4.98M migrated (19 GiB). 19:23: 22.3M scanned in three wakes, 4.21M migrated, kswapd 0.86M, 54 stalls, 2 pages swapped out |
| 5.9 s (19:09) | 8 % | one mixed burst: kcompactd 18.6M in 71 wakes, 5.3M migrated, kswapd 10.5M, direct reclaim 0.41M, 3,503 stalls, 582k swapped out |
| 4.7 s (17:06) and 4.7 s (19:23) | 13 % | bursts of 0.03-1.3 s per worker: kswapd scan (67.5M pages in five bursts; 24.9M in one), compaction stalls (6.6k; 3.7k), direct reclaim 0.21M and 0.69M, 7.4k and 68k pages swapped out |

Over the whole 17:06 replay: kswapd scanned 84.7M pages and direct reclaim 0.53M
(0.6 %), 7,006 direct compaction stalls, 8.04M pages migrated, 8,283 pages
swapped out, of which 172 pinned ones, while 380,529 pinned pages were moved
(0.05 % swapped); `MemFree` fell to 2.46 GiB. In the 19:23 replay 44,241 pinned
pages were moved and none swapped. Whatever moves a page of an unlocked pinned
block evicts the worker, and the page is mostly moved by compaction, not
swapped.

The repeat pair's `base2` replay (21:59, 11.7 s evicted in all, in nine bursts)
splits differently. The two largest, 6.3 s at +297.6 s and 3.6 s at +221.9 s
(85 %), came with kswapd scans of 2.67M and 13.2M pages, direct scans of 343k
and 59.7k, swap-outs of 352k and 2.15k pages and 574 and 1,256 compaction
stalls, while kcompactd scanned only 6.9k and 0.81M pages in them; the two
kcompactd passes (4.21M and 2.17M scanned) evicted 0.24 s and 0.52 s. One
replay is no share, but it shows that on this host the time also sits with
reclaim and swap-out, which is the part ranks 1 and 2 act on.

The fix arm ran under the same settings. Its second replay was as heavy as the
base replays that evicted (6.25M pages migrated against 5.4-8.0M) and had more
direct reclaim than any of them (1.67M pages, 8,862 compaction stalls, 1.98M
pages swapped out); in its third kcompactd scanned 17.5M pages and migrated
2.31M. Every pinned block of the workers was locked: none moved, none was
swapped, no worker was evicted. The workers' unlocked anonymous pages, which
are not KFD buffers, were moved 83,845 times and swapped out 639,267 times in
the same run without an eviction. So the settings below matter to a process
whose pinned pages are not locked; with the lock none is needed.

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
  reclaim and direct compaction, which these settings do not touch; it is not
  one of the long kcompactd passes above.

With 503 GiB of RAM, the gap between the kswapd wake-up (`low`) and its
sleep (`high`) is `max(min_free_kbytes/4, RAM x watermark_scale_factor/10000)`:
0.5 GiB today, so kswapd wakes below 0.59 GiB free and sleeps above 1.09 GiB.

Proposals, ranked by the evidence that the setting acts on the evicted time
above, then by the share of it the setting can touch. No sysctl is shown to
remove the 78 % that lies in the kcompactd passes; the first two rows touch the
13 % and part of the 8 %, and the lock is what removes the dependence on all of
them.

| Rank | Setting | Proposal | Evidence | Effect | Risk |
| --- | --- | --- | --- | --- | --- |
| 1 | `vm.watermark_scale_factor` | 10 to 100 | `MemFree` fell to 2.2-3.1 GiB in 9 of 10 loads (the repeat pair's two included); compaction stalls and direct reclaim in the bursts that hold 13 % of the evicted time (7,006 stalls in the 17:06 replay, 8,862 in the fix arm's second); kswapd did 91-99.8 % of the scan, so a runway helps the part direct reclaim did | `low` 0.59 to 5.1 GiB, `high` 1.09 to 10.1 GiB: kswapd starts with a 5 GiB runway | 5-10 GiB of RAM kept free |
| 1 | `vm.min_free_kbytes` | 91888 to 2097152 (2 GiB) | same | with the above `low` 7 GiB, `high` 12 GiB; direct reclaim only below 2 GiB | 2 GiB reserved |
| 2 | `vm.swappiness` | 60 to 1 | swap-out in bursts that hold 17 % of the evicted time, together with compaction; but 172 pinned pages were swapped against 380,529 moved (17:06) and none in the 19:23 replay | reclaim takes page cache (370+ GiB of it) before anonymous and shmem pages | an anonymous-heavy overload OOMs earlier instead of swapping |
| 3 | `vm.extfrag_threshold` | 500 to 1000 | [UNVERIFIED] the one sysctl that can stop kcompactd passes for large orders: the kernel does not compact a zone whose fragmentation index is at or below it, for orders above 3. It would touch the 78 % if the second trace shows wakes of order 4 and above [INFERENCE]. The 7.1 source was not at hand and that test has been reworked in recent kernels: check `mm/compaction.c` first | no compaction for requests of THP or huge-page size | THP and hugetlb allocations lose compaction (THP is `madvise` only here) |
| - | `vm.compaction_proactiveness` | keep 20 | 26 of 26 kcompactd passes followed kswapd passes and none came with kswapd idle; the hours without a load had none | none | |
| - | `vm.watermark_boost_factor` | keep 15000 | the boost of zone Normal was above 0 in 61 of 3,294 s of the fix arm (up to 0.84 GiB, 1 s of it in the third replay) and not in the passes at 81 GiB free | none | |
| - | loader | `posix_fadvise(DONTNEED)` after each shard, or direct I/O | the 168 GiB checkpoint read fills the page cache that kswapd then has to empty (it scanned 84.7M pages, 323 GiB, in the 17:06 replay) | the checkpoint stops filling the cache | code change |

Rank 1 cannot cure the replay's bursts: a demand 16 GiB above `MemFree`
exhausts any runway of 5-12 GiB. It only makes the allocating processes reclaim
and compact less often.

To try ranks 1 and 2 at runtime (not persisted; the values to go back to are the
ones above), the user runs:

```bash
sudo sysctl -w vm.watermark_scale_factor=100 vm.min_free_kbytes=2097152 vm.swappiness=1
sudo sysctl -w vm.watermark_scale_factor=10 vm.min_free_kbytes=91888 vm.swappiness=60  # back
```

and rank 3 only if the second trace shows wakes of order 4 and above:

```bash
sudo sysctl -w vm.extfrag_threshold=1000
sudo sysctl -w vm.extfrag_threshold=500  # back
```

A rerun of the proof's base arm (`ARMS=base`) under the new values, with a fresh
replay, shows whether a pass still evicts. Loads differ with the state of the
host: the base arm's replays migrated 8.04M, 5.42M and 5.87M pages and evicted
5.9-43.0 s, the other two migrated 25k and 0 and evicted nothing. Read a rerun
by the pages migrated: a setting helps if a load that migrates 5M pages or more
stops evicting; a load that migrates less proves nothing.

## Status

- Done: the fix and its tests; the sampler, the report (per proof window and
  kcompactd passes), the fdinfo check, the acceptance tool and their tests; the
  proof driver, its hook and their tests; the mechanism check; the 37-minute
  passive run, the three-hour run (16:57-19:57, no server of ours); the GPU
  proof (both arms, one tree) and its repeat pair (`fix2`, `base2`); the switch
  declared in `vllm/envs.py`; the step-rate gate.
- Result: `evicted_ms` stayed 0 in all four workers through three replays of
  `fix`, the second as heavy as the base replays that evicted (6.25M pages
  migrated against 5.4-8.0M) and the third a long kcompactd pass (17.5M pages
  scanned), where the base arm's heavy replay (5.87M) evicted them 4.9-5.8 s
  each; `fix2` stayed at 0 in a lighter replay (1.82M migrated) where `base2`
  was evicted 2.5-3.5 s per worker (5.95M). The lock covered every candidate
  and the fdinfo total equals `VmLck`; TTFT and ITL did not move. The first
  pair's driver verdict is FAILED on the output-throughput gate it had then
  (see "Latency and throughput"); it was not overridden. With the gate on the
  decode steps per second the four arms are PROVEN, and `fix2` against `base2`
  is +1.27 % (c8) and +0.95 % (c1) in steps per second (see "The repeat pair").
- Not done: the two bpftrace runs (the user's); any host setting; a real TP2
  neighbour (`NEIGHBOUR=cmd`) instead of the replay; locking the buffers of the
  API server and EngineCore processes (the fix arm's raw KFD log shows each of
  them evicted 20 times in the second load; they run no kernel); registering
  the other `VLLM_RDNA_*` recipe variables (14 still log `Unknown vLLM
  environment variable detected`); a rebuild of the extensions from the SHA
  (their provenance rests on hashes and file times, see "The repeat pair").

## Provenance

- Branch `fix/mlock-pinned-host`, from `f36f05b9b7` (a commit of `rdna_extras`,
  its merge base with `origin/rdna_extras` at `53cec79960`). Tree under test,
  first pair: `d892e6e7dbc79198c076a5812b738fed619062aa`; `vllm/` in it is
  byte-identical to `dd0eea91eb`. Repeat pair:
  `efbc0ec34ad3f00df47335e80f6c5d3ad87505f2` (the switch in `vllm/envs.py`).
  The other commits after `d892e6e7db` change tools, tests and this document
  only (`22b771b9d4`, `61fa64b176`, `736747642c`, `cd5d2d6737`, `4d90701dc9`,
  `8f496a4061`, `019895b519`, `bc2d2e2e99`, `8f2ac9ff07` and the one that holds
  this text).
- On par1-cs25: `~/w4a8_runs/evictfix-proof/` (`arms/base`, `arms/base/attempt1`,
  `arms/fix` with `proof/` = sampler output, `results/verdict.*`, `proof.log`),
  `~/w4a8_runs/evictfix-proof2/` (`arms/fix2`, `arms/base2`, the driver's
  verdict with the old gate, `proof.log`), `~/w4a8_runs/evictfix-proof4/` (the
  four arms linked, `results.out` = the verdict with the step-rate gate) and
  `~/w4a8_runs/evictfix-proof2-gate/` (the same for the repeat pair alone),
  `~/w4a8_runs/evictfix/` (the three-hour run `20261105-165723-light`, the
  37-minute run `20261105-155311`, the fdinfo logs, the zone sampler of the fix
  arm, `so_built.sha256`, `native_blobs.txt`, `so_freshness.txt` and
  `fresh_tu.py`). Exact-SHA trees
  `~/vllm-evictfix-d892e6e7db` and `~/vllm-evictfix-dd0eea91eb`; the repeat
  pair ran from `~/vllm-evictfix-efbc0ec34a` and its verdicts were computed
  from `~/vllm-evictfix-bc2d2e2e99`; the first driver ran from
  `~/vllm-evictfix-bdc1d8d425`, synced in place to `d892e6e7db` at 19:15:06.
