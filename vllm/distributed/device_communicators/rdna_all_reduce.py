# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aron Hsiao
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Push-based one-shot all-reduce for small TP messages on gfx1030 (2..8 ranks).

Ported from leapdragon/vllm-rdna2-qwen T44/T44b (Aron Hsiao). The VRAM-flag
protocol, abort-record decode, wedge marker, and rdna_ar_check() are the same
as that tree. Default-off, persist/self-test barriers, integer device index,
and PIX logging are this fork.

Opt-in via VLLM_RDNA_AR=1. Default is off. VLLM_FORCE_CUSTOM_ALL_REDUCE does
not enable this path. When enabled, eligible tensors dispatch ahead of stock
CUSTOM / PYNCCL. fp16, bf16, and fp32 are accepted, including the flat
non-contiguous spans custom all-reduce already allows.

Staging and flags are uncached device memory (peer announce is a posted P2P
store; we poll locally). Sequence numbers live on device (graph-capture
safe). One-shot only: messages up to min(VLLM_RDNA_AR_MAX_KB,
VLLM_RDNA_AR_ONESHOT_KB) (both default 64 KiB) use it, larger ones go to RCCL.
The push two-shot that used to cover larger messages was removed: it returned
zeros above the one-shot gate and made the boot self-test disable the whole
backend. A two-shot request (VLLM_RDNA_AR_ALGO=twoshot, or a MAX_KB above the
one-shot gate) is ignored with one warning.
VLLM_RDNA_AR_BLOCKS / VLLM_RDNA_AR_PACE pace PCIe push bursts.
The finisher sums the staged slices with 16-byte loads in fixed rank order
(fp32, bit-equal to the scalar loop); VLLM_RDNA_AR_VEC=0 restores the scalar
finisher and the old launch widths (A/B and rollback).
A one-shot wait for a peer that has not reached the collective yet is rank
skew, not a fault: a rank arrives seconds late after a JIT compile or a long
host step, minutes late when its host stalls, and the engine cannot step
without it anyway. So that wait is bounded only by VLLM_RDNA_AR_WAIT_MS
(default 600000, past the engine's 300 s execute timeout). Past ~2 s the
kernel reads the peer's own flag slot over P2P once a second: a peer that has
announced the collective while its store into our slot never lands is a lost
P2P write, aborted 10 s after the announce was seen. Waits past ~2 s that
complete are logged as a late peer; a thread logs waits that are still in
progress (every 30 s from 10 s on), naming the rank the others wait for, and
any eviction of this process's GPU queues by the kernel driver of 1 s or more
(KFD evicted_ms): a rank whose queues are evicted is a late peer to the rest.

T44b wedge handling: an abort records phase/peer/sequence in a host-mapped
word, and the abort is sticky in the kernel: later collectives on that
instance skip their waits (output = local input, sequence still announced so
peers move on), so a CUDA-graph step finishes instead of re-spinning to the
engine's execute timeout. rdna_ar_check() reads every live instance's record
once per engine step (the TP and the EP group each get their own instance)
and fails the step -- the engine stops with "rdna_ar wedged". A lost P2P write
also writes a marker under VLLM_CACHE_ROOT so the next boot stays on RCCL.
Results of the failing step are wrong by construction; nothing reroutes it
live.

Hierarchical mode (VLLM_RDNA_AR_HIER=auto|1|0, default auto): a group that spans two
PCIe switch islands of equal size (TP8 on 2 x 4 V620: each island is the GPUs
behind one root port, and the cross-root path through the CPU caps near
27 GB/s per direction and collapses with 3 or more independent flows) reduces
inside each island, exchanges one island partial per owner across the root
complex, and gathers inside the island. Messages below VLLM_RDNA_AR_HIER_MIN_KB
(default 20) keep the flat one-shot unchanged. From there the results carry
one more rounding to the working dtype than the flat sum (island partials) and
are identical on every rank. The one-shot gate of such a group is
VLLM_RDNA_AR_HIER_MAX_KB (default 1024; 0 = the flat gate), not MAX_KB /
ONESHOT_KB, which keep gating the all-gather and the argmax and the all-reduce
of every group the mode does not apply to. Islands come from the sysfs PCI
path of each rank's GPU (VLLM_RDNA_AR_HIER_ISLANDS="0,1,2,3|4,5,6,7" overrides
it); anything but two equal islands of 2..4 ranks stays flat. auto applies to a
world of 8 only and stays silent where it does not apply; 1 applies to any two
equal islands and warns where it cannot; 0 keeps the flat exchange.
VLLM_RDNA_AR_HIER_ALGO=auto|flat|hier (flat = the flat exchange with the island
arithmetic, for diagnostics), _X (owners per island) and _BLOCKS tune it.
"""

from __future__ import annotations

import glob
import os
import threading
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

_instances = 0
_MARKER_NAME = "rdna_ar_wedged"
# Matches RDNA_AR_ONESHOT_MAX in csrc/rocm/rdna_allreduce.cuh.
_ONESHOT_DEFAULT_KB = 64
# Algorithm ids of rdna_ar_all_reduce_algo, as RDNA_AR_ALGO_* in
# csrc/rocm/rdna_allreduce_hier.cuh (-1: by size).
ALGO_AUTO, ALGO_FLAT, ALGO_FLAT_ISL, ALGO_HIER = -1, 0, 1, 2
_ALGO_NAMES = {ALGO_AUTO: "auto", ALGO_FLAT_ISL: "flat", ALGO_HIER: "hier"}
_HIER_MAX_ISLAND = 4  # RDNA_AR_MAX_G
_HIER_DEFAULT_MIN_KB = 20
_HIER_DEFAULT_MAX_KB = 1024
_HIER_AUTO_WORLD = 8  # "auto" applies to 2 x 4 (the measured topology)
# Record phases (bits 8-11): 1 grid barrier, 2 flag wait (flat exchange), and
# the hier waits, as RDNA_AR_PH_* in csrc/rocm/rdna_allreduce_hier.cuh.
_PH_GRID = 1
_PH_STEPS = {
    3: "island reduce-scatter",
    4: "cross-root exchange",
    5: "island all-gather",
}
# Phases whose wait is for a peer's flag: a lost write there is a wedge.
_PH_FLAG_WAITS = (2, 3, 4, 5)
# Record bits, as RDNA_AR_REC_* in csrc/rocm/rdna_allreduce.cuh.
_REC_LIVE = 0x2  # report[1]: the wait is still in progress
_REC_SEEN = 0x4  # the peer's own flag slot showed the collective
# The live-wait thread logs a wait once it is this old, then at this cadence.
_WATCH_FIRST_MS = 10_000
_WATCH_EVERY_MS = 30_000
# KFD's running total of the time a process's GPU queues spent evicted (it
# grows when the queues come back). While a rank is evicted, its peers wait.
_KFD_EVICTED = "/sys/class/kfd/kfd/proc/{pid}/stats_*/evicted_ms"
_EVICT_LOG_MS = 1_000


def resolve_gate_kb() -> tuple[int, str | None]:
    """Effective one-shot gate in KiB, and why a requested setting was dropped.

    Two-shot no longer exists, so the gate is
    min(VLLM_RDNA_AR_MAX_KB, VLLM_RDNA_AR_ONESHOT_KB): larger messages fall
    back to RCCL instead of reaching a kernel that is not there.
    """
    max_kb = int(os.getenv("VLLM_RDNA_AR_MAX_KB", str(_ONESHOT_DEFAULT_KB)))
    oneshot_kb = int(os.getenv("VLLM_RDNA_AR_ONESHOT_KB", str(_ONESHOT_DEFAULT_KB)))
    if oneshot_kb <= 0:
        oneshot_kb = _ONESHOT_DEFAULT_KB
    gate_kb = min(max_kb, oneshot_kb)
    algo = os.getenv("VLLM_RDNA_AR_ALGO", "auto").strip().lower()
    dropped = []
    if algo in ("twoshot", "2stage"):
        dropped.append(f"VLLM_RDNA_AR_ALGO={algo}")
    if max_kb > gate_kb:
        dropped.append(f"VLLM_RDNA_AR_MAX_KB={max_kb}")
    if not dropped:
        return gate_kb, None
    what = " and ".join(dropped)
    return gate_kb, (
        f"{what} asked for the removed two-shot kernel; "
        f"using one-shot up to {gate_kb} KiB and RCCL above"
    )


@dataclass(frozen=True)
class HierConfig:
    """VLLM_RDNA_AR_HIER* of this process (see the module docstring)."""

    mode: str  # "auto" (default), "1" (warn when it cannot apply) or "0" (off)
    islands: str  # VLLM_RDNA_AR_HIER_ISLANDS; "" = detect from sysfs
    algo: int  # ALGO_AUTO, ALGO_FLAT_ISL or ALGO_HIER
    min_kb: int  # below it the legacy flat one-shot; from it the 3-phase algorithm
    max_kb: int  # largest all-reduce of a hier group (0 = the flat gate)
    owners: int  # ranks per island that cross the root complex (0 = by size)
    blocks: int  # blocks of a hier launch (0 = by size)
    notes: tuple[str, ...]  # env values that were not usable


def resolve_hier_config() -> HierConfig:
    notes: list[str] = []

    def int_knob(name: str, default: int, lo: int, hi: int) -> int:
        raw = os.getenv(name, "").strip()
        if not raw:
            return default
        try:
            val = int(raw)
        except ValueError:
            val = lo - 1
        if not lo <= val <= hi:
            notes.append(f"{name}={raw} is not an integer in {lo}..{hi}")
            return default
        return val

    raw_mode = os.getenv("VLLM_RDNA_AR_HIER", "auto").strip().lower() or "auto"
    mode = {"off": "0", "false": "0", "on": "1", "true": "1"}.get(raw_mode, raw_mode)
    if mode not in ("0", "1", "auto"):
        notes.append(f"VLLM_RDNA_AR_HIER={raw_mode} is not 0, 1 or auto")
        mode = "0"
    raw_algo = os.getenv("VLLM_RDNA_AR_HIER_ALGO", "auto").strip().lower() or "auto"
    algo = {"auto": ALGO_AUTO, "flat": ALGO_FLAT_ISL, "hier": ALGO_HIER}.get(raw_algo)
    if algo is None:
        notes.append(f"VLLM_RDNA_AR_HIER_ALGO={raw_algo} is not auto, flat or hier")
        algo = ALGO_AUTO
    return HierConfig(
        mode=mode,
        islands=os.getenv("VLLM_RDNA_AR_HIER_ISLANDS", "").strip(),
        algo=algo,
        min_kb=int_knob("VLLM_RDNA_AR_HIER_MIN_KB", _HIER_DEFAULT_MIN_KB, 0, 1 << 20),
        max_kb=int_knob("VLLM_RDNA_AR_HIER_MAX_KB", _HIER_DEFAULT_MAX_KB, 0, 1 << 20),
        owners=int_knob("VLLM_RDNA_AR_HIER_X", 0, 0, _HIER_MAX_ISLAND),
        blocks=int_knob("VLLM_RDNA_AR_HIER_BLOCKS", 0, 0, 32),
        notes=tuple(notes),
    )


def pci_island_key(pci_path: str) -> str | None:
    """The root port a GPU hangs off, from its resolved sysfs path.

    /sys/devices/pci0000:c0/0000:c0:03.1/0000:c5:00.0/... is behind root port
    0000:c0:03.1 of host bridge pci0000:c0: the first two components below
    /sys/devices. Two GPUs with different keys talk through the root complex.
    """
    parts = [p for p in pci_path.split("/") if p]
    if "devices" not in parts:
        return None
    key = parts[parts.index("devices") + 1 :][:2]
    return "/".join(key) if len(key) == 2 else None


def device_island_key(dev_idx: int) -> str | None:
    """pci_island_key of the GPU with torch index dev_idx (None if unknown)."""
    props = torch.cuda.get_device_properties(dev_idx)
    bdf = f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}"
    found = sorted(glob.glob(f"/sys/bus/pci/devices/{bdf}.*"))
    return pci_island_key(os.path.realpath(found[0])) if found else None


def _valid_islands(island_of_rank: list[int]) -> bool:
    g = len(island_of_rank) // 2
    return (
        len(island_of_rank) == 2 * g
        and 2 <= g <= _HIER_MAX_ISLAND
        and island_of_rank.count(0) == g
        and island_of_rank.count(1) == g
    )


def classify_islands(keys: list[str | None]) -> list[int] | None:
    """Island (0 = the one of rank 0, else 1) of every rank, or None.

    keys[r] is rank r's pci_island_key. Hier applies to exactly two distinct
    keys with the same number (2..4) of ranks each; anything else (one switch,
    three roots, 3 + 5 ranks, an unknown key) stays on the flat exchange.
    """
    if any(k is None for k in keys):
        return None
    distinct = list(dict.fromkeys(keys))
    if len(distinct) != 2:
        return None
    island_of_rank = [distinct.index(k) for k in keys]
    return island_of_rank if _valid_islands(island_of_rank) else None


def parse_island_override(spec: str, world_size: int) -> list[int]:
    """Island of every rank from "0,1,2,3|4,5,6,7" (ValueError if unusable)."""
    groups = spec.split("|")
    island_of_rank = [-1] * world_size
    try:
        for k, group in enumerate(groups):
            for r in (int(x) for x in group.split(",") if x.strip()):
                if not 0 <= r < world_size or island_of_rank[r] != -1:
                    raise ValueError(f"rank {r} is out of range or listed twice")
                island_of_rank[r] = k
    except ValueError as e:
        raise ValueError(f"VLLM_RDNA_AR_HIER_ISLANDS={spec!r}: {e}") from None
    if len(groups) != 2 or -1 in island_of_rank or not _valid_islands(island_of_rank):
        raise ValueError(
            f"VLLM_RDNA_AR_HIER_ISLANDS={spec!r} must split ranks 0..{world_size - 1} "
            f"into two groups of equal size 2..{_HIER_MAX_ISLAND}"
        )
    if island_of_rank[0] == 1:  # island 0 is the one of rank 0
        island_of_rank = [1 - k for k in island_of_rank]
    return island_of_rank


def marker_path() -> str:
    from vllm import envs

    return os.path.join(envs.VLLM_CACHE_ROOT, _MARKER_NAME)


def _fields(code: int) -> tuple[int, int, int, int]:
    """(phase, peer, measured wait in ms, sequence) of a kernel record."""
    return (
        (code >> 8) & 0xF,
        (code >> 12) & 0xF,
        ((code >> 16) & 0xFFFF) * 16,  # measured wait, 16 ms units
        (code >> 32) & 0xFFFFFFFF,
    )


def describe_abort(code: int, rank: int) -> str:
    """Decode the kernel's abort record into one sentence.

    Bit layout matches leapdragon/vllm-rdna2-qwen T44b (Aron Hsiao), plus the
    _REC_SEEN bit: whether the peer had announced the collective.
    """
    phase, peer, ms, seq = _fields(code)
    if phase == 1:
        what = (
            "its own blocks never reached the grid barrier "
            "(a launch on this GPU stalled)"
        )
    elif code & _REC_SEEN:
        what = (
            f"peer rank {peer} had announced it (its own flag slot showed the "
            "collective), but its posted P2P write into this rank's flag slot "
            "never arrived: the write was lost on this fabric"
        )
    else:
        what = (
            f"peer rank {peer} never reached it (its own flag slot, read over "
            "P2P, stayed behind): that rank stalled -- a host step, a JIT "
            "compile or its GPU queue -- or P2P reads from it fail"
        )
    step = _PH_STEPS.get(phase)
    where = f" ({step})" if step else ""
    return (
        f"rank {rank} timed out after ~{ms} ms of spinning at collective "
        f"#{seq}{where}: {what}"
    )


def describe_late(code: int, rank: int) -> str:
    """Decode the one-shot late-peer record (report[1], abort-record layout)."""
    _, peer, ms, seq = _fields(code)
    step = _PH_STEPS.get((code >> 8) & 0xF)
    where = f" ({step})" if step else ""
    msg = (
        f"rank {rank} waited ~{ms} ms for peer rank {peer} at collective #{seq}{where}"
    )
    if code & _REC_SEEN:
        msg += f" (peer rank {peer} had announced it; its write landed late)"
    return msg


def describe_wait(code: int, rank: int) -> str:
    """Decode a live record: a one-shot wait that is still in progress."""
    _, peer, ms, seq = _fields(code)
    if code & _REC_SEEN:
        state = (
            f"peer rank {peer} has announced it but its write has not landed "
            "here (aborted as a lost write 10 s after the announce)"
        )
    else:
        state = (
            f"peer rank {peer} has not reached it yet (that rank is stalled; "
            "the collective completes when it arrives)"
        )
    step = _PH_STEPS.get((code >> 8) & 0xF)
    where = f" ({step})" if step else ""
    return (
        f"rank {rank} has been waiting ~{ms} ms for peer rank {peer} at "
        f"collective #{seq}{where}: {state}"
    )


def _watch_once(logged: dict[int, tuple[tuple[int, int], int]]) -> None:
    """One pass of the live-wait watcher: log long in-progress waits.

    The model thread cannot report them: it is blocked behind the GPU that
    spins. `logged` maps an instance handle to the (sequence, peer) of the
    wait it last logged and the wait time it logged.
    """
    for inst in list(_live):
        if inst.disabled or inst._ops.rdna_ar_timed_out(inst.handle):
            continue
        code = int(inst._ops.rdna_ar_slow_info(inst.handle))
        if not code & _REC_LIVE:
            continue
        _, peer, ms, seq = _fields(code)
        if ms < _WATCH_FIRST_MS:
            continue
        prev = logged.get(inst.handle)
        if (
            prev is not None
            and prev[0] == (seq, peer)
            and ms - prev[1] < _WATCH_EVERY_MS
        ):
            continue
        logged[inst.handle] = ((seq, peer), ms)
        logger.warning("rdna_ar: %s", describe_wait(code, inst.rank))


def _watch_evictions(state: dict) -> None:
    """One pass of the eviction watch: log KFD queue evictions of >= 1 s.

    While the kernel driver has this process's GPU queues evicted, nothing it
    queued runs, so every peer waits in its next collective for this rank.
    `state` holds the sysfs path ("path", None when not exposed) and the
    last total read ("total").
    """
    path = state.get("path")
    if path is None:
        return
    try:
        with open(path) as f:
            total = int(f.read().strip())
    except (OSError, ValueError):
        return
    last = state.get("total")
    state["total"] = total
    if last is not None and total - last >= _EVICT_LOG_MS:
        rank = _live[0].rank if _live else "?"
        logger.warning(
            "rdna_ar: rank %s: the kernel driver had this process's GPU queues "
            "evicted for ~%d ms (KFD queue eviction, %s); its peers waited for "
            "it in their next collective meanwhile",
            rank,
            total - last,
            path,
        )


def _watch_live_waits() -> None:
    logged: dict[int, tuple[tuple[int, int], int]] = {}
    paths = sorted(glob.glob(_KFD_EVICTED.format(pid=os.getpid())))
    evictions: dict = {"path": paths[0] if paths else None}
    while True:
        time.sleep(1.0)
        _watch_once(logged)
        _watch_evictions(evictions)


_watcher: threading.Thread | None = None


def _start_watcher() -> None:
    global _watcher
    if _watcher is None:
        _watcher = threading.Thread(
            target=_watch_live_waits, name="rdna_ar_watch", daemon=True
        )
        _watcher.start()


# Every instance that passed its self-test, in creation order (one per
# process group: TP, EP, ...). Each has its own staging, sequence and abort
# word, so a wedge on one does not stall the others, but any wedge fails the
# step.
_live: list[RdnaOneShotAllReduce] = []


def rdna_ar_check() -> None:
    """Per-step wedge check of every live instance (no-op when none)."""
    for inst in _live:
        if not inst.disabled:
            inst.check()


class RdnaOneShotAllReduce:
    # Island (0/1) of every rank while the hierarchical all-reduce is on.
    hier_islands: list[int] | None = None
    _hier_cfg: HierConfig | None = None

    def __init__(self, group: ProcessGroup, device: torch.device) -> None:
        global _instances
        from vllm import _custom_ops as ops

        self.disabled = True
        self.handle = -1
        self._ops = ops
        # VLLM_RDNA_AR_GATHER=0 keeps all-gather / vocab argmax on RCCL (A/B and
        # rollback); the all-reduce path is unaffected.
        self._hip_collectives = os.environ.get("VLLM_RDNA_AR_GATHER", "1") != "0"
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        # One-shot only; messages above the gate go to RCCL. The hier gate
        # (a group of two islands) replaces it once the mode is on.
        flat_kb, dropped = resolve_gate_kb()
        self.gate_bytes = flat_kb * 1024
        if not (2 <= self.world_size <= 8):
            return
        if dropped is not None and self.rank == 0:
            logger.warning("rdna_ar: %s.", dropped)
        # T44b: a previous run on this machine wedged -- stay on RCCL
        # until the marker is removed.
        marker = marker_path()
        if os.path.exists(marker):
            try:
                with open(marker, encoding="utf-8") as f:
                    why = f.read().strip().replace("\n", " ")[:400]
            except OSError:
                why = "unreadable marker"
            logger.warning(
                "rdna_ar: disabled -- a previous run wedged on this machine "
                "(%s). Using RCCL for the small collectives. Delete %s to try "
                "the one-shot path again (a slow or ACS-redirected GPU P2P "
                "path is the usual cause), or set VLLM_RDNA_AR=0 to keep RCCL "
                "without this warning.",
                why,
                marker,
            )
            return
        # Integer index: some Torch APIs reject torch.device objects and
        # previously silently disabled this backend (PR #5).
        dev_idx = (
            device.index
            if device.index is not None
            else torch.accelerator.current_device_index()
        )
        gathered: list = [None] * self.world_size
        dist.all_gather_object(gathered, int(dev_idx), group=group)
        pix = False
        try:
            physical = [
                current_platform.visible_device_id_to_physical_device_id(int(d))
                for d in gathered
            ]
            pix = bool(current_platform.is_pix_connected(physical))
        except Exception as e:  # noqa: BLE001
            logger.warning("rdna_ar: PIX topology query failed (%s)", e)
        pix_votes: list = [None] * self.world_size
        dist.all_gather_object(pix_votes, pix, group=group)
        pix = all(bool(v) for v in pix_votes)
        device_ids = torch.tensor(gathered, dtype=torch.int64)
        my_name = f"/vllm_rdna_ar_{os.getpid()}_{_instances}"
        names: list = [None] * self.world_size
        dist.all_gather_object(names, my_name, group=group)
        shm_name = names[0]
        _instances += 1
        # Hier needs the islands before the staging exists: its gate can be
        # larger than the flat one, and the slots are sized for the larger.
        hier_cfg, hier_plan, hier_why = self._plan_hier(dev_idx, group)
        capacity_kb = flat_kb
        if hier_plan is not None:
            capacity_kb = max(flat_kb, hier_cfg.max_kb)

        # Ordered init: every rank runs every barrier no matter what happens
        # locally (a rank that bailed out of the loop deadlocked peers).
        packed = None
        err: str | None = None
        for r in range(self.world_size):
            if r == self.rank and err is None:
                try:
                    with torch.accelerator.device_index(dev_idx):
                        packed = ops.rdna_ar_init(
                            self.rank,
                            self.world_size,
                            device_ids,
                            capacity_kb * 1024,
                            shm_name,
                            flat_kb * 1024,
                        )
                except Exception as e:  # noqa: BLE001
                    err = str(e)
            dist.barrier(group=group)
        status: list = [None] * self.world_size
        dist.all_gather_object(status, err, group=group)
        if any(s is not None for s in status):
            logger.warning(
                "rdna_ar: disabled for this group -- init failed on some rank: %s",
                [s for s in status if s is not None][:1],
            )
            return
        assert packed is not None
        raw = packed.numpy().tobytes()
        self.handle = int.from_bytes(raw[:8], "little", signed=True)
        handles: list = [None] * self.world_size
        dist.all_gather_object(handles, raw[8:], group=group)
        buf = torch.frombuffer(bytearray(b"".join(handles)), dtype=torch.uint8).view(
            self.world_size, -1
        )
        err = None
        try:
            with torch.accelerator.device_index(dev_idx):
                ops.rdna_ar_connect(self.handle, buf.contiguous())
        except Exception as e:  # noqa: BLE001
            err = str(e)
        dist.all_gather_object(status, err, group=group)
        if any(s is not None for s in status):
            logger.warning("rdna_ar: disabled -- connect failed: %s", status)
            return
        dist.barrier(group=group)
        # Hierarchical mode (opt-in): needs the connected staging, and the
        # self-test below covers it.
        self.hier_islands = self._apply_hier(
            hier_cfg, hier_plan, hier_why, flat_kb, group
        )

        # Boot self-test: on boards where GPU P2P is slow or broken, init can
        # succeed while every collective then spins to its cap and aborts
        # WITHOUT writing the output.
        err = self._self_test(device, group)
        dist.all_gather_object(status, err, group=group)
        if any(s is not None for s in status):
            logger.warning(
                "rdna_ar: disabled -- boot self-test failed on some rank "
                "(weak GPU peer-to-peer on this board? falling back to RCCL): "
                "%s",
                [s for s in status if s is not None],
            )
            return
        dist.barrier(group=group)
        self.disabled = False
        _live.append(self)
        _start_watcher()
        hier = ""
        if self.hier_islands is not None and self._hier_cfg is not None:
            cfg = self._hier_cfg
            groups = [
                [r for r, k in enumerate(self.hier_islands) if k == i] for i in (0, 1)
            ]
            hier = (
                f"; hier islands {groups}, algo {_ALGO_NAMES[cfg.algo]}, hier from "
                f"{cfg.min_kb} KB, owners {cfg.owners or len(groups[0])}, "
                f"blocks {cfg.blocks or 'auto'}"
            )
        logger.info(
            "rdna_ar: one-shot all-reduce active (handle %d, rank %d/%d, "
            "devices %s, pix=%s, up to %d KB, RCCL above; "
            "blocks cap %s, pace %s, vec %s%s)",
            self.handle,
            self.rank,
            self.world_size,
            gathered,
            pix,
            self.gate_bytes // 1024,
            os.getenv("VLLM_RDNA_AR_BLOCKS", "auto"),
            os.getenv("VLLM_RDNA_AR_PACE", "0"),
            os.getenv("VLLM_RDNA_AR_VEC", "1"),
            hier,
        )

    def _plan_hier(
        self, dev_idx: int, group: ProcessGroup
    ) -> tuple[HierConfig, list[int] | None, str]:
        """Find out whether the hierarchical all-reduce applies to this group.

        Returns (config, island of every rank or None, why not). Every rank
        runs the same collectives whatever happens locally.
        """
        cfg = resolve_hier_config()
        if self.rank == 0:
            for note in cfg.notes:
                logger.warning("rdna_ar: %s; ignored.", note)
        if cfg.mode == "0":
            return cfg, None, ""
        key: str | None = None
        try:
            key = device_island_key(dev_idx)
        except Exception as e:  # noqa: BLE001
            logger.debug("rdna_ar: PCI path of device %s unknown (%s)", dev_idx, e)
        keys: list = [None] * self.world_size
        dist.all_gather_object(keys, key, group=group)
        islands: list[int] | None = None
        why = ""
        if cfg.islands:
            try:
                islands = parse_island_override(cfg.islands, self.world_size)
            except ValueError as e:
                why = str(e)
        else:
            islands = classify_islands(keys)
            why = f"root ports of ranks 0..{self.world_size - 1}: {keys}"
        if (
            islands is not None
            and cfg.mode == "auto"
            and self.world_size != _HIER_AUTO_WORLD
        ):
            islands = None
            why = (
                f"auto applies to {_HIER_AUTO_WORLD} ranks "
                f"(2 x {_HIER_AUTO_WORLD // 2}), this group has "
                f"{self.world_size}; VLLM_RDNA_AR_HIER=1 forces it"
            )
        return cfg, islands, why

    def _apply_hier(
        self,
        cfg: HierConfig,
        islands: list[int] | None,
        why: str,
        flat_kb: int,
        group: ProcessGroup,
    ) -> list[int] | None:
        """Turn the hierarchical all-reduce on in the connected instance.

        Returns the island of every rank, or None (flat exchange). Every rank
        runs the same collectives whatever happens locally.
        """
        if cfg.mode == "0":
            return None
        gate_kb = cfg.max_kb or flat_kb
        err: str | None = None
        if islands is not None:
            try:
                self._ops.rdna_ar_hier_init(
                    self.handle,
                    torch.tensor(islands, dtype=torch.int64),
                    cfg.owners,
                    cfg.min_kb,
                    cfg.algo,
                    cfg.blocks,
                    gate_kb,
                )
            except Exception as e:  # noqa: BLE001
                err = str(e)
        status: list = [None] * self.world_size
        dist.all_gather_object(status, err, group=group)
        if any(s is not None for s in status):
            # Ranks that did enable it must not run it against ranks that did
            # not: turn it off everywhere (an empty island list does that).
            try:
                self._ops.rdna_ar_hier_init(
                    self.handle, torch.empty(0, dtype=torch.int64), 0, 0, -1, 0, 0
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("rdna_ar: hier reset failed (%s)", e)
            islands, why = None, f"hier init failed on some rank: {err or status}"
        if islands is None:
            if self.rank == 0:
                log = logger.warning if cfg.mode == "1" else logger.info
                log(
                    "rdna_ar: hierarchical all-reduce (VLLM_RDNA_AR_HIER=%s) "
                    "does not apply, using the flat exchange: %s",
                    cfg.mode,
                    why or "not two equal islands of 2..4 ranks",
                )
            return None
        self._hier_cfg = cfg
        self.gate_bytes = gate_kb * 1024
        return islands

    def _self_test(self, device: torch.device, group: ProcessGroup) -> str | None:
        """Verified all-reduces on the fast path at three sizes.

        Returns an error string or None. Per size: one untimed warm-up, then
        REPEATS timed collectives judged on their minimum. With the
        hierarchical mode on, both of its algorithms also run at the largest
        and at an odd size, dtype by dtype, on position- and rank-dependent
        integers (exact in every dtype), so a chunk that lands in the wrong
        place fails. Never leave the barrier loop early -- a rank that stops
        calling barriers while its peer keeps looping deadlocks both.
        """
        repeats = 3
        err: str | None = None
        sync_dev = (
            device.index
            if device.index is not None
            else torch.accelerator.current_device_index()
        )
        # (elements, dtype, algorithm or None for the size dispatch, pattern)
        cases: list[tuple[int, torch.dtype, int | None, bool]] = [
            (1024, torch.float16, None, False),
            (4096, torch.float16, None, False),
            (self.gate_bytes // 2, torch.float16, None, False),
            (1024, torch.bfloat16, None, False),
        ]
        if self.hier_islands is not None:
            for hier_algo in (ALGO_FLAT_ISL, ALGO_HIER):
                cases += [
                    (self.gate_bytes // 2, torch.float16, hier_algo, True),
                    (4099, torch.float16, hier_algo, True),
                    (2049, torch.bfloat16, hier_algo, True),
                    (1537, torch.float32, hier_algo, True),
                ]
        try:
            with torch.accelerator.device_index(sync_dev):
                for trial, (numel, dtype, algo, pattern) in enumerate(cases):
                    if pattern:
                        idx = torch.arange(numel, device=device, dtype=torch.int64)
                        inp = ((idx * 5 + self.rank * 3) % 17).to(dtype)
                        expect = (
                            torch.stack(
                                [
                                    ((idx * 5 + r * 3) % 17)
                                    for r in range(self.world_size)
                                ]
                            )
                            .sum(0)
                            .to(torch.float32)
                        )
                    else:
                        inp = torch.full(
                            (numel,),
                            float(self.rank + 1) * (trial + 1),
                            dtype=dtype,
                            device=device,
                        )
                        expect = torch.full_like(
                            inp,
                            float(
                                (trial + 1)
                                * self.world_size
                                * (self.world_size + 1)
                                // 2
                            ),
                            dtype=torch.float32,
                        )
                    times: list[float] = []
                    for rep in range(repeats + 1):  # rep 0 = warm-up, untimed
                        dist.barrier(group=group)
                        logger.debug(
                            "[rdna_ar rank%d] trial%d rep%d barrier-out",
                            self.rank,
                            trial,
                            rep,
                        )
                        if err is not None:
                            continue
                        try:
                            t0 = time.perf_counter()
                            if algo is None:
                                out = self._ops.rdna_ar_all_reduce(self.handle, inp)
                            else:
                                out = self._ops.rdna_ar_all_reduce_algo(
                                    self.handle, inp, algo, 0
                                )
                            torch.accelerator.synchronize(sync_dev)
                            dt = time.perf_counter() - t0
                            logger.debug(
                                "[rdna_ar rank%d] trial%d rep%d call %.1fms",
                                self.rank,
                                trial,
                                rep,
                                dt * 1e3,
                            )
                            code = int(self._ops.rdna_ar_timeout_info(self.handle))
                            if code:
                                err = (
                                    f"spin-cap timeout in self-test trial "
                                    f"{trial} rep {rep} ({dt * 1e3:.0f} ms): "
                                    f"{describe_abort(code, self.rank)}"
                                )
                            elif not torch.equal(out.float(), expect):
                                bad = int((out.float() != expect).sum())
                                err = (
                                    f"wrong result in self-test trial {trial} "
                                    f"rep {rep} (algo {algo}, {numel} x {dtype}): "
                                    f"{bad} of {numel} elements differ, first "
                                    f"{out.float()[:4].tolist()} expected "
                                    f"{expect[:4].tolist()}"
                                )
                            elif rep > 0:
                                times.append(dt)
                        except Exception as e:  # noqa: BLE001
                            err = str(e)
                    if err is None:
                        best = min(times)
                        if best > 0.05:
                            err = (
                                f"self-test trial {trial} best "
                                f"{best * 1e3:.1f} ms of {repeats} for "
                                f"{numel * inp.element_size()} bytes (all: "
                                f"{', '.join(f'{t * 1e3:.1f}' for t in times)}"
                                f" ms; P2P too slow, RCCL will be faster)"
                            )
        except Exception as e:  # noqa: BLE001
            err = str(e)
        return err

    def should_use(self, inp: torch.Tensor) -> bool:
        return (not self.disabled) and self._ops.rdna_ar_can(self.handle, inp)

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        return self._ops.rdna_ar_all_reduce(self.handle, inp)

    def should_all_gather(self, inp: torch.Tensor) -> bool:
        """Last-dim all-gather of a small contiguous tensor via HIP push."""
        return (
            self._hip_collectives
            and (not self.disabled)
            and self._ops.rdna_ar_can_all_gather(self.handle, inp)
        )

    def all_gather(self, inp: torch.Tensor) -> torch.Tensor:
        return self._ops.rdna_ar_all_gather(self.handle, inp)

    def should_argmax(self, vals: torch.Tensor, idx: torch.Tensor) -> bool:
        return (
            self._hip_collectives
            and (not self.disabled)
            and self._ops.rdna_ar_can_argmax(self.handle, vals, idx)
        )

    def argmax(self, vals: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Global argmax over ranks of per-row (value, index); int64 [rows]."""
        return self._ops.rdna_ar_argmax(self.handle, vals, idx)

    def timed_out(self) -> bool:
        return (not self.disabled) and self._ops.rdna_ar_timed_out(self.handle)

    def _write_marker(self, msg: str) -> str | None:
        path = marker_path()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"world={self.world_size} {msg}\n"
                )
            return path
        except OSError as e:
            logger.warning("rdna_ar: could not write the wedge marker %s: %s", path, e)
            return None

    def check(self) -> None:
        """Fail the step if a captured collective aborted; log late peers."""
        if self.disabled:
            return
        slow_info = getattr(self._ops, "rdna_ar_slow_info", None)
        if slow_info is not None:
            slow = int(slow_info(self.handle))
            # A live record is a wait still in progress (the watcher logs it).
            if slow and not slow & _REC_LIVE and slow != getattr(self, "_last_slow", 0):
                self._last_slow = slow
                logger.warning(
                    "rdna_ar: late peer -- %s; the collective completed (that "
                    "rank was busy, e.g. JIT-compiling a kernel; not a wedge).",
                    describe_late(slow, self.rank),
                )
        code = int(self._ops.rdna_ar_timeout_info(self.handle))
        if code == 0:
            return
        self.disabled = True
        msg = describe_abort(code, self.rank)
        if (code >> 8) & 0xF in _PH_FLAG_WAITS and code & _REC_SEEN:
            path = self._write_marker(msg)
            logger.error(
                "rdna_ar: WEDGED -- %s. The one-shot all-reduce is disabled for "
                "this process; graph-captured steps cannot be re-routed live, "
                "so the engine stops here instead of grinding to the execute "
                "timeout. The next boot starts on RCCL automatically (marker: "
                "%s); VLLM_RDNA_AR=0 forces RCCL; delete the marker to retry "
                "P2P after checking ACS / IOMMU / slot topology.",
                msg,
                path or "not written",
            )
        else:
            logger.error(
                "rdna_ar: WEDGED -- %s. The one-shot all-reduce is disabled for "
                "this process; graph-captured steps cannot be re-routed live, "
                "so the engine stops here. No P2P write was lost, so no wedge "
                "marker is written: the next boot keeps the one-shot path.",
                msg,
            )
        raise RuntimeError(f"rdna_ar wedged: {msg} (see the log line above)")
