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
"""

from __future__ import annotations

import glob
import os
import threading
import time

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
    return (
        f"rank {rank} timed out after ~{ms} ms of spinning at collective #{seq}: {what}"
    )


def describe_late(code: int, rank: int) -> str:
    """Decode the one-shot late-peer record (report[1], abort-record layout)."""
    _, peer, ms, seq = _fields(code)
    msg = f"rank {rank} waited ~{ms} ms for peer rank {peer} at collective #{seq}"
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
    return (
        f"rank {rank} has been waiting ~{ms} ms for peer rank {peer} at "
        f"collective #{seq}: {state}"
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

# Two-lane mode calls rdna_ar_check() from both lane threads, and the list then
# holds both lanes' handles. check() reads host-visible records and may flip
# `disabled` and write the wedge marker, so one thread at a time; no
# device-wide synchronization happens inside the lock.
_check_lock = threading.Lock()


def rdna_ar_check() -> None:
    """Per-step wedge check of every live instance (no-op when none)."""
    if not _live:
        return
    with _check_lock:
        for inst in _live:
            if not inst.disabled:
                inst.check()


class RdnaOneShotAllReduce:
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
        # One-shot only; messages above the gate go to RCCL.
        max_kb, dropped = resolve_gate_kb()
        self.max_bytes = max_kb * 1024
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
                            self.max_bytes,
                            shm_name,
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
        with _check_lock:
            _live.append(self)
        _start_watcher()
        logger.info(
            "rdna_ar: one-shot all-reduce active (handle %d, rank %d/%d, "
            "devices %s, pix=%s, up to %d KB, RCCL above; "
            "blocks cap %s, pace %s, vec %s)",
            self.handle,
            self.rank,
            self.world_size,
            gathered,
            pix,
            max_kb,
            os.getenv("VLLM_RDNA_AR_BLOCKS", "auto"),
            os.getenv("VLLM_RDNA_AR_PACE", "0"),
            os.getenv("VLLM_RDNA_AR_VEC", "1"),
        )

    def _self_test(self, device: torch.device, group: ProcessGroup) -> str | None:
        """Verified all-reduces on the fast path at three sizes.

        Returns an error string or None. Per size: one untimed warm-up, then
        REPEATS timed collectives judged on their minimum. Never leave the
        barrier loop early -- a rank that stops calling barriers while its
        peer keeps looping deadlocks both.
        """
        repeats = 3
        err: str | None = None
        sync_dev = (
            device.index
            if device.index is not None
            else torch.accelerator.current_device_index()
        )
        try:
            with torch.accelerator.device_index(sync_dev):
                cases = (
                    (1024, torch.float16),
                    (4096, torch.float16),
                    (self.max_bytes // 2, torch.float16),
                    (1024, torch.bfloat16),
                )
                for trial, (numel, dtype) in enumerate(cases):
                    inp = torch.full(
                        (numel,),
                        float(self.rank + 1) * (trial + 1),
                        dtype=dtype,
                        device=device,
                    )
                    expect = float(
                        (trial + 1) * self.world_size * (self.world_size + 1) // 2
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
                            out = self._ops.rdna_ar_all_reduce(self.handle, inp)
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
                            elif not bool((out == expect).all()):
                                got = out.float().mean().item()
                                err = (
                                    f"wrong result in self-test trial {trial} "
                                    f"rep {rep}: mean {got:.2f}, expected "
                                    f"{expect:.1f}"
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
                                f"{numel * 2} bytes (all: "
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
        if (code >> 8) & 0xF == 2 and code & _REC_SEEN:
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
