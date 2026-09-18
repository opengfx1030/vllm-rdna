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
CUSTOM / PYNCCL.

Staging and flags are uncached device memory (peer announce is a posted P2P
store; we poll locally). Sequence numbers live on device (graph-capture
safe). VLLM_RDNA_AR_BLOCKS / VLLM_RDNA_AR_PACE pace PCIe push bursts;
VLLM_RDNA_AR_MAX_KB (default 64) bounds the fast path so prefill-sized
reduces stay on RCCL/CUSTOM. VLLM_RDNA_AR_SPIN_CAP bounds the wait.

T44b wedge handling: a spin-cap abort records phase/peer/sequence in a
host-mapped word. rdna_ar_check() reads it once per engine step and fails
the step with a marker under VLLM_CACHE_ROOT so the next boot stays on RCCL.
"""

from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

_instances = 0
_MARKER_NAME = "rdna_ar_wedged"


def marker_path() -> str:
    from vllm import envs

    return os.path.join(envs.VLLM_CACHE_ROOT, _MARKER_NAME)


def describe_abort(code: int, rank: int) -> str:
    """Decode the kernel's abort record into one sentence.

    Bit layout matches leapdragon/vllm-rdna2-qwen T44b (Aron Hsiao).
    """
    phase = (code >> 8) & 0xF
    peer = (code >> 12) & 0xF
    ms = (code >> 16) & 0xFFFF
    seq = (code >> 32) & 0xFFFFFFFF
    if phase == 1:
        what = (
            "its own blocks never reached the grid barrier "
            "(a launch on this GPU stalled)"
        )
    else:
        what = (
            f"peer rank {peer}'s flag never arrived "
            f"(the posted P2P write from GPU {peer} was lost or stalled "
            "on this fabric)"
        )
    return (
        f"rank {rank} timed out after ~{ms} ms of spinning at collective #{seq}: {what}"
    )


# False = not looked up yet; None = no TP / inactive; else the TP instance.
_active: RdnaOneShotAllReduce | None | bool = False


def rdna_ar_check() -> None:
    """Per-step wedge check; a no-op unless the fast path is active."""
    global _active
    if _active is False:
        try:
            from vllm.distributed.parallel_state import get_tp_group

            _active = getattr(get_tp_group().device_communicator, "rdna_ar_comm", None)
        except Exception:  # noqa: BLE001 -- no TP group (single rank / not init)
            _active = None
    if _active is not None and not _active.disabled:
        _active.check()


class RdnaOneShotAllReduce:
    def __init__(self, group: ProcessGroup, device: torch.device) -> None:
        global _instances
        from vllm import _custom_ops as ops

        self.disabled = True
        self.handle = -1
        self._ops = ops
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        # Decode-sized default; prefill chunks are faster on RCCL.
        max_kb = int(os.getenv("VLLM_RDNA_AR_MAX_KB", "64"))
        self.max_bytes = max_kb * 1024
        if not (2 <= self.world_size <= 8):
            return
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
            device.index if device.index is not None else torch.cuda.current_device()
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
                    with torch.cuda.device(dev_idx):
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
        raw = packed.numpy().tobytes()
        self.handle = int.from_bytes(raw[:8], "little", signed=True)
        handles: list = [None] * self.world_size
        dist.all_gather_object(handles, raw[8:], group=group)
        buf = torch.frombuffer(bytearray(b"".join(handles)), dtype=torch.uint8).view(
            self.world_size, -1
        )
        err = None
        try:
            with torch.cuda.device(dev_idx):
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
        logger.info(
            "rdna_ar: one-shot all-reduce active (handle %d, rank %d/%d, "
            "devices %s, pix=%s, max %d KB; blocks cap %s, pace %s)",
            self.handle,
            self.rank,
            self.world_size,
            gathered,
            pix,
            max_kb,
            os.getenv("VLLM_RDNA_AR_BLOCKS", "auto"),
            os.getenv("VLLM_RDNA_AR_PACE", "0"),
        )

    def _self_test(self, device: torch.device, group: ProcessGroup) -> str | None:
        """Verified all-reduces on the fast path at three sizes.

        Returns an error string or None. Per size: one untimed warm-up, then
        REPEATS timed collectives judged on their minimum. Never leave the
        barrier loop early -- a rank that stops calling barriers while its
        peer keeps looping deadlocks both.
        """
        debug = os.environ.get("VLLM_RDNA_AR_DEBUG") == "1"
        repeats = 3
        err: str | None = None
        sync_dev = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        try:
            with torch.cuda.device(sync_dev):
                for trial, numel in enumerate((1024, 4096, self.max_bytes // 2)):
                    inp = torch.full(
                        (numel,),
                        float(self.rank + 1) * (trial + 1),
                        dtype=torch.float16,
                        device=device,
                    )
                    expect = float(
                        (trial + 1) * self.world_size * (self.world_size + 1) // 2
                    )
                    times: list[float] = []
                    for rep in range(repeats + 1):  # rep 0 = warm-up, untimed
                        dist.barrier(group=group)
                        if debug:
                            print(
                                f"[rdna_ar rank{self.rank}] trial{trial} "
                                f"rep{rep} barrier-out",
                                flush=True,
                            )
                        if err is not None:
                            continue
                        try:
                            t0 = time.perf_counter()
                            out = self._ops.rdna_ar_all_reduce(self.handle, inp)
                            torch.cuda.synchronize(sync_dev)
                            dt = time.perf_counter() - t0
                            if debug:
                                print(
                                    f"[rdna_ar rank{self.rank}] trial{trial} "
                                    f"rep{rep} call {dt * 1e3:.1f}ms",
                                    flush=True,
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
        """Fail the step if a captured collective hit its spin cap."""
        if self.disabled:
            return
        code = int(self._ops.rdna_ar_timeout_info(self.handle))
        if code == 0:
            return
        self.disabled = True
        msg = describe_abort(code, self.rank)
        path = self._write_marker(msg)
        logger.error(
            "rdna_ar: WEDGED -- %s. The one-shot all-reduce is disabled for "
            "this process; graph-captured steps cannot be re-routed live, so "
            "the engine stops here instead of grinding to the execute "
            "timeout. The next boot starts on RCCL automatically (marker: "
            "%s); VLLM_RDNA_AR=0 forces RCCL; delete the marker to retry P2P "
            "after checking ACS / IOMMU / slot topology.",
            msg,
            path or "not written",
        )
        raise RuntimeError(f"rdna_ar wedged: {msg} (see the log line above)")
