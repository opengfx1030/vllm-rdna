# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dedicated-queue CU-masked streams for the two-lane engine (S3 R-S3-1/R-S3-2).

Four streams per worker process, created once at startup, in a fixed order, and
never destroyed:

======================  ==================  ==========================
name                    mask                used by
======================  ==================  ==========================
``A`` (decode)          ``[0, n)``          lane D while partitioned
``B`` (prefill)         ``[n, W)``          lane P while partitioned
``D_full``              ``[0, W)``          lane D solo
``P_full``              ``[0, W)``          lane P solo
======================  ==================  ==========================

Why dedicated streams: every CU-masked stream owns a KFD queue, while the
default pool streams share two queues at ``GPU_MAX_HW_QUEUES=2``. A lane whose
long kernel backlog sat on a shared queue would head-of-line-block the other
lane. Why never destroyed: destroying a stream an RCCL communicator had seen
killed a collective in the spike, and there is no way to give a KFD queue back
that RCCL keeps using (``design.md`` 3.10).

The CU mask is applied with ``hipExtStreamCreateWithCUMask`` through ctypes on
the already-loaded ``libamdhip64`` (same call the spike used, mask words of 32
bits). Each stream costs one KFD queue; the worker budget is 7 queues total and
8 is the hard limit (I6, section 6.1).
"""

import ctypes
import os
from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.v1.lane import Lane

logger = init_logger(__name__)

_MASK_WORD_BITS = 32

# CU-mask domain (one bit per WGP) of the parts the engine supports. The spike
# measured the V620 domain at 36 bits with 72 CUs (design.md 3.2/3.9) and
# hard-coded it; startup validates the device against it and refuses a mismatch.
_KNOWN_WGP_BITS: dict[str, int] = {"gfx1030": 36}

# KFD compute queues per process (I6): 3 baseline + 4 masked streams = 7.
KFD_QUEUE_BUDGET = 7
KFD_QUEUE_HARD_LIMIT = 8

# Creating a 5th masked stream in one process already slowed every collective
# in the spike, and a 7th hangs RCCL; the engine never needs more than four.
_MAX_LANE_STREAMS = 4


def device_wgp_count(device: torch.device | str | int | None = None) -> int:
    """Number of CU-mask bits (WGPs) of ``device``.

    HIP on gfx10x reports one "multi-processor" per WGP (a 72-CU V620 shows
    36), while other stacks report raw CUs. For a part with a known mask-domain
    size both conventions are accepted and anything else is refused; for an
    unknown part the CU count is halved.
    """
    props = torch.cuda.get_device_properties(device)
    arch = str(props.gcnArchName).split(":")[0]
    multiprocessors = int(props.multi_processor_count)
    expected = _KNOWN_WGP_BITS.get(arch)
    if expected is not None:
        if multiprocessors in (expected, expected * 2):
            return expected
        raise RuntimeError(
            f"two-lane: device {arch} reports {multiprocessors} "
            f"multi-processors, but the CU-mask domain must be {expected} bits "
            f"(one per WGP). Refusing to guess a lane split."
        )
    wgps = multiprocessors // 2 if multiprocessors % 2 == 0 else multiprocessors
    if wgps < 2:
        raise RuntimeError(
            f"two-lane: device {arch} reports {multiprocessors} "
            f"multi-processors; not enough CUs for two lanes."
        )
    return wgps


@dataclass(frozen=True)
class LaneStreams:
    """The four dedicated-queue streams of one worker process (R-S3-1)."""

    decode: torch.cuda.ExternalStream
    prefill: torch.cuda.ExternalStream
    decode_full: torch.cuda.ExternalStream
    prefill_full: torch.cuda.ExternalStream
    decode_wgps: int
    total_wgps: int
    # Raw HIP handles, kept so the streams are never garbage collected and for
    # logging; the process exits with them alive (R-S3-1).
    handles: tuple[int, int, int, int]

    def stream_for(self, lane: int | Lane, partitioned: bool) -> torch.cuda.Stream:
        """The stream lane ``lane`` runs a batch with ``partitioned`` on."""
        if Lane(lane) == Lane.DECODE:
            return self.decode if partitioned else self.decode_full
        return self.prefill if partitioned else self.prefill_full

    def name_for(self, lane: int | Lane, partitioned: bool) -> str:
        if Lane(lane) == Lane.DECODE:
            return "A" if partitioned else "D_full"
        return "B" if partitioned else "P_full"

    def masks(self) -> dict[str, str]:
        """Human-readable masks for the startup log."""
        return {
            "A": f"[0,{self.decode_wgps})",
            "B": f"[{self.decode_wgps},{self.total_wgps})",
            "D_full": f"[0,{self.total_wgps})",
            "P_full": f"[0,{self.total_wgps})",
        }


def _mask_words(wgps: range, num_words: int) -> list[int]:
    words = [0] * num_words
    for wgp in wgps:
        if not 0 <= wgp < num_words * _MASK_WORD_BITS:
            raise ValueError(f"WGP {wgp} is outside the {num_words}-word CU mask")
        words[wgp // _MASK_WORD_BITS] |= 1 << (wgp % _MASK_WORD_BITS)
    return words


def _load_hip_library() -> ctypes.CDLL:
    """The already-loaded ``libamdhip64`` (found through /proc/self/maps)."""
    path = None
    with open("/proc/self/maps", encoding="utf-8") as f:
        for line in f:
            if "libamdhip64" in line:
                path = line.split()[-1]
                break
    if path is None:
        raise RuntimeError(
            "two-lane: libamdhip64 is not loaded; cannot create CU-masked streams"
        )
    lib = ctypes.CDLL(path)
    lib.hipExtStreamCreateWithCUMask.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.hipExtStreamCreateWithCUMask.restype = ctypes.c_int
    return lib


def _create_masked_stream(
    lib: ctypes.CDLL, wgps: range, num_words: int, name: str
) -> tuple[torch.cuda.ExternalStream, int]:
    words = (ctypes.c_uint32 * num_words)(*_mask_words(wgps, num_words))
    handle = ctypes.c_void_p()
    rc = lib.hipExtStreamCreateWithCUMask(ctypes.byref(handle), num_words, words)
    if rc != 0 or handle.value is None:
        raise RuntimeError(
            f"two-lane: hipExtStreamCreateWithCUMask({name}, {list(wgps)}) "
            f"failed with rc={rc}"
        )
    return torch.cuda.ExternalStream(handle.value), int(handle.value)


_created: LaneStreams | None = None


def create_lane_streams(
    decode_wgps: int, device: torch.device | str | int | None = None
) -> LaneStreams:
    """Create (once) and return the process's four lane streams.

    A second call returns the same streams: they are created once and never
    destroyed (I5). Refuses a split that does not leave a WGP for each lane and
    a device dimension that does not match its CU-mask domain.
    """
    global _created
    total_wgps = device_wgp_count(device)
    if not 1 <= int(decode_wgps) < total_wgps:
        raise ValueError(
            f"two-lane: decode_wgps={decode_wgps} must satisfy 1 <= n < {total_wgps}"
        )
    if _created is not None:
        if _created.decode_wgps != int(decode_wgps):
            raise RuntimeError(
                f"two-lane: lane streams already exist with decode_wgps="
                f"{_created.decode_wgps}, refusing decode_wgps={decode_wgps}"
            )
        return _created

    # Force the HIP runtime to be loaded before scanning /proc/self/maps.
    torch.zeros(1, device=device or torch.accelerator.current_device_index())
    lib = _load_hip_library()
    num_words = (total_wgps + _MASK_WORD_BITS - 1) // _MASK_WORD_BITS
    n = int(decode_wgps)

    # Order matters and is part of the contract (R-S3-1).
    decode, h_a = _create_masked_stream(lib, range(0, n), num_words, "A")
    prefill, h_b = _create_masked_stream(lib, range(n, total_wgps), num_words, "B")
    # D_full's mask is the whole device, so a plain stream is equivalent and,
    # unlike a hipExtStreamCreateWithCUMask stream, it can host CUDA graph
    # capture. Capture reuses it, which keeps the process at 3 baseline + 4
    # lane queues (I6).
    decode_full = torch.cuda.Stream(device=device)
    h_df = int(decode_full.cuda_stream)
    prefill_full, h_pf = _create_masked_stream(
        lib, range(0, total_wgps), num_words, "P_full"
    )

    _created = LaneStreams(
        decode=decode,
        prefill=prefill,
        decode_full=decode_full,
        prefill_full=prefill_full,
        decode_wgps=n,
        total_wgps=total_wgps,
        handles=(h_a, h_b, h_df, h_pf),
    )

    # A HIP stream's KFD compute queue appears when its first command is
    # enqueued, not when the stream is created. Touch each stream once so the
    # startup census (R-S3-2) counts all four queues and so a later "queue
    # missing" cannot be mistaken for a small budget.
    probe_device = device
    if probe_device is None:
        probe_device = torch.accelerator.current_device_index()
    probe = torch.zeros(1, dtype=torch.uint8, device=probe_device)
    for stream in (decode, prefill, decode_full, prefill_full):
        with torch.cuda.stream(stream):
            probe.fill_(0)
        stream.synchronize()

    logger.info(
        "two-lane: created %d dedicated streams over a %d-WGP device (masks %s)",
        _MAX_LANE_STREAMS,
        total_wgps,
        _created.masks(),
    )
    return _created


@dataclass(frozen=True)
class KfdCensus:
    """KFD compute queues held by this process and by every other one."""

    own: int
    others: dict[int, int]

    @property
    def other_total(self) -> int:
        return sum(self.others.values())

    def describe(self) -> str:
        return (
            f"{self.own} KFD queues in this process, "
            f"{self.other_total} in {len(self.others)} other KFD "
            f"process(es) on this node"
        )


def _count_queues(queue_dir: str) -> int | None:
    try:
        return len(os.listdir(queue_dir))
    except FileNotFoundError:
        return 0
    except OSError:
        # Another user's process: the count is not readable, not zero.
        return None


def kfd_census(pid: int | None = None) -> KfdCensus:
    """Count KFD compute queues (R-S3-2).

    ``own`` is this process's queue count; ``others`` maps the pid of every
    other KFD process holding queues on the node to its count. The counts are
    per node, not per GPU: KFD does not expose a per-queue device mapping here,
    and queues of a co-tenant process at least share the runlist, so they are
    logged with the budget check.
    """
    me = os.getpid() if pid is None else pid
    proc_root = "/sys/class/kfd/kfd/proc"
    own = _count_queues(os.path.join(proc_root, str(me), "queues"))
    if own is None:
        raise RuntimeError(
            f"two-lane: cannot read this process's KFD queues under "
            f"{proc_root}/{me}; the KFD queue budget (I6) cannot be checked"
        )
    others: dict[int, int] = {}
    try:
        entries = os.listdir(proc_root)
    except OSError as e:
        logger.warning("two-lane: cannot list %s (%s)", proc_root, e)
        return KfdCensus(own=own, others={})
    for entry in entries:
        try:
            other_pid = int(entry)
        except ValueError:
            continue
        if other_pid == me:
            continue
        count = _count_queues(os.path.join(proc_root, entry, "queues"))
        if count:
            others[other_pid] = count
    return KfdCensus(own=own, others=others)


def check_kfd_budget(census: KfdCensus, *, context: str = "worker startup") -> None:
    """Enforce I6 / section 6.1 on a census (raise above the hard limit)."""
    if census.own > KFD_QUEUE_HARD_LIMIT:
        raise RuntimeError(
            f"two-lane: {census.own} KFD queues after {context}; the hard limit "
            f"is {KFD_QUEUE_HARD_LIMIT} (budget {KFD_QUEUE_BUDGET}: 3 baseline "
            f"+ 4 lane streams). {census.describe()}. Drop a communicator set or "
            f"a stream before serving: a runlist above 8 oversubscribes the GPU "
            f"and slows every collective."
        )
    if census.own == KFD_QUEUE_HARD_LIMIT:
        logger.warning(
            "two-lane: %s (at the hard limit; the target is %d). %s",
            context,
            KFD_QUEUE_BUDGET,
            census.describe(),
        )
        return
    logger.info(
        "two-lane: %s (target %d). %s",
        context,
        KFD_QUEUE_BUDGET,
        census.describe(),
    )
