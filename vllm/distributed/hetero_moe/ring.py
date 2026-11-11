# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step ring: how a hub host tells its servers' hosts that a step is coming.

The ring is a shared-memory file (``/dev/shm/hx_<plan_id>/ring_<hub>.bin``): one
writer (the hub host, ``StepRing.post`` before it launches a forward) and any
number of readers (one per server host). It carries only host-side facts the
device protocol cannot know before the step starts: the absolute epoch the hub's
``hx_begin_step`` will use, the row count ``T``, the graph bucket (0 = eager) and
the opcode of the step (``HX_OPCODE_MOE``; the server loop dispatches on it). A
server host replays the graph of that bucket, or runs its eager chain.

Layout (little endian)::

    header (64 B)   magic u64 | depth u32 | closed u32 | head u64 (steps posted)
    entry i (32 B)  step_no u64 | epoch_abs u64 | T u32 | bucket u16 | opcode u16
                    | check u64

Entry ``n`` (1-based step number) lives at ``i = (n - 1) % depth``. The writer
stores the entry, then ``head``; a reader that sees ``head >= n`` reads entry
``n`` and verifies ``check``, so a torn read is retried and a slot already
overwritten (reader more than ``depth`` steps behind) is reported, never
silently misread.
"""

import mmap
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path

from vllm.distributed.hetero_moe.hx_proto import HX_OPCODE_MOE

RING_MAGIC = 0x4858_5F52_494E_4731  # b"HX_RING1"
HEADER_BYTES = 64
ENTRY_BYTES = 32
DEFAULT_DEPTH = 64

_HEADER = struct.Struct("<QIIQ")
_ENTRY = struct.Struct("<QQIHHQ")
_MASK = (1 << 64) - 1


@dataclass(frozen=True)
class StepDesc:
    step_no: int  # 1-based count of posted steps
    epoch_abs: int  # epoch the hub's hx_begin_step will use for this step
    T: int
    bucket: int  # 0 = eager step
    opcode: int = HX_OPCODE_MOE


class RingOverrun(RuntimeError):
    """A reader fell more than ``depth`` steps behind the writer."""


def _check(step_no: int, epoch: int, T: int, bucket: int, opcode: int) -> int:
    h = (step_no * 0x9E3779B97F4A7C15) & _MASK
    h ^= (epoch * 0xC2B2AE3D27D4EB4F) & _MASK
    h ^= (((T << 32) | (bucket << 16) | opcode) * 0x165667B19E3779F9) & _MASK
    return (h ^ RING_MAGIC) & _MASK


class StepRing:
    def __init__(self, fd: int, mm: mmap.mmap, depth: int, writable: bool):
        self._fd = fd
        self._mm = mm
        self.depth = depth
        self.writable = writable
        self._posted = 0

    # ---- construction ---------------------------------------------------------

    @classmethod
    def create(cls, path: Path, depth: int = DEFAULT_DEPTH) -> "StepRing":
        """Writer side. Truncates any file of an earlier run."""
        if depth < 2:
            raise ValueError("ring depth must be >= 2")
        size = HEADER_BYTES + depth * ENTRY_BYTES
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
        os.ftruncate(fd, size)
        mm = mmap.mmap(fd, size)
        mm[:HEADER_BYTES] = _HEADER.pack(RING_MAGIC, depth, 0, 0).ljust(
            HEADER_BYTES, b"\0"
        )
        return cls(fd, mm, depth, True)

    @classmethod
    def open(cls, path: Path, timeout_s: float = 60.0) -> "StepRing":
        """Reader side: waits for the writer's file."""
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                fd = os.open(path, os.O_RDWR)
            except FileNotFoundError:
                fd = -1
            if fd >= 0:
                if os.fstat(fd).st_size >= HEADER_BYTES:
                    mm = mmap.mmap(fd, 0)
                    magic, depth, _, _ = _HEADER.unpack_from(mm, 0)
                    if magic == RING_MAGIC and depth >= 2:
                        if len(mm) != HEADER_BYTES + depth * ENTRY_BYTES:
                            mm.close()
                            os.close(fd)
                            raise RuntimeError(f"hx ring {path}: size/depth mismatch")
                        return cls(fd, mm, depth, False)
                    mm.close()
                os.close(fd)
            if time.monotonic() >= deadline:
                raise TimeoutError(f"hx ring {path} did not appear in {timeout_s:g} s")
            time.sleep(0.02)

    # ---- writer ---------------------------------------------------------------

    def post(
        self, epoch_abs: int, T: int, bucket: int, opcode: int = HX_OPCODE_MOE
    ) -> int:
        """Append a step; returns its 1-based number. Entry first, head last."""
        if not self.writable:
            raise RuntimeError("ring opened read-only")
        n = self._posted + 1
        off = HEADER_BYTES + ((n - 1) % self.depth) * ENTRY_BYTES
        _ENTRY.pack_into(
            self._mm,
            off,
            n,
            epoch_abs,
            T,
            bucket,
            opcode,
            _check(n, epoch_abs, T, bucket, opcode),
        )
        struct.pack_into("<Q", self._mm, 16, n)
        self._posted = n
        return n

    def close_writer(self) -> None:
        """Tell the readers the hub is shutting down cleanly."""
        if self.writable:
            struct.pack_into("<I", self._mm, 12, 1)

    # ---- reader ---------------------------------------------------------------

    @property
    def head(self) -> int:
        return struct.unpack_from("<Q", self._mm, 16)[0]

    @property
    def closed(self) -> bool:
        return struct.unpack_from("<I", self._mm, 12)[0] != 0

    def get(self, n: int) -> StepDesc | None:
        """Step ``n`` (1-based) when posted, else ``None``."""
        if self.head < n:
            return None
        off = HEADER_BYTES + ((n - 1) % self.depth) * ENTRY_BYTES
        for _ in range(100):
            step_no, epoch, T, bucket, opcode, chk = _ENTRY.unpack_from(self._mm, off)
            if step_no == n and chk == _check(n, epoch, T, bucket, opcode):
                return StepDesc(n, epoch, T, bucket, opcode)
            if step_no > n:
                raise RingOverrun(
                    f"hx ring: step {n} was overwritten (writer is at step {step_no})"
                )
            time.sleep(0)  # torn or not yet visible: retry
        raise RuntimeError(f"hx ring: entry {n} never became consistent")

    def close(self) -> None:
        try:
            self._mm.close()
        finally:
            os.close(self._fd)
